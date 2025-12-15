use crate::artifact::ivf_flat::{ChunkType, IvfFlatArtifact, PostingCodec};
use crate::ivf_flat::SearchResult;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use wasi_common::sync::WasiCtxBuilder;
use wasmtime::{Caller, Engine, Linker, Memory, Module, Store, TypedFunc};

#[derive(Default)]
struct HostState {
    artifact: Option<Arc<IvfFlatArtifact>>,
    file: Option<File>,
    cache: HashMap<u32, Vec<u8>>,
    raw_lens: Vec<u64>,
    lens: Vec<u32>,
    stats: FetchStats,
    cache_enabled: bool,
}

struct WasmState {
    wasi: wasi_common::WasiCtx,
    host: HostState,
}

impl HostState {
    fn set_artifact(&mut self, artifact: Arc<IvfFlatArtifact>) {
        self.raw_lens = artifact.footer().chunks.iter().map(|c| c.raw_len).collect();
        self.lens = artifact
            .footer()
            .chunks
            .iter()
            .map(|c| u32::try_from(c.len).unwrap_or(0))
            .collect();
        self.file = File::open(artifact.path()).ok();
        self.artifact = Some(artifact);
        self.cache.clear();
        self.stats = FetchStats::default();
        self.cache_enabled = true;
    }

    fn chunk_len(&mut self, chunk_id: u32) -> Result<u32> {
        if let Some(buf) = self.cache.get(&chunk_id) {
            self.stats.cache_hits += 1;
            self.stats.note_chunk(chunk_id, /*miss=*/false);
            return Ok(u32::try_from(buf.len()).unwrap_or(0));
        }
        Ok(self.lens.get(chunk_id as usize).copied().unwrap_or(0))
    }

    fn read_chunk_into(&mut self, chunk_id: u32, dst: &mut [u8]) -> Result<u32> {
        let bytes = if let Some(buf) = self.cache.get(&chunk_id) {
            self.stats.cache_hits += 1;
            self.stats.note_chunk(chunk_id, /*miss=*/false);
            buf.clone()
        } else {
            let artifact = self
                .artifact
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("artifact not set in host state"))?;
            let file = self
                .file
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("artifact file not open in host state"))?;
            let start = Instant::now();
            let bytes = artifact.read_chunk_bytes_by_index_from_file(file, chunk_id)?;
            self.stats.fetch_time_ns += start.elapsed().as_nanos() as u64;
            self.stats.chunks_fetched += 1;
            self.stats.compressed_bytes_in += bytes.len() as u64;
            if let Some(raw_len) = self.raw_lens.get(chunk_id as usize).copied() {
                self.stats.raw_bytes_decoded += raw_len;
            }
            self.stats.note_chunk(chunk_id, /*miss=*/true);
            if self.cache_enabled {
                self.cache.insert(chunk_id, bytes.clone());
            }
            bytes
        };

        if bytes.len() > dst.len() {
            return Ok(0);
        }
        dst[..bytes.len()].copy_from_slice(&bytes);
        Ok(bytes.len() as u32)
    }
}

#[derive(Debug, Default, Clone)]
pub struct FetchStats {
    pub cache_hits: u64,
    pub chunks_fetched: u64,
    pub compressed_bytes_in: u64,
    pub raw_bytes_decoded: u64,
    pub fetch_time_ns: u64,
    pub total_time_ns: u64,
    pub fetched_chunk_ids_sample: Vec<u32>,
    pub cache_miss_chunk_ids_sample: Vec<u32>,
}

impl FetchStats {
    fn note_chunk(&mut self, chunk_id: u32, miss: bool) {
        const LIMIT: usize = 64;
        if self.fetched_chunk_ids_sample.len() < LIMIT {
            self.fetched_chunk_ids_sample.push(chunk_id);
        }
        if miss && self.cache_miss_chunk_ids_sample.len() < LIMIT {
            self.cache_miss_chunk_ids_sample.push(chunk_id);
        }
    }
}

pub struct WasmIvfFlatKernel {
    store: Store<WasmState>,
    memory: Memory,
    alloc: TypedFunc<(u32, u32), u32>,
    dealloc: TypedFunc<(u32, u32, u32), ()>,
    search_batch: TypedFunc<(u32, u32, u32, u32, u32, u32, u32, u32, u32, u32), u32>,
    set_decoded_cache_budget: Option<TypedFunc<u32, ()>>,
    last_stats: Option<TypedFunc<u32, u32>>,
    decoded_cache_budget_bytes: u32,
    last_kernel_stats: KernelStats,
}

#[derive(Debug, Default, Clone)]
pub struct KernelStats {
    pub decode_time_ns: u64,
    pub compute_time_ns: u64,
    pub decoded_cache_hits: u64,
    pub decoded_cache_misses: u64,
    pub decoded_cache_bytes: u64,
}

impl WasmIvfFlatKernel {
    pub fn load(wasm_path: impl AsRef<Path>, artifact: Arc<IvfFlatArtifact>) -> Result<Self> {
        let engine = Engine::default();
        let module = Module::from_file(&engine, wasm_path.as_ref())
            .with_context(|| format!("load wasm {}", wasm_path.as_ref().display()))?;

        let mut linker = Linker::<WasmState>::new(&engine);
        wasi_common::sync::add_to_linker(&mut linker, |s| &mut s.wasi)?;

        linker.func_wrap("env", "host_chunk_len", |mut caller: Caller<'_, WasmState>, chunk_id: u32| -> u32 {
            let state = &mut caller.data_mut().host;
            match state.chunk_len(chunk_id) {
                Ok(len) => len,
                Err(_) => 0,
            }
        })?;

        linker.func_wrap(
            "env",
            "host_read_chunk",
            |mut caller: Caller<'_, WasmState>, chunk_id: u32, dst_ptr: u32, dst_len: u32| -> u32 {
                let res: Result<u32> = (|| {
                    let mem = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| anyhow::anyhow!("wasm export `memory` not found"))?;
                    let mut tmp = vec![0u8; dst_len as usize];
                    let n = {
                        let state = &mut caller.data_mut().host;
                        state.read_chunk_into(chunk_id, &mut tmp)?
                    };
                    if n == 0 {
                        return Ok(0);
                    }
                    mem.write(&mut caller, dst_ptr as usize, &tmp[..n as usize])?;
                    Ok(n)
                })();
                res.unwrap_or(0)
            },
        )?;

        let wasi = WasiCtxBuilder::new().inherit_stdio().build();
        let mut store = Store::new(
            &engine,
            WasmState {
                wasi,
                host: HostState::default(),
            },
        );
        store.data_mut().host.set_artifact(artifact);

        let instance = linker
            .instantiate(&mut store, &module)
            .with_context(|| "instantiate wasm module")?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow::anyhow!("wasm export `memory` not found"))?;

        let alloc = instance
            .get_typed_func::<(u32, u32), u32>(&mut store, "alloc_ffi")
            .with_context(|| "get export alloc_ffi")?;
        let dealloc = instance
            .get_typed_func::<(u32, u32, u32), ()>(&mut store, "dealloc_ffi")
            .with_context(|| "get export dealloc_ffi")?;
        let search_batch = instance
            .get_typed_func::<(u32, u32, u32, u32, u32, u32, u32, u32, u32, u32), u32>(
                &mut store,
                "ivf_flat_search_batch_ffi",
            )
            .with_context(|| "get export ivf_flat_search_batch_ffi")?;

        let set_decoded_cache_budget = instance
            .get_typed_func::<u32, ()>(&mut store, "ivf_set_decoded_cache_budget_ffi")
            .ok();
        let last_stats = instance
            .get_typed_func::<u32, u32>(&mut store, "ivf_last_stats_ffi")
            .ok();

        Ok(Self {
            store,
            memory,
            alloc,
            dealloc,
            search_batch,
            set_decoded_cache_budget,
            last_stats,
            decoded_cache_budget_bytes: 192 * 1024 * 1024,
            last_kernel_stats: KernelStats::default(),
        })
    }

    pub fn set_decoded_cache_budget_bytes(&mut self, bytes: u32) {
        self.decoded_cache_budget_bytes = bytes;
    }

    pub fn kernel_stats(&self) -> KernelStats {
        self.last_kernel_stats.clone()
    }

    pub fn reset_stats(&mut self) {
        self.store.data_mut().host.stats = FetchStats::default();
    }

    pub fn set_cache_enabled(&mut self, enabled: bool) {
        self.store.data_mut().host.cache_enabled = enabled;
        if !enabled {
            self.store.data_mut().host.cache.clear();
        }
    }

    pub fn stats(&self) -> FetchStats {
        self.store.data().host.stats.clone()
    }

    pub fn search(
        &mut self,
        artifact: &IvfFlatArtifact,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<SearchResult>> {
        let batches = self.search_batch(artifact, query, 1, k, nprobe)?;
        Ok(batches.into_iter().next().unwrap_or_default())
    }

    pub fn search_batch(
        &mut self,
        artifact: &IvfFlatArtifact,
        queries: &[f32],
        nq: usize,
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let dim = artifact.footer().dim as usize;
        let nlist = artifact.footer().nlist as usize;
        if nq == 0 {
            return Ok(vec![]);
        }
        if queries.len() != nq * dim {
            bail!(
                "queries len mismatch: expected {}, got {}",
                nq * dim,
                queries.len()
            );
        }
        if k == 0 {
            return Ok(vec![vec![]; nq]);
        }

        let centroids_id = artifact.find_chunk_id(ChunkType::Centroids, None)?;
        let posting_codec = artifact
            .footer()
            .chunks
            .iter()
            .find(|c| c.chunk_type == ChunkType::PostingList && c.list_id == Some(0))
            .map(|c| c.codec)
            .unwrap_or(PostingCodec::Raw);
        let posting_codec_id: u32 = match posting_codec {
            PostingCodec::Raw => 0,
            PostingCodec::RowIdDeltaVarintV1 => 1,
            PostingCodec::RawF16 => 2,
            PostingCodec::RowIdDeltaVarintV1F16 => 3,
        };

        let mut posting_ids = vec![0u32; nlist];
        for list_id in 0..nlist {
            posting_ids[list_id] =
                artifact.find_chunk_id(ChunkType::PostingList, Some(list_id as u32))?;
        }
        let mut dir = Vec::<u8>::with_capacity(16 + nlist * 4);
        dir.extend_from_slice(&(dim as u32).to_le_bytes());
        dir.extend_from_slice(&(nlist as u32).to_le_bytes());
        dir.extend_from_slice(&centroids_id.to_le_bytes());
        dir.extend_from_slice(&posting_codec_id.to_le_bytes());
        for id in posting_ids {
            dir.extend_from_slice(&id.to_le_bytes());
        }

        let queries_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(queries.as_ptr() as *const u8, queries.len() * 4)
        };

        let out_cap = k as u32;
        let out_len_bytes = (nq as u32) * out_cap * 8;
        let counts_len_bytes = (nq as u32) * 4;

        let dir_ptr = self.alloc.call(&mut self.store, (dir.len() as u32, 8))?;
        if dir_ptr == 0 {
            bail!("wasm alloc failed for dir");
        }
        self.memory.write(&mut self.store, dir_ptr as usize, &dir)?;

        let queries_ptr = self
            .alloc
            .call(&mut self.store, (queries_bytes.len() as u32, 4))?;
        if queries_ptr == 0 {
            self.dealloc.call(&mut self.store, (dir_ptr, dir.len() as u32, 8))?;
            bail!("wasm alloc failed for queries");
        }
        self.memory
            .write(&mut self.store, queries_ptr as usize, queries_bytes)?;

        let out_ptr = self.alloc.call(&mut self.store, (out_len_bytes as u32, 8))?;
        if out_ptr == 0 {
            self.dealloc.call(&mut self.store, (dir_ptr, dir.len() as u32, 8))?;
            self.dealloc
                .call(&mut self.store, (queries_ptr, queries_bytes.len() as u32, 4))?;
            bail!("wasm alloc failed for output");
        }
        let counts_ptr = self
            .alloc
            .call(&mut self.store, (counts_len_bytes as u32, 4))?;
        if counts_ptr == 0 {
            self.dealloc.call(&mut self.store, (dir_ptr, dir.len() as u32, 8))?;
            self.dealloc
                .call(&mut self.store, (queries_ptr, queries_bytes.len() as u32, 4))?;
            self.dealloc
                .call(&mut self.store, (out_ptr, out_len_bytes as u32, 8))?;
            bail!("wasm alloc failed for counts");
        }

        let nprobe = (nprobe.min(nlist).max(1)) as u32;
        let start_total = Instant::now();

        if let Some(setter) = &self.set_decoded_cache_budget {
            let _ = setter.call(&mut self.store, self.decoded_cache_budget_bytes);
        }

        let ok = self.search_batch.call(
            &mut self.store,
            (
                dir_ptr,
                dir.len() as u32,
                queries_ptr,
                nq as u32,
                dim as u32,
                k as u32,
                nprobe,
                out_ptr,
                out_cap,
                counts_ptr,
            ),
        )?;
        let total_time_ns = start_total.elapsed().as_nanos() as u64;
        if ok == 0 {
            bail!("wasm batch search failed");
        }

        let counts_bytes = self
            .memory
            .data(&self.store)
            .get(counts_ptr as usize..counts_ptr as usize + counts_len_bytes as usize)
            .ok_or_else(|| anyhow::anyhow!("counts slice out of bounds"))?;
        let mut counts = vec![0u32; nq];
        for i in 0..nq {
            let off = i * 4;
            counts[i] = u32::from_le_bytes(counts_bytes[off..off + 4].try_into().unwrap());
        }

        let out_bytes = self
            .memory
            .data(&self.store)
            .get(out_ptr as usize..out_ptr as usize + out_len_bytes as usize)
            .ok_or_else(|| anyhow::anyhow!("output slice out of bounds"))?;

        let mut results = Vec::<Vec<SearchResult>>::with_capacity(nq);
        for q in 0..nq {
            let mut one = Vec::<SearchResult>::with_capacity(counts[q] as usize);
            let base = q * (out_cap as usize) * 8;
            for i in 0..counts[q] as usize {
                let off = base + i * 8;
                let row_id = u32::from_le_bytes(out_bytes[off..off + 4].try_into().unwrap());
                let dist_bits =
                    u32::from_le_bytes(out_bytes[off + 4..off + 8].try_into().unwrap());
                one.push(SearchResult {
                    row_id,
                    distance: f32::from_bits(dist_bits),
                });
            }
            results.push(one);
        }

        self.dealloc
            .call(&mut self.store, (dir_ptr, dir.len() as u32, 8))?;
        self.dealloc
            .call(&mut self.store, (queries_ptr, queries_bytes.len() as u32, 4))?;
        self.dealloc
            .call(&mut self.store, (out_ptr, out_len_bytes as u32, 8))?;
        self.dealloc
            .call(&mut self.store, (counts_ptr, counts_len_bytes as u32, 4))?;

        self.store.data_mut().host.stats.total_time_ns = total_time_ns;

        // Pull kernel-side breakdown stats if available.
        self.last_kernel_stats = KernelStats::default();
        if let Some(getter) = &self.last_stats {
            let stats_ptr = self.alloc.call(&mut self.store, (5 * 8, 8))?;
            if stats_ptr != 0 {
                if getter.call(&mut self.store, stats_ptr)? != 0 {
                    let mut buf = vec![0u8; 5 * 8];
                    self.memory.read(&mut self.store, stats_ptr as usize, &mut buf)?;
                    let mut words = [0u64; 5];
                    for i in 0..5 {
                        let off = i * 8;
                        words[i] = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
                    }
                    self.last_kernel_stats = KernelStats {
                        decode_time_ns: words[0],
                        compute_time_ns: words[1],
                        decoded_cache_hits: words[2],
                        decoded_cache_misses: words[3],
                        decoded_cache_bytes: words[4],
                    };
                }
                let _ = self.dealloc.call(&mut self.store, (stats_ptr, 5 * 8, 8));
            }
        }

        Ok(results)
    }
}
