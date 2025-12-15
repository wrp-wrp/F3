use crate::artifact::ivf_flat::{ChunkType, IvfFlatArtifact, PostingCodec};
use crate::ivf_flat::SearchResult;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use wasmtime::{Caller, Engine, Linker, Memory, Module, Store, TypedFunc};

#[derive(Default)]
struct HostState {
    artifact: Option<Arc<IvfFlatArtifact>>,
    cache: HashMap<u32, Vec<u8>>,
    pending: HashMap<u32, Vec<u8>>,
    raw_lens: Vec<u64>,
    stats: FetchStats,
    cache_enabled: bool,
}

impl HostState {
    fn set_artifact(&mut self, artifact: Arc<IvfFlatArtifact>) {
        self.raw_lens = artifact.footer().chunks.iter().map(|c| c.raw_len).collect();
        self.artifact = Some(artifact);
        self.cache.clear();
        self.pending.clear();
        self.stats = FetchStats::default();
        self.cache_enabled = true;
    }

    fn chunk_len(&mut self, chunk_id: u32) -> Result<u32> {
        if let Some(buf) = self.cache.get(&chunk_id) {
            self.stats.cache_hits += 1;
            self.stats.note_chunk(chunk_id, /*miss=*/false);
            return Ok(u32::try_from(buf.len()).unwrap_or(0));
        }
        if let Some(buf) = self.pending.get(&chunk_id) {
            self.stats.note_chunk(chunk_id, /*miss=*/false);
            return Ok(u32::try_from(buf.len()).unwrap_or(0));
        }
        let artifact = self
            .artifact
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("artifact not set in host state"))?;
        let start = Instant::now();
        let bytes = artifact.read_chunk_bytes_by_index(chunk_id)?;
        self.stats.fetch_time_ns += start.elapsed().as_nanos() as u64;
        self.stats.chunks_fetched += 1;
        self.stats.compressed_bytes_in += bytes.len() as u64;
        if let Some(raw_len) = self.raw_lens.get(chunk_id as usize).copied() {
            self.stats.raw_bytes_decoded += raw_len;
        }
        self.stats.note_chunk(chunk_id, /*miss=*/true);
        if self.cache_enabled {
            self.cache.insert(chunk_id, bytes);
            Ok(u32::try_from(self.cache[&chunk_id].len()).unwrap_or(0))
        } else {
            self.pending.insert(chunk_id, bytes);
            Ok(u32::try_from(self.pending[&chunk_id].len()).unwrap_or(0))
        }
    }

    fn read_chunk_into(&mut self, chunk_id: u32, dst: &mut [u8]) -> Result<u32> {
        let mut bytes = if let Some(buf) = self.cache.get(&chunk_id) {
            self.stats.cache_hits += 1;
            self.stats.note_chunk(chunk_id, /*miss=*/false);
            buf.clone()
        } else if let Some(buf) = self.pending.remove(&chunk_id) {
            self.stats.note_chunk(chunk_id, /*miss=*/false);
            buf
        } else {
            // Fallback: load on-demand (should be rare because wasm calls host_chunk_len first).
            let len = self.chunk_len(chunk_id)?;
            if len == 0 {
                return Ok(0);
            }
            if let Some(buf) = self.cache.get(&chunk_id) {
                buf.clone()
            } else if let Some(buf) = self.pending.remove(&chunk_id) {
                buf
            } else {
                return Ok(0);
            }
        };

        if bytes.len() > dst.len() {
            return Ok(0);
        }
        dst[..bytes.len()].copy_from_slice(&bytes);
        let n = bytes.len() as u32;
        if self.cache_enabled && !self.cache.contains_key(&chunk_id) {
            self.cache.insert(chunk_id, std::mem::take(&mut bytes));
        }
        Ok(n)
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
    store: Store<HostState>,
    memory: Memory,
    alloc: TypedFunc<(u32, u32), u32>,
    dealloc: TypedFunc<(u32, u32, u32), ()>,
    search: TypedFunc<(u32, u32, u32, u32, u32, u32, u32, u32), u32>,
}

impl WasmIvfFlatKernel {
    pub fn load(wasm_path: impl AsRef<Path>, artifact: Arc<IvfFlatArtifact>) -> Result<Self> {
        let engine = Engine::default();
        let module = Module::from_file(&engine, wasm_path.as_ref())
            .with_context(|| format!("load wasm {}", wasm_path.as_ref().display()))?;

        let mut linker = Linker::<HostState>::new(&engine);

        linker.func_wrap("env", "host_chunk_len", |mut caller: Caller<'_, HostState>, chunk_id: u32| -> u32 {
            let state = caller.data_mut();
            match state.chunk_len(chunk_id) {
                Ok(len) => len,
                Err(_) => 0,
            }
        })?;

        linker.func_wrap(
            "env",
            "host_read_chunk",
            |mut caller: Caller<'_, HostState>, chunk_id: u32, dst_ptr: u32, dst_len: u32| -> u32 {
                let res: Result<u32> = (|| {
                    let mem = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| anyhow::anyhow!("wasm export `memory` not found"))?;
                    let mut tmp = vec![0u8; dst_len as usize];
                    let n = {
                        let state = caller.data_mut();
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

        let mut store = Store::new(&engine, HostState::default());
        store.data_mut().set_artifact(artifact);

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
        let search = instance
            .get_typed_func::<(u32, u32, u32, u32, u32, u32, u32, u32), u32>(
                &mut store,
                "ivf_flat_search_ffi",
            )
            .with_context(|| "get export ivf_flat_search_ffi")?;

        Ok(Self {
            store,
            memory,
            alloc,
            dealloc,
            search,
        })
    }

    pub fn reset_stats(&mut self) {
        self.store.data_mut().stats = FetchStats::default();
    }

    pub fn set_cache_enabled(&mut self, enabled: bool) {
        self.store.data_mut().cache_enabled = enabled;
        if !enabled {
            self.store.data_mut().cache.clear();
        }
    }

    pub fn stats(&self) -> FetchStats {
        self.store.data().stats.clone()
    }

    pub fn search(
        &mut self,
        artifact: &IvfFlatArtifact,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<SearchResult>> {
        let dim = artifact.footer().dim as usize;
        let nlist = artifact.footer().nlist as usize;
        if query.len() != dim {
            bail!("query dim mismatch: expected {dim}, got {}", query.len());
        }
        if k == 0 {
            return Ok(vec![]);
        }

        // Build directory buffer for wasm.
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

        let query_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(query.as_ptr() as *const u8, query.len() * 4)
        };

        let out_cap = k as u32;
        let out_len_bytes = (out_cap as usize) * 8;

        // Allocate and write buffers into wasm memory.
        let dir_ptr = self.alloc.call(&mut self.store, (dir.len() as u32, 8))?;
        if dir_ptr == 0 {
            bail!("wasm alloc failed for dir");
        }
        self.memory.write(&mut self.store, dir_ptr as usize, &dir)?;

        let query_ptr = self
            .alloc
            .call(&mut self.store, (query_bytes.len() as u32, 4))?;
        if query_ptr == 0 {
            self.dealloc.call(&mut self.store, (dir_ptr, dir.len() as u32, 8))?;
            bail!("wasm alloc failed for query");
        }
        self.memory
            .write(&mut self.store, query_ptr as usize, query_bytes)?;

        let out_ptr = self.alloc.call(&mut self.store, (out_len_bytes as u32, 8))?;
        if out_ptr == 0 {
            self.dealloc.call(&mut self.store, (dir_ptr, dir.len() as u32, 8))?;
            self.dealloc
                .call(&mut self.store, (query_ptr, query_bytes.len() as u32, 4))?;
            bail!("wasm alloc failed for output");
        }

        let nprobe = (nprobe.min(nlist).max(1)) as u32;
        let start_total = Instant::now();
        let out_n = self.search.call(
            &mut self.store,
            (
                dir_ptr,
                dir.len() as u32,
                query_ptr,
                dim as u32,
                k as u32,
                nprobe,
                out_ptr,
                out_cap,
            ),
        )?;
        let total_time_ns = start_total.elapsed().as_nanos() as u64;

        // Read results back.
        let out_bytes = self
            .memory
            .data(&self.store)
            .get(out_ptr as usize..out_ptr as usize + (out_n as usize) * 8)
            .ok_or_else(|| anyhow::anyhow!("output slice out of bounds"))?;
        let mut results = Vec::<SearchResult>::with_capacity(out_n as usize);
        for i in 0..out_n as usize {
            let off = i * 8;
            let row_id = u32::from_le_bytes(out_bytes[off..off + 4].try_into().unwrap());
            let dist_bits = u32::from_le_bytes(out_bytes[off + 4..off + 8].try_into().unwrap());
            results.push(SearchResult {
                row_id,
                distance: f32::from_bits(dist_bits),
            });
        }

        // Free wasm allocations.
        self.dealloc
            .call(&mut self.store, (dir_ptr, dir.len() as u32, 8))?;
        self.dealloc
            .call(&mut self.store, (query_ptr, query_bytes.len() as u32, 4))?;
        self.dealloc
            .call(&mut self.store, (out_ptr, out_len_bytes as u32, 8))?;

        // Record total wall time of the last search.
        self.store.data_mut().stats.total_time_ns = total_time_ns;

        Ok(results)
    }
}
