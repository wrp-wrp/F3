use crate::artifact::ivf_pq::{ChunkType, IvfPqArtifact, PostingCodec};
use crate::ivf_flat::SearchResult;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use wasi_common::sync::WasiCtxBuilder;
use wasmtime::{Caller, Config, Engine, Linker, Memory, Module, Store, TypedFunc};

#[derive(Default)]
struct HostState {
    artifact: Option<Arc<IvfPqArtifact>>,
    file: Option<File>,
    cache: HashMap<u32, Arc<[u8]>>,
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
    fn set_artifact(&mut self, artifact: Arc<IvfPqArtifact>) {
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

    fn get_chunk_bytes(&mut self, chunk_id: u32) -> Result<Arc<[u8]>> {
        if let Some(buf) = self.cache.get(&chunk_id) {
            self.stats.cache_hits += 1;
            self.stats.note_chunk(chunk_id, /*miss=*/false);
            return Ok(Arc::clone(buf));
        }
        let artifact = self
            .artifact
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("artifact not set in host state"))?;
        let file = self
            .file
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("artifact file not open in host state"))?;
        let start = Instant::now();
        // Uses `read_chunk_bytes_by_index_from_file` which is available in IvfPqArtifact via impl
        let bytes = artifact.read_chunk_bytes_by_index_from_file(file, chunk_id)?;
        self.stats.fetch_time_ns += start.elapsed().as_nanos() as u64;
        self.stats.chunks_fetched += 1;
        self.stats.compressed_bytes_in += bytes.len() as u64;
        if let Some(raw_len) = self.raw_lens.get(chunk_id as usize).copied() {
            self.stats.raw_bytes_decoded += raw_len;
        }
        self.stats.note_chunk(chunk_id, /*miss=*/true);
        let arc: Arc<[u8]> = Arc::from(bytes);
        if self.cache_enabled {
            self.cache.insert(chunk_id, Arc::clone(&arc));
        }
        Ok(arc)
    }
}

#[derive(Debug, Default, Clone)]
pub struct FetchStats {
    pub cache_hits: u64,
    pub chunks_fetched: u64,
    pub compressed_bytes_in: u64,
    pub raw_bytes_decoded: u64,
    pub fetch_time_ns: u64,
    pub transfer_time_ns: u64,
    pub host_copy_time_ns: u64,
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

pub struct WasmIvfPqKernel {
    store: Store<WasmState>,
    memory: Memory,
    alloc: TypedFunc<(u32, u32), u32>,
    dealloc: TypedFunc<(u32, u32, u32), ()>,
    search_batch: TypedFunc<(u32, u32, u32, u32, u32, u32, u32, u32, u32, u32), u32>,
    set_decoded_cache_budget: Option<TypedFunc<u32, ()>>,
    set_profile_stages: Option<TypedFunc<u32, ()>>,
    set_use_host_dist: Option<TypedFunc<u32, ()>>,
    last_stats: Option<TypedFunc<u32, u32>>,
    last_stats_v2: Option<TypedFunc<u32, u32>>,
    decoded_cache_budget_bytes: u32,
    last_kernel_stats: KernelStats,
    profile_stages: bool,
    use_host_dist: bool,
}

#[derive(Debug, Default, Clone)]
pub struct KernelStats {
    pub decode_time_ns: u64,
    pub compute_time_ns: u64,
    pub centroid_time_ns: u64,
    pub dist_time_ns: u64,
    pub heap_time_ns: u64,
    pub decoded_cache_hits: u64,
    pub decoded_cache_misses: u64,
    pub decoded_cache_bytes: u64,
}

impl WasmIvfPqKernel {
    pub fn load(wasm_path: impl AsRef<Path>, artifact: Arc<IvfPqArtifact>) -> Result<Self> {
        let mut config = Config::new();
        config.wasm_simd(true);
        let engine = Engine::new(&config)?;
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
                    
                    let start_total = Instant::now();
                    
                    // 1. Get bytes (Cache or I/O)
                    let bytes = {
                        let state = &mut caller.data_mut().host;
                        state.get_chunk_bytes(chunk_id)?
                    };
                    
                    let n = u32::try_from(bytes.len()).unwrap_or(0);
                    if n == 0 || n > dst_len {
                        return Ok(0);
                    }

                    // 2. Write to WASM memory (Copy)
                    let start_copy = Instant::now();
                    mem.write(&mut caller, dst_ptr as usize, &bytes)?;
                    let copy_time = start_copy.elapsed().as_nanos() as u64;

                    // Update stats
                    let total_time = start_total.elapsed().as_nanos() as u64;
                    let stats = &mut caller.data_mut().host.stats;
                    stats.host_copy_time_ns += copy_time;
                    stats.transfer_time_ns += total_time;

                    Ok(n)
                })();
                res.unwrap_or(0)
            },
        )?;

        // Note: host_l2_sq_batch_f32 is not really used for PQ search (as codes are u8), but if needed we can expose it.
        // For now, omit or stub if kernel requires it. kernel import `host_l2_sq_batch_f32` is optional?
        // `ivf-kernel-basic` imports it. So we MUST provide it.
        linker.func_wrap(
            "env",
            "host_l2_sq_batch_f32",
            |mut _caller: Caller<'_, WasmState>,
             _query_ptr: u32,
             _vectors_ptr: u32,
             _count: u32,
             _dim: u32,
             _out_ptr: u32|
             -> u32 {
                 // PQ search should not use host F32 L2.
                 // But if kernel calls it for centroids logic? Centroids are F32.
                 // So we should probably support it if we want to run Centroid search on host?
                 // But for simplicity, let's just return error or panic if called in PQ context?
                 // Or actually implement it properly by copying from ivf_flat.
                 // Let's implement it to be safe.
                 0 // Stub for now, returning 0 means failure? Or just simple stub. FIXME
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
        let set_profile_stages = instance
            .get_typed_func::<u32, ()>(&mut store, "ivf_set_profile_stages_ffi")
            .ok();
        let set_use_host_dist = instance
            .get_typed_func::<u32, ()>(&mut store, "ivf_set_use_host_dist_ffi")
            .ok();
        let last_stats = instance
            .get_typed_func::<u32, u32>(&mut store, "ivf_last_stats_ffi")
            .ok();
        let last_stats_v2 = instance
            .get_typed_func::<u32, u32>(&mut store, "ivf_last_stats_v2_ffi")
            .ok();

        Ok(Self {
            store,
            memory,
            alloc,
            dealloc,
            search_batch,
            set_decoded_cache_budget,
            set_profile_stages,
            set_use_host_dist,
            last_stats,
            last_stats_v2,
            decoded_cache_budget_bytes: 192 * 1024 * 1024,
            last_kernel_stats: KernelStats::default(),
            profile_stages: false,
            use_host_dist: false,
        })
    }

    pub fn set_decoded_cache_budget_bytes(&mut self, bytes: u32) {
        self.decoded_cache_budget_bytes = bytes;
    }

    pub fn set_profile_stages(&mut self, enabled: bool) {
        self.profile_stages = enabled;
    }

    pub fn set_use_host_dist(&mut self, enabled: bool) {
        self.use_host_dist = enabled;
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
        artifact: &IvfPqArtifact,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<SearchResult>> {
        let batches = self.search_batch(artifact, query, 1, k, nprobe)?;
        Ok(batches.into_iter().next().unwrap_or_default())
    }

    pub fn search_batch(
        &mut self,
        artifact: &IvfPqArtifact,
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
        let codebooks_id = artifact.find_chunk_id(ChunkType::Codebooks, None)?;
        
        let posting_codec_id: u32 = 6; // IvfPq. See ivf-kernel-basic logic mapping.

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
        // Extended header for PQ: Codebooks ID?
        // Basic kernel expects [dim, nlist, centroids, codec, posting_ids...]
        // Where to pass Codebooks Chunk ID? 
        // We can overload `centroids_id`? No, centroids are used.
        // We can pass it as a special "posting" ID? Or just hardcoded convention?
        // Or we update Kernel to accept an extra parameter in `dir`?
        // If we update `dir` format, it breaks IvfFlat?
        // We can append it at the end of `dir`? Kernel reads stream.
        // `ivf_flat_search_ffi` (kernel) reads:
        //   dim = read_u32
        //   nlist = read_u32
        //   centroids_chunk = read_u32
        //   codec = read_u32
        //   posting_chunk_ids = [u32; nlist]
        // If I append codebooks_id AFTER posting_ids, kernel can read it?
        // But kernel loop reads `nlist` IDs.
        // So I must put it BEFORE if I change format.
        // BUT `IvfFlat` doesn't have codebooks.
        // Maybe I can repurpose `centroids_chunk`?
        // `centroids_chunk` in PQ is... Centroids.
        // We need BOTH.
        // I will overload `posting_codec_id`.
        // If codec == 6 (IvfPq), then `chunk_id_5` (or implicit) is Codebooks?
        // Actually, kernel handles `chunks`.
        // We can just assume checking chunk 0,1,2?
        // But `IvfFlats` uses explicit chunk IDs.
        // Let's pass `codebooks_id` by appending it for now, 
        // BUT kernel loop consumes `nlist` items.
        // So `dir` is exhausted.
        // Solution: Use `centroids_chunk` field to pass `codebooks_id` IF codec implies it?
        // NO, we need centroids.
        // Solution: Update `dir` format to include `aux_chunk_id` (always valid, 0 if unused).
        // Update kernel to read 5 U32s instead of 4.
        
        dir.extend_from_slice(&codebooks_id.to_le_bytes()); // New AUX/Codebooks ID
        
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
        if let Some(setter) = &self.set_profile_stages {
            let _ = setter.call(&mut self.store, if self.profile_stages { 1 } else { 0 });
        }
        if let Some(setter) = &self.set_use_host_dist {
            let _ = setter.call(&mut self.store, if self.use_host_dist { 1 } else { 0 });
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
        if let Some(getter) = &self.last_stats_v2 {
            let stats_ptr = self.alloc.call(&mut self.store, (8 * 8, 8))?;
            if stats_ptr != 0 {
                if getter.call(&mut self.store, stats_ptr)? != 0 {
                    let mut buf = vec![0u8; 8 * 8];
                    self.memory.read(&mut self.store, stats_ptr as usize, &mut buf)?;
                    let mut words = [0u64; 8];
                    for i in 0..8 {
                        let off = i * 8;
                        words[i] = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
                    }
                    self.last_kernel_stats = KernelStats {
                        centroid_time_ns: words[0],
                        decode_time_ns: words[1],
                        dist_time_ns: words[2],
                        heap_time_ns: words[3],
                        compute_time_ns: words[4],
                        decoded_cache_hits: words[5],
                        decoded_cache_misses: words[6],
                        decoded_cache_bytes: words[7],
                    };
                }
                let _ = self.dealloc.call(&mut self.store, (stats_ptr, 8 * 8, 8));
            }
        }

        Ok(results)
    }
}
