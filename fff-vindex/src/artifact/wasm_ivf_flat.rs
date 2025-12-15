use crate::artifact::ivf_flat::{ChunkType, IvfFlatArtifact, PostingCodec};
use crate::ivf_flat::SearchResult;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use wasmtime::{Caller, Engine, Linker, Memory, Module, Store, TypedFunc};

#[derive(Default)]
struct HostState {
    artifact: Option<Arc<IvfFlatArtifact>>,
    cache: HashMap<u32, Vec<u8>>,
    raw_lens: Vec<u64>,
    stats: FetchStats,
}

impl HostState {
    fn set_artifact(&mut self, artifact: Arc<IvfFlatArtifact>) {
        self.raw_lens = artifact.footer().chunks.iter().map(|c| c.raw_len).collect();
        self.artifact = Some(artifact);
        self.cache.clear();
        self.stats = FetchStats::default();
    }

    fn get_or_load(&mut self, chunk_id: u32) -> Result<&[u8]> {
        if self.cache.contains_key(&chunk_id) {
            return Ok(self.cache.get(&chunk_id).unwrap().as_slice());
        }
        let artifact = self
            .artifact
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("artifact not set in host state"))?;
        let bytes = artifact.read_chunk_bytes_by_index(chunk_id)?;
        self.stats.chunks_fetched += 1;
        self.stats.compressed_bytes_in += bytes.len() as u64;
        if let Some(raw_len) = self.raw_lens.get(chunk_id as usize).copied() {
            self.stats.raw_bytes_decoded += raw_len;
        }
        self.cache.insert(chunk_id, bytes);
        Ok(self.cache.get(&chunk_id).unwrap().as_slice())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FetchStats {
    pub chunks_fetched: u64,
    pub compressed_bytes_in: u64,
    pub raw_bytes_decoded: u64,
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
            match state.get_or_load(chunk_id) {
                Ok(bytes) => u32::try_from(bytes.len()).unwrap_or(0),
                Err(_) => 0,
            }
        })?;

        linker.func_wrap(
            "env",
            "host_read_chunk",
            |mut caller: Caller<'_, HostState>, chunk_id: u32, dst_ptr: u32, dst_len: u32| -> u32 {
                let res: Result<u32> = (|| {
                    let bytes = {
                        let state = caller.data_mut();
                        state.get_or_load(chunk_id)?.to_vec()
                    };
                    if bytes.len() > (dst_len as usize) {
                        return Ok(0);
                    }
                    let mem = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| anyhow::anyhow!("wasm export `memory` not found"))?;
                    mem.write(&mut caller, dst_ptr as usize, &bytes)?;
                    Ok(bytes.len() as u32)
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

    pub fn stats(&self) -> FetchStats {
        self.store.data().stats
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

        Ok(results)
    }
}
