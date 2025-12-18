use anyhow::{anyhow, Context, Result};
use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::FixedSizeListArray;
use clap::Parser;
use fff_poc::reader::{FileReaderV2Builder, Projection, Selection};
use fff_vindex::artifact::ivf_flat::{
    build_ivf_flat_artifact_with_options, load_ivf_flat_artifact, IvfFlatArtifactBuildOptions,
    IvfFlatArtifactSearcher, PostingCodec,
};
use fff_vindex::artifact::wasm_ivf_flat::WasmIvfFlatKernel;
use fff_vindex::ivf_flat::{
    build_ivf_flat_sidecar, load_ivf_flat_index, search_ivf_flat, IvfFlatBuildOptions, SearchResult,
};
use serde_json::json;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    base_f3: PathBuf,

    /// Leaf column index of the vector column in the base file schema.
    #[arg(long)]
    vector_leaf_index: usize,

    /// Vector dimension.
    #[arg(long)]
    dim: usize,

    #[arg(long, default_value = "emb_ivf_flat")]
    index_name: String,

    #[arg(long, default_value_t = 1024)]
    nlist: usize,

    #[arg(long, default_value_t = 200_000)]
    train_sample: usize,

    #[arg(long, default_value_t = 1)]
    seed: u64,

    #[arg(long, default_value_t = 20)]
    max_kmeans_iters: usize,

    #[arg(long, default_value_t = 10)]
    k: usize,

    #[arg(long, default_value_t = 8)]
    nprobe: usize,

    /// Number of queries; queries are read from rows `[0..nq)`.
    #[arg(long, default_value_t = 1)]
    nq: usize,

    /// Build and query the IVF artifact container instead of the legacy ivf_flat file.
    #[arg(long, default_value_t = false)]
    artifact: bool,

    /// Run IVF search via a Wasm kernel (path to the `.wasm` module). Implies `--artifact`.
    #[arg(long)]
    artifact_wasm_kernel: Option<PathBuf>,

    /// Posting codec for artifact: `raw`, `row_id_delta_varint_v1`, `raw_f16`, `row_id_delta_varint_v1_f16`.
    #[arg(long, default_value = "raw")]
    artifact_posting_codec: String,

    /// Enable decoded posting cache for native artifact searcher.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    artifact_native_decoded_cache: bool,

    /// Clear native decoded cache before each measured iteration (forces cold behavior).
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    artifact_native_clear_each_iter: bool,

    /// Disable host-side chunk cache for wasm kernel (forces reads each time).
    #[arg(long, default_value_t = false)]
    artifact_wasm_no_cache: bool,

    /// Let the Wasm kernel call back into host for the L2 distance batch kernel (hybrid mode).
    #[arg(long, default_value_t = false)]
    artifact_wasm_host_dist: bool,

    /// Decoded posting cache budget inside Wasm kernel (bytes). Set 0 to disable.
    #[arg(long, default_value_t = 201_326_592)]
    artifact_wasm_decoded_cache_bytes: u32,

    /// Warmup iterations (builds index once, runs search multiple times).
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Repeated iterations to measure.
    #[arg(long, default_value_t = 5)]
    repeat: usize,

    /// Print per-iteration stats as JSON lines.
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Collect per-stage timing breakdown (adds overhead; intended for diagnosis, not peak perf).
    #[arg(long, default_value_t = false)]
    profile_stages: bool,

    /// Do not print the top-k results (only print stats).
    #[arg(long, default_value_t = false)]
    quiet: bool,

    /// Compute recall@k via brute-force scan over base vectors (can be expensive).
    #[arg(long, default_value_t = false)]
    recall: bool,

    /// Number of queries used for recall computation (defaults to min(nq, 10)).
    #[arg(long)]
    recall_queries: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let queries = read_first_vectors(&args.base_f3, args.vector_leaf_index, args.dim, args.nq)?;
    let recall_queries = args.recall_queries.unwrap_or(args.nq.min(10));
    let all_vectors = if args.recall {
        Some(read_all_vectors(
            &args.base_f3,
            args.vector_leaf_index,
            args.dim,
        )?)
    } else {
        None
    };

    let build_opts = IvfFlatBuildOptions {
        nlist: args.nlist,
        train_sample: args.train_sample,
        seed: args.seed,
        max_kmeans_iters: args.max_kmeans_iters,
    };

    let (index_path, results_batch) = if let Some(wasm_path) = &args.artifact_wasm_kernel {
        let posting_codec = match args.artifact_posting_codec.as_str() {
            "raw" => PostingCodec::Raw,
            "row_id_delta_varint_v1" => PostingCodec::RowIdDeltaVarintV1,
            "raw_f16" => PostingCodec::RawF16,
            "row_id_delta_varint_v1_f16" => PostingCodec::RowIdDeltaVarintV1F16,
            "raw_u8" => PostingCodec::RawU8,
            "row_id_delta_varint_v1_u8" => PostingCodec::RowIdDeltaVarintV1U8,
            other => return Err(anyhow!("unsupported --artifact-posting-codec: {other}")),
        };
        let index_path = build_ivf_flat_artifact_with_options(
            &args.base_f3,
            args.vector_leaf_index,
            args.dim,
            &args.index_name,
            build_opts,
            IvfFlatArtifactBuildOptions { posting_codec },
        )
        .with_context(|| "build ivf-flat artifact")?;
        if args.json {
            let index_bytes = std::fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0);
            println!(
                "{}",
                json!({
                    "event": "meta",
                    "engine": "wasm",
                    "base_f3": args.base_f3.to_string_lossy(),
                    "vector_leaf_index": args.vector_leaf_index,
                    "dim": args.dim,
                    "index_name": args.index_name,
                    "nlist": args.nlist,
                    "train_sample": args.train_sample,
                    "seed": args.seed,
                    "max_kmeans_iters": args.max_kmeans_iters,
                    "index_path": index_path.to_string_lossy(),
                    "index_bytes": index_bytes,
                    "nq": args.nq,
                    "k": args.k,
                    "nprobe": args.nprobe,
                    "posting_codec": args.artifact_posting_codec,
                    "cache_enabled": !args.artifact_wasm_no_cache,
                    "host_dist": args.artifact_wasm_host_dist,
                    "decoded_cache_budget_bytes": args.artifact_wasm_decoded_cache_bytes,
                    "profile_stages": args.profile_stages,
                })
            );
        }
        let artifact = load_ivf_flat_artifact(&index_path)?;
        let artifact = Arc::new(artifact);
        let mut kernel = WasmIvfFlatKernel::load(wasm_path, Arc::clone(&artifact))
            .with_context(|| "load wasm ivf-flat kernel")?;
        kernel.set_cache_enabled(!args.artifact_wasm_no_cache);
        kernel.set_use_host_dist(args.artifact_wasm_host_dist);
        kernel.set_decoded_cache_budget_bytes(args.artifact_wasm_decoded_cache_bytes);
        kernel.set_profile_stages(args.profile_stages);
        let nq = args.nq;

        // Warmup
        for _ in 0..args.warmup {
            kernel.reset_stats();
            let _ = if nq == 1 {
                kernel.search(&artifact, &queries, args.k, args.nprobe)?
            } else {
                kernel.search_batch(&artifact, &queries, nq, args.k, args.nprobe)?
                    .into_iter()
                    .next()
                    .unwrap_or_default()
            };
        }

        // Measure
        let mut last_results_batch: Vec<Vec<SearchResult>> = Vec::new();
        for it in 0..args.repeat {
            kernel.reset_stats();
            let run_start = Instant::now();
            let results_batch = if nq == 1 {
                vec![kernel.search(&artifact, &queries, args.k, args.nprobe)?]
            } else {
                kernel.search_batch(&artifact, &queries, nq, args.k, args.nprobe)?
            };
            let wall_ms = run_start.elapsed().as_secs_f64() * 1000.0;
            let stats = kernel.stats();
            let kstats = kernel.kernel_stats();
            if args.json {
                println!(
                    "{}",
                    json!({
                        "engine": "wasm",
                        "iteration": it,
                        "base_f3": args.base_f3.to_string_lossy(),
                        "vector_leaf_index": args.vector_leaf_index,
                        "dim": args.dim,
                        "index_name": args.index_name,
                        "nlist": args.nlist,
                        "train_sample": args.train_sample,
                        "seed": args.seed,
                        "max_kmeans_iters": args.max_kmeans_iters,
                        "nq": nq,
                        "k": args.k,
                        "nprobe": args.nprobe,
                        "posting_codec": args.artifact_posting_codec,
                        "cache_enabled": !args.artifact_wasm_no_cache,
                        "host_dist": args.artifact_wasm_host_dist,
                        "decoded_cache_budget_bytes": args.artifact_wasm_decoded_cache_bytes,
                        "profile_stages": args.profile_stages,
                        "wall_ms": wall_ms,
                        "kernel_total_ms": (stats.total_time_ns as f64) / 1e6,
                        "fetch_ms": (stats.fetch_time_ns as f64) / 1e6,
                        "transfer_ms": (stats.transfer_time_ns as f64) / 1e6,
                        "decode_ms": (kstats.decode_time_ns as f64) / 1e6,
                        "compute_ms": (kstats.compute_time_ns as f64) / 1e6,
                        "centroid_ms": (kstats.centroid_time_ns as f64) / 1e6,
                        "dist_ms": (kstats.dist_time_ns as f64) / 1e6,
                        "heap_ms": (kstats.heap_time_ns as f64) / 1e6,
                        "decoded_cache_hits": kstats.decoded_cache_hits,
                        "decoded_cache_misses": kstats.decoded_cache_misses,
                        "decoded_cache_bytes": kstats.decoded_cache_bytes,
                        "cache_hits": stats.cache_hits,
                        "chunks_fetched": stats.chunks_fetched,
                        "compressed_bytes_in": stats.compressed_bytes_in,
                        "raw_bytes_decoded": stats.raw_bytes_decoded,
                        "chunks_sample": stats.fetched_chunk_ids_sample,
                        "chunks_miss_sample": stats.cache_miss_chunk_ids_sample,
                    })
                );
            } else {
                println!(
                    "wasm it={} wall_ms={:.3} kernel_ms={:.3} fetch_ms={:.3} decode_ms={:.3} compute_ms={:.3} cache_hits={} fetched={} in={} raw={}",
                    it,
                    wall_ms,
                    (stats.total_time_ns as f64) / 1e6,
                    (stats.fetch_time_ns as f64) / 1e6,
                    (kstats.decode_time_ns as f64) / 1e6,
                    (kstats.compute_time_ns as f64) / 1e6,
                    stats.cache_hits,
                    stats.chunks_fetched,
                    stats.compressed_bytes_in,
                    stats.raw_bytes_decoded,
                );
            }
            last_results_batch = results_batch;
        }

        (index_path, last_results_batch)
    } else if args.artifact {
        let posting_codec = match args.artifact_posting_codec.as_str() {
            "raw" => PostingCodec::Raw,
            "row_id_delta_varint_v1" => PostingCodec::RowIdDeltaVarintV1,
            "raw_f16" => PostingCodec::RawF16,
            "row_id_delta_varint_v1_f16" => PostingCodec::RowIdDeltaVarintV1F16,
            "raw_u8" => PostingCodec::RawU8,
            "row_id_delta_varint_v1_u8" => PostingCodec::RowIdDeltaVarintV1U8,
            other => return Err(anyhow!("unsupported --artifact-posting-codec: {other}")),
        };
        let index_path = build_ivf_flat_artifact_with_options(
            &args.base_f3,
            args.vector_leaf_index,
            args.dim,
            &args.index_name,
            build_opts,
            IvfFlatArtifactBuildOptions { posting_codec },
        )
        .with_context(|| "build ivf-flat artifact")?;
        if args.json {
            let index_bytes = std::fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0);
            println!(
                "{}",
                json!({
                    "event": "meta",
                    "engine": "native_artifact",
                    "base_f3": args.base_f3.to_string_lossy(),
                    "vector_leaf_index": args.vector_leaf_index,
                    "dim": args.dim,
                    "index_name": args.index_name,
                    "nlist": args.nlist,
                    "train_sample": args.train_sample,
                    "seed": args.seed,
                    "max_kmeans_iters": args.max_kmeans_iters,
                    "index_path": index_path.to_string_lossy(),
                    "index_bytes": index_bytes,
                    "nq": args.nq,
                    "k": args.k,
                    "nprobe": args.nprobe,
                    "posting_codec": args.artifact_posting_codec,
                    "profile_stages": args.profile_stages,
                })
            );
        }
        let searcher =
            IvfFlatArtifactSearcher::open_with_options(&index_path, args.artifact_native_decoded_cache)?;
        for _ in 0..args.warmup {
            if args.artifact_native_clear_each_iter {
                searcher.clear_posting_cache();
            }
            for q in 0..args.nq {
                let query = &queries[q * args.dim..(q + 1) * args.dim];
                if args.profile_stages {
                    let _ = searcher.search_profiled(query, args.k, args.nprobe)?;
                } else {
                    let _ = searcher.search(query, args.k, args.nprobe)?;
                }
            }
        }
        let mut last = Vec::new();
        let mut last_results_batch: Vec<Vec<SearchResult>> = Vec::new();
        for it in 0..args.repeat {
            if args.artifact_native_clear_each_iter {
                searcher.clear_posting_cache();
            }
            let run_start = Instant::now();
            let mut centroid_ns: u64 = 0;
            let mut decode_ns: u64 = 0;
            let mut dist_ns: u64 = 0;
            let mut heap_ns: u64 = 0;
            let mut cache_hits: u64 = 0;
            let mut cache_misses: u64 = 0;
            let mut iter_results_batch: Vec<Vec<SearchResult>> = Vec::new();
            for q in 0..args.nq {
                let query = &queries[q * args.dim..(q + 1) * args.dim];
                if args.profile_stages {
                    let (r, s) = searcher.search_profiled(query, args.k, args.nprobe)?;
                    if args.recall {
                        iter_results_batch.push(r);
                    } else {
                        last = r;
                    }
                    centroid_ns += s.centroid_ns;
                    decode_ns += s.posting_decode_ns;
                    dist_ns += s.dist_ns;
                    heap_ns += s.heap_ns;
                    cache_hits += s.posting_cache_hits;
                    cache_misses += s.posting_cache_misses;
                } else {
                    let r = searcher.search(query, args.k, args.nprobe)?;
                    if args.recall {
                        iter_results_batch.push(r);
                    } else {
                        last = r;
                    }
                }
            }
            let wall_ms = run_start.elapsed().as_secs_f64() * 1000.0;
            if args.json {
                let compute_ns = centroid_ns
                    .saturating_add(decode_ns)
                    .saturating_add(dist_ns)
                    .saturating_add(heap_ns);
                println!(
                    "{}",
                    json!({
                        "engine": "native_artifact",
                        "iteration": it,
                        "base_f3": args.base_f3.to_string_lossy(),
                        "vector_leaf_index": args.vector_leaf_index,
                        "dim": args.dim,
                        "index_name": args.index_name,
                        "nlist": args.nlist,
                        "train_sample": args.train_sample,
                        "seed": args.seed,
                        "max_kmeans_iters": args.max_kmeans_iters,
                        "nq": args.nq,
                        "k": args.k,
                        "nprobe": args.nprobe,
                        "posting_codec": args.artifact_posting_codec,
                        "profile_stages": args.profile_stages,
                        "wall_ms": wall_ms,
                        "centroid_ms": (centroid_ns as f64) / 1e6,
                        "decode_ms": (decode_ns as f64) / 1e6,
                        "dist_ms": (dist_ns as f64) / 1e6,
                        "heap_ms": (heap_ns as f64) / 1e6,
                        "compute_ms": (compute_ns as f64) / 1e6,
                        "posting_cache_hits": cache_hits,
                        "posting_cache_misses": cache_misses,
                    })
                );
            } else {
                println!("native_artifact it={} wall_ms={:.3}", it, wall_ms);
            }
            if args.recall {
                last_results_batch = iter_results_batch;
            }
        }
        if args.recall {
            (index_path, last_results_batch)
        } else {
            (index_path, vec![last])
        }
    } else {
        let index_path = build_ivf_flat_sidecar(
            &args.base_f3,
            args.vector_leaf_index,
            args.dim,
            &args.index_name,
            build_opts,
        )
        .with_context(|| "build ivf-flat index")?;
        if args.json {
            let index_bytes = std::fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0);
            println!(
                "{}",
                json!({
                    "event": "meta",
                    "engine": "native_sidecar",
                    "base_f3": args.base_f3.to_string_lossy(),
                    "vector_leaf_index": args.vector_leaf_index,
                    "dim": args.dim,
                    "index_name": args.index_name,
                    "nlist": args.nlist,
                    "train_sample": args.train_sample,
                    "seed": args.seed,
                    "max_kmeans_iters": args.max_kmeans_iters,
                    "index_path": index_path.to_string_lossy(),
                    "index_bytes": index_bytes,
                    "nq": args.nq,
                    "k": args.k,
                    "nprobe": args.nprobe,
                })
            );
        }
        let index = load_ivf_flat_index(&index_path)?;
        for _ in 0..args.warmup {
            for q in 0..args.nq {
                let query = &queries[q * args.dim..(q + 1) * args.dim];
                let _ = search_ivf_flat(&index, query, args.k, args.nprobe)?;
            }
        }
        let mut last = Vec::new();
        let mut last_results_batch: Vec<Vec<SearchResult>> = Vec::new();
        for it in 0..args.repeat {
            let run_start = Instant::now();
            let mut iter_results_batch: Vec<Vec<SearchResult>> = Vec::new();
            for q in 0..args.nq {
                let query = &queries[q * args.dim..(q + 1) * args.dim];
                let r = search_ivf_flat(&index, query, args.k, args.nprobe)?;
                if args.recall {
                    iter_results_batch.push(r);
                } else {
                    last = r;
                }
            }
            let wall_ms = run_start.elapsed().as_secs_f64() * 1000.0;
            if args.json {
                println!(
                    "{}",
                    json!({
                        "engine": "native_sidecar",
                        "iteration": it,
                        "base_f3": args.base_f3.to_string_lossy(),
                        "vector_leaf_index": args.vector_leaf_index,
                        "dim": args.dim,
                        "index_name": args.index_name,
                        "nlist": args.nlist,
                        "train_sample": args.train_sample,
                        "seed": args.seed,
                        "max_kmeans_iters": args.max_kmeans_iters,
                        "nq": args.nq,
                        "k": args.k,
                        "nprobe": args.nprobe,
                        "wall_ms": wall_ms,
                    })
                );
            } else {
                println!("native_sidecar it={} wall_ms={:.3}", it, wall_ms);
            }
            if args.recall {
                last_results_batch = iter_results_batch;
            }
        }
        if args.recall {
            (index_path, last_results_batch)
        } else {
            (index_path, vec![last])
        }
    };

    if args.recall {
        let Some(all_vectors) = &all_vectors else {
            return Err(anyhow!("internal: recall enabled but base vectors not loaded"));
        };
        if results_batch.len() < recall_queries {
            return Err(anyhow!(
                "not enough results for recall: have {} queries, need {}",
                results_batch.len(),
                recall_queries
            ));
        }
        let recall = compute_recall_at_k(
            &results_batch,
            all_vectors,
            args.dim,
            &queries,
            args.k,
            recall_queries,
        );
        if args.json {
            println!(
                "{}",
                json!({
                    "event": "recall",
                    "nq": args.nq,
                    "recall_queries": recall_queries,
                    "k": args.k,
                    "nprobe": args.nprobe,
                    "posting_codec": args.artifact_posting_codec,
                    "recall_at_k": recall,
                    "base_rows": all_vectors.len() / args.dim,
                    "dim": args.dim,
                })
            );
        } else {
            println!("recall@{} over {} queries: {:.4}", args.k, recall_queries, recall);
        }
    }

    if !args.quiet {
        println!("index: {}", index_path.display());
        let display = results_batch.first().cloned().unwrap_or_default();
        println!("top{} (nprobe={}):", display.len(), args.nprobe);
        for r in display {
            println!("row_id={} dist={}", r.row_id, r.distance);
        }
    }
    Ok(())
}

fn read_first_vectors(base_f3: &PathBuf, leaf: usize, dim: usize, nq: usize) -> Result<Vec<f32>> {
    let file = File::open(base_f3).with_context(|| format!("open {}", base_f3.display()))?;
    let row_indexes: Vec<u64> = (0..nq as u64).collect();
    let mut reader = FileReaderV2Builder::new(Arc::new(file))
        .with_projections(Projection::All)
        .with_selection(Selection::RowIndexes(row_indexes))
        .build()
        .map_err(|e| anyhow!(e.to_string()))?;
    let batches = reader
        .read_file()
        .map_err(|e| anyhow!(e.to_string()))
        .with_context(|| "read first vector")?;
    let schema = batches
        .first()
        .ok_or_else(|| anyhow!("no batches"))?
        .schema();
    let batch = if batches.len() == 1 {
        batches[0].clone()
    } else {
        arrow::compute::concat_batches(&schema, &batches)?
    };
    if leaf >= batch.num_columns() {
        return Err(anyhow!(
            "vector_leaf_index out of range: {} >= {}",
            leaf,
            batch.num_columns()
        ));
    }
    let col = batch.column(leaf);
    let fsl = col
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| anyhow!("projected column is not FixedSizeListArray"))?;
    if fsl.value_length() as usize != dim {
        return Err(anyhow!(
            "dim mismatch: expected {}, got {}",
            dim,
            fsl.value_length()
        ));
    }
    let values = fsl.values().as_primitive::<arrow::datatypes::Float32Type>();
    let total = nq * dim;
    if values.values().len() < total {
        return Err(anyhow!(
            "not enough values for nq={}, dim={}: have {}",
            nq,
            dim,
            values.values().len()
        ));
    }
    Ok(values.values()[..total].to_vec())
}

fn read_all_vectors(base_f3: &PathBuf, leaf: usize, dim: usize) -> Result<Vec<f32>> {
    let file = File::open(base_f3).with_context(|| format!("open {}", base_f3.display()))?;
    let mut reader = FileReaderV2Builder::new(Arc::new(file))
        .with_projections(Projection::All)
        .with_selection(Selection::All)
        .build()
        .map_err(|e| anyhow!(e.to_string()))?;
    let batches = reader
        .read_file()
        .map_err(|e| anyhow!(e.to_string()))
        .with_context(|| "read all vectors")?;
    let mut out = Vec::<f32>::new();
    for batch in batches {
        if leaf >= batch.num_columns() {
            return Err(anyhow!(
                "vector_leaf_index out of range: {} >= {}",
                leaf,
                batch.num_columns()
            ));
        }
        let col = batch.column(leaf);
        let fsl = col
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| anyhow!("projected column is not FixedSizeListArray"))?;
        if fsl.value_length() as usize != dim {
            return Err(anyhow!(
                "dim mismatch: expected {}, got {}",
                dim,
                fsl.value_length()
            ));
        }
        if fsl.null_count() != 0 {
            return Err(anyhow!("vector column contains nulls"));
        }
        let values = fsl.values().as_primitive::<arrow::datatypes::Float32Type>();
        out.extend_from_slice(values.values());
    }
    if dim == 0 || out.len() % dim != 0 {
        return Err(anyhow!(
            "vector values len {} not divisible by dim {}",
            out.len(),
            dim
        ));
    }
    Ok(out)
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[derive(Copy, Clone)]
struct DistRow {
    dist: f32,
    row_id: u32,
}

impl Eq for DistRow {}

impl PartialEq for DistRow {
    fn eq(&self, other: &Self) -> bool {
        self.dist.total_cmp(&other.dist) == std::cmp::Ordering::Equal && self.row_id == other.row_id
    }
}

impl Ord for DistRow {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.row_id.cmp(&other.row_id))
    }
}

impl PartialOrd for DistRow {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn exact_topk_row_ids(vectors: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<u32> {
    let dim = dim.max(1);
    let n = vectors.len() / dim;
    let mut heap = std::collections::BinaryHeap::<DistRow>::new();
    for i in 0..n {
        let base = i * dim;
        let v = &vectors[base..base + dim];
        let d = l2_sq(query, v);
        if heap.len() < k {
            heap.push(DistRow {
                dist: d,
                row_id: i as u32,
            });
        } else if let Some(worst) = heap.peek().copied() {
            if d < worst.dist {
                let _ = heap.pop();
                heap.push(DistRow {
                    dist: d,
                    row_id: i as u32,
                });
            }
        }
    }
    let mut out: Vec<DistRow> = heap.into_iter().collect();
    out.sort_by(|a, b| a.dist.total_cmp(&b.dist));
    out.into_iter().map(|x| x.row_id).collect()
}

fn compute_recall_at_k(
    approx: &[Vec<SearchResult>],
    base_vectors: &[f32],
    dim: usize,
    queries: &[f32],
    k: usize,
    recall_queries: usize,
) -> f32 {
    let mut total = 0.0f32;
    let rq = recall_queries.min(approx.len());
    for qi in 0..rq {
        let query = &queries[qi * dim..(qi + 1) * dim];
        let gt = exact_topk_row_ids(base_vectors, dim, query, k);
        let gt_set: std::collections::HashSet<u32> = gt.into_iter().collect();
        let approx_set: std::collections::HashSet<u32> =
            approx[qi].iter().map(|r| r.row_id).collect();
        let hit = gt_set.intersection(&approx_set).count();
        total += (hit as f32) / (k as f32);
    }
    total / (rq as f32)
}
