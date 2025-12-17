use anyhow::{anyhow, Context, Result};
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
    build_ivf_flat_sidecar, load_ivf_flat_index, search_ivf_flat, IvfFlatBuildOptions,
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
}

fn main() -> Result<()> {
    let args = Args::parse();

    let queries = read_first_vectors(&args.base_f3, args.vector_leaf_index, args.dim, args.nq)?;

    let build_opts = IvfFlatBuildOptions {
        nlist: args.nlist,
        train_sample: args.train_sample,
        seed: args.seed,
        max_kmeans_iters: args.max_kmeans_iters,
    };

    let (index_path, results) = if let Some(wasm_path) = &args.artifact_wasm_kernel {
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
        let mut last_results = Vec::new();
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
            last_results = results_batch.into_iter().next().unwrap_or_default();
        }

        (index_path, last_results)
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
            for q in 0..args.nq {
                let query = &queries[q * args.dim..(q + 1) * args.dim];
                if args.profile_stages {
                    let (r, s) = searcher.search_profiled(query, args.k, args.nprobe)?;
                    last = r;
                    centroid_ns += s.centroid_ns;
                    decode_ns += s.posting_decode_ns;
                    dist_ns += s.dist_ns;
                    heap_ns += s.heap_ns;
                    cache_hits += s.posting_cache_hits;
                    cache_misses += s.posting_cache_misses;
                } else {
                    last = searcher.search(query, args.k, args.nprobe)?;
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
        }
        (index_path, last)
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
        for it in 0..args.repeat {
            let run_start = Instant::now();
            for q in 0..args.nq {
                let query = &queries[q * args.dim..(q + 1) * args.dim];
                last = search_ivf_flat(&index, query, args.k, args.nprobe)?;
            }
            let wall_ms = run_start.elapsed().as_secs_f64() * 1000.0;
            if args.json {
                println!(
                    "{}",
                    json!({
                        "engine": "native_sidecar",
                        "iteration": it,
                        "nq": args.nq,
                        "k": args.k,
                        "nprobe": args.nprobe,
                        "wall_ms": wall_ms,
                    })
                );
            } else {
                println!("native_sidecar it={} wall_ms={:.3}", it, wall_ms);
            }
        }
        (index_path, last)
    };

    if !args.quiet {
        println!("index: {}", index_path.display());
        println!("top{} (nprobe={}):", results.len(), args.nprobe);
        for r in results {
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
