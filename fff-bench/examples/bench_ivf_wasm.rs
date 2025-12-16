use clap::Parser;
use fff_vindex::artifact::ivf_flat::{IvfFlatArtifact, IvfFlatArtifactSearcher};
use fff_vindex::artifact::wasm_ivf_flat::{WasmIvfFlatKernel, FetchStats, KernelStats};
use fff_vindex::artifact::ivf_flat::{IvfFlatArtifactBuildOptions, PostingCodec};
use fff_vindex::ivf_flat::IvfFlatBuildOptions;
use fff_vindex::manifest::IndexManifest;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use rand::prelude::*;
use anyhow::{anyhow, Result};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    base_path: PathBuf,

    #[arg(short, long)]
    index_name: String,

    #[arg(short, long)]
    wasm_path: PathBuf,

    #[arg(long, default_value_t = 10)]
    nq: usize,

    #[arg(long, default_value_t = 10)]
    k: usize,

    #[arg(long, default_value_t = 10)]
    nprobe: usize,

    #[arg(long, default_value_t = 5)]
    warmup: usize,

    #[arg(long, default_value_t = 20)]
    iters: usize,

    #[arg(long)]
    build_if_missing: bool,

    #[arg(long, default_value_t = 1024)]
    build_nlist: usize,
    
    #[arg(long, default_value_t = 128)]
    dim: usize,

    #[arg(long)]
    csv: bool,

    #[arg(long, default_value = "raw")]
    posting_codec: String,
}

fn main() -> anyhow::Result<()> {
    // env_logger::init(); // Removed
    let args = Args::parse();

    // 1. Check/Build Index
    let index_dir = PathBuf::from(format!("{}.vindex", args.base_path.display()));
    let manifest_path = index_dir.join("manifest.json");
    
    if args.build_if_missing {
        let need_build = if !manifest_path.exists() {
            true
        } else {
            let manifest = IndexManifest::load(&manifest_path)?;
            manifest.get_by_name(&args.index_name).is_err()
        };

        if need_build {
            if !args.csv {
                println!("Building index {}...", args.index_name);
            }
            let opts = IvfFlatBuildOptions {
                nlist: args.build_nlist,
                train_sample: 50 * args.build_nlist, // Heuristic
                ..Default::default()
            };
            // Assuming the vector column is the last one or we know the index. 
            // For now, let's assume it is 0 if we generated the file specifically for this.
            // Or we try to find a suitable column. 
            // In a real scenario, we might want to pass this as arg.
            // Using 0 as default leaf index for now.
            let posting_codec = match args.posting_codec.as_str() {
                "raw" => PostingCodec::Raw,
                "row_id_delta_varint_v1" => PostingCodec::RowIdDeltaVarintV1,
                "raw_f16" => PostingCodec::RawF16,
                "row_id_delta_varint_v1_f16" => PostingCodec::RowIdDeltaVarintV1F16,
                _ => return Err(anyhow::anyhow!("unknown posting codec: {}", args.posting_codec)),
            };
            let artifact_opts = IvfFlatArtifactBuildOptions {
                posting_codec,
            };
            fff_vindex::artifact::ivf_flat::build_ivf_flat_artifact_with_options(&args.base_path, 0, args.dim, &args.index_name, opts, artifact_opts)?;
        }
    }

    let manifest = IndexManifest::load(&manifest_path)?;
    let entry = manifest.get_by_name(&args.index_name)?;

    let index_path = entry.path.clone();

    if !args.csv {
        println!("Loading index from {}", index_path.display());
    }

    // 2. Load Native Index (for baseline)
    let start = Instant::now();
    let native_searcher = IvfFlatArtifactSearcher::open(&index_path)?;
    let native_load_time = start.elapsed();
    if !args.csv {
        println!("Native artifact load time: {:?}", native_load_time);
    }

    // 3. Load Artifact & WASM Kernel
    let start = Instant::now();
    let artifact = Arc::new(IvfFlatArtifact::open(&index_path)?);
    let artifact_load_time = start.elapsed();
    if !args.csv {
        println!("Artifact open time: {:?}", artifact_load_time);
    }

    let start = Instant::now();
    let mut kernel = WasmIvfFlatKernel::load(&args.wasm_path, artifact.clone())?;
    let wasm_load_time = start.elapsed();
    if !args.csv {
        println!("WASM kernel load time: {:?}", wasm_load_time);
    }

    // Generate random queries
    let mut rng = StdRng::seed_from_u64(42);
    let mut queries = vec![0.0f32; args.nq * args.dim];
    // Generate normalized queries if metric is cosine, but here we assume L2
    for x in queries.iter_mut() {
        *x = rng.gen();
    }

    // Benchmark Native
    let mut native_latencies = Vec::with_capacity(args.iters);
    for i in 0..(args.warmup + args.iters) {
        let start = Instant::now();
        // Since native `search_ivf_flat` is single query, we loop
        for q in 0..args.nq {
            let q_vec = &queries[q * args.dim..(q + 1) * args.dim];
            let _ = native_searcher.search(q_vec, args.k, args.nprobe)?;
        }
        let elapsed = start.elapsed();
        if i >= args.warmup {
            native_latencies.push(elapsed);
        }
    }
    let native_p50 = native_latencies[args.iters / 2];
    let native_avg = native_latencies.iter().sum::<Duration>() / (args.iters as u32);
    if !args.csv {
        println!("Native Search: Avg={:?}, P50={:?}", native_avg, native_p50);
    }


    // Benchmark WASM (Cold Cache)
    // We can simulate cold cache by disabling cache or clearing it before each run
    let mut wasm_cold_latencies = Vec::with_capacity(args.iters);
    let mut wasm_cold_stats = FetchStats::default();
    
    for i in 0..(args.warmup + args.iters) {
        kernel.set_cache_enabled(false); // Disable cache to force fetch/decode
        kernel.reset_stats();
        
        let start = Instant::now();
        let _ = kernel.search_batch(&artifact, &queries, args.nq, args.k, args.nprobe)?;
        let elapsed = start.elapsed();
        
        if i >= args.warmup {
            wasm_cold_latencies.push(elapsed);
            let s = kernel.stats();
            wasm_cold_stats.chunks_fetched += s.chunks_fetched;
            wasm_cold_stats.compressed_bytes_in += s.compressed_bytes_in;
            wasm_cold_stats.raw_bytes_decoded += s.raw_bytes_decoded;
            wasm_cold_stats.fetch_time_ns += s.fetch_time_ns;
            wasm_cold_stats.transfer_time_ns += s.transfer_time_ns;
            wasm_cold_stats.host_copy_time_ns += s.host_copy_time_ns;
            wasm_cold_stats.total_time_ns += s.total_time_ns;
        }
    }
    let wasm_cold_avg = wasm_cold_latencies.iter().sum::<Duration>() / (args.iters as u32);
    if !args.csv {
        println!("WASM Cold Search: Avg={:?}", wasm_cold_avg);
    }


    // Benchmark WASM (Warm Cache)
    // Enable cache and warm it up
    kernel.set_cache_enabled(true);
    // Initial run to warm up
    let _ = kernel.search_batch(&artifact, &queries, args.nq, args.k, args.nprobe)?;

    let mut wasm_warm_latencies = Vec::with_capacity(args.iters);
    let mut wasm_warm_stats = FetchStats::default();
    let mut wasm_warm_kernel_stats = KernelStats::default();

    for i in 0..(args.warmup + args.iters) {
        kernel.reset_stats();
        let start = Instant::now();
        let _ = kernel.search_batch(&artifact, &queries, args.nq, args.k, args.nprobe)?;
        let elapsed = start.elapsed();
        
        if i >= args.warmup {
            wasm_warm_latencies.push(elapsed);
            let s = kernel.stats();
            let ks = kernel.kernel_stats();

            wasm_warm_stats.chunks_fetched += s.chunks_fetched;
            wasm_warm_stats.cache_hits += s.cache_hits; // Host cache hits
            wasm_warm_stats.total_time_ns += s.total_time_ns;
            wasm_warm_stats.host_copy_time_ns += s.host_copy_time_ns;
            
            wasm_warm_kernel_stats.decode_time_ns += ks.decode_time_ns;
            wasm_warm_kernel_stats.compute_time_ns += ks.compute_time_ns;
            wasm_warm_kernel_stats.dist_time_ns += ks.dist_time_ns;
            wasm_warm_kernel_stats.heap_time_ns += ks.heap_time_ns;
            wasm_warm_kernel_stats.decoded_cache_hits += ks.decoded_cache_hits;
            wasm_warm_kernel_stats.decoded_cache_misses += ks.decoded_cache_misses;
        }
    }
    let wasm_warm_avg = wasm_warm_latencies.iter().sum::<Duration>() / (args.iters as u32);
    if !args.csv {
        println!("WASM Warm Search: Avg={:?}", wasm_warm_avg);
    }

    // Benchmark WASM (Warm Cache + Host Dist)
    kernel.set_cache_enabled(true);
    kernel.set_use_host_dist(true);
    // Warmup run
    let _ = kernel.search_batch(&artifact, &queries, args.nq, args.k, args.nprobe)?;

    let mut wasm_hostdist_latencies = Vec::with_capacity(args.iters);
    let mut wasm_hostdist_stats = FetchStats::default();
    let mut wasm_hostdist_kernel_stats = KernelStats::default();

    for i in 0..(args.warmup + args.iters) {
        kernel.reset_stats();
        let start = Instant::now();
        let _ = kernel.search_batch(&artifact, &queries, args.nq, args.k, args.nprobe)?;
        let elapsed = start.elapsed();
        
        if i >= args.warmup {
            wasm_hostdist_latencies.push(elapsed);
            let s = kernel.stats();
            let ks = kernel.kernel_stats();

            wasm_hostdist_stats.chunks_fetched += s.chunks_fetched;
            wasm_hostdist_stats.cache_hits += s.cache_hits;
            wasm_hostdist_stats.total_time_ns += s.total_time_ns;
            wasm_hostdist_stats.host_copy_time_ns += s.host_copy_time_ns;
            
            wasm_hostdist_kernel_stats.decode_time_ns += ks.decode_time_ns;
            wasm_hostdist_kernel_stats.compute_time_ns += ks.compute_time_ns;
            wasm_hostdist_kernel_stats.dist_time_ns += ks.dist_time_ns;
            wasm_hostdist_kernel_stats.heap_time_ns += ks.heap_time_ns;
            wasm_hostdist_kernel_stats.decoded_cache_hits += ks.decoded_cache_hits;
            wasm_hostdist_kernel_stats.decoded_cache_misses += ks.decoded_cache_misses;
        }
    }
    let wasm_hostdist_avg = wasm_hostdist_latencies.iter().sum::<Duration>() / (args.iters as u32);
    if !args.csv {
        println!("WASM Warm+HostDist Search: Avg={:?}", wasm_hostdist_avg);
    }


        if args.csv {


            println!("native,{},{},{},0,0,0,0,0,{}", 
            args.nq, args.nprobe, native_avg.as_secs_f64()*1000.0, args.posting_codec);
        println!("wasm_cold,{},{},{},{},0,0,0,{},{}", 
            args.nq, args.nprobe, wasm_cold_avg.as_secs_f64()*1000.0,
            wasm_cold_stats.chunks_fetched as f64 / args.iters as f64, 
            (wasm_cold_stats.host_copy_time_ns as f64 / args.iters as f64) / 1_000_000.0,
            args.posting_codec);
        println!("wasm_warm,{},{},{},{},{},{},{},{},{}", 
            args.nq, args.nprobe, wasm_warm_avg.as_secs_f64()*1000.0,
            wasm_warm_stats.chunks_fetched as f64 / args.iters as f64,
            wasm_warm_kernel_stats.decoded_cache_hits as f64 / args.iters as f64,
            (wasm_warm_kernel_stats.decode_time_ns as f64 / args.iters as f64) / 1_000_000.0,
            (wasm_warm_kernel_stats.compute_time_ns as f64 / args.iters as f64) / 1_000_000.0,
            (wasm_warm_stats.host_copy_time_ns as f64 / args.iters as f64) / 1_000_000.0,
            args.posting_codec
        );
        println!("wasm_warm_hostdist,{},{},{},{},{},{},{},{},{}", 
            args.nq, args.nprobe, wasm_hostdist_avg.as_secs_f64()*1000.0,
            wasm_hostdist_stats.chunks_fetched as f64 / args.iters as f64,
            wasm_hostdist_kernel_stats.decoded_cache_hits as f64 / args.iters as f64,
            (wasm_hostdist_kernel_stats.decode_time_ns as f64 / args.iters as f64) / 1_000_000.0,
            (wasm_hostdist_kernel_stats.compute_time_ns as f64 / args.iters as f64) / 1_000_000.0,
            (wasm_hostdist_stats.host_copy_time_ns as f64 / args.iters as f64) / 1_000_000.0,
            args.posting_codec
        );
    }

    Ok(())
}
