use anyhow::{anyhow, Result};
use arrow_array::{FixedSizeListArray, Float32Array, RecordBatch};
use arrow_array::types::Float32Type;
use arrow_schema::{DataType, Field, Schema};
use fff_poc::options::FileWriterOptions;
use fff_poc::writer::FileWriter;
use fff_vindex::artifact::ivf_flat::{build_ivf_flat_artifact, IvfFlatArtifact, PostingCodec};
use fff_vindex::artifact::ivf_pq::{build_ivf_pq_artifact, IvfPqArtifact, IvfPqArtifactBuildOptions};
use fff_vindex::artifact::wasm_ivf_pq::WasmIvfPqKernel;
use fff_vindex::artifact::wasm_ivf_flat::WasmIvfFlatKernel;
use fff_vindex::ivf_flat::{IvfFlatBuildOptions, SearchResult};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BinaryHeap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tempfile::tempdir;

struct BenchMetrics {
    index_name: String,
    build_time_ms: u128,
    index_size_bytes: u64,
    recall_at_10: f32,
    p50_latency_us: f64,
    p99_latency_us: f64,
    avg_latency_us: f64,
}

struct NativeIvfFlatSearcher {
    artifact: IvfFlatArtifact,
    centroids: Vec<f32>,
    dim: usize,
    nlist: usize,
}

impl NativeIvfFlatSearcher {
    fn new(artifact: IvfFlatArtifact) -> Result<Self> {
        let centroids = artifact.read_centroids_f32()?;
        let dim = artifact.footer().dim as usize;
        let nlist = artifact.footer().nlist as usize;
        Ok(Self { artifact, centroids, dim, nlist })
    }

    fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Result<Vec<SearchResult>> {
        #[derive(PartialEq)]
        struct Pair(u32, f32);
        impl Eq for Pair {}
        impl PartialOrd for Pair {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.1.partial_cmp(&other.1)
            }
        }
        impl Ord for Pair {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
            }
        }

        // 1. Find nearest centroids
        let mut centroid_dists = Vec::with_capacity(self.nlist);
        for cid in 0..self.nlist {
            let c = &self.centroids[cid * self.dim..(cid + 1) * self.dim];
            let dist: f32 = query.iter().zip(c.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            centroid_dists.push((cid, dist));
        }
        centroid_dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        
        let mut heap = BinaryHeap::new();
        
        // 2. Scan postings
        for i in 0..nprobe.min(self.nlist) {
            let (cid, _) = centroid_dists[i];
            let (row_ids, vectors) = self.artifact.read_posting_list(cid as u32)?;
            for j in 0..row_ids.len() {
                let v = &vectors[j * self.dim..(j + 1) * self.dim];
                let dist: f32 = query.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                
                heap.push(Pair(row_ids[j], dist));
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
        
        let mut results = Vec::with_capacity(heap.len());
        while let Some(Pair(row_id, dist)) = heap.pop() {
            results.push(SearchResult { row_id, distance: dist });
        }
        results.reverse();
        Ok(results)
    }
}

// Add footer accessor to IvfFlatArtifact if generic access is hard?
// IvfFlatArtifact struct fields are private but it has `footer` field.
// Need to assume `footer()` method exists or field is public?
// `src/artifact/ivf_flat.rs` shows `pub struct IvfFlatArtifact { path, footer }` but fields private.
// I need getters. Or make fields public.
// I can view `IvfFlatArtifact` definition again. 
// It was:
// pub struct IvfFlatArtifact {
//     path: PathBuf,
//     footer: IvfFlatArtifactFooter,
// }
// I assume I can add a getter or use existing one.
// Wait, I am `bench_paper_eval.rs` which is OUTSIDE crate.
// I cannot access private fields.
// I will need to check if `footer()` getter exists.

fn generate_random_f3_file(path: &Path, n: usize, dim: usize, rng: &mut StdRng) -> Result<Vec<Vec<f32>>> {
    let f = File::create(path)?;
    let field = Field::new("vector", DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32
    ), false);
    let schema = Arc::new(Schema::new(vec![field]));
    
    let options = FileWriterOptions::default();
    let mut writer = FileWriter::try_new(schema.clone(), f, options)
        .map_err(|e| anyhow!("FileWriter new error: {:?}", e))?;
    
    let mut all_vectors = Vec::with_capacity(n);
    let mut flat_values = Vec::with_capacity(n * dim);
    
    for _ in 0..n {
        let mut vec = vec![0.0; dim];
        for i in 0..dim {
            vec[i] = rng.gen();
        }
        flat_values.extend_from_slice(&vec);
        all_vectors.push(vec);
    }
    
    let values_array = Float32Array::from(flat_values);
    let fsl = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(values_array),
        None,
    );
    
    let batch = RecordBatch::try_new(schema, vec![Arc::new(fsl)])?;
    writer.write_batch(&batch)
        .map_err(|e| anyhow!("FileWriter write_batch error: {:?}", e))?;
    writer.finish()
        .map_err(|e| anyhow!("FileWriter finish error: {:?}", e))?;
    
    Ok(all_vectors)
}

fn compute_ground_truth(vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<u32>> {
    queries.iter().map(|q| {
        let mut dists: Vec<(usize, f32)> = vectors.iter().enumerate().map(|(i, v)| {
            let d: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            (i, d)
        }).collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        dists.truncate(k);
        dists.iter().map(|(i, _)| *i as u32).collect()
    }).collect()
}

fn compute_recall(results: &[Vec<SearchResult>], ground_truth: &[Vec<u32>]) -> f32 {
    let mut total_recall = 0.0;
    for (res, gt) in results.iter().zip(ground_truth.iter()) {
        let res_set: std::collections::HashSet<u32> = res.iter().map(|r| r.row_id).collect();
        let gt_set: std::collections::HashSet<u32> = gt.iter().cloned().collect();
        let intersection = res_set.intersection(&gt_set).count();
        total_recall += intersection as f32 / gt.len() as f32;
    }
    total_recall / results.len() as f32
}

fn main() -> Result<()> {
    let dir = tempdir()?;
    let base_path = dir.path().join("base.f3");
    let dim = 128;
    let n_base = 20_000;
    let n_queries = 100;
    let k = 10;
    let nprobe = 20;

    let mut rng = StdRng::seed_from_u64(42);
    
    println!("Generating {} vectors...", n_base);
    let vectors = generate_random_f3_file(&base_path, n_base, dim, &mut rng)?;
    
    println!("Generating {} queries...", n_queries);
    let mut queries = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let mut q = vec![0.0; dim];
        for i in 0..dim {
            q[i] = rng.gen();
        }
        queries.push(q);
    }
    
    println!("Computing ground truth...");
    let ground_truth = compute_ground_truth(&vectors, &queries, k);
    
    // Build IvfFlat artifact (used by all tests)
    println!("\nBuilding IvfFlat artifact...");
    let flat_build_start = Instant::now();
    let flat_artifact_path = build_ivf_flat_artifact(
        &base_path, 0, dim, "flat_f16", 
        IvfFlatBuildOptions { nlist: 100, train_sample: 2000, seed: 42, max_kmeans_iters: 10 }
    )?;
    let flat_build_time = flat_build_start.elapsed().as_millis();
    let flat_size = std::fs::metadata(&flat_artifact_path)?.len();
    println!("Artifact built in {}ms, size: {} bytes", flat_build_time, flat_size);
    
    // --- IvfFlat Native (F16) ---
    {
        println!("\n--- IvfFlat Native (RowIdDeltaVarintV1F16) ---");
        
        let searcher = NativeIvfFlatSearcher::new(IvfFlatArtifact::open(&flat_artifact_path)?)?;
        let start_bench = Instant::now();
        let mut latencies = Vec::with_capacity(n_queries);
        let mut results = Vec::with_capacity(n_queries);
        for q in &queries {
            let t0 = Instant::now();
            let res = searcher.search(q, k, nprobe)?;
            latencies.push(t0.elapsed().as_micros() as f64);
            results.push(res);
        }
        let avg_lat = start_bench.elapsed().as_micros() as f64 / n_queries as f64;
        latencies.sort_by(|a, b| a.total_cmp(b));
        let p50 = latencies[n_queries / 2];
        let p99 = latencies[(n_queries as f64 * 0.99) as usize];
        let recall = compute_recall(&results, &ground_truth);
        
        println!("Recall: {:.4}, Latency: p50={:.1}us, p99={:.1}us, avg={:.1}us", recall, p50, p99, avg_lat);
    }
    
    let wasm_scalar_path = PathBuf::from("../target/ivf_kernel_basic.wasm");
    let wasm_simd_path = PathBuf::from("../target/ivf_kernel_basic_simd.wasm");
    
    if wasm_scalar_path.exists() {
        println!("\n--- IvfFlat Wasm Scalar (F16) ---");
        let artifact = Arc::new(IvfFlatArtifact::open(&flat_artifact_path)?);
        
        let mut kernel = WasmIvfFlatKernel::load(&wasm_scalar_path, artifact.clone())?;
        
        let start_bench = Instant::now();
        let mut latencies = Vec::with_capacity(n_queries);
        let mut results = Vec::with_capacity(n_queries);
         for q in &queries {
            // Warm up? No, measure cold/warm mix is fine for now.
            let t0 = Instant::now();
            let res = kernel.search(&artifact, q, k, nprobe)?;
            latencies.push(t0.elapsed().as_micros() as f64);
            results.push(res);
        }
        let avg_lat = start_bench.elapsed().as_micros() as f64 / n_queries as f64;
        latencies.sort_by(|a, b| a.total_cmp(b));
        let p50 = latencies[n_queries / 2];
        let p99 = latencies[(n_queries as f64 * 0.99) as usize];
        let recall = compute_recall(&results, &ground_truth);
         println!("Recall: {:.4}, Latency: p50={:.1}us, p99={:.1}us, avg={:.1}us", recall, p50, p99, avg_lat);
    }

    if wasm_simd_path.exists() {
        println!("\n--- IvfFlat Wasm SIMD (F16) ---");
        let artifact = Arc::new(IvfFlatArtifact::open(&flat_artifact_path)?);
        
        let mut kernel = WasmIvfFlatKernel::load(&wasm_simd_path, artifact.clone())?;
         let start_bench = Instant::now();
        let mut latencies = Vec::with_capacity(n_queries);
        let mut results = Vec::with_capacity(n_queries);
         for q in &queries {
            let t0 = Instant::now();
            let res = kernel.search(&artifact, q, k, nprobe)?;
            latencies.push(t0.elapsed().as_micros() as f64);
            results.push(res);
        }
        let avg_lat = start_bench.elapsed().as_micros() as f64 / n_queries as f64;
        latencies.sort_by(|a, b| a.total_cmp(b));
        let p50 = latencies[n_queries / 2];
        let p99 = latencies[(n_queries as f64 * 0.99) as usize];
        let recall = compute_recall(&results, &ground_truth);
         println!("Recall: {:.4}, Latency: p50={:.1}us, p99={:.1}us, avg={:.1}us", recall, p50, p99, avg_lat);
    }
    
    // --- IvfPq Wasm (Scalar/SIMD) ---
    {
        println!("\n--- IvfPq Wasm (m=16) ---");
        let start = Instant::now();
        let path = build_ivf_pq_artifact(
            &base_path, 0, dim, "pq_test",
            IvfFlatBuildOptions { nlist: 100, train_sample: 2000, seed: 42, max_kmeans_iters: 10 },
            IvfPqArtifactBuildOptions { 
                num_subspaces: 16, subspace_dim: 8, max_kmeans_iters: 10, seed: 42, train_sample: 2000 
            }
        )?;
        let build_time = start.elapsed().as_millis();
        let size = std::fs::metadata(&path)?.len();
        
        let artifact = Arc::new(IvfPqArtifact::open(&path)?);
        
        // Test with Scalar kernel
        if wasm_scalar_path.exists() {
            println!(">> Scalar Kernel:");
            let mut kernel = WasmIvfPqKernel::load(&wasm_scalar_path, artifact.clone())?;
             let start_bench = Instant::now();
            let mut latencies = Vec::with_capacity(n_queries);
            let mut results = Vec::with_capacity(n_queries);
            for q in &queries {
                let t0 = Instant::now();
                let res = kernel.search(&artifact, q, k, nprobe)?;
                latencies.push(t0.elapsed().as_micros() as f64);
                results.push(res);
            }
            let avg_lat = start_bench.elapsed().as_micros() as f64 / n_queries as f64;
            latencies.sort_by(|a, b| a.total_cmp(b));
            let p50 = latencies[n_queries / 2];
            let p99 = latencies[(n_queries as f64 * 0.99) as usize];
            let recall = compute_recall(&results, &ground_truth);
             println!("Build: {} ms, Size: {} bytes", build_time, size);
             println!("Recall: {:.4}, Latency: p50={:.1}us, p99={:.1}us, avg={:.1}us", recall, p50, p99, avg_lat);
        }
        
         // Test with SIMD Kernel
        if wasm_simd_path.exists() {
            println!(">> SIMD Kernel:");
            let mut kernel = WasmIvfPqKernel::load(&wasm_simd_path, artifact.clone())?;
             let start_bench = Instant::now();
            let mut latencies = Vec::with_capacity(n_queries);
            let mut results = Vec::with_capacity(n_queries);
            for q in &queries {
                let t0 = Instant::now();
                let res = kernel.search(&artifact, q, k, nprobe)?;
                latencies.push(t0.elapsed().as_micros() as f64);
                results.push(res);
            }
            let avg_lat = start_bench.elapsed().as_micros() as f64 / n_queries as f64;
            latencies.sort_by(|a, b| a.total_cmp(b));
            let p50 = latencies[n_queries / 2];
            let recall = compute_recall(&results, &ground_truth);
             println!("Recall: {:.4}, Latency: p50={:.1}us, avg={:.1}us", recall, p50, avg_lat);
        }
    }
    
    Ok(())
}
