//! Compare Lance IVF_PQ scanning vs. naive full-fragment reads and F3-style mini-IVF micro-filter.
//! Generates a synthetic dataset on disk once, builds a Lance IVF_PQ index, then runs knn queries.
//! Metrics: read bytes (rough estimate via returned batches), query latency. For IO stats you can
//! wrap the binary with `strace -e read,pread64 -c` or `perf stat -e io:*`.

use anyhow::Result;
use arrow_array_52::{FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator};
use arrow_schema_52::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::index::vector::VectorIndexParams;
use lance_index::IndexType;
use lance_index::DatasetIndexExt;
use clap::Parser;
use lance_linalg::distance::DistanceType;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug)]
struct BenchConfig {
    dim: usize,
    rows: usize,
    blocks: usize,
    k: usize,
    queries: usize,
    seed: u64,
    path: PathBuf,
}

#[derive(Parser, Debug)]
struct Args {
    /// Reuse existing Lance dataset/index if present (skip rebuild).
    #[arg(long)]
    reuse: bool,
}

fn make_data(cfg: &BenchConfig) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vec",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            cfg.dim as i32,
        ),
        false,
    )]));
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut values = Vec::with_capacity(cfg.rows * cfg.dim);
    for i in 0..cfg.rows {
        let base = (i / (cfg.rows / cfg.blocks).max(1)) as f32 * 50.0 + 5000.0;
        for _ in 0..cfg.dim {
            let noise: f32 = rng.gen_range(-5.0..5.0);
            values.push(base + noise);
        }
    }
    let flat = Float32Array::from(values);
    let list = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        cfg.dim as i32,
        Arc::new(flat) as _,
        None,
    )?;
    Ok(RecordBatch::try_new(schema, vec![Arc::new(list)])?)
}

async fn write_lance_dataset(cfg: &BenchConfig) -> Result<()> {
    if cfg.path.exists() {
        std::fs::remove_dir_all(&cfg.path)?;
    }
    let batch = make_data(cfg)?;
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    lance::dataset::Dataset::write(reader, &cfg.path.to_string_lossy(), None).await?;
    Ok(())
}

async fn build_ivfpq(cfg: &BenchConfig) -> Result<()> {
    let mut ds = DatasetBuilder::from_uri(cfg.path.to_string_lossy())
        .load()
        .await?;

    let params = VectorIndexParams::ivf_pq(
        (cfg.blocks * 2).max(8), // num_partitions
        8,                       // num_bits
        16,                      // num_sub_vectors (m)
        DistanceType::L2,
        15, // niter (kept small for quick demo)
    );
    ds.create_index(&["vec"], IndexType::Vector, None, &params, true)
        .await?;
    Ok(())
}

async fn run_queries(cfg: &BenchConfig) -> Result<()> {
    let ds = DatasetBuilder::from_uri(cfg.path.to_string_lossy())
        .load()
        .await?;
    let mut rng = StdRng::seed_from_u64(cfg.seed + 1);
    let mut queries = Vec::with_capacity(cfg.queries);
    for _ in 0..cfg.queries {
        let base_block = rng.gen_range(0..cfg.blocks) as f32;
        let base = base_block * 50.0 + 5000.0;
        let mut v = Vec::with_capacity(cfg.dim);
        for _ in 0..cfg.dim {
            let noise: f32 = rng.gen_range(-3.0..3.0);
            v.push(base + noise);
        }
        queries.push(v);
    }

    let t0 = Instant::now();
    let mut total_rows = 0usize;
    for q in &queries {
        let arr = Float32Array::from(q.clone());
        let mut scanner = ds.scan();
        scanner.nearest("vec", &arr, cfg.k)?;
        let results = scanner
            .try_into_stream()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        for b in results {
            total_rows += b.num_rows();
        }
    }
    let elapsed = t0.elapsed();
    let approx_bytes = total_rows * cfg.dim * std::mem::size_of::<f32>();
    println!("=== Lance IVF_PQ knn ===");
    println!(
        "config: rows={}, dim={}, blocks={}, k={}, queries={}",
        cfg.rows, cfg.dim, cfg.blocks, cfg.k, cfg.queries
    );
    println!(
        "result rows touched: {} over {} queries; approx bytes read: {} (~{:.2} MB); elapsed={:.2?}",
        total_rows,
        cfg.queries,
        approx_bytes,
        approx_bytes as f64 / (1024.0 * 1024.0),
        elapsed
    );
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = BenchConfig {
        dim: 128,
        rows: 32 * 1024,
        blocks: 32,
        k: 10,
        queries: 100,
        seed: 1234,
        path: std::env::temp_dir().join("f3_vs_lance_ivfpq.lance"),
    };
    println!("lance dataset path: {}", cfg.path.display());
    if !(cfg.path.exists() && args.reuse) {
        write_lance_dataset(&cfg).await?;
        build_ivfpq(&cfg).await?;
    }
    run_queries(&cfg).await?;
    Ok(())
}
