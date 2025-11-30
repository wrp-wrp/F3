//! Disk-based toy: vectors are block-aligned on disk; each block has mini-IVF centroids
//! and vectors ordered by centroid. A tiny WASM micro-index picks the best centroid per
//! block so we only read that centroid's slice, versus a baseline that reads full blocks
//! (approximate "Lance-style" full fragment read). This estimates how much back-read IO
//! can drop when block-level micro-filtering is used.

use anyhow::{Context, Result};
use clap::Parser;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::Instant;
use wasmtime::{AsContextMut, Engine, Instance, Module, Store, TypedFunc};

const MICRO_DIM: usize = 16;
const WASM_WAT: &str = r#"
(module
  (memory (export "memory") 8) ;; 8 pages = 512 KiB
  (func (export "select")
    (param $centroids i32) ;; f32[k*dim]
    (param $k i32)
    (param $dim i32)
    (param $query i32)     ;; f32[dim]
    (result i32)           ;; best centroid id
    (local $i i32) (local $j i32)
    (local $best i32) (local $bestd f32) (local $d f32) (local $tmp f32)
    (local.set $bestd (f32.const 3.4028235e38)) ;; max f32
    (block $exit
      (loop $outer
        (br_if $exit (i32.ge_u (local.get $i) (local.get $k)))
        (local.set $d (f32.const 0))
        (local.set $j (i32.const 0))
        (block $exit_inner
          (loop $inner
            (br_if $exit_inner (i32.ge_u (local.get $j) (local.get $dim)))
            ;; load query
            (local.set $tmp
              (f32.sub
                (f32.load (i32.add (local.get $query)
                                   (i32.shl (local.get $j) (i32.const 2))))
                ;; load centroid
                (f32.load
                  (i32.add
                    (local.get $centroids)
                    (i32.shl
                      (i32.add
                        (i32.mul (local.get $i) (local.get $dim))
                        (local.get $j))
                      (i32.const 2))))))
            (local.set $d (f32.add (local.get $d) (f32.mul (local.get $tmp) (local.get $tmp))))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $inner)
          )
        )
        (if (f32.lt (local.get $d) (local.get $bestd))
          (then
            (local.set $bestd (local.get $d))
            (local.set $best (local.get $i))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $outer)
      )
    )
    (return (local.get $best))
  )
)
"#;

#[derive(Clone)]
struct BlockMeta {
    center_scalar: f32,
    block_offset: u64,
    centroids_offset: u64,
    data_offset: u64,
    block_size: usize,
    k: usize,
    cluster_counts: Vec<usize>,
}

struct BenchConfig {
    dim: usize,
    block_size: usize,
    blocks: usize,
    k_micro: usize,
    top_blocks: usize,
    queries: usize,
    align: u64,
    tolerance_sigma: f32,
    tmp_path: PathBuf,
}

#[derive(Parser, Debug)]
struct Args {
    /// Reuse existing dataset/index file if present (skip rebuild).
    #[arg(long)]
    reuse: bool,
}

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) / a * a
}

fn build_dataset(
    cfg: &BenchConfig,
    rng: &mut StdRng,
    write_file: bool,
) -> Result<(Vec<BlockMeta>, Vec<Vec<f32>>)> {
    let mut metas = Vec::with_capacity(cfg.blocks);
    let mut query_vectors = Vec::with_capacity(cfg.queries);
    let mut offset: u64 = 0;
    let mut file = if write_file {
        Some(
            OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .read(true)
                .open(&cfg.tmp_path)?,
        )
    } else {
        None
    };

    for b in 0..cfg.blocks {
        offset = align_up(offset, cfg.align);
        let block_start = offset;
        let k = cfg.k_micro;
        // centroids (k * dim)
        let mut centroids: Vec<f32> = Vec::with_capacity(k * cfg.dim);
        let center_scalar = (b as f32 * 50.0) + 5000.0;
        for _ in 0..k {
            for _ in 0..cfg.dim {
                let base = rng.gen_range(-1.0..1.0) * 5.0;
                centroids.push(center_scalar + base);
            }
        }
        let centroids_bytes = bytemuck::cast_slice(&centroids);
        let centroids_offset = offset;
        if let Some(f) = file.as_mut() {
            f.seek(SeekFrom::Start(offset))?;
            f.write_all(centroids_bytes)?;
        }
        offset += centroids_bytes.len() as u64;

        // assign vectors to clusters, order by cluster
        let mut cluster_counts = vec![0usize; k];
        let mut vectors_per_cluster: Vec<Vec<f32>> = vec![Vec::new(); k];
        for _ in 0..cfg.block_size {
            let cid = rng.gen_range(0..k);
            cluster_counts[cid] += 1;
            let base = &centroids[cid * cfg.dim..(cid + 1) * cfg.dim];
            let mut vec = Vec::with_capacity(cfg.dim);
            for &c in base {
                let noise: f32 = rng.sample::<f32, _>(rand_distr::Normal::new(0.0, cfg.tolerance_sigma).unwrap());
                vec.push(c + noise);
            }
            vectors_per_cluster[cid].extend_from_slice(&vec);
        }

        // write vectors cluster by cluster
        let data_offset = offset;
        for cid in 0..k {
            let bytes = bytemuck::cast_slice(&vectors_per_cluster[cid]);
            if let Some(f) = file.as_mut() {
                f.write_all(bytes)?;
            }
            offset += bytes.len() as u64;
        }

        metas.push(BlockMeta {
            center_scalar,
            block_offset: block_start,
            centroids_offset,
            data_offset,
            block_size: cfg.block_size,
            k,
            cluster_counts: cluster_counts.clone(),
        });
    }
    if let Some(f) = file.as_mut() {
        f.flush()?;
        f.sync_all()?;
    }

    // build query vectors around random blocks/centroids
    for _ in 0..cfg.queries {
        let b = rng.gen_range(0..cfg.blocks);
        let center = metas[b].center_scalar;
        let mut q = Vec::with_capacity(cfg.dim);
        for _ in 0..cfg.dim {
            let noise: f32 = rng.sample::<f32, _>(rand_distr::Normal::new(0.0, cfg.tolerance_sigma).unwrap());
            q.push(center + noise);
        }
        query_vectors.push(q);
    }

    Ok((metas, query_vectors))
}

fn pick_blocks(query: &[f32], metas: &[BlockMeta], top_k: usize) -> Vec<usize> {
    let q_scalar = query.get(0).copied().unwrap_or(0.0);
    let mut pairs: Vec<(f32, usize)> = metas
        .iter()
        .enumerate()
        .map(|(i, m)| ((m.center_scalar - q_scalar).abs(), i))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pairs
        .into_iter()
        .take(top_k.min(metas.len()))
        .map(|(_, i)| i)
        .collect()
}

fn build_wasm(engine: &Engine) -> Result<Module> {
    let wasm_bytes = wat::parse_str(WASM_WAT)?;
    Module::new(engine, wasm_bytes).context("compile wasm micro index")
}

fn instantiate(engine: &Engine, module: &Module) -> Result<(Store<()>, Instance)> {
    let mut store = Store::new(engine, ());
    let instance = Instance::new(&mut store, module, &[])?;
    Ok((store, instance))
}

fn select_centroid(
    store: &mut Store<()>,
    instance: &Instance,
    centroids: &[f32],
    k: usize,
    dim: usize,
    query: &[f32],
) -> Result<usize> {
    let mut ctx = store.as_context_mut();
    let memory = instance
        .get_memory(&mut ctx, "memory")
        .context("memory export not found")?;
    let func: TypedFunc<(i32, i32, i32, i32), i32> =
        instance.get_typed_func(&mut ctx, "select")?;

    // layout: centroids at 0, query after that
    let centroid_ptr = 0usize;
    let centroid_bytes = bytemuck::cast_slice(centroids);
    memory.write(&mut ctx, centroid_ptr, centroid_bytes)?;
    let query_ptr = align_up(centroid_bytes.len() as u64, 64) as usize;
    let query_bytes = bytemuck::cast_slice(query);
    memory.write(&mut ctx, query_ptr, query_bytes)?;

    let best = func.call(
        &mut ctx,
        (
            centroid_ptr as i32,
            k as i32,
            dim as i32,
            query_ptr as i32,
        ),
    )? as usize;
    Ok(best.min(k - 1))
}

fn read_range(file: &mut File, offset: u64, len_bytes: usize, buf: &mut Vec<u8>) -> Result<()> {
    buf.resize(len_bytes, 0u8);
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = BenchConfig {
        dim: 128,
        block_size: 1024,
        blocks: 32,
        k_micro: 16,
        top_blocks: 4,
        queries: 100,
        align: 4096,
        tolerance_sigma: 6.0,
        tmp_path: std::env::temp_dir().join("f3_ann_micro_disk.bin"),
    };
    println!("temp file: {}", cfg.tmp_path.display());
    let mut rng = StdRng::seed_from_u64(1234);

    let need_rebuild = !(cfg.tmp_path.exists() && args.reuse);
    let (metas, queries) = build_dataset(&cfg, &mut rng, need_rebuild)?;
    let mut file = OpenOptions::new().read(true).open(&cfg.tmp_path)?;
    let engine = Engine::default();
    let module = build_wasm(&engine)?;
    let (mut store, instance) = instantiate(&engine, &module)?;

    // Preload centroids into RAM for quick micro-index eval (small footprint).
    let mut centroid_cache: Vec<Vec<f32>> = Vec::with_capacity(metas.len());
    for m in &metas {
        let bytes = (m.k * cfg.dim * std::mem::size_of::<f32>()) as usize;
        let mut buf = vec![0u8; bytes];
        read_range(&mut file, m.centroids_offset, bytes, &mut buf)?;
        let vec: Vec<f32> = bytemuck::cast_slice(&buf).to_vec();
        centroid_cache.push(vec);
    }

    // Baseline: read full blocks (approx Lance fragment read).
    let mut baseline_bytes = 0usize;
    let mut baseline_reads = 0usize;
    let mut scratch = Vec::new();
    let t0 = Instant::now();
    for q in &queries {
        let blocks = pick_blocks(q, &metas, cfg.top_blocks);
        for &bid in &blocks {
            let m = &metas[bid];
            let len = m.block_size * cfg.dim * std::mem::size_of::<f32>();
            read_range(&mut file, m.data_offset, len, &mut scratch)?;
            baseline_bytes += len;
            baseline_reads += 1;
        }
    }
    let baseline_elapsed = t0.elapsed();

    // Micro-index path: select centroid, read only that cluster slice.
    let mut micro_bytes = 0usize;
    let mut micro_reads = 0usize;
    let mut micro_vectors = 0usize;
    let t1 = Instant::now();
    for q in &queries {
        // project query to MICRO_DIM for the micro index
        let mut q_proj = vec![0f32; MICRO_DIM];
        q_proj.copy_from_slice(&q[..MICRO_DIM]);
        let blocks = pick_blocks(q, &metas, cfg.top_blocks);
        for &bid in &blocks {
            let m = &metas[bid];
            let centroids = &centroid_cache[bid];
            let best = select_centroid(&mut store, &instance, centroids, m.k, MICRO_DIM, &q_proj)?;
            let cluster_offset_elems: usize =
                m.cluster_counts.iter().take(best).sum::<usize>() * cfg.dim;
            let cluster_len = m.cluster_counts[best] * cfg.dim;
            let offset_bytes =
                m.data_offset + (cluster_offset_elems * std::mem::size_of::<f32>()) as u64;
            let len_bytes = cluster_len * std::mem::size_of::<f32>();
            if len_bytes > 0 {
                read_range(&mut file, offset_bytes, len_bytes, &mut scratch)?;
                micro_bytes += len_bytes;
                micro_vectors += cluster_len / cfg.dim;
                micro_reads += 1;
            }
        }
    }
    let micro_elapsed = t1.elapsed();

    println!("=== Disk micro-index toy (mini-IVF) ===");
    println!(
        "config: blocks={}, block_size={}, dim={}, k_micro={}, top_blocks={}, queries={}, align={}, sigma={}",
        cfg.blocks, cfg.block_size, cfg.dim, cfg.k_micro, cfg.top_blocks, cfg.queries, cfg.align, cfg.tolerance_sigma
    );
    println!(
        "baseline (full block ~ Lance-style): bytes={} (~{:.2} MB), reads={}, elapsed={:.2?}",
        baseline_bytes,
        baseline_bytes as f64 / (1024.0 * 1024.0),
        baseline_reads,
        baseline_elapsed
    );
    println!(
        "micro-index (read chosen centroid only): bytes={} (~{:.2} MB), reads={}, vectors={}, elapsed={:.2?}",
        micro_bytes,
        micro_bytes as f64 / (1024.0 * 1024.0),
        micro_reads,
        micro_vectors,
        micro_elapsed
    );
    if micro_bytes > 0 {
        println!(
            "byte reduction: {:.2}x; avg vectors per read: {:.1} / {}",
            baseline_bytes as f64 / micro_bytes as f64,
            micro_vectors as f64 / micro_reads.max(1) as f64,
            cfg.block_size
        );
    }

    Ok(())
}
