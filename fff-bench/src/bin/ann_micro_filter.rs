//! Minimal experiment: simulate block-level WASM micro-index filtering to estimate
//! how much vector back-read IO can be reduced. This is an in-memory toy model,
//! but it exercises a tiny WASM module that filters block-local scores.

use anyhow::{Context, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;
use wasmtime::{AsContextMut, Engine, Instance, Module, Store, TypedFunc};

const FILTER_WAT: &str = r#"
(module
  (memory (export "memory") 2) ;; 2 pages = 128 KiB
  (func (export "filter")
    (param $scores i32)   ;; pointer to u16 scores
    (param $len i32)      ;; number of scores
    (param $query i32)    ;; query score (u16 in low bits)
    (param $tol i32)      ;; tolerance (u16 in low bits)
    (param $out i32)      ;; pointer to u32 output ids
    (result i32)          ;; how many ids were written
    (local $i i32)
    (local $w i32)
    (local $val i32)
    (local $diff i32)
    (block $exit
      (loop $loop
        (br_if $exit (i32.ge_u (local.get $i) (local.get $len)))
        ;; load u16 score
        (local.set $val
          (i32.load16_u
            (i32.add (local.get $scores)
              (i32.shl (local.get $i) (i32.const 1)))
          )
        )
        (local.set $diff (i32.sub (local.get $val) (local.get $query)))
        ;; abs(diff)
        (if (i32.lt_s (local.get $diff) (i32.const 0))
          (then (local.set $diff (i32.sub (i32.const 0) (local.get $diff))))
        )
        ;; if within tolerance, write id
        (if (i32.le_u (local.get $diff) (local.get $tol))
          (then
            (i32.store
              (i32.add (local.get $out) (i32.shl (local.get $w) (i32.const 2)))
              (local.get $i)
            )
            (local.set $w (i32.add (local.get $w) (i32.const 1)))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )
    (return (local.get $w))
  )
)
"#;

struct BenchConfig {
    dim: usize,
    block_size: usize,
    blocks: usize,
    top_blocks: usize,
    queries: usize,
    tolerance: u16,
}

struct Block {
    center: u16,
    scores: Vec<u16>,
}

fn build_blocks(cfg: &BenchConfig, rng: &mut StdRng) -> Vec<Block> {
    (0..cfg.blocks)
        .map(|b| {
            let center = (b as u32 * 500 + 5_000) as u16;
            let mut scores = Vec::with_capacity(cfg.block_size);
            for _ in 0..cfg.block_size {
                // Cluster tightly around center to make filtering meaningful.
                let noise: i16 = rng.gen_range(-60..=60);
                let raw = center as i32 + noise as i32;
                let clamped = raw.clamp(0, u16::MAX as i32) as u16;
                scores.push(clamped);
            }
            Block { center, scores }
        })
        .collect()
}

fn build_wasm(engine: &Engine) -> Result<Module> {
    let wasm_bytes = wat::parse_str(FILTER_WAT)?;
    Module::new(engine, wasm_bytes).context("compile wasm micro-index")
}

fn instantiate(engine: &Engine, module: &Module) -> Result<(Store<()>, Instance)> {
    let mut store = Store::new(engine, ());
    let instance = Instance::new(&mut store, module, &[])?;
    Ok((store, instance))
}

fn run_filter(
    store: &mut Store<()>,
    instance: &Instance,
    scores: &[u16],
    query: u16,
    tolerance: u16,
) -> Result<usize> {
    let mut ctx = store.as_context_mut();
    let memory = instance
        .get_memory(&mut ctx, "memory")
        .context("memory export not found")?;
    let filter: TypedFunc<(i32, i32, i32, i32, i32), i32> =
        instance.get_typed_func(&mut ctx, "filter")?;

    const SCORES_PTR: usize = 0;
    const OUT_PTR: usize = 0x2000; // leave space for scores and alignment

    let bytes = bytemuck::cast_slice(scores);
    memory.write(&mut ctx, SCORES_PTR, bytes)?;

    let written = filter.call(
        &mut ctx,
        (
            SCORES_PTR as i32,
            scores.len() as i32,
            query as i32,
            tolerance as i32,
            OUT_PTR as i32,
        ),
    )? as usize;
    Ok(written)
}

fn pick_blocks(query_score: u16, blocks: &[Block], top_k: usize) -> Vec<usize> {
    let mut candidates: Vec<(u16, usize)> = blocks
        .iter()
        .enumerate()
        .map(|(idx, blk)| {
            let diff = query_score.abs_diff(blk.center);
            (diff, idx)
        })
        .collect();
    candidates.sort_by_key(|(diff, _)| *diff);
    candidates
        .into_iter()
        .take(top_k.min(blocks.len()))
        .map(|(_, idx)| idx)
        .collect()
}

fn main() -> Result<()> {
    let cfg = BenchConfig {
        dim: 128,
        block_size: 1024,
        blocks: 64,
        top_blocks: 6,
        queries: 200,
        tolerance: 24, // target ~1/4–1/3 vectors kept per block
    };
    let vector_bytes = cfg.dim * std::mem::size_of::<f32>();
    let mut rng = StdRng::seed_from_u64(42);

    let blocks = build_blocks(&cfg, &mut rng);
    let engine = Engine::default();
    let module = build_wasm(&engine)?;

    // Baseline: no block-level filtering, just read selected blocks.
    let baseline_start = Instant::now();
    let mut base_bytes = 0usize;
    let mut base_reads = 0usize;
    for _ in 0..cfg.queries {
        let target_block = rng.gen_range(0..cfg.blocks);
        let query_score = blocks[target_block].center + rng.gen_range(0..=10);
        let selected = pick_blocks(query_score, &blocks, cfg.top_blocks);
        base_reads += selected.len();
        base_bytes += selected.len() * cfg.block_size * vector_bytes;
    }
    let baseline_elapsed = baseline_start.elapsed();

    // With micro-index filtering.
    let mut filtered_bytes = 0usize;
    let mut filtered_reads = 0usize;
    let mut kept_vectors = 0usize;
    let mut store_instance = instantiate(&engine, &module)?;
    let filtered_start = Instant::now();
    for _ in 0..cfg.queries {
        let target_block = rng.gen_range(0..cfg.blocks);
        let query_score = blocks[target_block].center + rng.gen_range(0..=10);
        let selected = pick_blocks(query_score, &blocks, cfg.top_blocks);
        filtered_reads += selected.len();
        for idx in selected {
            let kept = run_filter(
                &mut store_instance.0,
                &store_instance.1,
                &blocks[idx].scores,
                query_score,
                cfg.tolerance,
            )?;
            kept_vectors += kept;
            filtered_bytes += kept * vector_bytes;
        }
    }
    let filtered_elapsed = filtered_start.elapsed();

    let avg_keep = kept_vectors as f64 / (cfg.queries * cfg.top_blocks) as f64;
    println!("=== WASM block micro-index toy run ===");
    println!(
        "config: blocks={}, block_size={}, dim={}, queries={}, top_blocks={}, tolerance={}",
        cfg.blocks, cfg.block_size, cfg.dim, cfg.queries, cfg.top_blocks, cfg.tolerance
    );
    println!(
        "baseline (no filter): bytes={} (~{:.2} MB), block_reads={}, elapsed={:.2?}",
        base_bytes,
        base_bytes as f64 / (1024.0 * 1024.0),
        base_reads,
        baseline_elapsed
    );
    println!(
        "with micro-index:    bytes={} (~{:.2} MB), block_reads={}, elapsed={:.2?}",
        filtered_bytes,
        filtered_bytes as f64 / (1024.0 * 1024.0),
        filtered_reads,
        filtered_elapsed
    );
    if filtered_bytes > 0 {
        println!(
            "byte reduction: {:.2}x; avg kept per block: {:.1} / {} vectors",
            base_bytes as f64 / filtered_bytes as f64,
            avg_keep,
            cfg.block_size
        );
    }

    Ok(())
}
