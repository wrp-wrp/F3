use anyhow::{Context, Result};
use clap::Parser;
use fff_vindex::ivf_flat::l2_sq;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;
use wasi_common::sync::WasiCtxBuilder;
use wasmtime::{Caller, Config, Engine, Linker, Module, Store};

#[derive(Parser, Debug)]
struct Args {
    /// Path to `ivf_kernel_basic.wasm` compiled for `wasm32-wasip1`.
    #[arg(long, default_value = "target/wasm32-wasip1/release/ivf_kernel_basic.wasm")]
    wasm: PathBuf,

    /// Vector dimension.
    #[arg(long, default_value_t = 128)]
    dim: usize,

    /// Number of candidate vectors.
    #[arg(long, default_value_t = 16_384)]
    count: usize,

    /// Outer loop iterations (total distance ops = `iters * count`).
    #[arg(long, default_value_t = 200)]
    iters: usize,

    /// Warmup calls for both native and wasm.
    #[arg(long, default_value_t = 2)]
    warmup: usize,

    /// Print JSON.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    json: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dim = args.dim;
    let count = args.count;
    let iters = args.iters;

    let mut query = vec![0.0f32; dim];
    let mut vectors = vec![0.0f32; count * dim];
    fill_f32(&mut query, 0x1234_5678_9abc_def0);
    fill_f32(&mut vectors, 0x0fed_cba9_8765_4321);

    let (wasm_ctx, wasm_query_ptr, wasm_vectors_ptr, wasm_out_ptr) =
        prepare_wasm(&args.wasm, &query, &vectors)?;

    let native = measure_native(&query, &vectors, count, dim, iters, args.warmup);
    let wasm = measure_wasm(
        wasm_ctx,
        wasm_query_ptr,
        wasm_vectors_ptr,
        wasm_out_ptr,
        count,
        dim,
        iters,
        args.warmup,
    )?;

    let native_ns_per_op = native.elapsed_ns as f64 / (iters as f64 * count as f64);
    let wasm_ns_per_op = wasm.elapsed_ns as f64 / (iters as f64 * count as f64);

    if args.json {
        println!(
            "{}",
            json!({
                "dim": dim,
                "count": count,
                "iters": iters,
                "ops": iters * count,
                "native": {
                    "elapsed_ns": native.elapsed_ns,
                    "ns_per_op": native_ns_per_op,
                    "checksum": native.checksum,
                },
                "wasm": {
                    "elapsed_ns": wasm.elapsed_ns,
                    "ns_per_op": wasm_ns_per_op,
                    "checksum": wasm.checksum,
                    "wasm_path": args.wasm.to_string_lossy(),
                },
                "ratio_wasm_over_native": wasm_ns_per_op / native_ns_per_op,
            })
        );
    } else {
        println!(
            "dim={} count={} iters={} native={:.2}ns/op wasm={:.2}ns/op ratio={:.3}",
            dim,
            count,
            iters,
            native_ns_per_op,
            wasm_ns_per_op,
            wasm_ns_per_op / native_ns_per_op
        );
    }
    Ok(())
}

fn fill_f32(out: &mut [f32], mut state: u64) {
    for v in out {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mantissa = ((state >> 32) as u32) & 0x007f_ffff;
        let bits = 0x3f80_0000 | mantissa; // [1.0, 2.0)
        let f = f32::from_bits(bits) - 1.0;
        *v = f;
    }
}

struct Timed {
    elapsed_ns: u64,
    checksum: f32,
}

fn measure_native(query: &[f32], vectors: &[f32], count: usize, dim: usize, iters: usize, warmup: usize) -> Timed {
    for _ in 0..warmup {
        let _ = native_loop(query, vectors, count, dim, iters / 10 + 1);
    }
    let start = Instant::now();
    let checksum = native_loop(query, vectors, count, dim, iters);
    let elapsed_ns = start.elapsed().as_nanos() as u64;
    Timed { elapsed_ns, checksum }
}

#[inline(never)]
fn native_loop(query: &[f32], vectors: &[f32], count: usize, dim: usize, iters: usize) -> f32 {
    let mut acc = 0.0f32;
    for _ in 0..iters {
        for i in 0..count {
            let base = i * dim;
            let v = &vectors[base..base + dim];
            acc += l2_sq(query, v);
        }
    }
    std::hint::black_box(acc)
}

struct WasmCtx {
    store: Store<WasmState>,
    memory: wasmtime::Memory,
    dealloc: wasmtime::TypedFunc<(u32, u32, u32), ()>,
    micro: wasmtime::TypedFunc<(u32, u32, u32, u32, u32, u32), u32>,
}

struct WasmState {
    wasi: wasi_common::WasiCtx,
}

fn prepare_wasm(wasm_path: &PathBuf, query: &[f32], vectors: &[f32]) -> Result<(WasmCtx, u32, u32, u32)> {
    let mut config = Config::new();
    config.wasm_simd(true);
    let engine = Engine::new(&config)?;
    let module = Module::from_file(&engine, wasm_path)
        .with_context(|| format!("load wasm module {}", wasm_path.display()))?;

    let mut linker = Linker::new(&engine);
    wasi_common::sync::add_to_linker(&mut linker, |s: &mut WasmState| &mut s.wasi)?;

    linker.func_wrap("env", "host_chunk_len", |_caller: Caller<'_, WasmState>, _chunk_id: u32| -> u32 {
        0
    })?;
    linker.func_wrap(
        "env",
        "host_read_chunk",
        |_caller: Caller<'_, WasmState>, _chunk_id: u32, _dst_ptr: u32, _dst_len: u32| -> u32 { 0 },
    )?;

    let wasi = WasiCtxBuilder::new().inherit_stdio().build();
    let mut store = Store::new(&engine, WasmState { wasi });
    let instance = linker.instantiate(&mut store, &module)?;

    let memory = instance
        .get_memory(&mut store, "memory")
        .context("wasm export `memory` not found")?;
    let alloc = instance
        .get_typed_func::<(u32, u32), u32>(&mut store, "alloc_ffi")
        .context("get export alloc_ffi")?;
    let dealloc = instance
        .get_typed_func::<(u32, u32, u32), ()>(&mut store, "dealloc_ffi")
        .context("get export dealloc_ffi")?;
    let micro = instance
        .get_typed_func::<(u32, u32, u32, u32, u32, u32), u32>(
            &mut store,
            "l2_microbench_query_vs_vectors_f32_ffi",
        )
        .context("get export l2_microbench_query_vs_vectors_f32_ffi")?;

    let query_bytes = bytemuck::cast_slice(query);
    let vectors_bytes = bytemuck::cast_slice(vectors);
    let query_ptr = alloc.call(&mut store, (query_bytes.len() as u32, 4))?;
    let vectors_ptr = alloc.call(&mut store, (vectors_bytes.len() as u32, 4))?;
    let out_ptr = alloc.call(&mut store, (4, 4))?;
    memory.write(&mut store, query_ptr as usize, query_bytes)?;
    memory.write(&mut store, vectors_ptr as usize, vectors_bytes)?;
    memory.write(&mut store, out_ptr as usize, bytemuck::bytes_of(&0.0f32))?;

    Ok((
        WasmCtx {
            store,
            memory,
            dealloc,
            micro,
        },
        query_ptr,
        vectors_ptr,
        out_ptr,
    ))
}

fn measure_wasm(
    mut ctx: WasmCtx,
    query_ptr: u32,
    vectors_ptr: u32,
    out_ptr: u32,
    count: usize,
    dim: usize,
    iters: usize,
    warmup: usize,
) -> Result<Timed> {
    for _ in 0..warmup {
        let _ = ctx.micro.call(
            &mut ctx.store,
            (
                query_ptr,
                vectors_ptr,
                count as u32,
                dim as u32,
                (iters / 10 + 1) as u32,
                out_ptr,
            ),
        )?;
    }

    let start = Instant::now();
    let ok = ctx.micro.call(
        &mut ctx.store,
        (
            query_ptr,
            vectors_ptr,
            count as u32,
            dim as u32,
            iters as u32,
            out_ptr,
        ),
    )?;
    let elapsed_ns = start.elapsed().as_nanos() as u64;
    anyhow::ensure!(ok != 0, "wasm microbench returned 0");

    let mut out = [0u8; 4];
    ctx.memory.read(&mut ctx.store, out_ptr as usize, &mut out)?;
    let checksum = f32::from_le_bytes(out);

    let _ = ctx.dealloc.call(&mut ctx.store, (query_ptr, (dim * 4) as u32, 4));
    let _ = ctx.dealloc.call(
        &mut ctx.store,
        (vectors_ptr, (count * dim * 4) as u32, 4),
    );
    let _ = ctx.dealloc.call(&mut ctx.store, (out_ptr, 4, 4));

    Ok(Timed { elapsed_ns, checksum })
}
