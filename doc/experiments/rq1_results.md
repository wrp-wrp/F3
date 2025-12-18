# RQ1 Results: Wasm Cost Boundary (Batch + SIMD)

This note summarizes the outputs produced by:
- `exp_scripts/rq1_simd_ablation.sh`
- `scripts/compare_rq1_simd_ablation.py`

## Artifacts

Latest (after fixing steady-state host-copy for hot postings):
- SIMD build CSV: `results/rq1_simd_ablation_v3/simd/all_results.csv`
- no-SIMD build CSV: `results/rq1_simd_ablation_v3/nosimd/all_results.csv`
- Compare command:
  - `python3 scripts/compare_rq1_simd_ablation.py results/rq1_simd_ablation_v3`

Baseline (older, kept for reference):
- SIMD build CSV: `results/rq1_simd_ablation/simd/all_results.csv`
- no-SIMD build CSV: `results/rq1_simd_ablation/nosimd/all_results.csv`
- Compare command:
  - `python3 scripts/compare_rq1_simd_ablation.py results/rq1_simd_ablation`

Parameters (from scripts):
- dataset: synthetic 100k vectors, dim=128 (generated under the output dir)
- index: IVF-Flat, `nlist=1024`, `nprobe=20`, `k=10`
- codec: `raw` (f32 postings)
- warmup/iters: `WARMUP=3`, `ITERS=10` (defaults in `exp_scripts/rq1_batch_sweep.sh`)

## Key signal for RQ1

### 1) Steady-state host copy can be (almost) eliminated

Fix applied:
- `wasm-libs/ivf-kernel-basic/src/lib.rs`: avoid unconditional `fetch_chunk(posting_chunk_id)` before checking the kernel-side decoded cache.
- Without this, even “warm” runs still copy posting bytes host→Wasm; and cache misses effectively fetched/copies twice (once unconditionally, once for decode+insert).

Using the SIMD build, warm-cache mode (`wasm_warm`, codec `raw`, v3 results):

| nq | native_ms | wasm_warm_ms | ratio (wasm/native) | host_copy_ms |
|---:|---:|---:|---:|---:|
| 1 | 0.031 | 0.315 | 10.13 | 0.007 |
| 2 | 0.061 | 0.451 | 7.36 | 0.008 |
| 4 | 0.127 | 0.475 | 3.72 | 0.009 |
| 8 | 0.308 | 0.605 | 1.96 | 0.008 |
| 16 | 0.633 | 0.999 | 1.58 | 0.009 |
| 32 | 1.217 | 1.731 | 1.42 | 0.008 |
| 64 | 2.321 | 3.474 | 1.50 | 0.010 |
| 100 | 3.948 | 6.589 | 1.67 | 0.012 |

Interpretation:
- warm 场景下 `host_copy_ms` 已经接近 0（约 0.01ms/batch 量级），说明“跨边界搬运”并不是不可消除的硬瓶颈：只要工作集能驻留在 Wasm（decoded cache 命中），就可以避免重复拷贝。
- `nq>=8` 后 `wasm/native` 比例进入 1.4–2.0× 区间；剩余开销主要来自 Wasm runtime/boundary + centroid 扫描 + heap 等，而不是 posting bytes 传输。

一个具体点（v2 → v3，对 SIMD `wasm_warm`, `nq=100`）：
- `host_copy_ms`: 3.660ms → 0.012ms
- `avg_latency_ms`: 11.210ms → 6.589ms

### 2) SIMD ablation (sanity check)

The SIMD vs no-SIMD comparison is in:
- `python3 scripts/compare_rq1_simd_ablation.py results/rq1_simd_ablation_v6_nq100_iters100_raw`
- `python3 scripts/compare_rq1_simd_ablation.py results/rq1_simd_ablation_v6_nq100_iters100_f16`

Important note (fixing the ablation):
- On `wasm32-wasip1`, `simd128` may be enabled by default, so a “no-SIMD” build must explicitly pass `-C target-feature=-simd128`.
- This is fixed in `exp_scripts/rq1_simd_ablation.sh` (older runs like v3/v4/v5 are not clean SIMD-vs-noSIMD evidence).

Results (with `WARMUP=5`, `ITERS=100`, `NQ_LIST=100`):
- `codec=raw`: `wasm_warm` SIMD `4.987ms` vs no-SIMD `19.259ms` (≈3.86× slower without SIMD); compute `3.410ms` vs `13.379ms` (≈3.92×).
- `codec=raw_f16`: `wasm_warm` SIMD `5.564ms` vs no-SIMD `19.753ms` (≈3.55× slower without SIMD); compute `3.850ms` vs `13.741ms` (≈3.57×).

## Next evidence step for RQ1

To make SIMD impact “pop” as a clean RQ1 figure:
- rerun the sweep with a compute-heavier or smaller-transfer regime (e.g. `raw_f16` or PQ), and compare SIMD vs no-SIMD again.
