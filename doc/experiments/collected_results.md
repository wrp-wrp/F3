# Collected Results Snapshot

This file is a *snapshot* of result artifacts currently present in the workspace under `results/`, plus a normalized view of the IVF+Wasm benchmark CSV.

For the consolidated TODO / next-step plan for vector indexing, see `doc/experiments/vector_index_todo.md`.

If you have additional local runs (e.g. SIFT `*.jsonl` output directories from `scripts/run_sift_ivf_experiments.sh`), drop them under `results/` (or rerun the scripts) and we can re-collect.

Generated on: 2025-12-17

## 1) Inventory (`results/`)

| path | size_kb |
|---|---:|
| results/combined_nimble_projection_times.csv | 2.0 |
| results/combined_orc_projection_times.csv | 2.0 |
| results/combined_orc_read_times.csv | 1.3 |
| results/combined_orc_read_times_original_cpp.csv | 1.1 |
| results/cr.csv | 0.6 |
| results/decomp.csv | 0.8 |
| results/f3_proj_100cols_unchecked.svg | 505.3 |
| results/f3_proj_100kcols.svg | 328.5 |
| results/f3_proj_1kcols.svg | 931.7 |
| results/f3_proj_1kcols_unchecked.svg | 559.7 |
| results/iounit_checksum-read.csv | 0.2 |
| results/iounit_checksum-write.csv | 0.2 |
| results/ivf_wasm_bench/all_bench_results.csv | 1.2 |
| results/ivf_wasm_bench/vectors.f3.vindex/manifest.json | 2.6 |
| results/lance_benchmark_results/base_comparison.csv | 0.2 |
| results/lance_benchmark_results/data_cache_bytes_comparison.csv | 0.2 |
| results/lance_benchmark_results/max_page_bytes_comparison.csv | 0.2 |
| results/lance_benchmark_results/version_comparison.csv | 0.1 |
| results/nimbleUncomp_read_times.csv | 0.3 |
| results/nimble_col_size_laion.csv | 0.3 |
| results/nimble_col_size_laion.txt | 145.7 |
| results/nimble_projection_times_no_loadSchema.csv | 0.3 |
| results/nimble_read_times.csv | 0.3 |
| results/nimble_read_times_0.csv | 0.3 |
| results/parquet_analysis.json | 111.4 |
| results/rg_size_laion/chunk_size_rg1048576.json | 0.3 |
| results/rg_size_laion/chunk_size_rg131072.json | 2.2 |
| results/rg_size_laion/chunk_size_rg262144.json | 1.1 |
| results/rg_size_laion/chunk_size_rg524288.json | 0.6 |
| results/rg_size_laion/chunk_size_rg65536.json | 4.0 |
| results/wasm_micro.csv | 6.7 |
| results/wasm_micro_debuginfo_optnone.csv | 0.8 |
| results/wasm_size.csv | 9.5 |

## 1.1) New RQ1 artifacts

- `results/rq1_simd_ablation/simd/all_results.csv`
- `results/rq1_simd_ablation/nosimd/all_results.csv`
- `results/rq1_simd_ablation_v2/simd/all_results.csv`
- `results/rq1_simd_ablation_v2/nosimd/all_results.csv`
- `results/rq1_simd_ablation_v3/simd/all_results.csv`
- `results/rq1_simd_ablation_v3/nosimd/all_results.csv`
- `results/rq1_simd_ablation_v4_f16/simd/all_results.csv`
- `results/rq1_simd_ablation_v4_f16/nosimd/all_results.csv`
- `results/rq1_simd_ablation_v6_nq100_iters100_raw/simd/all_results.csv`
- `results/rq1_simd_ablation_v6_nq100_iters100_raw/nosimd/all_results.csv`
- `results/rq1_simd_ablation_v6_nq100_iters100_f16/simd/all_results.csv`
- `results/rq1_simd_ablation_v6_nq100_iters100_f16/nosimd/all_results.csv`

## 2) Vector Index: IVF+Wasm benchmark (`results/ivf_wasm_bench/all_bench_results.csv`)

Source: `exp_scripts/wasm_ivf_bench.sh` produces the CSV and data under `results/ivf_wasm_bench/`.

Notes:
- The CSV schema is:
  `mode,nq,nprobe,avg_latency_ms,chunks_fetched,decoded_cache_hits,decode_time_ms,compute_time_ms,host_copy_time_ms,codec`
  (older runs may have a stale header; rerun `exp_scripts/wasm_ivf_bench.sh` to regenerate.)

Normalized table (per codec × mode):

| codec | mode | avg_ms | compute_ms | host_copy_ms | chunks_fetched | decoded_cache_hits |
|---|---|---:|---:|---:|---:|---:|
| raw | native | 3.384 | 0.000 | 0.000 | 0 | 0 |
| raw | wasm_warm | 9.452 | 3.676 | 3.256 | 0 | 0 |
| raw | wasm_warm_hostdist | 7.655 | 2.849 | 2.828 | 0 | 0 |
| raw | wasm_cold | 18.643 | 0.000 | 2.097 | 2001 | 0 |
| raw_f16 | native | 3.713 | 0.000 | 0.000 | 0 | 0 |
| raw_f16 | wasm_warm | 4.732 | 3.227 | 0.008 | 0 | 2000 |
| raw_f16 | wasm_warm_hostdist | 4.868 | 3.334 | 0.009 | 0 | 2000 |
| raw_f16 | wasm_cold | 4.771 | 0.000 | 0.008 | 1 | 0 |
| row_id_delta_varint_v1 | native | 3.393 | 0.000 | 0.000 | 0 | 0 |
| row_id_delta_varint_v1 | wasm_warm | 8.561 | 3.205 | 2.843 | 0 | 0 |
| row_id_delta_varint_v1 | wasm_warm_hostdist | 8.220 | 3.169 | 2.645 | 0 | 0 |
| row_id_delta_varint_v1 | wasm_cold | 20.160 | 0.000 | 2.127 | 2001 | 0 |
| row_id_delta_varint_v1_f16 | native | 3.457 | 0.000 | 0.000 | 0 | 0 |
| row_id_delta_varint_v1_f16 | wasm_warm | 4.786 | 3.255 | 0.008 | 0 | 2000 |
| row_id_delta_varint_v1_f16 | wasm_warm_hostdist | 4.605 | 3.111 | 0.008 | 0 | 2000 |
| row_id_delta_varint_v1_f16 | wasm_cold | 4.774 | 0.000 | 0.008 | 1 | 0 |

Related write-ups:
- `doc/experiments/ivf_wasm_benchmark_results.md`
- `doc/experiments/ivf_artifact_current_status.md` (includes SIFT100K end-to-end tables, stage profiling, and microbench notes)

## 3) IO On-Demand Performance (New Proof)

Measured via `exp_scripts/io_on_demand_benchmark.sh` (2025-12-18).

| Metric | Traditional (Full Load) | F3 (On-Demand WASM) |
| :--- | :--- | :--- |
| **Data Fetched** | 25.41 MB (100%) | **0.98 MB (4%)** |
| **First Query Latency** | 22.92 ms | **1.26 ms (18x faster)** |
| **Steady State Latency** | 0.44 ms | 0.44 ms |

### 3.1) Storage Breakdown (1M Vectors)
- **IVF-Flat**: Structure < 1%, Data > 99%.
- **HNSW (M=32)**: Structure ~25%.
