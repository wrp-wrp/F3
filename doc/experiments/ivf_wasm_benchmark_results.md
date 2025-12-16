# WASM Vector Index Benchmark Results

**Date:** 2025-12-16
**Status:** Completed
**Focus:** Performance analysis of WASM-based IVF-Flat index vs Native implementation, with a focus on data movement overhead and SIMD acceleration.

## 1. Experimental Setup

*   **Dataset:** Synthetic, 100,000 vectors, 128 dimensions, Float32.
*   **Index:** IVF-Flat, `nlist=1024`, `nprobe=20`.
*   **Query Batch:** `nq=100`, `k=10`.
*   **Platform:** Darwin (macOS), Apple Silicon (ARM64).
*   **WASM Runtime:** Wasmtime (via `fff-ude-wasm`).
*   **Compilation:** `wasm32-wasip1` with `+simd128` enabled.

## 2. Benchmark Results Summary

Results are aggregated from `results/ivf_wasm_bench/all_bench_results.csv` (Warm Cache scenario).

| Metric | Native | WASM (Raw F32) | WASM (Raw F16) |
| :--- | :--- | :--- | :--- |
| **Total Latency (ms)** | **3.32** | 9.45 | **4.73** |
| **Ratio to Native** | 1.0x | 2.85x | **1.42x** |
| **Data Movement (Copy) (ms)** | 0 | 3.26 | ~0 |
| **Compute Time (ms)** | ~3.32 | 3.68 | 3.23 |
| **Overhead/Boundary (ms)** | 0 | 2.51 | 1.50 |

## 3. Key Findings

### 3.1. SIMD is Critical
Enabling SIMD (`+simd128`) for the WASM build resulted in a **3x speedup** for F32 compute, bringing WASM compute time (3.68ms) very close to Native (3.32ms). Without SIMD, WASM compute is ~13ms+, making it uncompetitive.

### 3.2. The F16 Sweet Spot
**F16 (Half-Precision) is the optimal configuration for WASM vector indexing.**
*   **Performance:** WASM F16 achieves **78% of Native F16 performance** (4.73ms vs 3.71ms).
*   **Why?** F16 data size is 50% of F32. This allows the WASM kernel to cache the *entire* working set in its linear memory (within the default budget).
*   **Zero Copy:** Once cached, `host_copy_ms` drops to **0**. The execution becomes purely CPU-bound inside the WASM sandbox, eliminating the primary bottleneck of data movement.

### 3.3. Data Movement Bottleneck (F32)
For F32 data which exceeds the cache or requires streaming:
*   **Overhead:** Data movement (Host -> WASM Copy) accounts for **~34.5%** (3.26ms) of the total latency.
*   **Boundary Cost:** The remaining overhead (~2.5ms) comes from function call overhead and WASM memory bounds checking.
*   **Conclusion:** For bandwidth-heavy tasks (like F32 scanning), the cost of crossing the Host-WASM boundary is significant.

### 3.4. Host Offloading (HostDist)
*   **F32:** Offloading distance calculation to the Host (`WASM Warm + Host Dist`) reduces latency slightly (9.45ms -> 7.65ms), primarily by leveraging Host's superior AVX/NEON implementation and bypassing some WASM memory management.
*   **F16:** Offloading offers **no benefit** (sometimes slower due to call overhead) because WASM SIMD is already efficient enough, and the data is already resident in WASM memory.

## 4. Recommendations

1.  **Adopt F16/Quantization:** Use `raw_f16` or quantized encodings for WASM indexes. This maximizes cache efficiency and minimizes the expensive Host-to-WASM data copy.
2.  **Keep Compute in WASM:** With F16 and SIMD enabled, WASM is fast enough. Complex logic (filtering, reranking) should stay in WASM to maintain architectural decoupling.
3.  **Optimize Chunk Size:** Ensure data chunks are sized to balance between granular fetching and call overhead. 
4.  **SIMD is Mandatory:** Always compile WASM kernels with `target-feature=+simd128`.

## 5. Artifacts
*   Script: `exp_scripts/wasm_ivf_bench.sh`
*   Code: `fff-bench/examples/bench_ivf_wasm.rs`
*   Kernel: `wasm-libs/ivf-kernel-basic`
