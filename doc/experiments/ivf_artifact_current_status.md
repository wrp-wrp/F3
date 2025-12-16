# IVF “Index Artifact + Wasm Kernel” 当前能跑的实验（已跑通）

这份文档记录：当前代码已经实现了什么、能跑哪些实验、以及一次已完成的 SIFT100K 基线实验结果（原始 JSONL 日志不进 git，路径保存在本地 `results/` 下）。

## 已实现的能力

- IVF-Flat 索引 artifact 容器（chunk + footer 目录 + base file binding）
  - 代码：`fff-vindex/src/artifact/ivf_flat.rs`
  - chunk：`centroids`、`list_offsets`、`posting_list(list_id)`
- posting list 的无损压缩（目前只压 `row_id`，向量仍是 `f32` 原样存）
  - codec：
    - `raw`：`[count][row_ids][vectors]`
    - `row_id_delta_varint_v1`：`[count][first][uleb128 deltas][vectors]`
- posting list 的向量半精度压缩（`f16`，会引入数值误差；目前只对 IVF-Flat 的 posting vectors 生效）
  - codec：
    - `raw_f16`：`[count][row_ids][vectors(f16)]`
    - `row_id_delta_varint_v1_f16`：`[count][first][uleb128 deltas][vectors(f16)]`
- Wasm IVF 搜索内核（解码+搜索在 Wasm 内部执行；I/O 由 host 回调提供）
  - Wasm：`wasm-libs/ivf-kernel-basic/src/lib.rs`
  - Host runtime（Wasmtime + WASI + chunk cache + fetch stats）：`fff-vindex/src/artifact/wasm_ivf_flat.rs`
- 端到端 SIFT100K 跑通脚本
  - 下载数据：`scripts/download_sift_100k.sh`
  - 生成 base `.f3`：`fff-bench/examples/sift_build_f3.rs`
  - 实验矩阵：`scripts/run_sift_ivf_experiments.sh`
- Native artifact 的“热数据”缓存（让 `f16` 更小但不慢）
  - `IvfFlatArtifactSearcher` 会在查询过程中缓存 *已解码* 的 posting lists（row_ids + vectors(f32)），避免每次 query 都重复 f16->f32 转换/重复读文件。
  - 代码：`fff-vindex/src/artifact/ivf_flat.rs`（`IvfFlatArtifactSearcher.posting_cache`）

## 现在能做的“预期实验”（可复现）

### E0：基线矩阵（已经跑通）
对比三条读取/执行路径 + 编码/缓存开关：

- `native_sidecar`：旧的 sidecar `.ivf_flat`
- `native_artifact`：artifact + native 查询
- `wasm_artifact`：artifact + Wasm 内核查询（cache on/off）

矩阵目前固定参数（见脚本）：`nlist=64, nprobe=16, k=10`，并跑 `nq=1/32`，`repeat=10` 输出 JSONL。

### E1：Size 拆分（已经能做）
利用 artifact footer 里的 chunk 目录，按 chunk_type/codec 汇总 `len/raw_len`，用于解释“压缩到底压了索引的哪一部分”。

脚本：`scripts/ivf_artifact_chunk_breakdown.py`

### E2：Cold/Warm I/O 行为（已经能做）
Wasm 路径具备 chunk cache + stats（`chunks_fetched / cache_hits / compressed_bytes_in / fetch_ms`），可以直接对比 cold vs warm。

### E3：Batch 效应（已经能做）
Wasm 内核提供 batch API（一次传 `nq>1` 个 query），可以扫 `nq` 看吞吐/跨边界成本。

> 还没做但论文必须补的：Recall@k（需要 ground truth / brute-force 或外部库计算）。

## 已跑结果（SIFT100K，最新一次）

- 运行脚本：`bash scripts/run_sift_ivf_experiments.sh`
- 本地结果目录：`results/sift_ivf_artifact_20251215_233632/`
- 汇总方式：对每个 `.jsonl` 的 `wall_ms` 取 p50/p95/p99（n=10）

| case | index_bytes | p50_ms | p95_ms | p99_ms | 备注 |
|---|---:|---:|---:|---:|---|
| native_sidecar_nq1 | 51,633,360 | 1.019 | 1.101 | 1.101 | IVF sidecar（旧路径） |
| native_sidecar_nq32 | 51,633,360 | 28.157 | 28.599 | 28.599 | nq=32 |
| native_artifact_raw_nq32 | 51,640,895 | 27.685 | 27.775 | 27.775 | artifact + raw（decoded cache 生效后接近 sidecar） |
| native_artifact_delta_nq32 | 51,355,625 | 27.720 | 28.214 | 28.214 | artifact + row_id delta-varint |
| native_artifact_raw_f16_nq32 | 26,041,127 | 27.693 | 28.119 | 28.119 | artifact + f16 vectors（~2x 更小，p50 追平） |
| native_artifact_delta_f16_nq32 | 25,755,857 | 27.678 | 28.117 | 28.117 | artifact + f16 vectors（~2x 更小，p50 追平） |
| wasm_artifact_raw_cache_nq32 | 51,640,895 | 78.345 | 80.515 | 80.515 | Wasm + host chunk cache |
| wasm_artifact_delta_cache_nq32 | 51,355,625 | 81.329 | 84.611 | 84.611 | Wasm + host chunk cache |
| wasm_artifact_raw_f16_cache_nq32 | 26,041,127 | 48.622 | 50.526 | 50.526 | Wasm + **kernel decoded cache** + f16 vectors |
| wasm_artifact_delta_f16_cache_nq32 | 25,755,857 | 48.281 | 49.055 | 49.055 | Wasm + **kernel decoded cache** + f16 vectors |
| wasm_artifact_delta_nocache_nq32 | 51,355,625 | 93.569 | 98.037 | 98.037 | Wasm + **no host cache**（会大量重复 fetch chunk） |
| wasm_artifact_delta_cache_nq1 | 51,355,625 | 3.093 | 3.235 | 3.235 | nq=1 |

说明：
- `index_bytes` 目前下降很小，是因为 **只压了 row_id**，posting chunk 里的向量仍是 `f32` 原样存（体积大头在向量）。
- `*_f16` codec 将 posting vectors 存为 `f16`，所以 index 大小约减半；native 路径由于 `IvfFlatArtifactSearcher` 缓存了已解码 posting（热），所以 p50 基本追平 `f32`。
- Wasm 路径为了证明“动态解压在 Wasm 内也不慢”，新增了 **kernel 内 decoded cache + decode/compute breakdown**：在 warm 场景下 `decode_ms≈0`（解码在 warmup 完成），主要开销落在内核的计算阶段。
- Wasm `nocache` 会显著增加 `fetch_ms / compressed_bytes_in`（见对应 `.jsonl` 的每行 stats 字段）。

### “差距主要在 compute”是怎么证明的？

Wasm 的每次迭代 JSON 行现在包含这些字段（来自 `vector_ivf_flat_demo`）：
- `fetch_ms`：host 拉 chunk 的耗时（I/O）
- `transfer_ms`：host 将 chunk 传入 Wasm 线性内存的耗时（包括 cache hit 时的拷贝/写入）
- `decode_ms`：Wasm 内核里 posting 解码耗时（主要用于 `*_f16` 的动态解压/解码）
- `compute_ms`：Wasm 内核里遍历 posting + 距离计算（扫描阶段）的耗时
- `kernel_total_ms`：Wasm 内核总耗时（包含 centroid 距离、heap/topk、扫描等所有内核工作）

因此，在 **host cache 开启** 且 **kernel decoded cache 开启** 的 warm 场景下，如果观测到：
- `fetch_ms≈0`
- `transfer_ms≈0`
- `decode_ms≈0`
而 `kernel_total_ms` 仍显著大于 0，则剩余时间只能来自 **内核计算**（centroid 选择/heap/topk/距离扫描等）。

一个具体例子（来自 `results/sift_ivf_artifact_20251215_233632/wasm_artifact_raw_f16_cache_nq32.jsonl`）：
- `fetch_ms=0.0`，`decode_ms=0.0`
- `compute_ms≈47.7ms`，`kernel_total_ms≈47.9ms`

这说明在 warm 情况下，“动态解压”已经不再是瓶颈，差距主要来自计算（尤其是 posting 扫描 + 距离核）。

### Stage profiling（native vs wasm：centroid/decode/dist/heap）

为了把 “遍历索引 / 距离计算 / topk(heap)” 的时间拆开做归因，`vector_ivf_flat_demo` 增加了 `--profile-stages`：
- native artifact：通过 `IvfFlatArtifactSearcher::search_profiled()` 输出 `centroid_ms / decode_ms / dist_ms / heap_ms`
- wasm artifact：kernel 侧通过 `ivf_last_stats_v2_ffi` 输出同名字段；host 侧仍输出 `fetch_ms / transfer_ms`

脚本（warm only，便于快速定位瓶颈）：`bash scripts/run_sift_ivf_aligned_profile.sh`  
本地结果目录（一次样例）：`results/sift_ivf_aligned_profile_20251216_001400/`

脚本（strict cold/warm，对齐缓存策略）：`bash scripts/run_sift_ivf_aligned_profile_strict.sh`  
本地结果目录：`results/sift_ivf_aligned_profile_strict_YYYYMMDD_HHMMSS/`

> 注意：stage profiling 会把距离计算与 heap 更新拆成多 pass，会额外引入开销；它的用途是“时间归因”，不是“峰值性能”。

#### Stage profiling 结果（SIFT100K，nq=32，warm，p50 over iterations）

| case | codec | index_bytes | p50_wall_ms | fetch_ms | transfer_ms | centroid_ms | decode_ms | dist_ms | heap_ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native_f32_warm | raw | 51,640,895 | 28.518 | - | - | 0.069 | 0.000 | 27.723 | 0.679 |
| wasm_f32_warm | raw | 51,640,895 | 58.597 | 0.000 | 6.963 | 0.113 | 0.000 | 49.865 | 1.297 |
| native_f16_warm | raw_f16 | 26,041,127 | 28.361 | - | - | 0.067 | 0.000 | 27.581 | 0.668 |
| wasm_f16_warm | raw_f16 | 26,041,127 | 50.467 | 0.000 | 0.001 | 0.114 | 0.000 | 48.906 | 1.303 |

### SIMD 公平性（论文里应该怎么做）

你说得对：为了公平，SIMD 必须作为一个明确的实验维度，而不是“某边默认开、某边默认关”。

建议论文报告两组设置（两边对齐），并把构建 flags 写进结果目录（可复现）：
- **Portable baseline（公平）**：native/wasm 都使用默认编译配置（不要求 wasm `simd128`），强调可移植与“默认体验”。
- **SIMD baseline（公平）**：native 使用 `-C target-cpu=native`，Wasm 使用 `-C target-feature=+simd128`（并确认 Wasmtime 支持 simd）。

两组都报 `Recall@k` + `p50/p99 latency`，并在图注明确说明编译/运行配置（否则审稿人会认为不公平）。

对应脚本（会把 `RUSTFLAGS` 写入 `results/.../build_config.txt`）：
- 严格对齐（cold/warm）：`bash scripts/run_sift_ivf_aligned_fair.sh portable` / `bash scripts/run_sift_ivf_aligned_fair.sh simd`
- Stage profiling（warm）：`bash scripts/run_sift_ivf_aligned_profile_fair.sh portable` / `bash scripts/run_sift_ivf_aligned_profile_fair.sh simd`

## 严格对齐实验（native vs wasm，cold/warm）

为避免“native 热、wasm 冷”或缓存策略不一致，提供了严格对齐脚本：

- 运行：`bash scripts/run_sift_ivf_aligned.sh`
- 本地结果目录（一次样例）：`results/sift_ivf_aligned_20251216_001446/`

该脚本固定 `nq=32, nlist=64, nprobe=16, k=10`，并对 `f32/raw` 与 `f16/raw_f16` 各自跑：
- `native warm`：native decoded posting cache 开
- `native cold`：native decoded posting cache 关
- `wasm warm`：host chunk cache 开 + wasm kernel decoded cache 开（对 f16）
- `wasm cold`：host chunk cache 关 + wasm kernel decoded cache 关（对 f16）

一轮汇总（p50 wall_ms，见 `results/sift_ivf_aligned_20251216_001446/`）：
- `native_f32_warm ≈ 27.616ms`，`native_f32_cold ≈ 59.186ms`
- `wasm_f32_warm ≈ 57.634ms`，`wasm_f32_cold ≈ 89.745ms`
- `native_f16_warm ≈ 28.177ms`，`native_f16_cold ≈ 102.289ms`
- `wasm_f16_warm ≈ 50.215ms`，`wasm_f16_cold ≈ 158.466ms`

#### 严格对齐 Stage profiling（SIFT100K，nq=32）

Stage profiling 表来自 `bash scripts/run_sift_ivf_aligned_profile_strict.sh`（每个 case 都开启 `--profile-stages`），因此：

- native 侧会给出 `centroid/decode/dist/heap/compute`
- Wasm 侧会给出 `fetch/transfer`（host）+ `centroid/decode/dist/heap/compute`（kernel）

为了让表格有参考价值，下面只保留 *SIMD 严格对齐* 的最新一组（见下节，目录：`results/sift_ivf_aligned_profile_strict_20251216_121958/`）。

#### 严格对齐 Stage profiling（Wasm `simd128` / native SIMD）

为避免“native 没开 SIMD、Wasm 开了 SIMD”造成的错觉，本文档只保留当前代码的 *SIMD 对齐* 表（见下节）。

#### 严格对齐 Stage profiling（native 显式 SIMD L2 + Wasm `simd128`）

该表用于回答“Wasm 内动态解压 + 搜索的阶段耗时，是否能和 native 对齐”。对齐方式：

- native：`l2_sq()` 使用显式 SIMD（aarch64 NEON / x86 SSE/AVX），不依赖 `-C target-cpu=native`
- Wasm：编译时开启 `simd128`（`WASM_RUSTFLAGS='-C target-feature=+simd128'`）

本地结果目录（一次样例）：`results/sift_ivf_aligned_profile_strict_20251216_121958/`

| case | codec | cache | p50_wall_ms | p50_fetch_ms | p50_transfer_ms | p50_centroid_ms | p50_decode_ms | p50_dist_ms | p50_heap_ms | p50_compute_ms | chunks_fetched |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native_f32_warm | raw | posting_cache=on | 10.790 | - | - | 0.030 | 0.000 | 9.967 | 0.737 | 10.736 | - |
| native_f32_cold | raw | posting_cache=off | 43.711 | - | - | 0.036 | 32.922 | 9.650 | 0.722 | 43.342 | - |
| wasm_f32_warm | raw | host_cache=on | 19.516 | 0.000 | 7.121 | 0.034 | 0.000 | 10.748 | 1.357 | 12.111 | 0 |
| wasm_f32_cold | raw | host_cache=off | 53.016 | 27.061 | 40.992 | 0.043 | 0.000 | 10.228 | 1.288 | 11.510 | 513 |
| native_f16_warm | raw_f16 | posting_cache=on | 11.212 | - | - | 0.032 | 0.000 | 10.360 | 0.761 | 11.153 | - |
| native_f16_cold | raw_f16 | posting_cache=off | 84.722 | - | - | 0.037 | 74.187 | 9.527 | 0.727 | 84.385 | - |
| wasm_f16_warm | raw_f16 | host_cache=on + kernel_decoded_cache=on | 14.071 | 0.000 | 0.001 | 0.037 | 0.000 | 12.446 | 1.471 | 13.909 | 0 |
| wasm_f16_cold | raw_f16 | host_cache=off + kernel_decoded_cache=off | 133.969 | 12.259 | 19.752 | 0.046 | 100.266 | 11.679 | 1.442 | 13.129 | 513 |

#### 距离计算核 microbench（native vs Wasm）

Stage profiling 的 `dist_ms` 会受计时埋点影响（尤其是 Wasm 内部 `Instant` 采样），因此补一个“只测距离核吞吐”的 microbench：

- native：直接调用 `fff_vindex::ivf_flat::l2_sq()`
- Wasm：调用导出的 `l2_microbench_query_vs_vectors_f32_ffi()`，在 Wasm 内部做 tight loop（host 只做一次调用）

命令（示例）：

```bash
# build wasm (no simd)
CARGO_TARGET_DIR=/tmp/mb_wasm_nosimd cargo build -p ivf-kernel-basic --target wasm32-wasip1 --release
# build wasm (simd128)
RUSTFLAGS='-C target-feature=+simd128' CARGO_TARGET_DIR=/tmp/mb_wasm_simd cargo build -p ivf-kernel-basic --target wasm32-wasip1 --release
# build + run microbench (native)
CARGO_TARGET_DIR=/tmp/f3_native_strict cargo build -p fff-bench --release --example l2_kernel_microbench
/tmp/f3_native_strict/release/examples/l2_kernel_microbench --wasm /tmp/mb_wasm_simd/wasm32-wasip1/release/ivf_kernel_basic.wasm
# scalar↔scalar 对齐（native 强制 scalar；Wasm 也不编译 simd128）
/tmp/f3_native_strict/release/examples/l2_kernel_microbench --native-scalar --wasm /tmp/mb_wasm_nosimd/wasm32-wasip1/release/ivf_kernel_basic.wasm
```

本机一次样例（`dim=128, count=8192, iters=1000, warmup=3`，各跑 7 次取 median，checksum 对齐）：

| 对齐组 | native ns/op (median) | wasm ns/op (median) | ratio (wasm/native) |
|---|---:|---:|---:|
| scalar ↔ scalar（`--native-scalar` / wasm no-simd） | 30.618 | 47.245 | 1.541 |
| simd ↔ simd（native 默认 / wasm `simd128`） | 6.721 | 9.805 | 1.458 |


### Size 拆分（同一份 index 文件）

以 `row_id_delta_varint_v1` 为例（`python3 scripts/ivf_artifact_chunk_breakdown.py ...`）：

- `centroids/raw`: `32,768` bytes
- `list_offsets/raw`: `520` bytes
- `posting_list/row_id_delta_varint_v1`: `51,313,752` bytes（`raw_len=51,600,256`，比例 `~0.9944`）

## 如何复现/汇总

1) 跑实验：

```bash
bash scripts/run_sift_ivf_experiments.sh
```

2) 汇总某次结果目录（文本）：

```bash
python3 scripts/summarize_ivf_jsonl.py results/sift_ivf_artifact_YYYYMMDD_HHMMSS
```

3) 看 artifact 的 chunk/codec size 拆分：

```bash
python3 scripts/ivf_artifact_chunk_breakdown.py data/sift/sift100k.f3.vindex/sift_ivf.ivf_flat.artifact
```
