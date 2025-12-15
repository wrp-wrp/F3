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
- Wasm IVF 搜索内核（解码+搜索在 Wasm 内部执行；I/O 由 host 回调提供）
  - Wasm：`wasm-libs/ivf-kernel-basic/src/lib.rs`
  - Host runtime（Wasmtime + WASI + chunk cache + fetch stats）：`fff-vindex/src/artifact/wasm_ivf_flat.rs`
- 端到端 SIFT100K 跑通脚本
  - 下载数据：`scripts/download_sift_100k.sh`
  - 生成 base `.f3`：`fff-bench/examples/sift_build_f3.rs`
  - 实验矩阵：`scripts/run_sift_ivf_experiments.sh`

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
- 本地结果目录：`results/sift_ivf_artifact_20251215_170211/`
- 汇总方式：对每个 `.jsonl` 的 `wall_ms` 取 p50/p95/p99（n=10）

| case | index_bytes | p50_ms | p95_ms | p99_ms | 备注 |
|---|---:|---:|---:|---:|---|
| native_sidecar_nq1 | 51,633,360 | 1.017 | 1.019 | 1.019 | IVF sidecar（旧路径） |
| native_sidecar_nq32 | 51,633,360 | 27.775 | 29.267 | 29.267 | nq=32 |
| native_artifact_raw_nq32 | 51,640,895 | 63.387 | 65.765 | 65.765 | artifact + raw posting |
| native_artifact_delta_nq32 | 51,355,625 | 64.446 | 66.121 | 66.121 | artifact + row_id delta-varint |
| wasm_artifact_raw_cache_nq32 | 51,640,895 | 77.844 | 80.540 | 80.540 | Wasm + cache |
| wasm_artifact_delta_cache_nq32 | 51,355,625 | 83.853 | 87.286 | 87.286 | Wasm + cache |
| wasm_artifact_delta_nocache_nq32 | 51,355,625 | 100.036 | 103.197 | 103.197 | Wasm + **no cache**（会大量重复 fetch chunk） |
| wasm_artifact_delta_cache_nq1 | 51,355,625 | 3.238 | 3.363 | 3.363 | nq=1 |

说明：
- `index_bytes` 目前下降很小，是因为 **只压了 row_id**，posting chunk 里的向量仍是 `f32` 原样存（体积大头在向量）。
- Wasm `nocache` 会显著增加 `fetch_ms / compressed_bytes_in`（见对应 `.jsonl` 的每行 stats 字段）。

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
