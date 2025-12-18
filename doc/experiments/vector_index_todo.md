# Vector Index TODO (Executable Index Artifact + Wasm IVF)

目标：把向量索引从“库/代码依赖”升级为一个**可执行索引工件（Executable Index Artifact）**：索引文件自描述（chunks + 目录 + base binding + 校验），并可携带/选择执行内核（native/Wasm）与搜索策略，使其具备可移植、可演进、可复现、可治理的属性。

本文件是仓库中关于向量索引（IVF 为主）的**唯一 TODO**。此前分散的 TODO/plan 已合并到这里。

## 0. 研究问题（写作与实验围绕）

- **RQ1 可移植性与成本**：同一索引工件跨宿主运行是否可行？Wasm 相比 native 的成本边界在哪里（batch、SIMD、跨边界搬运/调用）？
- **RQ2 压缩协同执行**：把 codec 与执行内核固化到工件后，能否做到“更小但不慢”（或同延迟更高 recall）？收益来自 I/O 变少、解压量变少、还是 cache 命中变好？
- **RQ3 演进性**：不升级客户端，只替换索引工件（codec/参数/策略/kernel）能否获得确定收益，并保持向后兼容与一致性校验？

## 0.1 RQ1 证据包（必须产出“强证据”）

RQ1 的目标不是“证明 Wasm 很快”，而是给出**可复现、可解释的成本边界**：在什么条件下 Wasm 的额外开销可以被 batch/SIMD/缓存摊薄到一个小常数。

已跑通的 RQ1 结果摘要：`doc/experiments/rq1_results.md`。
最新 RQ1 实验输出（含修复后 “steady-state host copy≈0”）：`results/rq1_simd_ablation_v3/`。

### Claim-RQ1.1：Wasm 的主要额外成本来自跨边界数据搬运与边界开销，而非距离计算本身

**对照设计（只改 1 个因素）：**
- 固定算法与参数：IVF-Flat，`nlist/nprobe/k/dim` 固定。
- 变量：`nq`（batch size）。
- 输出：`avg_latency_ms` 与 `host_copy_time_ms`（或 stage profiling 的 `transfer_ms`），并给出 `ratio_to_native`。

**验收标准：**
- `nq` 增大时，`avg_latency_ms / nq` 下降并趋于平稳。
- `host_copy_time_ms`（或 `transfer_ms`）在总耗时中的占比随 `nq` 下降或保持“可解释的小常数”。

**一键复现：**
- 脚本：`exp_scripts/rq1_batch_sweep.sh`
- 输出目录：`results/rq1_batch_sweep/`
- 汇总脚本：`scripts/summarize_rq1_batch_sweep.py`

### Claim-RQ1.2：SIMD 是 Wasm 成本边界的必要条件（否则计算部分失真）

**对照设计：**
- 同一套参数与数据集，唯一变量：Wasm kernel 是否开启 `+simd128`。
- 输出：`avg_latency_ms` 与（若可用）`compute_time_ms`。

**验收标准：**
- `+simd128` 显著降低 compute 相关耗时，使“boundary/copy”成为可见主瓶颈（否则 RQ1 的归因会混淆）。
  - 注意：在 `wasm32-wasip1` 上 `simd128` 可能是默认开启的；要做“无 SIMD”对照需要显式传 `-C target-feature=-simd128`。

**一键复现：**
- 脚本：`exp_scripts/rq1_simd_ablation.sh`
- 输出目录：`results/rq1_simd_ablation_v3/`（可自定义；旧结果在 `results/rq1_simd_ablation/`）
- 对比脚本：`scripts/compare_rq1_simd_ablation.py`
  - 建议强证据用例：`WARMUP=5 ITERS=100 NQ_LIST="100" CODECS="raw raw_f16" bash exp_scripts/rq1_simd_ablation.sh results/rq1_simd_ablation_v6_nq100_iters100`

### Claim-RQ1.3：当工作集能常驻 Wasm（例如 f16 或更强压缩 + decoded cache），跨边界搬运可近似消失

**对照设计：**
- 固定 `nq`，比较 `raw_f32` vs `raw_f16`（或 SQ8）在 warm cache 下的 `host_copy_time_ms/decoded_cache_hits`。

**验收标准：**
- warm 场景下 `host_copy_time_ms` 接近 0（或显著下降），并且 `decoded_cache_hits` 很高；总耗时更接近 native。

## 1. 现状盘点（以代码为准）

### 1.1 已完成（能跑通）

- [x] IVF-Flat 索引构建/查询（native）与 sidecar 路径：`fff-vindex/src/ivf_flat.rs`
- [x] 多索引 manifest（base binding + entries）：`fff-vindex/src/manifest.rs`
- [x] IVF-Flat artifact 容器（chunk 目录 + footer JSON + base binding + checksum 校验）：`fff-vindex/src/artifact/ivf_flat.rs`
- [x] Artifact native 查询（按 list 拉 posting，支持 posting decode cache）：`fff-vindex/src/artifact/ivf_flat.rs` (`IvfFlatArtifactSearcher`)
- [x] Wasm IVF-Flat kernel（batch search、stage profiling、decoded cache、可选 host distance kernel）：`wasm-libs/ivf-kernel-basic/src/lib.rs`
- [x] Wasm host runtime（wasmtime + chunk cache + stats/transfer）：`fff-vindex/src/artifact/wasm_ivf_flat.rs`
- [x] 可复现实验脚本（SIFT100K 矩阵、严格对齐 profiling、CSV/JSONL 汇总、chunk breakdown）：`scripts/`
- [x] Recall/ground-truth 基础实现：
  - `fff-vindex/tests/recall.rs`
  - `fff-vindex/tests/sift_100k.rs`（ignored，本地数据集）

### 1.2 已部分完成（存在“跑得通但不稳/不全”的缺口）

- [~] IVF-PQ artifact 构建 + Wasm 路径（kernel 已有 PQ 分支；wrapper 可用但不完善）：
  - `fff-vindex/src/artifact/ivf_pq.rs`
  - `wasm-libs/ivf-kernel-basic/src/lib.rs`（`posting_codec == 6`）
  - `fff-vindex/src/artifact/wasm_ivf_pq.rs`（`host_l2_sq_batch_f32` 目前 stub）

- [~] SQ8 / U8 posting codec：
  - 代码中已出现 `PostingCodec::{RawU8, RowIdDeltaVarintV1U8}` 与解码/距离计算路径（host+kernel 各有一部分）。
  - 但 `quantization_params` 的“统一来源/持久化语义”还没有收敛成稳定契约，实验也未形成闭环（recall/latency/size）。

## 2. 统一目标与“验收标准”（每个阶段必须产出什么）

### 2.1 统一输出（每次跑实验必须具备）

- [ ] **Recall@k**（至少 Recall@10）+ **p50/p95/p99 latency**
- [ ] **Index size breakdown**（按 chunk_type/codec）
- [ ] **每次 query 的证据链**（可采样）：`chunks_fetched`, `compressed_bytes_in`, `raw_bytes_decoded`, `decode_time`, `dist_time`, `heap_time`
- [ ] **配置记录**（必须写进输出）：dataset、dim、nlist、nprobe、nq、SIMD 开关、cache 开关、decoded cache budget、codec/params

### 2.2 “策略随索引走”演进实验（必须能一键复现）

- [ ] 同一个 host 二进制不变，只替换 index artifact 文件：
  - `index_v1`：baseline（例如 f16）
  - `index_v2`：仅改策略/codec（例如 SQ8 或 reorder+early-stop）
  - 对比：size/latency/recall 并给出归因（bytes/time）

## 3. Backlog（按优先级，从“能形成研究结论”开始）

### P0（最高优先级）：把“研究结论链”补齐

- [x] 把 Recall@k 纳入 `vector_ivf_flat_demo` 的输出（native/wasm 都可跑），复用 brute-force/ground truth 逻辑（`fff-bench/examples/vector_ivf_flat_demo.rs` 支持 `--recall [--recall-queries N]`）。
- [x] 固化一个“最小实验矩阵”（codec × cache × nq × nprobe）并产出可比较的表格：`exp_scripts/p0_min_matrix.sh` + `scripts/summarize_p0_min_matrix.py`。
- [x] 修复/统一 `results/ivf_wasm_bench/all_bench_results.csv` 的字段（`exp_scripts/wasm_ivf_bench.sh` 的 header 已与行列对齐）。

> RQ1 优先：先把 batch sweep（`nq` 轴）与 SIMD 对照跑成“强证据表”，再继续做更复杂的 codec/策略。

### P1：SQ8（U8）闭环（对应“更小但不慢”的主张）

- [ ] Host 构建端：统一计算并写入 `quantization_params`（全局 min/max 或 per-dim/per-block 策略先选一个最简单可解释的）。
- [ ] Artifact writer：将 quant params 写入 footer，并在 manifest 中记录（便于外部工具读取）。
- [ ] Wasm kernel：从 `dir` 或 chunk/meta 中读取 quant params，完成 `u8 -> f32`（或直接 u8 距离核）并支持 SIMD。
- [ ] Bench：把 SQ8 加入与 f16 同级的对比矩阵（native/wasm；cold/warm；nq sweep）。
- [ ] 产出：SQ8 的 size/latency/recall Pareto，并解释收益来自哪里（bytes/time）。

### P2：维度重排 + early-stop（做出“策略随索引走”的最小演进点）

- [ ] Host 构建：训练统计（方差/能量）并生成 permutation；写入 footer。
- [ ] Wasm kernel：使用 permutation 做“先算高贡献维度”的距离累加，并在超过 topk worst 时 early-stop。
- [ ] 演进实验：同一 host，`index_v1`（无策略）vs `index_v2`（reorder+early-stop），对比 p99/吞吐与 recall。

### P3：IVF-PQ 收敛（作为扩展章节/后续主线）

- [ ] 修复 `fff-vindex/src/artifact/wasm_ivf_pq.rs` 的 stub（至少实现必要的 host 回调一致性）。
- [ ] 完善 PQ artifact 的 chunk 切分与元信息（codebooks/params/transform/codes/row_ids）。
- [ ] 增加 PQ 的 recall/latency/size 实验矩阵，并纳入统一输出格式。

### P4：接口/规范稳定化（避免后期推倒）

- [ ] 固化 ABI/codec id 与版本化策略（dir layout/aux chunk id 等），写成稳定 spec（从本文件拆出 `spec.md`）。
- [ ] 支持 `host_get_chunks([id...])` 批量拉取，验证小 chunk 粒度下 p99 的失败模式并给出策略（合并/预取/批量）。

## 4. 建议的“接下来两周”执行方案（按最小闭环）

### Week 1：把指标闭环打通（P0）

1) `vector_ivf_flat_demo` 增加 `--recall`（或默认在 `--json` 输出里加 recall 字段），实现 brute-force ground truth（对 SIFT 可限制 queries 数）。
2) 统一输出 schema（JSONL/CSV），确保每条结果都记录 config + size/bytes/time。
3) 修复/整理合成基准 CSV 的列定义，避免 header/row 不一致。

### Week 2：先做 SQ8 的最小闭环（P1）

1) 选定 quant params（全局 min/max），把它作为 footer 的稳定字段写入并贯通 host+wasm。
2) 扩展脚本：在 SIFT 与 synthetic 两个数据集上跑 `f32/f16/sq8`，输出 recall/latency/size 三元组。
3) 如果 recall 掉得明显，再讨论改成 per-dim/per-block 或 residual quantization（但先用最简单版本产出一条 Pareto 曲线）。

## 5. 常用入口（方便开工）

- IVF 入口：`fff-vindex/src/ivf_flat.rs`
- IVF-Flat artifact：`fff-vindex/src/artifact/ivf_flat.rs`
- Wasm kernel：`wasm-libs/ivf-kernel-basic/src/lib.rs`
- Wasm host runtime：`fff-vindex/src/artifact/wasm_ivf_flat.rs`
- Demo/runner：`fff-bench/examples/vector_ivf_flat_demo.rs`
- 脚本：`scripts/run_sift_ivf_experiments.sh`、`scripts/run_sift_ivf_aligned_profile_strict.sh`
