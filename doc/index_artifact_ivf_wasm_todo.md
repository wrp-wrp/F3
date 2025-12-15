# IVF 索引工件（Executable Index Artifact）+ Wasm 内核：SIGMOD 导向的设计备忘与 TODO（主打 IVF）

目标：把现有 `fff-vindex` 的 IVF（先 `IvfFlat`，再 `IvfPq`）从“Rust 代码里的索引实现”升级为一个**可移植、可压缩、可演进、可治理（governable）**的索引工件（`index.bin`），并用 Wasm 内核在查询时完成“按需/部分解压 + 搜索”，host 仅负责 I/O 与缓存。

SIGMOD 导向：本文不主张“发明新 ANN/新压缩算法”，而是主张一个数据库系统视角的抽象与契约：
**把索引变成可执行的、有自描述与一致性契约的“数据工件”**，从而让压缩/搜索策略/执行内核可以随索引一起发布、演进与复现。

本文件用于“存档想法 + 详细 TODO”，以便后续按图施工。

---

## 0. 核心主张（论文叙事用，避免“只是新格式”的误解）

- **Executable Index Artifact（抽象 + 契约）**：索引不是库/代码依赖，而是一个自描述二进制工件（chunks + 目录 + 绑定 + 校验 + Wasm 内核），并且明确“谁负责 I/O、谁负责执行、谁负责一致性”。
- **Compression-as-a-Codec（机制，而非新算法）**：索引的结构化压缩（ids/posting/codes）以 chunk codec 形式固化在工件里；Wasm 在 `search()` 内按需解码/部分解压执行。codec 可以用已知方法（delta/bitpack/EF/zstd…），创新点是“工件化 + 可插拔执行”而不是 codec 本身。
- **Strategy-Follows-Index（演进与复现）**：搜索策略/参数（nprobe、early-stop、维度重排、预取计划、rerank 等）随索引文件长期保存；客户端不升级，仅换索引文件即可“升级”策略，并保持可复现（同工件→同结果/同统计）。

## 0.1 研究问题（Research Questions，写作与实验都围绕它）

- **RQ1 可移植性与成本**：同一索引工件是否能在不同宿主（语言/平台）运行？Wasm 相比 native 的成本边界在哪里（batch、SIMD、跨边界调用次数）？
- **RQ2 压缩协同执行**：把 codec 与执行内核一起固化到工件后，能否做到“更小但不慢”（或在相同延迟下更高 recall）？收益来自 I/O 变少、解压量变少、还是 cache 命中变好？
- **RQ3 演进性**：在不升级客户端的情况下，仅替换索引工件（codec/参数/策略/kernel）能否获得确定的收益，并保持向后兼容与一致性校验？

---

## 1. 设计边界（必须明确）

### 1.1 Host/Wasm 职责边界

- Host：I/O（本地文件/对象存储）、mmap/read、chunk cache、并发/异步桥接、结果拼装。
- Wasm：chunk codec 解码/部分解压、距离计算/打分、候选合并、topk、可选 rerank。

### 1.2 两种运行模式（都要支持，优先 B）

- **A. Preloaded**：host 把整个 `index.bin`（或常驻 chunk）拷进 Wasm memory；Wasm 自行按目录切片解码。
- **B. Chunk Fetch（推荐默认）**：Wasm 在 `search()` 内通过导入函数请求 `chunk_id -> bytes`（支持批量拉取），host 负责读+cache；从调用者视角仍然只调用一次 `search()`。

### 1.3 非目标（第一版先不做）

- 增量更新索引（append-only delta index 可留到 discussion）。
- 多租户/多版本并存的复杂管理（先用 manifest/路径管理即可）。
- 通用查询语言（先把 IVF 检索跑通，过滤可复用已有 `kwargs` 体系）。

### 1.4 SIGMOD 审稿常见质疑点（提前规避）

- **“这是不是在发明另一个索引文件格式？”**：回答要点是：本文核心是 *executable artifact* 的契约（目录/绑定/治理/演进/复现）与执行架构（host I/O + wasm 执行），格式只是承载该契约的最小载体。
- **“Wasm 为什么必要？”**：回答要点是：它提供可移植执行与演进边界（同工件跨宿主运行、在不升级客户端情况下更新 kernel/codec/策略），并用实验量化其成本与可控优化（batch/SIMD/减少跨界调用）。
- **“压缩不是你发明的”**：承认并利用：用已有 codec 做出 *机制* 的系统收益与可演进性，而非 claim 新算法。

---

## 2. Index Artifact 文件格式（v0，最小可用）

### 2.1 总体布局

- **Header（固定大小）**
  - magic（例如 `F3IDX\0\0\0`）
  - format_version
  - endianness / alignment
  - footer_offset
  - header_checksum（可选）
- **Body（Chunk 区）**
  - 一串 chunk：每个 chunk 是压缩后的 bytes（可选加上对齐 padding）
- **Footer（目录区）**
  - base binding：`schema_checksum/data_checksum/size`（复用 `fff-vindex/src/manifest.rs` 的概念）
  - global meta：index kind（ivf_flat/ivf_pq）、dim、metric、build_params 等
  - wasm modules：一个或多个 kernel（模块 bytes + module_id + ABI 版本）
  - chunk directory：`chunk_id -> {type, offset, len, raw_len, codec_id, codec_params, checksum, requires[], kernel_id}`

### 2.1.1 一致性契约（必须写清楚，否则容易翻车）

索引工件必须声明它绑定的数据快照（snapshot），至少包含：

- `base_path`（可选：用于可读性，不用于强一致性判断）
- `base_size`、`base_schema_checksum`、`base_data_checksum`（强校验）
- `row_id_semantics`：是否为 `row_number`（0..N-1）或需要 `rowid_remap`（若未来支持重排/删除）

加载时：
- 默认 **拒绝** 绑定不一致的索引（除非显式 `--force`），并输出诊断信息（哪个字段不匹配）。

### 2.2 Chunk 粒度（针对 IVF）

**IvfFlat v0 建议最小 chunk 集合：**
- `centroids`（f32[nlist*dim]）
- `list_offsets`（u64[nlist+1]）
- `postings_meta`（每个 list 的 {row_ids_len, payload_len} / 可选）
- `posting_list[i]`：第 i 个 list 的 posting（row_ids + vectors 或 row_ids + vector_ref）

**IvfPq v0 追加 chunk：**
- `codebooks`、`pq_params`、（可选）`transform`（PCA/OPQ/permutation）
- `codes_shard[j]`：按 shard/list 分组的 pq codes
- `row_ids_shard[j]`：codes 对应的 row_ids

### 2.3 codec 层级（建议两层）

- **容器级通用压缩（可选）**：zstd/lz4（host 或 wasm 均可实现，但 v0 建议先 host）。
- **结构化 codec（主角）**：对 row_ids/postings/codes 做 delta/bitpack/EF 等（应放在 wasm 内核里，支持边解码边遍历）。

---

## 3. Wasm Kernel ABI（v0：小而可扩展）

### 3.1 导出函数（Wasm -> host 调用点）

建议最小导出集合（名称与签名后续可用 wit/component model 固化；v0 可先用 C ABI）：

- `check(meta_ptr, meta_len, out_ptr) -> errno`
  - 返回支持的 `features`（metric、SIMD、batch、支持的 codec_id 等）。
- `init(index_meta_ptr, index_meta_len, kwargs_ptr, kwargs_len, out_handle_ptr) -> errno`
- `search(handle, queries_ptr, nq, dim, topk, kwargs_ptr, kwargs_len, out_result_ptr) -> errno`
- `free_result(result_ptr)` / `destroy(handle)`

### 3.2 导入函数（host -> Wasm，供 Wasm 拉取 chunk）

优先设计为“按 chunk_id 拉取”，避免任意 offset 读：

- `host_get_chunk(chunk_id, out_ptr, out_len_ptr) -> errno`
  - 返回一个“只读 bytes 视图”，约定其生命周期（例如：直到下一次 `host_release_chunk` 或直到 `search` 返回）。
- `host_get_chunks(chunk_ids_ptr, n, out_table_ptr) -> errno`（批量版，推荐）
- `host_release_chunk(chunk_id)`（可选）

注意：若要跨语言易适配，优先采用“host 复制到 wasm memory”的协议；若追求极致性能，再做 zero-copy 变体。

### 3.3 把跨边界开销控制成“可证明的小常数”

SIGMOD 风险点之一是“Wasm 只是慢”。因此 ABI 必须内置两条约束：

- **强制 batch**：`search(nq>1)` 为主路径，避免 per-query 频繁出入 wasm。
- **批量取 chunk**：优先 `host_get_chunks([id...])`，确保每次 query 的 host<->wasm 往返次数与 `nprobe` 同阶，而不是与 posting 内部迭代同阶。

---

## 4. 代码落地 TODO（按里程碑）

下面是“能落地 + 能写论文 + 能跑基线”的执行顺序。

### Milestone A：IvfFlat 容器化（不引入 Wasm，先把 artifact 跑通）

- [ ] A1：在 `fff-vindex` 新增 `artifact` 模块（或新 crate `fff-iartifact`），定义：
  - [ ] `IndexArtifactHeader` / `IndexArtifactFooter` / `ChunkDesc`
  - [ ] `CodecId`（先枚举：`raw`、`lz4`、`zstd`、`delta_varint`…，后两者可先占位）
  - [ ] `KernelId`（先支持 `native`，后续再 `wasm:module_id`）
- [ ] A2：实现 `write_index_artifact_ivf_flat(...)`：
  - [ ] 从现有 `fff-vindex/src/ivf_flat.rs` 构建 `IvfFlatIndex`
  - [ ] 写入 header/body/footer（chunk 目录可先非常简单：centroids/list_offsets/postings）
  - [ ] base binding 写入 footer（复用 `BaseFileBinding` 字段）
- [ ] A3：实现 `load_index_artifact(...) -> ArtifactView`：
  - [ ] 校验 magic/version
  - [ ] 校验 base binding（size/schema_checksum/data_checksum）
  - [ ] 解析 chunk 目录（提供 `get_chunk_desc(type/id)`）
- [ ] A4：提供 native 查询路径：`search_ivf_flat_artifact_native(view, query, ...)`
  - [ ] 只为验证容器正确性：能跑通 topk 结果一致性
- [ ] A5：更新/新增 bench demo：
  - [ ] `fff-bench/examples/vector_ivf_flat_demo.rs` 增加 artifact build/load/search 分支
  - [ ] 输出 size breakdown（按 chunk）

**验收标准：** `IvfFlat` 的 artifact 版本在 recall/latency 与现有 `fff-vindex` 内存结构一致（允许 tiny 误差），并能在一台机器复现实验。

### Milestone B：Wasm IVF Kernel v0（只做搜索，不做结构化压缩）

- [ ] B1：新增一个 wasm-lib（例如 `wasm-libs/ivf-kernel-basic`）：
  - [ ] 导出 `init/search/destroy`（v0 C ABI）
  - [ ] 先假设 chunk bytes 是 raw（不压缩），Wasm 内部解析 posting + 距离计算
- [ ] B2：host 侧 runtime：
  - [ ] 复用 `fff-ude-wasm` 的 wasmtime 封装思路，抽出“通用 Wasm 调用器”（更适合 `search` 这种 batch API）
  - [ ] 实现 `host_get_chunks` 导入函数：从 artifact view + cache 返回 chunk bytes
- [ ] B3：打通调用链：
  - [ ] Rust 调用者只调用一次 `search()`，Wasm 内部按需拉取 list/posting chunk
  - [ ] 支持 batch query（nq>1）以降低 host<->wasm 往返成本
- [ ] B4：对比 native：
  - [ ] 记录 wasm vs native 的 p50/p99、QPS、CPU

**验收标准：** Wasm IVF Flat 能跑通，性能达到“可用”的基线（先不追 75–85%，但要能解释开销来源）。

### Milestone C：把“部分解压/结构化 codec”内置进 Wasm（论文主打点之一）

- [ ] C1：定义结构化 codec 的接口（Wasm 内核内部）：
  - [ ] `decode_posting_iter(bytes, params) -> iterator(row_id, payload)`
  - [ ] 支持“边解码边遍历”（不要强制整块 materialize）
- [ ] C2：实现第一批稳赢 codec（lossless）：
  - [ ] `row_ids`: delta + varint / bitpacking
  - [ ] `list_offsets`: delta（可选）或 raw
  - [ ] `codes`（为 IvfPq 预留）：bitpacking/byte-aligned unpack
- [ ] C3：artifact writer 侧支持为 chunk 选择 codec：
  - [ ] footer chunk desc 记录 `codec_id + params + raw_len`
  - [ ] writer 负责把 raw -> encoded（此处 encoded 算法可以在 host 写，Wasm 只需 decode；或两边都实现以支持 future re-encode）
- [ ] C4：实验链路：
  - [ ] size 降幅（按 chunk 分解）
  - [ ] query 解压字节数（每次 query 解了哪些 chunk、解了多少 raw_len）
  - [ ] latency 影响与解释（是否因为减少 I/O 或减少解压量获益）

**验收标准：** 在相同 recall 下，artifact size 明显下降，p99 不显著退化（或退化可通过 batch/SIMD/减少 chunk 拉取次数修复）。

### Milestone D：IvfPq（让“策略随索引走”变成实证）

- [ ] D1：实现/完善 `IvfPq` artifact chunk 切分：
  - [ ] codebooks/params/transform 单独 chunk
  - [ ] codes/row_ids 按 shard 或按 list 分 chunk
- [ ] D2：Wasm 内核支持 PQ 路径：
  - [ ] LUT 构建与 batch scoring
  - [ ] rerank 可选（用原始向量回表或用残差）
- [ ] D3：策略固化（至少选 1–2 个“换文件升级”可展示的点）：
  - [ ] 维度重排 permutation（来自训练）
  - [ ] early-stop 参数（子空间误差界/分段累积上界）
  - [ ] nprobe 自适应策略（根据 query 统计）
- [ ] D4：演进性实验：
  - [ ] 同一个 host 二进制不变，加载 `index_v1.bin` 与 `index_v2.bin`
  - [ ] v2 只改策略/codec/params，展示 size/latency/recall 改善

---

## 5. 评测与论文产物 TODO（与里程碑绑定）

### 5.1 必须有的指标与日志

- [ ] Recall@k、p50/p95/p99 latency、QPS vs 并发
- [ ] Index size breakdown（按 chunk_type）
- [ ] 每次 query 的 I/O/解压统计：`chunks_fetched`, `compressed_bytes`, `raw_bytes`, `decode_time`
- [ ] Wasm vs native 的开销归因：跨边界调用次数、复制次数、SIMD 开关、batch 大小

### 5.1.1 SIGMOD 级别的“证据链”输出（建议实现为可开关的 tracing）

每次 `search()` 至少记录：

- `nprobe`、`lists_touched`
- `chunks_fetched`（id 列表可采样）、`compressed_bytes_in`、`raw_bytes_decoded`
- `time_breakdown`：fetch、decode、distance、topk merge（粗粒度即可）

用这些证据链回答：收益到底来自哪里？是否真的“按需解压”？是否存在 I/O 碎片化？

### 5.2 Baseline 与 ablation（写作必须）

- [ ] Native IVF（同数据布局）vs Wasm IVF
- [ ] 不压缩 vs 通用压缩（zstd/lz4）vs 结构化 codec（Wasm decode）
- [ ] “全量解码” vs “按需解码”（通过 chunk 粒度与 nprobe 控制）
- [ ] 策略开关 ablation（至少 1 项：维度重排或 early-stop）

### 5.2.1 需要提前准备的“外部基线”（避免被质疑闭门造车）

尽量至少覆盖一种工业/学术常用实现作为参照（即便只在单机上跑）：

- IVF Flat / IVF PQ：例如 Faiss 的 IVF（同 nlist/nprobe/dim），对比 size/latency/recall 的趋势与数量级。

注：论文不必宣称“比 Faiss 更快”，但要证明你的机制带来明确的 **可移植/可演进/可治理** 与 **压缩协同执行** 的系统优势，并量化其代价。

### 5.2.2 失败模式实验（SIGMOD 加分项）

- [ ] 当 chunk 粒度过细时：跨边界调用与小 I/O 导致 p99 恶化（展示并给出建议策略：批量取 chunk、合并小 chunk、预取）
- [ ] 当压缩过强时：decode 成为瓶颈（展示并给出 codec 选择/参数建议）

### 5.3 代码组织建议（减少返工）

- [ ] 把 artifact 规范写成稳定文档（本文件未来拆成 `spec.md` + `todo.md`）
- [ ] 把 ABI/codec id 固化（版本化），避免后期“接口推倒重来”
- [ ] bench 与数据处理脚本收敛到 `fff-bench`/`scripts`，确保可复现

---

## 6. 当前仓库落点提示（便于开工）

- IVF 实现入口：`fff-vindex/src/ivf_flat.rs`（现成 build/load/search）
- 多索引 catalog：`fff-vindex/src/manifest.rs`
- Wasm runtime 参考：`fff-ude-wasm/src/lib.rs`（wasmtime 封装、内存传参模式）
- kwargs 体系（可复用到 index kernel）：`fff-ude/src/kwargs.rs` + `format/kwargs.md`

---

## 7. “最小可 SIGMOD”范围建议（防止 scope 爆炸）

如果时间紧，优先保证以下组合能闭环并且有强实验：

- **IvfFlat artifact + Wasm kernel + 结构化 row_id/posting codec（至少 1 个）**
- **一个“换文件升级”点**（例如仅升级 codec 或仅升级 early-stop/维度重排）
- **强证据链**（chunks/bytes/time breakdown）+ **native baseline**

IvfPq 可作为“扩展性与更大收益”的后续章节/附录，但不要让它成为主线阻塞项。
