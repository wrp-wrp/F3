# 外置向量索引（IVFFlat → IVFPQ mixed-bit）工程实现文档

本文档描述在本仓库（F3/fff-poc）中新增“外置向量索引”能力的工程方案与实施步骤。目标是：

- 主数据文件仍为 F3（`.f3`），索引文件存在外部 sidecar，不修改主文件数据区。
- 同一数据文件支持多个索引，并可在读取/查询时选择。
- 第一阶段优先实现 `IVFFlat`（验证接口与闭环）。
- 第二阶段实现 `IVFPQ`，并支持 **mixed-bit PQ**：不同子量化器（子空间）分配不同 bit 数（per-subquantizer bit allocation）。
- 工程上保持模块化：索引能力放入独立 crate，不反向污染 `fff-poc` 核心逻辑；`fff-poc` 只补齐通用能力（如多行回表、FixedSizeList 读写）。

---

## 1. 落盘形态与多索引组织

给定数据文件：`/path/data.f3`

推荐索引目录结构：

- `/path/data.f3.vindex/manifest.json`
- `/path/data.f3.vindex/<index_name>.ivf_flat`（IVFFlat 索引本体，自定义格式）
- `/path/data.f3.vindex/<index_name>.ivf_pq`（IVFPQ 索引本体，自定义格式，第二阶段）

说明：

- `manifest.json` 作为 catalog，列出所有索引条目（多索引）。
- 每个索引条目是一个独立文件，便于版本化、替换、对比评测。

---

## 2. 基础假设与 RowId 定义

本项目当前的读写方式是典型 footer-based 格式：数据写在前，Footer + Postscript 在末尾，并包含 checksum/offset 等信息；读端按文件尾部定位元数据。并且对象存储 reader 侧默认假设文件大小不会变化。

因此向量索引实现默认采用：

- **主数据文件写完不改（不可变）**
- `row_id = 逻辑行号（0..N-1）`

好处：索引里存储 `row_id` 即可稳定引用数据行，不需要额外“主键→行号”映射。

---

## 3. 向量列类型（数据文件）

索引构建需要从数据文件读取“向量列”。建议目标类型为：

- `FixedSizeList<Float32>(dim)`

原因：

- 维度固定，利于训练/量化/距离计算，且避免 List 的变长 offsets 开销。
- 便于后续 PQ code 固定长度。

**实现注意**：当前 `fff-poc` 的 logical encoder/decoder 尚未覆盖 `FixedSizeList`，需要补齐（见第 8 节）。

---

## 4. Manifest（多索引 catalog）定义

`manifest.json` 建议结构：

- `base`：绑定主文件（路径、大小、schema_checksum、data_checksum）
- `indexes[]`：索引条目列表

每个条目至少包含：

- `name`：索引名（选择用）
- `path`：索引文件相对/绝对路径
- `kind`：`ivf_flat` / `ivf_pq`
- `vector_leaf_index`：向量列在 Arrow schema 的 leaf 索引（与 Projection 保持一致）
- `dim`、`metric`
- `build_params`（nlist、train_sample、seed…）
- `quantization`（IVFPQ 才需要，包含 mixed-bit 配置）

索引加载时必须校验 `base` 信息，避免索引与数据文件错配。

---

## 5. IVFFlat 索引文件格式（自定义二进制，便于 WASM codec）

IVFFlat 的核心数据长度天然不一致：`centroids(nlist*dim)`、`offsets(nlist+1)`、`row_ids(N)`、`vectors(N*dim)`，并且索引访问模式偏“按 list / block 随机读”。

因此索引文件更适合做成**自定义二进制结构**（sidecar），并把压缩/解码/距离计算作为可插拔 codec（后续可由 WASM 管理）。

当前实现的最小落盘格式（little-endian）：

- magic: `b"F3IVFF1\\0"`（8 bytes）
- version: `u32`
- dim: `u32`
- nlist: `u32`
- vector_leaf_index: `u32`
- base_schema_checksum: `u64`
- base_data_checksum: `u64`
- centroids_len: `u64`（元素个数，f32）
- list_offsets_len: `u64`（元素个数，u64，= nlist+1）
- row_ids_len: `u64`（元素个数，u32）
- vectors_len: `u64`（元素个数，f32，= row_ids_len*dim）
- payload：依次写入 `centroids[f32]`、`list_offsets[u64]`、`row_ids[u32]`、`vectors[f32]`

索引元信息以 `manifest.json` 为准（多索引 catalog、base 绑定校验、build params、quantization 等）。

---

## 6. IVF 公共层（Flat/PQ 共用）

推荐在 `fff-vindex` crate 内分层：

- `ivf::kmeans`：训练 coarse centroids（L2；cosine 可在外部先 normalize）
- `ivf::assign`：将向量分配到最近 centroid，构造 CSR posting lists
- `ivf::search`：对 query 选 nprobe 个 centroid，得到 candidate ranges

IVFFlat/IVFPQ 差异只在 posting payload 与打分方式。

---

## 7. IVFPQ mixed-bit（第二阶段）

这里的 mixed-bit 指：

- per-subquantizer bit allocation：`nbits[j]`（第 j 个子量化器分配的 bit 数），而不是“距离加权/逐维缩放”。

设计要点：

- 先做 `IVF + PQ` 的标准流程，再扩展到 mixed-bit：
  - 固定 `m` 个子空间
  - `nbits: Vec<u8>` 长度为 `m`
  - `total_bits = Σ nbits[j]`，若固定则 `code_bytes = ceil(total_bits/8)` 固定
- 索引落盘：
  - `codes: FixedSizeBinary(code_bytes)`（固定长度更好）
  - `pq_codebooks`：按实现选择存为若干列或一列 binary（需可重建 LUT）
- 查询：
  - 对 query 构建 LUT（每个子空间一个 lookup table）
  - 对每个 code 以查表求和方式得到近似距离/相似度
  - 需要时再用原始向量精排（可回表或存储重建向量）

建议接口抽象：

- `trait Quantizer`：`train/encode/prepare_query/score/meta`
- `QuantizerMeta`：记录 `m/nbits/packing/total_bits` 等用于读写与兼容检查

---

## 8. 必要的 fff-poc 通用能力补齐

### 8.1 多行回表（Selection::RowIndexes）

当前 reader 对 `Selection::RowIndexes(Vec<u64>)` 的实现只读取 `row_indexes[0]`，无法一次回表 topK 多行。向量检索回表必须支持：

- 在一个 row group 内读取多个 row_id
- 将离散 row_id 合并为若干 ranges，减少 I/O 与解码开销
- 最终按请求顺序 gather 并组成 RecordBatch

该增强属于通用能力，应实现于 `fff-poc`，并被 `fff-vindex` 调用。

### 8.2 FixedSizeList 读写支持（数据文件向量列）

为了让数据文件能用 `FixedSizeList<Float32>(dim)` 表示向量列，需要在 logical encoder/decoder 中增加对 `DataType::FixedSizeList` 的支持。

建议逻辑层映射为两个物理列：

- validity：`Boolean`（一列）
- values：子元素类型（例如 `Float32`），长度为 `num_rows * dim`（一列）

decoder 再将 (validity, values) 组合成 `FixedSizeListArray`。

---

## 9. 实施步骤（可执行清单）

Phase 1（IVFFlat 闭环）：

1. 新增 crate `fff-vindex`（workspace member）
2. 实现 `manifest.json` 读写 + `IndexCatalog`（多索引）
3. 实现 IVFFlat 构建（kmeans + assign + 写索引 `.ivf_flat`）
4. 实现 IVFFlat 查询（nprobe 召回 + vectors 精排）
5. 提供 `fff-bench/examples/` demo（build + search）

Phase 2（回表）：

1. `fff-poc` 实现多 row_id Selection
2. `fff-vindex` 实现 `search_and_fetch`（topK row_id 回表读取其它列）

Phase 3（IVFPQ mixed-bit）：

1. Quantizer 抽象与 `PqMixedBitQuantizer` 实现
2. 索引 schema 增加 `codes` + `codebooks`
3. 与 IVFFlat 的多索引选择联动（优先级可配置）
