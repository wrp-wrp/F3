# F3 文件格式结构说明

本文基于 `format/File.fbs` 中的 FlatBuffer 定义及 `fff-poc/src/{writer,reader}` 的实现，对 F3 文件的二进制布局做一个中文说明，方便在 macOS/Debian 以外的环境中理解和实现兼容的读写器。

## 1. 顶层布局

F3 文件采用“数据 + 元数据 + Postscript”自描述布局，整体顺序如下：

```text
┌──────────────────────────────────────────┐
│ Row Group 0                              │
│   Column Chunks + EncUnits               │
├──────────────────────────────────────────┤
│ ... more row groups ...                  │
├──────────────────────────────────────────┤
│ Row Group R                              │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ Wasm Binaries (可选)                     │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ Column/RowGroup Metadata sections (|A|)  │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ Statistics / 其他扩展区 (|B|, 可为空)    │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ Footer (FlatBuffer Footer root)          │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ Postscript (固定 32B + "F3")             │
└──────────────────────────────────────────┘
```

`Postscript` 位于尾部，包含元数据尺寸、Footer 尺寸、压缩算法、数据校验和、Schema 校验和以及版本号。Reader 始终以 `MAGIC="F3"` 验证合法性（参见 `reader::read_postscript`）。

## 2. Row Group 与列块

每个 Row Group 保存一组行，其内部再按照列切分为多个 `Chunk`（代码里也称 IOUnit）。要点如下：

- **Chunk / IOUnit**  
  - 拥有 `offset`、`size`、`num_rows` 等基本信息；  
  - 可以被分割为多个 `EncUnit`（编码单元），每个 EncUnit 默认覆盖固定数量的值（64K，可由编码控制），并可以指定压缩及 `Encoding`（Plain、Vortex cascade、Custom WASM 等）。
- **字典策略**  
  - `DictionaryEncoding` 支持 `NoDictionary`、`LocalDictionary`、`SharedDictionary` 三种。共享字典的块被集中存放在 Footer 的 `shared_dictionary_table` 里，与列块之间通过 `dictionary_positions` 建立映射。
- **Wasm 关联**  
  - `Encoding` 可以引用 `WASMEncoding`（`wasm_id`、子 EncUnit 尺寸等）。Reader 根据 wasm_id 到 `OptionalMetadataSections["WASMBinaries"]` 中取出实际字节。

行组本身的“轻量信息”存储在 Footer 的 `RowGroups` 表中（行数、偏移、大小），而每个列的详细 `ColumnMetadata` 则放在单独的 `MetadataSection`，通过 `RowGroupMetadata.col_metadatas` 指向以便投影时只读取需要的列。

## 3. Footer 内容

Footer 是整个文件最核心的 FlatBuffer 对象，包含：

1. **Arrow Schema**：序列化后的 IPC 消息（由 `fff-poc::writer` 使用 Arrow IPC API 生成）。Reader 以此恢复列及逻辑类型。
2. **LogicalTree**：比 Arrow schema 更贴近物理布局的逻辑节点树（Flat/List/Struct/组合等）。
3. **RowGroups**：每个行组的 `row_counts / offsets / sizes`，以及指向列级元数据的 `RowGroupMetadata` 数组。
4. **OptionalMetadataSections**：命名的扩展段。目前“WASMBinaries”用于保存 Wasm 解码器，也可以扩展存储列 UUID、ZoneMap 等信息。
5. **EncodingVersions**：记录各 `EncodingType` 的语义版本（`SemVer`），便于读写双方做兼容性协商。
6. **SharedDictionaryTable**：集中存放共享字典块和其引用关系。

Writer 在 `fff-poc/src/writer.rs` 中生成上述结构，并在写 Footer 前将 Arrow schema 的序列化结果参与 `schema_checksum` 计算，保证跨语言读取时能够判定 Schema 是否被篡改。

## 4. Wasm 二进制段

F3 的可扩展性来自嵌入式 Wasm 解码器：

- 所有 Wasm 模块以顺序的方式写入数据区之后、元数据之前。
- Footer 中的 `OptionalMetadataSections` 会记录每个模块的 `offset/size/compression_type`，并以名称（目前固定为 "WASMBinaries"）区分。
- EncUnit 的 `encoding` 字段引用 `wasm_id`，Reader 会按需加载并缓存（参见 `fff-ude-wasm` 与 `fff-poc/src/decoder`）。

这种设计允许新的编码算法无需升级核心库即可在文件内部发布。

## 5. Postscript 与校验

Postscript 的布局（均为小端）：

| 字段 | 字节数 | 说明 |
| --- | --- | --- |
| metadata_size | 4 | 元数据总长度（从 `MetadataSection` 开始到 Footer 结束） |
| footer_size | 4 | Footer FlatBuffer 的字节数 |
| footer_compression | 1 | 目前固定为 `CompressionType::Uncompressed` |
| checksum_type | 1 | 目前为 `ChecksumType::XxHash` |
| data_checksum | 8 | 实际列数据的校验和，写入顺序与 schema 一致 |
| schema_checksum | 8 | Arrow schema IPC 消息的校验和 |
| major_version | 2 | 主版本号 |
| minor_version | 2 | 次版本号 |
| magic | 2 | ASCII `"F3"` |

Reader 首先读取末尾固定大小的 Postscript，进而一次性请求 [metadata_size] 范围的数据，无需随机 I/O 多次跳转。

## 6. 读写流程概览

**写入**（`fff-poc::writer`）：
1. 根据输入的 Arrow RecordBatch/Array，将列数据编码为 EncUnit，顺序写入列块区域，同时累积 `data_checksum`。
2. （可选）写入 Wasm 模块或共享字典块。
3. 构建 ColumnMetadata、RowGroupMetadata 等 FlatBuffer 对象，将它们以 `MetadataSection` 形式附加到文件末尾。
4. 生成 Footer（含 schema、logical tree、row group 摘要、可选段等），写入后更新 `metadata_size`。
5. 在文件尾部追加 Postscript。

**读取**（`fff-poc::reader`）：
1. 通过 `read_postscript` 获取 Footersize、校验和、版本信息。
2. 读取 Footer，获取 RowGroup 摘要和可选段位置。
3. 根据投影/筛选需要，定位并加载列元数据，再按 EncUnit 读取数据块；当遇到 `EncodingType::CUSTOM_WASM` 或未内置的编码时，加载相应的 Wasm 模块执行解码。
4. 根据 Schema 恢复 Arrow 数组或执行自定义计算。

## 7. 参考目录

- `format/File.fbs`：格式权威定义。
- `fff-format/`：由 FlatBuffer 生成的 Rust 绑定。
- `fff-poc/src/writer.rs`、`fff-poc/src/reader/mod.rs`：具体写入/读取逻辑。
- `fff-ude-wasm/` 与 `wasm-libs/`：Wasm 解码器与示例。

希望这份中文文档能帮助你在需要深入了解 F3 布局、实现新的 Reader/Writer 或调试 Wasm 解码器时快速定位关键结构。
