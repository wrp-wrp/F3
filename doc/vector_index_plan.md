# F3 向量索引支持技术路线

## 1. 格式与元数据扩展

1. 在 `format/File.fbs` 扩展以下结构：
   - `VectorIndexDescriptor`：描述 `index_id`、目标列、算法类型、距离度量、版本号、优先级、量化信息、Wasm 模块引用，以及指向真实索引数据的 `MetadataSection`。
   - `QuantizationSpec` / `QuantizationSegment`：支持 per-dimension（或 per-segment）记录量化方法、bit 宽、子空间范围、额外参数（字节数组）。
   - 可选 `VectorIndexSection`：若希望索引主体/辅助数据分段存储，可以复用 `MetadataSection` 列表。
2. Footer 新增 `vector_indexes: [VectorIndexDescriptor]`；OptionalMetadataSections 中约定 `"VectorIndexes"` 名称空间存放索引原始字节，`"VectorIndexWasm"` 存放索引查询逻辑 Wasm 模块（也可沿用 `"WASMBinaries"`，但建议独立命名以便区分）。
3. Descriptor 中的 `algo_type`、`metric` 使用枚举；`custom_params`（字节数组）用于未来扩展/自定义算法；旧版 reader 不认该 section 时可安全跳过，保持兼容。

## 2. Writer 侧接口设计

1. `FileWriterOptions` 新增 `vector_indexes: Vec<VectorIndexConfig>`：定义目标列、所需算法、量化参数、优先级等。
2. 引入 `trait VectorIndexBuilder`：
   ```rust
   trait VectorIndexBuilder {
       fn build(&self, column: &dyn Array, cfg: &VectorIndexConfig) -> Result<VectorIndexArtifact>;
   }
   ```
   `VectorIndexArtifact` 包含：
   - `blob: Vec<u8>`（索引主体）
   - `quantization_meta: QuantizationSpec`
   - `descriptor_extras`（如版本、统计信息）
   - `wasm_module: Option<Vec<u8>>`（索引查询/量化逻辑的 Wasm 模块）
3. Writer 流程：列块刷完后 → 遍历 `vector_indexes` → 调用 builder 得到 artifact → 将 `blob` 写入 metadata 区域（记录 offset/size/compression），如有 `wasm_module` 亦写入 `"VectorIndexWasm"` 区域 → 构造 `VectorIndexDescriptor` 推入 Footer，并在 descriptor 中记录 `blob_section`、`wasm_section`、`quantization_meta`、`custom_params`。
4. 数据校验：把索引 blob 计入 `metadata_size` 和 data checksum（如有需要），确保文件整体校验一致。

## 3. Reader 侧接口设计

1. `VectorIndexRegistry`：解析 Footer 中的 `vector_indexes`，按列/算法建索引，提供查询接口。
2. `VectorIndexHandle` trait：
   ```rust
   trait VectorIndexHandle {
       fn query(&self, spec: QuerySpec) -> Result<Vec<ScoredRow>>;
   }
   ```
   - `QuerySpec` 指定 k 近邻、radius、需要的算法/索引 ID 等。
3. Registry 负责：
   - 根据 `descriptor.blob_section` 到 OptionalMetadata 读取索引 blob；
   - 若 `descriptor.wasm_section` 存在或 `algo_type == CUSTOM_WASM`，加载对应 Wasm 模块，将 `quant_spec/custom_params` 传入初始化；
   - 本地实现可优先使用，若缺失则回退到 Wasm；
   - 缓存索引实例，避免重复加载。
4. Reader API 示例：
   ```rust
   let idx = reader.vector_index("embedding").with_algo("HNSW")?;
   let neighbors = idx.query(QuerySpec::knn(query_vec, 10));
   ```
5. 当前实现内置了 `encode_bruteforce_index` 编码器和 `FileReaderV2::vector_knn_l2` 查询入口，可直接对 brute-force 索引执行 recall 实验；其他算法可以通过自定义 blob + Wasm 扩展。

## 4. 量化与多索引策略

1. `QuantizationSpec` 支持 `segments: Vec<QuantizationSegment>`，允许：
   - 不同维度区间使用不同的 PQ、OPQ、SQ；
   - 每段自定义 bit 宽、参数（旋转矩阵、codebook 等）。
2. 同一列可写入多个索引：`vector_indexes` 中存在多个 descriptor，`priority` 或 `usage_hint` 字段指示推荐顺序，Reader 也允许应用指定 `index_id`。
3. `custom_params`（字节数组）为特定算法提供附加配置；若 reader 不识别，可通过 Wasm decoder 解析执行。复杂的 per-dimension 权重或自定义量化逻辑都能通过 `QuantizationSegment + custom_params + wasm_module` 组合描述。

## 5. 实施顺序

1. **Schema 落地**：修改 FlatBuffer schema，更新 `fff-format` 生成的代码，补充新的枚举/常量。
2. **接口骨架**：在 `fff-poc` 中定义 `VectorIndexConfig/Builder/Artifact`、`VectorIndexRegistry/Handle` 等结构，先以 stub 实现保证编译通过。
3. **写入端打通**：实现一个基础索引（例如 brute-force + 标量量化），写入真实 blob，完成 Footer 描述；新增单元/集成测试验证读回正确。
4. **读取端打通**：解析 descriptor、加载 blob，暴露查询 API；实现简单查询（例如线性扫描 + SQ 解码）。
5. **扩展算法与量化**：引入 HNSW、IVF-PQ 等真正 ANN 算法，实现 `QuantizationSegment` 的序列化/反序列化，并验证多索引/多量化场景。
6. **工具与文档更新**：在 `fff-bench` 增加 vector workload，扩写 README 与 `doc/f3_file_structure_cn.md` 说明新 optional section、配置方法和 API，用例示范。

此路线保证：接口先行（便于多方并行开发）、文件格式兼容旧 reader、后续可以无缝叠加更多索引与量化策略。
