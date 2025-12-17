# WASM Vector Index Research Plan: SQ8 Quantization & Dimension Reordering

**Date:** 2025-12-16
**Status:** In Progress
**Focus:** Implementing Scalar Quantization (SQ8) and Dimension Reordering within the F3+WASM vector indexing framework to achieve higher compression ratios and optimize query performance, while maintaining architectural decoupling.

---

## 1. Overall Goal

The primary goal is to demonstrate that WASM can be effectively used to implement custom, data-specific compression and search strategies for vector indexes, delivering near-native performance while offering superior flexibility and portability compared to traditional native implementations. This research aims to prove that an "Executable Index Artifact" can bring significant advantages in memory-constrained or evolving environments.

---

## 2. Detailed TODO List

### Phase 1: 基础设施升级 (Support Quantization) - **进行中**

*   **1.1. 扩展 Artifact 定义**：在 `fff-vindex/src/artifact/ivf_flat.rs` 中
    *   在 `PostingCodec` 枚举中添加 `RawU8` 和 `RowIdDeltaVarintV1U8`。（**已完成**）
    *   在 `IvfFlatArtifactFooter` 结构中添加 `quantization_params: Option<serde_json::Value>` 字段，用于存储 SQ8 量化参数。（**已完成**）
    *   `write_ivf_flat_artifact_file` 函数签名已更新以接收 `quantization_params`。（**已完成**）
*   **1.2. 实现 Host-side SQ8 构建器**：在 `fff-vindex/src/artifact/ivf_flat.rs` 中
    *   添加 `quantize_vector`, `dequantize_vector`, `find_min_max` 辅助函数。（**已完成**）
    *   修改 `build_ivf_flat_artifact_with_options` 函数：
        *   在 `assign_ivf_flat` 调用之后，如果 `artifact_options.posting_codec` 是 U8 类型，则计算 `all_vectors` 的全局 `min_val` 和 `max_val`。（**进行中**）
        *   根据 `min_val`, `max_val` 计算 `scale` 和 `zero_point`。（**进行中**）
        *   将这些参数封装成 `serde_json::Value`，赋值给 `quantization_params`。（**进行中**）
        *   将 `quantization_params` 传递给 `write_ivf_flat_artifact_file`。（**进行中**）
    *   修改 `write_ivf_flat_artifact_file` 函数：
        *   在 `match artifact_options.posting_codec` 中增加 `PostingCodec::RawU8` 和 `PostingCodec::RowIdDeltaVarintV1U8` 分支。（**进行中**）
        *   在这些分支中，使用 `quantization_params` 中的 `scale` 和 `zero_point`，将 `index.vectors` 中对应的 `f32` 切片量化为 `u8` 向量。（**进行中**）
        *   将量化后的 `u8` 向量写入文件。（**进行中**）
        *   更新 `manifest.add_or_replace_index` 调用，使其包含 `quantization_params`。（**进行中**）
*   **1.3. 实现 WASM Kernel V2**：在 `wasm-libs/ivf-kernel-basic/src/lib.rs` 中
    *   修改 `ivf_flat_search_batch_ffi` 函数：
        *   从 `dir` 参数或 `Host` 导入函数获取 `quantization_params`（主要是 `scale` 和 `zero_point`）。
        *   在处理 `PostingCodec::RawU8` 和 `PostingCodec::RowIdDeltaVarintV1U8` 时：
            *   读取 `u8` 向量数据。
            *   在计算距离之前，使用 `scale` 和 `zero_point` 将 `u8` 向量动态反量化为 `f32`。
            *   使用 SIMD-accelerated L2 距离函数计算距离。
    *   添加 `l2_sq_u8` 函数（或其他名称），用于计算 `f32` 查询向量与 `u8` 向量之间的 L2 距离。
    *   可能需要修改 `PostingCodec` 的 ID 映射，以反映新的 U8 类型。

### Phase 2: 索引制导的优化 (The "Feature" - Reordering) - **待定**

*   **2.1. 维度重排策略**：在 Host 侧构建器中
    *   计算 `all_vectors` 中每个维度的方差。
    *   生成一个 `permutation` 向量，将维度按方差降序排列。
    *   将 `permutation` 存储在 `IvfFlatArtifactFooter` 的 `quantization_params` 中（或新增一个 `reordering_params` 字段）。
    *   在量化/编码 Posting List 向量时，按照 `permutation` 重新组织向量的维度顺序。
*   **2.2. 早停逻辑**：在 WASM 内核中
    *   从 footer 中读取 `permutation`。
    *   修改距离计算流程：先计算重排后的前 N 个维度，如果当前距离已经超过 Top-K 堆中的最差距离，则提前终止计算。
    *   这个 `N_fast_dims` 可以是固定值，也可以是根据索引元数据（如 `variance_threshold`）动态决定。

### Phase 3: 实验与验证 - **待定**

*   **3.1. 扩展 `bench_ivf_wasm.rs`**：
    *   添加对新的 `PostingCodec::RawU8` 和 `PostingCodec::RowIdDeltaVarintV1U8` 的命令行支持。
    *   确保正确加载和使用新的 `quantization_params`。
*   **3.2. 运行基准测试**：
    *   `Native F32`
    *   `WASM F16` (现有)
    *   `WASM SQ8` (新)
    *   `WASM SQ8 + Reorder` (新)
*   **3.3. 分析结果**：
    *   对比 Index Size。
    *   对比 Recall@K。
    *   对比 Latency (Total, Compute, Copy, Decode)。
    *   特别关注 SQ8 的内存减少和早停带来的性能提升。

---

## 3. 修改需要使用的知识和思考

*   **SQ8 量化原理**：
    *   `min/max` 量化：找到向量集中所有值的全局最小值 `min_val` 和最大值 `max_val`。
    *   `scale = 255.0 / (max_val - min_val)`
    *   `zero_point = -min_val * scale`
    *   量化（f32 -> u8）：`q = round(val * scale + zero_point)`。需要 `clamp(0, 255)`。
    *   反量化（u8 -> f32）：`val = (q - zero_point) / scale`。
    *   精度损失：SQ8 是有损压缩，需要注意对 Recall 的影响。可能需要考虑 QPS 与 Recall 的帕累托最优。
*   **WASM SIMD (`simd128`)**：
    *   `l2_sq_u8` 的实现：查询向量仍是 `f32`，索引向量是 `u8`。因此需要先将 `u8` 向量反量化为 `f32`，然后才能进行 `f32` 的 L2 距离计算。
    *   反量化本身需要用 SIMD 实现以提高效率：`v128_load_u8x16`, `v128_sub_f32`, `v128_mul_f32` 等操作。
*   **Rust `byteorder` 和 `serde_json`**：
    *   用于二进制数据的读写和 JSON 格式的元数据序列化/反序列化。
*   **Rust `slice::windows(2)` 和 `wrapping_sub`**：用于处理 Delta Varint 编码。
*   **`serde_json::Value` 结构**：方便存储异构的量化参数。
*   **维度重排策略**：
    *   方差计算：遍历所有向量，计算每个维度上的方差。
    *   排序与映射：根据方差对维度索引进行排序，生成一个 `dim_permutation: Vec<u32>` 向量。
    *   应用重排：在编码向量（F32, F16, U8）之前，根据 `dim_permutation` 重新组织向量的维度顺序。
*   **早停逻辑**：
    *   在 WASM 距离计算循环中，引入 `current_worst_distance`。
    *   计算到一定维度数量（例如 `N_fast_dims`）后，累加的 `current_dist_sq` 与 `current_worst_distance_sq` 比较，如果 `current_dist_sq > current_worst_distance_sq`，则 `return f32::MAX` 提前退出。
    *   这个 `N_fast_dims` 可以是固定值，也可以是根据索引元数据（如 `variance_threshold`）动态决定。
