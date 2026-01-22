# LakeVector: 基于 F3 "Index-as-Codec" 的云原生向量检索系统
## Workshop 投稿研究计划书

**项目代号**: LakeVector
**核心理念**: Index-as-Codec (索引即编解码), Zero-I/O Pruning (零 I/O 剪枝)
**目标平台**: Future-proof File Format (F3)

---

## 1. 项目意义 (Significance & Motivation)

### 1.1 核心问题：云存储上的向量检索 "不可能三角"
在当前的云原生架构中，向量数据库面临着严峻的挑战。对象存储（如 AWS S3）虽然提供了极低的存储成本，但其**高延迟 (High Latency)** 和 **按请求计费 (Per-Request Cost)** 的特性，使得传统的向量索引算法（如 HNSW, IVFPQ）难以直接运行。
*   **内存索引**：太贵，无法扩展到十亿/百亿级规模。
*   **磁盘索引 (DiskANN)**：依赖 SSD 的随机读性能 (IOPS)，在 S3 上会导致严重的读放大和极高的延迟。
*   **现有列存 (Parquet/Lance)**：虽然支持扫描，但在高维向量和复杂元数据过滤的混合查询下，仍然包含大量无效的数据传输。

### 1.2 我们的解法：重新定义 "索引"
LakeVector 提出并在 Workshop 中探讨的核心观点是：**在 S3 时代，索引不应该是一个独立的数据结构（如树或图），而应该内嵌于数据压缩编解码 (Codec) 之中。**

我们利用 **F3 (Future-proof File Format)** 的 **UDE (User-Defined Encoding)** 机制，将检索逻辑（剪枝、过滤、排序）下推到 Codec 层。这使得我们能够实现 **"透明索引"**，即在不改变上层查询引擎的情况下，通过底层的智能 Codec 实现 **Zero-I/O 剪枝** —— 在发起任何 S3 数据读取之前，仅凭极小的元数据就排除 90% 以上的无关数据块。

### 1.3 社区贡献
*   **新范式**: 提出 "Index-as-Codec" 设计模式，为云原生存储格式设计提供新思路。
*   **新架构**: 证明了基于 Wasm 的存储格式可以安全、高效地执行复杂的下推逻辑（如神经网络解压）。
*   **开源落地**: 基于 F3 的开源实现，为学术界和工业界提供一个可复现的 S3 向量检索基准。

---

## 2. 可行性评估 (Feasibility Assessment)

我们基于 **F3** 现有的架构能力进行了详细评估，结论是 **High Feasibility (高度可行)**。

| 关键挑战 | 解决方案 | F3 支持情况 |
| :--- | :--- | :--- |
| **如何不读数据就过滤？** | **Grid Bitmap (网格位图)**: 在 Footer 中存储全局粗粒度位图。F3 读取器优先读取 Footer。 | ✅ **Footer Metadata**: F3 支持自定义 Footer Section，可直接存入 RoaringBitmap。 |
| **如何避免 S3 随机读？** | **Partition-Level Micro-Ordering**: 在写入时对数据进行语义聚类和物理排序，确保存储连续。 | ✅ **Writer Logic**: 可以在写入 F3 `Chunk` 前进行预处理排序。 |
| **如何处理复杂过滤？** | **Wasm Smart Codec**: 将过滤逻辑编译为 Wasm，随文件分发。 | ✅ **fff-ude**: F3 核心特性，支持 Wasm 解码器和参数传递 (kwargs)。 |
| **神经压缩性能？** | **Coarse/Fine Latents**: 类似 DiskANN 思想，但用于压缩的数据流分离。 | ✅ **EncUnit**: F3 支持自定义 `EncUnit` 结构，适合存放 Latent Codes。 |

**前期验证 (Preliminary Results)**:
*   F3 现有的 Wasm 解码器开销在可接受范围内 (微秒级)。
*   Lance 等项目已经证明了在 S3 上进行向量扫描的可行性，LakeVector 在此基础上增加索引剪枝，性能只会更优。

---

## 3. 详细研究计划 (Research Plan)

本计划旨在通过 10-12 周的时间，完成从原型开发到论文/Workshop 投稿的全过程。

### **Phase 1: 问题量化与基线建立 (Weeks 1-2)**
*   **目标**: 用数据证明 "现有方案在 S3 上很慢/很贵"。
*   **任务**:
    1.  部署 **MinIO** 模拟 S3 环境（增加人工延迟）。
    2.  利用 `fff-bench` 对比 **Lance** (S3 mode) 和 **Parquet**。
    3.  测试数据集：SIFT1B (1M/10M/100M 子集) + 高基数标量字段 (Timestamp)。
    4.  输出：Baseline 延迟、S3 请求数、甚至 "美元成本/查询" 曲线图。

### **Phase 2: 核心组件实现 - "The Three Pillars" (Weeks 3-6)**
*   **目标**: 在 F3 上实现 LakeVector 的 MVP (最小可行性产品)。
*   **Pillar I: 物理排序 (Writer)**
    *   在 `fff-poc` writer 中集成 IVF 聚类和标量排序逻辑。
*   **Pillar II: 全局位图 (Layout)**
    *   在 F3 Footer 中定义并写入 `LAKEVECTOR_GRID`。
    *   实现 Reader 端的 "Zero-I/O" 检查逻辑：`if !bitmap.check(query) { return; }`。
*   **Pillar III: 透明索引 (UDE)**
    *   开发 `LakeVectorCodec` (Rust -> Wasm)。
    *   实现 `Init(kwargs)` 接口，接收查询谓词并进行 `EncUnit` 级别的精确过滤。

### **Phase 3: 神经压缩集成 (Weeks 7-9)**
*   **目标**: 进一步压缩存储体积，减少网络带宽。
*   **任务**:
    1.  训练一个简单的 PQ 或 AutoEncoder 模型。
    2.  将其推理逻辑嵌入到 Pilliar III 的 Wasm Codec 中。
    3.  实现 "粗糙码 (Coarse)" 存 Metadata，"精细码 (Fine)" 存 Data 的分离存储。

### **Phase 4: 评估与写作 (Weeks 10-12)**
*   **目标**: 撰写 Workshop 论文。
*   **实验**:
    *   **End-to-End**: LakeVector vs Lance vs DiskANN (S3 optimized)。
    *   **Ablation**: 证明 "Bitmap 剪枝" 和 "物理排序" 各自带来的提升。
*   **写作**:
    *   强调 "Index-as-Codec" 的新颖性。
    *   讨论在 F3 这种 "可扩展格式" 上做研究的便捷性。

---

## 4. 预期成果 (Expected Outcomes)

1.  **Workshop Short Paper (4-6 pages)**: 投递至 **DBTest (SIGMOD Workshop)**, **ADMS (VLDB Workshop)** 或 **CloudDB**。
2.  **Open Source Prototype**: 在 F3 仓库中合并 `LakeVector` 扩展，作为展示 F3 强大扩展性的旗舰案例。
3.  **Benchmark Suite**: 一套专门针对 S3 向量检索的基准测试工具。
