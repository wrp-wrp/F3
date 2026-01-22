# LakeVector 研究计划：基于 F3 实现 "Index-as-Codec"

**状态**: 草稿 (Draft)
**目标平台**: F3 (Future-proof File Format)
**核心概念**: LakeVector - 一种面向云原生对象存储 (S3) 的静态向量存储格式，核心理念是 "Index-as-Codec"。

---

## 1. 执行摘要 (Executive Summary)

本文档概述了 **LakeVector** 的研究与实现计划。LakeVector 是一种专为高延迟对象存储 (S3) 设计的向量存储格式。我们将利用 **F3 (Future-proof File Format)** 生态系统，特别是其 **用户定义编码 (UDE)** 和 **Wasm 解码器** 能力，来实现 "Index-as-Codec" (索引即编解码) 的愿景。

核心假设是：通过将索引逻辑（静态位图、微排序）直接嵌入到压缩编解码器中（通过 F3 的 UDE 机制），我们可以在 S3 上实现高性能的向量检索，而无需外部索引文件或繁重的随机 I/O。

## 2. 与 F3 架构的对齐 (Architecture Alignment)

F3 为 LakeVector 提供了完美的基础，原因如下：
*   **fff-ude (User Defined Encoding)**: 允许我们要定义处理 Index-as-Codec 逻辑的自定义 "Smart Codecs"。
*   **Wasm Decoders**: 使复杂的解码/过滤逻辑（如神经解压、位图检查）具有可移植性，并能嵌入到文件中。
*   **fff-bench**: 现有的、强大的基准测试套件，用于与 Lance, Parquet 和 ORC 进行对比。

### 组件映射
| LakeVector 组件 | F3 模块 / 实现路径 |
| :--- | :--- |
| **网格位图 (Grid Bitmap)** | 存储在 F3 文件 Footer (Metadata) 或专用的 Metadata 列中。 |
| **Codec 注入 (精确位图)** | 在 `fff-ude` / `fff-encoding` 中实现为自定义编码。 |
| **神经压缩 (Neural Compression)** | 在 `fff-ude-wasm` 中实现为基于 Wasm 的 UDE。 |
| **物理布局 (Sort/Micro-Ordering)** | 在 `Writer` 逻辑中实现（在编码前进行排序）。 |

---

## 3. 研究路线图 (Research Roadmap)

### 第一阶段：建立基准 (Weeks 1-2)
**目标**：利用 F3 的基准测试工具，量化 "S3 随机读取问题"。

*   **任务 1.1**: 扩展 `fff-bench` 以支持 S3 延迟模拟（如果尚未存在），或针对实际 S3/MinIO 运行基准测试。
*   **任务 1.2**: 在标准向量数据集（SIFT1B 子集, GIST1M）上对 **Lance** 进行基准测试（通过 `fff-bench` 中的 `read_lance`）。
    *   指标：延迟 (Latency)、请求计数 (Request Count)、传输数据量 (Data Transferred)。
    *   场景：Top-K 向量搜索、混合搜索 (向量 + 标量)。
*   **任务 1.3**: 对 **Parquet** (scan-based) 进行基准测试，作为 "无索引" 的性能基线。

### 第二阶段："Index-as-Codec" 原型 (Weeks 3-5)
**目标**：验证 "索引" 可以作为 "编解码器" 实现。

*   **任务 2.1**: 在 Rust 中实现 `GridBitmapCodec` (原生 `fff-encoding`)。
    *   该 Codec 允许在解码完整 Chunk 之前 "Peek" (查看) 头部 (位图)。
*   **任务 2.2**: 集成到 F3 格式中。
    *   修改 `fff-poc` writer，以便在传递给 Codec 之前对数据进行排序 (分区级微排序)。
    *   将 "全局网格位图" 存储在 F3 Footer 中。
*   **任务 2.3**: 验证 "Zero-IO 剪枝" (Zero-IO Pruning)。
    *   测量在完全下载/解码 Main Body 之前跳过了多少 Chunk。

### 第三阶段：神经压缩与 Wasm (Weeks 6-8)
**目标**：实现高压缩率和检索质量。

*   **任务 3.1**: 为目标数据集训练轻量级的 AutoEncoder/Quantizer。
*   **任务 3.2**: 在 **WebAssembly** 中实现解码器。
    *   使用 `fff-ude-wasm` 封装神经解码器。
*   **任务 3.3**: 实现 "粗/细" (Coarse/Fine) 分离。
    *   Coarse Latents -> Metadata / Fast Stream.
    *   Fine Latents -> Data Stream (Wasm 解码)。

### 第四阶段：全系统评估 (Weeks 9-10)
**目标**：证明 LakeVector (F3) 在 S3 上优于 Lance。

*   **任务 4.1**: 使用 `fff-bench` 进行端到端基准测试。
*   **任务 4.2**: 消融研究 (排序 vs 位图 vs 神经压缩 的影响)。

---

## 4. 实验设计 (Experimental Design)

### 4.1 数据集
*   **SIFT1M / GIST1M**: 用于纯向量搜索基线。
*   **Synthetic Hybrid (合成混合数据)**: 增加高基数标量字段（时间戳、价格）的 SIFT1M，用于测试混合搜索。

### 4.2 指标
*   **IO 效率**: 读取字节数 / 总字节数 (Bytes Read / Total Bytes)。
*   **请求效率**: S3 GET 请求数量。
*   **延迟**: P95 查询延迟。
*   **准确率**: Recall@K。

### 4.3 成功标准 (Success Criteria)
*   **剪枝**: >80% 的数据 Chunk 通过 Metadata/Bitmap 被跳过，无需下载 Main Body。
*   **延迟**: < 内存索引 (In-Memory Index) 在 S3 上的 2 倍。
*   **存储**: 与标准 PQ/OPQ 相当。

---

## 5. 下一步行动 (Next Steps)
1.  初始化 `fff-bench` 环境并复现当前的 Lance vs F3 基准测试。
2.  在 F3 仓库中创建分支 `feature/lakevector-research`。
3.  开始 **第一阶段：建立基准**。
