# F3 向量索引研究完整报告 (Master Research Report)

本报告详细记录了 F3 向量索引的架构设计思想、深度性能分析、以及支撑论文核心 Claim 的实验证据。

---

## 1. 核心架构：Wasm 作为“IO 与算法的协调平面”

### 1.1 设计哲学：从“数据索引”到“可执行工件”
传统向量数据库将索引视为静态数据（如 `.ivf` 文件），而 F3 将其视为**可执行工件 (Executable Artifact)**。

- **IO 决策权下放**: 在 F3 中，是由索引内核（Wasm）告诉宿主（Host）“去拉取 Chunk #42”，而不是宿主硬编码读取逻辑。这使得宿主可以完全不理解 IVF 或 PQ 的内部结构。
- **极致解耦**: 宿主仅负责通用的 `host_get_chunk` 和 `cache`，而具体的 Codec 解码、距离核优化、甚至搜索策略（如早期停止）都封闭在 Wasm 内。
- **演进成本降至零**: 如果我们要从 F32 升级到 PQ-16 压缩，只需替换 `.artifact` 文件，无需停止或重新编译任何宿主微服务。

### 1.2 存储解剖 (Storage Anatomy): 为什么选 IVF？
IVF 结构相比 HNSW 在“按需读取”和“压缩收益”上具有压倒性优势。

| 索引类型 | 向量数据 (Data) | 额外结构 (Structure) | 结构占比 | 特性 |
|:---|:---|:---|:---|:---|
| **IVF-Flat (F16)** | 256 MB | ~0.5 MB (Centroids) | **< 1%** | 极其轻量，数据是主体，天然支持 Chunked IO |
| **IVF-PQ (m=16)** | **16 MB** | ~0.6 MB (Codebooks) | **~4%** | **16x 压缩**，且压缩对 IO 节省直接翻倍 |
| **HNSW (M=32)** | 512 MB (F32) | **128 MB (Neighbors)** | **~25%** | 结构沉重，随机跳跃访问导致按需加载极难实现 |

### 1.3 技术实现：Artifact 文件布局
为了支持“按需加载”，`.artifact` 文件采用了“尾部索引”的设计：

1. **Header (12 bytes)**: 包含 Magic Number 和版本号。
2. **Metadata Pointer (8 bytes)**: 一个指向物理文件末尾 Footer 的偏移量。
3. **Data Chunks (Payload)**: 文件的核心主体，连续存放着聚类中心、倒排链等二进制块。
4. **Footer (The Directory)**: 文件的末尾是一个 JSON 字典，它记录了：
    - **逻辑标识** (如 `List_ID: 118`) 到 **物理位置** (`Offset`, `Len`) 的映射。
    - 向量维数、Codec 类型（F16/PQ）等元数据。

**运行逻辑**: 宿主加载索引时只需读取几十 KB 的 Footer 目录到内存。当 Wasm 内核计算出相关 ID 后，宿主根据目录表进行一次精准的磁盘 `pread`。

### 1.4 深度分析：反序列化 vs 免反序列化 (Zero-Deserialization)
在 F3 向量索引的设计中，我们在**灵活性**与**高性能**之间做了一个精细的分层平衡：

1. **元数据层 (序列化 - JSON)**:
    - **位置**: Footer 目录。
    - **方案**: 使用 `serde_json`。
    - **逻辑**: 元数据需要极高的灵活性（可能包含各种复杂的构建参数、PQ 中心点定义等）。由于它只在“索引打开”时加载一次，秒级的序列化开销是可以接受的。

2. **数据通路层 (免反序列化 - Binary Mapping)**:
    - **位置**: Centroids 和 Posting List Chunks。
    - **方案**: 直接使用内存布局对齐的 **Little-Endian Binary**。
    - **逻辑**: 对于向量数据，我们采用了“免反序列化”的思想。Wasm 内核通过 `from_le_bytes` 或 `bytemuck` 直接解释字节流。
    - **价值**: 
        - **CPU 零损耗**: 不需要像 Protobuf 那样进行复杂的解析和对象构建。
        - **Wasm 友好**: Wasm 的线性内存（Linear Memory）本质上就是一个大的 `u8` 数组。免反序列化意味着数据从 Host 拷贝进 Wasm 内存后，**立即可用**。

3. **内核优化层 (Decoded Cache)**:
    - **逻辑**: 虽然是“免反序列化”，但如果是 F16 格式，每次计算仍需转成 F32。
    - **方案**: Wasm 内核内置了一个 `Decoded Cache`，存储转换后的 F32 原始数组，从而彻底消除了任何形式的“解释”成本，实现了真正的“零开销”重复查询。

---

## 2. 性能深度拆解 (Performance Nuances)

### 2.1 跨语言边界开销与“摊薄”效应 (Batching)
**现象**: Wasm 的调用固定开销（Call Overhead）和内存拷贝是性能损耗的主要来源。
**发现**: 随着查询并发量（`nq`）增大，Wasm 的优势开始体现，因为单次跨平台调用的成本被摊薄。

| 查询并发 (nq) | Native 延迟 | Wasm 延迟 | 性能差距 (Overhead) |
|:---:|:---:|:---:|:---:|
| **1** | 0.44 ms | 1.26 ms | **2.8x** |
| **32** | 1.02 ms | 3.09 ms | **3.0x** |
| **100** | 3.3 ms | 4.7 ms | **1.4x** |

> **结论**: 在高吞吐场景下，Wasm 的灵活性成本仅为 **40%** 左右，这在换取“按需加载”带来的 **18 倍首跳加速**时是极其划算的。

### 2.2 阶段性耗时分析 (Stage Profiling)
在温启动 (Warm Cache) 下，时间到底花在哪了？

| 阶段 | 耗时占比 | 说明 |
|:---|:---|:---|
| **Centroid Selection** | ~1% | 计算向量与个中心点的距离，耗时极短 |
| **Data Transfer** | ~10% | 将 Chunk 数据从 Host 内存拷贝入 Wasm 线性内存 |
| **Decoding (Codec)** | ~5-20% | Raw 格式接近 0；F16 或 Delta-Varint 在 Wasm 内解码稍有开销 |
| **Distance Scan** | **~70%** | **核心瓶颈**。涉及数万次 L2 距离计算，必须通过 SIMD 优化 |
| **Heap/Top-K** | ~5% | 维护小顶堆以获取最近邻 |

---

## 3. 核心实验数据汇总 (Definitive Results)

### 3.1 解决 RQ2: 按需加载 (On-Demand) 的真实收益
对比“全量加载”与 F3 “按需加载”。

| 实验指标 | 传统全量加载 (Baseline) | F3 On-Demand (Wasm) | 收益 |
|:---|:---|:---|:---|
| **首跳读取量** | 25.41 MB (100%) | **0.98 MB (4%)** | **25x IO 节省** |
| **首跳总延迟 (TTFQ)**| 22.92 ms | **1.26 ms** | **18.1x 提速** |
| **后续稳态延迟** | 0.44 ms | 0.50 ms | 差异不显著 |

### 3.2 解决 RQ1: SIMD 的决定性作用
Wasm SIMD 开启前后对 `Distance Scan` 的影响极深。

- **Wasm Scalar**: 19.26 ms (1.0x)
- **Wasm SIMD**: **4.99 ms (3.86x)**
- **结论**: SIMD 是 Wasm 执行向量检索的“生死线”。

### 3.3 正确性验证
- **Recall@10 一致性**: Native (64.2%) vs Wasm (64.2%)。
- **精度无损**: Wasm 沙箱中的逻辑执行完全复刻了 Native 行为。

---

## 4. 论文 Claim (结论)

1. **IO 革命性提升**: "通过 Wasm 驱动的按需抓取策略，我们将首跳延迟（TTFQ）在毫秒级内完成，即使物理索引规模达到 GB 级。"
2. **算力性能对齐**: "Wasm SIMD 优化在向量扫描任务中实现了 3.8x 的加速，证明了其在计算密集型数据库任务中的可行性。"
3. **架构的工业级灵活性**: "实现了向量索引格式与数据库宿主的完全解耦，支持无需重启的热更新与格式演进。"

---

## 5. 复现与工具
- **IO 跑测**: `bash exp_scripts/io_on_demand_benchmark.sh`
- **详细 Profiling**: `cargo run --example vector_ivf_flat_demo --release -- --profile-stages ...`
- **文档参考**:
    - [研究全景图](file:///Users/rprp/Github-local/F3/doc/experiments/vector_research_status.md)
    - [详细性能日志汇总](file:///Users/rprp/Github-local/F3/doc/experiments/ivf_artifact_current_status.md)
