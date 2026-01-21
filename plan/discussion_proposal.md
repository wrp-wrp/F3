# Fusion-F3: Data Lake 上的 Filtered ANNS 研究提案

> **讨论用精简版** | 2026-01-20

---

## 场景：S3 / 云存储 (Data Lake)

**核心约束：**
| 特性 | 数值 | 影响 |
|-----|------|------|
| 首字节延迟 (TTFB) | 50-100ms | 图索引的随机跳转代价极高 |
| 聚合吞吐量 | 100+ Gbps | 顺序扫描有优势 |
| 存储成本 | ~$20/TB/月 | 比 RAM 便宜 80x |
| 不可变性 | Append-only | 无法维护动态索引结构 |

**典型查询：**
```sql
SELECT * FROM documents
WHERE tenant_id = 'company_A' AND created_at > '2024-01-01'
ORDER BY embedding <-> query_vector LIMIT 10;
```

---

## 观察

### 观察 1：标量压缩已有成熟的统计信息机制
- Parquet/ORC 存储 **Min/Max** 支持 Skip Scanning
- 字典编码存储 **Dictionary Page**
- 可选存储 **Bloom Filter** 支持等值查询

### 观察 2：向量格式开始内置索引
- Lance 直接在文件里嵌入 **IVF-PQ 索引**
- 存储聚类中心 (Centroids)、量化码本 (Codebooks)
- 问题：格式与算法紧耦合，难以升级

### 观察 3：F3 的独特能力
- **WASM 自解码**：压缩算法可编程，不固化在格式规范里
- **Decoupled IOUnit**：可以精确读取文件中的某几个 64KB 块
- **可扩展元数据**：可以存储任意统计信息

---

## 核心想法

### 想法 1：有损压缩 ≈ 索引
向量量化 (VQ/PQ) 本质上是：
1. **聚类** → 产生 Centroids（天然的粗粒度索引）
2. **残差编码** → 产生 Codebooks（支持压缩域计算）

> **压缩的副产品就是索引，不需要额外构建**

### 想法 2：利用 F3 的 WASM 可编程性
- 让压缩算法通过标准化 API **暴露统计信息**
- 查询引擎 **无需理解压缩细节** 就能做剪枝
- 这是一种 **"黑盒谓词下推"** 机制

### 想法 3：标量过滤 + 向量检索需要协同
- 单独的标量剪枝或向量剪枝都不够
- 需要 **Bitmap 交集** 实现联合剪枝

---

## 问题分析

### 问题 1：标量压缩 + 向量索引能否完全分开设计？

**分析：**

| 方案 | 优点 | 缺点 |
|-----|------|------|
| 完全分离（两个黑盒） | 各自独立演进 | 无法联合剪枝，物理布局冲突 |
| 外挂 HNSW（像 Lance） | 索引效果好 | 需要额外存储，S3 随机访问代价高 |

**结论**：在 S3 场景下，**完全分离效果有限**。需要至少在以下层面协同：
- 物理布局：数据如何排序存储
- 元数据：标量统计 + 向量统计需要一起用于剪枝

---

### 问题 2：如何让通用引擎利用黑盒压缩的统计信息？

**核心挑战**：
- 查询引擎 (DuckDB/Spark) 不理解 WASM Codec 内部
- 但需要利用 Codec 产生的统计信息做剪枝

**提议的解决方案 —— Standardized Interaction Protocol (SIP)：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Query Engine (Host)                       │
│  "我有 query vector Q 和 filter city='NY'"                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   SIP (标准化协议)       │
              │  • inspect()            │
              │  • prune()              │
              │  • search_with_filter() │
              └────────────┬────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    WASM Codec (Guest)                        │
│  "我有 10 个聚类中心，Cluster 3,7 距离 Q 最近"              │
└─────────────────────────────────────────────────────────────┘
```

**API 分级设计：**

| Level | 能力 | 接口 | 代价 |
|-------|-----|------|------|
| L0 | 完整解压 | `decode(bytes) -> vectors` | O(N) |
| L1 | 统计信息 | `inspect() -> Centroids/Radius` | O(1) |
| L2 | 块级剪枝 | `can_prune(query, threshold) -> bool` | O(1) |
| L3 | 稀疏解压 | `decode_sparse(bytes, bitmap) -> vectors` | O(σ·N) |
| L4 | 压缩域计算 | `search_compressed(bytes, query) -> topk` | O(N_compressed) |

**关键问题**：黑盒 Codec 愿意暴露多少信息？
- 最小暴露：Centroid + Radius（足够做三角不等式剪枝）
- 最大暴露：支持压缩域 ADC 计算

---

### 问题 3：标量压缩 + 向量索引联合设计是否真的有效？

**需要研究的 Filtered ANNS 文献：**

| 论文 | 方法 | 与我们的关系 |
|-----|------|-------------|
| ACORN (SIGMOD'24) | Predicate-aware graph navigation | 在图索引层做过滤感知 |
| NHQ (VLDB'23) | Hybrid IVF + filtered posting | 内存场景，IVF 内置过滤 |
| Milvus Filtered Search | Bitmap + HNSW | 后过滤为主 |

**联合设计的潜在收益：**

1. **物理布局统一**：Z-Order 同时优化标量范围查询和向量聚类
2. **元数据复用**：压缩产生的 Centroid 同时用于向量剪枝
3. **执行融合**：Bitmap & Cluster 交集，减少解压量

**待验证假设**：
- Z-Order 在高维向量降维后是否还有效？
- 联合布局比分离布局能提升多少剪枝率？

---

### 问题 4：前过滤 vs 后过滤 vs 融合过滤

**三种策略对比：**

| 策略 | 执行流程 | 失效场景 |
|-----|---------|---------|
| **Pre-Filter** | 先标量过滤 → 再向量搜索 | 低选择性：过滤后仍有 1 亿行，需全部算距离 |
| **Post-Filter** | 先向量 TopK → 再标量过滤 | 高选择性：TopK 全被过滤掉，召回消失 |
| **Fusion** | 两侧同时剪枝 → 交集 → 稀疏解压 | 需要额外的选择率估计开销 |

**Fusion 策略的执行流程：**

```
Step 1: 标量剪枝
  city='NY' → Bitmap_A (Rows 1, 10, 25...)

Step 2: 向量剪枝
  dist(Q, centroids) → Cluster 3,7 相关 → Bitmap_B

Step 3: 交集
  Final_Bitmap = Bitmap_A & Bitmap_B

Step 4: 稀疏解压
  只解压 Final_Bitmap 标记的行
```

**关键决策点**：如何自动选择策略？

需要一个 **Selectivity Estimator**：
- 估计标量过滤后剩多少行 (σ_scalar)
- 估计向量过滤后剩多少行 (σ_vector)
- 根据 σ_scalar × σ_vector 决定策略

**可能的启发式规则：**
- σ_scalar < 5%: Pre-Filter 优先
- σ_scalar > 50%: Post-Filter 优先
- 中间情况: Fusion

---

## 本质问题

> **在一个高延迟、不可变的存储介质（Data Lake）上，解决 Filtered ANNS 问题。**

**与传统向量数据库的关键差异：**

| 维度 | 内存向量库 (Milvus) | Data Lake (S3) |
|-----|-------------------|----------------|
| 延迟假设 | 微秒级 | 50-100ms |
| 更新模式 | 可变 | Append-only |
| 索引维护 | 实时更新 | 写入时一次性构建 |
| 主要瓶颈 | 内存带宽 | 网络 I/O + 解压 CPU |

**核心洞察**：
- HNSW 等图索引依赖大量随机访问，在 S3 上 **不可行**
- 正确的方向是 **拥抱扫描，但通过统计信息和物理布局将扫描数据量降至最低**

---

## 设计点

### 设计点 1：前/后/融合过滤如何选择？

**提议方案：Cost-Based Selection**

```rust
fn choose_strategy(σ_scalar: f32, σ_vector: f32, block_size: usize) -> Strategy {
    let pre_cost = σ_scalar * VECTOR_COMPUTE_COST;
    let post_cost = TOP_K * SCALAR_FILTER_COST + (1.0 - σ_scalar) * WASTED_COMPUTE;
    let fusion_cost = BITMAP_INTERSECT_COST + σ_scalar * σ_vector * SPARSE_DECODE_COST;

    // 选择代价最小的
    min(pre_cost, post_cost, fusion_cost)
}
```

**待解决**：
- σ_scalar 可以从 Min/Max 统计估计
- σ_vector 怎么估计？需要 Codec 提供 `estimate_selectivity(query, threshold)` 接口

---

### 设计点 2：解压代价怎么考虑？

**F3 当前的解压架构：**
```
Chunk (行组) → EncUnit (64KB) → MiniBlock (1K 行)
```

**代价因素：**

| 因素 | 影响 | 优化方向 |
|-----|------|---------|
| I/O 代价 | S3 GET 请求次数 | Read Coalescing 合并请求 |
| 解压 CPU | WASM vs Native | 重计算委托给 Host Function |
| 部分解压 | 不同编码支持度不同 | 优先选择支持 slice 的编码 |
| 压缩域计算 | PQ 支持 ADC | 跳过解压直接查表算距离 |

**代价模型公式：**

```
Cost_L0 (Full Scan)   = S_block / T_net + S_block / T_cpu_decode
Cost_L3 (Sparse Skip) = N_req(σ) × L_net + S_block × σ / T_cpu_decode
```

**Break-even Point**：通常在 σ < 20% 时，Sparse Skip 有优势。

---

### 设计点 3：外挂索引 vs 内置索引

**提议：内置 IVF + 量化，外挂作为可选项**

| 方案 | 存储方式 | 优点 | 缺点 |
|-----|---------|------|------|
| **内置 IVF-PQ** | Centroid 存在 Footer | 零额外存储，压缩即索引 | 精度有限 |
| **外挂 HNSW** | 独立的 .idx 文件 | 召回率高 | S3 随机访问代价高 |

**建议的分层设计：**

```
Layer 0: 内置 IVF (必须)
  - 聚类中心存在文件元数据
  - 支持粗粒度剪枝 (三角不等式)
  - 零额外 I/O

Layer 1: 可选外挂索引 (可选)
  - 类似 Lance 的 HNSW sidecar
  - 适合热数据或高精度场景
  - 需要额外存储和 I/O
```

---

### 设计点 4：什么时候一列数据应该被看作向量的一部分？

**场景分析：**

```
Schema: [user_id, timestamp, category, embedding[768]]
```

**问题 1**：timestamp 和 category 是否应该参与向量的聚类？

| 策略 | 做法 | 效果 |
|-----|------|------|
| 纯向量聚类 | 只用 embedding 聚类 | 向量检索快，标量过滤慢 |
| 纯标量排序 | 按 timestamp 排序 | 标量过滤快，向量检索慢 |
| **联合聚类** | Z-Order(timestamp_bits, cluster_id) | 两边都还行 |

**问题 2**：多个向量列怎么处理？

```
Schema: [id, text_embedding[768], image_embedding[512]]
```

- 方案 A：分别聚类，各自存 Centroid
- 方案 B：Concat 后联合聚类（可能损失精度）
- 方案 C：只对主查询列聚类，其他列跟随

**待研究**：工作负载分析决定哪种策略最优。

---

### 设计点 5：黑盒标量压缩算法的 API 设计

**现有 F3 Codec API：**
```rust
trait Decoder {
    fn decode(&self) -> Result<ArrayRef>;              // 完整解压
    fn slice(&mut self, start: usize, stop: usize);    // 部分解压
}
```

**提议扩展的 VectorCodec API：**

```rust
trait VectorCodec {
    // === L1: 统计信息暴露 ===
    fn get_manifest() -> CodecManifest;  // 声明支持的能力
    fn inspect_geometry(header: &[u8]) -> GeometryInfo;

    // === L2: 块级剪枝 ===
    fn can_prune(query: &[f32], threshold: f32, stats: &[u8]) -> bool;

    // === L3: 稀疏解压 ===
    fn decode_sparse(blob: &[u8], bitmap: &[u8]) -> Vec<Vector>;

    // === L4: 压缩域计算 ===
    fn search_with_filter(
        blob: &[u8],
        query: &[f32],
        bitmap: &[u8],  // 标量过滤结果
        k: u32
    ) -> Vec<(RowId, Score)>;

    // === 新增：选择率估计 ===
    fn estimate_selectivity(query: &[f32], threshold: f32) -> f32;
}

struct GeometryInfo {
    geometry_type: GeometryType,  // Sphere | HyperRectangle
    center: Vec<f32>,             // 聚类中心
    radius: f32,                  // 覆盖半径
    num_vectors: u64,             // 向量数量
}
```

**关键设计决策**：
- Codec 只需实现它能支持的 Level
- 引擎根据 `get_manifest()` 动态选择执行路径
- 最差情况退化到 L0 (Full Scan)

---

## 待讨论的开放问题

1. **Z-Order 在高维的有效性**：向量先降维到多少维再做 Z-Order 位交错？

2. **元数据存储位置**：Centroid 存在 Footer 还是每个 RowGroup Header？

3. **WASM 性能税**：压缩域计算在 WASM 里做是否可接受？还是需要 Host Function？

4. **与现有系统的兼容**：DuckDB/Spark 如何调用 SIP 协议？需要 Connector 改造？

5. **实验设计**：
   - 数据集：LAION-400M / Wikipedia / Deep1B
   - Baseline：Parquet (暴力扫描) / Lance (专用格式)
   - 指标：延迟 / 召回率 / 存储开销

---

## 一句话总结

> **在 S3 Data Lake 上，通过「压缩-索引二象性」和「黑盒谓词下推协议」，实现零额外存储开销的 Filtered ANNS。**

---

*最后更新: 2026-01-20*
