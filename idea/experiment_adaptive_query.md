# 实验计划：自适应查询处理 (Adaptive Query Processing) 验证

**核心假设 (Hypothesis)**: 
对于同一个 F3 物理文件（相同的索引/排序布局），在不同的查询选择率 (Selectivity) 下，存在不同的最优执行算法。
具体来说，**位图跳跃 (Bitmap Skip)** 在低选择率下最优，而 **SIMD 扫描 (Vectorized Scan)** 在高选择率下最优。LakeVector 的 Wasm 机制应当能自动识别并切换，从而逼近理论最优边界。

---

## 1. 实验设置 (Setup)

### 1.1 数据生成
*   **总量**: 1000 万行 (10M Rows) 向量数据 (128-dim Float32)。
*   **物理布局**: 
    *   按照 `ClusterID` 分区。
    *   分区内按照 `Timestamp` 排序。
    *   每 64K 行一个 `Chunk` (IOUnit)。
*   **标量列**: 
    *   `Tag`: 低基数整数 (0-100)，模拟标签点查。
    *   `Score`: 浮点数，模拟范围查询。

### 1.2 待测算法 (Kernels)

我们将使用 Rust 实现并通过 `fff-ude-wasm` 编译三种 Wasm Kernel：

#### Kernel A: 暴力 SIMD (Always Scan)
*   **逻辑**: 无视任何过滤条件，直接加载所有 Chunk 的 Latent Codes。使用 SIMD (AVX2/NEON) 对全部数据进行解压和距离计算。
*   **场景**: 假设它是 "Baseline"，代表现有 Parquet/Lance 在不做特殊优化时的行为。

#### Kernel B: 位图跳跃 (Bitmap Skip)
*   **逻辑**: 
    *   读取 Footer 的 `RoaringBitmap`。
    *   计算出需要保留的 RowID 集合。
    *   在解码 `EncUnit` 时，使用 `Gather` 指令或条件分支，只解压位图中存在的行。
*   **场景**: 针对稀疏数据访问优化。

#### Kernel C: 自适应融合 (LakeVector Adaptive)
*   **逻辑**: 
    *   在 `Init()` 阶段，根据输入的 Query Predicate 预估选择率 (Selectivity Estimation)。
    *   `if selectivity < THRESHOLD`: 调用 Kernel B 的逻辑。
    *   `else`: 调用 Kernel A 的逻辑。
*   **关键点**: 确定这个 `THRESHOLD` (拐点) 的位置。

---

## 2. 实验步骤 (Procedure)

### 2.1 寻找拐点 (Finding the Cross-Over)
**目标**: 测出 Kernel A 和 Kernel B 性能曲线的交点。

1.  **控制变量**:
    *   查询类型: `Tag == X` (点查)。
    *   通过调整数据分布，使得满足 `Tag == X` 的行数比例从 **0.01%** 逐渐增加到 **50%**。
2.  **测量指标**:
    *   **Latency**: 端到端查询时间。
    *   **CPU Cycles**: 纯计算开销。
    *   **Branch Misses**: 分支预测失败率 (预期 Kernel B 在高选择率下会暴涨)。
3.  **预期结果**:
    *   在 Selectivity < 1% 时，Kernel B 极快 (e.g., 10ms vs A 的 500ms)。
    *   在 Selectivity > 5-10% 时，Kernel B 性能急剧下降，甚至慢于 Kernel A (因为分支预测失败和随机内存访问)。
    *   交点即为 **Optimal Threshold**。

### 2.2 验证自适应性 (Verifying Adaptivity)
**目标**: 验证 Kernel C 是否能自动贴合最优曲线。

1.  将测量出的 Threshold 硬编码进 Kernel C (或实现简单的 Cost Model)。
2.  在随机生成的混合负载下运行 Kernel C。
3.  **主要指标**: 
    *   **Regret Ratio**: `(Time_C - Min(Time_A, Time_B)) / Min(Time_A, Time_B)`。
    *   目标是 Regret Ratio < 5% (即几乎总是做出了正确的选择)。

---

## 3. 结果展示 (Visualization)

我们将绘制一张核心图表用于论文/Proposal：

*   **X轴**: Selectivity (Log Scale: 0.01%, 0.1%, 1%, 10%, 100%)
*   **Y轴**: Latency (ms)
*   **曲线**:
    *   🔴 **SIMD Scan (Baseline)**: 平坦的直线 (性能与选择率无关，总是全扫)。
    *   **🔵 **Bitmap Skip**: 斜率为正的曲线 (选择率越高越慢)。
    *   🟢 **LakeVector (Adaptive)**: 实际上是 `Min(Red, Blue)` 的下包络线。

---

## 4. 进阶：反馈机制与状态管理 (Feedback & State)

用户指出的核心挑战：**在 Data Lake (S3) 场景下，计算节点（Reader）往往是无状态且短暂的（Serverless/Ephemeral）。** 内存中的统计信息会在进程销毁时丢失，因此对于"低频查询"，内存缓存无效。

我们修正后的双模态状态管理方案：

### 4.1 方案 A：基于日志的进化 (Log-Structured Evolution)
这是最符合您设想的架构，类似于 LSM-Tree 的 WAL 思想。

1.  **微量追加 (Micro-Append / Query Log)**
    *   **动作**: 每次查询结束后，Reader 并不直接修改模型，而是 Append 一条微小的 **"Query Log"** 到文件末尾（或 Sidecar 文件 `query_log.bin`）。
    *   **内容**: `Struct { Selectivity: f32, Latency: i32, AlgoUsed: u8 }` (约 16 字节)。
    *   **代价**: 极低。如果是 S3，可以使用 Batch Append 或 Write-behind 策略。

2.  **定期合并 (Periodic Compaction / Model Update)**
    *   **触发**: 当 Append 的 Log 数量达到阈值 (e.g., N=1000) 或周期性触发。
    *   **动作**: 
        *   读取所有未处理的 Logs。
        *   **Training**: 在 Wasm 内运行轻量级回归算法 (Linear Regression)，更新 `THRESHOLD` 或 Cost Model 系数。
        *   **Compaction**: 将新的 Wasm 参数写入 Footer，并**标记旧的 Logs 为已过期** (Garbage Collection)。
    *   **收益**: 这种机制实现了 **"Online Learning"**，查询越多，Cost Model 越准，最终收敛到物理环境的最优解。

### 4.2 方案 B：影子文件 (The "Sidecar" Way)
*   **适用场景**: 原始数据只读 (Immutable) 或并发写冲突严重。
*   **机制**: 
    1.  对于文件 `data.f3`，Reader 尝试读取 `data.f3.stats` (Shadow File)。
    2.  如果存在，Wasm 加载该文件中的统计信息覆盖默认参数。
    3.  定期通过后台任务 (Compaction Job) 将 Shadow File 的信息 Merge 回主文件。

### 4.3 核心算法：轻量级多维在线学习 (Lightweight Multi-dim Online Learning)
这也是用户关注的痛点：*“光看 Selectivity 不够，K 的大小也很重要，嵌入 AI 又太重。”*
我们的解法是：**Online SGD on Polynomial Features (基于多项式特征的在线梯度下降)**。

1.  **特征工程 (Features)**:
    我们不只看 $s$ (Selectivity)，而是构建一个特征向量 $\mathbf{x}$:
    $$ \mathbf{x} = [1, s, k, s \cdot k, s^2, \log(k)] $$
    *   $s$: 标量过滤比例。
    *   $k$: Top-K 的 K 值。
    *   $s \cdot k$: **交互项 (Interaction Term)**，捕捉 "既要查很多点，又要拿很多 K" 的非线性成本。

2.  **模型 (Model)**:
    预测耗时 $\hat{y} = \mathbf{w}^T \mathbf{x}$。
    这是一个纯向量点积操作，在 Wasm 中仅需几十个时钟周期，**极度轻量**，完全不是"复杂的 AI"。

3.  **在线学习 (Online Learning)**:
    每次查询结束后，根据真实耗时 $y_{real}$，使用 **SGD** 更新权重 $\mathbf{w}$:
    $$ \mathbf{w} \leftarrow \mathbf{w} - \eta \cdot (y_{real} - \hat{y}) \cdot \mathbf{x} $$
    
    *   **无需 heavy runtime**: 没有 PyTorch/TensorFlow，只是十几行 Rust 代码的手写矩阵乘法。
    *   **捕捉复杂关系**: 它可以自动学到 "当 K 很大时，即使 s 很小，Bitmap Skip 也可能变慢" 这种细微的 Trade-off。

4.  **决策 (Decision)**:
    *   Cost(AlgoA) = $w_A \cdot x$
    *   Cost(AlgoB) = $w_B \cdot x$
    *   选择 Cost 较小的那个。

**结论**: 我们不需要一个全功能的神经网络。一个 6-10 维参数的线性模型配合交互特征，足以拟合绝大多数数据库算子的 Cost Curve。这正是 F3 Wasm "Micro-Intelligence" 的最佳体现。
    
**结论**: 在 Data Lake 场景下，**持久化到 S3 (Persistence on S3)** 是反馈机制生效的必要条件。内存仅用于单个 Batch 或高频交互式 Session 的临时加速。

---

## 5. 意义 (Significance)

该实验将强有力地证明：
1.  **静态索引是不够的**：没有任何一种单一的静态索引结构能同时处理好稀疏和稠密查询。
2.  **代码即数据 (Code-is-Data) 的价值**：只有通过 Wasm 将“选择算法的逻辑”下推到存储层，才能在不修改上层引擎代码的情况下，实现这种细粒度的性能优化。

### 5.1 为什么一定要 "演进" (Why Evolution Matters)?
用户可能会问：*“为什么不能由 Writer 写死一个阈值 (e.g. 1%)？”*
演进式查询的价值在于应对 **环境异构性 (Heterogeneity)**：

*   **硬件差异**: 
    *   在 AWS EC2 上，S3 带宽很高，Scan 更快 -> 阈值可能是 5%。
    *   在家庭宽带上，下载很慢，Bitmap Skip 更快 -> 阈值可能是 10%。
    *   静态写死的参数无法同时适应这两种环境。通过 "Log & Learn"，文件能在不同的运行环境下自动收敛到该环境的最优解。
*   **数据分布漂移**:
    *   写入时数据是均匀分布的。
    *   但在查询侧，用户可能只盯着某个热点 Range 查。全局最优不等于局部最优。反馈机制能捕捉到这种 **Workload Skew**。
