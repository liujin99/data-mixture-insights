# Survey + RegMix-D 深度分析

> 两篇论文联合分析：领域全景综述 + RegMix 动态化扩展

---

## Part A: Survey — Rethinking Data Mixture for LLMs

> arXiv:2505.21598 | CUHK-Shenzhen (Ruoyu Sun 组) | 2025.5 | 投稿 ACL ARR 2025 February

### A.1 论文定位

这是数据配比领域**第二篇综述**（第一篇是 Albalak et al. 2024 的 data selection 大综述中一小节）。本文首次将 data mixture 作为独立研究方向进行系统梳理。

**作者团队**：Ruoyu Sun（孙若愚，CUHK-SZ，优化理论方向，h-index 较高），Junchi Yang（杨骏驰，共同一作）。团队在优化理论方面有积累，对 min-max / bi-level 框架的分析有深度。

### A.2 方法分类体系

本文提出的细粒度分类是对 Albalak et al. (2024) 的 offline/online 二分法的扩展：

```
Data Mixture Methods
├── Offline Methods (静态权重 w*)
│   ├── Heuristic-based: Uniform, Proportional
│   ├── Algorithm-based
│   │   ├── Min-Max Optimization: DoReMi
│   │   └── Bi-Level Optimization: DoGE, ScaleBiO
│   └── Function Fitting-based: DML, BiMix, D-CPT, AutoScale, RegMix, MDE
│
└── Online Methods (动态权重 w^t)
    ├── Online Min-Max: MFTCoder, ShearedLLaMa, Velocitune
    ├── Online Mixing Laws: Skill-It, Aioli
    └── Others: ODM (multi-armed bandit), ADO (domain scaling laws)
```

**QuaDMix 在此分类中的位置**：Function Fitting-based Offline Method（使用 LightGBM 作为回归模型，拟合 domain weights → validation loss 的映射）。

### A.3 Function Fitting 方法对比（Table 2 核心）

| 方法 | Mixing Law 形式 | 参数 | 聚合函数 |
|------|----------------|------|---------|
| DML | c_j + b_j·exp(Σ A_ij·w_i) | c,b∈R^k, A∈R^{k×k} | Σ h_j·l_val,j(w) |
| BiMix | A_j/w_j^α · (B_j/s^β + C_j) | A,B,C,α,β∈R^k | (1/k)Σ l_val,j |
| AutoScale | (D_0 + w_j·D)^{-γ} + c_j | D_0,γ,c∈R^k | (1/k)Σ l_val,j |
| RegMix | M_j(w) [LightGBM] | 回归模型 | 单域最小化 (Pile-CC) |
| MDE | M_j(w; MDE features) | 回归模型 + data expert features | (1/k)Σ l_val,j |

**关键观察**：
- DML/BiMix/AutoScale 使用**参数化函数**（指数、幂律），参数数量 O(k²) 或 O(k)
- RegMix/MDE 使用**非参数回归**（LightGBM），更灵活但需要更多训练数据
- **QuaDMix 也属于非参数回归类**，但搜索空间更大（198 维 vs RegMix 的 17 维）
- 大多数方法用 (1/k)Σ l_val,j 作为聚合函数，但 RegMix 只最小化单域 (Pile-CC) — 这是一个已知弱点

### A.4 离线方法的三大缺陷

论文明确指出离线方法面临的三个迁移问题：

| 问题 | 描述 | 影响 QuaDMix？ |
|------|------|---------------|
| **Model Transfer** | 最优权重在不同架构/tokenizer 间不迁移 | ✅ 如果目标模型架构变化 |
| **Model Scale** | 小 proxy 的最优权重在大模型上失效 | ✅✅ 核心问题（14M → 7B+） |
| **Data Scale** | 数据规模增大时最优权重偏移 | ✅ 训练数据量变化时 |

**Function fitting 方法的潜在优势**：DML 和 AutoScale 通过 scaling law 外推来缓解 scale 问题，但只有这两个方法显式纳入 scale 因子。RegMix 和 QuaDMix **没有** scale 外推能力。

### A.5 在线方法的统一视角

论文最重要的理论贡献是将在线方法与离线框架建立联系：

**Online Min-Max 方法的统一更新规则**：
```
w_i^t = w_i^{t-t_update} · exp(V_i^t) / Σ_j w_j^{t-t_update} · exp(V_j^t)
```

区别仅在于 V_i^t 的定义：
- **DoReMi_online**: V_i^t = max{l_train,i(θ^t) - l_train,i(θ_ref), 0}（训练 excess loss）
- **ShearedLLaMa**: V_i^t = max{l_val,i(θ^t) - l_ref,i, 0}（验证 excess loss）
- **Velocitune**: V_i^t = clamp{(l_val,i - l_ref,i) / (l_init,i - l_ref,i), 0, 1}（归一化学习速度）

**Online Mixing Law 方法**（Skill-It, Aioli）：
- 维护时变矩阵 A_ij^t，表示域 i 对域 j 的影响
- 更新规则类似但使用 A_ij^t 替代 V_i^t

**对 QuaDMix 的启示**：如果将 QuaDMix 从 offline 转为 online，最简单的路径是采用 Velocitune 的归一化学习速度作为信号，在训练过程中动态调整域权重。

### A.6 三大挑战与未来方向

#### 挑战 1: 最优域权重难以找到
- 静态最优权重假设等价于固定 curriculum，但全局最优 curriculum 计算上不可行
- **建议**：采用 online 策略，在训练过程中动态优化

#### 挑战 2: 自然域定义过粗
- GLaM 8 域 vs Pile 22 域：DoReMi 在 Pile 上效果更好
- 细粒度域发现方法：聚类（embedding + k-means）或 **loss trajectory 作为样本特征**（Chen et al. 2023 发现 loss trajectory 显著优于其他特征）
- **对 QuaDMix 的启示**：22 个 Dewey Decimal 域已经是较细粒度的划分，但可能仍不够细。loss trajectory 作为聚类特征是一个值得探索的方向

#### 挑战 3: Loss 指标 vs 任务表现
- **最关键的发现**：大多数方法找到的最优权重**只比 uniform sampling 略好**（Jiang et al. 2024），甚至**不如 uniform sampling**（Chen et al. 2024）
- 这直接质疑了整个 data mixture 领域的价值主张
- **对 QuaDMix 的启示**：这与用户观察到的 "QuaDMix < quality-top-k" 高度一致。复杂优化可能确实在 loss 空间有效，但在任务表现空间无效

### A.7 可信度评估

| 维度 | 评分 | 理由 |
|------|------|------|
| 作者团队 | ★★★★☆ | Ruoyu Sun 在优化理论方面有积累 |
| Venue | ★★★☆☆ | 投稿 ACL ARR 2025 Feb，尚未被接收 |
| 分类体系 | ★★★★★ | 比 Albalak et al. 2024 更细粒度，建立了 online-offline 联系 |
| 实验验证 | ★★☆☆☆ | 纯综述，无新实验 |
| 对 QuaDMix 价值 | ★★★★☆ | 提供了清晰的方法定位和挑战分析 |

### A.8 QuaDMix 未被提及

**整篇综述没有提到 QuaDMix 或 CLIMB/CliMBMix**。这意味着：
1. QuaDMix 在学术界的知名度仍然很低
2. 综述覆盖的方法截至 2025 年初，QuaDMix 可能发表时间较晚或影响力尚未扩散
3. 这也说明 QuaDMix 尚未被社区作为 baseline 或参考方法

---

## Part B: RegMix-D — Dynamic Data Mixing via Proxy Training Trajectories

> arXiv:2606.18663 | U Tokyo + NII (Yoshimasa Tsuruoka 组) | 2026.6 | Work in Progress

### B.1 核心思想

**一句话**：RegMix 只用 proxy 训练的终点 loss 拟合回归模型，RegMix-D 用**完整 loss 轨迹**拟合，从而实现动态配比。

**关键观察**：每次 proxy 训练产生一条完整的 loss 曲线，不只是一个终点值。这些中间点包含了"在训练不同阶段，哪种配比最优"的信息。

### B.2 方法详解

#### Trajectory-Conditioned Regression

给定 N 个切换点，将 proxy 训练分为 N+1 个区间 {t_1, ..., t_{N+1}}。

对每个 proxy i 和每对相邻区间 (t_l, t_{l+1})，构建一个训练样本：

```
输入: (t_l, m^(i), l_l^(i))  →  输出: l_{l+1}^(i)
```

即：给定当前步数 t、当前配比 m、当前 loss l，预测下一个区间的 loss。

**与 RegMix 的区别**：
- RegMix: f(m) → l_endpoint（全局映射，一个配比一个终点）
- RegMix-D: f_N(t, m, l) → l_next（局部转移，条件于当前状态）

#### Offline 模式（Algorithm 1）

1. 从初始 loss l_0 开始
2. 在每个切换点 s_j，Dirichlet 采样 100K 候选配比
3. 选择使 f_N(s_j, m, l_{j-1}) 最小的 top-128 平均作为 m_j*
4. 更新 l_j = f_N(s_j, m_j*, l_{j-1})（递归使用预测值）
5. 映射到目标训练：s_j^target = s_j · (T_target / T_p)

#### Online 模式（Algorithm 2）

与 Offline 相同，但在每个切换点使用**目标模型的实际观测 loss** 替代递归预测值：

```
l̃_{S_j} = l̂_{S_j} · (P_t / P_p)^β
```

其中 β=0.05 是跨尺度修正因子，P_t/P_p = 1000（1B / 1M）。

### B.3 实验结果

| 方法 | Proxies | 模式 | 13-task Avg |
|------|---------|------|-------------|
| Human | — | — | 44.80 |
| DoReMi | — | — | 47.10 |
| RegMix | 512 | Static | 47.41 |
| RegMix-D | 128 | Offline | 47.93 |
| RegMix-D | 128 | Online | 48.15 |
| RegMix-D | 512 | Offline | 48.09 |
| RegMix-D | **512** | **Online** | **48.44** |

**关键发现**：
1. **128 proxies 的 RegMix-D > 512 proxies 的 RegMix**：用 25% 的 proxy 计算量就超越了原版
2. **Online > Offline**：使用目标模型实际 loss 比递归预测更可靠
3. **改进广泛分布**：11/13 个任务达到 top-1 或 top-2，不是集中在某类任务
4. **增加 proxy 数量收益递减**：128→512 只带来 ~0.2% 提升

### B.4 消融实验

| 参数 | 最优值 | 敏感度 |
|------|--------|--------|
| 切换点 N | 5 | 中等（3,7,9 也优于 RegMix） |
| 跨尺度修正 β | 0.05 | 低（最大变化 0.25） |
| Proxy 大小 | 1M = 60M = 120M | **几乎无差异**（< 0.05） |

**Proxy 大小消融**是最重要的发现：1M、60M、120M proxy 给出几乎相同的结果。这与 RegMix 原论文的发现一致，进一步验证了**小 proxy 在 Pile 数据集上是足够的**。

### B.5 开销分析

| 组件 | 开销 |
|------|------|
| Proxy 训练 | 1M proxy ≈ 0.5 hr/个 (H800)，128 个 = 64 GPU-hours |
| Offline 模式 | 目标训练 +0%（schedule 预生成） |
| Online 模式 | 目标训练 **+0.37%**（仅切换点查询回归模型） |

### B.6 动态配比可视化

Pile-CC 权重轨迹（RegMix 静态 = 0.87）：
- Offline: segment 2 降至 0.555（早期向非主导域重新分配），后期回升
- Online: 类似趋势但波动更小（因为基于实际观测而非递归预测）

**直觉**：训练早期应该更多探索非主导域，后期收敛回主导域。

### B.7 可信度评估

| 维度 | 评分 | 理由 |
|------|------|------|
| 作者团队 | ★★★☆☆ | U Tokyo (Tsuruoka 组)，NLP 领域知名，但非 LLM 大厂 |
| Venue | ★★☆☆☆ | arXiv only，"Work in Progress"，尚未投稿 |
| 实验规模 | ★★★☆☆ | 1B target / 25B tokens，中等规模 |
| 方法简洁性 | ★★★★★ | 极其简洁的扩展，无额外机制 |
| 可复现性 | ★★★★☆ | 基于 RegMix 开源框架，代码应可获取 |
| 对 QuaDMix 价值 | ★★★★★ | 直接可应用于 QuaDMix 的 LightGBM surrogate |

### B.8 局限性

1. **单域验证 loss**：只用 Pile-CC 作为优化目标（继承 RegMix 的弱点）
2. **未与 Aioli/TiKMiX 对比**：因为不同语料/域划分/规模/benchmark，难以公平比较
3. **1B 验证**：未在更大模型上验证
4. **线性时间映射**：proxy 切换点通过线性映射到目标训练，假设 loss 曲线结构相似

---

## Part C: 联合分析与 QuaDMix 启示

### C.1 Survey 框架下的 RegMix-D 定位

在 Survey 的分类体系中：
- **RegMix** = Offline / Function Fitting-based（静态权重）
- **RegMix-D Offline** = 仍然属于 Offline（预生成 schedule），但引入了**时间维度**
- **RegMix-D Online** = 跨越了 Offline/Online 边界：回归模型在 proxy 阶段离线训练，但在目标训练阶段在线部署

Survey 的分类框架**没有覆盖这种混合模式**。RegMix-D 实际上是一种 "Offline-trained Online" 方法，这可能是 Survey 未来版本需要增加的新类别。

### C.2 对 QuaDMix 的直接适用性

#### 最高优先级：将 RegMix-D 的轨迹回归应用于 QuaDMix

QuaDMix 当前使用 LightGBM 拟合 `domain_weights → validation_loss`（终点 loss）。RegMix-D 的改进路径：

1. **在 proxy 训练中记录 loss 轨迹**：不只是终点 loss，每隔 N 步记录一次
2. **训练 trajectory-conditioned LightGBM**：输入 (step, weights, current_loss) → 输出 next_loss
3. **部署 Offline 模式**：预生成 N 个切换点的配比 schedule
4. **部署 Online 模式**（可选）：在目标训练中用实际 loss 查询回归模型

**预期收益**：
- 基于 RegMix-D 的结果，+0.5-1.0% 平均提升（47.41 → 48.44 = +1.03）
- 用更少的 proxy 达到相同效果（128 vs 512 = 75% 节省）
- Online 模式额外开销 < 0.5%

#### 中等优先级：跨尺度 loss 修正

RegMix-D 的跨尺度修正公式：
```
l̃ = l_target · (P_t / P_p)^β, β = 0.05
```

QuaDMix 的 proxy (14M) 到 target (7B+) 的尺度比为 500x。如果采用类似修正：
- β = 0.05 时，修正因子 = 500^0.05 ≈ 1.36
- 这意味着 target loss 需要乘以 1.36 才能与 proxy loss 对齐
- 但 QuaDMix 不直接做 loss 对齐（LightGBM 直接学习映射），所以这个修正可能不直接适用

#### 低优先级：Survey 的域粒度建议

Survey 建议用 loss trajectory 作为样本特征进行聚类，发现更细粒度的域。QuaDMix 的 22 个 Dewey Decimal 域已经较细，但如果性能仍不理想，可以考虑：
- 在 proxy 训练中记录每个样本的 loss trajectory
- 用 trajectory 做聚类，发现"模型视角"的自然域
- 这与 DoGraph (#33) 的"人类域 ≠ 模型域"发现一致

### C.3 Survey 挑战 3 的深层含义

Survey 指出"最优权重只比 uniform 略好甚至更差"，这与用户观察到的 "QuaDMix < quality-top-k" 高度一致。可能的解释：

1. **Loss 空间 vs 任务空间的不一致**：在 proxy loss 空间找到的最优解，在下游任务空间不是最优
2. **搜索空间过大**：198 维搜索空间中，LightGBM 的预测精度随维度指数下降
3. **Proxy 信号噪声**：14M proxy 的 loss 信号包含大量噪声，复杂优化放大了噪声

**结论**：Survey 和 RegMix-D 共同指向一个方向——**减少优化复杂度，增加信号质量**。RegMix-D 通过利用轨迹信息增加信号质量（不增加搜索维度），是一个更安全的改进方向。

### C.4 行动建议优先级

| 优先级 | 行动 | 预期收益 | 成本 |
|--------|------|---------|------|
| P0 | 在 proxy 训练中记录 loss 轨迹，训练 trajectory-conditioned LightGBM | +0.5-1% | 低（修改 logging + 重训 LightGBM） |
| P1 | 减少 proxy 数量（从当前数量减至 128-256） | 节省 50-75% proxy 计算 | 无 |
| P2 | 实现 Offline schedule 模式（N=5 切换点） | 动态配比 | 低 |
| P3 | 实现 Online 模式（需要跨尺度 loss 修正） | 额外 +0.2-0.3% | 中（需要目标训练中间 loss 反馈） |
| P4 | 将验证目标从单域 (Pile-CC 等价) 改为多域聚合 | 参考 MDE 发现 | 中 |

---

## Part D: 两篇论文的关键数据点

### D.1 RegMix-D 实验设置

| 参数 | 值 |
|------|-----|
| Target 模型 | 1B TinyLlama |
| Proxy 模型 | 1M TinyLlama variant |
| 训练数据 | Pile (17 域) |
| 训练 tokens | 25B |
| Proxy 训练步数 | 1,000 steps (1M tokens/step) |
| 优化器 | AdamW, lr=4e-4, weight_decay=0.1 |
| Context length | 2048 |
| Global batch size | 512 |
| 回归模型 | LightGBM |
| 验证目标 | Pile-CC validation loss |
| 候选采样 | Dirichlet 100K, top-128 平均 |
| 切换点 | N=5 |
| 跨尺度修正 | β=0.05 |
| GPU | 1 H800 (proxy), 8 GPUs (target) |

### D.2 Survey 方法覆盖范围

| 类别 | 方法数 | 代表方法 |
|------|--------|---------|
| Offline / Heuristic | 2 | Uniform, Proportional |
| Offline / Min-Max | 1 | DoReMi |
| Offline / Bi-Level | 2 | DoGE, ScaleBiO |
| Offline / Function Fitting | 6 | DML, BiMix, D-CPT, AutoScale, RegMix, MDE |
| Online / Min-Max | 4 | MFTCoder, DoReMi_online, ShearedLLaMa, Velocitune |
| Online / Mixing Law | 2 | Skill-It, Aioli |
| Online / Others | 2 | ODM, ADO |
| **总计** | **19** | — |
