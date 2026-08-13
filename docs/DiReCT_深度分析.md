# DiReCT 深度分析

**论文**: Towards Efficient LLMs Annealing with Principled Sample Selection
**arXiv**: 2605.31175 (2026.05.29)
**作者**: Yuanjian Xu (HKUST-GZ / MSRA 实习生), Jianing Hao (HKUST-GZ), Wanbo Zhang (Fudan), Zhong Li (MSRA), Guang Zhang (HKUST-GZ)
**Venue**: ICML 2026
**机构**: HKUST(GZ) + Microsoft Research Asia
**代码**: https://github.com/xuyj233/Direct

---

## 一、核心论点

### 1.1 问题定位

Annealing（冷却）阶段是预训练最后的关键阶段，决定最终模型质量。现有策略依赖经验启发式（域过滤、上下文扩展），缺乏优化理论基础。

### 1.2 核心洞察：损失景观的谱几何

- 稳定训练末期：梯度在**高曲率（stiff）方向**振荡，有效下降信号弱
- Annealing 需要双目标：① 在低曲率（flat）方向提供强下降信号 ② 抑制高曲率方向的噪声
- **假设**：某些训练 schedule 的经验成功源于隐式满足了这些谱约束

### 1.3 方法：DiReCT (Directionally-Restrained Constrained Training)

将样本选择形式化为**约束优化问题**：
- 选样本使其梯度在 flat 子空间投影大（驱动沿谷底下降）
- 同时在 stiff 子空间投影小（避免不稳定）

---

## 二、方法详解

### 2.1 Validation Hessian 矩阵

在验证集上计算 Hessian 矩阵 H_val 的特征分解：

```
H_val = V Λ V^T,  Λ = diag(λ_1, ..., λ_d)
```

- 高特征值方向 = stiff（高曲率）→ 不稳定
- 低特征值方向 = flat（低曲率）→ 泛化

### 2.2 谱子空间划分

用曲率阈值 ε 划分：
- **Stiff 子空间 S_⊥**：{j | λ_j > ε} — 高曲率方向，需抑制
- **Flat 子空间 S_∥**：{j | λ_j ≤ ε} — 低曲率方向，需增强

### 2.3 样本梯度投影

每个训练样本 x_i 的梯度在特征向量 v_j 上的投影：

```
g_{i,j} = ⟨∇ℓ_pre(x_i; θ_s), v_j⟩
```

### 2.4 优化目标

```
max_w  ‖Σ_i w_i · g_{i, I_∥}‖²     (最大化 flat 子空间梯度能量)
s.t.    Σ_j λ_j Σ_i w_i g_{i,j}² ≤ τ_⊥  (stiff 子空间能量约束)
        Σ_i w_i = K                        (选择 K 个样本)
```

### 2.5 随机投影近似

直接计算 H_val ∈ ℝ^{d×d}（d~10^9）不可行。用随机投影 R ∈ ℝ^{k×d}（k≪d）：

```
z_j = R · ∇ℓ(x_j^val; θ_s) ∈ ℝ^k
H̃ = (1/M) Σ z_j z_j^T ∈ ℝ^{k×k}
```

- 存储 O(Nd) → O(Nk)
- 论文 GPT-2-Medium: k=34，覆盖 94.5% 谱能量

### 2.6 SCA 求解器

问题是非凸（二次目标 + 组合约束）。用 Successive Convex Approximation：
1. 松弛 w_i ∈ [0,1]
2. 用线性代理替代二次目标
3. 每次迭代解一个 LP
4. 投影梯度上升 + Dykstra 投影
5. 收敛后四舍五入得到离散选择

### 2.7 理论保证

**Theorem 4.4 (Flat-direction preference)**：沿低特征值方向移动会增加 flatness S(θ) = Tr(H_ρ^{-1})，并降低该方向曲率。

**PAC-Bayes 泛化界**：flatness 直接关联泛化保证，曲率惩罚缩小 → 更好的 train-test gap。

---

## 三、实验结果

### 3.1 实验设置

| 维度 | 设置 |
|------|------|
| 模型 | GPT-2-Medium (355M), Llama-1.1B |
| 预训练数据 | SlimPajama (GPT-2), The Pile (Llama) |
| Annealing 数据 | **1.8M 样本, 2.1GB**（MathPile 1.27M + StarCoderData 0.23M + Dolma Longmino 0.30M） |
| 验证集 | 0.18M 样本（10% split） |
| 选择比例 | 80%（从 1.62M 训练样本中选 ~1.3M） |
| Annealing 超参 | E=5 epochs, B=4, lr=1e-5, cosine + warmup 0.1, clip 1.0 |
| DiReCT 超参 | τ_⊥=0.1, k=34 (GPT-2), SCA max_iter=20, δ=1e-4 |

### 3.2 主结果

| 方法 | GPT-2-Medium 平均 | Llama-1.1B 平均 |
|------|-------------------|-----------------|
| Uniform | 30.2 | 37.9 |
| Perplexity-based | 31.0 | 38.2 |
| Loss-based | 30.4 | 38.3 |
| GradNorm (IS) | 31.3 | 38.9 |
| InfoBatch | 32.1 | 40.1 |
| **DiReCT** | **32.3 (+2.1)** | **40.3 (+2.4)** |

**关键发现**：
- DiReCT 在两个模型规模上均最高
- 数学/代码 benchmark 提升最大（GSM8K +3.4, HumanEval +5.0 @1.1B）
- 通用 benchmark（HellaSwag, PiQA）波动小 → uniform 已足够

### 3.3 学到的选择逻辑

- **偏好高 loss 样本**（Pearson r=+0.811）→ 模型未学好的模式提供更强下降信号
- **偏好长序列**→ 更丰富的语义依赖 → 梯度在 flat 子空间分量更大

### 3.4 超参指导

- ε 通过谱肘检测确定（累积能量曲线拐点）
- GPT-2-Medium: k=34, ε=158.88, 覆盖 94.5% 谱能量
- τ_⊥ 与 k 耦合：调 τ_⊥ 找最优 k

---

## 四、可扩展性分析

### 4.1 模型规模

| 维度 | 能力 |
|------|------|
| 355M | ✅ 测试 |
| 1.1B | ✅ 测试 |
| 1.5B+ | 🟡 未测试，但 sketching 方法定义上可处理（k≈34-100 << d） |

随机投影将 d≈10^9 降到 k≈34-100，存储 O(Nk)，Hessian 近似只需验证集 forward pass。

### 4.2 数据规模

| 维度 | 论文 | 用户场景 |
|------|------|---------|
| Annealing 数据 | **1.8M 样本 (2.1GB)** | 100B tokens |
| 训练样本数 N | ~1.62M | ~10^8-10^9 |

**论文明确承认的限制**：

> "gathering gradients across the full D_train still requires a forward pass, which is **nontrivial at trillion-token scale**. We leave dynamic re-partitioning and parameter-subset backpropagation for gradient collection as **future work**."

### 4.3 计算成本

- 验证集 Hessian 近似：M 个 forward+backward（M~10^4-10^5，成本可控）
- 训练集梯度投影：N 个 forward+backward（**100B 数据不可行**）
- SCA 求解器：T_max=20 次迭代，LP 求解
- **论文未报告 GPU 小时数**

### 4.4 定位限制

- **Annealing 阶段专用**，不是主训练阶段方法
- **样本级选择**，不是域级配比方法
- 适用于冷却阶段的小数据子集（1-5B tokens），不适用于全量 100B 数据

---

## 五、业界认可度

| 维度 | 数据 |
|------|------|
| Semantic Scholar 引用 | **0**（2026.05 提交，极新） |
| GitHub Stars | **2** |
| GitHub Forks | 0 |
| 代码更新 | 2026.07 最后 push |
| 头部厂商 tech report 引用 | **无** |
| Venue | ICML 2026 |
| 来源 | MSRA + HKUST(GZ) |

### 引用 Velocitune 的关系

DiReCT 在 related work 中引用 Velocitune 作为 stable training 阶段代表方法，自身聚焦 annealing 阶段。两者是**互补关系**：
- Velocitune：stable training 阶段的动态域配比
- DiReCT：annealing 阶段的样本级选择

---

## 六、对用户场景的适用性

### 用户配置
- 100B STEM 数据 + 1.5B+ 模型
- 4 STEM 域，6 benchmark
- 中训练（不是 annealing）

### 适用性矩阵

| 维度 | 判断 | 说明 |
|------|------|------|
| 域级配比 | ❌ | 样本级选择，不是域级 |
| 中训练 | 🟡 | 论文是 annealing 阶段，不是主训练 |
| 100B 数据 | ❌ | 论文明确说 trillion-token 是 future work |
| 1.5B 模型 | 🟡 | 论文测到 1.1B，1.5B 接近 |
| 冷却阶段 | ✅ | 如果用户有冷却阶段且数据是小子集 |
| 代码开源 | ✅ | https://github.com/xuyj233/Direct |

### 实际价值

**对用户的主训练配比问题**：不适用（样本级 + annealing 阶段 + 100B 不可行）

**对用户的冷却阶段**：如果中训练有冷却阶段，且冷却数据是 1-5B tokens 的子集，DiReCT 可以用于该子集的样本选择。但论文未在 >1.1B 模型上验证。

**理论启发**：
1. **Flat vs stiff 方向**：验证集 Hessian 的谱几何提供了新的数据选择视角
2. **高 loss 样本优先**：annealing 阶段应选模型未学好的样本（r=+0.811）
3. **长序列优先**：更丰富的语义依赖 → flat 子空间分量大

---

## 七、总结

DiReCT 的核心价值：
1. **首次为 annealing 阶段提供优化理论基础** — Hessian 谱几何 → 有原则的样本选择
2. **Flat-direction preference 定理** — 沿低曲率方向移动增加 flatness → PAC-Bayes 泛化保证
3. **随机投影可扩展** — O(Nd) → O(Nk)，k≈34
4. **学到的选择逻辑可解释** — 偏好高 loss + 长序列

**对用户的核心限制**：100B 数据不可行（论文明确 future work），样本级而非域级，annealing 阶段而非主训练。仅适用于冷却阶段小子集。
