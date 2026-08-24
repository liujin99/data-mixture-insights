# LLM 数据选择配比问题的形式化定义

> 本文档用数学语言严格定义数据配比优化问题，分析目标函数的五层困难结构，并论证代理迁移路径在信息论意义上的不可解性。
>
> 综合 69 篇论文的分析提炼，作为项目的理论根基。

---

## 一、问题设定

### 1.1 符号表

| 符号 | 类型 | 含义 | 本项目实例 |
|------|------|------|-----------|
| $M$ | $\mathbb{R}^+$ | 模型参数量 | 1.5B |
| $T$ | $\mathbb{R}^+$ | 训练 token 预算 | 1.8B |
| $K$ | $\mathbb{N}$ | 数据域数量 | 待定（ClimbLab 域分类） |
| $\mathcal{D} = \{D_1, \ldots, D_K\}$ | — | 各域数据池（$|D_k| \gg T$） | ClimbLab 1.2T 子集 |
| $\mathcal{E} = \{e_1, \ldots, e_L\}$ | — | 评估任务集 | {GSM8K, MATH, MMLU, ARC, ...} |
| $\boldsymbol{w} = (w_1, \ldots, w_L)$ | $\mathbb{R}_+^L$ | 评估任务权重（$\sum w_\ell = 1$） | STEM 加权 |
| $N$ | $\mathbb{N}$ | 允许的训练运行次数 | 1-2 |
| $S_e$ | $\mathbb{R}^+$ | 任务 $e$ 的知识复杂度（事实总数） | GSM8K $\sim$8K |
| $\boldsymbol{r}$ | $\Delta^{K-1}$ | **决策变量**：域配比向量 | — |

### 1.2 决策变量

$$\boldsymbol{r} = (r_1, \ldots, r_K) \in \Delta^{K-1}$$

其中 $\Delta^{K-1}$ 为 $K-1$ 维概率单纯形：

$$\Delta^{K-1} = \left\{\boldsymbol{r} \in \mathbb{R}^K \,\middle|\, r_k \geq 0,\; \sum_{k=1}^K r_k = 1\right\}$$

$r_k$ 表示域 $k$ 在训练数据中占的 token 比例。

### 1.3 训练过程

$$\theta^*(\boldsymbol{r};\, M,\, T,\, \mathcal{D}) = \text{Train}\!\left(M,\; \mathcal{S}(\boldsymbol{r}, \mathcal{D}, T),\; T\right)$$

其中 $\mathcal{S}(\boldsymbol{r}, \mathcal{D}, T)$ 为按配比 $\boldsymbol{r}$ 从各域 $\mathcal{D}$ 中采样共 $T$ 个 token 的数据集。$\theta^*$ 为训练后的模型参数。

本项目设定：基座已预训练于 ClimbLab 1.2T（800 tokens/param，充分），中训练为 continual pre-training（在基座权重上继续训练）。

---

## 二、形式化定义

### 2.1 原始优化问题（accuracy 目标）

$$\boxed{\boldsymbol{r}^* = \arg\max_{\boldsymbol{r} \in \Delta^{K-1}} F(\boldsymbol{r};\, M,\, T,\, \mathcal{D},\, \mathcal{E})}$$

目标函数：

$$F(\boldsymbol{r};\, M,\, T,\, \mathcal{D},\, \mathcal{E}) = \sum_{e \in \mathcal{E}} w_e \cdot A_e\!\left(\theta^*(\boldsymbol{r};\, M,\, T,\, \mathcal{D})\right)$$

其中 $A_e(\theta) \in [0, 1]$ 为模型 $\theta$ 在任务 $e$ 上的 accuracy。

### 2.2 loss 代理形式（多数论文实际优化的）

实践中 $F$ 不可微且评估昂贵，多数方法转而优化 validation loss：

$$\boldsymbol{r}^*_\mathcal{L} = \arg\min_{\boldsymbol{r} \in \Delta^{K-1}} \mathcal{L}(\boldsymbol{r};\, M,\, T,\, \mathcal{D},\, \mathcal{V})$$

$$\mathcal{L}(\boldsymbol{r};\, M,\, T,\, \mathcal{D},\, \mathcal{V}) = \frac{1}{|\mathcal{V}|}\sum_{x \in \mathcal{V}} -\log p_{\theta^*(\boldsymbol{r})}(x)$$

其中 $\mathcal{V}$ 为验证集。**关键警告**：$\boldsymbol{r}^*_\mathcal{L} \neq \boldsymbol{r}^*$ 一般成立（详见第五节）。

### 2.3 计算约束

$$\text{s.t.} \quad \text{eval-count}(F) \leq N$$

每次 $F(\boldsymbol{r})$ 的评估需要一次完整训练（成本 $\propto M \cdot T$）。本项目 $N = 1\text{-}2$，即 **1-2 shot 优化**——连坐标降维都不够。

### 2.4 bilevel 形式（文献视角）

Survey #2 (#50) 将配比优化形式化为概率单纯形上的 bilevel 问题：

$$\min_{\boldsymbol{r} \in \Delta^{K-1}} \quad \underbrace{\mathcal{L}_{\text{outer}}(\theta^*(\boldsymbol{r});\, \mathcal{V})}_{\text{外层：验证集 loss}}$$

$$\text{s.t.} \quad \theta^*(\boldsymbol{r}) = \underbrace{\arg\min_\theta \mathcal{L}_{\text{inner}}(\theta;\, \boldsymbol{r}, \mathcal{D}, T)}_{\text{内层：训练}}$$

DoReMi、FastMix、QuaDMix 均为此 bilevel 问题的不同近似求解策略。区别在于如何处理内层（训练）的高成本：proxy 替代、可微逼近、或 Bayesian optimization。

---

## 三、目标函数 $F(\boldsymbol{r})$ 的性质

以下五条性质共同构成问题的困难性。前三条是 $F$ 本身的性质，后两条是代理范式的困难。

### 3.1 黑箱性

$F$ 不可微、无梯度、无解析式。$\nabla_{\boldsymbol{r}} F$ 不存在（训练是黑箱）。每次评估需要完整训练。$N=1\text{-}2$ 意味着标准黑箱优化方法（Bayesian optimization、网格搜索、坐标降维）均不可行——它们至少需要 $O(K)$ 次评估。

### 3.2 相变结构（不连续性）

由 Phase Transitions (#01, Gu et al., NeurIPS'25 Spotlight)，对复杂度为 $S_e$ 的任务 $e$，accuracy 关于配比 $r_e$（$e$ 对应域的配比）存在**尖锐相变**：

$$A_e(\boldsymbol{r};\, M,\, T) \approx \begin{cases} 0 & \text{若 } r_e < r_{\text{thres}}(M, S_e) \\ & \text{或 } T < T_{\text{thres}}(r_e, M, S_e) \\[4pt] g_e(r_e, T;\, M, S_e) & \text{否则（单调递增）} \end{cases}$$

临界配比幂律：

$$\boxed{r_{\text{thres}}(M, S_e) \propto \frac{S_e}{M^{\alpha+1}}, \quad \alpha \approx 0.283,\; \alpha+1 = 1.283}$$

训练步数在阈值以下的增长：

| $r_e$ 区域 | $T_{\text{thres}}$ 增长 | 拟合 |
|-----------|------------------------|------|
| $r_e > 0.4$ | 线性 $\propto 1/r_e$ | — |
| $r_e < 0.4$ | 指数 $\propto e^{B/r_e}$, $B \approx 1.514$ | $R^2 = 0.998$ |
| $r_e < 0.25$ | 超指数（偏离幂律 $2.9\times$） | $T(r) = C \cdot r^{-3.839}$ |

**推论**：$F(\boldsymbol{r})$ 在 $r_e = r_{\text{thres}}$ 处导数不连续。$F$ 不是光滑函数。假设光滑性的优化方法（梯度下降、Bayesian optimization with RBF kernel）在阈值附近失效。

**极端实验**：$r = 0.2$, $M = 70\text{M}$, 训练 $256\text{B}$ token（每条 biography 出现 $\sim$3000 次）$\to$ accuracy 仍 $\approx 0\%$。低于阈值训再久等于零。

### 3.3 尺度依赖性

$F$ 的景观**定性**地依赖于 $M$：

$$\exists\, M_1 \neq M_2:\quad \arg\max_{\boldsymbol{r}} F(\boldsymbol{r};\, M_1) \neq \arg\max_{\boldsymbol{r}} F(\boldsymbol{r};\, M_2)$$

不是精度差异，是**最优解位置不同**。两个机制：

#### 机制 1：相变阈值差异 $\to$ 不同 regime

| 模型 | $r_{\text{thres}}$（GSM8K 估算） | $r = 0.7$ 时状态 |
|------|------|------|
| 14M (proxy) | $\sim$0.30-0.40 | 阈值附近，极度敏感 |
| 1.5B (target) | $\sim$0.06-0.08 | 阈值 10 倍以上，线性区 |

阈值附近的最优配比关注"如何跨过阈值"，线性区的最优配比关注"如何最大化利用已跨阈值的域"。两者的最优解**方向可以相反**。

QuaDMix 实验验证：14M 搜出的配比在 1.5B 上 6 个任务中 4 个**比 Random 还差**——方向反了。

#### 机制 2：容量竞争 $\to$ 背包问题对不同 $M$ 有不同解

模型为最优有界容量学习器（Phase Transitions, Theorem 4.3），在容量 $M$ 下最小化期望 test loss = **分数背包问题**：

$$\min_{\boldsymbol{c}} \sum_{k=1}^K V_k(c_k;\, M) \quad \text{s.t.} \quad \sum_{k=1}^K c_k \leq M$$

- $V_k(c_k;\, M)$ = 域 $k$ 分配 $c_k$ 容量时的 test loss 下降量（边际价值）
- $M$ 改变 $\Rightarrow$ 各域边际价值排序改变 $\Rightarrow$ 背包解改变

知识密集数据"值得学"的条件：

$$\frac{r_e}{1 - r_e} > \frac{D^- F_{P_2}(M)}{D^+ F_{P_1}(0)}$$

其中 $D^- F_{P_2}(M)$ 和 $D^+ F_{P_1}(0)$ 分别为 web 数据和知识数据的 loss 对容量的导数。

**14M**（容量极稀缺）：某些 STEM 子域的边际价值 $<$ web $\to$ 最优"少给 STEM"
**1.5B**（容量充足）：同样子域的边际价值 $>$ web $\to$ 最优"多给 STEM"
$\to$ **方向相反**

Capacity Competition (#38) 从理论上解释：有限模型容量在域间分配，导致域 loss 全局耦合。proxy（小容量）和 target（大容量）的容量竞争机制完全不同。

### 3.4 状态配置

定义域 $k$ 在模型 $M$ 下的状态：

$$\sigma_k(M) = \begin{cases} \text{post-transition} & r_k > r_{\text{thres}}(M, S_k) \\ \text{pre-transition} & r_k < r_{\text{thres}}(M, S_k) \end{cases}$$

状态向量 $\boldsymbol{\sigma}(M) = (\sigma_1(M), \ldots, \sigma_K(M))$ 决定了 $F$ 的**局部结构**（哪些域在相变前、哪些在相变后）。当 $M_p \ll M_t$：

$$\boldsymbol{\sigma}(M_p) \neq \boldsymbol{\sigma}(M_t)$$

不同状态配置 $\Rightarrow$ 不同局部结构 $\Rightarrow$ 不同最优解。这是第四节不可解性论证的基础。

---

## 四、代理范式与不可解性

### 4.1 代理范式

给定 proxy 模型 $M_p \ll M_t$（target），代理方法在 $M_p$ 上以低成本（可做数百次训练）搜索配比，然后迁移到 $M_t$：

$$\hat{\boldsymbol{r}} = h\!\left(\mathcal{L}_p(\boldsymbol{r}_1), \ldots, \mathcal{L}_p(\boldsymbol{r}_n)\right) \approx \boldsymbol{r}^*_t = \arg\max_{\boldsymbol{r}} F_t(\boldsymbol{r})$$

其中 $\mathcal{L}_p = \mathcal{L}(\cdot;\, M_p)$, $F_t = F(\cdot;\, M_t)$, $h$ 为推断函数。

具体实例：

| 方法 | $h$ 的形式 | proxy | 搜索空间 |
|------|-----------|-------|---------|
| DoReMi | mirror descent on proxy loss | 280M | simplex |
| RegMix / QuaDMix | 回归（LightGBM/OLS）proxy loss $\to$ target loss | 14M-280M | 198 维 |
| FastMix | 可微 bilevel（闭式梯度） | 280M | simplex |
| LogFloor (#68) | proxy 动态瓶颈顺序 replay | 70M | 顺序空间 |

### 4.2 可行性条件

代理有效的**充要条件**：

$$\arg\max_{\boldsymbol{r}} F_p(\boldsymbol{r}) \approx \arg\max_{\boldsymbol{r}} F_t(\boldsymbol{r})$$

即 proxy 最优 $\approx$ target 最优。等价地：

$$\boldsymbol{\sigma}(M_p) \approx \boldsymbol{\sigma}(M_t)$$

即两模型处于相同的状态配置。

### 4.3 不可解性定理

**命题 1（绝对不可解）**：若 $\exists$ 任务 $e$ 使得 $r_{\text{thres}}(M_p, S_e) > 1$（proxy 永远无法学会任务 $e$），则：

$$F_p(\boldsymbol{r})\Big|_{\text{task } e} = 0 \quad \forall\, \boldsymbol{r} \in \Delta^{K-1}$$

$F_p$ 在任务 $e$ 上为**常数函数**，关于 $\boldsymbol{r}^*_t$ 包含**零信息**。任何推断函数 $h$ 无法从常数中提取信息。

**命题 2（regime 不匹配不可解）**：若 $\boldsymbol{\sigma}(M_p) \neq \boldsymbol{\sigma}(M_t)$（状态配置不同），则 $F_p$ 和 $F_t$ 处于**不同 regime**：

- $F_p$ 提供关于"状态 $\boldsymbol{\sigma}(M_p)$ 下最优"的信息
- 我们需要"状态 $\boldsymbol{\sigma}(M_t)$ 下最优"
- 两者的关系**不可先验确定**——可以从"方向相同"到"方向相反"

QuaDMix 实验验证了最坏情况：方向相反 $\to$ 比 Random 还差。

**信息论解释**：

相变后的最优配比可能依赖于**相变后才会出现的表示空间结构** $\Phi_{\text{post}}(M_t)$。若 $\Phi_{\text{post}}(M_t)$ 在 $M_p$ 上不存在或不可观测，则：

$$I\!\left(F_p(\boldsymbol{r});\; \boldsymbol{r}^*_t\right) = 0$$

即 $F_p$ 与 $\boldsymbol{r}^*_t$ 的互信息为零。**不是"方法不够好"的问题，是"信号不存在"的问题。**

Why Larger Models Learn More (#52) 的实证支持：小模型把神经元分配给高频/低复杂度域，稀有复杂域被牺牲——不是"学得慢"而是"学不了"。容量不够时优先级排序本身就不同。

### 4.4 实证验证

| 证据 | 含义 | 来源 |
|------|------|------|
| 14M gsm8k R²=0.751（within-model），但 QuadMix 在 1.5B gsm8k 最差 | 高 R² 也不保证 cross-model 迁移 | QuaDMix 实验 |
| 静态 marginal mixture 仅保留 10.3% 收益（vs 正确顺序 60.3%） | proxy 的价值在顺序不在配比 | LogFloor #68 消解 |
| 70M→12B 顺序迁移在无相变任务（bAbI/MMLU）成功 | 无相变时难度层级尺度不变 → 迁移可行 | LogFloor #68 |
| 有相变任务（STEM accuracy）未测试 | 有相变时迁移未验证——恰好是 QuaDMix 撞的墙 | LogFloor #68 局限 |
| QuadMix Final Loss=0.9166（最低），但 6/4 任务最差 | 14M 优化的 loss 方向在 1.5B accuracy 上反向 | QuaDMix 实验 |

---

## 五、loss-accuracy 鸿沟

即使放弃代理、在 target scale $M_t$ 上直接用 loss 搜索（如 DomainPilot #39），仍有根本问题：**loss 最优 $\neq$ accuracy 最优**。

### 5.1 形式定义

$$\boldsymbol{r}^*_\mathcal{L} = \arg\min_{\boldsymbol{r}} \mathcal{L}(\boldsymbol{r};\, M_t) \quad \neq \quad \boldsymbol{r}^* = \arg\max_{\boldsymbol{r}} F(\boldsymbol{r};\, M_t)$$

两者最优解一般不重合。鸿沟有两个来源。

### 5.2 来源 1：close vs distant minimizer

Nexus (#51, 清华+字节跳动) 发现：loss landscape 存在**多个极小值**，优化器可能收敛到不同 minimizer：

- **Close minimizer**：各域在表示空间中都有覆盖，accuracy 高
- **Distant minimizer**：部分域被牺牲，表示空间偏斜，accuracy 低
- 两者的 validation loss **几乎相同**（差 0.004），但 GSM8K accuracy 差 **15 分**（44.0 vs 59.0）

这意味着 $\mathcal{L}(\boldsymbol{r})$ 作为 $F(\boldsymbol{r})$ 的代理，**系统性缺失了一个维度**——"优化器收敛到 close 还是 distant minimizer"。这不是噪声问题（3B 模型上仍然如此），是 loss 信号本身的结构性盲区。

Nexus 的解法：通过最大化各域梯度余弦相似性，强制模型收敛到 close minimizer。代码改动 $\sim$20 行，几乎零计算开销。但这只缓解、不解决 loss-accuracy 鸿沟——Nexus 是正交干预，不是配比优化。

### 5.3 来源 2：loss 的相变滞后

Loss 在 phase transition **之前**就开始下降（只是 accuracy 不变）。因此 $\mathcal{L}(\boldsymbol{r})$ 的景观比 $F(\boldsymbol{r})$ 更光滑：

| 函数 | $r < r_{\text{thres}}$ | $r > r_{\text{thres}}$ | 在阈值处 |
|------|----------------------|----------------------|---------|
| $\mathcal{L}(\boldsymbol{r})$ | 平滑下降 | 继续下降 | 连续 |
| $F(\boldsymbol{r})$ | $= 0$（常数） | 快速上升 | **导数不连续** |

$\mathcal{L}$ 把相变"抹平"了——光滑化的代价是**丢失了相变信息**。$\mathcal{L}$ 的最优可能在相变前就达到（loss 已经降够了但 accuracy 还没涌现），而 $F$ 的最优需要跨过相变。

Schaeffer 2023 (#03) 指出：accuracy 的"涌现"可能部分是评估指标阈值造成的假象，但对下游 accuracy 目标而言，假象不影响结论——我们关心的是 accuracy 值本身，不管它是"真涌现"还是"指标阈值效应"。

### 5.4 QuaDMix 直接实证

QuaDMix 实验提供了 $\boldsymbol{r}^*_\mathcal{L} \neq \boldsymbol{r}^*$ 的直接证据：

| 配比来源 | Final Loss（越低越好） | GSM8K | 6 任务中劣于 Random |
|---------|----------------------|-------|-------------------|
| QuadMix（14M loss 优化） | **0.9166**（最低） | 0.0697（最差） | 4/6 |
| Random | 1.4508 | 0.0804 | — |
| Manual (3:7) | 1.3583 | 0.0834 | — |

QuadMix 的 training loss 最低——14M proxy 确实找到了降低 loss 的配比。但这个配比在 1.5B accuracy 上 4/6 任务比 Random 还差。**loss 最优和 accuracy 最优方向相反。**

### 5.5 LogFloor 的绕行与未验证的桥

LogFloor (#68) 试图绕过 loss-accuracy 鸿沟：

- 用 log-loss **下降顺序**（瓶颈排序）而非 loss 值本身
- 在 bAbI/MMLU 上验证 70M→12B 顺序迁移成功（30.9% token 节省）
- 但 bAbI/MMLU 是无相变任务（70M 就能取得 loss 进展且 accuracy 提升）

**未验证的桥**：log-loss 下降顺序是否等于 accuracy 涌现顺序？

| 条件 | 无相变任务（bAbI） | 有相变任务（STEM） |
|------|-------------------|-------------------|
| loss 下降和 accuracy 提升的关系 | 高度相关 | loss 下降可能领先 accuracy 涌现很久 |
| 难度排序跨尺度不变？ | 成立 | **可能不成立**（相变前后重排） |
| LogFloor 验证？ | ✅ | ❌ |

这个桥 LogFloor 没有搭——恰好是 QuaDMix 撞的同一面墙。

### 5.6 实践启示

loss 信号在配比优化中的**可用边界**：

| 用途 | 可行性 | 理由 |
|------|--------|------|
| 粗略方向引导（loss 高的域多给） | ⚠️ 有条件可行 | loss 确实反映模型在哪些域学得不够 |
| 精确配比预测（loss 最优 = accuracy 最优） | ❌ 不可行 | close/distant minimizer + 相变滞后 |
| 域级学习能力排序（loss 下降速度排序） | ⚠️ 无相变任务可行 | LogFloor 在 bAbI/MMLU 验证 |
| 域级学习能力排序（有相变任务） | ❌ 未验证 | loss 排序可能 ≠ accuracy 涌现排序 |
| 早期 stopping 信号（val_loss 不再降） | ✅ 可行 | Phase Transitions 论文使用 val_loss +0.05 标准 |

> **loss 是配比优化的必要信息源，但不是充分信息源。** 它可以告诉"哪些域学得不够"（方向性），但不能告诉"精确给多少"（定量），更不能预测"相变后什么配比最优"。

---

## 六、本项目的问题实例

### 6.1 参数

$$M = 1.5\text{B},\quad T = 1.8\text{B},\quad N = 1\text{-}2$$

$$\mathcal{E} = \{\text{GSM8K},\, \text{MATH},\, \text{MMLU},\, \text{ARC},\, \text{GPQA},\, \ldots\}, \quad \boldsymbol{w} = \text{STEM 加权}$$

基座预训练于 ClimbLab 1.2T（800 tokens/param），中训练为 continual pre-training。

### 6.2 关键特征

| 特征 | 值 | 含义 |
|------|------|------|
| $r_{\text{thres}}$(1.5B, GSM8K) | $\sim$0.06-0.08 | 用户 $r=0.7$ 是阈值 $\sim$10 倍 $\to$ 在线性增长区 |
| 数学 token | $\sim$1.26B（$r=0.7 \times 1.8\text{B}$） | DeepSeekMath 参考线 150B 的 $\sim$1% $\to$ 数据量生死瓶颈 |
| $N$ | 1-2 | 不可搜索，必须 one-shot 或 heuristic |
| STEM 相变状态 | 近临界（1.5B 在涌现边缘） | 景观极度敏感，配比微小变化可能导致大 accuracy 波动 |
| 基座状态 | 已充分预训练 | 容量竞争已"用完"基础分配，中训练调整风味 |

### 6.3 域状态分类

| 状态 | 条件 | 含义 | 对策 |
|------|------|------|------|
| 远超阈值 | $r_k \gg r_{\text{thres}}$ | 域已充分学习，边际收益递减 | 可以减少配比 |
| 刚跨阈值 | $r_k \gtrsim r_{\text{thres}}$ | 正在相变中，极度敏感 | 精确配比关键但不可预测 |
| 低于阈值 | $r_k < r_{\text{thres}}$ | accuracy = 0%，loss 在降 | 需大幅增加配比或接受不学 |
| 已收敛 | 域数据量 $\to$ 饱和 | 更多 token 无收益 | 不增加配比 |

### 6.4 可行性边界

$$\underbrace{\text{proxy 预测}}_{\text{不可行（第四节）}} \quad \underbrace{\text{target-scale 搜索}}_{\text{不可行（}N=1\text{-}2\text{）}} \quad \underbrace{\text{loss 引导（target scale）}}_{\text{可能但 loss}\neq\text{acc（第五节）}} \quad \underbrace{\text{heuristic}}_{\text{可行但不保证最优}} \quad \underbrace{\text{正交干预}}_{\text{Nexus 等优化器方法}}$$

---

## 七、小结

### 7.1 五层困难结构

| 层次 | 困难 | 数学表述 | 来源 |
|------|------|---------|------|
| **第一层：黑箱优化** | $F$ 不可微，$N=1\text{-}2$ shot | $\nabla_{\boldsymbol{r}} F$ 不存在 | 计算约束 |
| **第二层：相变结构** | $F$ 在 $r_{\text{thres}}$ 处不连续 | $A_e = 0$ if $r < r_{\text{thres}}$; else $g(r)$ | Phase Transitions #01 |
| **第三层：尺度依赖** | $F$ 景观随 $M$ 定性改变 | $\boldsymbol{\sigma}(M_p) \neq \boldsymbol{\sigma}(M_t)$ | 背包理论 + 容量竞争 #38 |
| **第四层：代理不可解** | proxy 信号 $\perp$ target 最优 | $I(F_p; \boldsymbol{r}^*_t) = 0$ | 信息论论证 |
| **第五层：loss-accuracy 鸿沟** | $\boldsymbol{r}^*_\mathcal{L} \neq \boldsymbol{r}^*$ | 同 loss 差 15 分 | Nexus #51 |

### 7.2 核心判断

> **数据配比问题 = 黑箱 + 相变 + 尺度依赖 + 代理不可解 + loss-accuracy 鸿沟。**
>
> 五层困难叠加，其中第四层（代理不可解）在信息论意义上不可逾越——当 $\boldsymbol{\sigma}(M_p) \neq \boldsymbol{\sigma}(M_t)$ 时，proxy 信号与 target 最优的互信息为零。这解释了 QuaDMix 的失败不是 bug 而是必然。
>
> 第五层（loss-accuracy 鸿沟）意味着即使放弃代理、在 target scale 操作，loss 引导也只能提供方向性参考，不能精确预测 accuracy 最优。
>
> 在 $N=1\text{-}2$ 的约束下，可行的路径是：(a) target-scale loss 引导 + 启发式约束，(b) 正交干预（优化器层面降低配比敏感度），(c) 接受不可预测性、用 heuristic 配比。

---

## 附录：相关文档索引

| 文档 | 关系 |
|------|------|
| `QuaDMix失败分析_相变理论启示.md` | 第四节实证 + Phase Transitions 理论详解 |
| `核心未解决问题_相变与代理迁移.md` | 第四节各方法应对策略汇总 |
| `LogFloor_深度分析.md` | 第四节 + 第五节的 LogFloor 案例分析 |
| `STEM能力涌现分析.md` | 第三节数据量瓶颈 + 第六节项目实例 |
| `Nexus_深度分析.md` | 第五节 close/distant minimizer 详解 |
| `FastMix_DomainAwareScalingLaws_深度分析.md` | 第二节 bilevel 形式 + scaling law 方法 |
