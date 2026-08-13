# CAMEL 深度分析 v3

**论文**: Capacity-Aware Mixture Law Enables Efficient LLM Data Optimization
**arXiv**: 2603.08022v2 (2026.03.09 提交, 2026.05.06 v2)
**作者**: Jingwei Li, Xinran Gu, Jingzhao Zhang
**Venue**: ICML 2026
**架构**: DeepSeek V3 MoE, 590M-A12M ~ 7B-A150M → 55B-A1.2B 外推

---

## 一、核心方法

### 1.1 问题定位：Mid-Training

论文**明确针对 mid-training 场景**（非 from-scratch pretraining）：

> "Mid-training is a crucial period when the model quickly builds essential skills such as abstract reasoning, math problem solving, coding, and planning. In cases like mid-training, data quality is prioritized over data quantity, as model ability can be easily weakened when mixed with low-quality, high-volume web text."

**与用户场景完全一致**：用户做中训练数据配比，3:7 通用:STEM。

### 1.2 Capacity-Aware Mixture Law (CAMEL)

**核心公式 (Theorem 2.2)**：

```
L_val(r, M) = C + Σ_i K_i / (<t_i, r>^α_i · M^β_i)
```

| 符号 | 含义 |
|------|------|
| r | 混合配比向量（在概率单纯形上） |
| M | 模型大小（总容量） |
| <t_i, r> | 内在域 i 在配比 r 下的有效权重 |
| α_i | 配比对 loss 的指数影响（域特定） |
| β_i | 模型大小对 loss 的指数影响（域特定） |
| K_i | 域特定常数 |
| C | 高阶项（常数） |

**关键区别**：CAMEL **联合建模**配比和模型大小。之前的 DML (Ye et al. 2025) 和 SODM (Shukhor et al. 2025) 将两者分开建模。

**理论基础**：从训练损失最小化的约束优化问题（Eq. 1）出发，在 Assumption 2.1（域间指数接近 + 稳定内在域权重）下，通过 KKT 条件求解，得到验证损失的表达式（Theorem 2.2）。证明见 Appendix D。

### 1.3 容量分配视角

**核心假设**：预训练是一个隐式的容量分配优化问题：

```
min_{m̃}  L_train(r; m̃) = Σ_i η_i(r) · ℓ_i(m̃_i)
s.t.     Σ_i m̃_i ≤ M
```

其中 ℓ_i(m̃_i) = C_i + A_i / m̃_i^{a_i} 是域 i 的训练损失（幂律形式）。

模型在总容量 M 约束下，自然地将容量分配到各内在域。**不同域的容量分配不成比例**：

实验测量（Figure 3）：在固定混合比下训练 small/medium/large 三个模型规模，测量各域 loss 改善率：

```
R_i = Δ_i^(L) / Δ_i^(M) = (ℓ_small - ℓ_large) / (ℓ_small - ℓ_medium)
```

结果：
- **R_math ≈ 2.4**（数学域容量吸收慢）
- **R_know ≈ 4.3**（知识域容量吸收快）

如果有效参数随模型大小等比缩放，则 R_i 应在所有域上相同。但 R_math ≠ R_know 表明**有效参数分配不随模型大小线性增长，而是动态调整**。

**这直接解释了为什么小模型 proxy 配比不能迁移到大模型** — 不同域的容量分配随模型大小变化的速率不同。

### 1.4 Loss-to-Benchmark 预测定律

**核心公式 (Eq. 3)**：

```
Acc_b(L) = C_b + A_b / (1 + exp(k_b^T · L + B_b))
```

其中 **L = (L_1, ..., L_N)** 是来自**多个验证集**的 loss 向量，b 是目标 benchmark。

**关键设计**：
- 之前的方法：单个 val_loss → benchmark（不可靠，Lourie et al. 2025 证实）
- CAMEL：**多个 val_loss → benchmark**（多维度信号）
- 采用 logistic 形式，扩展自 Dubey et al. (2024) Llama 3 和 Bhagia et al. (2024)

**理由**：benchmarks（如 MMLU）需要综合多种能力（知识+推理+语言理解），单一 val_loss 不足以预测。

**拟合数据**：在 14 个常用 benchmark 上验证，预测误差低（Figure 4）。但需要 >200 个验证集来拟合 loss-to-benchmark 映射。

### 1.5 端到端流水线

```
配比 r → L_b(r, M) [benchmark-proxy loss] → Acc_b(r, M) [benchmark accuracy]
```

1. 先用多个验证集 loss 拟合 loss-to-benchmark 映射（成本低：单次训练获得多个 checkpoint 的 loss-accuracy 对）
2. 定义 benchmark-proxy loss: L_b = k_b^T · L（降维，减少参数数量）
3. 用 CAMEL 定律拟合 L_b(r, M)（成本高：每个配比需独立训练）
4. 外推到目标模型大小 M_target，搜索最优 r* = argmax_r A(r, M_target; w)

### 1.6 Hourglass 采样策略

在固定计算预算下，**沙漏型采样**（两端多、中间少）最小化外推误差：

- **最小模型**：多采样（成本低，提供低 M 锚点）
- **最大模型**：适当采样（锚定外推端点）
- **中间模型**：少采样（信息量低，性价比差）

**对比 5 种策略**（Figure 7）：hourglass < diamond < triangle < inverted triangle < rectangle

**实践意义**：如果要做多规模实验拟合 scaling law，不要均匀分配算力，应优先最小和最大规模。

### 1.7 内在域 (Intrinsic Domains)

**概念**：不要求显式域划分，用 k 个"内在域"（latent components）解释验证 loss 的变化。

- t_ij ≥ 0：内在域 i 在数据集 j 中的比例，满足 Σ_i t_ij = 1
- η_i(r) = <t_i, r>：混合比 r 诱导的内在域 i 的有效权重
- k 是超参数，论文证明方法对 k 的选择鲁棒（Appendix B.2）

**实践意义**：198 个显式域可以通过 PCA/NMF 等方法投影到 k=5-10 个内在域，降低优化维度。

### 1.8 Cooldown 阶段

**论文描述**（Appendix A.2）：

> "For each model scale, training proceeds in two stages. We first pretrain the model up to its compute-optimal number of tokens, as estimated by scaling laws. This is followed by a cooldown stage with an additional 20B tokens, during which the data mixture is systematically varied."

**与其他头部厂商的对照**：
- GLM-5：三阶段上下文扩展（32K→128K→200K），中训练后 200K 阶段在 128K 窗口内也提升
- Nemotron 3 Ultra：Phase 1 (15T diversity) → Phase 2 (5T quality)
- Qwen3：S1 (30T) → S2 (5T reasoning) → S3 (long-ctx)

**对用户的价值**：数据配比在 cooldown/annealing 阶段最重要，用户的中训练实验正是这个阶段。

---

## 二、实验结果

### 2.1 Benchmark 结果（Table 1, 55B-A1.2B 目标模型）

| 方法 | MMLU | ARC-C | BBH | GSM8K | MATH | HumanEval | C-Eval | 加权平均 |
|------|------|-------|-----|-------|------|-----------|-------|---------|
| Human | 0.573 | 0.535 | 0.429 | 0.652 | 0.325 | 0.354 | 0.637 | 0.496 |
| DoReMi | 0.569 | 0.529 | 0.437 | 0.674 | 0.336 | 0.450 | 0.627 | 0.514 |
| DML | 0.579 | 0.552 | 0.454 | 0.696 | 0.393 | 0.353 | 0.631 | 0.519 |
| SODM | 0.566 | 0.541 | 0.453 | 0.705 | 0.399 | 0.421 | 0.628 | 0.526 |
| **CAMEL** | 0.577 | **0.553** | **0.455** | 0.693 | 0.372 | **0.451** | **0.653** | **0.532** |

- CAMEL 加权平均最高（0.532）
- 比次优 SODM 高 +0.6%（绝对）
- 比人类设计高 +3.6%（绝对）
- 比模型大小无关方法（DoReMi）高 +1.8%
- 部分 benchmark CAMEL 不是最优（MMLU: 0.577 vs DML 0.579, MATH: 0.372 vs SODM 0.399）

### 2.2 Held-out Benchmark 泛化（Figure 8）

7 个未见 benchmark 上 CAMEL 仍然最高 → 不是过拟合到优化目标。

### 2.3 成本

- 比基线减少 **50% 混合优化成本**
- 用不到目标模型一次完整训练的成本即可找到高质量混合
- +3% downstream benchmark 性能

### 2.4 数据域

训练数据 5 个域：English, Chinese, Code, Math, Knowledge。
- **数据源完全不透明** — 论文只给定性描述（如"cleaned web & narrative corpora"、"open-source repositories"），不给具体数据集名
- 验证集需要 >200 个，但论文不列具体是哪些
- 优化用 7 个 benchmark：MMLU, ARC-C, BBH, GSM8K, MATH, HumanEval, C-Eval

---

## 三、作者与开源分析

### 3.1 作者背景

| 作者 | 单位 | 研究背景 | 备注 |
|------|------|---------|------|
| **Jingwei Li** (一作) | 清华期智研究所 (IIIS) | 控制论/RL/ML 理论 | **CAMEL 是首篇 LLM 数据论文**，之前无 LLM 训练经验 |
| **Xinran Gu** | 清华 IIIS + 上海期智研究院 | 分布式训练/数据混合理论 | 博士生 (guxr24@mails.tsinghua.edu.cn) |
| **Jingzhao Zhang** (通讯) | 清华 IIIS + 上海期智研究院 | 优化理论 | IIIS 助理教授 (jingzhaoz@tsinghua.edu.cn) |

**Jingwei Li 的 7 篇论文轨迹**（全部与 Jingzhao Zhang 合著）：
1. CAMEL (2026.3) — 首篇 LLM 数据论文
2. Finite Sample Analyses (2025.9) — 控制论
3. Nonlinear Implicit Bias (2025.5) — ML 理论
4. Black-Box MIA for Diffusion (2024.5) — 安全
5. SIPO (2023.10) — RL
6. Online Control (2023.6) — 控制论
7. Online Policy Optimization (2022.9) — RL

**关键观察**：从控制论/RL 直接跳到 55B MoE 数据配比，中间无 LLM 训练过渡。

### 3.2 期智研究所 (IIIS)

- 清华大学交叉信息研究院（IIIS），由姚期智先生创立
- 理论计算机科学研究为主
- Jingzhao Zhang 组：优化理论 + ML 理论

**关联论文 — Phase Transitions (2505.18091, NeurIPS'25 Spotlight)**：
同一组的理论前序工作（Xinran Gu + Jingzhao Zhang），研究数据混合引起的相变现象。

Phase Transitions 作者：
- Xinran Gu — 清华 IIIS + 上海期智研究院
- Kaifeng Lyu — 清华 IIIS（访问 UC Berkeley Simons Institute 期间完成）
- Jiazheng Li — 清华 College of AI + 上海期智研究院
- Jingzhao Zhang — 清华 IIIS + 上海期智研究院

**注意**：Kaifeng Lyu 和 Jiazheng Li **不在 CAMEL 论文中**。Phase Transitions 有 4 位作者，CAMEL 只有 3 位。

### 3.3 Phase Transitions vs CAMEL 对比

| 维度 | Phase Transitions | CAMEL |
|------|-------------------|-------|
| 规模 | Pythia 14M-6.9B | DeepSeek V3 MoE 590M-55B |
| 数据 | 合成传记 + FineWeb-Edu | 5 域（English/Chinese/Code/Math/Knowledge），不透明 |
| 理论 | 容量分配 + knapsack + 相变 | 容量分配 + CAMEL 定律 |
| 硬件 | 8-16 张 RTX 6000 Ada / 8 A100 | **未描述** |
| 致谢 | ✅ 有（National Key R&D Program + 上海期智） | ❌ **无致谢章节** |
| 资助声明 | ✅ 有 | ❌ **无资助声明** |
| 代码 | 未开源 | 未开源 |
| 算力来源 | 学术（明确披露） | **未知** |
| Venue | NeurIPS'25 Spotlight | ICML 2026 |

**关键差异**：同一组作者在 Phase Transitions 中明确写了致谢和硬件，CAMEL 却完全没有。这不是疏忽 — 是刻意省略。

### 3.4 产业合作疑点

**疑点汇总**：

| 疑点 | 说明 |
|------|------|
| 无致谢 | Phase Transitions 有完整致谢，CAMEL 没有 |
| 无资助声明 | Phase Transitions 有 National Key R&D Program，CAMEL 没有 |
| 无硬件描述 | Phase Transitions 描述了 GPU 型号和数量，CAMEL 没有 |
| 规模跳跃 | Phase Transitions 用 8-16 GPU，CAMEL 用 55B MoE（~1000x 算力） |
| 数据源不透明 | 只给定性描述，不给数据集名 |
| 无代码/数据/验证集列表 | 无法复现 |
| 架构选择 | 点名用 DeepSeek V3 架构（非 LLaMA/GLM/Qwen） |
| 作者无 LLM 背景 | Jingwei Li 从控制论直接跳到 55B MoE |

**可能的合作方分析**：

| 解释 | 可信度 | 理由 |
|------|--------|------|
| DeepSeek 合作 | ~50% | 架构 + 无致谢 + 数据不透明 + 55B 规模 |
| 其他中国公司 | ~20% | 也可能用 DeepSeek V3 开源架构 |
| 纯学术（国家算力） | ~20% | 重点研发计划 + IIIS 影响力 |
| 上海期智/清华自有 | ~10% | 55B MoE 规模很难 |

**注意**：以上为推断，**没有确凿证据**。作者单位确认为清华期智研究所，但 55B MoE 的算力来源不明。目前不做定论，**不将其视为 DeepSeek 工作来引用**。

### 3.5 开源状态

**完全无开源**：
- 无 GitHub 仓库
- 无 HuggingFace 模型/数据
- 无代码/数据可用性声明
- 无"代码即将发布"承诺
- 论文中数据源只给定性描述

**对比同领域开源情况**：

| 论文 | Venue | 开源 |
|------|-------|------|
| PreSelect | ICML'25 | ✅ 代码+数据+分类器 |
| Chameleon | ICML'25 | ✅ 代码 |
| DoReMi | NeurIPS'23 | ✅ 代码 |
| RegMix | ICLR'25 | ✅ 代码 |
| DML | ICLR'25 | ✅ 代码 |
| SODM | arXiv | ✅ 代码 |
| **CAMEL** | ICML'26 | ❌ |
| InfoLaw | ICML'26 | ❌ |
| FastMix | ICLR'26 | ❌ |
| ByDoRe | ICML'26 | ❌ |

**趋势**：2025 及以前基本开源，2026 开始普遍不开源。

---

## 四、可信度评估

### 4.1 理论可信度（高）

- **ICML 2026 接收**（venue 确认）
- **清华期智研究所**（知名 ML 理论组，姚期智创立）
- **理论推导完整**：Theorem 2.2 + Appendix D 证明（KKT 条件 + Assumption 2.1 求解）
- **前序理论**：Phase Transitions (NeurIPS'25 Spotlight) 提供容量分配理论基础
- **55B-A1.2B 验证**：大规模验证非 toy experiment
- **7+7 benchmark 验证**（含 held-out），非过拟合
- **与 4 个基线全面对比**（DoReMi, DML, SODM, Human-designed）

### 4.2 实验可信度（中，不可验证）

- **无代码/数据**：无法复现
- **数据源不透明**：无法使用相同数据验证
- **验证集不列表**：>200 验证集但不知道是哪些
- **+0.6% over SODM**：提升不大，可能在噪声范围内
- **部分 benchmark 不是最优**：MMLU, MATH, GSM8K
- **算力来源不明**：无法评估实验的充分性
- **作者无 LLM 经验**：Jingwei Li 首篇 LLM 论文

### 4.3 综合判断

| 维度 | 可信度 | 理由 |
|------|--------|------|
| 理论框架 | **高** | 数学证明 + ICML 评审 + 知名组 |
| 容量分配视角 | **高** | Phase Transitions 独立验证 |
| R_math ≠ R_know | **中高** | 可独立验证但未验证 |
| loss-to-benchmark 映射 | **中** | 概念合理但需 >200 验证集 |
| hourglass 采样 | **中** | 有实验支撑但规模有限 |
| 55B 实验结果 | **低** | 无法复现，算力来源不明 |
| +0.6% over SODM | **低** | 可能在噪声内 |

**整体定位**：理论可信可借鉴，实验不可全信。与 PreSelect（开源+ICML'25）相比可信度低一档。

---

## 五、与 QuaDMix 的关联分析

### 5.1 信号对比

| 维度 | QuaDMix | CAMEL |
|------|---------|-------|
| 信号 | 1M proxy val_loss（单点） | 多验证集 loss 向量 → benchmark |
| proxy→target 迁移 | 直接迁移（已知有问题） | 显式建模 M，CAMEL 定律外推 |
| benchmark 预测 | 无（用 val_loss 替代） | loss-to-benchmark logistic 映射 |
| 容量建模 | 无 | 容量分配优化问题 |
| 计算预算分配 | 均匀分配 336 实验 | hourglass 策略（两端多中间少） |
| mid-training 适用性 | 通用 | **明确针对 mid-training** |
| 域分组 | 198 显式域 | 5 内在域 |

### 5.2 CAMEL 如何解释 QuaDMix 的核心问题

#### 问题 1: val_loss 不预测推理性能
**CAMEL 的解法**：不是用单个 val_loss，而是用**多个验证集 loss 向量**通过 logistic 映射预测 benchmark。不同 benchmark 需要不同能力组合，多维信号可以捕捉。

**对 QuaDMix 的启示**：不要用单一 STEM val_loss 做信号，应该用**多个验证集**（通用、STEM、数学、代码等各自 val_loss）组合预测 benchmark。

#### 问题 2: proxy→target 迁移失败
**CAMEL 的解法**：显式建模模型大小 M。CAMEL 定律中 L_val 依赖于 M^β_i，不同域的 β_i 不同，因此小模型最优配比 ≠ 大模型最优配比。R_math ≠ R_know 证明了不同域容量分配不成比例。

**对 QuaDMix 的启示**：QuaDMix 的 1M proxy 实验结果需要通过 CAMEL 定律外推到目标模型大小，而不是直接迁移。R_math ≈ 2.4 vs R_know ≈ 4.3 表明 STEM 域在目标规模下可能比小 proxy 显示的更有价值。

#### 问题 3: 搜索空间过大（198 维）
**CAMEL 的解法**：使用**内在域**概念。k 个内在域（k << 198）通过数据组合权重 <t_i, r> 来表示。论文证明方法对 k 的选择鲁棒。

**对 QuaDMix 的启示**：198 域需要先投影到 5-10 个内在域。CAMEL 提供了理论基础 — 内在域是验证 loss 的隐变量。

### 5.3 关键限制

1. **假设 2.1 要求域间指数接近**（α_i ≈ ā + ε_i）：如果 198 域中某些域的 α_i 差异很大，CAMEL 定律可能不适用
2. **需要多个验证集**：CAMEL 的 loss-to-benchmark 映射需要 >200 个验证集来拟合。QuaDMix 目前只有 1 个 STEM 验证集
3. **5 个域 vs 198 域**：CAMEL 只用 5 个域，QuaDMix 有 198 个。需要先分组
4. **MoE 架构**：CAMEL 用 DeepSeek V3 MoE，用户模型架构可能不同
5. **无开源代码**：无法直接复现或使用

---

## 六、可借鉴的 6 个具体点

即使不将 CAMEL 视为 DeepSeek 工作，其理论框架仍有 6 个可借鉴点：

### 6.1 容量分配视角（最核心）

**CAMEL 公式**：L_val(r, M) = C + Σ K_i / (<t_i, r>^α_i · M^β_i)

**借鉴方式**：
- 不再直接优化 val_loss，而是建模不同域的容量吸收速率
- 拟合 CAMEL 公式外推到目标规模
- 解释 QuaDMix 失败的根因：1M proxy 的 M 太小，容量分配模式与目标模型不同

**可行性**：在已有 336 个 proxy 实验上拟合 CAMEL 公式，检查是否能预测最优配比。

### 6.2 R_math ≠ R_know（可独立验证）

**CAMEL 发现**：R_math ≈ 2.4, R_know ≈ 4.3

**借鉴方式**：
- 在用户自己的数据上验证：训练 3 个模型规模，测 STEM vs 通用域的 R 值
- 如果 R_STEM ≠ R_general，就证明了小 proxy 不可用
- STEM 域（特别是数学）在目标规模下可能比小 proxy 显示的更有价值

**可行性**：需要 3 个模型规模的训练数据。如果有 QuaDMix 的多规模实验结果可直接计算。

### 6.3 Loss-to-Benchmark 映射（信号设计）

**CAMEL 公式**：Acc_b(L) = C_b + A_b / (1 + exp(k_b^T L + B_b))

**借鉴方式**：
- 不再优化 val_loss 本身 → 优化 benchmark 预测值
- 用多个验证集（不只一个）→ 多维信号
- 与 PreSelect 的预测强度殊途同归：都认为 loss↔benchmark 相关性才是正确信号

**可行性**：需要在 proxy 实验上同时跑 benchmark，拟合映射关系。

### 6.4 内在域分组（198→5-10 方案）

**CAMEL 方法**：k 个内在域，t_ij = 内在域 i 在数据集 j 中的比例

**借鉴方式**：
- 198 域 → PCA/NMF 聚类到 5-10 个内在域
- 降低了优化维度（198→5-10），使搜索可行
- 不需要人工分组，数据驱动
- 论文证明方法对 k 的选择鲁棒

**可行性**：可以直接在 198 域的 loss/质量特征上做 PCA/NMF。

### 6.5 Hourglass 采样（实验设计）

**CAMEL 发现**：固定算力下，hourglass 策略（小+大模型多采样，中间少采样）误差最低

**借鉴方式**：
- 如果要训练多个小模型拟合 CAMEL 公式，用 hourglass 策略分配算力
- 最小规模（如 100M）和最大规模（如 1B）多跑实验
- 中间规模少跑

**可行性**：在做新一轮实验时直接采用此策略。

### 6.6 Cooldown 阶段定位

**CAMEL 描述**：pretrain → cooldown 20B tokens（混合比系统变化）

**借鉴方式**：
- 数据配比在 cooldown/annealing 阶段最重要
- 与 GLM-5、Nemotron、Qwen3 的中训练描述一致
- 用户的中训练实验就是 cooldown 阶段

**可行性**：直接确认当前实验定位正确。

---

## 七、与 PreSelect 的互补关系

### 7.1 方法对比

| 维度 | PreSelect | CAMEL |
|------|-----------|-------|
| **信号设计** | 预测强度（loss↔benchmark 秩相关） | loss-to-benchmark logistic 映射 |
| **粒度** | 文档级 | 域级（内在域） |
| **规模外推** | 无（单规模） | 有（M 在公式中） |
| **域分组** | 不需要 | 内在域（k << n） |
| **实验设计** | 6 模型 forward pass | hourglass 采样 |
| **阶段** | 从头预训练 | **中训练** |
| **开源** | ✅ 代码+数据+分类器 | ❌ 无 |
| **Venue** | ICML'25 | ICML'26 |
| **规模验证** | 3B/100B | 55B-A1.2B MoE |
| **proxy 模型** | 6 个 Llama (7B-65B) forward pass | 8 个 DeepSeek V3 MoE 小模型训练 |

### 7.2 互补分析

**PreSelect 的优势**：
- 开源，可直接使用
- 信号设计简单（只需 forward pass，不需训练）
- 文档级粒度更精细
- ICML'25 评审通过，已被引用验证

**CAMEL 的优势**：
- 明确针对中训练
- 有规模外推框架（CAMEL 定律）
- 有域分组方案（内在域）
- 有实验设计指导（hourglass）
- 55B 规模验证

**两者共同的核心洞察**：
- **benchmark 相关性 > val_loss** — 两者从不同角度得出同一结论
- PreSelect：用 loss↔benchmark 秩相关选数据
- CAMEL：用 loss→benchmark logistic 映射优化配比

### 7.3 综合方案草图

结合两者优势，用户的实施方案可以是：

```
步骤 1: 域分组
  198 域 → PCA/NMF → 5-10 个内在域（CAMEL 方法）

步骤 2: 信号计算
  用 4-6 个开源模型 forward pass 算每个内在域的 loss（PreSelect 方法）

步骤 3: 预测强度/映射
  算域级 loss↔STEM benchmark 相关性
  （PreSelect 信号 + CAMEL logistic 映射）

步骤 4: 规模外推
  拟合 CAMEL capacity-aware law，外推到目标规模（CAMEL 框架）

步骤 5: 优化
  优化 loss-to-benchmark 映射，得到最优域配比
```

**成本估算**：
- 4-6 个开源模型 forward pass：~25 H100 小时
- 3 个规模 × 5-10 个配比实验：~100-200 GPU 小时
- 总计：远低于 QuaDMix 的 336 实验

---

## 八、行动建议

### 8.1 立即可做（零成本）

1. **验证 R_math ≠ R_know**：在 QuaDMix 已有的多规模 proxy 实验上，计算各域的 R 值。如果 R_STEM ≠ R_general，直接证明小 proxy 不可用

2. **198 域 → 内在域分组**：在 198 域的 loss/质量特征上做 PCA/NMF，检查能否降到 5-10 个内在域

3. **拟合 CAMEL 公式**：将 336 个实验的 (r, M_proxy, L_val) 数据用 CAMEL 公式拟合，外推到 M_target，看预测的最优 r* 是否不同于 LightGBM 回归的结果

### 8.2 短期可做（低成本）

4. **收集多个验证集 loss**：增加通用、数学、代码等验证集，构成多维 loss 向量

5. **拟合 loss-to-benchmark 映射**：在已有 proxy 实验上，如果有 benchmark 结果，拟合 logistic 映射

6. **域级预测强度**：用 PreSelect 方法（4-6 个开源模型 forward pass）算域级 loss↔benchmark 秩相关

### 8.3 中期可做（需算力）

7. **Hourglass 采样实验**：训练 3 个模型规模（小/中/大）× 5-10 个配比，用 hourglass 策略分配算力

8. **CAMEL 定律拟合 + 外推**：用步骤 7 的数据拟合 CAMEL 公式，外推到目标规模

---

## 九、总结

CAMEL 是目前索引中**最直接对应用户场景**的论文：

1. **明确 mid-training** — 与用户完全一致
2. **loss-to-benchmark 映射** — 直击 val_loss→benchmark gap 痛点
3. **容量分配解释 proxy 失效** — R_math ≠ R_know，不同域容量分配不成比例
4. **55B-A1.2B 验证** — 最大规模验证
5. **ICML 2026** — 高可信度

**但需注意**：
- 无开源代码/数据，实验不可复现
- 算力来源不明，55B 实验无法独立验证
- +0.6% over SODM 可能在噪声内
- 理论可借鉴，实验不可全信

**与 PreSelect 互补**：PreSelect 提供开源信号设计，CAMEL 提供理论框架和规模外推。两者结合是用户中训练数据配比问题的最有前景方案。

**最高 ROI 行动**：
1. 在 QuaDMix 现有 336 个 proxy 实验上拟合 CAMEL 定律
2. 收集多个验证集 loss，拟合 loss-to-benchmark 映射
3. 检查 R_STEM ≠ R_general，验证容量分配不成比例
4. 198 域 → PCA/NMF → 5-10 个内在域
