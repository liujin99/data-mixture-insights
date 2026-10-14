# MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes

**论文**: arXiv:2509.24945v3 (ICLR 2026)
**作者**: Changsheng Zhao, Ernie Chang, Zechun Liu, Chia-Jung Chang, Wei Wen, Chen Lai, Rick Cao, Yuandong Tian, Raghuraman Krishnamoorthi, Yangyang Shi, Vikas Chandra
**机构**: Meta AI (FAIR)
**代码**: https://github.com/facebookresearch/MobileLLM-R1
**模型**: https://huggingface.co/collections/facebook/mobilellm-r1
**状态**: ICLR 2026 接收（v3 终稿，2026.02.27 提交）

---

## 一句话总结

**sub-billion 模型 + ~2T 精选数据（4.2T 含重采样）即可涌现推理能力**，用 Qwen3 36T 语料的 11.7% 达到可比性能。核心方法是 cross-capability self-influence 驱动的数据配比——预训练用 AutoMixer 离线算影响分数定配比，中训练用模型-数据协同演化（influence>0 过滤 + 自适应再配比），影响分数收敛到 0 = 数据榨干。

---

## 0. 训练管线全貌

```
随机初始化
    │
    ▼
预训练 Phase 1 (2T tokens) ──── 广覆盖：FineWeb-Edu 63.75% + 种子推理数据
    │
    ▼
预训练 Phase 2 (2T tokens) ──── 向推理倾斜：Math 23.33% + FineWeb-Edu 54.83%
    │                         （从 1.8T 唯一数据中重采样到 4T）
    ▼
MobileLLM-R1-base (140M / 360M / 950M)
    │
    ▼
中训练 Phase 1 (100B tokens) ── 广知识：DCLM 37% + Code 20.7% + Math 11.6%
    │                         （Dolmino 语料 + influence 过滤 + Llama-3.1-8B 蒸馏）
    ▼
中训练 Phase 2 (100B tokens) ── 向推理倾斜：Math 29.1% + Code 29.1% + CC-Math 19.4%
    │
    ▼
Post-training
    ├── General SFT (866k, Tulu3-SFT, 2 epoch)
    └── Reasoning SFT (6.2M, OpenMath+OpenScience+OpenCode, 4 epoch)
    │
    ▼
MobileLLM-R1 (最终模型)
```

**三阶段设计哲学**：预训练种下推理潜力 → 中训练压缩知识 + 域强化 → 后训练激发推理。每一阶段都有 P1(广覆盖)→P2(向目标域倾斜) 的两阶段配比演变。

---

## 1. 模型架构

| 模型 | Layers | Attn Heads | KV Heads | Dim | Hidden Dim | Params |
|---|---|---|---|---|---|---|
| MobileLLM-R1-140M | 15 | 9 | 3 | 576 | 2048 | 140.2M |
| MobileLLM-R1-360M | 15 | 16 | 4 | 1024 | 4096 | 359.4M |
| MobileLLM-R1-950M | 22 | 24 | 6 | 1536 | 6144 | 949.2M |

基于 MobileLLM + LLaMA3.2 架构：128k 词表、QK-Norm（稳定小模型训练）、input/output embedding 权重共享（参数效率）。Base 模型上下文 4k，后训练后扩展到 32k。

---

## 2. 预训练配比方法论（§2，核心贡献）

### 2.1 LOO 分析：哪些数据源真正有用

**方法**：每次剔除一个数据集训练 baseline，测量 Code/Math/Knowledge 三类 capability-probing 集上的 NLL 变化（Eq.1）。

**capability-probing 集构建**（Hierarchical Rejection Sampling）：
1. FineWeb-Edu 分类器筛选（score > 4）
2. Ask-LLM 打分，取 top 10%（判断推理相关性）
3. 语义去重 → 每个数据集约 10,000 条代表性样本

**三个域的分类**（单数据集可属多域）：
- Code (C): StarCoder, StackExchange, Nemotron-Code, Cosmopedia, Natural Reasoning, peS2o
- Math (M): OpenWebMath, FineMath, Algebraic Stack, Nemotron-Math, Cosmopedia, Natural Reasoning, peS2o
- Knowledge (K): FineWeb-Edu, Wikipedia, Arxiv, Cosmopedia, Nemotron-Science, Natural Reasoning, peS2o

**LOO 关键发现（Fig.3）**：

| 发现 | 细节 |
|---|---|
| **FineWeb-Edu 是跨域"胶水"** | 剔除后所有能力（含 code、math）均最大幅度下降——web 数据连接异构域 |
| **StarCoder→数学 > OpenWebMath→代码** | **反转常识**：此前普遍认为数学数据对代码贡献大（Lewkowycz 2022），实际反过来 |
| 域专用数据主要利好本域 | StarCoder 主利好 code（及 math），数学语料主利好 math |
| Wikipedia 对 math/code 贡献小 | 但作为结构化事实知识源仍不可缺 |

### 2.2 Cross-Capability Self-Influence 配比（AutoMixer 框架）

**影响分数定义**（Eq.2）：

$$I(x_i, x_{\text{test}}; \theta) = -\nabla_\theta L(x_{\text{test}}; \theta^*)^\top H^{-1}_{\theta^*} \nabla_\theta L(x_i; \theta^*)$$

直接算 Hessian 逆不可行，AutoMixer (Chang et al., ACL 2025) 提供高效近似。

**Self-influence vs Cross-influence**：
- Self：训练样本和 probing 集同域（如 StarCoder 样本 → Code probing 集）
- Cross：训练样本和 probing 集异域（如 StarCoder 样本 → Math probing 集）

**多检查点加权**：T=10 个均匀间隔 checkpoint，线性递增权重 $\alpha_{c,t} \propto t$（强调后期训练）。

**联合影响分数**（Eq.4）：

$$I_{\text{joint}}(x_i) = \sum_{c \in \{C,M,K\}} \sum_{t=1}^{T} \alpha_{c,t} \cdot I(x_i; \theta_{c,t})$$

**数据集级采样权重**（Eq.5）：

$$w_g = \frac{\rho_g}{\sum_{g'} \rho_{g'}}, \quad \rho_g = \frac{1}{N_g} \sum_{x_i \in g} I_{\text{joint}}(x_i) \cdot s_i$$

其中 $N_g$ 为数据集 token 数，$s_i$ 为样本长度。**长样本的 influence 被长度加权**（因为训练成本更高）。

**结果（Fig.4）**：influence-derived mixture 在 Code、Math、Knowledge 三域 ppl 上**全面优于 uniform sampling**，且全程未接触任何 benchmark 测试集（benchmark-free）。

### 2.3 预训练配比表（Table 5）

| 数据集 | 唯一 tokens (B) | Phase 1 Mix | Phase 2 Mix |
|---|---|---|---|
| FineWeb-Edu | 1300 | **63.75%** | **54.83%** |
| StarCoder | 263.8 | 10.66% | 0.52% |
| Arxiv | 28 | 6.36% | 1.32% |
| OpenWebMath | 12.6 | 6.93% | **23.33%** |
| Wiki | 3.7 | 5.03% | 0.14% |
| StackExchange | 19.6 | 5.03% | 0.86% |
| Algebraic stack | 12.6 | 2.25% | 1.26% |
| FineMath | 34 | — | **8.01%** |
| Facebook natural reasoning | 1.8 | — | 3.18% |
| Nemotron math | 15 | — | 3.01% |
| Cosmopedia | 25 | — | 2.70% |
| Nemotron code | 16 | — | 0.72% |
| Algebraic stack | 12.6 | — | 1.26% |
| Nemotron science | 2 | — | 0.03% |
| peS2o | 50 | — | 0.08% |
| **总唯一 tokens** | **~1.8T** | | |

**P1→P2 配比演变模式**：
- Math 类（OpenWebMath + FineMath + Nemotron math）：6.93% → **34.35%**（含 FineMath 8.01%）
- Code 类（StarCoder + Nemotron code）：10.66% → 1.24%
- Web/Edu（FineWeb-Edu）：63.75% → 54.83%
- 从 1.8T 唯一数据重采样到 4T（Phase 1 + Phase 2 各 2T），意味着**高质量数据被重复 ~2.3×**

---

## 3. 中训练：知识压缩（§3，与本项目最相关）

### 3.1 Data-Model 协同演化

中训练是预训练和后训练之间的桥梁：**压缩预训练获得的知识，最大化目标任务表现**。每阶段 100B tokens。

**与预训练配比的关键区别**：预训练用 AutoMixer 离线算 influence（固定配比），中训练用**模型自身**在线算 influence（配比随训练演化）。

**三步迭代**：

1. **Sample-level rejection sampling**（Eq.6）：只保留 influence > 0 的样本
   $$D_t = \{x_i \in D_{\text{raw}} : I(x_i; \theta_t) > 0\}$$
   模型自过滤：正贡献样本保留，零/负贡献丢弃。

2. **Dataset-level adaptive mixing**：聚合 influence 到数据集级，按 Eq.4-5 重新计算配比

3. **训练 + 重复**（Eq.7）：$\theta_{t+1} = \text{MidTrain}(\theta_t, D_t)$，新模型提供更精细的 influence 估计

### 3.2 影响分数收敛 = 数据耗尽信号

**Fig.5 核心发现**：
- Phase 1：influence 分数分布宽，正负都有（模型欠训练，判断力弱）
- Phase 2：分布**压缩向零**——极端贡献消失，大部分样本影响趋于零
- **收敛到零 = 数据集信息已被充分利用，无法再提供增益**

> "As training progresses, the influence scores of data samples increasingly concentrate around zero or negative values, indicating near-complete utilization of the informative content in the dataset and convergence of the process."

**实操结论**：两个阶段即可收敛（"two stages suffice"），无需更多轮。

### 3.3 子采样数据 > 原始数据

**Fig.6**：influence 过滤后的子采样数据在 MMLU 上**全程优于原始数据**，且更稳定（原始数据在 30k 步附近有明显性能下探）。无论是否加 Llama-3.1-8B 蒸馏，子采样都更优。

### 3.4 中训练配比表（Table 6，post sub-sampling）

| 数据集 | Phase 1 Mix | Phase 2 Mix |
|---|---|---|
| Dolmino DCLM Baseline | **37.03%** | 6.51% |
| Nemotron-Pretraining-Code-v1 | 20.69% | **29.10%** |
| Dolmino Math | 11.63% | **29.10%** |
| peS2o | 11.41% | 2.01% |
| StarCoder | 6.90% | 9.70% |
| FLAN | 4.10% | 0.72% |
| Nemotron-CC-Math-v1 | 3.45% | **19.40%** |
| Wiki | 2.66% | 0.47% |
| StackExchange | 2.12% | 2.00% |
| Benchmark Set (train splits) | 0.97% | (余量) |

**P1→P2 配比演变**：
- Math 类（Dolmino Math + CC-Math）：15.08% → **48.50%**
- Code 类（Nemotron Code + StarCoder）：27.59% → 38.80%
- 通用知识（DCLM + FLAN + peS2o + Wiki）：55.20% → 9.71%
- **Benchmark train splits 直接加入**（GSM8K/ARC/BoolQ 等 train 集）——aligning training with downstream evaluation

### 3.5 中训练技术细节

- **知识蒸馏**：Llama-3.1-8B-Instruct 作为 teacher，KL divergence on logits
- 优化器：Adam，LR 线性衰减到零，peak 3.6e-4
- BS=4, seq=4k, 50k steps, 16×8 GPUs, 1-2 天/阶段
- 语料基于 **Dolmino**（OLMo 2 的中训练语料）+ 额外 math/code 增强

---

## 4. 后训练发现

### 4.1 分阶段 SFT > 混合训练（Table 1）

| 策略 | MATH | GSM8K | LCBv6 | MMLU |
|---|---|---|---|---|
| 无 Tulu，直接 M+C+S | 56.2 | 68.2 | 13.1 | 44.0 |
| 有 Tulu，再 M+C+S | **57.8** | **68.5** | **13.7** | 43.7 |
| Tulu + (M+C+S) 混合 | 56.2 | 53.1 | 14.9 | 44.0 |

**先对齐再上推理数据**（staged）全面优于混合训练，尤其在 math 和 general reasoning 上。

### 4.2 RL 对小模型无益（§D.2）

- Base + RL (GRPO)：GSM8K 57.0
- SFT：GSM8K **74.0**
- SFT + 额外 RL：无显著提升

**结论**：小模型容量有限，SFT 提供的结构化监督 > RL 的噪声自探索。大模型的"RL 探索 → 蒸馏小模型"范式更有效。

### 4.3 Science 推理数据有跨域迁移

OpenScienceReasoning-2 不仅提升科学基准，还迁移到 math 和 code（Table 1）。但引入 math/code 数据会降低 MMLU（小模型容量有限，域间竞争）。

---

## 5. 附录关键发现

### 5.1 RankMe：预训练表示质量的低成本代理（§D.1）

**问题**：预训练 MMLU 无法预测中训练后性能（pre-training MMLU 几乎不随 LR 变化，但 mid-training 后差异巨大）。

**RankMe**（Eq.8，源自视觉自监督）：

$$\text{RankMe}(Z) = \exp\left(-\sum_{k=1}^{\min(N,K)} p_k \log p_k\right), \quad p_k = \frac{\sigma_k(Z)}{\sum_i \sigma_i(Z)} + \epsilon$$

其中 $\sigma_k(Z)$ 为 output embedding 的奇异值。

| Learning Rate | Pre-train MMLU | RankMe | **Post Mid-train MMLU** |
|---|---|---|---|
| 4e-3 | 27.92 | **21.98** | **36.31** |
| 2e-3 | 27.67 | 20.19 | 33.95 |
| 1e-3 | 26.87 | 14.22 | 29.02 |
| 4e-4 | 27.50 | 7.81 | 27.55 |

**RankMe 与下游性能强正相关**，高 LR → 高表示秩 → 更多参数被有效利用 → 更强泛化。**预训练 loss 不预测下游，但表示结构（秩）预测。**

### 5.2 计算开销（§D.3）

| 环节 | GPU hours |
|---|---|
| 预训练影响分数计算 | ~5,100 |
| 中训练影响分数计算 | ~1,100 |
| Hierarchical Rejection Sampling | ~200 |
| LOO ablation | ~400（每数据集 ~50） |
| **数据策展总计** | **~6,800** |
| 预训练 | ~28,000 |
| 中训练 | ~10,000 |
| 后训练 | ~600 |
| **训练总计** | **~38,600** |

**数据策展 = 总成本的 ~15%**。Influence 计算（6,200 GPU hours）≈ 训练的 16%。

---

## 6. 与本项目的系统关联

### 6.1 场景匹配度

| 维度 | 用户场景 | MobileLLM-R1 | 匹配 |
|---|---|---|---|
| 模型规模 | 1.5B | 950M (22 层) | ✅ 最接近的公开报告 |
| 中训练 token | 1.8B | 100B×2 = 200B | ⚠️ 差 100×（用户预算更紧） |
| 中训练性质 | continual pre-train（ClimbLab 基座上） | 预训练后中训练 | ✅ 同构 |
| 目标域 | STEM accuracy | math/code/knowledge | ✅ 直接对应 |
| 可复现 | 需要 | 全开源（recipe+配比+代码） | ✅ |
| 评测 | STEM accuracy | MATH500/GSM8K/AIME | ✅ |

### 6.2 对用户场景的可迁移洞察

**① P1→P2 配比演变模式（最高价值）**

R1 在预训练和中训练中都采用"P1 广覆盖 → P2 向目标域倾斜"：
- 预训练 Math：6.93% → 34.35%（含 FineMath）
- 中训练 Math+Code：42.67% → 87.30%

**对用户的启示**：1.8B tokens 中训练如果分两段，前段含通用数据，后段向 STEM 大幅倾斜。用户当前 r=0.7（70% STEM）可视为"P2 风格"的单段配比——若有机会分段，前段可降至 ~40% STEM，后段升至 70%+。

**② Influence→0 收敛判据（操作价值最高）**

R1 的核心可操作发现：**influence 分数分布收敛到零 = 数据被榨干**。

**⚠️ Confound 警告（用户提出）**：influence→0 的信号受两个混淆因素污染，R1 未做任何控制实验剥离：

1. **学习率衰减**：R1 中训练两阶段 LR 均线性衰减到零（Table 4）。LR→0 → 参数更新→0 → 训练梯度 $\nabla_\theta L(x_i) \to 0$ → influence（梯度内积）机械趋零。这是 LR schedule 的必然结果，与数据是否"榨干"无关。
2. **正常 loss 收敛**：即使 LR 不衰减，模型在任何数据集上训练足够久，loss 收敛 → 梯度自然缩小 → influence→0。这是训练收敛的必然副产物，不是数据特有的"耗尽"。

要区分"数据耗尽"与"收敛假象"需要：固定 LR（不衰减）看 influence 是否仍→0；或固定模型状态算 influence（不随训练移动）。R1 两个控制实验都没做。Fig.5 的"分布压缩向零"完全可用 LR 衰减 + 模型收敛解释。

**对退化判据的影响**：val loss 斜率停滞有同样的 confound——停滞可能是 LR 衰减或模型收敛导致的，不一定是数据耗尽。但实操上影响不大：不管停滞的原因是什么，"停滞了就切换 P1→P2 或停训"都是合理决策。只是不能把停滞**解读为**"这批数据没用了"（可能是"这个 LR 没用了"或"模型到极限了"）。

**对用户的退化应用**（N=1-2 约束下无法做完整 co-evolution）：
- 训练中定期测量 STEM val loss 的下降速率
- loss 下降停滞 ≈ R1 的 influence→0 ≈ 该配比/数据源已耗尽（**含 confound**：也可能是 LR 衰减或模型收敛）
- 可作为**何时切换 P1→P2 的触发器**或 early-stopping 信号
- 不需要计算 influence score，只需监控 val loss 曲线斜率

**③ StarCoder→数学迁移（配比直觉修正）**

R1 LOO 实证：StarCoder 对数学的收益 > OpenWebMath 对代码的收益。

**对用户的启示**：STEM 配比中不应只看数学语料，**代码数据可能是被低估的 STEM 助推剂**。若 STEM 数据量不足（1.26B math 仅参考线 1%），可用代码数据部分替代——R1 的 P2 中训练里 Code 占 38.80%，Math 占 48.50%，Code 几乎与 Math 等量。

**④ 唯一数据 1.8T → 重采样到 4T（重复 ~2.3×）**

R1 从 1.8T 唯一数据重采样到 4T 训练 tokens，高质量数据被重复约 2.3×。这与 #71 DomainRepetition 的"loss 低的域可多重复"一致。

**对用户的启示**：1.8B tokens 中训练如果 STEM 数据池 < 1.26B，重复是合理的（R1 重复 2.3× 仍有效）。但需监控是否过相变阈值（Phase Transitions #01：r=0.7 ≫ r_thres≈0.06-0.08，在有效区）。

**⑤ RankMe 作为预训练质量代理**

R1 发现预训练 loss/MMLU 不预测中训练后性能，但 RankMe（表示秩）强相关。

**对用户的启示**：ClimbLab 1.2T 基座的中训练效果好不好，不能只看预训练 loss——可计算基座模型的 embedding 秩（RankMe），作为中训练潜力的低成本诊断。

**⑥ Benchmark train splits 直接加入中训练**

R1 在中训练 P2 直接加入 GSM8K/ARC/BoolQ 等 train splits（占 0.97%），"aligning training with downstream evaluation"。

**对用户的启示**：若 STEM 评测集有 train split，可直接以小比例（~1%）混入中训练——R1 的实证表明这有效且不过拟合。

### 6.3 不可直接迁移的部分

| R1 方法 | 为什么不能直接搬 | 退化方案 |
|---|---|---|
| AutoMixer 在线 influence（预训练） | 需要 10 个 checkpoint + 3 个域专用模型 → ~5,100 GPU hours | 用 LOO 静态配比替代 |
| 中训练 co-evolution（多轮迭代） | 需要 2+ 轮中训练 → 与 N=1-2 冲突；且 influence→0 受 LR 衰减+loss 收敛 confound | 用 val loss 斜率作为单次退化判据（**含 confound**：停滞也可能是 LR 衰减/模型收敛导致） |
| Llama-3.1-8B 蒸馏 | 需要 teacher 模型 + KL divergence 计算 | 若有 teacher 可选，否则跳过 |
| 200B 中训练 | 用户仅 1.8B（差 100×） | R1 的配比比例仍可参考，但收敛行为不同 |

### 6.4 与已有论文的关联

| 论文 | 关联 |
|---|---|
| **MAI-Thinking-1 (#73)** | 同为"influence/utility 驱动配比"路线；MAI 是 35B/1T 规模（尺度外推风险大），R1 是 sub-B（尺度无缝）。MAI 的 rank non-invariance 与 R1 的"checkpoint 加权"（后期更重要）呼应 |
| **Nemotron-CLIMB (#74)** | 用户的 ClimbLab 1.2T 基座来源；R1 中训练也用 Nemotron-CC-Math/Code，数据生态交叉 |
| **OLMo 2 (#75)** | R1 中训练语料基于 Dolmino（OLMo 2 的中训练语料），直接复用 |
| **Phase Transitions (#01)** | R1 P1→P2 向目标域倾斜，与相变理论一致（先过阈值再线性增长）；用户 r=0.7 ≫ r_thres |
| **DomainRepetition (#71)** | R1 从 1.8T 重复到 4T（2.3×），loss 低的域多重复——与 #71 的 loss-correlated 规则一致 |
| **AutoMixer (ACL 2025)** | R1 的核心方法来源（同作者 Ernie Chang） |
| **Tulu 3 (#81)** | R1 后训练 General SFT 用 Tulu3-SFT |

---

## 7. 局限性

1. **尺度差距**：中训练 200B vs 用户 1.8B（差 100×）。R1 的两阶段收敛行为在 1.8B 下是否成立未知——数据量太少可能根本无法收敛到 influence→0
2. **从零预训练 vs continual pre-train**：R1 是从随机初始化预训练后中训练；用户基座已在 ClimbLab 1.2T 上充分预训练（TPP=800），中训练起点不同
3. **Influence 计算成本**：~6,200 GPU hours（预训练 5,100 + 中训练 1,100），占训练成本 16%——用户 1-2 runs 预算下不可承受完整流程
4. **域分类粒度粗**：仅 C/M/K 三域，用户 STEM 场景可能需要更细（STEM 内部 math/physics/bio 交互不同）
5. **蒸馏依赖**：中训练用 Llama-3.1-8B 蒸馏，效果中蒸馏馏贡献占比未消融（无法区分 influence 过滤 vs 蒸馏各自贡献）
6. **无相变建模**：influence→0 收敛与 Phase Transitions 的关系未分析。influence 高可能因为近相变（数据量不够），也可能因为域本身难——两种情况最优策略不同
7. **后训练驱动**：最终推理性能大量依赖 SFT 数据（6.2M reasoning samples），预训练+中训练的"推理潜力"只有在高质量 SFT 后才兑现
8. **influence→0 的 confound（用户指出）**：influence score 本质是梯度内积（Eq.2），两个梯度都趋零时乘积必然趋零。R1 中训练 LR 线性衰减到零 → 参数更新→0 → 梯度→0 → influence→0 是 LR schedule 的机械结果，非数据特异耗尽。且即使 LR 不衰减，loss 收敛也使梯度→0。R1 缺关键控制实验（固定 LR / 固定模型状态算 influence），无法区分"数据耗尽"vs"收敛假象"。Fig.5 的"分布压缩向零"完全可用 LR 衰减 + 模型收敛解释。**这使 influence→0 作为"数据耗尽信号"的可信度从"操作判据"降级为"必要非充分信号"**——停滞了可能是数据没了，也可能是 LR/模型到极限了

---

## 8. 总结评估

| 维度 | 评分 | 说明 |
|---|---|---|
| 理论深度 | ★★★★☆ | influence-based 配比有完整数学框架（Eq.2-7），co-evolution 有收敛性分析 |
| 实验质量 | ★★★★★ | ICLR'26 接收，LOO ablation + influence 收敛实验 + RankMe 诊断，实验设计严谨 |
| 可信度 | ★★★★★ | Meta AI FAIR 团队，全开源（代码+权重+配比+数据源），ICLR peer-reviewed |
| 与配比项目相关性 | ★★★★★ | 83 篇中与用户 1.5B/1.8B 中训练场景匹配度最高的报告；尺度无缝、阶段同构 |
| 可操作性 | ★★★☆☆ | 方法主体不可直接搬（需多轮训练+influence 计算）；但 LOO 发现、配比比例、收敛判据可退化为启发式 |
| 风险 | 中 | 配比比例和 P1→P2 演变模式可直接参考；influence co-evolution 需降级使用 |

**核心价值**：本报告提供了一套**完整的 influence-driven 数据配比方法论的全流程实现**（预训练 AutoMixer + 中训练 co-evolution + 后训练 staged SFT），且在 sub-billion 尺度上验证——是 83 篇中**唯一与用户场景（1.5B 中训练）尺度匹配且方法论完整的报告**。

**对用户场景的诚实评估**：R1 的方法主体（在线 influence 迭代）与 N=1-2 黑箱约束冲突，无法直接复刻。但其**退化产物**极具价值：
1. P1→P2 配比演变模式（广覆盖→目标域倾斜）——**可直接用于 1.8B 的段内配比设计**
2. StarCoder→数学迁移——**修正 STEM 配比直觉，代码数据可部分替代数学语料**
3. influence→0 / val loss 斜率停滞作为数据耗尽信号——**1-2 runs 下的单次退化判据**（⚠️ 含 LR 衰减 + 收敛 confound，是必要非充分信号）
4. 中训练配比表（P2: Math 48.5% + Code 38.8%）——**直接作为 1.8B 配比的参照系**
5. RankMe 作为基座质量诊断——**预训练 loss 不够，表示秩更准**

**优先级**：深读第一优先。超过 MAI-Thinking-1（35B 尺度外推风险大），是本项目 83 篇中对用户场景**参考价值最高的单篇报告**。
