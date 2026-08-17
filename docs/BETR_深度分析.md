# BETR: Benchmark-Targeted Ranking 深度分析

**分析日期**: 2026-08-17

---

## 1. 论文概览

| 维度 | 信息 |
|------|------|
| 标题 | Language Models Improve When Pretraining Data Matches Target Tasks |
| arXiv | 2507.12466 (2025.07) |
| 机构 | **Apple** (10 人团队) |
| 作者 | David Mizrahi, Anders Boesen Lindbo Larsen, Jesse Allardice, Suzie Petryk, Yuri Gorokhov, Jeffrey Li*, Alex Fang, Josh Gardner†, Tom Gunter, Afshin Dehghan |
| Venue | preprint（2025.07 提交，尚未见 venue 标注） |
| 代码 | **未开源**（FastText + Arctic-Embed 均为开源组件，可复现） |
| 引用 | **17 篇**（Semantic Scholar） |
| 与 QuaDMix 相关性 | **中** — 文档级选择 vs 域级配比，互补而非替代 |

### 作者背景

| 作者 | 机构 | 备注 |
|------|------|------|
| David Mizrahi (一作) | Apple | — |
| Josh Gardner | Apple → **Anthropic** | "now at Anthropic" |
| Jeffrey Li | UW (实习 at Apple) | — |
| Alex Fang | — | DCLM 核心成员 |
| Afshin Dehghan (通讯) | Apple | 计算机视觉背景 |

### 核心引用关系

- **被 Phase Transitions 论文引用**（NeurIPS'25 Spotlight, #01）— 数据配比相变论文将 BETR 作为 baseline
- **被 NAG 论文引用**（ByteDance, #23）— 作为主要 target-oriented baseline 对比
- 引用 RegMix、DoReMi 作为 related work（group-level 方法对比）

---

## 2. 核心问题

**Target-oriented document selection**：给定目标 benchmark，从大规模预训练数据池中选择最有助于提升该 benchmark 性能的文档。

### 与域级配比的区别

| 维度 | BETR | QuaDMix / ClimbMix |
|------|------|---------------------|
| 粒度 | **文档级**（每篇文档打分） | 域级（每类数据一个权重） |
| 输出 | 排序+过滤（top X%） | 混合权重（α₁:α₂:...:αₖ） |
| 信号 | benchmark embedding 相似度 | proxy val_loss / benchmark accuracy |
| proxy 需要 | **不需要** | 需要（QuaDMix 1M / ClimbMix d10 70M） |

论文明确自我定位：
> "Group-level methods optimize mixture weights directly but only at coarse granularity, while document-level methods usually rely on proxies refined through benchmark feedback. BETR explores a more direct path by explicitly aligning pretraining documents with benchmark training examples."

**关键区别**：BETR 解决的是"哪些文档好"，QuaDMix/ClimbMix 解决的是"各域比例多少"。两者层次不同，可以叠加使用。

---

## 3. 方法

### 3.1 三步流程（极简）

```
Step 1: Embed — 10M 预训练文档 + benchmark 训练集 → Arctic-Embed L 2.0 共享空间
Step 2: Score  — 文档按 benchmark 相似度排序（max rank aggregation）
Step 3: Filter — 训 FastText 二分类（top 10% vs bottom 90%）→ 全量打分 → 筛 top X%
```

### 3.2 Embedding

| 设计选择 | 取值 | 理由 |
|----------|------|------|
| 采样文档数 | **10M**（~0.1% 数据池） | 捕获 web 文本多样性，计算可控 |
| Target 粒度 | **每条 benchmark 样本独立 embed** | 实验显示优于 centroid（Section 5.5） |
| Embedding 模型 | **Arctic-Embed L 2.0** | BERT-like encoder，8192 max length |
| 备选模型 | GTE Large v1.5, GTE-Qwen2 1.5B | 消融显示 embedding 选择影响极小 |

### 3.3 Scoring（Rank-based）

每条 benchmark 样本对所有文档排序，文档得分取所有 benchmark 样本中的**最佳排名**（max aggregation）：

$$S_j = \max_i \{v(r_{ij})\}$$

其中 $r_{ij}$ 是文档 $j$ 在 benchmark 样本 $i$ 排序中的排名，$v(r)$ 是 rank-to-value 函数。

**为什么用 rank 不用 cosine similarity**：
- rank 更鲁棒（不依赖绝对相似度值）
- 论文测试了 mean/max + log₂(1/r)/1/r 等组合，**max + best rank 最优**
- 即使 top-ranked 文档与 benchmark 的 cosine 也仅 ~0.5（web text 与 benchmark 差异大）

### 3.4 FastText Classifier

| 参数 | 值 |
|------|-----|
| 任务 | 二分类：top 10% (positive) vs bottom 90% (negative) |
| 正样本 | BETR score 排名 top 10% 的 10M 采样文档 |
| 负样本 | bottom 90%，下采样平衡类别 |
| 验证准确率 | **75.6%** |
| 替代方案 | 302M LM classifier (82.4% acc) — **反而更差** |

**反直觉发现**：FastText（75.6% 准确率）匹配或超过 302M LM classifier（82.4% 准确率）。论文结论："classification accuracy is not the key metric for effective data selection" — 这与 DCLM 的发现一致（简单 FastText 击败复杂方法）。

### 3.5 Targeting 策略

| 策略 | 说明 | 评估方式 |
|------|------|---------|
| **Evaluation-Aware (EA)** | 直接用评估 benchmark 的训练集做 target | 评估同 benchmark test set |
| **Evaluation-Blind (EB)** | 用 39 个与评估集不重叠的 diverse benchmark 做 target | 评估 held-out benchmark |

EB 策略验证泛化性：target 和 evaluate 用完全不同的 benchmark，仍优于 baseline。

---

## 4. 实验设置

### 4.1 规模

| 维度 | 数值 |
|------|------|
| 模型数 | **500+** |
| FLOPs 范围 | 10¹⁹ ~ 10²² |
| 模型大小 | 7B (主要), scaling law 53 models/dataset |
| 训练量 | 140B (7B-1x) / 1.4T (7B-10x) |
| 数据池 | DCLM-RefinedWeb (24T), Nemotron-CC (6.3T) |
| Scaling law | 每数据集 53 模型，bootstrap 量化不确定性 |

### 4.2 Core Benchmarks (10 个)

| Benchmark | 类型 | STEM 相关 |
|-----------|------|-----------|
| MMLU | 知识 | ✅ (含物理/化学/生物/数学) |
| ARC-Easy | 科学推理 | ✅ |
| ARC-Challenge | 科学推理 | ✅ |
| SciQ | 科学知识 | ✅ |
| PIQA | 常识推理 | — |
| HellaSwag | 常识完成 | — |
| Lambada OpenAI | 语言建模 | — |
| TriviaQA | 事实知识 | — |
| WebQuestions | 事实知识 | — |
| Winogrande | 常识推理 | — |

### 4.3 Noncore Benchmarks

39 个 diverse benchmark（与 Core 不重叠），用于 EB 策略的泛化验证。

### 4.4 训练阶段

**⚠️ 仅预训练（from scratch），非 continual pre-training**

论文明确声明：
> "We do not verify if pretraining improvements persist after post-training."

---

## 5. 结果

### 5.1 主结果：Compute Multiplier

| 对比 | DCLM-RefinedWeb | Nemotron-CC |
|------|----------------|-------------|
| BETR vs No Filter | **4.7x** | **4.7x** |
| BETR vs DCLM-Baseline | **1.8x** | **2.5x** |
| BETR vs Nemotron-CC HQ | — | **2.8x** |
| BETR Target-Noncore vs Baseline | 1.5x | 2.1x |

> Compute multiplier X = BETR 只需 1/X 的计算量达到 baseline 同等性能。

### 5.2 STEM 基准详细（7B-10x, Nemotron-CC, Target-Core）

| Benchmark | No Filter | DCLM-Baseline | BETR Target-Core | vs No Filter | vs Baseline |
|-----------|-----------|---------------|-------------------|-------------|------------|
| **ARC-C** | 51.5 | 53.5 | **54.7** | +3.2 | +1.2 |
| **ARC-E** | 82.2 | 82.4 | **84.0** | +1.8 | +1.6 |
| **SciQ** | 95.8 | 97.0 | 96.4 | +0.6 | -0.6 |
| **MMLU** | 60.8 | 67.3 | **68.3** | **+7.5** | +1.0 |
| PIQA | 80.9 | 80.4 | **83.0** | +2.1 | +2.6 |
| HellaSwag | 60.9 | 60.8 | **64.1** | +3.2 | +3.3 |
| Lambada | 71.5 | 72.3 | **74.9** | +3.4 | +2.6 |
| TriviaQA | 45.3 | 44.2 | **50.2** | +4.9 | +6.0 |
| WebQuestions | 27.6 | 27.5 | 25.5 | -2.1 | -2.0 |
| Winogrande | 72.1 | 74.7 | 74.0 | +1.9 | -0.7 |
| **Core Avg** | **64.9** | **65.9** | **67.5** | **+2.6** | **+1.6** |

**STEM 观察**：
- MMLU 提升最大（+7.5），compute multiplier 5.2x
- ARC-C/E 中等提升
- SciQ 略降（-0.6），可能因 SciQ 本身已接近天花板（96%+）
- TriviaQA 提升最大（+4.9），知识 benchmark 普遍受益更多

### 5.3 Per-Benchmark Compute Multiplier（vs No Filter, Nemotron-CC）

| Benchmark | CM | 含义 |
|-----------|-----|------|
| ARC-Easy | **10.4x** | BETR 只需 10% 计算量 |
| PIQA | **9.5x** | — |
| MMLU | **5.2x** | — |
| SciQ | 4.8x | — |
| ARC-Challenge | 4.5x | — |
| HellaSwag | 3.7x | — |
| TriviaQA | 3.5x | — |
| Lambada | 2.9x | — |
| WebQuestions | 2.8x | — |
| Winogrande | 1.8x | — |

**知识类 benchmark（MMLU, TriviaQA, SciQ）受益最大**，说明 BETR 的 benchmark 相似度排序确实在选知识密集型文档。

### 5.4 关键发现：过滤率随规模变化

$$F_{opt}(C) = 4 \times 10^{-5} \times C^{0.25}$$

| 规模 | FLOPs | 最优过滤率 | 含义 |
|------|-------|-----------|------|
| 小模型 | 10²⁰ | **top 3%** | 极激进过滤 |
| 中模型 | 10²² | top 10% | 中等 |
| 大模型 | 10²³ | top 30% | 宽松 |

**关键洞察**：
- **小模型需要更激进过滤** — 1.5B 模型属于小模型，应该用 top 3-5%
- 大模型已具备足够容量消化多样数据，过滤太激进反而有害
- 这与 Phase Transitions 论文一致：小模型容量有限，必须集中在最相关数据上
- 这也与 Capacity Competition 论文一致：小模型把神经元分配给高频域，稀有域被牺牲

### 5.5 个体 Targeting：精确控制能力

Table 2 关键发现：
- 每行 = 用单个 benchmark 做 target，评估所有 10 个 Core benchmark
- **对角线占优**：每个 targeted benchmark 在自己列上排名最高或并列最高
- **聚类效应**：targeting ARC-Easy 同时提升 ARC-Challenge、SciQ、MMLU（知识类聚类）
- **trade-off**：targeting Winogrande 提升 Lambada（语言理解聚类）但损害 MMLU（知识类）

**多目标退化**（NAG 论文数据）：
- Single-target avg: 48.8%
- Multi-target (6 benchmarks) avg: 44.4% (**-0.5%**)
- BETR 在多目标时 embedding 相似度会冲突

### 5.6 Domain Targeting vs Benchmark Targeting

| Target | Core Avg (7B-1x) |
|--------|-------------------|
| No filter | 56.0 |
| GitHub domain | 52.3 (**-3.7**) |
| arXiv domain | 52.6 (**-3.4**) |
| Wikipedia domain | 55.9 |
| StackExchange domain | 56.3 |
| OH-2.5 + ELI5 | 60.2 |
| DM Mathematics | 60.4 |
| **All above domains** | 59.6 |
| **Noncore benchmarks** | **60.9** |

**关键洞察**：benchmark targeting > domain targeting。即使 targeting 完全不同的 benchmark 集（Noncore），也比 targeting 任何单一 domain 好。这说明 benchmark 驱动的选择比人工直觉的 domain 选择更有效。

---

## 6. 成本分析

| 步骤 | 计算量 | 时间估计 |
|------|--------|---------|
| Embed 10M docs (Arctic-Embed L 2.0) | ~1-2 GPU 小时 (A100) | ~1h |
| Embed benchmark 训练集 (~14K 样本) | 可忽略 | <1min |
| Rank scoring (10M docs × 14K benchmarks) | CPU, 可并行 | ~30min |
| Train FastText classifier | CPU | <5min |
| Score full corpus (~10B docs) | FastText 推理 | 数小时 (可分布式) |
| **总计** | **~2-4 GPU 小时** | **<1 天** |

**对比其他方法**：

| 方法 | 成本 | 类型 |
|------|------|------|
| **BETR** | ~2-4 GPU 小时 | 文档级过滤 |
| QuaDMix (336 proxy) | ~336 GPU 小时 | 域级配比 |
| ClimbMix (14 proxy) | ~5 NPU 小时 | 域级配比 |
| Data Mixing Agent | ~1996 GPU 小时 | 域级配比 |
| NAG (提取) | ~192 GPU 小时 | 文档级选择 |

BETR 是所有方法中**成本最低**的。

---

## 7. 与用户场景的适配分析

### 7.1 用户配置

| 维度 | 用户 | BETR 论文 |
|------|------|-----------|
| 模型 | 1.5B | 7B |
| 训练量 | 1.8B tokens | 140B-1.4T tokens |
| 训练阶段 | **continual pre-training** | from-scratch pretraining |
| 数据池 | ~100B STEM (Essential-Web v1.0) | 24T (DCLM) / 6.3T (Nemotron-CC) |
| STEM 占比 | 70% (数学42%/物理8.4%/化学8.4%/生物11.9%) | 未专门针对 STEM |
| Benchmarks | ARC-C/E, GPQA, GSM8K, MATH, MMLU-STEM | 10 Core + 39 Noncore |
| 过滤率需求 | top 3-5%（小模型） | 论文验证 top 3% 适用小模型 |

### 7.2 适配评估

| 维度 | 评估 | 理由 |
|------|------|------|
| **成本** | ✅ 极低 | ~2-4 GPU 小时，远低于其他方法 |
| **信号源** | ✅ benchmark 相似度 | 不用 val_loss，绕过 QuaDMix 根因 |
| **小模型友好** | ✅ 过滤率幂律 | 1.5B 属小模型，top 3-5% 最优 |
| **STEM targeting** | ✅ 可直接 target STEM benchmarks | ARC-C, MMLU-STEM, GPQA, GSM8K |
| **无 proxy 间隙** | ✅ 不用 proxy 模型 | 避免了 1M→target 或 70M→730M 迁移问题 |
| **训练阶段** | ⚠️ 论文是预训练 | 用户是 continual pre-training，论文未验证 |
| **问题层次** | ⚠️ 文档级 ≠ 域级 | 用户需要域级配比权重，BETR 输出是文档排序 |
| **多目标** | ⚠️ 退化 -0.5% | 用户需 target 6 个 benchmark，可能有冲突 |
| **GPQA 数据量** | ⚠️ 极少 | GPQA Diamond 训练集仅 ~100 样本，embedding 不稳定 |
| **开源** | ⚠️ 未开源 | 但 FastText + Arctic-Embed 均开源，可复现 |

### 7.3 与 ClimbMix 的互补关系

BETR 和 ClimbMix 解决不同层次的问题，可以**两阶段串联**：

```
Stage 1: BETR 文档过滤
  - 输入：~100B STEM 数据 (22 FDC 域)
  - Target：ARC-C, MMLU-STEM, GPQA, GSM8K, MATH 训练集
  - 输出：top 5% 文档 (~5B tokens 高质量 STEM 子集)
  - 成本：~2-4 GPU 小时

Stage 2: ClimbMix 域级配比
  - 输入：BETR 过滤后的 ~5B tokens 高质量数据
  - 搜索：22 FDC 域的最优混合权重
  - 输出：α* (最优配比)
  - 成本：~5 NPU 小时

Stage 3: 目标训练
  - 用 α* 配比在 1.5B 模型上 continual pre-training
```

**为什么这两阶段有协同效应**：
1. BETR 先去除噪声文档 → 提高域内同质性 → ClimbMix 的 LightGBM 回归更准确
2. BETR 过滤后数据量减少 → ClimbMix 搜索空间缩小 → 更少的 proxy 实验
3. 两者都绕过 val_loss 信号 → 一致的信号体系

### 7.4 与 Phase Transitions 的关系

BETR 被我们的 Phase Transitions 分析论文引用。两者的关系：

| 维度 | BETR | Phase Transitions |
|------|------|-------------------|
| 层次 | 文档级选择 | 域级配比 |
| 核心发现 | 小模型需激进过滤 | 小模型低于相变阈值 |
| 一致性 | ✅ 过滤率幂律 | ✅ 容量竞争 |

**一致性解释**：
- Phase Transitions：小模型容量有限，稀有域知识无法跨过相变阈值
- BETR：小模型需要激进过滤，只保留最相关文档
- 两者本质上都在说：**小模型容量有限，必须集中资源**

**BETR 对相变的潜在帮助**：
- BETR 过滤后，每 token 的"有效事实密度" D 可能提高（去除无关事实）
- 如果 D 从 < D_thres 提升到 > D_thres → 可能跨过相变阈值
- 但这不是 BETR 的设计目标，效果需实验验证

---

## 8. 消融实验关键发现

### 8.1 Embedding 选择影响极小

| Embedding 模型 | Core Avg (7B-1x) |
|----------------|-------------------|
| GTE-Base v1.5 | 62.0 |
| GTE-Large v1.5 | 62.2 |
| GTE-Large + Reranker | 62.3 |
| GTE-Qwen2 1.5B | 62.1 |
| **Arctic-Embed L v2** | **62.5** |
| FastText (无 embedding) | 61.8 |

差异仅 0.7 分，embedding 模型不是关键瓶颈。

### 8.2 Scoring 函数

| 聚合 | 值函数 | Core Avg |
|------|--------|----------|
| mean | log₂(1/r) | 61.5 |
| max | log₂(1/r) | 62.0 |
| **max** | **1/r (best rank)** | **62.5** |
| max | 1/r² | 62.2 |

Max + best rank 最优，但差异不大。

### 8.3 Classifier 选择（反直觉）

| 分类器 | 验证准确率 | Core Avg |
|--------|-----------|----------|
| **FastText** | **75.4%** | **62.5** |
| 302M LM (classify) | 82.4% | 62.0 |
| 302M LM (regress) | — | 61.5 |

**分类准确率更高反而效果更差**。论文推测：FastText 的低准确率可能起到了正则化效果，避免过拟合到 benchmark 表面特征。

### 8.4 方法完整性消融

| 消融 | Core Avg | 含义 |
|------|----------|------|
| 完整 BETR | 62.5 | — |
| Target test set (非 train set) | 62.3 | 几乎无差（去偏差） |
| 无 decontamination | 62.4 | 几乎无差 |
| 直接 FastText on benchmarks vs web | 58.0 | **-4.5！** 说明 embedding+rank 步骤关键 |

---

## 9. 引用分析（17 篇）

### 值得关注的引用

| # | 论文 | 年份 | 关系 |
|---|------|------|------|
| 1 | **Data Mixing Can Induce Phase Transitions** | 2025 NeurIPS | **我们的 Phase Transitions 论文** — BETR 被 phase transitions 引用 |
| 2 | **NAG: Neuron-Activated Graph** | 2026 | 直接竞品，用 BETR 做 baseline |
| 3 | **FLUX: Data Worth Training On** | 2026 | 后续工作 |
| 4 | **Capacity-Aware Mixture Law** | 2026 | 容量竞争理论 |
| 5 | **Prescriptive Scaling** | 2026 | scaling law 方法论 |
| 6 | BioAlchemy | 2026 | 生物领域应用 BETR |
| 7 | Removing Noise, not Finding Gold | 2025 | 质量过滤方法 |
| 8 | Linear Model Merging | 2026 | 多模态配比 |

### 影响力对比

| 论文 | 引用数 | 开源 |
|------|--------|------|
| **BETR** | **17** | ❌ |
| ADAPT | 1-2 | ✅ |
| Data Mixing Agent | 4-5 | ❌ |
| Phase Transitions | — | ❌ |

BETR 在 document-level selection 领域有较强影响力。

---

## 10. 局限性

### 10.1 论文自述

| 局限 | 说明 |
|------|------|
| **仅预训练** | "We do not verify if pretraining improvements persist after post-training" |
| **数据量不足** | "top 10% of Nemotron-CC yields only 630B tokens, far below 15T+ tokens typical in production" |
| **硬过滤** | "focus exclusively on hard filtering at fixed thresholds, rather than softer weighting" |
| **FastText 之谜** | "FastText's strong performance despite lower classification accuracy remains unexplained" |
| **单语言** | "Applying BETR to multilingual or code data would likely require replacing FastText's whitespace tokenization" |

### 10.2 额外观察

| 局限 | 说明 |
|------|------|
| **无 continual pre-training 验证** | 用户场景是中训练 annealing，论文未覆盖 |
| **7B 规模验证** | 用户 1.5B 更小，但幂律外推支持小模型 |
| **多目标退化** | NAG 论文显示 -0.5%，用户需 target 6 benchmarks |
| **GPQA 训练样本极少** | ~100 样本，embedding 不稳定 |
| **无 code** | FastText + Arctic-Embed 开源，但 pipeline 未开源 |
| **domain targeting 效果差** | Table 3 显示 targeting "GitHub" 反而 -3.7 分 |

---

## 11. 可借鉴要点

### 11.1 直接可用

| 要点 | 实现方式 | 成本 |
|------|---------|------|
| **benchmark 相似度做数据过滤** | Arctic-Embed + FastText pipeline | ~2-4 GPUh |
| **小模型用 top 3-5% 过滤率** | F_opt 幂律外推到 1.5B | 零（仅调参数） |
| **FastText 足够** | 不需要 LM classifier | 降低成本 |
| **max rank aggregation** | 比均值和 cosine 更鲁棒 | 零（算法选择） |
| **多样 benchmark targeting** | 用 Noncore 策略（39 diverse benchmarks） | 零（数据选择） |

### 11.2 与 QuaDMix/ClimbMix 互补

| 路线 | 做法 | 成本 |
|------|------|------|
| **两阶段** | BETR 过滤 → ClimbMix 配比 | ~2-4 GPUh + 5 NPUh |
| **替代信号** | 用 BETR score 替代 val_loss 做域内质量排序 | ~2-4 GPUh |
| **验证 Phase Transitions** | 过滤后检查物理/化学是否跨过相变阈值 | 零（分析） |

### 11.3 理论启示

1. **benchmark 驱动 > 直觉驱动**：BETR 的 benchmark targeting 优于任何 domain targeting，说明人工直觉（"arXiv 好做数学"）不如直接匹配 benchmark
2. **简单方法胜出**：FastText 击败 LM classifier，再次验证 DCLM 的发现
3. **过滤率是规模依赖的**：固定过滤率在不同规模上次优，需随规模调整
4. **文档级与域级正交**：BETR 解决文档级选择，QuaDMix/ClimbMix 解决域级配比，可以叠加

---

## 12. 复现指南

### 12.1 依赖

```python
# 开源组件
pip install fasttext sentence-transformers pyarrow pandas numpy

# Embedding 模型 (HuggingFace)
# Arctic-Embed L 2.0: Snowflake/arctic-embed-l-v2.0
# 备选: GTE Large v1.5: thenlper/gte-large-v1.5
```

### 12.2 Pipeline 实现

```python
# Step 1: Embed
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("Snowflake/arctic-embed-l-v2.0")
doc_embeddings = model.encode(sampled_docs, batch_size=256)
benchmark_embeddings = model.encode(benchmark_examples, batch_size=256)

# Step 2: Score (max rank aggregation)
from scipy.stats import rankdata
scores = np.zeros(len(sampled_docs))
for be_emb in benchmark_embeddings:
    sims = doc_embeddings @ be_emb
    ranks = rankdata(-sims)  # higher sim = lower rank number
    scores = np.maximum(scores, 1.0 / ranks)

# Step 3: FastText classifier
import fasttext
threshold = np.percentile(scores, 90)
labels = [f"__label__{'pos' if s > threshold else 'neg'}" for s in scores]
# 训练 FastText, 然后对全量语料打分
```

### 12.3 适配用户场景的修改

1. **Target benchmarks**: 用 ARC-C, MMLU-STEM, GPQA, GSM8K, MATH 的训练集替代 Core benchmarks
2. **过滤率**: 用 top 5% 而非 10%（1.5B 小模型）
3. **数据池**: 用 Essential-Web v1.0 STEM 子集（~100B tokens, 22 FDC 域）
4. **后续步骤**: BETR 过滤后数据可直接喂给 ClimbMix 做域级配比

---

## 13. 与项目内已有论文的关系图

```
                    BETR (#64)
                   /         \
                  / document    \ benchmark
                 / level         \ signal
                /                 \
    NAG (#23) —— competitor    ClimbMix —— domain level
    neuron activation           iterative bootstrapping
         |                           |
    Phase Transitions (#01)    QuaDMix —— val_loss signal
    capacity limited            root cause: val_loss ≠ downstream
```

- **BETR vs NAG**: 同为 target-oriented document selection，但 BETR 用 embedding 相似度，NAG 用神经元激活图。NAG +1.0 over BETR。
- **BETR vs ClimbMix**: 不同层次（文档级 vs 域级），可以串联使用。
- **BETR vs QuaDMix**: BETR 用 benchmark 信号绕过 val_loss 问题，与 ClimbMix 同一洞察但更简单。
- **BETR → Phase Transitions**: 过滤率幂律与容量竞争理论一致，小模型需激进过滤。

---

## 14. 总结

| 维度 | 评价 |
|------|------|
| **方法创新性** | ★★★☆☆ — 简单有效，"把 implicit 的 benchmark 优化变 explicit" |
| **实验严谨性** | ★★★★★ — 500+ 模型，scaling law，bootstrap 不确定性量化 |
| **STEM 适用性** | ★★★☆☆ — MMLU +7.5 有潜力，但未专门针对 STEM 子学科 |
| **用户场景适配** | ★★★☆☆ — 预训练 vs 中训练，文档级 vs 域级，但成本低值得尝试 |
| **成本** | ★★★★★ — 所有方法中最低（~2-4 GPUh） |
| **可信度** | ★★★★☆ — Apple 团队，500+ 模型验证，17 引用 |

**一句话总结**：BETR 是最低成本的 benchmark-driven 数据选择方法，不能替代域级配比但可以作为预处理步骤提升数据质量，与 ClimbMix 串联使用。
