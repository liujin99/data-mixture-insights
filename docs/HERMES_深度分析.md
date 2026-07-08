# HERMES 深度分析

## 基本信息
- **论文**: HERMES: A Multi-Granularity Labeling Substrate for Pre-training Data Mixtures
- **arXiv**: 2607.02266v1, 2026年7月2日
- **机构**: Wizard Quant (北京) + 北京大学 + 中国科学技术大学
- **作者**: Ziyun Qiao, Yue Min, Ruining Chen, Yujun Li
- **场景**: Pretraining 数据配比（与 QuaDMix 同层）
- **PDF**: `papers/25_HERMES_MultiGranularity_Labeling.pdf`

## 核心论点

**瓶颈不在 mixer，而在 label system。**

现有方法（DoReMi、RegMix、Chameleon 等）都假设语料已经被分好组，然后优化组权重。但分组方式本身（provenance、topic taxonomy、flat embedding cluster）固定在一个语义轴和一个粒度上。改变粒度需要重新聚类。

HERMES 的核心贡献不是更好的聚类算法，而是一个**一次标注、多粒度可读的层级标签基底**。

## 核心方法

### Learned Semantic Transform (LST) + 3-stage RVQ

1. **Frozen Encoder**: 1024-dim 句子级编码器，对 ~50M 文档生成 embedding，只算一次
2. **LST**: 单层线性变换 + normalize，将 embedding 旋转到量化友好的几何空间
   - 三个损失：结构保持、量化重建、正交性
3. **3-stage RVQ**: 每级 K=256 的 codebook，级联残差量化
   - 输出层级码 (c₁, c₂, c₃)

### 粒度作为前缀长度

| 粒度 | 码 | 名义数 | 实际观测数 |
|------|-----|--------|-----------|
| L₁ | c₁ | 256 | 256 |
| L₁₂ | (c₁, c₂) | 65,536 | 65,408 |
| L₁₂₃ | (c₁, c₂, c₃) | 16.7M | 129,955 |

**关键**：一次标注，通过前缀长度切换粒度，无需重新聚类。

### 两阶段采样

P(x | sampler) = w_{b₁} · π_{b₁}(b_{≤ℓ}) · ρ_{b_{≤ℓ}}(x)

- **Stage 1 (L₁ outer weight)**: 256 个 L₁ bucket 的权重
  - Uniform: 等权
  - DoReMi: group-DRO 在 120M/2.5B proxy 上学习
- **Stage 2 (sub-bucket sampler)**: 在每个 L₁ 祖先内部选择子桶和文档
  - Max-entropy coverage: 等权子桶，所有文档可选
  - Quality top-30%: 按大小加权子桶，只选 FineWeb-Edu top-30% 文档

## 实验结果

### 设置
- **模型**: 1B LLaMA-style decoder
- **训练**: 25B tokens
- **语料**: 内部 ~50M 文档
- **评估**: 16 个能力子任务，4 个能力族（Basic Skills, Science QA, Language Modeling, Others）
- **指标**: Avg. = 族等权宏平均准确率

### 完整排行榜 (Top 10)

| Rank | 配置 | L₁ Outer | Stage-2 | Avg. | z |
|------|------|----------|---------|------|---|
| **1** | **HERMES + L₁₂ quality top-30%** | **DoReMi** | **L₁₂ quality top-30%** | **0.4222** | **+1.628** |
| 2 | HERMES + L₁ max-entropy | Uniform | L₁ max-entropy | 0.4155 | +1.048 |
| 3 | KMeans-256 + L₁ max-entropy | Uniform | L₁ max-entropy | 0.4153 | +1.059 |
| 4 | WO-Format | Uniform | random | 0.4134 | +0.949 |
| 5 | WO-Topic + Topic DRO | DoReMi | random | 0.4082 | +0.508 |
| 6 | WO-Topic + Topic max-entropy | Uniform | random | 0.4079 | +0.469 |
| 7 | HERMES + global L₁₂₃ max-entropy | DoReMi | global L₁₂₃ max-entropy | 0.4061 | +0.317 |
| 8 | HERMES + L₁-local quality top-30% | DoReMi | L₁-local quality top-30% | 0.4045 | +0.114 |
| 9 | WO-Format + Format DRO | DoReMi | random | 0.4029 | +0.128 |
| 10 | HERMES + L₁ max-entropy | DoReMi | L₁ max-entropy | 0.4028 | +0.048 |

### Finding A: Stage-2 规则在 L₁₂ 上很重要

固定 DoReMi L₁ outer weight + L₁₂ 粒度：
- Max-entropy coverage: Avg. 0.3969
- Quality top-30% (corrected reader): Avg. **0.4222**
- **差距: +0.0253 (z +2.09)**

Quality top-30% 同时改变两个轴：子桶质量分布（等权 vs 按大小）和文档资格（所有 vs top-30%）。

### Finding B: L₁₂ 优势在 L₁₂₃ 上消失

| 粒度 | Quality top-30% | Coverage | 差距 |
|------|----------------|----------|------|
| L₁₂ | 0.4222 | 0.3969 | **+0.0253** |
| L₁₂₃ | 0.3988 | 0.3986 | +0.0002 (噪声) |

**原因**: Candidate competition — L₁₂₃ 子桶中位大小从 2,271 缩到 429（5.3x），36% 的 L₁₂₃ 桶少于 30 个文档，桶内 top-30% 不再是全局 top-30% 的稳定代理。

### L₁=256 聚类方法无差异

五种 256-way 聚类方法（KMeans, MiniBatchKMeans, BisectingKMeans, Plain RVQ, HERMES）在标准聚类指标上完全一致：
- 平均余弦相似度差异 < 0.003
- 有效聚类数 236-247
- 熵 [5.46, 5.51]

下游配对训练：HERMES vs KMeans 差 0.0002 Avg.，数值上打平。

### DoReMi vs Uniform 在 L₁ 上的意外发现

| 分组 | Uniform Avg | DoReMi Avg | 差距 |
|------|------------|------------|------|
| HERMES | 0.4155 | 0.4028 | **-0.0127** |
| KMeans | 0.4153 | 0.3889 | **-0.0264** |
| WO-Format | 0.4134 | 0.4029 | **-0.0105** |
| WO-Topic | 0.4079 | 0.4082 | +0.0003 |

**4 组中 3 组 DoReMi 不如 Uniform！** DoReMi 的 group-DRO 上权重 loss 高的组（OCR 碎片、代码模板），但这些组对 16 任务能力评估贡献很小。

## 计算成本

- 20 个 1B/25B 训练：约 100 GPU-days（8 GPU, FP32）
- HERMES 标注：~50M 文档一次标注，复用所有实验
- LST 训练：AdamW, LR 3e-4, 10 epochs, 8-GPU DDP
- Proxy (DoReMi): 120M 参数, 2.5B tokens

## 与 QuaDMix 的关系

### HERMES 论文对 QuaDMix 的定位

HERMES 将 QuaDMix 归类为 **sample-wise selector**（逐样本打分器），明确列为 **out of scope** baseline。理由是：在 50M 文档 × 1B 模型 × 25B token 的规模下，QuaDMix 等逐样本方法"costly to score, tune, and validate"，且大多数不提供显式的混合控制接口。

### 核心差异

| 维度 | HERMES | QuaDMix |
|------|--------|---------|
| **层级** | 标签系统（分组方式） | 混合器（权重优化） |
| **粒度** | 256 → 65k → 130k（前缀可调） | 固定 ~30 个领域 |
| **质量信号** | FineWeb-Edu top-30% | 5 个 FastText 分类器 |
| **优化方式** | DoReMi/Uniform outer + Stage-2 规则 | LightGBM surrogate + 90 维优化 |
| **标注成本** | 一次标注，多粒度复用 | 每次需重训 proxy |
| **核心发现** | 粒度 × 采样规则有交互效应 | 质量 + 多样性联合优化 |

### 关键洞察

1. **粒度存在最优点，不是越细越好**：
   - L₁₂ (65k buckets) > L₁ (256) > L₁₂₃ (130k)
   - L₁₂₃ 因 candidate competition 反而退化
   - QuaDMix 的 ~30 领域可能在"太粗"一端

2. **Quality top-30% 在合适粒度上效果显著**：
   - +0.0253 Avg. 在 L₁₂ 上，仅靠桶内质量过滤
   - 这与 quality-top-k 在你实验中赢的现象一致

3. **DoReMi 在 L₁ 上不如 Uniform**：
   - Group-DRO 上权重 loss 高的组，但这些组对下游能力没用
   - QuaDMix 的 proxy 信号如果类似 DoReMi，可能也有这个问题

4. **标签系统是可分离的设计轴**：
   - HERMES 证明分组方式和混合策略是两个独立的设计轴
   - QuaDMix 把两者耦合在一起（领域定义 = 分组方式 = 优化单元）

## 对 QuaDMix 的启示

### 直接可借鉴

1. **多粒度标签系统**：
   QuaDMix 当前用固定 ~30 个领域。可以考虑用 HERMES 的 RVQ 方法构建层级标签，在不同粒度上做配比优化。特别是 L₁₂ (~65k buckets) 可能是更优的粒度。

2. **桶内质量过滤**：
   HERMES 证明在合适粒度下，简单的桶内 quality top-30% 就能带来 +0.0253 的提升。QuaDMix 可以在每个领域内部加质量过滤层。

3. **DoReMi 的教训**：
   DoReMi 在 L₁ 上不如 Uniform，因为 group-DRO 优化了错误的目标（worst-group loss vs capability avg）。QuaDMix 的 proxy 优化目标也需要审视。

4. **Candidate competition 边界**：
   桶太小（< ~400 文档）时，桶内质量排序不稳定。QuaDMix 的领域如果太小，质量过滤可能无效。

### 局限性

- 只在 1B/25B 验证，未验证 7B+ 规模
- 内部语料和编码器，不可直接复现
- 只测试了 DoReMi 和 Uniform 两种 outer weight
- 未与 QuaDMix/Chameleon/OP-Mix 直接比较
- 单 seed 训练，统计显著性有限

## 可信度评估

**优势**：
- 实验设计严谨，控制变量充分
- 明确区分"聚类更好"和"标签系统更好"两种解释，并排除了前者
- Candidate competition 诊断有直接测量
- 完整的 20 配置排行榜
- 代码开源

**劣势**：
- Wizard Quant 不是知名机构，0 引用
- 内部语料 + 内部编码器，可复现性差
- 1B 模型规模偏小
- 单 seed，统计显著性存疑
- 未与主流方法（RegMix、Chameleon）直接对比

## QuaDMix 相关性评级

**★★★★☆ 高**

HERMES 直接解决 pretraining 数据配比问题，且核心发现（粒度 × 质量过滤交互效应、candidate competition 边界、DoReMi 不如 Uniform）对 QuaDMix 有直接指导意义。特别是：
- 验证了 quality-top-k 在合适粒度下有效的理论支撑
- 提供了"粒度太细反而退化"的量化边界（中位桶大小 ~400 文档）
- 指出标签系统和混合策略应该作为独立设计轴
