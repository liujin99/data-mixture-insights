# Data Mix Insights

LLM 预训练数据配比算法的业界研究洞察项目。

## 项目目标

追踪和分析数据配比（data mixture）领域的最新研究进展，为 QuaDMix 等配比算法的实现和优化提供理论指导和实践参考。

## 核心问题

1. **Proxy 可靠性**：小模型 proxy 的信号是否可靠？如何修复？
2. **Scaling Law 外推**：小模型拟合的 scaling law 能否外推到大模型？
3. **相变现象**：数据配比中的相变如何影响训练效果？
4. **质量信号**：如何定义和量化数据质量？

## 目录结构

```
data-mix-insights/
├── papers/          # 论文 PDF (25 篇)
├── docs/            # 分析文档
│   ├── 数据配比业界洞察分析报告.md      # 初始分析报告
│   ├── 高价值论文深度分析.md            # TOP 3 论文精读
│   ├── 最新论文分析_2025-2026.md        # 2025-2026 新发现
│   ├── CAMEL_Proxy_Reliability_深度分析.md  # CAMEL + Proxy 可靠性深度分析
│   ├── Proxy_Reliability_论文总结.md        # Proxy 可靠性完整总结
│   ├── LLMSurgeon_DataChef_深度分析.md      # LLMSurgeon + DataChef 分析
│   ├── GroupMATES_深度分析.md               # Group-MATES 组级数据选择分析
│   ├── Chameleon_深度分析.md                # Chameleon KRLS 领域配比分析
│   ├── PiKE_ADAPT_OPMix_深度分析.md        # PiKE + ADAPT + OP-Mix 深度分析
│   ├── NAG_NeuronActivatedGraph_深度分析.md  # NAG 神经元激活图数据选择分析
│   ├── CausalMix_深度分析.md                  # CausalMix 因果推断数据配比分析
│   ├── HERMES_深度分析.md                     # HERMES 多粒度标签基底分析
│   └── FastMix_DomainAwareScalingLaws_深度分析.md  # FastMix + Domain-Aware SL 深度分析
└── README.md
```

## 论文索引

### 第一梯队：高可信度 + 高价值

| # | 论文 | 机构 | Venue | 关键贡献 |
|---|------|------|-------|---------|
| 01 | Phase Transitions in Data Mixing | 清华 | NeurIPS'25 Spotlight | 揭示配比相变现象，理论证明 |
| 03 | Scaling Laws for Optimal Data Mixtures | Apple | arXiv | 参数化 scaling law，跨模态验证 |
| 04 | DeMix: Model Merging | 小红书+东大 | ICML'26 | Model merging 替代 proxy 训练 |
| 09 | CAMEL: Capacity-Aware Mixture Law | 清华 | arXiv | Phase Transitions 的实践版 |
| 10 | Proxy Reliability | Princeton/Berkeley | ICLR'26 | 降低 LR 修复 proxy 信号 |
| 20 | Chameleon | EPFL | **ICML'25** | KRLS 领域权重，免重训迁移 |
| 21 | PiKE: Adaptive Data Mixing | Google Research+USC | arXiv | 梯度交互自适应配比，1.9x 收敛加速 |
| 26 | FastMix: Gradient Descent Mixing | 腾讯混元+港大 | **ICLR'26** | 可微 bilevel 优化，1 proxy 550x 加速 |
| 27 | Domain-Aware Scaling Laws | MIT+Microsoft Research | arXiv | 域间协同量化 γ/σ，R²=0.91 |

### 第二梯队：大厂经验

| # | 论文 | 机构 | Venue | 关键贡献 |
|---|------|------|-------|---------|
| 11 | DataMan | 阿里 Qwen | ICLR'25 | 14 条质量标准，447B 标注 |
| 12 | UtiliMax / MEDU | Meta FAIR | arXiv | Token-count baseline，200x 加速 |
| 13 | Repetition-Aware Scaling | Apple | arXiv | 高质量数据可重复 15-20x |
| 14 | Synthetic Data in Pretraining | Meta | EMNLP'25 | 30% 合成 + 70% 自然 |
| 22 | ADAPT: Online Reweighting | Cambridge+Oxford | **ICLR'26** | 在线 reweighting 优于离线选择，泛化性强 |

### 第三梯队：探索方向

| # | 论文 | 机构 | Venue | 关键贡献 |
|---|------|------|-------|---------|
| 02 | Mixture Scaling Under Constraints | Apple | arXiv | 数据受限场景配比 |
| 05 | Data Mixing Laws | 复旦 | ICLR'25 | 嵌套 scaling laws |
| 06 | Multi-fidelity BO | Columbia | arXiv | 贝叶斯优化框架 |
| 07 | Beyond Language Modeling | Meta FAIR | arXiv | 多模态预训练 |
| 08 | CMR Scaling Law | - | EMNLP'24 | 跨模态 scaling |
| 15 | OP-Mix | NYU+DeepMind | arXiv | LoRA 插值 proxy |
| 16 | Data Organization | Microsoft | ACL'26 | 数据顺序优化 |
| 17 | LLMSurgeon | MBZUAI+UCL | ACL'26 | 从生成文本反推预训练配比 |
| 18 | DataChef | 复旦+上海AI Lab | arXiv | RL 自动生成 adaptation 配方 |
| 19 | Group-MATES | CMU+Meta | arXiv | 组级数据选择，建模数据交互效应 |
| 23 | NAG: Neuron-Activated Graph | ByteDance+UCSC+UNC | arXiv | 神经元激活图做 target-oriented 数据选择，training-free |
| 24 | CausalMix: Causal Inference | 清华+蚂蚁集团 | arXiv (2026.7) | 因果推断框架做 SFT 配比，DML 正交化 + CATE |
| 25 | HERMES: Multi-Granularity Labeling | Wizard Quant+北大+中科大 | arXiv (2026.7) | 层级 RVQ 标签基底，粒度作为前缀长度，粒度×质量过滤交互效应 |
| 26 | FastMix: Gradient Descent Data Mixing | 腾讯混元+港大+中大 | **ICLR 2026** | 可微 bilevel 优化，1 proxy 550x 加速，梯度内积信号 |
| 27 | Domain-Aware Scaling Laws | MIT CSAIL+Microsoft Research | arXiv (2026.7) | 量化域间协同 γ 和 σ，观测拟合，R²=0.91 |

## 核心洞察

### 1. Proxy 可靠性是核心问题

三篇论文从不同角度指向同一个问题：
- **Phase Transitions** (NeurIPS'25): 小 proxy 信号可能完全无效（理论）
- **Proxy 可靠性** (ICLR'26): 超参数设置会导致结论翻转（实证）
- **CAMEL** (2026.3): 用 capacity-aware 方法修正不可靠性（方法）

**结论**：1M proxy 是最大的风险点，需要立即修复。

### 2. Scaling Law 的局限性

- Scaling law 本质是经验拟合，不是第一性原理
- 小模型拟合的 scaling law 在 >10B 模型上常常失效
- Phase Transitions 证明了为什么：知识密集数据存在相变，power law 无法预测

**结论**：Scaling law 作为理解趋势的工具有价值，但作为精确预测工具不够可靠。

### 3. 简单方法可能是最好的

- **Token-count heuristic** (Meta UtiliMax): 优于人工调参和 learned mix
- **降低 LR** (Proxy 可靠性): 简单 patch 解决大问题
- **15-20x 重复** (Apple): 比想象中高得多

**结论**：不要过度工程化，先试试简单方法。

### 4. 组级效应是真实且显著的

Group-MATES (CMU+Meta) 用实验量化了数据交互效应：
- 个体选择 vs 组级选择性能差距 **2x**
- 关系权重（embedding cosine similarity）是建模交互的最有效组件
- 选择开销随模型规模增大而递减（3B 上仅 3.4%）

**结论**：QuaDMix 的二次型 Q 矩阵方向正确，但可以考虑用更轻量的 cosine similarity 替代。

### 5. LLM 预训练中梯度冲突极少，应利用而非修复

PiKE (Google Research) 实证发现大规模预训练中任务梯度几乎不冲突：
- PCGrad、NashMTL 等冲突缓解方法在 LLM 中**基本无效**
- 最优动态权重取决于**梯度范数**和**梯度方差**两个标量
- 自适应调整比最优静态权重**一致更好**

**结论**：QuaDMix 可以利用梯度范数+方差作为配比信号，不需要复杂的冲突缓解。

### 6. 离线方法存在泛化风险

ADAPT (Cambridge, ICLR'26) 揭示离线数据选择的根本缺陷：
- 离线选择在 in-domain 上表现好，但 **OOD 泛化严重退化**（LESS: -11.5%）
- 硬过滤牺牲数据多样性，用记忆替代泛化
- 在线 reweighting 保持全部数据，动态调整权重，泛化更强

**结论**：QuaDMix 作为离线方法，评估时必须包含 OOD benchmark，避免过度优化特定领域。

### 7. 神经元级信号是未被充分利用的维度

NAG (ByteDance+UCSC+UNC) 证明 LLM 内部神经元激活模式可以有效表征数据质量：
- 仅 0.12% 的神经元构成"功能骨架"，关掉即导致 23.5% 性能崩塌
- FFN UP 层信号最丰富，跨层聚合比单层好 4.1%
- Training-free，192 GPU-hours 一次性提取可复用于任意 target
- 与质量信号（FineWeb-Edu）正交，可叠加

**结论**：NAG 解决样本级选择，QuaDMix 解决领域级配比，两者互补。NAG 论文自己指出 QuaDMix 可以改进其 multi-target 设置。

### 8. 因果推断视角：不存在单一最优配比，最优配比取决于数据状态

CausalMix (清华+蚂蚁集团, 2026.7) 将数据配比转化为因果边际收益估计：
- 用 Double Machine Learning 正交化隔离数据状态的混杂效应
- 3 个协变量（Normalized_Loss, HES, Writing_Style）分别对应 Complexity, Difficulty, Quality
- 因果模型在 tulu-3 上训练，成功迁移到**完全未见的数据集和模型架构**（Qwen3-4B）
- CATE Interpreter 发现：Knowledge 数据对困难数据有负效应（技能冲突）

**结论**：QuaDMix 的 90 维搜索在找静态最优解，而 CausalMix 证明最优配比应该随数据状态变化。可借鉴 DML 正交化改进 LightGBM surrogate，用 top-100 平均替代单一最优点。

### 9. 可微 bilevel 优化实现 550x 搜索加速，但工业数据风险高

FastMix (腾讯混元, ICLR 2026) 将配比权重从采样概率重参数化为 loss 权重，使 α 可微：
- 1 个 proxy（2-layer LLaMA 256d，与 QuaDMix **架构完全相同**）联合优化模型参数和配比
- 梯度 = 验证集梯度 · 领域训练梯度的内积（方向对齐信号）
- Pre-training 14 benchmarks 平均 48.2（RegMix 47.2, CLIMB 47.5），搜索成本 1.3 GPU-hours（RegMix 720h）
- SFT 场景优势更大：65.4 vs CLIMB 59.9

**关键警告**：论文自己承认 < 0.5B proxy 在工业数据上收敛不稳定。QuaDMix 的 14M proxy 远小于 0.5B。代码中熵正则化 λ=0（论文声称 1e-5），只用 lm_head 梯度（论文未讨论）。

### 10. 域间协同可量化，且协同 ≠ 语义重叠

Domain-Aware Scaling Laws (MIT+MSR, 2026.7) 从 52 个公开模型拟合域间协同：
- **一阶 γ**：Math→HumanEval +1.34, Books→HumanEval -0.90, Code→IFEval +0.88
- **二阶 σ**：Code×Science +2.55, Code×Math +2.40, Math×Books 负值
- R² 从 Chinchilla 0.41 提升到 FO 0.92（HumanEval），最差 R²=0.866
- 验证实验：optimal vs anti-optimal 在 150M 上 HumanEval BPB 差 0.337
- **协同 ≠ 重叠**：Code×Science embedding cosine=0.31（低），但 σ=+2.55（强协同）

**局限**：观测拟合（非因果），8 域只有 2 个有效 PCA 方向，验证仅 30M/150M。

### 11. 瓶颈在标签系统而非 mixer，粒度存在最优点

HERMES (Wizard Quant+北大+中科大, 2026.7) 证明数据配比的瓶颈不在混合策略优化，而在分组方式本身：
- 3-stage RVQ 一次标注 ~50M 文档，通过前缀长度切换粒度（256 → 65k → 130k buckets）
- **L₁₂ (65k) 粒度下 quality top-30% 带来 +0.0253 Avg.**，但 L₁₂₃ (130k) 上优势消失
- 原因：candidate competition — 桶中位大小从 2,271 缩到 429，桶内排序不再稳定
- **DoReMi 在 L₁ 上不如 Uniform**（4 组中 3 组），group-DRO 优化了错误的目标
- 五种 256-way 聚类方法在标准指标上完全一致，证明增益来自层级结构而非聚类质量

**结论**：QuaDMix 的 ~30 领域可能在"太粗"一端，但粒度也不是越细越好（~400 文档/桶是下界）。标签系统和混合策略应作为独立设计轴。桶内质量过滤在合适粒度下效果显著，再次验证 quality-top-k 方向正确。

## 对 QuaDMix 的行动建议

### 立即行动

1. **修复 Proxy LR** — 降低 proxy 的 learning rate，验证与大模型结果的相关性
2. **实现 Capacity-Aware Mixture Law** — 在 LightGBM 回归中加入 capacity-aware 特征
3. **记录域梯度统计量** — 在 proxy 训练中记录每个域的梯度范数和方差（FastMix + PiKE）
4. **加入 pairwise 域特征** — 在 LightGBM 特征中加入 u_k × u_{k'} 捕捉二阶协同（Domain-Aware SL）

### 短期行动

5. **对比 Token-Count Baseline** — 实现 token-count heuristic 作为 baseline
6. **调整高质量数据重复次数** — 高质量数据重复 15-20x
7. **参考 DataMan 质量标准** — 检查 Eq.1 的质量信号是否覆盖 14 条标准
8. **验证集域覆盖度检查** — 确保 CAP v1/OpenHermes 不偏向特定域（Domain-Aware SL + Google MDE）
9. **PCA 检查搜索空间有效维度** — 对 proxy 实验数据做 PCA（Domain-Aware SL 发现 8 域只有 2 个有效方向）

### 长期探索

10. **探索 FastMix 在线优化模式** — 550x 搜索加速，但需注意工业数据风险
11. **探索 LoRA 插值 Proxy** — 比训完整模型便宜，比 model merging 灵活
12. **扩展到数据顺序优化** — 顺序是另一个重要维度
13. **合成数据配比** — 如果数据池包含合成数据，使用 30% 配比

## 更新日志

- **2026-06-22**: 初始分析，收集 8 篇论文，完成 TOP 3 深度分析
- **2026-06-23**: 扩展搜索，新增 8 篇论文，发现 CAMEL 和 Proxy 可靠性两篇关键论文
- **2026-06-30**: 新增 LLMSurgeon (ACL'26) 和 DataChef (arXiv)，完成深度分析，评估与 QuaDMix 相关性（低）
- **2026-07-03**: 新增 Group-MATES (CMU+Meta)，完成深度分析，评估组级数据选择对 QuaDMix 的启示
- **2026-07-03**: 新增 Chameleon (EPFL, ICML'25)，完成深度分析，KRLS 领域配比方法与 QuaDMix 高相关
- **2026-07-03**: 新增 PiKE (Google Research, arXiv) 和 ADAPT (Cambridge, ICLR'26)，完成深度分析；确认 OP-Mix (paper 15/23) 为同一篇论文
- **2026-07-03**: 新增 NAG (ByteDance+UCSC+UNC, arXiv)，完成深度分析，神经元激活图做 target-oriented 数据选择，与 QuaDMix 互补
- **2026-07-06**: 新增 CausalMix (清华+蚂蚁集团, arXiv 2026.7)，完成深度分析，因果推断框架做 SFT 数据配比，DML 正交化 + CATE Interpreter
- **2026-07-08**: 新增 HERMES (Wizard Quant+北大+中科大, arXiv 2026.7)，完成深度分析，层级 RVQ 标签基底，粒度×质量过滤交互效应，与 QuaDMix 高相关 (★★★★☆)
- **2026-07-20**: 新增 FastMix (腾讯混元+港大+中大, ICLR 2026) 和 Domain-Aware Scaling Laws (MIT+MSR, arXiv 2026.7)，完成深度分析。关键发现：FastMix proxy 架构与 QuaDMix 完全相同；论文自述 <0.5B proxy 工业数据不稳定；域间协同可量化且协同≠语义重叠

## License

MIT
