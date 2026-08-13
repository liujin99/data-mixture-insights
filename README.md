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
├── papers/          # 论文 PDF (54 篇, 本地存储不入库)
├── docs/            # 分析文档
│   ├── 数据配比业界洞察分析报告.md      # 初始分析报告
│   ├── 高价值论文深度分析.md            # TOP 3 论文精读
│   ├── 最新论文分析_2025-2026.md        # 2025-2026 新发现
│   ├── CAMEL_深度分析_v3.md                      # CAMEL v3 深度分析（作者调查+6可借鉴点+PreSelect互补）
│   ├── CAMEL_Proxy_Reliability_深度分析.md  # CAMEL + Proxy 可靠性深度分析
│   ├── Proxy_Reliability_论文总结.md        # Proxy 可靠性完整总结
│   ├── LLMSurgeon_DataChef_深度分析.md      # LLMSurgeon + DataChef 分析
│   ├── GroupMATES_深度分析.md               # Group-MATES 组级数据选择分析
│   ├── Chameleon_深度分析.md                # Chameleon KRLS 领域配比分析
│   ├── PiKE_ADAPT_OPMix_深度分析.md        # PiKE + ADAPT + OP-Mix 深度分析
│   ├── NAG_NeuronActivatedGraph_深度分析.md  # NAG 神经元激活图数据选择分析
│   ├── CausalMix_深度分析.md                  # CausalMix 因果推断数据配比分析
│   ├── HERMES_深度分析.md                     # HERMES 多粒度标签基底分析
│   ├── FastMix_DomainAwareScalingLaws_深度分析.md  # FastMix + Domain-Aware SL 深度分析
│   ├── PreSelect_深度分析.md                         # PreSelect (ICML'25) 深度分析
│   ├── Survey_RegMixD_深度分析.md                    # Survey 综述 + RegMix-D 深度分析
│   ├── 新论文批量分析_202607.md                      # 5 篇新论文分析 (MixAtlas, WebGraphMix, PreSelect 等)
│   ├── 新论文批量分析_2026下半年.md                  # 14 篇新论文分析 (InfoLaw, DomainPilot, HDS, Solar Open 2 等)
│   ├── 业界模型技术报告_数据配比分析.md              # Kimi K3 + DeepSeek V4/V3/Math/LLM 数据配比分析
│   ├── 一线大厂annealing阶段数据配比分析.md          # Qwen3/Phi-4/DeepSeek V3/Kimi/Gemma3 annealing 阶段分析
│   ├── Nexus_深度分析.md                             # Nexus 优化器深度分析（非数据配比但直接影响配比价值判断）
│   ├── 小Proxy不可用_多论文交叉验证总结.md            # 6 篇论文交叉验证：1M proxy+val_loss 路径不可修复
│   ├── 头部厂商TechReport_数据配比深度分析.md         # DeepSeek-V4/GLM-5/Qwen3/Nemotron 3 frontier 规模数据配比分析
│   ├── Velocitune_深度分析.md                          # Velocitune (ACL'25) 学习速度动态域配比 + scaling law 稳定性分析
│   ├── DiReCT_深度分析.md                              # DiReCT (ICML'26) Hessian 谱几何冷却阶段样本选择
│   ├── DataMixingAgent_深度分析.md                     # Data Mixing Agent (ACL'26) RL 学习域配比策略，Velocitune 升级版
│   └── ADAPT_深度分析_v2.md                            # ADAPT (ICLR'26) 在线 reweighting + 四篇联合对比表
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
| 26 | **FastMix: Gradient Descent Mixing** | 腾讯混元+港大 | **ICLR'26** | **[高价值]** 可微 bilevel 优化，1 proxy 550x 加速，proxy 架构与 QuaDMix 完全相同，Section 4.3 工业踩坑经验直接适用 |
| 27 | **Domain-Aware Scaling Laws** | MIT+Microsoft Research | arXiv (2026.7) | **[高价值]** 域间协同量化 γ/σ，R²=0.91，协同≠语义重叠，pairwise 特征可直接加入 LightGBM |
| 30 | **WebGraphMix** | Princeton (Danqi Chen) | arXiv | 网页图中心度作为 training-free 正交质量信号，与 FastText 正交，乘法组合 +1.5% |
| 31 | PreSelect | HKUST+Vivo AI | **ICML'25** | Predictive strength + fastText，10x 计算缩减，开源 |
| 37 | **InfoLaw** | ByteDance | **ICML'26** | **[高价值]** 信息论 scaling law，同时建模配比权重+重复递减收益，7B/425B 验证，0.15% loss 误差 |
| 38 | **Explaining Data Mixing Scaling Laws** | - | **ICML'26** | **[高价值]** 理论解释配比 scaling law：Capacity Competition + Noise Reduction，理论背书 quality-top-k，解释 proxy→target 失效 |

### 第二梯队：大厂经验

| # | 论文 | 机构 | Venue | 关键贡献 |
|---|------|------|-------|---------|
| 11 | DataMan | 阿里 Qwen | ICLR'25 | 14 条质量标准，447B 标注 |
| 12 | UtiliMax / MEDU | Meta FAIR | arXiv | Token-count baseline，200x 加速 |
| 13 | Repetition-Aware Scaling | Apple | arXiv | 高质量数据可重复 15-20x |
| 14 | Synthetic Data in Pretraining | Meta | EMNLP'25 | 30% 合成 + 70% 自然 |
| 22 | ADAPT: Online Reweighting | Cambridge+Oxford | **ICLR'26** | 在线 reweighting 优于离线选择，泛化性强 |
| 36 | **RegMix-D: Dynamic Mixing** | U Tokyo+NII | arXiv (2026.6) | **[高价值]** RegMix 动态化扩展，loss 轨迹回归，128 proxy (25% 预算) 超越 RegMix 512，Online +0.37% 开销，proxy 大小 1M-120M 无差异 |
| 40 | **HDS: Holistic Data Scheduler** | Dang, Ma, Liao | **KDD'26** (S2 DBLP 确认) | 在线 RL（SAC）数据调度，三目标奖励（质量+域间影响+weight norms），Pile 省 44% 迭代，MMLU +7.2%，代码开源 |
| 41 | TANDEM: Twin Networks | Kuaishou (12 作者) | arXiv (2026.6) | Bi-level 配比优化，twin networks（proxy+reference）差值度量数据效能，理论保证 |
| 42 | **Solar Open 2** | Upstage | Tech Report (2026.7) | 250B-A15B MoE 业界实践：quality+rarity 策展 + mixture-ratio 优化，20T→10T 精炼 |
| 61 | **Velocitune** | Microsoft | **ACL'25** | 学习速度动态域配比：V=(L_cur-L_target)/(L_init-L_target)，同模型 scaling law 预测目标 loss，exp 权重更新。绕过 QuaDMix 跨模型外推问题。scaling law 稳定性不透明（逐域误差未报告）。11 引用 |
| 62 | **DiReCT** | MSRA+HKUST(GZ) | **ICML'26** | Annealing 阶段样本选择：Hessian 谱几何 → flat/stiff 子空间 → 选 flat 方向梯度大的样本。随机投影 k≈34。355M-1.1B 验证。100B 数据不可行（论文明确 future work）。代码开源 |
| 63 | **Data Mixing Agent** | Microsoft | **ACL'26** | **[高价值]** RL 学习域配比策略（Velocitune 升级版）。2.1M agent + CQL 离线 RL + 50M proxy 384 轨迹（1996 GPUh 一次性）。benchmark 分数作 reward（绕过 val_loss 问题）。跨模型+跨领域泛化。3B 验证 +3.02% over RegMix |

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
| 29 | MixAtlas: GP Surrogate | Apple+UW+VTech | arXiv | GP surrogate + GP-UCB 主动采样，同预算比 LightGBM 高 4-7 点，0.5B→7B 迁移验证 |
| 32 | GRASP: Subset Interaction | Wizard Quant+USTC | arXiv | 子集交互感知数据归因，二次几何惩罚，rank correlation 翻倍 |
| 33 | DoGraph: Gradient Dynamics | HKUST-GZ+HIT+OpenCSG | arXiv | 梯度视角重定义域，人类域≠模型域，graph-constrained reweighting |
| 35 | Survey: Data Mixture for LLMs | CUHK-Shenzhen (Ruoyu Sun) | arXiv (2025.5) | 领域综述：19 种方法分类，3 大挑战，指出最优权重仅略优于 uniform |
| 39 | **DomainPilot** | 单作者 (He Zhang) | arXiv (2026.7) | **[独立再发现]** 域级 loss 引导两阶段优化，~30 行 patch 工业部署，Qwen3-1.7B SFT +2-3.8%。单作者无 venue 无机构，但 2026-08 独立搜索中从不同角度重新发现并列为 Tier 1 优先深读。工程思路高参考价值，结论需验证 |
| 43 | DataOrchestra | SJTU (Pengfei Liu) | arXiv (2026.7) | 样本级策展编排，orchestrator 决定 drop/untouch/clean，0.5B-7B from scratch |
| 44 | DecoupleMix | Meituan (7 作者) | arXiv (2026.7) | VLM 配比解耦（inter-class 搜索 + intra-class 凸优化），proxy 最优比例迁移到更大尺度 |
| 45 | CuraWeb | Meituan (15 作者) | arXiv (2026.7) | 质量/冗余/多样性联合优化，2T-token 语料，3B 规模 +1.8% |
| 46 | Reasoning Quality Emerges Early | UCLA (Mirzasoleiman) | **ICML'26** (Poster) | 前 100 推理 token loss + 扰动 checkpoint 检测难度，91% 更高效 |
| 47 | Online Selection Is Implicit Alignment | 3 作者 | arXiv (2026.7) | 在线数据选择隐式改变行为偏好（拒绝率/冗长度/谄媚度） |
| 48 | CoDA: Co-Adaptive LoRA | 4 作者 | arXiv (2026.7) | 无标签域参与度控制，competence+affinity 信号，熵正则化 QP，半数据超 uniform |
| 49 | DataFlex | 北大 (Bin Cui) + Conghui He + Weinan E | arXiv (2026.3) | 统一数据中心训练框架（selection+mixture+reweighting），基于 LLaMA-Factory |
| 50 | Survey #2: Data Mixing | Tianjin U (Deyi Xiong) | Data Intelligence (2026, 期刊同行评审) | 第三篇综述，bilevel on simplex 形式化，inverse data mixing 方向 |

### 第四梯队：相关方向（非数据配比但直接影响配比价值判断）

| # | 论文 | 机构 | Venue | 关键贡献 |
|---|------|------|-------|---------|
| 51 | **Nexus: Same Loss, Better Downstream** | 清华+字节跳动 Seed | arXiv (2026.4) | **[高价值]** 优化器方法，非数据配比。同一 pretraining loss GSM8K 差 15%。Nexus 通过梯度相似性收敛到 close minimizer，自动平衡欠优化域（隐式数据混合器）。与数据配比正交，可叠加 |
| 52 | **Why Larger Models Learn More** | U Chicago+Apple+Anthropic | arXiv (2026.5) | **[新发现]** 容量竞争：小模型把神经元分配给高频/低复杂度域→稀有复杂域被牺牲。OLMo 4M-4B 实验。解释 1M proxy 为何不能学 STEM 推理 |
| 53 | **RISE: Scalable Data Attribution** | Vendor AI | arXiv (2026.4) | **[新发现]** Forward-only 数据归因，112× 存储压缩，扩展到 32B。影响信号集中在 output layer。闭环：RISE 选数据持续 pretraining → consistent improvement |
| 54 | **Kernel Surrogate for Task Attribution** | U Chicago (Hongyang Zhang) | **ICLR'26** | **[新发现]** 核函数代理模型 > 线性代理。25% 更高 LOO 相关性，数据选择 40% 提升。梯度估计 <2% 误差无需重训。可替代 QuaDMix LightGBM |
| 55 | **K-EXAONE 2.0** | LG AI Research | Tech Report (2026.8) | 750B MoE/37B activated。**"difficulty-focused mid-training"** — 难度聚焦中训练设计。continual pre-training → mid-training → post-training |
| 56 | **Beyond Rephrasing: Book-Level Organization** | - | arXiv (2026.7) | **[中训练专用]** 书籍级组织的合成教材，686K 教材/32B tokens/15000+ 学科。书级组织（非重写）+1.09 平均提升。Llama3-8B 验证 |
| 57 | **DeepSeek-V4** | DeepSeek-AI | Tech Report (2026.4) | 1.6T/49B + 284B/13B MoE, 32T+33T tokens。提到"mid-training phase"加入 agentic 数据但无配比细节。训练稳定性: Anticipatory Routing + SwiGLU Clamping |
| 58 | **GLM-5** | 智谱 AI + 清华 | Tech Report (2026.2) | **[高价值]** 744B/40B MoE, 28.5T tokens。**头部厂商唯一专门 Mid-Training 章节**：三阶段上下文递进(32K→128K→200K, 1T+500B+50B) + 160B SWE 数据 + 合成长上下文 |
| 59 | **Qwen3** | Alibaba Qwen | Tech Report (2025.5) | 235B/22B MoE, 36T tokens。三阶段预训练(S1通用30T→S2推理5T→S3长上下文)。**实例级配比优化**超越域级方法(DoReMi/RegMix)，30T+ tokens 细粒度标注 |
| 60 | **Nemotron 3 Ultra** | NVIDIA | Tech Report (2026.6) | **[高价值]** 550B/55B MoE, 20T tokens。**数据配比最透明**：19 类别 Phase1/Phase2 比例公开，两阶段 diversity→quality 课程(75%/25%)，SFT-in-pretrain，消融数据丰富 |

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

### 12. 网页图拓扑是完全正交的质量信号

WebGraphMix (#30, Princeton Danqi Chen 组) 证明了一个惊人的发现：
- Common Crawl host-level 网页图的中心度（betweenness/katz）与 FastText 质量分数**正交**
- 中心 host → 符号/算法推理 (+1.4%)，边缘 host → 事实知识 (+1.2%)
- 1:1 中心/边缘混合 + 乘法组合 = **43.8%** vs quality-only 42.3% vs random 39.8%
- **18/18 种组合配置均超过 quality-only baseline**
- 完全 training-free，一次性 9 GPU-hours

**结论**：网页图中心度可作为 QuaDMix FastText 质量分数的独立补充维度，实现成本极低。

### 13. GP Surrogate 在同预算下优于 LightGBM

MixAtlas (#29, Apple+UW) 在相同 proxy 预算下直接对比 GP vs LightGBM (RegMix)：
- 5D 搜索空间：GP 高 4.18 点（50.68 vs 46.50）
- 10D 搜索空间：GP 高 7.06 点（47.18 vs 40.12）
- GP 提供原生不确定性量化 σ(h)，支持 GP-UCB 主动探索
- 0.5B proxy → 7B target 迁移成功（+8.5%），跨族 Qwen2→Qwen2.5 也成功（+3.3%）

**局限**：GP 的 O(n³) 扩展性限制高维应用。QuaDMix 的 198 维空间可能超出 GP 能力，但低维子空间（5-10 主域）可行。

### 14. 人类定义的域 ≠ 模型感知的域

DoGraph (#33, HKUST-GZ) 通过 Theorem 2.2 证明：域差异编码在梯度几何中，人类标签定义的域与模型梯度感知的域可能不一致。

**结论**：QuaDMix 的 22 个 Dewey Decimal 域是否对 14M proxy 有意义？建议在 proxy 训练中记录 per-domain 梯度分布，用 PCA 检查域结构是否与梯度分布对齐。

### 15. 动态配比是最低成本的改进路径

RegMix-D (#36, U Tokyo+NII) 用最简洁的扩展证明动态配比的威力：
- **核心改动**：从终点 loss 回归 → 完整 loss 轨迹回归，输入 (step, weights, current_loss) → next_loss
- **128 proxy (25% 预算) 的 RegMix-D 超越 512 proxy 的 RegMix**：+0.52 avg（47.93 vs 47.41）
- **Online 模式**：在目标训练切换点用实际 loss 查询回归模型，额外开销仅 +0.37%
- **Proxy 大小消融**：1M = 60M = 120M（差异 < 0.05），验证小 proxy 在 Pile 上足够
- **跨尺度修正**：l̃ = l_target · (P_t/P_p)^β, β=0.05，对 β 不敏感（最大变化 0.25）

**结论**：QuaDMix 只需修改 proxy 训练 logging（记录轨迹）+ 重训 LightGBM（加 step/current_loss 特征），即可获得 +0.5-1% 提升，成本极低。这是目前所有改进建议中 ROI 最高的。

### 16. 最优配比可能只比 uniform 略好（甚至更差）

Survey (#35, CUHK-SZ) 明确指出：大多数方法找到的最优权重**只比 uniform sampling 略好**（Jiang et al. 2024），甚至**不如 uniform sampling**（Chen et al. 2024）。

三大挑战：
1. 全局最优 curriculum 计算上不可行
2. 自然域定义过粗（GLaM 8 域 vs Pile 22 域效果差异大）
3. Loss 指标与任务表现之间存在 gap

**结论**：这与 "QuaDMix < quality-top-k" 的观察高度一致。复杂优化在 loss 空间可能有效，但在任务空间无效。应降低对精确配比的期望，聚焦信号质量而非搜索复杂度。

### 17. Scaling Law 从经验拟合走向机制解释，理论背书 quality-top-k

两篇 ICML 2026 论文从理论角度解释数据配比：

**InfoLaw (#37, ByteDance)** 把预训练建模为信息积累：质量→信息密度，重复→尺度依赖递减收益。7B/425B 验证，loss 预测误差仅 0.15%。重复维度是 QuaDMix 完全缺失的关键变量。

**Explaining Data Mixing Scaling Laws (#38)** 识别两个机制：
- **Capacity Competition**：有限模型容量在域间分配，导致域 loss 全局耦合。这从理论上解释 proxy→target 失效——14M proxy 容量极稀缺，容量竞争机制与 7B target 完全不同，最优配比必然偏移
- **Noise Reduction**：最优权重向"更难学"的域偏移以最小化噪声。这从理论上**直接背书 quality-top-k**——高质量数据噪声更少，最优权重自然向其偏移

**结论**：QuaDMix 的 quality-first 设计有理论依据。198 维搜索空间严重过参数化（#38 用更少参数达到更高精度）。应考虑引入 InfoLaw 式重复递减收益建模。

### 18. 在线/动态配比方法全面爆发，QuaDMix 静态配比日益孤立

2026 下半年新方法几乎全部是动态/在线的：DomainPilot (两阶段域 loss 引导)、HDS (RL 在线调度)、TANDEM (twin networks)、CoDA (熵正则化 QP)、ADAPT (在线 reweighting)。Solar Open 2 业界报告也明确采用配比优化。

**HDS (#40, KDD 2026)** 引入三目标奖励：data-driven（质量）+ loss-driven（域间影响）+ **model-driven（weight norms）**。weight norms 是 QuaDMix 缺失的模型状态信号，可反映"模型正在学什么"。Pile 省 44% 迭代，MMLU +7.2%。

**DomainPilot (#39)** 证明域级 loss 监控 + 两阶段优化（~30 行 patch 注入 Megatron-LM）即可在 Qwen3-1.7B SFT 上 +2-3.8%，不增加成本。

**结论**：领域整体向动态迁移，QuaDMix 的静态 LightGBM 回归越来越孤立。最低成本动态化路径：① 记录 proxy loss 轨迹（RegMix-D）→ ② 目标训练加域 loss 监控 patch（DomainPilot）→ ③ 如需更强效果加 model-driven reward（HDS）。

### 19. 业界 250B 级模型确认配比优化有效，反驳"仅略优于 uniform"

Solar Open 2 (#42, Upstage, 250B-A15B MoE) 技术报告明确披露：quality + rarity 策展 + mixture-ratio optimization，将 20T 候选池精炼为 10T 混合，"在同等 token 预算下超越 Solar Open 1 配方"。DeepSeek-V3 (671B/37B) 训练 14.8T "diverse and high-quality" tokens，全程零 spike。

**结论**：Survey (#35) "最优权重仅略优于 uniform" 基于学术小规模实验。业界前沿大模型明确采用配比优化并报告收益——**在工业规模上配比优化确实有效且值得投入**，直接支持 QuaDMix 的价值主张。

### 20. Rarity（数据稀有度）是低成本正交质量信号

Solar Open 2 (#42) 业界实践使用 quality + rarity 双信号策展。Rarity = 1/文档频次 或 1/重复度，计算成本极低（语料统计），与现有质量信号正交。

**结论**：QuaDMix 当前 5 个质量信号都是"内容质量"维度，缺少"数据稀有度"维度。Rarity 可作为第 7 个正交信号（与 PreSelect predictive strength 第 6 维、WebGraphMix 中心度正交维度并列），是本次搜索中 ROI 最高的新信号。

### 21. 优化器可替代数据配比的部分功能（val_loss 不充分的终极证明）

Nexus (#51, 清华+字节跳动) 从优化器角度证明了一个惊人的事实：

- **同一 pretraining loss（差 0.004），GSM8K 差 15 分**（44.0 vs 59.0）
- 不是 proxy 太小的问题——3B 模型上仍然如此。val_loss 信号本身缺失了"优化器收敛到 close 还是 distant minimizer"这一完整维度
- Nexus 通过最大化各域梯度余弦相似性，强制模型收敛到 close minimizer
- **Nexus = 隐式数据混合器**：数据少的域自动被补偿，降低对人工配比的敏感度

**对 QuaDMix 的影响**：
1. 这是 val_loss 不充分的**最强实锤**——不是噪声问题，是整个优化维度缺失
2. 提供了第三条破局路线（Route C）：不改配比，改优化器，与 Route A/B 正交可叠加
3. 如果 Nexus 真能自动平衡域，QuaDMix 精确配比的价值打折扣
4. 代码改动极小（~20 行），几乎零计算开销，可立即在 nanochat 中验证

**结论**：Nexus 是低成本、低风险的正交方法。它不替代数据配比优化，但可能显著降低对精确配比的依赖。建议立即验证。

## 对 QuaDMix 的行动建议

### 立即行动

1. **修复 Proxy LR** — 降低 proxy 的 learning rate，验证与大模型结果的相关性
2. **实现 Capacity-Aware Mixture Law** — 在 LightGBM 回归中加入 capacity-aware 特征
3. **记录域梯度统计量** — 在 proxy 训练中记录每个域的梯度范数和方差（FastMix + PiKE）
4. **加入 pairwise 域特征** — 在 LightGBM 特征中加入 u_k × u_{k'} 捕捉二阶协同（Domain-Aware SL）
5. **记录 proxy loss 轨迹 + 训练 trajectory-conditioned LightGBM** — RegMix-D 证明这是 ROI 最高的改进（+0.5-1%，成本仅修改 logging + 重训回归模型）
6. **加入 Rarity 信号（第 7 维质量特征）** — Solar Open 2 (#42) 业界验证，rarity = 1/文档频次，计算成本极低，与现有 5 个质量信号正交（预期 +0.5-1%）

### 短期行动

7. **加入网页图中心度信号**（#30）— 计算 Common Crawl host-level betweenness centrality，乘法组合 + 50/50 混合，预期 +1.5%，成本 9 GPU-hours
8. **对比 Token-Count Baseline** — 实现 token-count heuristic 作为 baseline
9. **调整高质量数据重复次数** — 高质量数据重复 15-20x；参考 InfoLaw (#37) 建模重复递减收益
10. **参考 DataMan 质量标准** — 检查 Eq.1 的质量信号是否覆盖 14 条标准
11. **验证集域覆盖度检查** — 确保 CAP v1/OpenHermes 不偏向特定域（Domain-Aware SL + Google MDE）
12. **PCA 检查搜索空间有效维度** — 对 proxy 实验数据做 PCA（Domain-Aware SL 发现 8 域只有 2 个有效方向；Explaining DMSL #38 用更少参数更高精度）
13. **检查域定义与 proxy 梯度一致性**（#33）— 验证 22 个 Dewey Decimal 域是否与模型梯度感知对齐
14. **在目标训练加域 loss 监控 patch**（DomainPilot #39）— ~30 行代码注入训练框架，在切换点查询回归模型调整配比，Qwen3-1.7B 验证 +2-3.8%
15. **验证 Noise Reduction 理论**（Explaining DMSL #38）— 检查 quality-top-k 是否符合"最优权重向难域偏移"的理论预测

### 长期探索

16. **探索 GP Surrogate**（#29）— 在低维子空间（5-10 主域）上尝试 GP + GP-UCB 主动采样
17. **探索 FastMix 在线优化模式** — 550x 搜索加速，但需注意工业数据风险
18. **探索 LoRA 插值 Proxy** — 比训完整模型便宜，比 model merging 灵活
19. **扩展到数据顺序优化** — 顺序是另一个重要维度
20. **合成数据配比** — 如果数据池包含合成数据，使用 30% 配比
21. **探索 model-driven reward（weight norms）**（HDS #40）— 作为第三信号（质量+域间影响+模型状态），RL 在线调度 Pile 省 44% 迭代、MMLU +7.2%
22. **探索 InfoLaw 式重复递减收益建模**（#37）— 在 overtraining 场景下显式建模重复的尺度依赖递减收益
23. **用 DataFlex (#49) 框架做标准化 baseline 对比** — 统一 sample selection/mixture/reweighting 三种范式
24. **在 nanochat 中验证 Nexus 优化器**（#51）— ~20 行代码实现 Algorithm 3（inner_model NSGD + 位移 pseudo-gradient），3:7 配比不变，验证 STEM+通用域是否同时提升。与数据配比正交，可叠加 Route A/B
25. **采用两阶段 diversity→quality 课程**（Nemotron 3 #60）— 前 75% 多样性优先，后 25% 质量优先。NVIDIA 550B 规模验证。与 QuaDMix quality-top-k 思路一致
26. **分阶段上下文+难度递进**（GLM-5 #58）— 中训练分阶段：先扩上下文，再加难度数据。200K 阶段增强 128K 窗口内性能
27. **考虑加入 SFT-style 数据**（Nemotron 3 #60）— sft-code/sft-stem/sft-general 在预训练中有效（Akter 2026 frontloading）
28. **考虑补充代码数据**（GLM-5/Qwen3/Nemotron 3）— 所有头部厂商中训练都含 code ≥15%，当前无 code 是最大差异
29. **长期升级到实例级配比**（Qwen3 #59）— 从域级配比升级到实例级，需要细粒度标签（educational value/fields/domains/safety）

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
- **2026-07-21**: 新增 5 篇数据配比/选择论文 (#29-33)，完成深度分析。关键发现：网页图中心度正交于 FastText（#30）；GP surrogate 同预算优于 LightGBM（#29）；人类域≠模型域（#33）。移除 #28 (LR Decay, 训练调度) 和 #34 (MIRA, mid-training)，聚焦数据配比/选择。总计 32 篇论文
- **2026-07-22**: 完成 PreSelect (ICML'25) 深度分析。核心发现：predictive strength 与 QuaDMix 5 个质量信号完全不重叠，可作为第 6 维；文档级选择比域级选择高 4.8%；同族模型至关重要；开源 fastText 分类器可直接集成
- **2026-07-24**: 新增 Survey (#35, CUHK-SZ) 和 RegMix-D (#36, U Tokyo+NII)，完成联合深度分析。关键发现：RegMix-D 用 loss 轨迹回归实现动态配比，128 proxy (25% 预算) 超越 RegMix 512，是 ROI 最高的改进路径；Survey 指出最优权重仅略优于 uniform，与 QuaDMix < quality-top-k 观察一致。总计 34 篇论文
- **2026-07-28**: 扩展搜索范围至业界大模型技术报告，新增 14 篇论文 (#37-50)，完成批量分析。核心发现：① InfoLaw (ICML'26, ByteDance) 信息论 scaling law 同时建模配比+重复，7B/425B 验证 0.15% 误差；② Explaining Data Mixing Scaling Laws (ICML'26) 理论解释 Capacity Competition + Noise Reduction，背书 quality-top-k；③ Solar Open 2 (250B MoE) 业界确认配比优化有效，反驳"仅略优于 uniform"；④ DomainPilot 域级 loss 引导两阶段优化 ~30 行 patch 部署；⑤ HDS (KDD'26) RL 在线调度三目标奖励；⑥ Rarity 数据稀有度信号是低成本正交新维度。**可信度验证**：ICML'26 (InfoLaw/Explaining DMSL/Reasoning Quality) + KDD'26 (HDS) + Data Intelligence 期刊 (Survey #2) 均经 arXiv comment/S2 DBLP 确认；DomainPilot 降级至第三梯队（单作者无 venue无机构）。总计 48 篇论文
- **2026-08-10**: 新增 Nexus (#51, 清华+字节跳动 Seed, arXiv 2026.4)，完成深度分析。**非数据配比论文，但直接影响配比价值判断**。核心发现：① 同一 pretraining loss（差 0.004），GSM8K 差 15 分——val_loss 不充分的终极证明，不是 proxy 太小的问题而是信号本身缺失完整维度；② Nexus 优化器通过最大化各域梯度余弦相似性收敛到 close minimizer，自动平衡欠优化域（隐式数据混合器），降低对人工配比的敏感度；③ 数据混合实验 Math10/40/70 证明 Nexus 自动补偿数据少的域；④ 与数据配比正交可叠加（Route C），代码改动 ~20 行几乎零计算开销；⑤ SFT 后优势保持。新增第四梯队"相关方向"类别。总计 49 篇论文
- **2026-08-10**: 新一轮 arXiv 搜索（5 个新角度：梯度/影响力信号、scaling law、中训练、influence function、课程学习），新增 5 篇论文 (#52-56)。关键发现：① 独立再发现 DomainPilot (#39) 为 Tier 1——从不同搜索角度重新发现其高相关性，升级其状态；② Why Larger Models Learn More (#52) 从容量竞争角度解释 1M proxy 为何不能学 STEM 推理——小模型把神经元分配给高频域，稀有复杂域被牺牲；③ RISE (#53) forward-only 数据归因扩展到 32B，112× 压缩，可替代 val_loss 信号；④ Kernel Surrogate (#54, ICLR'26) 核函数代理 > 线性代理 40%，可替代 LightGBM；⑤ K-EXAONE 2.0 (#55) LG 750B MoE 披露"difficulty-focused mid-training"设计；⑥ Beyond Rephrasing (#56) 中训练专用合成教材。总计 54 篇论文。**注意**：Tier 1/2 中 DomainPilot/CAMEL/ADAPT/Explaining DMSL/Phase Transitions/Scaling Laws for Optimal Mixtures 已在索引中，本次搜索独立验证其高相关性
- **2026-08-10**: 新增 4 篇头部厂商 frontier 规模 Tech Report 深度分析 (#57-60)：DeepSeek-V4 (1.6T/49B, 32T tokens)、GLM-5 (744B/40B, 28.5T)、Qwen3 (235B/22B, 36T)、Nemotron 3 Ultra (550B/55B, 20T)。**核心发现**：① GLM-5 是头部厂商唯一有专门 Mid-Training 章节的论文——三阶段上下文递进(32K→128K→200K, 1T+500B+50B) + 160B SWE 数据；② Nemotron 3 Ultra 数据配比最透明——19 类别 Phase1/Phase2 比例公开，两阶段 diversity→quality 课程(75%/25%)，SFT-in-pretrain；③ Qwen3 实例级配比优化超越域级方法(DoReMi/RegMix)，三阶段预训练 S2 推理阶段本质是中训练；④ DeepSeek-V4 最不透明，仅一句话提到 mid-training。**关键差距**：所有头部厂商中训练都含 code ≥15%，用户当前无 code 数据是最大差异。新增文档 `docs/头部厂商TechReport_数据配比深度分析.md`。总计 58 篇论文
- **2026-08-13**: 完成 CAMEL 深度分析 v3 更新。**新增内容**：① 作者调查——确认清华期智研究所 (IIIS) 所属，Phase Transitions 有完整致谢但 CAMEL 无致谢/无资助/无硬件描述，55B MoE 算力来源不明；② 开源状态确认——完全无开源（无 GitHub/HF/代码声明/数据集名），对比同领域 2026 论文普遍不开源趋势；③ 可信度重评——理论可信（ICML'26+数学证明），实验不可验证（无代码/数据/算力不明），不做 DeepSeek 合作定论；④ 6 个可借鉴点梳理（容量分配视角/R_math≠R_know/loss-to-benchmark 映射/内在域分组/hourglass 采样/cooldown 定位）；⑤ PreSelect 互补分析——PreSelect 提供开源信号设计，CAMEL 提供理论框架和规模外推，两者结合是最有前景方案。新增文档 `docs/CAMEL_深度分析_v3.md`
- **2026-08-13**: 新增 3 篇微软/MSRA 团队数据配比方法深度分析 (#61-63)。**核心发现**：① Velocitune (ACL'25) 用学习速度动态调整域权重，通过同模型 scaling law 预测目标 loss，绕过 QuaDMix 跨模型外推问题——但 scaling law 稳定性不透明（逐域误差未报告，去掉 target loss → -4.1%）；② Data Mixing Agent (ACL'26) 是 Velocitune 同团队升级版，用 RL (CQL) 自动学习配比策略，benchmark 分数作 reward 完全绕过 val_loss 信号问题，跨模型+跨领域泛化（训一次用多次），3B 验证 +3.02% over RegMix，但一次性成本 1996 GPU 小时（384 轨迹）；③ DiReCT (ICML'26) annealing 阶段 Hessian 谱几何样本选择，100B 数据不可行（论文明确 future work），仅适用于冷却阶段小子集。**四篇联合对比**（含 ADAPT #22）已写入 `docs/ADAPT_深度分析_v2.md`。**结论**：Data Mixing Agent 是目前最对口的替代方案（域级 + benchmark reward + 跨模型泛化），但需评估 1996 GPU 小时成本和 50M→1.5B+ 规模迁移风险。新增文档 `docs/Velocitune_深度分析.md`、`docs/DiReCT_深度分析.md`、`docs/DataMixingAgent_深度分析.md`。总计 63 篇论文

## License

MIT
