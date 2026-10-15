# Data Mixture Insights

LLM 预训练数据配比算法的业界研究洞察项目。

> **English**: A tier-graded research-insight library on LLM pretraining **data mixture** — 140 papers tracked and adjudicated (T1–T4) under a standing credibility rubric (experiment scale / institution / community traction). Focus scenario: 1.5B-model mid-training (~1.8B tokens) evaluated on STEM accuracy. T1 highlights: scale-induced rank non-invariance (MAI-Thinking-1), capability-aligned OOD validation (SuperValid), merge-as-proxy mixture search (MergeMix), phase transitions in data mixing. Notes are in Chinese; the tier tables and quick index are language-light.

> **免责声明**：本项目所有分析、评估和结论均为个人主观看法，针对特定场景（1.5B 模型 + ~1.8B token 中训练，STEM accuracy 评估）进行的判断，不代表任何官方立场，也不构成对任何论文或方法的权威评价。论文的梯队划分、价值标记和行动建议仅为本项目内部工作参考。

## 项目目标

追踪和分析数据配比（data mixture）领域的最新研究进展，为 QuaDMix 等配比算法的实现和优化提供理论指导和实践参考。

## 核心问题

1. **Proxy 可靠性**：小模型 proxy 的信号是否可靠？如何修复？
2. **Scaling Law 外推**：小模型拟合的 scaling law 能否外推到大模型？
3. **相变现象**：数据配比中的相变如何影响训练效果？
4. **质量信号**：如何定义和量化数据质量？

## 目录结构

```
data-mixture-insights/
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
│   ├── ADAPT_深度分析_v2.md                            # ADAPT (ICLR'26) 在线 reweighting + 四篇联合对比表
│   ├── BETR_深度分析.md                                # BETR (Apple, 2025.07) benchmark 相似度文档级过滤，最低成本
│   ├── TaskAgnosticInfluence_深度分析.md               # Task-Agnostic Influence (COLM'26) 数据贡献动态分析，分阶段配比轨迹实证
│   ├── DIDS_深度分析.md                                 # DIDS (EMNLP'25) FIM 指导域影响 + loss 轨迹动态配比，直接对比 Velocitune 赢 3 分
│   ├── MIRA_深度分析.md                                 # MIRA (2026) 中训练源感知 rubric 筛选，25B 半量匹配 50B 全量
│   ├── STEM能力涌现分析.md                              # STEM 能力规模涌现分析：业界共识 + 相变定律 + DeepSeekMath 锚点 + nanochat-npu 1.5B 可行性判断
│   ├── QuaDMix失败分析_相变理论启示.md                   # QuaDMix 1.5B 失败根因（proxy gap + 相变阈值差异 + 容量分配相反）+ Phase Transitions 论文核心信息 + 域级相变 trade-off + 修复方向
│   ├── LogFloor_深度分析.md                               # LogFloor (arXiv 2026.08) Proxy 动态瓶颈顺序跨尺度迁移 + 顺序消解实验 + loss 绕过 accuracy 涌现的局限
│   ├── 核心未解决问题_相变与代理迁移.md                     # 综合分析：相变前信号能否预测相变后最优配比？各方法应对策略与失败分析，信息论视角的不可解性论证
│   ├── LLM数据选择配比问题定义.md                          # 问题形式化定义：符号/优化问题/目标函数五层困难/代理不可解性定理/loss-accuracy 鸿沟/本项目实例
│   ├── MixtureExperiment_深度分析.md                      # Scheffé 混合实验设计+I-最优设计；域间交互真实且重要；稀疏 Scheffé 跨尺度保持 ranking
│   ├── DomainRepetition_深度分析.md                        # 域级最优重复次数与 val loss 强负相关；同 TPP proxy 可迁移（清华+字节 Seed）
│   ├── Puro2B_深度分析.md                                  # Puro-2B (清华, 2026.08) 2B/1.4T 从零复现：课程/CMA 2.40× 最大算法收益 + proxy 分位切片协议 + post-training 存活实验
│   ├── CapacityDependent_深度分析.md                       # Capacity-Dependent (COLM'26) 1.5B-8B 三阶段容量模式：1.5B No Catch-up→选高似然数据 + 容量约束双重投影理论
│   ├── UltraData_深度分析.md                               # UltraData (OpenBMB, 2026.02) L0-L4 分层管理深读：L3 vs L1 MATH500 +9.24pp（格式重构>筛选）+ tiered vs mix +1.49pp + 两级验证协议
│   ├── MiniCPM5_2B_深度分析.md                             # MiniCPM5-2B (2026.09.07) 四 checkpoint 链深读：Base vs Midtrain diff 定标实验设计 + 配方黑箱边界
│   ├── Marin535B_Harrier_深度分析.md                       # Marin 535B Hero Run / Harrier 数据管线（2026.08-09 开源项目）：40 域×5 质量桶混合空间 + 三层权重选择协议 + 两级聚类/按内容类型校准质量分桶（ClimbMix 复现对照）
│   ├── NemotronCLIMB_深度分析.md                           # Nemotron-CLIMB (NVIDIA, 2025.04) 深读：ClimbLab 构建法 + 112-run 迭代搜索 + CPT vs from-scratch 配比形状结论（2026-10-13 补档）
│   ├── MergeMix_深度分析.md                                # MergeMix (人大高瓴+蚂蚁, 2026.01) 深读：merge-as-proxy 四步机制 + Taylor 一阶等价/二阶交叉项 + 12 配比真训保真 ρ>0.9 + Mini-MergeMix 用户版草稿（2026-10-13）
│   ├── SuperValid_深度分析.md                              # SuperValid (蚂蚁, 2026.05) 深读：capability 级 OOD 验证集四步管线 + 三层证据（16 模型观测/消融/2T 切换受控）+ CAP v2 用户版草稿（2026-10-13）
 │   └── 论文总表.md                                      # 全 134 篇论文统一汇总表（编号+梯队+标记+机构+Venue+文档路径+关键贡献）
└── README.md
```

## 可信度评估准则（2026-10-12 用户指定）

对论文/报告的评估，除既有可信度分层规则（**定理 > 受控机制/反例 > 方向性结论 > 具体数字**；"X 有用"正结论最易随规模蒸发，反例/算术账最耐用）外，以下三项为**准入级准则**，评估任何工作前必须核对：

1. **实验规模（重大参考因素）**：学术界常在小规模模型/数据上得出结论，**并不必然适用于大规模模型/数据**。必须显式记录实验规模（模型参数量+数据量+训练 tokens）并与工业界主流规模（>100B 参数/数十 T tokens）对照——相差 2 个数量级以上默认标 ⚠️ 规模外推风险；有多尺寸扫描（跨尺度验证）或工业规模佐证的可放宽。本项目目标场景锚点：1.5B 模型 / 1.8B tokens 中训练。
2. **作者与发布机构**：作者背景（理论/统计/工程）、机构类型（头部厂商实验室 / 知名学术组 / 初创 / 单作者无机构）是可信度的重要考量，录入时在行内标注。
3. **业界关注度**：受业界关注程度（被大厂 TR 引用、工业采用、社区讨论度）是可信度的重要正信号；缺乏关注度的工作在引用其结论前须回溯一手证据。

## 论文索引

### 重点论文快速索引

> 汇总所有标记论文（共 42 篇），附深度分析文档路径。**全部 140 篇论文统一总表见 [`docs/论文总表.md`](docs/论文总表.md)**（含梯队、标记、机构、Venue、文档路径）。

| # | 论文 | 标记 | Venue | 深度分析文档 | 一句话核心价值 |
|---|------|------|-------|-------------|--------------|
| 01 | **Phase Transitions in Data Mixing** | **[高价值/重要]** | NeurIPS'25 Spotlight | `PhaseTransitions_深度分析_v2.md` + `QuaDMix失败分析_相变理论启示.md` | 配比相变理论根基，r_thres ∝ M^(-α)，本项目 STEM 涌现分析与 QuaDMix 失败根因的理论基础 |
| 26 | FastMix | [高价值] | ICLR'26 | `FastMix_DomainAwareScalingLaws_深度分析.md` | 可微 bilevel 优化 550x 加速，proxy 架构同 QuaDMix，<0.5B proxy 工业不稳定警告 |
| 27 | Domain-Aware Scaling Laws | [高价值] | arXiv | `FastMix_DomainAwareScalingLaws_深度分析.md` | 域间协同量化 γ/σ，R²=0.91，pairwise 特征可加入 LightGBM |
| 36 | RegMix-D | [高价值] | arXiv | `Survey_RegMixD_深度分析.md` | 动态 loss 轨迹回归，128 proxy 超 512，ROI 最高改进路径 |
| 37 | InfoLaw | [高价值] | ICML'26 | `新论文批量分析_2026下半年.md` | 信息论 scaling law，同时建模配比+重复递减收益，0.15% 误差 |
| 38 | Explaining Data Mixing SL | [高价值] | ICML'26 | `ExplainingDMSL_深度分析_v2.md` + `新论文批量分析_2026下半年.md` | Capacity Competition + Noise Reduction 理论，背书 quality-top-k |
| 51 | Nexus | [高价值] | arXiv | `Nexus_深度分析.md` | 同 loss 差 15 分，优化器级隐式配比器，val_loss 不充分终极证明 |
| 58 | GLM-5 | [高价值] | Tech Report | `头部厂商TechReport_数据配比深度分析.md` | 头部厂商唯一专门 Mid-Training 章节，三阶段上下文递进 |
| 60 | Nemotron 3 Ultra | [高价值] | Tech Report | `头部厂商TechReport_数据配比深度分析.md` | 数据配比最透明，19 类别 Phase1/Phase2 比例公开 |
| 63 | Data Mixing Agent | [高价值] | ACL'26 | `DataMixingAgent_深度分析.md` | RL 学习配比策略，benchmark 作 reward 绕过 val_loss，+3.02% over RegMix |
| 64 | BETR | [高价值] | arXiv | `BETR_深度分析.md` | benchmark 相似度文档级选择，成本最低（~2-4 GPUh），2.1x CM |
| 66 | DIDS | [高价值] | EMNLP'25 | `DIDS_深度分析.md` | FIM+loss 轨迹动态配比，赢 Velocitune 3 分，开源，无 scaling law 依赖 |
| 52 | Why Larger Models Learn More | [新发现] | arXiv | `小Proxy不可用_多论文交叉验证总结.md` | 容量竞争：小模型牺牲稀有复杂域，解释 1M proxy 不能学 STEM |
| 53 | RISE | [新发现] | arXiv | *(索引+changelog 2026-08-10)* | Forward-only 数据归因，112× 压缩，扩展到 32B |
| 54 | Kernel Surrogate | [新发现] | ICLR'26 | *(索引+changelog 2026-08-10)* | 核函数代理 > 线性代理 40%，可替代 LightGBM |
| 65 | Task-Agnostic Influence | [有意思] | COLM'26 | `TaskAgnosticInfluence_深度分析.md` | task-agnostic 数据贡献度量（有循环论证批判，★★★☆☆） |
| 39 | DomainPilot | [独立再发现] | arXiv | `DomainPilot_深度分析.md` | 域级 loss 引导两阶段优化，~30 行 patch，无 proxy 无 loss≠value 风险 |
| 68 | LogFloor | [高价值] | arXiv | `LogFloor_深度分析.md` | Proxy 动态瓶颈顺序可跨尺度迁移（70M→12B），顺序消解 60.3%→10.3%→6.0%，但 loss 绕过 accuracy 涌现 |
| 70 | Data Mixing as Mixture Experiment | [有意思] | arXiv | `MixtureExperiment_深度分析.md` | Scheffé 混合实验设计+I-最优设计；域间成对交互真实（web 文本是交互放大器） |
| 73 | **MAI-Thinking-1** | [高价值] | Tech Report | `MAI_Thinking_1_深度分析.md` | §2.5 数据配比 6 页：rank non-invariance（千 proxy 实证否定 RegMix 假设）、hierarchical local/global search、memorization-aware epoch capping、完整配比表 |
| 74 | Nemotron-CLIMB | [高价值] | arXiv | `NemotronCLIMB_深度分析.md` | **用户基座预训练数据来源**：聚类迭代搜索（ClimbLab 1.2T + ClimbMix 400B），优于 DoReMi/RegMix |
| 82 | **MobileLLM-R1** | [高价值] | ICLR'26 | `MobileLLM_R1_深度分析.md` | **2025-2026 最新**：cross-capability self-influence 配比+LOO ablation+Pareto，2T tokens 推理涌现，与 MAI §2.5 可比 |
| 85 | D³-MOPD | [有意思] | arXiv | *(索引+changelog 2026-09-02)* | 按域收敛信号（reverse-KL）零开销动态调度域配比，N=1-2 友好 |
| 84 | **Puro-2B** | [高价值] | arXiv | `Puro2B_深度分析.md` | 2B/1.4T 从零复现（$6.9K）；课程/CMA 配方 2.40× 最大算法收益；proxy 分位切片协议；post-training 存活实验（+1.77~2.02 pp 穿透 SFT） |
| 88 | **Capacity-Dependent Data Selection** | [高价值] | **COLM 2026** | `CapacityDependent_深度分析.md` | 1.5B-8B 数学蒸馏三阶段容量模式：1.5B 处 No Catch-up 最深区→选高似然（学生分布匹配）数据；容量约束双重投影理论 |
| 94 | KD During Mid-Training | [新发现] | arXiv | *(索引+changelog 2026-09-03)* | 蒸馏目标设计（非数据配比，用户管线无教师不可直接用）；间接价值="数据价值依赖训练阶段"机制+条件性适用（将来掺教师生成数据时 forward KL 的坑与 Switch 修复） |
| 95 | REER-PT | [有意思] | arXiv | *(索引+changelog 2026-09-03)* | PPL 作优化信号修复（而非过滤）难 continuation：插入推理注释，680M +2.07 pp；与 PPL 过滤线互补 |
| 96 | PLES | [新发现] | arXiv | *(索引+changelog 2026-09-03)* | 不确定性缩减采集函数估计最优超参 scaling law，<1/10 网格算力；proxy 实验设计可借鉴（选最能减少外推不确定性的点） |
| 98 | OpenEuroLLM Scaling Laws | [新发现] | arXiv | *(索引+changelog 2026-09-03)* | LR/batch scaling law + WSD annealing 增益广谱测量 + overtraining regime 显式建模；全部预训练 run 开源可当 proxy 拟合外部验证集 |
| 101 | **UltraData: Tiered Data Management** | [高价值] | arXiv | `UltraData_深度分析.md` | L0-L4 分层数据管理框架+全开源；UltraData-Math 290B 在 MiniCPM-1.2B 上 MATH500 +3.62pp vs Nemotron-CC（用户 1.5B 规模重合参照）；**深读：L3 vs L1 +9.24pp（格式重构>筛选）、Math-L3 跨域迁移、tiered vs mix +1.49pp、Decay Verification=用户 CPT 同构协议** |
| 106 | Curriculum as Transport | [有意思] | COLM'26 | *(索引+changelog 2026-09-14)* | Wasserstein 课程路径解耦 ordering/exposure/端点平滑/pacing；无普适最优顺序、收益非曝光效应——三档顺序实验设计参考；⚠️ 合成套件非 LM 预训练 |
| 110 | **Marin 535B Hero Run / Harrier 数据管线** | [高价值] | 开源工程项目 | `Marin535B_Harrier_深度分析.md` | 535B-A23B/18T 完全开源项目：40 语义域×5 质量桶混合空间+BO/swarm/梯子三层权重选择+三阶段换混；**配比轴**：共享前缀多种子协议、uncheatable held-out、外推回测校正；**聚类轴（ClimbMix 复现）**：5k→40 两级聚类、按内容类型校准质量分桶、微簇基直方图修复 CLIMB 位置性预测器缺陷；⚠️ BPB 目标非 STEM accuracy，数值不迁移 |
| 112 | **Stress-testing Alignment Midtraining** | [有意思] | Arcadia Impact（UK AISI 合作，伦敦非营利） | | 第一个 AMT 系统 stress-test（虚构零污染设定，106B 全参，1M-1B tokens）：2% 冲突 SFT 推翻 190M token 中训练动机、EFT 构成>剂量、held-out 知识被 EFT 侵蚀、描述不如示范、RL 后"知识在行为不用"。对用户=中训练阶段约束：收益须 SFT 后测（#84 互证）、语料须含示范（#101 互证）、未引发知识拿不到；⚠️ 价值观轴≠知识轴，数字不迁移 |
| 113 | **Conditional Transfer from Controlled Pretraining Mixtures to Code** | [有意思] | arXiv 2610.11548（单作者 Ohad Rubin，无 code） | | **混料比例取中间值最好（75/25）；按训练 loss 自动调比例会把真正有用的数据砍掉**——loss 降得快≠最后有用（单作者受控实验：引方向不引数字） |
| 114 | **Generalization Dynamics of LM Pre-training** | [高价值/新发现] | arXiv 2609.33150（Berkeley/Stanford/GDM，无 code） | | **训更久≠更聪明：常规成绩继续涨，深层泛化/鲁棒不涨甚至倒退**（4.5T 检查点经相同后训练全面优于 4.9T 终点）；相邻检查点深层能力会剧烈跳变而常规分数平稳——**中间检查点必须保存、终点要比较后再选**（细节：核心问题文档 §2.10） |
| 117 | **OptiSelect: How does the Optimizer Shape Data Curriculum?** | [有意思] | arXiv 2610.03432（EPFL Jaggi 组，无 code） | | **边训边挑数据不是免费的：挑数据要多读一倍 token，算总账反而不如不挑**；且有没有用取决于优化器（Lion/Muon 下基本白挑），打分方法与优化器要分开选 |
| 122 | **MDE: Mixtures of Data Expert Models** | [高价值] | arXiv 2502.15950（2025.02，Google DeepMind，补档） | *(索引+changelog 2026-10-12)* | 选对验证目标比换方法重要得多：end-task 验证 loss 与下游相关 0.844，域 loss 只有 0.282；k 个域专家小模型的 loss 可廉价近似任意混合配比，当回归特征后排名 0.81→0.95 |
| 124 | **Repetition Mismatch** | [新发现] | EMNLP 2026 Main | *(索引+changelog 2026-10-12)* | 小规模配比实验外推失败的元凶=重复率失配（预算变大时小数据集的重复率在变，最优配比随之漂移）；子采样到目标重复率再搜：1/16 tokens 在 1.17B 上把最优配比误差从 0.85 压到 0.10 |
| 130 | **MergeMix** | [高价值] | arXiv 2601.17858（人大高瓴+蚂蚁，ICML 模板） | `MergeMix_深度分析.md` | 把模型合并权重当配比搜索的低耗代理（每域 5B token 专家≈试验预算 2.5%）：8B/16B MoE 工业级 mid-training 持平/超过人工调参、成本降 100×、12 配比真训对照 ρ>0.9；**8B 搜出的配比直接用于 16B 仍超人工基线**（proxy→target 跨尺度正面证据）；沿 #74 Nemotron 四域划分；静态一次搜够+math 最优点在混合区（r<1.0 证据链）。**2026-10-13 用户裁定升 T1** |
| 132 | **SuperValid** | [高价值] | arXiv 2605.28179（蚂蚁集团） | `SuperValid_深度分析.md` | capability 级 OOD 验证集构造配方（benchmark→知识因子→语料检索→改写扩展，6 域 86k 样本离线造一次全 run 复用）：16 开源模型观测 MSE 1.51e-3（Qwen2.5-1.5B 在列）；**2T tokens 受控切换实验：IID loss 大跳而 SuperValid loss 平稳**——扫配比期间的公平监控信号；#122 MDE 的工具化，CAP v1 升级配方。**2026-10-13 用户裁定升 T1** |
| 134 | **Validity Threats for FM Research** | [有意思] | arXiv 2606.05029（Tübingen AI Center+Vienna） | *(索引+changelog 2026-10-13)* | 方法论元批判：proxy 实验/观测研究/单 run 设计各有特征性效度威胁（借社会科学四类效度）；**跨尺度可预测性无法只靠 proxy 实验自证，必须更大尺度验证**——400M proxy 线与 N=1-2 设计的威胁清单 |
| 137 | **How LR Decay Wastes Your Best Data** | [高价值] | arXiv 2511.18903（清华 PACMAN，**#84/#125 同组**） | *(索引+changelog 2026-10-14)* | 升序质量课程必须配恒定 LR——标准衰减把课程收益吃掉；修复=温和衰减或 checkpoint 平均替代衰减（1.5B/30B，+1.64%）；与 #84 Puro CMA、#130 MergeMix WSM 三方独立收敛；**即 2026-07 移除的旧 #28，按配方五轴口径重录** |

> **标记说明**：`[高价值/重要]` = 项目理论根基；`[高价值]` = 直接可用方法/数据；`[新发现]` = 新视角启发；`[有意思]` = 有价值但需批判看待；`[独立再发现]` = 不同搜索角度独立验证高相关。

### 第一梯队：高可信度 + 高价值

| # | 论文 | 机构 | Venue | 关键贡献 |
|---|------|------|-------|---------|
| 01 | **Phase Transitions in Data Mixing** | 清华 | NeurIPS'25 Spotlight | **[高价值/重要]** 揭示配比相变现象，理论证明 STEM 知识获取存在尖锐相变 `r_thres ∝ M^(-α)`，(M, r) 联合曲面，本项目 STEM 涌现分析的理论根基 |
| 03 | Scaling Laws for Optimal Data Mixtures | Apple | arXiv | 参数化 scaling law，跨模态验证 |
| 04 | DeMix: Model Merging | 小红书+东大 | ICML'26 | Model merging 替代 proxy 训练 |
| 09 | CAMEL: Capacity-Aware Mixture Law | 清华 | arXiv | Phase Transitions 的实践版 |
| 10 | Proxy Reliability | Princeton/Berkeley | ICLR'26 | 降低 LR 修复 proxy 信号 |
| 21 | PiKE: Adaptive Data Mixing | Google Research+USC | arXiv | 梯度交互自适应配比，1.9x 收敛加速 |
| 27 | **Domain-Aware Scaling Laws** | MIT+Microsoft Research | arXiv (2026.7) | **[高价值]** 域间协同量化 γ/σ，R²=0.91，协同≠语义重叠，pairwise 特征可直接加入 LightGBM |
| 31 | PreSelect | HKUST+Vivo AI | **ICML'25** | Predictive strength + fastText，10x 计算缩减，开源 |
| 37 | **InfoLaw** | ByteDance | **ICML'26** | **[高价值]** 信息论 scaling law，同时建模配比权重+重复递减收益，7B/425B 验证，0.15% loss 误差 |
| 38 | **Explaining Data Mixing Scaling Laws** | - | **ICML'26** | **[高价值]** 理论解释配比 scaling law：Capacity Competition + Noise Reduction，理论背书 quality-top-k，解释 proxy→target 失效 |
| 58 | **GLM-5** | 智谱 AI + 清华 | Tech Report (2026.2) | **[高价值]** 744B/40B MoE, 28.5T tokens。**头部厂商唯一专门 Mid-Training 章节**：三阶段上下文递进(32K→128K→200K, 1T+500B+50B) + 160B SWE 数据 + 合成长上下文。**2026-10-12 用户裁定 T4→T1（可信度准则：工业规模 744B/28.5T+唯一 Mid-Training 章节）** |
| 60 | **Nemotron 3 Ultra** | NVIDIA | Tech Report (2026.6) | **[高价值]** 550B/55B MoE, 20T tokens。**数据配比最透明**：19 类别 Phase1/Phase2 比例公开，两阶段 diversity→quality 课程(75%/25%)，SFT-in-pretrain，消融数据丰富。**2026-10-12 用户裁定 T4→T1（可信度准则：工业规模 550B/20T+19 类配比全公开）** |
| 73 | **MAI-Thinking-1** | Microsoft AI | Tech Report (2026.08) | **[高价值]** 35B/1T MoE, 30T+3.55T tokens。§2.5 数据配比 6 页：Pareto frontier + 183 proxy 模型；**rank non-invariance**（千个 proxy 实证否定 RegMix rank invariance 假设，stem-heavy 小尺度赢但 code-heavy 大尺度反超）；hierarchical local/global search；memorization-aware epoch capping（NLL<0.01 分数）；8× 重复上限；完整配比表（Code 54.6%/STEM 15.8%/Math 5.4%）；中训练 STEM:Code:Other=35:55:10。**2026-10-12 用户裁定升 T1**（与 #74 CLIMB/#110 Marin 并列第一梯队） |
| 74 | **Nemotron-CLIMB** | NVIDIA | arXiv (2025.04) | **[高价值]** **用户基座预训练数据来源**。聚类迭代搜索（embed→cluster→proxy→predictor），ClimbLab 1.2T（20 clusters）+ ClimbMix 400B；优于 DoReMi/RegMix；1B 模型超 Llama-3.2-1B +2.0% |
| 110 | **Marin 535B Hero Run / Harrier 数据管线** | Open Athena（Stanford 起源：David Hall+Percy Liang；embedding=微软 harrier-oss-v1） | 开源工程项目（2026.08-09） | **[高价值]** 535B-A23B/18T tokens 完全开源训练项目（代码/issue/权重/语料构成/逐 checkpoint 指标全公开）。**混合空间**：292 源→嵌入（harrier-oss-v1-0.6b，1024d int8）→5k 微簇→40 语义域（每域≥1% 人口）×5 质量桶（按内容类型校准截断）=200 单元格，23.11T tokens。**权重选择三层**：BO（monotonic regret @d768）→d512 共享前缀多种子 swarm（新混/旧混各 3 种子从同一 checkpoint 恢复训练至终点比 final BPB）→缩放梯子事前验证（60-80% 检查点线性外推+小宽度回测残差校正→1.07× 等算力加速）；桶级曝光上限 2.96 epochs；7 个零权重 uncheatable held-out。**三阶段调度**：initial/main/cooldown，Q4 质量桶冷却期 22.9→28.0%（annealing-on-quality 第三独立来源）。**对用户**：① r 扫描协议（共享前缀+多种子+held-out 零曝光）直接套用；② ClimbMix 复现对照：两级聚类+最小人口约束、质量按内容类型校准、微簇基直方图特征 h=V·w 解决重聚类搁浅（CLIMB 位置性预测器被点名反面教材——重聚类即重触发 112-run 扫描）。⚠️ 目标函数 macro BPB 非 STEM accuracy（#68 同构陷阱），535B/18T 差 4 数量级，数值不迁移 |
| 130 | **MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging** | 人大高瓴+蚂蚁集团 | arXiv (2026.01, 2601.17858，ICML 模板) | **[高价值]** merge-as-proxy：每域 5B token 专家（试验预算 2.5%）→参数空间合并权重=配比性能代理→对下游 benchmark 选配比；Taylor 展开论证权重混合≈数据混合（math/code 双域受控验证）。8B/16B MoE（各 1.4B 激活，**AdamW** 同用户栈）工业级 mid-training（沿 #74 四域划分、恒定 LR+merge 16 checkpoints 模拟 annealing、200B 预算）：math/code/reasoning 超人工调参基线（已上线生产）且其余域近无损，**成本降 100×、12 配比真训对照排序保真 ρ>0.9（知识类 0.98）、8B 配比直接迁移 16B 仍超人工**——proxy→target 跨尺度正面证据（与 #73 反例对照）；静态一次搜够（50% 处重搜≈初始——最优配比是 mid-training 的稳定内在属性）；**math 最优点在混合区不在纯 math**（r<1.0 证据链）；4 粗域→16 细簇层级搜索全自动超手工；与 #04 DeMix 构成 merge 家族"搜索代理 vs 最终合并"分工。**2026-10-13 用户裁定升 T1**（准则：8B/16B 工业规模+12 配比真训保真验证；⚠️ 无 venue、MoE、内部 benchmark） |
| 132 | **SuperValid: Capability-Aligned OOD Validation** | 蚂蚁集团 | arXiv (2026.05, 2605.28179) | **[高价值]** IID val loss 在配比变化时会漂、capability 级 OOD 验证集不漂。四步管线：benchmark 题目→LLM 抽知识因子→FineWeb 检索（OLMOTrace）→LLM 过滤→**改写扩展**成知识密集验证文本（6 域 86,102 样本，离线造一次全 run 复用）。三层证据：①16 开源模型观测（Qwen2.5-0.5B 到 Kimi-K2，**Qwen2.5-1.5B 在列**）loss-能力 sigmoid 拟合 MSE 1.51e-3；②消融：挖空题目 7.29e-3→只检索 2.31e-3→完整管线 6.73e-4（**改写扩展贡献最大**）；③**关键受控：2T tokens from-scratch 两阶段（第二阶段加合成比例）——IID loss 大跳而 SuperValid loss 平稳**；193M-6.17B MoE 上 log-linear scaling law 成立。#122 MDE"验证目标选择>方法选择"（§26）的工具化——CAP v1 升级配方（检索池换成 ClimbLab/ClimbMix）。**2026-10-13 用户裁定升 T1**（准则：2T 受控+16 模型观测+蚂蚁工业组；⚠️ 无 venue——4.5 个月新文引用 1 为时间因素非质量信号、观测为相关性、自训系列 active 10M-324M 偏小（观测覆盖至 Kimi-K2 级）、2T 实验模型尺寸未明说、中文评测烙印） |

### 第二梯队：大厂经验

| # | 论文 | 机构 | Venue | 关键贡献 |
|---|------|------|-------|---------|
| 11 | DataMan | 阿里 Qwen | ICLR'25 | 14 条质量标准，447B 标注 |
| 12 | UtiliMax / MEDU | Meta FAIR | arXiv | Token-count baseline，200x 加速 |
| 13 | Repetition-Aware Scaling | Apple | arXiv | 高质量数据可重复 15-20x |
| 14 | Synthetic Data in Pretraining | Meta | EMNLP'25 | 30% 合成 + 70% 自然 |
| 20 | Chameleon | EPFL | **ICML'25** | KRLS 领域权重，免重训迁移。**2026-10-12 梯队裁定 T1→T2（可信度准则：实验最大仅 1.2B）** |
| 22 | ADAPT: Online Reweighting | Cambridge+Oxford | **ICLR'26** | 在线 reweighting 优于离线选择，泛化性强 |
| 26 | **FastMix: Gradient Descent Mixing** | 腾讯混元+港大 | **ICLR'26** | **[高价值]** 可微 bilevel 优化，1 proxy 550x 加速，proxy 架构与 QuaDMix 完全相同，Section 4.3 工业踩坑经验直接适用。**2026-10-12 梯队裁定 T1→T2（可信度准则：预训练验证最大 1B）** |
| 30 | **WebGraphMix** | Princeton (Danqi Chen) | arXiv | 网页图中心度作为 training-free 正交质量信号，与 FastText 正交，乘法组合 +1.5%。**2026-10-12 梯队裁定 T1→T2（可信度准则：规模核查 400M/1B 模型、8B/28B tokens，距工业 2+ 数量级；无 venue）** |
| 36 | **RegMix-D: Dynamic Mixing** | U Tokyo+NII | arXiv (2026.6) | **[高价值]** RegMix 动态化扩展，loss 轨迹回归，128 proxy (25% 预算) 超越 RegMix 512，Online +0.37% 开销，proxy 大小 1M-120M 无差异 |
| 40 | **HDS: Holistic Data Scheduler** | Dang, Ma, Liao | **KDD'26** (S2 DBLP 确认) | 在线 RL（SAC）数据调度，三目标奖励（质量+域间影响+weight norms），Pile 省 44% 迭代，MMLU +7.2%，代码开源 |
| 41 | TANDEM: Twin Networks | Kuaishou (12 作者) | arXiv (2026.6) | Bi-level 配比优化，twin networks（proxy+reference）差值度量数据效能，理论保证 |
| 42 | **Solar Open 2** | Upstage | Tech Report (2026.7) | 250B-A15B MoE 业界实践：quality+rarity 策展 + mixture-ratio 优化，20T→10T 精炼 |
| 61 | **Velocitune** | Microsoft | **ACL'25** | 学习速度动态域配比：V=(L_cur-L_target)/(L_init-L_target)，同模型 scaling law 预测目标 loss，exp 权重更新。绕过 QuaDMix 跨模型外推问题。scaling law 稳定性不透明（逐域误差未报告）。11 引用 |
| 62 | **DiReCT** | MSRA+HKUST(GZ) | **ICML'26** | Annealing 阶段样本选择：Hessian 谱几何 → flat/stiff 子空间 → 选 flat 方向梯度大的样本。随机投影 k≈34。355M-1.1B 验证。100B 数据不可行（论文明确 future work）。代码开源 |
| 63 | **Data Mixing Agent** | Microsoft | **ACL'26** | **[高价值]** RL 学习域配比策略（Velocitune 升级版）。2.1M agent + CQL 离线 RL + 50M proxy 384 轨迹（1996 GPUh 一次性）。benchmark 分数作 reward（绕过 val_loss 问题）。跨模型+跨领域泛化。3B 验证 +3.02% over RegMix |
| 64 | **BETR: Benchmark-Targeted Ranking** | **Apple** | arXiv (2025.07) | **[高价值]** benchmark 相似度文档级数据选择。Arctic-Embed + rank scoring + FastText 三步流程，500+ 模型 scaling law 验证。2.1x CM over DCLM-Baseline（4.7x over 无过滤）。**最小模型需最激进过滤**（F_opt ∝ C^0.25，小模型 top 3%，大模型 top 30%）。FastText (75.6% acc) 击败 302M LM classifier (82.4% acc)。成本最低（~2-4 GPUh）。17 引用。被 Phase Transitions 论文引用。文档级 ≠ 域级，但可与 ClimbMix 串联使用（BETR 过滤 → ClimbMix 配比） |
| 65 | **Task-Agnostic Training Data Influence** | NAIST+NII+早稻田+东大+哥本哈根IT+东北大 | **COLM'26** | **[有意思]** task-agnostic 数据贡献度量。**深度批判**：θ* 内生性导致三层循环论证（梯度幅度混淆+自指+域比例偏向），交叉反转可由 loss 动态更简洁解释（先易后难）。对配比决策指导性有限（★★★☆☆），但连接到 Velocitune 的 loss 动态信号。**2026-10-12 梯队裁定 T1→T2（梯队与 [有意思] 标记不齐+规模待核）** |
| 66 | **DIDS** | HKUST+MetaX+Alibaba+浙师大+苏大 | **EMNLP'25** | **[高价值]** FIM 指导域影响 + loss 轨迹动态配比。绕过 Velocitune scaling law 依赖。直接对比 Velocitune 赢 3 分（62.3 vs 59.3）。10% 数据超全量训练。开源代码（原型质量，两套不一致实现）。梯度聚类默认关闭。额外开销 1.9%。**2026-10-12 梯队裁定 T1→T2（可信度准则：验证仅 939k 样本 SFT/1B tokens、自述 100B 聚类瓶颈、原型级关注度；[高价值] 标记保留——在线方法方向价值不变）** |
| 67 | **MIRA** | SJTU+UBC+Langboat+北航+IQuest | arXiv (2026.05) | 中训练源感知 rubric 筛选。自锚定 rubric 发现（Kimi-K2.6 教师 → 15 锚定维度/组 → Qwen3.5-35B 学生蒸馏）。25B 半量匹配 50B 全量（64.20 vs 63.83）。可靠性 mask 12% 源-维度对。样本级筛选非域级配比，与 DIDS 互补（MIRA 筛域内 → DIDS 配域间）。无代码无 venue |
| 68 | **LogFloor** | Choi et al. (Stanford/KAIST, Cioffi) | arXiv (2026.08) | **[高价值]** Proxy 动态瓶颈顺序可跨尺度迁移。70M→12B replay 省 30.9%。顺序消解：正确 60.3%→静态 10.3%→逆转 6.0%→随机 6/8 命中。用 Qwen2.5-1.5B。**核心局限**：用 loss 绕过 accuracy 涌现，loss 顺序 ≠ accuracy 涌现顺序（STEM 相变未验证），用无相变任务（bAbI/MMLU）回避了核心问题 |
| 71 | Scaling Domain Data Repetition | 清华+字节 Seed | arXiv (2026.08) | 域级最优重复次数与 val loss 强负相关（非数据量）；同 TPP proxy 可迁移。**对用户场景帮助有限**：TPP≈1.2 vs 验证 TPP=20，数据量瓶颈非重复策略。**2026-10-12 梯队裁定 T1→T2（价值错位：TPP≈1.2 与用户 TPP≈1200 不同区）** |
| 72 | **Cross-Difficulty Optimization Dynamics** | 复旦 (Zhikai Ding, Ziyi Ye) | arXiv (2026.08) | Relative Transfer 度量跨难度知识迁移+TDCS 动态采样。课程学习优化动力学解释。概念上接近域间交互（难度级）+在线适应（DomainPilot 家族）。场景为 post-training 课程学习 |
| 75 | **OLMo 2** | Allen AI (Team OLMo) | arXiv (2024.12) | Dolmino Mix 1124 数据混合；annealing curriculum（高质量数据收尾）；**全开源**（数据+代码+recipe+中间检查点）；single-source microanneals 方法（被 MAI-Thinking-1 §2.5.4 直接引用）。**§4.5 Checkpoint Soups（原文核实）**：中训练收尾独立 anneal 多次（7B: 3×50B 不同数据顺序）再平均权重 |
| 76 | **Llama 3 Herd of Models** | Meta | arXiv (2024.07) | 15T tokens 预训练数据；annealing phase（高质量数据收尾）；quality classifiers pipeline；最广泛引用的预训练数据工程参考 |
| 77 | Phi-4 Technical Report | Microsoft | arXiv (2024.12) | 合成数据配比为核心创新；"beyond distillation"（合成数据超越教师模型）；数据质量优先策略 |
| 80 | SmolLM2: When Smol Goes Big | Hugging Face (Ben Allal et al.) | arXiv (2025.02) | 4 阶段 evolving mixture，ablation 驱动配比调整。**已偏旧**（2025.02，18 个月前），参考价值有限 |
| 82 | **MobileLLM-R1** | Meta AI | **ICLR'26** (arXiv 2509.24945, 2025.09) | **[高价值]** **§2 配比 3 页+附录，与 MAI-Thinking-1 §2.5 可比**：cross-capability self-influence 配比方法（AutoMixer 框架）、LOO ablation、Pareto frontier。2 阶段配比表（P1: Web 63.75%/Code 10.66%/Math 6.93%；P2: Math 23.33%）。2T tokens 足够推理涌现（Qwen3 36T 的 11.7%）。**StarCoder 对数学收益 > OpenWebMath 对代码**（反转常识）。Mid-training influence score 收敛=数据耗尽信号 |
| 83 | **MobileLLM-Pro** | Meta Reality Labs | arXiv (2025.11, 2511.06719) | §4 offline data-mix simulation（SDM 框架）+ LOO continued tuning + 配比表（FineWeb-Edu 89.75%/Code 4.66%/Math 1.92%）。1.4T tokens。Specialist model merging 做域级 annealing。静态 offline 不如 R1 的迭代式 |
| 84 | **Puro-2B** | 清华 PACMAN (Wenguang Chen 组, Kaifeng Lyu 等 11 人) | arXiv (2026.08, 2608.27370) | **[高价值]** **Qwen2-1.5B 同款架构从零复现**：RTX 5090、$6.9K、1.4T tokens（700 TPP，与 ClimbLab 800 同区）。**深读**（`Puro2B_深度分析.md`）：课程/CMA 数据配方 2.40× 最大算法收益（376 桶源内升序课程+恒定 LR 续训+6 checkpoint 平均，模型平均单独无益）；proxy 分位切片协议（0.6B 共享基座+80% 占比爬坡+q00-q75）；**post-training 存活实验**（课程优势穿透匹配 SFT +1.77~2.02 pp）；P1→P2 配比 Math 7.2%→18.3%；Nemotron-CC-v2 主力（与 ClimbLab 同生态）。同组前置机制论文 **#137 LR Decay**（恒定 LR/checkpoint 平均组件的受控机制证据）。⚠️ 污染审计缺失+proxy 容量错位 |
| 88 | **Capacity-Dependent Data Selection for Reasoning** | Virginia Tech (Ruoxi Jia 组) | **COLM 2026** (arXiv 2026.08, 2608.13721) | **[高价值]** **1.5B-8B 数学 reasoning 蒸馏受控实验**（Llama 交叉验证）：三阶段容量模式——1.5B/3B **No Catch-up**（1.5B：1 epoch avg 24.16 vs 14.77，10/10 全胜，低似然 5 epochs 仍追不上）、7B 不稳定反转、8B 持续反转（avg 70.69 vs 66.64，epoch 2.25 交叉后持续领先）；小模型学低似然=死记；容量约束双重投影理论（数据张成 ∩ 学生可表示子空间）。**直接命中用户场景：1.5B STEM 数据应偏向基座 PPL 可及段**。**深读**：`CapacityDependent_深度分析.md` |
| 101 | **UltraData: Tiered Data Management** | OpenBMB/面壁智能 (Yudong Wang 等) | arXiv (2026.02, 2602.09003) | **[高价值]** **L0-L4 分层数据管理框架**（MiniCPM 系列数据底座 + #84 Puro-2B 的 UltraData-Math 出处）：L0 数学专用解析→L1 启发式清洗去重→L2 专有模型标注种子蒸馏轻量分类器全量质量分级→L3 重写/合成精修（Q&A/对话/教科书）；核心主张=数据按层级**跨训练阶段策略性分配**。**UltraData-Math 290B tokens**（L1 170.5B/L2 33.7B/L3 88B）：MiniCPM-1.2B 实证 MATH500 +3.62pp vs Nemotron-CC 4plus、GSM8K +3.34，code/general 不掉——**1.2B 与用户 1.5B 规模重合的最直接公开数学预训练参照**。对用户：L2 学习式分级=PPL 分位过滤互补方案；L2/L3 可作 STEM 池高质量补充源；中训练取高层级数据。**深读**（`UltraData_深度分析.md`，2026-09-09）：① **L2 vs L1 增益小（MATH500 +1.42/GSM8K -1.74）而 L3 vs L1 +9.24pp MATH500/+7.13 GSM8K**——纯筛选不够、格式重构（QA/多轮对话/教科书化）才是数学跃迁来源，PPL 过滤线的定位校准；② **Math-L3 跨域迁移**：English +3.45（BBH +5.87/OBQA +15.40）、Code +3.80（MBPP +4.56）——STEM 重仓买通用推理的直接 1.2B 实证；③ **tiered vs mix 同配比 +1.49pp、后段增长 1.7×**（出场时机独立于配比；常识广度任务 HellaSwag/PIQA 轻微回退=深度换广度信号）；④ 三种验证协议（Efficient 10B 快筛/**Decay 100B=1.3T 基座+30/70 混合续训+LR 指数衰减 7.5e-4→3.75e-5**）=用户 proxy→主实验两级结构的工业版模板。⚠️ 30% 目标占比与用户 70% 方向相反、100B vs 1.8B 尺度差、L3 合成依赖 72B 教师（只能复用不能自产）。可信度高（官方，论文+数据+代码全开源） |

| 111 | **AutoData: Agentic Search for Pre-training Data Selection** | Weco AI 初创+UvA（NVIDIA DGX Cloud 算力；Yuxiang Wu 通讯） | arXiv (2026.09.17, 2609.19754，在投) | agent 在**可执行选择算法空间**（打分+分层+随机规则 DSL）搜索预训练数据选择，替代权重空间优化——**底座与用户栈同构**（nanochat GPT-2 + **ClimbMix 全池** 400B/5.53 亿文档选 2.6%，proxy d8/0.42B tokens/单 H100 10min）。**①精炼池质量轴已死**：DCLM 分类器倒退（0.9954 vs random 0.9484）、PPL 中段过滤倒退、RegMix（K=24/128 Dirichlet/LightGBM）≈random——与 Marin "LightGBM=弱基线"互证，Climb 级语料只有内容构成（r 轴）有残值；**②效果量校准**：最优选择 vs 随机仅 bpb ~0.1% 相对、CORE 22 任务全不显著（即使以 CORE 为搜索目标）——小尺度 accuracy 噪声主导（门3 旁证），proxy 发现 125M→897M 有效→1.3B 消失（支持 400M 海选+1.5B 决赛两点式）；**③**subsample 种子隔离协议（固定子样本只变种子、4 种子并行、配对 t 检验）+ 零过滤配方原语（多样性门控 PPL/三角核带状卫生分/格内收缩/Gumbel 锦标赛/1/3-1/2 随机骨架/**源序块先验**——利用 ClimbMix shard 顺序簇结构）可搬。⚠️ 1 天预印本无评审、code 已放出（MIT，2026-09-18）、from-scratch 选子集≠用户全量再加权 regime、摘要 CORE 不显著有轻度 overclaim |

| 115 | **When do data mixtures improve scaling laws?** | ISTA+Yale 理论组（Mondelli=Yale/ISTA 双栖、Misiakiewicz=Yale Stats，FWF+ERC 资助） | arXiv (2026.09, 2609.38011) | **配比轴理论边界**：高维混合回归（共享函数/异构协方差/规模异速增长 n₂=n^γ₂）minimax+ridge 显式 scaling law——**混合严格优于任一单数据集 ⟺ 辅助谱重尾（互补覆盖 δ>1）且规模适中增长（γ_c<γ₂<1）**，否则只动 prefactor；固定比例+重尾辅助渐近退化为辅助主导（正分布偏移）。词频级操作化：辅助域应覆盖目标域低频 token，两域分布太近则无改善；81.5M nanoGPT/SlimPajama 定性验证（γ₂=1.25 改善、0.75/1.75 不改善）。**对用户**：源选择=选互补覆盖而非冗余堆质量（与 #38 容量竞争互补）；r 最优值随预算增长漂移有理论根据（固定 r=0.7 边界+1）。⚠️ 风格化线性模型+81.5M 从零单实验，1.5B CPT 定性外推 |
| 122 | **MDE: Optimizing Pre-training Data Mixtures with Mixtures of Data Expert Models** | Google DeepMind/Research（Belenki, Agarwal, Shi, Toutanova） | arXiv (2025.02, 2502.15950, preprint) | **[高价值]** **验证目标选择 > 方法选择**（Table 1，510M/50K steps）：优化目标 avg-et（11 个 end-task 验证域）与下游 All 准确率 Spearman **0.844** vs avg-sp（7 个 SlimPajama 域，DoGe/DML 常用目标）**0.282**——换验证目标的收益远大于方法间差异；本项目 6 处深度分析引用的"Google MDE"即此文（2026-10-12 补档）。**方法**：Mixture of Data Experts——每域一个小专家模型，loss 加权聚合近似任意混合 loss（理论保证），作回归附加特征（~25 混合样本）；排名 Spearman 0.81→0.95（GBM）/0.65→0.95（线性）；仅训 k 个专家独立搜配比（3× 省算力）。**对用户**：CAP v1 域覆盖度检查依据；域专家 loss=400M proxy 线廉价特征（专家=小模型无容量错位）；RegMix 同族（LightGBM min-leaf=20 不适合低样本）。⚠️ SlimPajama/生成排序基准非 STEM accuracy；Marin #6326 "MDE 式特征工程"反例警示上限 |
| 123 | **What Is a Repeated Token Worth?** | FloatAI+ETH Zurich（Chai）+Xiong（独立研究者） | arXiv (2026-10, 2610.05591, 无 venue) | 多 epoch 定价几何：重复代价 ∝ z=(R-1)N/U 单变量定律（Chinchilla 骨架）；固定语料时更大模型更快到最优 epoch（127M→2B：15.3→3.6）；临界 epoch 数随每参数预算增长、几乎不随模型大小变；compute-optimal 同步长大 size+epochs。**14M-2B 八尺寸 SlimPajama/CommonCrawl（2B 达用户量级、跨尺度扫描）**。对用户：TPP≈1200 过训练区直接理论。⚠️ 非数学语料、无 venue、机构弱（FloatAI/独立研究者）、关注度未知 |
| 124 | **Repetition Mismatch: Why Data Mixture Experiments Don't Scale** | Imperial College London（Marek Rei 组） | **EMNLP 2026 Main**（arXiv 2606.07597） | **[新发现]** 小规模配比外推失败元凶=**重复率失配**：高质量数据集小→预算增长时其重复率改变→最优配比漂移，小实验无法预见；**修复=子采样匹配目标重复率**：1/16 目标 tokens 在 1.17B 上恢复最优配比（误差 0.10 vs 0.85）；不受控需 19%/44%/94% 预算（2/3/4 视界）；三源 757M 两次受控视界即恢复。**"是重复动力学而非规模本身决定小实验泛化"——直接命中本项目 proxy 外推核心问题**；400M proxy 线实验设计修正 |
| 125 | **Larger Datasets Can Be Repeated More** | 清华 PACMAN（#84 同组，Kaifeng Lyu 理论） | **ICLR 2026**（arXiv 2511.13421） | **定理级**：有效复用率 E(K,N)（一遍需放大多少倍数据才等价 K-epoch）——K 小时 E≈K（线性收益），K 大时饱和于随 N 增长的值（强凸 Θ(log N)）→**大数据集更耐重复**；修正 Muennighoff 2023"≤4 epoch 无差"（我们 #13 的引证源）：该常数实际依赖数据规模与分布；LLM 实证验证。对用户：1.26B 数学 token 重复预算上界的定理参照；与 #123 互补 |
| 126 | **How Much Is an AI Token Worth?** | UMD+Pangram Labs 等 | arXiv (2026.09, 2609.40295) | wild AI 文本配比轴：800 模型扫 AI:human 掺入比例——数据饥饿模型先降 human loss 后饱和反转为害、高预算模型几乎立即有害，Chinchilla 预测不了变号；benefit/harm 双项 scaling law（小模型拟合预测 3.6× 大模型、误差低 41%）。建议：目标为 human text 时过滤 AI 文本、**先重复 human 再扩 AI 语料**（连重复轴）、human/AI 分开报 val loss。开源 WildAI 83B 标注语料+800 模型。**训练用 nanochat recipe+Muon（与用户栈同 recipe）**。尺寸已核：19.9M-973M（268M 拟合→973M 验证）。⚠️ 最大 973M<1.5B、无 venue |
| 131 | **Optimal Splitting of Language Models from Mixtures to Specialized Domains** | Apple（**#03 原班**：Seto/Ablin/Filippova/Ye/Bethune/Katharopoulos/Grangier） | arXiv (2026.03, 2603.19149) | **split model training 的最优切换点**：scaling law 拟合 L_k(N,D,D')（DCLM 聚 K=16 域、1.3B/2.7B/6.7B、预训练扫 0-340B）定"通用预训练多少 token 后再专门化"——①CPT from scratch 显著差于先预训练；②早切（~40B）≈全量预训练；③t_S 两侧准确率先升后降=预测可信。对用户：正面支持"ClimbLab 基座+STEM CPT"结构；**⚠️ 论文的专门化阶段喂纯域数据，未测 CPT 内通用/专门混合比例——与 r=0.7 只是结构类比，非其结论**；§25 CPT vs from-scratch 的 Apple 侧证据。⚠️ 无 venue、多专门模型设定 |
| 137 | **How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining** | 清华 PACMAN（**#84 Puro-2B/#125 同组**：Kairong Luo/Kaifeng Lyu/Wenguang Chen 等 8 人） | arXiv 2511.18903（2025.11，v3 2026.05，引用 13/11 月，无 venue） | **[高价值]** **顺序×调度耦合轴首个受控机制证据**：升序质量课程与衰减 LR 方向冲突（好数据压轴时 LR 已太低、模型"接不住"）——恒定 LR 下课程显著超随机打乱、标准衰减下优势消失。修复：①温和衰减；②模型平均替代衰减（最后几个 checkpoint 加权平均）；组合后 1.5B/30B tokens、多种质量度量下 +1.64%（git 旧档 +2.7%，v3 缩水，引当前版）。**三方独立收敛**："用权重平均替代 LR 衰减"=#84 Puro CMA（同组产品化）+#130 MergeMix WSM（独立组）——2026 年课程/merge 时代重复出现的配方组件。**历史**：即 2026-07-22 移除的旧 #28（"训练调度非数据配比"），按数据配方五轴口径重录。对用户：三档顺序实验必须锁死 LR 调度变量。⚠️ 无 venue、课程轴→配比轴是类比非同构 |
| 138 | **Bridging Compute- and Data-Optimal Pretraining** | Harvard（Qin, Alvarez-Melis）+MIT CSAIL（Hamidieh） | arXiv 2607.25271（2026.07，无 venue） | **CD scaling laws**：统一 compute-optimal（数据自由增长）与 data-optimal（语料固定）两端——token 有效性函数 η 给"派生 token（多 epoch 重复/改写）值多少新鲜 token"定价（完美替代→零价值）；14M-600M/Dolma-3 拟合重复与改写两策略。η 非常数：联合依赖模型尺寸/TPP/派生数据量、随语料扩张**饱和**；compute-bound/data-bound/model-bound 三区，**经典 compute-optimal 在大部分实际相关区域次优**。对用户：重复轴（#123/#125/#124）的统一框架版；**oversampling 定价器（2026-10-14 场景修正：池 100B≫预算 1.8B，数据充裕非 data-bound）**——全局默认全用新鲜，重复只在"簇权重×预算>簇大小"的局部发生，值不值按 η×质量溢价 vs 次优新鲜簇判（负面结论：池远大于预算时优化精力应花在选择轴而非重复轴）；#101 L3 改写与重复同框架定价，蒸馏亦为派生策略（连 #94）。⚠️ η 拟合区间 14M-600M 外推未验证、无 venue |

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
| 69 | Bayesian Domain Reweighting | 西安交大 (Zongben Xu) | arXiv (2026.07) | Dirichlet 分布 + Gamma 先验直接推断域权重，批评 scaling law/rank invariance 结构假设，声称比 search-based 方法用更少数据 |
| 78 | DataComp-LM (DCLM) | 多机构 (Li, Fang et al.) | arXiv (2024.06) | 240T 语料受控实验框架；model-based filtering；412M-7B scale；DCLM-Baseline 2.6T tokens 达 64% MMLU |
| 79 | Data Mixing Survey #3 | CUHK-SZ (Chen, Miao, Xiong) | arXiv (2026.03) | 41 页综述：bilevel on simplex 形式化；Static（rule-based/learning-based）vs Dynamic（adaptive/externally-guided）分类法；覆盖 DoReMi/RegMix/BiMix/Chameleon/Domain2Vec 等 |
| 81 | Tulu 3 | Allen AI + UW (Lambert et al.) | arXiv (2024.11) | **Post-training 配比方法论**：skill-specific 上界模型 → 优化配比逼近上界。Table 15 混合比例。8B 和 70B 最优配比不同（Scale dependence post-training 版） |
| 85 | D³-MOPD | 苏州大学 NLP 团队特征 (10 人) | arXiv (2026.08, 2608.24987) | **[有意思]** 多教师蒸馏的动态域调度：域收敛速度差异大→固定配比浪费算力；**复用 per-domain reverse-KL 信号做零开销调度**。"按域收敛信号动态重调度"与 R1 influence→0 触发器同家族且零开销（N=1-2 友好）。可信度中（无 venue，苏大团队） |
| 89 | GEM: Geometric Entropy Mixing | Yue Min et al. | **ICML 2026 Poster** (arXiv 2026.05, 2605.26121) | 数据策展重构为超球面变分问题+混合平衡正则；批评人工分类学本体错位与欧氏聚类各向异性缺陷。可信度中高（ICML'26 Poster） |
| 90 | Always Learning, Always Mixing | NYU (Cho, Linzen) + MIT (Sharma) | arXiv (2026.05, 2605.15220) | 跨 pretraining/continual/adaptation 全生命周期统一混料框架；批评现有方法单阶段割裂、依赖单阶段 proxy。可信度高（NYU 教授级团队） |
| 91 | Mix, Don't Tune | ENS Paris/INRIA (Ablin) + DTU (Frellsen) 等 | arXiv (2026.05, 2605.13225) | 数据受限（重复训练）场景：混入高资源辅助语言数据 > 超参调节（高 weight decay）。与用户 TPP≈1.2 数据受限设定相关。可信度中高（欧洲学术团队） |
| 94 | KD During Mid-Training Favors Reasoning over Factual Recall | Meta (Zettlemoyer, Yih, Zeng) + Princeton (Stella Li) + CMU (Pang Wei Koh) 等 12 人 | arXiv (2026.09, 2609.01532) | **[新发现]**（判级修正自 [高价值]，方法本体是蒸馏目标设计非数据配比）forward KL 蒸馏在 mid-training 行为反转：pre-training 双赢，mid-training reasoning 涨但**拖慢 factual recall**；归因=教师置信度跨域不对称×学生演化知识状态。Switch Distillation（教师熵路由）修复且穿透 post-training。间接价值：①"数据价值依赖训练阶段"（数据价值轴新维度，连接 #01/#84）；②条件性适用——STEM 池将来掺教师生成数据时立即直接相关。可信度高（Meta+Princeton+CMU，代码开源） |
| 95 | REER-PT | StepFun (Jiaheng Liu, Ge Zhang, Wenhao Huang 等 10 人) | arXiv (2026.08, 2608.30627) | **[有意思]** PPL 作优化信号做**数据增强（而非过滤）**：识别"难预测但可从上文推断"的 continuation，离线插入简短推理注释重建 context→continuation 连接（长度/泄漏约束），保持 NTP 目标。680M 对照：知识/推理 benchmark 最高 +2.07 pp，PPL 降 0.42-7.29。与用户 PPL 过滤线互补：超难数据不丢弃而是修复（补中间步），"difficult but inferable"=用户中难度带定义。可信度中高（StepFun 数据团队，无 venue，680M 规模） |
| 96 | PLES: Power-Law Entropy Search | Meta Ax/BoTorch 团队 (Balandat, Bakshy, Eriksson) + Cornell (Ament) | arXiv (2026.09, 2609.01431) | **[新发现]** 成本感知 multi-fidelity BO 估计**最优超参 scaling law**：采集函数改为"最大程度降低 scaling law 估计整体不确定性/单位算力"（非优化单目标），自然偏好信息量大的小规模实验；真实 LLM 训练验证，<1/10 网格算力。与 #06 同线但目标不同（估计 law 本身）。**对 proxy 搜索线**：选最能减少 1.5B 外推不确定性的 400M 实验点，而非盲扫。可信度高（Ax/BoTorch 原班团队） |
| 98 | OpenEuroLLM Scaling Laws | OpenEuroLLM 联盟 (Jülich Jitsev / Turku Pyysalo / Hamburg Franke) + Aaron Klein (BOHB 作者) | arXiv (2026.08, 2608.28308) | **[新发现]** LR/batch size 联合+边际 scaling law（随容量与数据规模演化）；WSD 调度 annealing 增益广谱测量（stable→decay 阶段最优超参是否迁移）；**undertraining/overtraining regime 用显式交互 scaling form 建模**——用户 TPP≈1200 同区（与 #37 InfoLaw 互补）；**全部预训练 run 开源**，可当 proxy 拟合的外部验证集。可信度高（欧洲联盟+AutoML 权威 Klein） |
| 103 | Building Multilingual Bridges | Cohere Labs (Mofakhami, Fadaee, Kreutzer, Berard 等 8 人) | arXiv (2026.09, 2609.10445) | 推理是语言无关行为：英语推理主干+多语言非推理数据即可跨语言迁移——Tiny Aya L2-Thinker（3.35B）SFT 阶段数据构成实现 60 语言 in-language reasoning 93%+。与 #101 Math-L3 跨域迁移同构（"数据混合作为泛化支柱"的跨语言版证据）。⚠️ SFT 阶段+多语言场景，与用户 CPT 配比不同层。可信度高（Cohere Labs Aya 系列，权重+数据开源） |
| 104 | Climate-ModernBERT | UZH+McGill/Mila+Princeton 学术组 (Yu, Raj, Stammbach, Leippold 等) | arXiv (2026.09, 2609.07798) | 域适应 CPT 的 merge-vs-mix 受控对照：各语料独立专门化后**参数空间合并**优于联合混合训练，更好保留异构语料互补信息；学术语料是最强适应信号（9 基准 76.3 F1，+2.8）。与 #75 OLMo2 soups/#84 Puro CMA 权重平均家族同谱系 CPT 实例。⚠️ encoder+判别式评测+小增益，向生成式 1.5B CPT 迁移性弱。可信度高（模型+checkpoint 全开源） |
| 106 | **Curriculum as Transport** | Microsoft Research (Chin, Alvarez-Melis) | **COLM'26** (arXiv 2609.09099) | **[有意思]** 课程学习解耦框架（Wasserstein 轨迹分离 ordering/exposure/端点平滑/pacing）：①无普适最优课程顺序；②课程收益≠累积曝光效应（优于 exposure-matched 静态采样）；③pacing 与端点平滑性决定课程在难度谱哪段有效。**用户三档顺序实验的直接设计参考**（支持"N=1 赢了白赚"预期管理；与 #101 tiered vs mix +1.49pp 互证）。⚠️ 合成套件非 LM 预训练，向 1.5B CPT 迁移未验证。可信度高（COLM'26+MSR） |
| 107 | Amortizing Scaling Law Construction Costs | TU Eindhoven+ELLIS Tübingen/MPI-IS (Jha, Vanschoren, Klein 等 9 人) | arXiv workshop (2026.09, 2609.05016) | scaling law 拟合只需 best-loss frontier——BO 采集顺序（渐进扩大 compute 预算+surrogate 幻想评估补全网格）省 10-100× vs 稠密网格。与 #96 PLES 同族（不确定性缩减采集），新增=compute 渐进采集。对用户 400M proxy 协议：proxy 点选取顺序与外推有效性的参考。⚠️ 5 页 workshop 无评审。可信度中高（Vanschoren/Klein AutoML 团队） |
| 113 | **Conditional Transfer from Controlled Pretraining Mixtures to Code** | Ohad Rubin（单作者，无可查机构） | arXiv (2026.10, 2610.11548) | **[有意思]** 三信号分解（diagnosticity/teachability/transfer）+受控混比边实验（70% 固定 CodeParrot+30% 三源单纯形，~180M MoE/15.7B tokens，统一 SFT 桥后 HumanEval）：可教性≠迁移（最可教的策展套件纯用反而低于零合成基线）；最优混比为 75% 目标对齐+25% 策展内部点（条件价值非单调，与 r=0.7 哲学及 #14 同构）；**Ado 在线调度（loss 可减性）5k 步内压掉对齐源、落后匹配固定混 2.4-11.0 pp，~88k 步才回摆**——同尺度 loss 信号指错方向的受控反例（#51/#97/AutoData 线）。统计纪律超同侪（置换检验/BH-FDR/5 万配对 bootstrap）。**可信度中低**⚠️：单作者无机构无代码、关键对比 95% CI 跨零——取方向性结论，数字勿引用 |
| 127 | **SPOKES: Optimizing for Diverse Pretraining Data Selection** | UW NLP（Lee/Choi/Zettlemoyer/Koh）+A*STAR | arXiv (2026.06, 2606.15216) | 集合级多样性直接优化（G-Vendi 分数+指数梯度）：质量×多样性联合选择 FineWeb/DCLM 平均 +1.4/+1.5，超语义去重与质量过滤；单独多样性仅 +0.4/+0.5——多样性是质量的放大器不是替代品。对用户=过度精选丢多样性的校准信号（#118 同族）。尺寸已核：LLaMA-1B/100B tokens from scratch（单尺寸）。⚠️ 无 venue |
| 128 | **Educational or Just Wikipedia-Style?** | Warsaw Univ. of Technology | arXiv (2026.05, 2605.23721) | 学习式质量分类器的风格混淆陷阱：维基风格化改写即可翻转过滤决策，FineWeb-Edu CQF ~7% 文档被翻转放行——分类器分的是"维基风格"非"教育性"。L2 学习式分级（#101）稳健性警告；PPL 分位过滤受此陷阱较小 |
| 129 | **Domain Mixture Design via Log-Likelihood Differences** | 京都大学（Shimodaira 组） | arXiv (2026.03, 2603.16622) | LL 空间几何定域权重：模型=LL 空间中的点，训练更新方向对准目标模型方向→域权重（蒸馏的配比替代，pretraining/CPT 设定）。NanoGPT 一致降 KL。目标错位（用户无目标模型、目标为 STEM accuracy）；与 #29 MixAtlas 同族。⚠️ 小规模 |
| 133 | **Shift is Good: Mismatched Data Mixing Improves Test Performance** | UChicago+清华+TTIC（Medvedev/**Kaifeng Lyu**/Zhiyuan Li/**Srebro**） | arXiv (2025.10, 2510.25108) | **定理级：训练配比≠测试配比"几乎总是更优"**——只要各子群误差函数非常数，最优训练混合比例 generically 不等于测试分布比例（power law/记忆/组合技能/迁移四情形刻画；Corollary 7.4）。r=0.7 的理论执照：训练混合不必匹配评测构成，留 30% 其他数据可以是 generic 最优而非妥协（与 #113 内部点/#14 30% 合成/#01 预算依赖同谱系）。⚠️ GPT-2 ~50M 合成验证、无 venue |
| 134 | **Validity Threats for Foundation Model Research** | Tübingen AI Center+Vienna（König/Pawelczyk/**von Luxburg**/Bordt） | arXiv (2026.06, 2606.05029) | **[有意思]** 方法论元批判（借社会科学四类效度）：proxy 实验用外部+构念效度换统计+内部效度——**跨尺度/recipe 可预测性无法只靠 proxy 实验自证，必须更大尺度验证**；观测研究面临混淆+效应异质；单 run 设计受单元间干扰。全库方法学自觉轴（#10/#35/#121）：400M proxy 线、N=1-2 单 run、benchmark 直评三类威胁的系统性清单。⚠️ 立场论文无新实验、引用 0 |
| 135 | **Relative Generalization Invariance of LLM Pretraining** | Yale+Sea AI Lab+NUS/NTU/A*STAR（Fengzhuo Zhang/**Zhuoran Yang**/Pang/Du 等） | arXiv (2026.09, 2609.33016) | **任意两 token 的 val-loss 差跨优化器与中等架构变化近似不变**（Adam/Lion/Soap/Muon 间成立，NTK/mean-field 单独解释被排除，二次模型定理刻画）；换数据流（FineWeb→CodeParrot）强烈破坏 RGI——**域间相对损失差=数据敏感、优化器/架构不敏感的信号**。Layer 4 反例线索：支持用相对量（域 loss 差/ranking）当 proxy 目标（400M→1.5B 更稳）、AdamW vs Muon 差异影响小的正面证据；⚠️ 跨容量是否保持未明（与 #73/#01 张力）、主证据 124M-0.7B、2.5 周新文 |
| 139 | **The Kinetics of Training** | Lei Dong（**独立研究者单作者无机构**） | arXiv 2607.27281（2026.07，40 页，无 venue） | **驱动成核速率定律**：能力形成=电路零件一次对齐的无部分学分事件（速率=sites×attempts×drive×exp(−βK)−destruction，预注册+冻结常数）；**浓度门槛点燃**（混合过门槛 10/10 点燃、门槛下 0/12，#01 相变的动力学重述）+**监控盲区**（过临界步能力永不点燃而 val loss 全程平滑——门3 只能测的机制论证，已录核心问题文档 §4.5）+**precursor 预测点燃时间 5% 中位误差**（若复现=相变前信号预测相变后的第一个正面线索，与 §4.1 信息论论证张力最大处）。⚠️ **当假说源不当证据源**（独立研究者单作者、主证据 Pythia 消融），数字不引用 |
| 140 | **Problems with Chinchilla Approach 2** | Generation Lab+Columbia+UT Austin+Amazon（Czech/Xu/Wang/Held/Elmatad） | arXiv 2603.22339（2026.03，无 venue） | isoFLOP 抛物线拟合的**系统性偏差**（非噪声）：网格宽度/采样不对中/loss 面不对称三源——无噪声合成数据也偏；套 Llama 3 公开数据=参数分配少投 6.5%（$3.8e25 FLOP 的 6.5%、$1.4M 浪费）；Approach 3+VarPro 修复。对用户：一切 scaling law 外推（400M→1.5B、#123/#138 拟合）的引用层纪律，与 #134 同轴、#107 互补（怎么采点+怎么拟合）。⚠️ 纯方法学无自有训练实验、直接操作面小 |
| 141 | **Unified Data Selection for LLM Reasoning（HES）** | USTC（Feng 组）+阿里+NUS（录入时机构纠错：非 NTU） | arXiv 2605.22389（2026.05，无 venue） | **免训练推理数据质量信号**：样本内 top 0.5% 最高熵 token 的熵之和（HES）——SFT top 20% ≈ 全量（31.34 vs 31.83，STEM 反超 GPQA 51.96 vs 47.66）、最低 20% 崩到 11.75；**0.6B 打分≈8B 自选**（免训练信号跨尺度稳定，#135 同向）；RL 段 Pos-High/Neg-Rand 1.5B 模型超全批且半量数据。对用户：质量-效用张力线正面证据——**熵轴（难度/信息量）比风格轴（质量分类器 #128）更接近训练效用**；池内质量分层候选信号（nanochat 栈可直接算）。⚠️ SFT/RFT/RL 后训练范式非 CPT、≤8B |

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
| 59 | **Qwen3** | Alibaba Qwen | Tech Report (2025.5) | 235B/22B MoE, 36T tokens。三阶段预训练(S1通用30T→S2推理5T→S3长上下文)。**实例级配比优化**超越域级方法(DoReMi/RegMix)，30T+ tokens 细粒度标注 |
| 70 | Data Mixing as Mixture Experiment | Yicheng Mao, Hongru Du | arXiv (2026.08) | **[有意思]** 经典 Scheffé 混合实验设计+I-最优设计。域间成对交互真实（web 文本是交互放大器）；稀疏 Scheffé 跨尺度保持 ranking（nuance Layer 4）；I-最优设计省 25% proxy runs。**可信度存疑**：2 作者无 venue 无机构，经验声明仅仿真验证，方法本身基于 60 年经典统计。**2026-10-12 梯队裁定 T3→T4（无 venue 仅仿真）** |
| 86 | Instella-MoE | AMD (Instella 系列团队, 13 人) | Tech Report (2026.09, 2609.00791) | 16B/2.8B active MoE，pre-train→**mid-train**→long-context→SFT(feedback-driven data curation)→DPO 全管线开放。AMD 新增 mid-training 配比数据点。可信度高（厂商 TR） |
| 87 | SPT | Yufei Sun 等 3 人（无可查机构） | arXiv (2026.08, 2608.26563) | Skill Pre-Training：SkillCorpus（公开 skill 包）做因果 LM 中训练目标。mid-training 数据形态参考（agentic）。**可信度低**⚠️ 3 作者无机构无 venue，结论引用前需独立验证 |
| 92 | MidTool | UC Berkeley + UW (8 人) | arXiv (2026.08, 2608.20314) | 中训练工具使用语料合成管线；引述"定向中训练可强化数学/科学推理"。mid-training 数据形态参考（agentic 方向）。可信度高（Berkeley 团队+开源数据模型） |
| 93 | OctoLong | Würzburg UKP Lab (Gurevych) | arXiv (2026.08, 2608.05141) | 跨仓库代码上下文中训练增强长上下文；AST 管线构造长程依赖语料。mid-training 数据形态参考（长上下文方向）。可信度高（知名 NLP 实验室） |
| 97 | Qwen3.8-Next Architecture | Alibaba Qwen (36 人) | Tech Report (2026.08, 2608.30320) | 125B/A6B 架构 TR（GDN 混合+QSA+n-gram embedding+Muon）。配比相关仅两点：① **loss 与下游 accuracy 不总同向**（n-gram 词表扩大 loss 单调降但 accuracy 饱和）——新增 loss≠accuracy 数据点（支撑 #51 Nexus 线）；② continued-pretraining 阶段全注意力层替换为 QSA，Muon 使最优 LR/batch 上移。可信度高（Qwen 官方 TR） |
| 99 | SPARK: Skeleton-Guided Reasoning Synthesis | Yu Li, Qizhi Pei, Lijun Wu (MSRA 数学推理方向) 等 7 人 | arXiv (2026.08, 2608.30214) | 科学文献→推理数据合成：claim-evidence-derivation 推理骨架 → 四类任务（机制推理/假设证伪/定量推导/边界校准）+一致性验证，Spark-234K。**STEM 数据形态参考**（与 #92/#93 同族）。⚠️ **#88 容量警示**：明确"substantially higher difficulty"——1.5B 中训练勿直接 bulk 灌入（够不着区），更适合后续 SFT 资源。可信度中高（MSRA 数学推理团队，无 venue） |
| 100 | Where Induction Runs Out | Sabilashan Ganeshan（单作者，无可查机构） | arXiv (2026.08, 2608.29411) | MDL 两部分编码做难度信号：OEIS benchmark 实测记忆而非归纳（"apparent competence tracks recognition"——与 #88 小模型死记同构）；**MDL = 廉价免污染难度信号**，与用户 PPL 过滤互补的难度度量视角（显式分离结构 vs 死记）。**可信度中低**⚠️：单作者无可查机构，但预注册+代码数据开源（zenodo） |
| 102 | MiniCPM5-2B | OpenBMB/面壁智能 | Release (2026.09.07) | 2B dense（LlamaForCausalLM，131k ctx），2B 级开源 SOTA（avg 53.9 超 4B 级）。**对用户独一无二：Base→Midtrain→SFT→Final 四 checkpoint 全链开源**——可精确 diff 工业级 2B 模型 mid-training 前后 STEM accuracy 增益，为用户 1.5B mid-training 期望收益定标。训练配方：base（stable+decay）→mid-training（强化目标能力+适配分布）→post（SFT 400B deep-thinking→RL→OPD 16 专家 reverse-KL 合并）。语料全开源（UltraData 家族，见 #101）。**深读**（`MiniCPM5_2B_深度分析.md`）：diff 定标实验设计定稿——只取 Δpp 不取绝对分（tokenizer/架构不同）；若 diff 小=能力播种假说→1.8B 中训练期望管理与评测协议设计；RL+OPD 平均 +10.96 须先扣减；⚠️ mid-training 配方黑箱（token 数/配比未公开，diff 只测总增益）、131k 长上下文成分混杂、仅 2B 有 Midtrain checkpoint（1B 无）。可信度高（官方，全开源） |
| 105 | **Data Scarcity and Model Sparsity: MoEs Overfit More to Repeated Data** | Stanford+UW (Jha, Li, Leskovec, Liang, Zettlemoyer) | arXiv (2026.09, 2609.11917) | 重复数据对 MoE 伤害远大于 dense：80M dense 8× 重复近无损 vs MoE 4× 开始退化、随稀疏度（总参数而非激活参数）加剧；路由早期固定，专家专门化与重复过拟合相关。重复轴（#13/#37/#73 重复上限）的 MoE 边界数据点——**用户 dense 1.5B 无直接相关，架构转 MoE 时生效**。可信度高（Stanford/UW scaling-laws 团队） |
| 108 | Edu-QuRating | UCL 背景学术组 (Garrod, Ince 等 9 人) | arXiv (2026.09, 2609.09425) | 教育价值评分从单标量扩展为多维 rubric（QuRating 后续）：LLM judge 配对标注→蒸馏可复用打分器→按维度组合策展，9 预训练基准聚合超 FineWeb-Edu 基线（增益集中于特定任务）；打分器兼作 GRPO reward。质量评分轴备查（与 #101 L2 学习式分级同方向）。可信度中（学术组，无 venue） |
| 109 | Data Scout | IBM Research (Garg, Zahid, Ludwig 等 6 人) | arXiv (2026.09, 2609.05766) | 反转"过滤 CommonCrawl"范式：LLM 将根主题扩展为 taxonomy+数千搜索查询→定向爬取（非流行度驱动）→子域级分类器探针准入（相关性在子域粒度有锐利边界）。语料生产轴备查；**若将来 STEM 池数据量不足（爬取式扩充替代合成），此为现成管线**。可信度高（IBM Research） |
| 112 | **Stress-testing Alignment Midtraining** | Arcadia Impact（伦敦 AI safety 非营利；与 UK AISI alignment 团队紧密合作，UK AISI Alignment Project 资助；Daniel Tan 等 6 人） | arXiv (2026.09.17, 2609.20412) | **[有意思]** 第一个对 AMT（base→post 之间用对齐文档继续预训练）的系统 stress-test。虚构零污染设定（Dispatch/Python 4 双设定互证），三段管线（AMT 全参 1M-1B tokens→最小 IFT→LoRA EFT），gemma-3-12b/27b、GLM-4.5-Air 106B。五条脆性发现（方向性可信、单种子数字勿引用）：**2% 冲突 SFT（~80k tokens）推翻 190M token 中训练动机**（90%→13%）；EFT 构成>中训练剂量；held-out 泛化弱且 EFT 主动侵蚀未引发知识；去示范换说理 uplift 缩水 0.35-0.73×；SFT→RL 后动机利用 34%→8%（thinking 逐字引用章程却选反面）；gemma-3-4b 全剂量无感。对用户=**中训练阶段约束**：收益须 SFT 后测（#84 互证）、语料须含示范（#101 L3 互证）、未引发知识拿不到、RL"在而不动"风险、剂量曲线平快。⚠️ 价值观轴≠知识/STEM 轴不迁移数字、公开复刻 AMT 或弱于前沿、三周零社区消化 |
| 114 | **Generalization Dynamics of LM Pre-training** | UC Berkeley (Jiaxin Wen, Dawn Song, Lijie Chen) + Stanford/GDM (Zhengxuan Wu) | arXiv (2026.09, 2609.33150) | **[高价值/新发现]** **mode-hopping**：预训练全程 LMs 在鹦鹉式（模式匹配）与智能式（泛化）计算间频发突发跳变（OLMo3-32B answer+1 评测相邻 checkpoint 81%→0%→81.7%，而 loss 与 10 个常规 benchmark 全程平稳）；排除评测噪声/指标选择假象/标准优化动力学三个零假设（单步梯度稳定、K=5 checkpoint 平均只缓解不修复）；容量分配解读（泛化回路与早学浅回路竞争，每个数据窗口决定谁赢）。两应用：**继续预训练/中训练改善 in-distribution 不改善泛化/对齐**（4.5T vs 4.9T checkpoint 经相同 post-training：GPQA 36.3% vs 29.8%、prefilling 鲁棒性 53% vs 21%）+数据子集可定向控制泛化动力学。已录 `核心未解决问题_相变与代理迁移.md` §2.10。可信度高（顶级团队+公开 checkpoint 全程扫描+4 seeds）；⚠️ 32B 预训练尺度、泛化敏感指标（常规 benchmark 平稳→对用户 STEM accuracy 直接威胁有限）、评估套件未放出 |
| 116 | **Fisher-Guided Submodular Data Selection for CPT** | UIC (Zhiling Lan, Yan Yan) + Cisco Research | arXiv (2026.10, 2610.02593) | **CPT 数据选择=遗忘控制 + 参数空间机制**：参数不可知选择（PPL/分类器/随机）造成 post-CPT Fisher 对角在**高 Fisher 坐标（已承诺方向）系统性下漂**——灾难性遗忘的 Fisher 几何解释（"loss 信号在参数空间指错方向"机制级证据，#51/#97/#113 线）；anchor（√Λ⊙g）/frontier（g⊘√Λ）双几何 log-det 子模目标+Sieve-Streaming 单遍流式；TinyLlama-1.1B/Llama-3.1-8B 医疗 CPT：**1B 选择 tokens 超 10B replay（≥10× token 效率）**且约束遗忘。**2026-10-12 用户裁定降 T4**：医疗单域场景窄、方法收益不迁移；机制层警告保留引用。⚠️ 样本级选择非源配比、无代码 |
| 117 | **OptiSelect: How does the Optimizer Shape Data Curriculum?** | EPFL (Simin Fan, Martin Jaggi 等) | arXiv (2026.10, 2610.03432) | **[有意思]** **首个优化器×在线数据选择系统研究**：理论四定理（选择增益=优化器诱导效用分数的可判别性；Lion/Muon 符号型几何可判别性坍缩增益封顶；AdamW/Sophia 对角自适应几何上界严格更好；最优过采样率推导）。124M/720M、4.2B/14B tokens（Chinchilla 最优）、FineWeb-HQ、下游任务 proxy：实测与理论一致（AdamW/AdEMAMix 增益显著、Lion 无益、D-Muon 尺度依赖）；**诚实反思：增益是样本效率非算力效率**（λ=2 读 2× tokens，等 read-tokens 对比全面落后标准训练）；**打分几何解耦**：D-Muon 做优化器时 AdamW 二阶矩几何反而是最佳打分几何。**对用户**：在线选择轴 Muon 时代边界数据点+打分/优化几何解耦原则。⚠️ 小规模 from-scratch，1.5B CPT 定性外推。**2026-10-12 梯队裁定 T3→T4（可信度准则：124M/720M 差工业 2+ 数量级）** |
| 118 | **What Pre/Midtraining Make Learnable from Rewards?** | Chiwun Yang (CityU HK) + Xiaoyu Li (UNSW) | arXiv (2026.09, 2609.38446, 160 页) | **pre/midtraining 供给 RL 什么的理论**：构造"训练奖励上一致、held-out 答案不同"的机制对，证明任务无关源观测消除歧义；有限采样 Adam 路径构造（源预测→奖励适应同参数两段）。Qwen2.5 实验：八世界 Sequential 82.61% vs 私有随机源 44.15%；独立 Memory 确认 75.32% vs 49.86%；memory replay 在奖励适应期间保持已获取检索；GSM8K/HotpotQA 分解入场 accuracy/奖励增益/终点。**对用户**：#112 发现⑤（RL 后"知识在行为不用"）的理论对应——**RL 只能在已存在计算中选择，中训练语料须供应可被奖励调用的计算形态**。⚠️ 纯理论+构造任务族 |
| 119 | **AutoDataBench** | Ruifeng Yuan 等 | arXiv (2026.09, 2609.40097, 有代码) | **数据智能受控测试床**：隔离评估 agent"理解/操纵/改进训练数据"能力（诊断/组织/构造×工具/检索/知识注入），固定非数据因素——对照 auto-research 基准多因素无法归因的缺陷；另测训练前预测 vs 训练后实测（data-effect 推理证据）；**复用轨迹做 mid-training 提升下游 coding**。对用户：#111 AutoData 的评估基础设施对应物（agentic 数据优化生态若成气候的标准化参照）。⚠️ agent 基准非配比方法；与 Weco AutoData 同名不同家 |
| 120 | **Multilingual Quality Classifiers** | EPFL (Martin Jaggi 组) | arXiv (2026.10, 2610.11585) | **英文质量分类器的多语言扩展**：encoder 嵌入+小 MLP，标签=英文分类器×机翻文本分数；1B/3B/8B 保持 model-based 过滤基线性能且不伤区域/文化知识基准；打分标准可跨语言迁移（含训练外语言）。对用户：质量评分轴（#108/#78 线）多语言扩展，英文 STEM 场景备查。⚠️ 多语言偏用户轴外 |
| 121 | **Scaling Down the Scaling Laws** | Joe Dwyer（单作者） | arXiv (2026.10, 2610.06387, 综述) | 资源受限 LLM 训练综述：scaling law→compute-optimal 演进，重点参数/token/**数据效率**/受限环境；覆盖数据剪枝、量化、LoRA、边缘优化；指出企业级 scaling 原理向小模型/受限算力泛化的缺口。对用户：数据效率轴（TPP≈1200 过训练区、#98 同区）背景综述，无新方法新数据。⚠️ **单作者无 venue 纯二手综合**——仅作索引锚点，引用结论须回溯一手 |
| 136 | **Multi-task Code LLMs: Data Mix or Model Merge?** | RPI+Cisco+IBM Research | arXiv (2026.01, 2601.21115) | mix vs merge 受控对照（Qwen/DeepSeek Coder 2B/7B，code 生成+摘要 SFT）：**7B 上 merge 更优**（保留 96% 专门化性能，最优 merge HumanEval 92.7% vs 专门微调 90.9%）、**2B 上 data mixing 更优**——#104（encoder 侧 merge>mix）的生成式/SFT 侧补集；用户 1.5B 落"mix 侧"量级；merge 家族（#04/#84/#104/#130）尺度边界数据点。⚠️ SFT 微调非预训练/CPT、窄场景 |
| 142 | **WARP: Weight-Space Analysis for Recovering Training Data Portfolios** | UW-Madison（Huang/Goyal/Cooper/Sala；录入时机构纠错：非 CMU） | arXiv 2607.01686 + **ICML'26 Workshop on Weight-Space Symmetries**（有代码） | **逆向配比**：从公开权重反推微调数据配比——base↔微调模型间 merge 插值造伪 checkpoint 模拟训练轨迹，Mimic Score 梯度对齐按域池化成几何足迹→softmax 读出/MLP 投影回域比例；BERT/GPT-2 MAE 0.046/0.104，超样本级成员推断、**伪轨迹>真轨迹**（LERP 单调平滑 vs 真实路径噪声）。merge 家族分工补全：#130 正向搜索代理/#04 替代训练+本篇逆向取证；与 #17 LLMSurgeon 同逆向轴不同信号源（权重几何 vs 生成行为）。⚠️ BERT/GPT-2-Small 微调+分类数据集（类当域）距 LM 预训练 2+ 数量级，scaling to LLMs 为 future work |

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

### 22. 核心未解决问题：相变前信号无法预测相变后最优配比

综合 127 篇论文分析，数据配比领域存在一个**贯穿全局的理论鸿沟**：

**问题**：在能力涌现（phase transition）之前，能否从小 proxy 信号预测相变之后的最优数据配比？

**各方法的应对与失败**：

| 方法 | 策略 | 失败机制 |
|------|------|---------|
| QuaDMix/DoReMi/RegMix | proxy 搜静态配比 | proxy 最优 ≠ target 最优（已验证失败）；第四解释=重复率失配（#124，**可修复**，见 §24） |
| LogFloor (#68) | proxy 搜动态顺序 | 用无相变任务绕过；loss 顺序 ≠ accuracy 涌现顺序（未验证） |
| DomainPilot | 不用 proxy，target scale loss | loss ≠ accuracy（Nexus 证明同 loss 差 15 分） |
| Bayesian Reweight (#69) | 直接从数据推断 | 批评 scaling law 假设，但本质仍用 loss 信号 |
| Phase Transitions (#01) | 理论预测相变阈值 | 只预测"何时涌现"≠预测"什么配比最优" |
| BETR/Data Mixing Agent | 直接用 accuracy | 成本高或不做域级配比 |

**为什么可能根本无法解决**（信息论视角）：
- 相变后的最优配比可能依赖于**相变后才会出现的表示空间结构**
- 如果该结构在相变前不存在或不可观测，任何基于相变前信号的方法在信息论意义上都不可能成功
- Capacity Competition (#38) 理论解释：proxy（小容量）和 target（大容量）的容量竞争机制完全不同，最优配比必然偏移
- Why Larger Models Learn More (#52) 实证：小模型把神经元分配给高频域，稀有复杂域被牺牲——不是"学得慢"而是"学不了"

**核心判断**：
> QuaDMix 的失败不是 bug 而是必然。proxy 搜索路径（无论静态还是动态）在有 phase transition 的能力上不可迁移。所有论文要么选择不需要预测的场景（LogFloor 用无相变任务），要么用 loss 绕开 accuracy 涌现（DomainPilot），要么用 target 信号在线调整（但这不是"预测"而是"反应"）。**这个鸿沟目前没有任何论文跨越。**

**2026-10 更新（#124 Repetition Mismatch, EMNLP 2026 Main）**：该鸿沟中被证明有一部分并非信息论缺失，而是**机械性的重复率失配**——高质量源在小/大预算下的重复率不同，最优配比随之漂移；把数据子采样到目标重复率即可修复（1/16 tokens 恢复 1.17B 最优配比，误差 0.10 vs 不控制 0.85）。"根本无法解决"的断言应收窄为容量/相变部分；预算轴部分可机械修复（见 §24 与核心问题文档推论 7）。

详见 `docs/核心未解决问题_相变与代理迁移.md`

### 23. 域间交互真实且重要，结构化响应面可能跨尺度稳定

两篇新论文从不同角度回应了此前"域间交互缺失"和"代理不可解"的开放问题：

**Data Mixing as Mixture Experiment (#70, arXiv:2608.23922)** [有意思] 将 proxy 配比搜索重新定位为经典混合实验设计（Scheffé 1958）：

- **域间成对交互是真实且重要的**——弱域通过与 web 文本组合变得有利。Scheffé 二阶模型 $\mathcal{L}(\boldsymbol{r}) \approx \sum \beta_k r_k + \sum \beta_{kk'} r_k r_{k'}$ 中的 $\beta_{kk'}$ 就是用户此前指出的域间关联 $\sigma_{kk'}$，且可通过稀疏化有效估计
- **稀疏 Scheffé 模型跨尺度保持 mixture ranking**——对 Layer 4（代理不可解性）的重要 nuance：无结构预测器（LightGBM/神经网络）不可迁移（QuaDMix 失败），但**结构化（稀疏低阶多项式）响应面可能跨尺度稳定**。不推翻 Layer 4，只细化——仍是 loss 空间，不解决 Layer 5
- **I-最优设计省 25% proxy runs**——实验设计效率直接降低 proxy 搜索成本
- **⚠️ 可信度存疑**：2 作者无 venue 无机构，经验声明仅仿真验证。但方法学根基（Scheffé 混合实验设计）本身是 60 年经典统计，框架价值独立于论文可信度。**保留为 [有意思]：框架有参考价值，具体经验声明需批判看待**

**Scaling Domain Data Repetition (#71, arXiv:2608.14071, 清华+字节 Seed)** 发现域级最优重复次数与 val loss 强负相关（非数据量决定），同 TPP 可跨尺度迁移。

- 团队可信（清华+字节 Seed），但**对用户当前场景帮助有限**：用户中训练 TPP≈1.2 vs 论文验证 TPP=20，迁移条件不直接满足；用户瓶颈是数据量不足（1.26B vs 参考 150B），非重复策略优化

**结论**：域间交互 $\sigma_{kk'}$ 不仅存在，而且可以用经典统计工具（Scheffé 混合实验设计）在 proxy 层面估计。代理不可解性针对的是无结构黑箱——施加正确结构先验（稀疏低阶多项式在单纯形上）可能部分恢复跨尺度信息。但关键限制：这些仍在 loss 空间，Layer 5（loss-accuracy 鸿沟）不受影响。**#70 的框架思路值得参考，但不应作为高可信依据；#71 对用户场景的战术价值需等数据量问题解决后才有意义。**

### 24. 重复率失配：proxy 外推失败的第四解释，且是唯一可机械修复的

**Repetition Mismatch (#124, Imperial College, EMNLP 2026 Main, [新发现])** 定位了小规模配比实验外推失败的头号元凶：

- **机制**：高质量数据集小、训练预算大时，它在小实验和大实验中的**重复率不同**，最优配比随重复率漂移——小实验根本"看不见"这个变量
- **修复**：把数据**子采样到目标重复率**再搜，1/16 tokens 就能把 1.17B 模型的最优配比逼到 0.10 以内（不控制则误差 0.85）；不受控达到同精度需多训练视界，耗 19%-94% 目标预算
- **核心论断**："是重复动力学而非规模本身决定小实验是否可泛化；数据重复应作为配比优化的一等公民变量"

与既有三大失败解释的关系：容量竞争（#38）、相变表示空间（#01+#52）、val_loss 维度缺失（Nexus #51）沿模型尺寸/信号轴；**重复率失配沿预算轴，与三者正交，且唯一可机械修复**（子采样即可，无需新理论）。

重复轴现在有完整的"理论+实证+新语料"三件套参照：
- **#125 定理（ICLR 2026, 清华 PACMAN）**：数据集越大越耐重复（强凸情形饱和点 Θ(log N)），"≤4 epoch 几乎无差"不是普适常数——1.26B 数学 token 在 1.8B 预算下的重复预算上界有定理参照
- **#123 实证几何（14M-2B 八尺寸扫描）**：重复代价由单变量 z=(R-1)·N_total/U 决定，临界 epoch 数几乎不随模型大小变（proxy 外推利好：大小只动预算项）
- **#126 AI token 变号（800 模型受控扫描）**：掺 AI 文本对数据饥饿模型先降 human loss 后饱和再反转为害、对高预算模型几乎立即有害，Chinchilla 式 law 预测不了这种**变号**；实操=先重复 human 文本、再考虑扩 AI 语料

**对用户的直接修正（行动建议 item 7）**：数学池 1.26B token 在 1.8B 预算下 ~1.4 epoch；400M proxy 线若用 1/16 预算且不子采样则只有 ~0.09 epoch——重复率失配 16 倍。proxy 搜索前必须先按目标重复率子采样。

### 25. CPT 与 from-scratch 的最优配比形状不同：CPT 尖锐、from-scratch 均衡

**Nemotron CLIMB (#74, NVIDIA, T1)** 深读结论（`docs/NemotronCLIMB_深度分析.md`）：

- CLIMB 搜索设定与用户场景**同构**：10T-token 基座（62M/350M/1B，WSD 训练）上搜 40B-token CPT 配比，proxy+LightGBM+迭代搜索；1B/40B CPT 后 60.41 vs RegMix 59.37 / DoReMi 59.16
- **§6 关键发现**：CPT 的最优配比是**尖锐的**——少数域（C8/C9/C18/C19）主导；from-scratch 的最优配比需要**均衡**。两者搜索结果不可互换
- **直接支持 r=0.7**：用户的激进 STEM 占比是"CPT 形状"，有 NVIDIA 实验背书
- **警告**：ClimbMix 数据集内部构成是 from-scratch 优化的（偏均衡）——**不能直接抄为中训练配比**，只适合当语料池
- proxy 尺寸收益温和（62M→350M 仅 +0.3）——用户的 400M proxy 在合理区间
- **坑**：预测器有位置性（Marin 批评：重聚类使已搜配比失效；修复=微簇基直方图 h=V·w）；112 次 proxy 搜索 ≈5040 GPUh（≈0.8× 一次目标训练）——N=1-2 用户不可复现，只能借鉴结论而非流程
- **Apple 侧补强（#131, 2026-10-13 新录）**：Optimal Splitting（#03 原班）用 scaling law 定"通用预训练→专门域 CPT"的最优切换点 t_S——各预算下 CPT from scratch 都显著差于先预训练再专门化，且 t_S 两侧准确率先升后降（内在最优存在，预训练过多也变差）。正面支持"ClimbLab 基座+STEM CPT"结构；**⚠️ 边界（2026-10-13 用户裁定收敛）**：论文的专门化阶段喂纯域数据，未测 CPT 内通用/专门混合比例——与 r=0.7"CPT 中保留 30% 非 STEM"只是结构类比，非其结论

### 26. 验证目标的选择比配比方法的选择更重要

**MDE (#122, Google DeepMind, [高价值], arXiv 2502.15950)**：

- **Table 1（510M/50K steps）**：优化目标与下游 All 准确率的 Spearman——用 11 个 end-task 验证域的 loss 做 objective **ρ=0.844**，用 7 个 SlimPajama 域 loss（DoGe/DML 常用目标）只有 **0.282**——"换验证目标"的收益远大于"换配比方法"
- **Mixture of Data Experts**：每域训一个小专家模型，loss 按混合权重加权聚合 ≈ 任意混合配比的 val loss（附 Proposition 理论保证）；作为回归特征后配比排名预测 0.81→0.95，仅需 ~25 个真实混合 run 做监督，独立搜配比省 3× 算力
- **对用户**：CAP v1/OpenHermes 验证集的域覆盖度检查（行动建议相关条目）的理论根据——validation mixture 本身是最大的杠杆，先检查它再谈换方法
- **工具化落地（#132 SuperValid, 蚂蚁, 2026-10-13 新录）**：capability 级 OOD 验证集的构造配方——benchmark→知识成分蒸馏→语料检索扩展→LLM 过滤改写成验证样本；2T tokens 受控实验里 IID val loss 随配比变化漂移而 SuperValid loss 平稳（MSE 1.51e-3 跨架构/尺度/数据分布）。CAP v1 若要升级为"capability 级 OOD 验证集"，这是现成管线；与 MDE 互补：MDE 证明"选对验证目标"收益大，SuperValid 给出"怎么造对目标"的工业配方

### 27. 模型合并可以当配比搜索的低耗代理，且工业级跨尺度迁移成功（2026-10-13 新增）

**MergeMix (#130, 人大高瓴+蚂蚁, [高价值], arXiv 2601.17858)**：

- **方法**：每域只训小 token 专家（K=4，沿 #74 Nemotron 四域划分，5B≈试验预算的 2.5%），在**参数合并权重空间**里对下游 benchmark 搜最优配比——"权重混合≈数据混合"有 Taylor 展开轨迹分解论证
- **工业级验证**：8B/16B MoE（各 1.4B 激活）mid-training（200B 预算，恒定 LR+merge 16 checkpoints 模拟 annealing）：math/code/reasoning 超**已上线生产的人工调参基线**、其余域近无损；对比 Adapted RegMix（#36）也占优
- **两个关键数字**：搜索成本降 **100×**；排序保真 Spearman **ρ>0.9**
- **最重要的证据（跨尺度）**：8B 上搜出的配比**直接用于 16B mid-training 仍超人工基线**——proxy→target 配比迁移在工业尺度成功，是 400M→1.5B 计划的正面外推证据；与 #73 MAI rank non-invariance（反例）对照读：跨尺度迁移"常常成功但非保证"，机制边界在容量差与相变区（#01）
- **与 merge 家族分工**：#04 DeMix=分开训再合并当**最终模型**；MergeMix=合并当**搜索代理**；#104/#136=merge vs mix 的受控对照。用户 N=1-2 下合并推理成本近零，是 proxy 搜索之外的第二条低耗路径（但 5B 专家成本对用户仍偏高，结构上可缩）

### 28. 相对信号比绝对信号耐造：跨优化器稳定的损失差 + 错配可有益的理论（2026-10-13 新增）

**RGI (#135, Yale+Sea AI Lab, arXiv 2609.33016)**：

- **现象**：任意两个 token 的 val-loss 之差，跨优化器（Adam/Lion/Soap/Muon）与中等架构变化**近似不变**——这些选择对 token 损失是"近似均匀平移"；换数据流（FineWeb→CodeParrot）则**强烈破坏**该不变性（0.7B 验证），且有二次模型定理刻画（GD/Adam/Muon 统一）
- **含义**：**域间相对损失差=数据敏感、优化器/架构不敏感的信号**。①用相对量（域 loss 差/ranking）当 proxy 目标，比绝对 loss 更可能跨尺度保持——400M→1.5B 同栈不同尺寸的 proxy 协议应优先输出相对量；②用户 AdamW vs #126 nanochat Muon 的优化器差异对相对结构影响小（换优化器≈均匀平移）
- **边界（待深读）**：RGI 验证的是优化器/中等架构变化，**跨容量**（小 proxy→大 target 的容量差）是否保持未明——与 #73 rank non-invariance、#01 相变的张力是 Layer 4（代理不可解性）的正面战场

**Shift is Good (#133, UChicago+清华+TTIC Srebro/Lyu, arXiv 2510.25108)**：

- **定理级结论**：训练配比≠测试配比不但不亏反而"**几乎总是更优**"——只要各子群误差函数非常数，最优训练混合比例 generically 不等于测试分布比例（power law/记忆/组合技能/迁移四情形+Corollary 7.4）
- **含义**：**r=0.7 的理论执照**——训练混合不必匹配评测构成（STEM 评测近 100% STEM，训练留 30% 其他数据可以是 generic 最优而非妥协）；把"为什么不是 r=1.0"的回答从经验直觉（#113 内部点/#14 30% 合成/#01 预算依赖）升级到定理层。实操提醒：错配的方向与幅度依赖各域误差曲线形状——具体偏离多少仍须 proxy 实验定，理论只保证"≠测试分布"这一点

## 对 QuaDMix 的行动建议

### 立即行动

1. **修复 Proxy LR** — 降低 proxy 的 learning rate，验证与大模型结果的相关性
2. **实现 Capacity-Aware Mixture Law** — 在 LightGBM 回归中加入 capacity-aware 特征
3. **记录域梯度统计量** — 在 proxy 训练中记录每个域的梯度范数和方差（FastMix + PiKE）
4. **加入 pairwise 域特征** — 在 LightGBM 特征中加入 u_k × u_{k'} 捕捉二阶协同（Domain-Aware SL）
5. **记录 proxy loss 轨迹 + 训练 trajectory-conditioned LightGBM** — RegMix-D 证明这是 ROI 最高的改进（+0.5-1%，成本仅修改 logging + 重训回归模型）
6. **加入 Rarity 信号（第 7 维质量特征）** — Solar Open 2 (#42) 业界验证，rarity = 1/文档频次，计算成本极低，与现有 5 个质量信号正交（预期 +0.5-1%）
7. **子采样匹配目标重复率（400M proxy 线实验设计修正）**（#124）——每个 proxy 配比实验前，把各数据池子采样到"每源 epoch 与 1.8B 目标预算一致"：重复率失配是外推失败的机械解释（同尺寸 1.17B 小预算→全预算同样失败；受控后 1/16 tokens 即恢复最优配比，误差 0.10 vs 不控制 0.85）。例：r=0.7 下数学池 1.26B token 目标预算走 ~1.4 epoch，proxy 预算 1/16 不子采样则变 ~0.09——重复动力学完全不同。见核心问题文档 §2.11

### 短期行动

8. **加入网页图中心度信号**（#30）— 计算 Common Crawl host-level betweenness centrality，乘法组合 + 50/50 混合，预期 +1.5%，成本 9 GPU-hours
9. **对比 Token-Count Baseline** — 实现 token-count heuristic 作为 baseline
10. **调整高质量数据重复次数** — 高质量数据重复 15-20x；参考 InfoLaw (#37) 建模重复递减收益。**⚠️ 修正（#125/#123，2026.10）**：固定重复倍数非普适常数——大数据集更耐重复（E(K,N) 饱和值随 N 增长，修正 Muennighoff "≤4 epoch 无差"），重复上界应按池大小与预算显式计算（定价几何 z=(R-1)N/U）
11. **参考 DataMan 质量标准** — 检查 Eq.1 的质量信号是否覆盖 14 条标准
12. **验证集域覆盖度检查** — 确保 CAP v1/OpenHermes 不偏向特定域（Domain-Aware SL + Google MDE）
13. **PCA 检查搜索空间有效维度** — 对 proxy 实验数据做 PCA（Domain-Aware SL 发现 8 域只有 2 个有效方向；Explaining DMSL #38 用更少参数更高精度）
14. **检查域定义与 proxy 梯度一致性**（#33）— 验证 22 个 Dewey Decimal 域是否与模型梯度感知对齐
15. **在目标训练加域 loss 监控 patch**（DomainPilot #39）— ~30 行代码注入训练框架，在切换点查询回归模型调整配比，Qwen3-1.7B 验证 +2-3.8%
16. **验证 Noise Reduction 理论**（Explaining DMSL #38）— 检查 quality-top-k 是否符合"最优权重向难域偏移"的理论预测

### 长期探索

17. **探索 GP Surrogate**（#29）— 在低维子空间（5-10 主域）上尝试 GP + GP-UCB 主动采样
18. **探索 FastMix 在线优化模式** — 550x 搜索加速，但需注意工业数据风险
19. **探索 LoRA 插值 Proxy** — 比训完整模型便宜，比 model merging 灵活
20. **扩展到数据顺序优化** — 顺序是另一个重要维度
21. **合成数据配比** — 如果数据池包含合成数据，使用 30% 配比
22. **探索 model-driven reward（weight norms）**（HDS #40）— 作为第三信号（质量+域间影响+模型状态），RL 在线调度 Pile 省 44% 迭代、MMLU +7.2%
23. **探索 InfoLaw 式重复递减收益建模**（#37）— 在 overtraining 场景下显式建模重复的尺度依赖递减收益
24. **用 DataFlex (#49) 框架做标准化 baseline 对比** — 统一 sample selection/mixture/reweighting 三种范式
25. **在 nanochat 中验证 Nexus 优化器**（#51）— ~20 行代码实现 Algorithm 3（inner_model NSGD + 位移 pseudo-gradient），3:7 配比不变，验证 STEM+通用域是否同时提升。与数据配比正交，可叠加 Route A/B
26. **采用两阶段 diversity→quality 课程**（Nemotron 3 #60）— 前 75% 多样性优先，后 25% 质量优先。NVIDIA 550B 规模验证。与 QuaDMix quality-top-k 思路一致
27. **分阶段上下文+难度递进**（GLM-5 #58）— 中训练分阶段：先扩上下文，再加难度数据。200K 阶段增强 128K 窗口内性能
28. **考虑加入 SFT-style 数据**（Nemotron 3 #60）— sft-code/sft-stem/sft-general 在预训练中有效（Akter 2026 frontloading）
29. **考虑补充代码数据**（GLM-5/Qwen3/Nemotron 3）— 所有头部厂商中训练都含 code ≥15%，当前无 code 是最大差异
30. **长期升级到实例级配比**（Qwen3 #59）— 从域级配比升级到实例级，需要细粒度标签（educational value/fields/domains/safety）
31. **使用 I-最优设计 + 稀疏 Scheffé 模型替代随机采样 + LightGBM**（#70, [有意思]）— proxy 实验设计优化：用经典 DOE 方法选择配比实验点（省 25% runs），用稀疏二阶 Scheffé 模型替代黑箱预测器获得可解释的域主效应+交互分解。特别关注 web×math 交互项 $\beta_{\text{web},\text{math}}$。**注意**：论文可信度存疑（无 venue），但方法本身是 60 年经典统计工具
32. **按域 val loss 决定重复策略**（#71）— 测量各域 val loss，loss 低的域多重复，loss 高的域少重复。用同 TPP 小 proxy 调好后迁移。**注意**：中训练 TPP≈1.2 与 Chinchilla TPP=20 差异大，迁移不直接；用户当前瓶颈是数据量而非重复策略，此建议为战术级

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
- **2026-08-17**: 新增 BETR 深度分析 (#64, Apple, arXiv 2025.07)。**核心发现**：① benchmark 相似度驱动文档级数据选择，3 步流程（embed → rank score → FastText filter），成本仅 ~2-4 GPUh，是所有方法中最低；② 500+ 模型 scaling law 验证，2.1x compute multiplier over DCLM-Baseline，MMLU +7.5 分；③ 过滤率幂律 F_opt(C)=4×10⁻⁵·C^0.25 — 小模型需 top 3% 激进过滤，大模型 top 30% 宽松，与 Phase Transitions 容量竞争理论一致；④ FastText (75.6% acc) 击败 302M LM classifier (82.4% acc) — 分类准确率不是关键；⑤ benchmark targeting > domain targeting（Noncore 60.9 > arXiv 52.6）；⑥ 被 Phase Transitions 论文引用，17 Semantic Scholar 引用。**适配评估**：文档级≠域级配比，不能替代 ClimbMix，但可作为低成本预处理步骤（BETR 过滤 → ClimbMix 配比 两阶段串联）。论文仅验证预训练，未验证 continual pre-training。新增文档 `docs/BETR_深度分析.md`。总计 64 篇论文
- **2026-08-18**: 新增 Task-Agnostic Training Data Influence 深度分析 (#65, NAIST+NII+早稻田+东大+哥本哈根IT+东北大, **COLM 2026 录用**)。**核心发现**：① task-agnostic 数据贡献度量——不依赖下游任务，定义为"梯度更新让模型参数向最终预训练参数靠近的程度"；② 检查点近似（用相邻检查点参数差代替 batch 更新）无需重训，事后从公开 Pythia/PolyPythia 检查点直接估算；③ **中期贡献达峰**（~40k 步），早晚期较低；④ **文学→STEM 交叉反转**：早期文学类高贡献、STEM 低贡献，后期反转——为业界"后期增加 STEM 配比"提供归因层面实证；⑤ 高 PPL（困难）文本在中期贡献占比从 20%→25%，低 PPL 20%→15%；⑥ 末期 opponent（负贡献）样本升至 8~10%，暗示后期重质而非重量；⑦ task-specific (TracIn) 无法稳定复现文学→STEM 交叉——两种视角应互补。**可靠性验证**：检查点近似晚期 Pearson >0.94，跨 6 个模型规模+多种子一致（小模型阶段转换推迟）。**局限性**：观察性非因果、"贡献"≠下游性能、仅 Pythia+Pile 验证。**与配比项目相关性 ★★★★☆**：直接支撑分阶段配比策略，PPL 分桶可作质量信号补充。新增文档 `docs/TaskAgnosticInfluence_深度分析.md`。总计 65 篇论文
- **2026-08-18**: 更新 #65 深度批判分析。**三层循环论证**：① 梯度幅度混淆——高 PPL→大梯度→对齐项线性增大→贡献偏高（论文未做控制实验）；② θ* 自指性——θ* 由被评估数据训出，样本梯度方向和"未来要走的方向"天然相关（对个体样本偏弱，域级成立）；③ 域比例偏向——数据多的域拉偏 θ*，该域贡献系统性偏高。**交叉反转 = loss 动态而非数据价值**：文学→STEM 交叉可用 loss 下降速度差异统一解释（先易后难），而非独立的数据价值信号——"STEM 后期贡献高"只是因为 STEM 当前 loss 还没降。**建设性结论**：该加什么数据看当前 loss（非循环），而非看贡献（循环），与 Velocitune (#61) 学习速度方法不谋而合。相关性评级 ★★★★☆→★★★☆☆（下调）。更新文档 `docs/TaskAgnosticInfluence_深度分析.md`
- **2026-08-18**: 新增 DIDS (#66, HKUST+MetaX+Alibaba+浙师大+苏大, **EMNLP 2025**) 和 MIRA (#67, SJTU+UBC+Langboat+北航+IQuest, arXiv 2026.05) 深度分析。从 Velocitune 11 篇引用中筛选出最适合用户场景的方法。**DIDS 核心发现**：① FIM 指导的域影响度量（KL 散度二阶 Taylor 近似 → Fisher 对角近似）+ loss 轨迹指数衰减拟合 + 边际递减 → EMA 动态配比；② **直接 benchmark 对比 Velocitune 赢 3 分**（多任务 62.3 vs 59.3，单任务 63.7 vs 60.5）；③ 用 10% 数据（100k vs 929k）超过全量训练；④ **绕过 Velocitune 最大风险**——不需要 scaling law；⑤ 额外开销仅 1.9%，500M proxy 够用；⑥ 消融：FIM -2.9, loss 轨迹 -2.8, 梯度聚类 -1.8。**代码仓库深度分析**：两套不一致的 FIM 实现（可执行脚本用 Fisher kernel 内积 vs utils.py 用 KL quadratic 差分平方）；梯度聚类默认关闭（num_clusters=0）；更原则性变体（R² 限定潜力因子）在 utils.py 未使用；研究原型质量。**MIRA 核心发现**：① 中训练样本级筛选（非域级配比），自锚定 rubric 发现（Kimi-K2.6 教师 → 15 锚定维度/组 → Qwen3.5-35B-A3B MoE 学生蒸馏）；② 25B 半量匹配/超过 50B 全量（macro avg 64.20 vs 63.83）；③ 可靠性 mask 12% 源-维度对（MAE ≥ 1）；④ PPL/DSIR 有长度依赖，MIRA 长度鲁棒；⑤ 学生蒸馏成本 <4% 中训练成本（~250 vs ~7,680 H800 GPUh）。**定位**：DIDS 做域间配比，MIRA 做域内筛选——互补关系（MIRA 筛 → DIDS 配）。DIDS 相关性 ★★★★☆（域级+开源+赢 Velocitune+无 scaling law，但规模和代码质量顾虑），MIRA 相关性 ★★★☆☆（中训练匹配但层面不同+无代码无 venue）。新增文档 `docs/DIDS_深度分析.md`、`docs/MIRA_深度分析.md`。总计 67 篇论文
- **2026-08-19**: 新增 STEM 能力涌现分析文档 `docs/STEM能力涌现分析.md`。**核心问题**：STEM（数学/推理）是规模涌现的能力吗？多大模型/多少数据才涌现？应用于本项目 nanochat-npu 1.5B 中训练的可行性判断。**综合三层共识**：① 经典涌现叙事（Wei 2022）GSM8K ~10²² FLOPs；② 指标假象修正（Schaeffer 2023）accuracy 阈值造成的阶跃假象但对下游 accuracy 目标仍成立；③ 相变理论（Phase Transitions #01, NeurIPS'25 Spotlight）`r_thres ∝ M^(-α)` 尖锐相变 + (M, r) 联合曲面 + 低于阈值训再久=0%。**业界锚点**：DeepSeekMath 1.3B + 150B 数学 token ≈ 21% GSM8K；7B + 280B 数学 ≈ 24%；Nexus 规模门控 GSM8K 增益 130M≈0 → 3B+15（STEM 收益 1B 以下几乎不存在）。**本项目判断**：基座 1.5B 预训练于 ClimbLab 1.2T（充分，800 tokens/param）；中训练 1.8B tokens（1.2× 参数量），3:7 配比 → 1.26B 数学 token，**仅为 DeepSeekMath 参考线（120-280B）的 ~1%**。配比方向正确（r_thres ∝ M^(-α) 支持小模型高 r，70% 合理），但数据量是生死瓶颈——**1.5B + 1.26B 数学 token 下 STEM 涌现基本不可能**，预期 GSM8K 个位数。当前中训练定位为"基座风味调整/数据管线验证"，真正 STEM 涌现需模型放大到 3B+ + 数学 token 提到 ~100B 量级。同时将 Phase Transitions (#01) 标记为 **[高价值/重要]** 并更新论文索引描述为其作为本项目 STEM 涌现分析理论根基
- **2026-08-20**: 新增 QuaDMix 失败分析与相变理论启示文档 `docs/QuaDMix失败分析_相变理论启示.md`。**核心内容**：① **失败根因 = proxy gap（14M → 1.5B）**——Phase Transitions 理论解释：14M 与 1.5B 相变阈值不同（14M r_thres>0.30 vs 1.5B≈0.06-0.08）→ 不同 regime；背包问题对不同 M 有不同解 → 14M 最优"少给某些子域" vs 1.5B 最优"多给" → 方向相反 → 比 Random 还差。R² within-model（14M→14M）≠ cross-model transfer（14M→1.5B），gsm8k R²=0.751 但 QuadMix 反而最差 = 直接证据。② **Phase Transitions 论文完整信息**：两个相变（模型大小+配比）、幂律 r_thres ∝ S/M^1.283（α≈0.283）、训练步数随 1/r 超指数增长（r<0.4）、背包理论+边际价值、两个策略（随机子采样 4x + CKM 20x）、推理任务也有相变、proxy 警告原文+5 引用、continual pre-training 验证+early stopping 标准（val_loss +0.05）、计算成本表。③ **域级相变与 STEM trade-off**：每个域有自己的 r_thres（S 不同）；高数学配比损害其他 STEM 任务（背包 trade-off）；数学得分提升需足够 token 走完相变——r=0.7>>r_thres 在线性区但 1.26B 是参考 150B 的 ~1% → "刚跨阈值还没走完"；域状态分类表（远超阈值/刚跨/低于/已收敛）。④ **修正"小Proxy不可用"文档**：14M proxy 有信号（R²=0.30-0.84），不是随机噪声，但信号来自不同 regime 不传递到 1.5B accuracy。⑤ **行动方向**：修复 14M→1.5B 直接搜索（198→22 维, CLIMB 式 30→10→3）；Step 0 验证（30 min 推理判断 val_loss→accuracy 是否成立）；DomainPilot 兜底（无 proxy gap, 无 loss≠value, ~30 行 patch）；两条腿走路建议。新增文档 `docs/QuaDMix失败分析_相变理论启示.md`。⑥ **新增"重点论文快速索引"表**：在论文索引顶部汇总所有 [高价值/重要]/[高价值]/[新发现]/[有意思]/[独立再发现] 标记论文（共 17 篇），附深度分析文档路径，便于快速定位重点论文。确认 arXiv:2608.13515（Task-Agnostic Influence, #65）已在项目中记录
- **2026-08-20**: 新增 `docs/论文总表.md` — 全 67 篇论文统一汇总表（按编号排序，含梯队+标记+机构+Venue+文档路径+关键贡献），附统计摘要和标记/梯队说明。README 顶部新增免责声明（个人主观+特定场景+非官方）。README 重点索引表精简为链接到总表
- **2026-08-24**: 新一轮 arXiv 搜索（data mixture / data selection / scaling law data），新增 2 篇论文 + 2 篇深度分析文档。**LogFloor (#68, arXiv:2608.14936)**：小 proxy 模型搜索数据配比的新视角——不搜静态配比，搜**动态瓶颈解决顺序**。70M→12B replay 省 30.9%。顺序消解实验极干净（正确 60.3%→静态塌缩 10.3%→逆转 6.0%→随机 6/8 命中），独立验证 QuaDMix "静态配比不可迁移"的失败归因。**核心局限**：用 log-loss 不用 accuracy 绕过 phase transition，用 bAbI/MMLU 无相变任务回避了核心问题；loss 下降顺序 ≠ accuracy 涌现顺序——恰好是 QuaDMix 撞的同一面墙。**Bayesian Domain Reweighting (#69, arXiv:2607.27928, 西安交大 Zongben Xu)**：Dirichlet+Gamma 先验直接推断域权重，批评 scaling law/rank invariance 结构假设，声称比 search-based 方法用更少数据（仅索引，待深读）。**新增综合分析文档** `docs/核心未解决问题_相变与代理迁移.md`：综合 69 篇论文，系统性梳理核心理论鸿沟——"相变前信号能否预测相变后最优配比"，论证各方法应对策略与失败机制，从信息论视角论证该问题可能根本无法解决（QuaDMix 失败不是 bug 而是必然）。新增文档 `docs/LogFloor_深度分析.md`。总计 69 篇论文
- **2026-08-24**: 新增问题形式化定义文档 `docs/LLM数据选择配比问题定义.md`。用数学语言严格定义数据配比优化问题：**第一节**符号表+决策变量（$K-1$ 维单纯形）；**第二节**三层等价形式（accuracy 目标 $F$ / loss 代理 $\mathcal{L}$ / bilevel on simplex）+计算约束 $N=1\text{-}2$；**第三节**目标函数五条性质（黑箱性、相变结构 $r_{\text{thres}} \propto S/M^{1.283}$、尺度依赖=背包+容量竞争、状态配置 $\boldsymbol{\sigma}(M)$）；**第四节**代理范式定义+可行性条件+**不可解性定理**（命题1：proxy 无法学会 $\Rightarrow F_p$ 为常数 $\Rightarrow$ 零信息；命题2：$\boldsymbol{\sigma}(M_p) \neq \boldsymbol{\sigma}(M_t)$ $\Rightarrow$ 不同 regime $\Rightarrow$ 方向可反；信息论：$I(F_p; \boldsymbol{r}^*_t)=0$）+实证验证表；**第五节 loss-accuracy 鸿沟展开**（5.1 形式定义 $\boldsymbol{r}^*_\mathcal{L} \neq \boldsymbol{r}^*$、5.2 close vs distant minimizer=Nexus 同 loss 差 15 分、5.3 loss 相变滞后=光滑化代价是丢失相变信息、5.4 QuaDMix 直接实证=Final Loss 最低但 accuracy 最差、5.5 LogFloor 绕行与未验证的桥、5.6 实践启示=loss 是必要非充分信息源）；**第六节**本项目实例化（参数表+域状态分类+可行性边界）；**第七节**五层困难叠加小结。核心判断：数据配比=黑箱+相变+尺度依赖+代理不可解+loss-accuracy 鸿沟，第四层信息论不可逾越
- **2026-08-27**: 新一轮 arXiv 搜索（Semantic Scholar bulk API，9 个查询角度），新增 3 篇论文 (#70-72) + 2 篇深度分析文档。**Data Mixing as Mixture Experiment (#70, arXiv:2608.23922, Yicheng Mao & Hongru Du)** [有意思]：将 proxy 数据配比搜索重新定位为经典**混合实验设计**（Scheffé 1958）——域=成分，配比=比例，proxy run=实验点，val loss=响应面。三个核心发现：① **域间成对交互真实且重要**——弱域通过与 web 文本组合变好，直接回应此前用户指出的域间关联 σ_kk' 缺失问题；② **稀疏 Scheffé 模型跨尺度保持 mixture ranking**——对 Layer 4 代理不可解性的 nuance：无结构不可迁移，结构化可能可迁移（仍是 loss 空间，不解决 Layer 5）；③ **I-最优设计省 25% proxy runs**。**可信度存疑降级为 T3 [有意思]**：2 作者无 venue 无机构，Semantic Scholar 查无可靠轨迹，经验声明仅仿真验证。方法学根基（Scheffé 1958）本身可信，框架有参考价值，具体经验声明需批判看待。**Scaling Domain Data Repetition (#71, arXiv:2608.14071, 清华+字节 Seed, Jingzhao Zhang)**：域级最优重复次数与 val loss 强负相关（非数据量决定），同 TPP proxy 可迁移。团队可信但**对用户场景帮助有限降级为 T1 无标记**：用户中训练 TPP≈1.2 vs 验证 TPP=20，数据量瓶颈（1.26B vs 参考 150B）非重复策略。**Cross-Difficulty Optimization Dynamics (#72, arXiv:2608.17268, 复旦)**：Relative Transfer 度量跨难度知识迁移+TDCS 动态采样。概念上接近域间交互（难度级）+在线适应，但场景为 post-training 课程学习。新增文档 `docs/MixtureExperiment_深度分析.md`、`docs/DomainRepetition_深度分析.md`。总计 72 篇论文

- **2026-08-27**: 用户指定 MAI-Thinking-1 (microsoft.ai/pdf/mai-thinking-1.pdf) 为高价值参考，搜索类似工作。MAI-Thinking-1 §2.5 "Selecting a Data Mixture for Training" 6 页，含 Pareto frontier、**rank non-invariance**（千个 proxy 模型实证否定 RegMix rank invariance 假设——stem-heavy 小尺度赢但 code-heavy 大尺度反超）、hierarchical local/global search、memorization-aware epoch capping（NLL<0.01 分数）、完整配比表（Table 5: Code 54.6%/STEM 15.8%/Math 5.4%，Math 5.28× 重复最高）、中训练 STEM:Code:Other=35:55:10。新增 9 篇论文 (#73-81)。**Frontier 规模 Tech Report**：**#73 MAI-Thinking-1** (T2 [高价值], Microsoft AI)：35B/1T MoE，109 页，rank non-invariance 直接回应本项目 Layer 3 代理迁移问题；**#74 Nemotron-CLIMB** (T1 [高价值], NVIDIA)：**用户基座预训练数据来源**（ClimbLab 1.2T + ClimbMix 400B），聚类迭代搜索优于 DoReMi/RegMix；**#75 OLMo 2** (T2, Allen AI)：Dolmino Mix 1124，全开源，single-source microanneals 被 MAI §2.5.4 直接引用；**#76 Llama 3** (T2, Meta)：15T tokens，annealing phase；**#77 Phi-4** (T2, Microsoft)：合成数据配比为核心；**#80 SmolLM2** (T2, Hugging Face)：1.7B/11T，4 阶段 evolving mixture。**方法论参考**：#78 DataComp-LM (T3, 240T 受控实验框架)、#79 Data Mixing Survey #3 (T3, 41 页综述)、#81 Tulu 3 (T3, post-training 配比方法论)
- **2026-09-02**: 新一轮 arXiv 搜索（2026-08-25→09-02 窗口 + 上轮盲区补漏），新增 10 篇论文 (#84-93)。**窗口内新论文**：① **Puro-2B (#84, 清华 thu-pacman, 2608.27370)** [T2，深读候选第一优先]：Qwen2-1.5B 同款架构从零复现，RTX 5090 消费卡 $4.4-6.9K、1.4T tokens FP8，数据/代码/权重 Apache 2.0 全开源；含 Puro Cost Scaling Law 与 **"预训练数据课程如何塑造 post-training 后性能"的 end-to-end case study**（全管线可控才做得了的实验，与用户 ClimbLab 基座+中训练场景同构）；② **D³-MOPD (#85, 2608.24987)** [有意思]：多教师蒸馏场景发现域收敛速度差异大→固定配比浪费算力，复用 per-domain reverse-KL 信号做**零开销动态域调度**——与 MobileLLM-R1 influence→0 触发器同家族但更轻量（N=1-2 友好）；③ **Instella-MoE (#86, AMD)**：16B/2.8B active MoE 全管线开放（含 mid-training+feedback-driven data curation）；④ SPT (#87)：skill 包做中训练语料。**上轮盲区补录**：⑤ **Capacity-Dependent Data Selection for Reasoning (#88, Virginia Tech Ruoxi Jia 组, 2608.13721)** [有意思]：**1.5B-8B 数学 reasoning 蒸馏受控实验**，Fast-Fit/Slow-Gain 模式——小模型用高似然（贴近学生分布）数据早期更快更稳，低似然数据仅对大模型+长训练有益（小模型无法吸收→浅层重复行为）。**直接命中用户场景：1.5B STEM 数据选择应偏向学生分布匹配数据**；⑥ GEM (#89)：超球面变分配比框架；⑦ Always Learning, Always Mixing (#90, NYU)：全生命周期统一混料；⑧ Mix, Don't Tune (#91)：数据受限场景混料优于调参（与用户 TPP≈1.2 相关）；⑨ MidTool (#92)/OctoLong (#93)：中训练数据形态（agentic/长上下文）。**本次同步校准历史统计漂移**：按行级重算，总表实际 81+10=**91 篇**（原统计 83 虚高 2；梯队真实分布 T1=17/T2=25/T3=36/T4=13；"有深度分析=65" 修正为"引用深度分析路径 45，docs 实际文件 31"）。总计 91 篇论文
- **2026-09-02（可信度核查）**：对 #84-93 全部 10 篇新论文做作者/机构/venue 核查（arXiv API comment 字段 + HTML 全文机构提取）。**关键修正**：① **#88 Capacity-Dependent 升级 [有意思]→[高价值]**——arXiv comment 显示 **COLM 2026 接收**，且为 Virginia Tech Ruoxi Jia 组（数据归因/influence 方向知名学者），1.5B-8B 受控实验直接命中用户场景，三项全强；② **#89 GEM 确认 ICML 2026 Poster 接收**（此前按无 venue 评价，可信度上调至中高）；③ **#84 Puro-2B 作者全名单核实**：Wenguang Chen（清华 PACMAN 实验室教授）+ Kaifeng Lyu（UChicago，scaling law 方向学者），11 作者 62 页完整报告，Apache 2.0 全开源——可信度高；④ **#87 SPT 可信度低**⚠️：3 作者无可查机构、无 venue、无 comment，仅作方向记录；⑤ **#86 Instella-MoE 移 T3→T4**（对齐头部厂商 TR 归类惯例，与 #55-60 同档）；⑥ 其余：#85 D³-MOPD 苏大团队（中）、#90 Always Learning 为 NYU Cho/Linzen 教授团队（高）、#91 Mix Don't Tune 为 ENS/INRIA Ablin 等欧洲团队（中高）、#92 MidTool 为 Berkeley+UW 团队且开源（高）、#93 OctoLong 为 Würzburg UKP Lab Gurevych 组（高）。**可信度分布**：高 6 篇 / 中高 2 篇 / 中 1 篇 / 低 1 篇（SPT）
- **2026-09-02（原文核实与纠错）**：用户对 checkpoint 平均的业界实况提出质疑，下载 Llama 3（92 页）/Gemma 1/2/3/DeepSeek-V3/Qwen2.5/OLMo 2/MiniCPM 七篇报告原文逐篇检索权重平均用法。**核实结果**：① **DeepSeek-V3 预训练用 EMA**（CPU 异步维护参数 EMA，用于早期估计 LR 衰减后性能）；② **OLMo 2 §4.5 中训练 Checkpoint Soups**——最终高质量混合独立 anneal 多次（7B: 3×50B tokens 各不同随机数据顺序）再平均权重（"find a better local minimum"），**恰在 mid-training 阶段**，已补入 #75 总表条目；③ Gemma 3 仅后训练 weight averaging（WARP）；Llama 3/Qwen2.5 预训练无。**纠错三处**：Gemma 1/2 "EMA" 命中实为 math**ema**tics 子串（撤回）；MiniCPM arXiv 版检索不到 model soup（撤回，疑记忆混淆自社区博客）；Llama 3 checkpoint 平均（撤回，只有数据 annealing）。Puro 消融解释了不普适原因：衰减 LR 轨迹上直接平均无效，多数家收尾是 decay-to-zero 故试之无感。同时就用户对 PPL 选数据的追问补充 CapacityDependent 文档：**新颖性 ≠ 可学性**（高 PPL 直觉对 8B+ 正确、对 1.5B 失效——容量条件性）；宏观（域级 70% STEM 本身就是"加强没见过"）vs 微观（池内文档级过滤）；**域混淆警告**（PPL 阈值须源内分位）；**打分模型用 1.5B 基座而非 400M 代理**（student-grounded，400M 系统性高估难度=Puro 容量错位陷阱）；**采样式打分协议**（每源抽 1k-10k 篇定分布，全池逐文档打分≈2× 中训练 run 计算量，不可取）。
- **2026-09-02（深读 Top2）**：完成 **#84 Puro-2B**（`docs/Puro2B_深度分析.md`，升级 [高价值]）与 **#88 Capacity-Dependent**（`docs/CapacityDependent_深度分析.md`）两篇深度分析，通读 62 页 TR 全文 + COLM'26 正式版 32 页全文。**#84 Puro-2B 核心发现**：① 五大效率组件中**课程/CMA 数据配方贡献 2.40×（最大单项）**——376 桶源内升序质量课程（跨组件配比桶内近似保持）+ step-218k 恒定 LR 续训 + 6 checkpoint 等权平均；消融显示**模型平均单独无益**（uniform −0.42/curriculum +0.01），必须配恒定 LR 续训（+0.63）；课程与 LR 调度耦合设计（与 R1 的 LR confound 同主题）；② **Proxy 分位切片协议**：Qwen3-0.6B 共享基座（86B tokens）+ 2000 步续训（8.4B/条件）+ 候选占比 0→80% 爬坡（控制分布冲击）+ q00-q75 分位窗口把源级分数变成源内决策信号；PCA 只做描述不做目标（方法论自觉）；③ **Post-training 存活实验**（最有价值）：课程优势穿透匹配 SFT——聚焦数学 +1.77 pp / 放大数学 +2.02 pp（3 次重复一致）/ 广泛指令 +1.59 pp（13/15 任务）——"配比收益必须用 SFT 后下游 accuracy 衡量"直接强化本项目 Layer 5 应对策略；④ P1→P2 配比：English 73.2%→59.4%、**Math 7.2%→18.3%**、尾段 SFT 1.3%；主力组件 Nemotron-CC-v2 HQ（P2 448B，与用户 ClimbLab 同生态）+ UltraData-Math 118B；700 TPP 与 ClimbLab 800 同区、Phase 2 单遍与用户 TPP≈1.2 同构；⑤ ⚠️ 污染审计缺失（作者自述）+ 0.6B proxy 选 2B 数据的容量错位未检验。**#88 Capacity-Dependent 核心发现**：① **三阶段容量模式**（比摘要的二元反转更细）：低容量（≤3B/Llama3.2-3B）**Fast-Fit/No Catch-up**、中间容量（Qwen2.5-Math-7B）不稳定反转（峰值可达 6/10 但 20 checkpoint 仅 5 时点持续占优）、高容量（8B）**持续反转**（5 epochs avg 低 70.69 vs 高 66.64，epoch 2.25 交叉后持续领先至 epoch 5）；② 小模型学低似然数据=**死记**（1.5B 重复题干卡死、3B 中途重复、7B 完成）；③ 理论：容量约束**双重投影**（教师知识差距须同时落在数据张成空间 ∩ 学生可表示子空间），"容量对齐数据是低容量蒸馏必要条件"；难度应是 **student-grounded**（学生似然）而非 trace 长度等外部代理；新增难度选择有效性的第三轴**容量轴**（前两轴：数据量 Sorscher/学习阶段 Weinshall）；④ 对自家 best-checkpoint envelope 报告口径的诚实二次审计（Peak Accessibility vs Persistent Superiority 两标准）；⑤ **对用户场景的直接推论：1.5B 处 No Catch-up 最深区 → STEM 中训练数据应偏向基座 PPL 可及段（用 ClimbLab 基座打分选分位），勿灌超难合成推理数据；N=1-2 短预算下高似然更安全（低似然回本需 ≥7B+20+ checkpoint 长训练）**；⑥ **两篇拼出的开放问题**：Puro 用 0.6B proxy 选 2B 数据——#88 证明选择价值容量依赖，proxy 容量错位是配比信号失真的新来源（Layer 4 新实例，已写入两份文档交叉引用）。总计 91 篇论文（标记 25 篇，深度分析文档 33 个）
- **2026-09-02（深读 #73 MAI-Thinking-1）**：完成 `docs/MAI_Thinking_1_深度分析.md`（109 页微软官方 TR 全文通读，§2.5 配比 6 页 + §2.2.1 ladder + 附录 C 长上下文）。**核心发现**：① **Rank non-invariance 工业级实证**（§2.5.2/Fig.6）——几千个 760M-4B proxy 做 RegMix 式 forecasting，stem-heavy-mix 小尺度 STEM NLL 领先但 **23B×20T 训练中途被 code-heavy-mix 反超**（曲线交叉），归因两个高质量低多样性 STEM 源（11.8%→0.3% 权重差）；RegMix 理论根基被一手实验否定，且反转在 23B/20T 才暴露；微软补救=所有消融上 scaling ladder（恒定 TPP 多尺度曲线）看 scaling 行为而非固定尺度排名+最终候选 2.8× 计算验证；② **配比目标函数 Eq.3**：0.5×Coding+0.175×STEM+0.175×Math+0.1×GK+0.05×Multi（约 40 个内部 NLL 基准，Math 单列）——多目标显式加权压单目标；③ **最终配比 Table 5**（30T）：Code 54.6%/STEM 15.8%/Math 5.4%（5.28× 重复最多）/Web 14.9%（0.55×）/多语言 0.06%——**稀缺源多重复、富源不重复**，8× 全局上限；④ **中训练 §2.5.4**（用户阶段直接对标）：STEM:Code:Other=35:55:10，**数据全部来自预训练语料不引入新源**（只过滤+重加权+重打包），类内用 OLMo 2 single-source microanneals，STEM reasoning 用 Bloom Analyze 层级过滤（Essential AI taxonomy），**memorization-aware epoch capping**（两 checkpoint 间验证损失改善中 NLL<0.01 token 占比=记忆化分数→按源动态分配重复预算）；⑤ **上下文扩展调配比无效**（附录 C.1 负结果）：试过加权重长文档/调域配比全部无益，最优=沿用中训练同配比仅重打包；⑥ **TPP 匹配**：微软主 run 500-1000 TPP，用户 1.5B×1.8B≈1200 TPP 同属"过度训练紧凑模型"区——配比行为应参考高 TPP 协议。**rank non-invariance 实例已并入 `核心未解决问题_相变与代理迁移.md` §2.7**（proxy 信号随规模改变符号的最强业界证据，400M→1.5B 与微软 5B→23B 尺度差同数量级，proxy 结论需中间验证点）。另：应用户要求删除此前误写入 nanochat-npu 的 ppl_filter.py，研究文档恢复纯方法论描述，nanochat-npu 已确认 untouched（git status 干净）。总计 91 篇论文（标记 25 篇，深度分析文档 34 个）
- **2026-09-02（数字纠错）**：用户追问 #88 收益量化时复核 Table 1/2 原文，发现此前记录有拼接错误："8B 反转 AIME 43.33 vs 6.67" 实为把 8B 低似然的 43.33 与 **1.5B** 列的 6.67 拼在一起。已修正（`CapacityDependent_深度分析.md` + 本 README 两处）：正确对比为 **8B AIME（5 epochs）低 43.33 vs 高 40.00**；更稳的反转证据用平均分（8B：低 62.62→70.69 (+8.07) vs 高 66.16→66.64 (+0.48)，epoch 2.25 交叉后持续领先）。同时补入 1.5B 收益量化（此前缺失）：**1 epoch avg 24.16 vs 14.77，10/10 数据集全胜，MATH500 +12.8/GPQA +12.1/KAOYAN +11.1；低似然再训 4 epochs 最好 16.50 仍够不到高似然 1 epoch 的 24.16（No Catch-up 定量实锤）**。另澄清：PPL 过滤建议中的 [q0.5%, q95%] 分位带为工程默认值（非论文数字，论文用 argmax/argmin 二分无阈值），已在文档中标注出处。
- **2026-09-02（#88 实验设计深挖）**：用户连续追问"argmax/argmin 二分到底是什么/似然公式/与代理模型冲突"，`CapacityDependent_深度分析.md` 更新：① **§0.1 学生似然定义**：∏ₜ πθ₀(yₜ|x,y<ₜ)，实现=frozen 学生基座前向 logprob 求和；⚠️ 求和式使 D_high 偏向简短回答（长度混淆，论文未讨论，用户过滤方案应用平均 NLL 规避）；② **§0.2 实验设计逻辑**（配对受控实验）：同题钉死只换回答→学生似然推两极造 D_high/D_low→同配置双学生对决→5 尺寸重复得容量模式；③ **§4.2 场景差异升级为四维表**（数据本体=教师 trace vs 原始文档/选择单位=同题配对 vs 跨文档过滤/训练方式=SFT vs continual pretraining/比较结构=数据集对决 vs 过滤决策）——**+9.4 pp 是配对蒸馏效应大小，文档池按 PPL 过滤的收益无人量化过**；④ **与代理模型线的关系定论**：#88 与 proxy 线天然冲突（数据价值随容量变），按信号粒度分层处理——400M 只做粗方向筛（可信）、配比排名/最优数值需 1.5B 中间验证点（存疑）、文档级 PPL 判断必须 1.5B 自打（不可信，400M 系统性高估难度）。（PPL/NLL/似然的数学补习内容为会话内讨论，不属论文内容，未入文档。）
- **2026-09-02（高价值论文关键 Takeaway 汇总入总表）**：应用户要求，`论文总表.md` 表头后新增 **"[高价值] 论文的关键 Takeaway"** 板块——全部 17 篇 [高价值] + 1 篇 [高价值/重要] 论文各压缩为一条针对用户场景（1.5B/1.8B tokens/STEM accuracy/N=1-2）的一句话行动向结论，含 #84 Puro-2B（数据配方是最大杠杆 2.40×、配比收益穿透 SFT 才算数、P2 Math 18.3% 可当初始候选）与 #88 Capacity-Dependent（1.5B 别灌超难、打分用 1.5B 自己、+9.4 pp 是蒸馏数字不可平移）。目的：打开总表即可一目了然看到每篇高价值论文最关键的东西，无需点进深度分析文档。
- **2026-09-03（新一轮 arXiv 搜索 08-30→09-03）**：8+7 个角度（data mixture/data selection/mid-training+continual pretraining/curriculum/scaling law×data/domain weighting/perplexity×filtering/STEM×pretraining/annealing/synthetic data/ti:"mid-training"/data ordering/checkpoint averaging/influence/KD），新增 4 篇（#94-97）：① **#94 KD During Mid-Training**（初判 [高价值] T2，**同日判级修正为 [新发现] T3**，见下条；Meta+Princeton+CMU：Zettlemoyer/Stella Li/Pang Wei Koh，代码开源 facebookresearch）——**forward KL 蒸馏在 mid-training 阶段行为反转**（pre-training 双赢 vs mid-training 拖慢 factual recall），归因=教师置信度跨域不对称×学生演化知识状态，Switch Distillation（教师熵路由）修复且收益穿透 post-training；机制层"数据价值依赖训练阶段"与 #88/#01 直接连线；② **#95 REER-PT**（[有意思] T3，StepFun 数据团队）——PPL 作优化信号**修复而非过滤**难 continuation（插入推理注释），680M +2.07 pp，与用户 PPL 过滤线互补（超难数据补中间步 vs 丢弃）；③ **#96 PLES**（[新发现] T3，Meta Ax/BoTorch 原班团队）——不确定性缩减采集函数估计最优超参 scaling law（<1/10 网格算力），proxy 实验设计可借鉴（选最能减少 1.5B 外推不确定性的 400M 点，而非盲扫）；④ **#97 Qwen3.8-Next**（T4，Qwen 官方 TR）——架构为主，但含 loss≠accuracy 新数据点（n-gram 词表 loss 单调降而 accuracy 饱和）。已同步：总表 4 行、快速索引 25→28 篇、T2/T3/T4 索引表。排除：Instella-MoE（已为 #86）、Probing Factual Knowledge Transfer（EMNLP'26，多语言事实迁移，偏题）、SMELT/TuringLLM（架构/物理 AI 领域）。总计 95 篇论文（标记 28 篇，深度分析文档 34 个）
- **2026-09-03（#94 判级修正）**：用户指出 #94 KD During Mid-Training 内容主体是蒸馏（训练目标设计）而非数据配比/选择——复核确认：**方法本体（Switch Distillation）是蒸馏损失函数修改，用户管线（raw 文档+NTP，无教师）用不上**，初判 [高价值]（=直接可用方法）过宽，降级为 **T3 + [新发现]**。保留的真实价值：① 机制层"数据价值依赖训练阶段"（教师置信度跨域不对称 × 学生演化知识状态）——数据价值轴（数据量 Sorscher / 容量 #88 / 训练阶段本篇）新维度；② 条件性直接相关：将来 STEM 池掺教师生成数据（REER-PT 式注入或 KL 蒸馏）时，forward KL 拖慢事实记忆的坑立即生效，Switch（教师熵路由）是现成修复。已同步：总表 #94 行改写、takeaway 板块移除该行（板块仅收录 [高价值]）、README 快速索引标记改 [新发现]、T2→T3 索引表迁移。标记论文总数不变（28 篇）。
- **2026-09-03（第二轮搜索 08-28→09-04，再补 20 个新角度）**：data pruning/sample selection、quality filtering、token/example-level weighting、SLM、dedup/repetition、data curation、compute-optimal/data budget、ti:"pre-training data"、selective LM/reference model、knowledge acquisition、math reasoning corpus、annealing×LM、mixture ratio、data attribution、textbook、web crawl、loss prediction/proxy model、curriculum×pretraining、domain adaptation×CPT、reasoning data。**结论：本窗口数据配比相关工作已被前轮基本覆盖，无强新命中**；新增 3 篇边缘相关（#98-100）：① **#98 OpenEuroLLM Scaling Laws**（**[新发现]** T3，Jülich/Turku 联盟+**Aaron Klein**——BOHB 作者/AutoML 权威）——LR/batch 联合+边际 scaling law、WSD annealing 增益广谱测量、**overtraining regime 显式 scaling form 建模**（用户 TPP≈1200 同区，与 #37 互补）、全部 run 开源（proxy 拟合外部验证集）；② **#99 SPARK**（T4，Qizhi Pei+Lijun Wu MSRA 数学推理团队）——科学文献推理数据合成（Spark-234K），STEM 数据形态参考，⚠️ 明确高难度数据+1.5B=够不着区（#88 警示）；③ **#100 Where Induction Runs Out**（T4，⚠️ 单作者无可查机构但预注册+开源）——MDL 两部分编码难度信号，"apparent competence tracks recognition not induction"（与 #88 死记同构），PPL 的互补难度度量（显式分离结构 vs 死记）。排除：DIEM（RFT 样本挖掘）、CPR（SFT 路由）、Verification-Aware Training（投机解码）、Output Format Confounds（指令微调）。总计 98 篇论文（标记 29 篇，深度分析文档 34 个）
- **2026-09-08（用户指定：MiniCPM5-2B 新开源工作调研）**：MiniCPM5-2B（OpenBMB，2026-09-07 发布，2B dense/131k ctx/2B 级 SOTA avg 53.9）调研完成，新增 2 篇（#101-102）：① **#101 UltraData: Tiered Data Management**（**[高价值]** T2，arXiv 2602.09003）——**L0-L4 分层数据管理框架**，MiniCPM 系列数据底座 + #84 Puro-2B 的 UltraData-Math 118B 出处；L0 数学专用 HTML parser（magic-html+LaTeX 标准化）→L1 启发式清洗+文档去重→L2 专有模型标注种子**蒸馏轻量分类器**全量质量分级→L3 重写/合成精修；核心主张=数据按层级**跨训练阶段策略性分配**（pre-train/mid-train/alignment 各取所需层级）。**UltraData-Math 290B tokens**（L1 web 170.5B/L2 精选 33.7B/L3 精修 88B）在 **MiniCPM-1.2B** 上实证 MATH500 37.02（+3.62 vs Nemotron-CC 4plus）/GSM8K +3.34 且 code/general 不掉——**1.2B 与用户 1.5B 规模几乎重合的数学域预训练增益公开参照**。对用户三重价值：L2 学习式分级=PPL 分位过滤的互补方案；L2/L3=STEM 池高质量补充源（Apache-2.0）；中训练取高层级数据的分层分配原则；② **#102 MiniCPM5-2B**（T4）——**Base→Midtrain→SFT→Final 四 checkpoint 全链开源**（公开生态首次 Midtrain checkpoint 单独发布）：用户可下载 Base+Midtrain 直接 diff 工业级 2B mid-training（math/code/长上下文适配）的 STEM accuracy 增益，给自己 1.5B mid-training 期望收益定标；评测参照（final）：AIME'25/'26 86.5/86.5、MATH-500 94.6、GPQA-D 70.2。已同步：总表 2 行+takeaway 板块 #101 一行、快速索引 29→30 篇、T2/T4 索引表。总计 100 篇论文（标记 30 篇，深度分析文档 34 个）
- **2026-09-09（深读 #101/#102 双连）**：完成 `docs/UltraData_深度分析.md`（arXiv 2602.09003 论文 27 页全文）与 `docs/MiniCPM5_2B_深度分析.md`（官方 README 全文）。**#101 UltraData 核心发现**：① 三种数据验证协议（Efficient 10B 快筛/Pre-train 120B/**Decay 100B 定论**——1.3T 完成基座+30% 待验+70% 默认混合续训+LR 指数衰减 7.5e-4→3.75e-5，μP+WSD），**Decay Verification 与用户 continual pre-training（ClimbLab 基座+1.8B 混合续训）结构同构**——400M proxy→1.5B 主实验两级结构的工业版模板（协议细节可抄）；② **L2 vs L1 增益小（MATH500 +1.42/GSM8K -1.74）而 L3 vs L1 +9.24pp/+7.13**——学习式筛选不够、格式重构（难度分层 QA/7 persona 师生对话/多风格改写/知识点教科书模块）才是数学能力跃迁来源；对用户 PPL 过滤线的定位校准：过滤是必要非充分，别高估；③ **Math-L3 跨域迁移**：English avg +3.45（OBQA +15.40/BBH +5.87/ARC-E +5.29）、Code +3.80（MBPP +4.56/HumanEval +3.05）、Chinese +1.89——高质量数学数据是通用推理驱动力，r=0.7 STEM 重仓合理性的直接 1.2B 实证（但 tiered 在常识广度任务 HellaSwag/PIQA/SIQA/Winogrande 轻微回退=深度换广度信号，通用防护仍需监控）；④ **tiered vs mix 同配比对照**（120B 从零，域分布 50/25/8/17，L1:L2:L3=1:1:1 仅出场方式不同）：顺序出场 L1→L2→L3 总 avg +1.49pp、Math avg 4.60 vs 2.36 近 2×、后段增长 1.7×（28.35→31.66 vs 28.26→30.17）——**出场时机是配比之外的独立轴**（与 #84 课程同族；用户 70/30 协议期间不动，配比定论后作独立小实验，⚠️ 120B vs 1.8B 尺度差 66×）；⑤ L2 学习式分级（种子标注→蒸馏轻量分类器）与 PPL 分位过滤正交可叠加但实证增益小；⑥ L0 数学专用解析（magic-html+MathML/KaTeX/AsciiMath→LaTeX 标准化）是数学数据第一道门槛；⑦ 局限：L2 仅 preview、N=1 无方差、mid-training 直接证据间接（主张 L2/L3 用于 mid-training 但实证是 decay verification+从零）、L3 教师合成存在 MATH500/GSM8K 泄漏面。**#102 MiniCPM5-2B 核心结论**：diff 定标实验设计定稿——下载 Base+Midtrain（BF16 各 ~5GB，走镜像站）同协议跑用户 STEM accuracy，**只取 Δpp 不取绝对分**（tokenizer/架构不同）；解读框架：若 diff 小=能力播种假说（增益在 SFT/RL 才兑现，呼应 #94 训练阶段依赖性）→直接影响 1.8B 中训练期望管理与评测协议设计；官方评测数字含 RL+OPD 平均 +10.96 须先扣减；⚠️ mid-training token 数/配比黑箱（diff 只能测总增益不能归因）、131k 长上下文成分混杂、仅 2B 有 Midtrain checkpoint（MiniCPM5-1B 无）。已同步：总表 2 行深读注记+takeaway #101 更新、快速索引 #101、T2/T4 索引表、文件树（并修正总表树注释 91→100）。总计 100 篇论文（标记 30 篇，深度分析文档 36 个）
- **2026-09-11（用户决策记录）**：① **MiniCPM5-2B diff 定标实验不做**（工程成本与收益不匹配，已记入 `MiniCPM5_2B_深度分析.md` §3 备查）——改为零成本静态定标：引用 #101 decay verification 数字（1.2B+100B L3 数据→MATH500 37.02）作为"≈用户规模+50× 预算+顶层质量"的增益上限参照；② 顺序实验具体化（用户询问后定稿）：STEM 池按**数据源质量层级**分三档（3plus/未分档 web→4plus→L3 级合成），1.26B STEM tokens 三段各 ~420M 低档先喂高档压轴，通用 30% 全程均匀陪跑，同总量同配比唯一变量=顺序，N=1"赢了白赚"型；⚠️ 前提=池内有真实质量方差（与掺 L3 源绑定）、66× 尺度差可能缩到噪声
- **2026-09-14（MiniCPM5 TR 追查→MiniCPM4 TR 数据章节）**：用户要求抓 MiniCPM5 TR 细节。**发现 1：MiniCPM5 无独立 TR**——总表 #102 原引的 2506.07900 实为系列 TR=**MiniCPM4 TR**（2025.06，44 页），引用错误已修正；GitHub README 其余 arXiv 链接（InfLLM-V2/HyPE/JustRL/Rethinking-OPD）均为架构/post-training 论文，与配比无关。**发现 2：MiniCPM4 TR §2.2/§2.3 通读，四条发现（用户筛选后仅第 1 条入库）**：① **ScalingBench**（✅ 已记入 `核心未解决问题_相变与代理迁移.md` §2.8）——小模型下游 accuracy 不可用的解法：下游验证集+强模型生成解题轨迹+测**条件 NLL**，loss 与 performance 呈 sigmoid（0.36B-4B 拟合/7B+80B 外推吻合）；直接命中用户 400M proxy 信号问题（nanochat token 级 loss 可直接实现）；⚠️ 偏旧（2024 方法/TR 2025.06 复用，#97 loss≠accuracy 解耦数据点构成反证压力）、sigmoid 是跨尺度校准非跨配比 rank 保证（§2.7 仍适用）、轨迹质量依赖强模型——**用户判断：有价值但偏旧，标记为参考线不作主路径**；② Efficient Verification 成本锚点（1B 模型单次数据验证 110 GPUh vs 从零 1200；协议与 #101 完全一致=生态标准方法学交叉验证）——不入库；③ fastText 质量分类器完整配方（dim 256/lr 0.1/ngram 3/minFreq 5/3 epochs/阈值 0.5；15T tokens 80 CPU<1000h；正种子高效验证法选出/负种子多源随机抽）——不入库；④ <10B 开源模型可做 L3 式重构（教科书+论坛范式，弱化"L3 只能复用不能自产"警示，数学域未验证）——不入库。MiniCPM4 TR 本身不建总表条目（15 个月偏旧+单点价值）。总计 100 篇论文（标记 30 篇，深度分析文档 36 个）
- **2026-09-14（ScalingBench 业界认可度调研）**：用户追问该做法业界认可度。溯源：ScalingBench 出自 **Densing Law of LLMs**（arXiv 2412.04315，2024.12，THU+ModelBest，Xiao Chaojun 等，无正式 venue，论文主结论=能力密度每 3.3 个月翻倍）。调研结论（已补入 `核心未解决问题_相变与代理迁移.md` §2.8）：① **方法家族（两步 loss→accuracy 映射）是业界主流**——论文自述 Inspired by Llama-3（Meta 工业标准做法），RegMix（benchmark NLL 配比回归目标）同族——信号家族有头部实验室背书；② **具体实现认可度中等偏弱**——52 引用/21 个月（2025:24/2026:27），引用构成绝大多数是"3.3 个月翻倍"头条结论的跨领域泛引，方法学采用者几乎只有 OpenBMB/ModelBest 自家（MiniCPM4 TR 复用、VoxCPM2 TR 2026 引用）；③ **条件 NLL on CoT 轨迹（相对 RegMix/Llama3 的差异化技巧）无外部系统性复验**；2026 风向对 loss→accuracy 映射更谨慎（#97/#73）。**定位修正：非"偏旧旁技"，而是"Meta 验证过的信号家族 + OpenBMB 内部用 15 个月的实现、外部零复验"——激活前必须在自己栈上复验 sigmoid 校准与 CoT 轨迹两步**。工具记录：arXiv API/S2 持续限流（仅 citations 端点间歇可用），Densing Law PDF 已存 `/tmp/opencode/n3/densing.pdf`
- **2026-09-14（新一轮配比论文 sweep 09-04→09-14 窗口）**：用户聚焦数据配比/配方轴（暂不关注数据合成/生产；#74 Nemotron-CLIMB 用户已在复现，暂不深读）。arXiv API 与 S2 双双限流，改走四通道：arxiv listing（`/list/cs.CL|cs.LG/recent?show=1000`，覆盖 09-07→09-14 公告 1492 篇标题；月度 listing 路径 404）+ hf-mirror daily pages（09-04 残缺日，05/06 周末无公告）+ `/abs/` 批量抓取（标题/摘要/作者）+ `/html/` 全文（机构核查）。新增 7 篇（#103-109），**窗口内无 [高价值] 级命中——09-03 两轮"配比领域安静"的判断在 9 月中旬继续成立**：① **#106 Curriculum as Transport**（T3 **[有意思]**，COLM'26，MSR Alvarez-Melis）——Wasserstein 课程路径解耦 ordering/exposure/端点平滑/pacing 四因素，固定预算受控比较；核心发现：**无普适最优顺序（强上下文依赖）+ 课程收益不能仅由累积曝光解释（优于 exposure-matched 静态采样）+ pacing/端点平滑决定课程在难度谱哪段有效**——三档顺序实验的直接设计参考（与 #101 tiered vs mix 同曝光对照互证）；⚠️ 合成套件（12 任务/33 难度轴）非 LM 预训练；② **#103 Building Multilingual Bridges**（T3，Cohere Labs，Tiny Aya L2-Thinker 3.35B）——SFT 阶段"数据混合作为泛化支柱"：推理是语言无关行为，经精心混合跨语言迁移、无需每语言推理监督（与 #101 Math-L3 跨域迁移同构）；⚠️ SFT+多语言非用户 CPT 层；③ **#104 Climate-ModernBERT**（T3，UZH+McGill/Mila）——域适应 CPT 语料构成对照：**参数空间合并（独立专门化 checkpoint）优于联合混合训练**、更好保留异构语料互补信息；学术语料最强适应信号；merge-vs-mix 少见受控对照=#75/#84 权重平均家族 CPT 新数据点；⚠️ encoder+判别式+领域 NLP；④ **#107 Amortizing Scaling Law Construction**（T3，TU/e+ELLIS Tübingen/MPI，workshop 5 页）——scaling law 拟合的 BO 采集：compute 渐进采集+surrogate 幻想补全，10-100× 节省；#96 PLES 同族，proxy 点选取顺序参考；⑤ **#105 MoE 重复数据**（T4，Stanford+UW：Jha/Margaret Li/Leskovec/Liang/Zettlemoyer）——MoE 4× 重复开始退化 vs dense 8× 近无损，随稀疏度加剧；用户 dense 无关，重复轴 MoE 边界数据点；⑥ **#108 Edu-QuRating**（T4，QuRating 后续多维教育评分，蒸馏打分器，9 基准超 FineWeb-Edu；选择轴当前不聚焦）；⑦ **#109 Data Scout**（T4，IBM Research，LLM taxonomy→定向爬取领域预训练语料，反过滤范式；生产轴当前不聚焦，STEM 池将来缺数据可回看）。排除：LeDQeR（企业数据质量规则生成，非 LM）、MoE/剪枝/视觉关键词噪声。已同步：总表 7 行+统计区修正（91→107；修正旧计数漂移：T1=17/T2=26/T3=43/T4=21、文档引用 77/仅索引 30）、快速索引 30→31 篇（#106）。总计 107 篇论文（标记 31 篇，深度分析文档 36 个）
- **2026-09-15（用户指定：Stanford 语义聚类 + 535B 开源项目调研）**：定位为 **Marin 535B-A23B Hero Run**（Stanford 起源：David Hall + Percy Liang 2024 创建，现 Open Athena 主导；535B 总参/23B 激活 MoE、18T tokens、2026-08-21 启动训练中；launch note 2026-09-03）。深读其 GitHub issue 体系（#8435 hero run、#9126 换混对照、#7998/#8208/#8303 聚类管线、#8452 权重提升、#8425 uncheatable、#6969 桶无关代理设计、#8633 迁移 GP、#8453 多轴标注 roadmap）、harrier_mix JSON（200 单元格三阶段权重）、簇总览页，新增 **#110（T1 [高价值]，首个工程项目条目）**+`Marin535B_Harrier_深度分析.md`。**核心发现（配比轴）**：① **混合空间**=40 语义域×5 质量桶（23.11T tokens，292 源）；② **权重选择三层协议**：BO（monotonic regret，d768）→d512 共享前缀多种子 swarm（新/旧混各 3 种子从同一 step-393 checkpoint 恢复、训练至终点比 final BPB——**直接适用于用户 r 扫描设计**，且其"60M@22% 进度 Spearman 0.298"=早期检查点排序不稳定的又一独立证据，与 Proxy Reliability 结论一致）→缩放梯子事前验证（60-80% 检查点线性外推+小宽度回测残差校正，1.07× 等算力加速判定换混值得）；③ **桶级曝光上限 2.96 epochs**（重复度硬约束进混合选择）；④ 7 个零权重 uncheatable held-out 数据集防指标污染；⑤ 三阶段调度数值：Q4 质量桶冷却期 22.9→28.0%（annealing-on-quality 第三独立来源）、agent transcripts 簇全程上升 1.3→6.6%、math 簇下降 4.3→1.4%（⚠️ BPB 目标函数差异，不构成对 r=0.7 的反证，存档不行动）。**核心发现（聚类轴，ClimbMix 复现直接相关）**：两级聚类（5k 球面 k-means 微簇→加权分裂+爬山粗化→40 域，源均衡采样下限+每域≥1% 人口）+质量正交轴（nemotron_88k 融合分+内容类型分类器→**按内容类型校准的 Q0-Q4 截断**：同一分数对代码/数学/散文含义不同）+嵌入兼用于模糊去重；**最重要**：mixing_via_embeddings 设计（#6969）以冻结微簇基直方图 h=V·w 重参数化混合，使代理模型对桶重定义稳健——**CLIMB（用户基座来源）被点名为反面教材：位置性预测器，重聚类即重触发 112-run 扫描**；复现启示=先冻结微簇基再改簇数/质量定义，已积累的 proxy 结论可保留。**可信度**：高于绝大多数论文（全程可审计：预注册 18T Paloma loss=2.04、权重 parquet/对照 CSV/analyze.py 全公开、自我诚实标注"swarm 选混又评混非独立验证"）；Marin 8B 对 Llama 3.1 8B 16/19、129B 预注册命中 1% 内、Delphi 300× 预测误差 0.2% 内。**边界**：目标函数 macro BPB 非 STEM accuracy（#68 LogFloor 同构陷阱未解），尺度差 4 数量级，数值不迁移。工具通道记录：GitHub API（issues 正文 JSON）+ raw.githubusercontent（权重 JSON/文档）+ GCS 公开桶（簇总览/对照报告）+ HN Algolia 搜索（项目定位）全通；openathena.ai 博客页 401 需登录（正文经 HN 转述与 marin.community 补齐）。已同步：总表 #110 行+takeaway+统计（107→108，T1=18，[高价值]=18）、快速索引 31→32（#110）、T1 索引表。总计 108 篇论文（标记 32 篇，深度分析文档 37 个）
- **2026-09-15（Marin 讨论沉淀 + 语料/代理深挖）**：用户深挖讨论后新增两处沉淀。① `Marin535B_Harrier_深度分析.md` 新增 §8.5（research.md 硬数据）：**LightGBM（literal RegMix）= 弱基线，生产代理 DSP（手写饱和/惩罚参数化）OOF 0.914 > 900 列特征工程 0.86**——数据少噪声大时结构先验赢黑盒，参数化定律流派（InfoLaw/#27）再加一票；**60M→300M identity transfer Spearman 仅 0.777**（proxy 不可靠第三独立来源）；形状问题（平滑饱和 vs 相变的 misspecification，r=0.7 远超 1.5B 阈值故局部安全、危险仅在 400M proxy 与小 r 探索）；N=1-2 防自欺三件套（预注册/赢者诅咒/独立性标注）；② `核心未解决问题_相变与代理迁移.md` 新增 §1.3 **三道门分解**：门1 知识获取阈值（数据轴，r=0.7 已过）/ 门2 走完相变的训练量（r<0.4 超指数贵、r=0.7 便宜但 1.8B 是否走完不可先验）/ 门3 accuracy 起跳点（能力轴，= 核心问题本体，只能测）——推论：中途定期评 benchmark 并记录起跳时机（比终点分数信息量大）、跨 regime 只带锚点不带公式。③ 语料构成提取（簇总览页源富集信息）：Marin 18T 大量复用 **Nemotron 语料家族**（nemotron_cc_v2/v2_1 各质量档+synthetic、nemotron_code、nemotron_cc_math_v1 含 4plus_mind、nemotron_specialized）+ 多语言（hplt_v3/common_corpus/finetranslations/finepdfs/dolma4pdfs/**climblab-ja**）+ 代码补充（stack-v3/swe-rebench）+ 专门源（uspto/edgar/institutional_books）；自产部分主要是 agentic transcripts 与再处理管线——**与用户基座（ClimbLab 系）有直接语料亲缘**
- **2026-09-18（增量 sweep 09-14→09-18 + AutoData 可信度核查/深读）**：arxiv listing 增量 + 6 组短语检索（data mixing/mixture/curation/selection/pruning/mixture weights）。窗口内最相关 **#111 AutoData**（T2，Weco AI+UvA，2609.19754）：agentic 搜索数据选择算法（AIDE 式 agent 在打分+分层+随机规则程序空间搜索，proxy 回路 d8/0.42B tokens/单 H100 10min/200 步×3 LLM）。可信度核查：1 天预印本在投（致谢含"感谢审稿人"）、**code 仓库 404 未放出**、NVIDIA DGX Cloud 算力、Karpathy 生态（底座 nanochat、引 autoresearch、动机引用 NanoChat "FineWeb-Edu→ClimbMix 省 27% wall-clock"）；统计纪律尚可（多 seed+配对 t 检验+诚实 limitations）。**底座与用户栈同构（nanochat + ClimbMix 全池选 2.6%）**，深读三结论：① **精炼池质量轴已死**（DCLM 倒退 0.9954 vs random 0.9484、PPL 中段倒退、RegMix K=24/128 Dirichlet/LightGBM≈random——与 Marin LightGBM 弱基线互证→r 轴是 Climb 级语料唯一残值方向）；② **效果量校准**：最优选择 vs 随机 bpb ~0.1% 相对、CORE 22 任务全不显著（即使以 CORE 为搜索目标）+ proxy 迁移 897M 有效→1.3B 消失（支持 400M 海选+1.5B 决赛两点式；门3"accuracy 轴小尺度噪声主导"旁证）；③ **subsample 协议**（固定子样本只变种子、4 种子并行、配对 t 检验）+ 零过滤配方原语（多样性门控 PPL/分位数三角核带状卫生分/topic×format 576 格 shrinkage centering/3 文档 Gumbel 锦标赛/1/3-1/2 随机骨架/**源序块先验**——agent 自己发现 ClimbMix shard 顺序携带簇结构）；LLM 标注特征（n_factual/n_rsteps/n_rerrors，提示词在附录 A.3）仅 +0.2σ 不值。⚠️ 摘要称 CORE 提升但正文不显著（轻度 overclaim）。其余窗口候选存档不深读：Stress-testing Alignment Midtraining（2609.20412，110B/1B tokens AMT 假设检验，中训练阶段参照）、2609.18960（合成数据选择器审计排名 vs 下游效用 Spearman ρ=0.04——代理不可靠又一独立证据，已录核心问题文档 §1.3/§2.9）、QVAC Genesis III（191B 合成 STEM 语料 19 域×难度分层，合成轴不聚焦）、ZGCM-1（2609.13356，全开源 7B 配方：渐进上下文课程 16K→64K→256K+MDP 化中训练）、SEA-LION-v4.8（Nemotron-3 CPT 报告，与 ClimbLab 同族基座做法）。Marin 侧动态：hero run 30.5%（5.48T/18T tokens，step 118,901/390,251，ETA 11/27）、#8870 静默 GPU 挂死以 PDL-off 构建压住（2 天零复发，根因未捕获确认）、#9212 SFT 重复/eval 泄漏审计例程化（159 源/1.13 亿对话）。已同步：总表 #111+统计（108→109，T2=27，仅索引 31，无标记 67）、T2 索引表。总计 109 篇论文（标记 32 篇，深度分析文档 37 个）
- **2026-10-10（对话跨 3 周恢复：AMT 录入 + #111 修正 + 三周空窗 sweep + Marin 复查）**：发现上次会话（09-18）后对话跨约三周恢复（Arcadia 官网 10/9 构建 + 系统时间 10/10 确认）。① 录入 **#112 Stress-testing Alignment Midtraining**（T4 **[有意思]**，Arcadia Impact 伦敦 AI safety 非营利，与 UK AISI alignment 团队紧密合作、UK AISI Alignment Project + Coefficient Giving 资助，Daniel Tan（Resolution）/Andrew Draganov 等 6 人，arXiv 2609.20412）：第一个 AMT 系统 stress-test（虚构零污染 Dispatch/Python 4 双设定，gemma-3-12b/27b+GLM-4.5-Air 106B 全参，1M-1B tokens）。五条脆性发现（方向性可信、单种子数字勿引用）：**2% 冲突 SFT（~80k tokens）推翻 190M token 中训练动机**（90%→13%）；EFT 构成>中训练剂量；held-out 泛化弱且 EFT 主动侵蚀未引发知识；去示范换说理 uplift 缩水 0.35-0.73×；SFT→RL 后动机利用 34%→8%（thinking 逐字引用章程却选反面）；gemma-3-4b 全剂量无感。对用户=中训练阶段约束：收益须 SFT 后测（#84 互证）、语料须含示范（#101 L3 互证）、未引发知识拿不到、RL"在而不动"风险、剂量曲线平快；⚠️ 价值观轴≠知识/STEM 轴不迁移数字、公开复刻 AMT 或弱于前沿、三周零社区消化。② **#111 修正**：AutoData 代码已放出（WecoAI/AutoData，MIT，2026-09-18 当日推送——深读时 404 系时点差异，10-10 复核 8 stars），总表/索引表备注已更新。③ **三周空窗 sweep（09-18→10-10）**：8 组短语检索+listing，候选 10 篇待深读裁决：**Generalization Dynamics of LM Pre-training**（2609.33150，Berkeley/Stanford/GDM：Wen/Wu/Song/Lijie Chen——"mode-hopping"：预训练全程 LMs 在鹦鹉式与智能式计算间突然跳变，直击核心问题文档的相变轴）⭐、**Conditional Transfer from Controlled Pretraining Mixtures to Code**（2610.11548，单作者 Ohad Rubin——diagnostic/teachable/transfers 三信号分解，loss 下降≠下游迁移的受控实验）⭐、**When do data mixtures improve scaling laws?**（2609.38011，IST Austria Mondelli/Misiakiewicz——高维混合回归理论：辅助数据何时真改进 scaling law）⭐、**Trajectory Soup**（2609.37169，SJTU+蚂蚁——中训练算力天花板与共享 checkpoint 分叉/轨迹合并）、**Fisher-Guided Submodular Data Selection for CPT**（2610.02593——CPT 遗忘控制选择，用户 CPT 阶段同构）、**OptiSelect**（2610.03432，EPFL Jaggi——优化器感知在线选择，Muon 时代选择理论）、What Pre/Midtraining Make Learnable from Rewards?（2609.38446，理论——中训练供给 RL 的信息/计算，与 #112 发现⑤互补）、AutoDataBench（2609.40097，非 Weco 同名——数据智能测试床）、Multilingual Quality Classifiers（2610.11585，EPFL——质量轴多语言，低优先）、Scaling Down the Scaling Laws（2610.06387，单作者综述，低优先）。④ **Marin 复查（10-10）**：hero run **63.1%**（step 246,175/390,251，11.4T/18T tokens），**ETA 提前至 11-11**（9 月估计为 11-27）；10-06 Kubernetes 误删训练任务事故后无故障运行；David Hall 对新旧混 end-of-run loss 做 forecast-only 拟合（loss floors 重拟合）；cooldown 前 router capacity 是公开问题；#8818 决定 layers 0/1 近死路由专家留到预训练后再研究。已同步：总表 #112+#111 修正+统计（109→110，T4=22，[有意思]=4，有标记 26，仅索引 32）、快速索引 32→33（#112）、T4 索引表。总计 110 篇论文（标记 33 篇，深度分析文档 37 个）
- **2026-10-12（10 候选逐篇裁决：录 3 不录 7）**：应用户指令按优先度逐篇分析 10-10 sweep 的 10 篇候选，**新增录入流程门槛（用户决策）：记入总表前先审视必要性，无必要不录**。**录入 3 篇**：① **#113 Conditional Transfer**（T3 **[有意思]**，2610.11548，单作者 Ohad Rubin 无机构无代码）——三信号分解（诊断性/可教性/迁移性）+受控混比边（70% 固定+30% 三源单纯形，~180M MoE/15.7B tokens/统一 SFT 桥）：可教性≠迁移（最可教策展套件纯用 15.9 低于零合成基线 16.5）、最优为 75% 对齐+25% 策展内部点 22.6（条件价值非单调，与 r=0.7 哲学及 #14 同构）、**Ado 在线调度追 loss 可减性丢迁移**（free-ratio 5k 步内把对齐源压到 <5%、终点 1.2-1.4%，落后匹配固定混 2.4-11.0 pp，1k 步视界时间近视 ~88k 步才回摆）——**同尺度 loss 信号指错方向的受控反例**（#51/#97/AutoData 线，区别于 proxy 跨尺度问题）；统计纪律超同侪（置换 2000 次/BH-FDR/5 万配对 bootstrap 含 checkpoint 重选）；⚠️ 关键对比 95% CI 跨零（P(Δ>0) 82.7-93.3%）、MoE 小规模单下游（HumanEval）——**引方向不引幅度**。② **#114 Generalization Dynamics**（T4 **[新发现]**，2609.33150，Berkeley/Stanford/GDM：Jiaxin Wen/Zhengxuan Wu/Dawn Song/Lijie Chen）——**mode-hopping**：预训练全程在鹦鹉式与智能式计算间频发突发跳变（OLMo3-32B answer+1 相邻 checkpoint 81%→0%→81.7%，而 loss 与常规 benchmark 平稳）；OLMo3/Apertus 公开 checkpoint 全程扫描（9-90× Chinchilla 预算）+系统性排除三零假设（评测噪声/指标假象/优化动力学——单步稳定、K=5 平均只缓解不修复）；容量分配解读（泛化回路与浅回路竞争，数据窗口决定谁赢）；两应用：**继续预训练/中训练改善 in-distribution 不改善泛化/对齐**（4.5T vs 4.9T checkpoint 相同 post-training 后 GPQA 36.3% vs 29.8%、prefilling 鲁棒 53% vs 21%——用户中训练收益测什么的直接警示，与 #94 同族）+数据子集可定向控制泛化动力学（32B 小实验）；⚠️ 32B 尺度+泛化敏感指标（常规 benchmark 平稳→对 STEM accuracy 直接威胁有限）、无 code。**核心问题文档新增 §2.10（mode-hopping 证据+与三道门关系+#113 ADO 反例一段）+§3 表一行**。③ **#115 When do data mixtures improve scaling laws?**（T2，2609.38011，IST Austria Mondelli/Misiakiewicz 理论组，FWF+ERC）——高维混合回归三层理论（minimax 变分刻画/ridge 确定性等价/幂律显式 scaling law）：**混合严格优于任一单数据集 ⟺ 辅助谱重尾（互补覆盖 δ>1）且规模适中增长（γ_c<γ₂<1）**，否则只动 prefactor 不动指数；固定比例+重尾辅助渐近退化为辅助主导（正分布偏移）；词频级操作化（辅助域覆盖目标域低频 token，分布太近无改善）；81.5M nanoGPT/SlimPajama 定性验证（γ₂=1.25 改善/0.75、1.75 不改善）；对用户=源选择选互补覆盖而非冗余堆质量（与 #38 互补）+**r 最优值随预算增长漂移的理论根据**（固定 r=0.7 边界+1）；⚠️ 风格化线性模型+81.5M 单实验。**不录 7 篇（处置记录，均经全文/摘要核实后裁决）**：**Trajectory Soup**（2609.37169，蚂蚁 Ling+SJTU，高可信）——深读确认**分支变量=优化配方**（data-shuffling seed/LR/batch/调度/Muon 动量，附录表 8 逐一核实）**非数据配比**（原文明示 mixture 变体留作 future work）；串行中训练饱和渐近线 ~67.2（600B tokens/1.3B active MoE，与用户 1.8B 预算差 ~300×不构成约束）；汤合收益 +0.4-0.5 分且过 SFT 保留，但选择信号与报告分数不独立（D.5 自认）+用户 N=1-2 不可操作——**若未来出现"配比分叉+汤合"论文再跟踪**；Fisher-Guided CPT（2610.02593，UIC Lan/Yan + Cisco）——样本级选择器非源配比轴，Fisher 漂移机制（loss 选择侵蚀高 Fisher 已承诺坐标）有趣但用户固定比例精选池 CPT 用不上；OptiSelect（2610.03432，EPFL Jaggi）——from-scratch 在线选择×优化器交互理论，非用户 regime；What Pre/Midtraining Make Learnable from Rewards?（2609.38446，160 页理论）——RL 轴，#112 发现⑤已方向性覆盖；AutoDataBench（2609.40097）——agent 数据智能基准非配比方法（与 Weco AutoData 同名不同家）；Multilingual Quality Classifiers（2610.11585，EPFL）——多语言质量过滤偏轴；Scaling Down the Scaling Laws（2610.06387）——单作者综述。**统计区行级重算校准**（修正历史漂移：标记计数旧记 26/68，行级实为 33/77；本次录入后 35/78）。已同步：总表 3 行+统计+相互参照、快速索引 33→35 篇（#113/#114）、T2/T3/T4 索引表、核心问题文档 §2.10+§3。总计 113 篇论文（标记 35 篇）
- **2026-10-12（补录 6 篇：录入口径修正）**：用户纠正录入门槛口径——**"样本级选择也是数据选择配比的同类问题、在线选择也是关键问题；只要跟数据选择/配比相关都应记录"**（此前我按"源配比直接可用性"收窄，过严；此决策已记录）。据新口径补录 6 篇：**#116 Fisher-Guided Submodular CPT**（T3，UIC Zhiling Lan/Yan Yan+Cisco Research）——CPT 数据选择=遗忘控制：**参数不可知选择（PPL/分类器/随机）造成 post-CPT Fisher 对角在高 Fisher 已承诺坐标系统性下漂**（灾难性遗忘的参数空间机制，"loss 指错方向"机制级证据，#51/#97/#113 线）；anchor/frontier 双几何 log-det 子模目标+Sieve-Streaming 单遍流式；TinyLlama-1.1B/Llama-3.1-8B 医疗 CPT **1B 选择 tokens 超 10B replay（≥10× token 效率）**；⚠️ 样本级非源配比、无代码。**#117 OptiSelect**（T3，EPFL Jaggi）——首个优化器×在线选择系统研究：选择增益=优化器诱导效用分数的可判别性（Lion/Muon 符号几何坍缩封顶、AdamW/Sophia 对角自适应更优，四定理）；124M/720M 实测一致；**诚实反思：增益是样本效率非算力效率**（λ=2 等 read-tokens 对比全面落后标准训练）；**打分几何解耦**（D-Muon 优化器下 AdamW 几何反而是最佳打分几何）。**#118 What Pre/Midtraining Make Learnable from Rewards?**（T4，Chiwun Yang CityU HK+Xiaoyu Li UNSW，160 页理论）——#112 发现⑤的理论对应：**RL 只能在已存在计算中选择，中训练语料须供应可被奖励调用的计算形态**（八世界 82.61% vs 44.15%、memory replay 保持检索）。**#119 AutoDataBench**（T4，Ruifeng Yuan 等，有代码）——数据智能受控测试床（诊断/组织/构造×工具/检索/知识注入，固定非数据因素）+#111 AutoData 的评估基础设施对应物；复用轨迹做 mid-training 提升 coding。**#120 Multilingual Quality Classifiers**（T4，EPFL Jaggi 组）——质量评分轴（#108/#78 线）多语言扩展，英文场景备查。**#121 Scaling Down the Scaling Laws**（T4，⚠️ 单作者纯二手综述）——数据效率轴（TPP≈1200 过训练区、#98 同区）背景索引锚点。**Trajectory Soup 维持不录**（用户质询后复核：分支变量=优化配方非数据配比、原文明示 mixture 留作 future work——连数据选择都算不上，属真正轴外；如需改判随时可补）。已同步：总表 6 行+统计（113→119；T3=46、T4=27、仅索引 41、无标记 84）、T3/T4 索引表、文件树。总计 119 篇论文（标记 35 篇）
- **2026-10-12（用户标记裁定）**：用户对 9 篇优先度解析的反馈裁定——**#114 Generalization Dynamics 升 [高价值/新发现]**（用户："这个高价值"；已按高价值规则补总表 Takeaway 板块行：单 checkpoint 泛化读数不可信、中间 checkpoint 必须保存、继续训练改善 in-distribution 不改善泛化；⚠️ 边界保留——32B 尺度+泛化敏感指标，对 STEM accuracy 直接威胁有限）；**#117 OptiSelect 加 [有意思]**（用户："比较有意思"；优化器×在线选择系统研究+诚实算力账+打分/优化几何解耦）；**#118 "别把 general 数据滤太狠"结论获用户认可**（"有一定价值"，未加标记）。同步：快速索引 35→36（+#117 行）、标记统计（[高价值/新发现] 1、[新发现] 6、[有意思] 8；有标记 36/无标记 83）、标记说明表新增组合标记行。核实修正：#115 机构细化为 ISTA+Yale 理论组（原文 dual affiliation：Misiakiewicz=Yale Stats & DS、Mondelli=Yale/ISTA 双栖，此前误记纯 IST Austria）；#116 机构确认 UIC（University of Illinois Chicago）+Cisco Research。总计 119 篇论文（标记 36 篇）
- **2026-10-12（用户梯队裁定第二批+#115 可信度评估）**：**#116 Fisher CPT 降 T3→T4**（用户裁定："场景太小（医疗），可能不适用我们的场景，等级得低"——方法收益仅医疗单域证据不迁移；机制层警告"loss 选择系统性磨损高 Fisher 已承诺坐标"保留引用）；**#113 确认维持 T3 低位**（用户认可——T3 已是配比轴内最低档，T4 留给轴外相关）；**#115 可信度评估后保 T2**（用户质询"真实 LLM 训练场景是否适用"——评估结论：数学层高可信（minimax 上/下界+可证伪预言验证），LLM 适用性=原则级（互补覆盖选源/预算条件 r/辅助亚线性扩展），原则与 #27/#38/#14/#101 LLM 原生证据线独立收敛、本篇为其统一理论解释，δ/γ₂ 定量条件不迁移；总表条目已补评估注记）。同步：总表 #116 梯队+注记、README #116 行 T3 表移入 T4 表、统计 T3=45/T4=28。总计 119 篇论文（标记 36 篇）
- **2026-10-12（用户指定关键结论）**：**#114 "训更久 ≠ 更聪明"被用户指定为关键结论**——继续预训练/中训练只改善常规（in-distribution）成绩，不改善深层泛化/鲁棒甚至变差（4.5T checkpoint 经相同 post-training 后 GPQA 36.3% vs 29.8%、prefilling 鲁棒性 53% vs 21%，全面优于 4.9T final）。已置入总表 Takeaway 板块 #114 行（加注"2026-10-12 用户指定关键结论"）；操作含义：中训练终点不默认取 final，后期 checkpoint 逐一 STEM 直评比较后再选。总计 119 篇论文（标记 36 篇）
- **2026-10-12（用户格式裁定：takeaway 可读性）**：用户反馈"记录的 takeaway 令人费解，无法一眼看懂关键结论"——**确立记录规范：一句话白话版关键结论置前，细节与数字置后**（适用今后所有 takeaway/一句话核心价值列）。已按此改写快速索引 #113/#114/#117 三行。总计 119 篇论文（标记 36 篇）
- **2026-10-12（总表全行一句话 takeaway 化）**：应用户要求（"这版很不错，总表里面的关键结论都改成类似的形式（一句话takeaway）"），**全 119 行统一加 `**【一句话】**` 白话前缀**——#113-#121 用用户已批准版本，其余 110 篇据行内容新拟；**原细节全部保留在后**（一句话=速览层，原内容=细节层，两层并存）。总计 119 篇论文（标记 36 篇）
- **2026-10-12（补录 #122 MDE：引用成档）**：用户发现重点论文缺 MDE（Climb=#74/Marin=#110/MAI=#73 均已在库）——"Google MDE"被本项目 6 处深度分析长期当背景引用（Survey_RegMixD 方法分类、FastMix/PreSelect 行动建议、Marin #6326 反例、README 行动建议）却从未立项成行。已溯源核实补录为 **#122**（arXiv 2502.15950，Google DeepMind，Belenki/Agarwal/Shi/Toutanova，2025.02）：**Table 1：end-task 验证目标与下游 ρ=0.844 vs 域 loss 目标 0.282**（"验证目标选择 > 方法选择"，loss≠accuracy 轴最硬量化之一）；MDE 特征（k 个域专家小模型 loss 聚合近似混合 loss，理论保证）作回归特征 → 排名 0.81→0.95、仅训 k 个专家（3× 省算力）。**T2 [高价值]**（⚠️ 2025.02 preprint 补档非新发现；SlimPajama/生成排序基准非 STEM accuracy）。统计：119→120 篇（T2 28→29、标记 36→37、[高价值] 19→20、无标记 83 不变）。总计 120 篇论文（标记 37 篇）
- **2026-10-12（用户梯队裁定第三批：#73 MAI 升 T1）**：用户指定 Climb/Marin/MAI 三个工作全部放 T1——#74 Nemotron-CLIMB、#110 Marin 原本即 T1，实际变更 **#73 MAI-Thinking-1 T2→T1**（Microsoft AI 35B/1T MoE TR：千 proxy rank non-invariance 实证、memorization-aware epoch capping、中训练 STEM:Code:Other=35:55:10 与用户同区）。README 行从第二梯队表移入第一梯队表（加裁定注记），总表梯队+统计同步（T1 18→19、T2 29→28，总 120/标记 37 不变）。总计 120 篇论文（标记 37 篇）
- **2026-10-12（用户准则裁定：可信度评估三要素）**：应用户指令，三项考量确立为项目**准入级可信度准则**并写入 README 新增《可信度评估准则》章节（置于论文索引前）：①**实验规模为重大参考因素**（学术小规模结论不必然适用大规模，须显式记录模型/数据规模并对照工业规模，差 2 个数量级以上默认标 ⚠️ 规模外推风险）；②**作者与发布机构**；③**业界关注度**。总表免责声明同步指向该准则。背景：前向引用扫描发现两篇重复轴新论文（2610.05126：≤520M 小规模；2610.05591：14M-2B 八尺寸扫描），用户裁定规模权重显著上调——2610.05126 因 ≤520M 可信度不足暂缓录入。总计 120 篇论文（标记 37 篇）
- **2026-10-12（用户裁定：录入三篇重复轴论文，可信度准则首次应用）**：①**#123 What Is a Repeated Token Worth?**（2610.05591，ETH Zurich Chai+Xiong，无 venue）：多 epoch 定价几何 z=(R-1)N/U 单变量定律；14M-2B 八尺寸扫描（2B 达用户量级——用户认可"可信度还可以"）；②**#124 Repetition Mismatch**（**EMNLP 2026 Main**，Imperial College London 四作者）**[新发现]**：小规模配比外推失败元凶=**重复率失配**，子采样匹配目标重复率可 1/16 tokens 在 1.17B 恢复最优配比（0.10 vs 0.85）——直接命中本项目 proxy 外推核心问题；③**#125 Larger Datasets Can Be Repeated More**（**ICLR 2026**，清华 PACMAN，作者表与 #84 Puro-2B 重叠）：E(K,N) 定理——大数据集更耐重复（饱和值随 N 增长，强凸 Θ(log N)），修正 Muennighoff"≤4 epoch 无差"（#13 引证源）。三篇构成重复轴"实证几何+外推机制修复+理论"三角；**2610.05126（≤520M）维持暂缓**。统计：120→123 篇（T2 28→31、[新发现] 6→7、标记 37→38、无标记 85）。总计 123 篇论文（标记 38 篇）
- **2026-10-12（用户梯队裁定第四批：全库可信度准则审视，8 项调整）**：按新准则（实验规模/作者机构/业界关注度）全库审视，用户批准执行——**降级 6 项**：#26 FastMix T1→T2（预训练验证最大 1B、容量错位风险自记）、#20 Chameleon T1→T2（最大仅 1.2B）、#71 Domain Data Repetition T1→T2（TPP≈1.2 与用户 TPP≈1200 不同区、行内自记帮助有限）、#65 Task-Agnostic Influence T1→T2（梯队与 [有意思] 标记不齐+规模待核）、#117 OptiSelect T3→T4（124M/720M 差工业 2+ 数量级）、#70 Mixture Experiment T3→T4（无 venue 仅仿真）；**升级 2 项**：#58 GLM-5、#60 Nemotron 3 Ultra **T4→T1**（工业规模+唯一 Mid-Training 章节/19 类配比全公开。注：审视提案误记其原梯队为 T2，实为 T4，终点 T1 经用户批准不变）。保留不动：#01/#03/#09/#38（定理级豁免）、#10/#21/#27/#31（机构或规模合格，#31 已核 3B/100B tokens）、#73/#74/#110（工业规模）。#04 DeMix/#30 WebGraphMix/#66 DIDS 规模待补核。统计：T1 19→17、T2 31→35、T3 45→43、T4 28 不变，总 123/标记 38 不变。总计 123 篇论文（标记 38 篇）
- **2026-10-12（用户梯队裁定第五批：#30/#66 规模补核降级，#04 核查保留）**：按新准则补核三篇 T1 的实验规模（arXiv 原文+本地深度分析交叉）后用户批准执行——**降级 2 项**：#66 DIDS T1→T2（**[高价值] 标记保留**——目标模型 8B 够格，但训练仅 1B tokens、验证为 939k 样本 post-training SFT 非预训练/中训练、论文自述梯度聚类 100B 规模计算瓶颈、GitHub 1 star/3 commits 原型级关注度；机构 HKUST+MetaX+Alibaba 与 EMNLP'25 venue 合格）、#30 WebGraphMix T1→T2（400M/1B 模型、8B/28B tokens DataComp-LM 管线，距工业 2+ 数量级；Princeton 但无 venue）；**保留 1 项**：#04 DeMix T1（arXiv 2602.00747 核查：7 component×30B=211B tokens、Qwen3-1.7B/8B 双尺寸验证、22T token 数据集开源、ICML'26 accepted——三要素全过）。统计：T1 17→15、T2 35→37，总 123/标记 38 不变。总计 123 篇论文（标记 38 篇）
- **2026-10-13（重复率轴落地：核心问题文档 §2.11 + 行动建议修正）**：#124 Repetition Mismatch 的"重复率失配"机制写入 `核心未解决问题_相变与代理迁移.md`——① §1.3 新增推论 7（外推失败**第四种解释**：独立于模型尺寸、作用于预算轴、**可修复**）；② 新增 §2.11 完整节：机制（每源 epoch 随预算变、最优配比随之漂移）/证据（子采样匹配目标重复率后 1/16 tokens 恢复最优配比 0.10 vs 0.85）/与前三种解释（相变阈值随 M、容量竞争、checkpoint 跳变——均模型状态轴）的关系/**用户场景映射**（400M proxy 线实验设计修正：r=0.7 下数学池 1.26B token 目标预算 ~1.4 epoch vs proxy 1/16 预算不子采样 ~0.09；重复上界三参照 #125 E(K,N)/#123 定价几何/#73 NLL<0.01 capping；与门2 交叉——budget 参数化应携带 per-source epoch）；③ §3 汇总表新增行。README 行动建议：立即行动新增第 7 条"子采样匹配目标重复率"；原第 9 条（高质量数据重复 15-20x）加 ⚠️ 修正注记（#125/#123：固定倍数非普适、按池大小与预算显式计算）；全节重编号（现 32 条）
- **2026-10-13（前向引用扫描候选逐篇裁决：录 4 不录 6）**：10-12 前向引用扫描留下的 10 篇候选逐篇摘要全文+作者核查后裁决。**录入 4 篇**：① **#126 How Much Is an AI Token Worth?**（T2，2609.40295，UMD+UChicago 等）——wild AI 文本配比轴：800 模型扫 AI:human 掺入比例、benefit/harm 双项变号 scaling law（小模型拟合预测 3.6× 大模型、误差低 41%）、建议"先重复 human 再扩 AI 语料"与重复轴直接连线、WildAI 83B 标注语料+全模型开源；② **#127 SPOKES**（T3，2606.15216，UW Choi/Zettlemoyer/Koh+A*STAR）——G-Vendi 集合级多样性直接优化，质量×多样性联合 +1.4/+1.5 超语义去重/质量过滤（单独多样性仅 +0.4/+0.5：多样性是放大器不是替代品）；③ **#128 Educational or Just Wikipedia-Style**（T3，2605.23721，波兰组待核）——学习式质量分类器风格混淆陷阱（FineWeb-Edu CQF ~7% 文档可被维基风格化改写翻转放行），L2 学习式分级（#101）稳健性警告；④ **#129 Domain Mixture Design via LL-diff**（T3，2603.16622，京都 Shimodaira 统计组）——LL 空间几何定域权重（蒸馏的配比替代、pretraining/CPT 设定），目标错位但配比轴罕见 CPT 设定。**不录 6 篇（处置记录）**：TESS（2610.02092，实验场景=LLM 安全/指令微调 post-training，与用户 CPT 错位）、SynPro/Data-Bound Scaling（2605.17849，合成生产轴暂不聚焦，"格式重构>纯筛选"已由 #101 L3 覆盖）、Proxy Compression（2602.04289，名字撞车——tokenizer/压缩训练方案非 proxy 模型轴）、KletterMix（2606.03773，德语语料策展全报告，语料生产轴+语种域外）、Project Greenhouse（2610.11922，开源 agentic search reranker，搜索轴）、Visible but Not Yet Curatable（2608.28819，数字图书馆学术工件元数据轴）。统计：总 123→127（T2 38、T3 46），标记 38 不变，仅索引 49。总计 127 篇论文（标记 38 篇）
- **2026-10-13（Marin 盯梢快查）**：repo 当日仍活跃（10-13 push）。Hero run issue #8435 最新动态：① 10-09 bethehand 发布**独立预注册预测**（#9977）：最终 dropless Paloma macro 中位数 **2.035**、80% CI [2.023, 2.047]（数据至 step 227,999）——与 08-19 官方预注册 2.04 一致或略好；2.04 vs 2.10 差距主要溯源到 lm_head FLOPs 是否计入 compute；② 10-01 mcwitt-agent 发布 4K→8K seqlen 切换前单机架测量（#9615）：8K 比 4K 少 1.6% tokens/s、专家分配丢弃率升 4.5×（1.9e-4→8.3e-4，均在 sender 侧；16K 代价 3.1%/16×）。日程不变：cooldown 切换 ~10 月底（step ~312k/80%）、**ETA 11-11 预注册对答案**（原 2.04）
- **2026-10-13（#74 Nemotron-CLIMB 深读补档：悬空引用建成）**：建 `NemotronCLIMB_深度分析.md`（arXiv 2504.13161，v2 2025-11-30，NVIDIA+Georgia Tech）。**对用户最重要的三个结论**：① **§6 CPT vs from-scratch 配比形状**：连续预训练下"少数域拿走大部分权重"（C8/C9/C18/C19 主导）、from-scratch 需要更均衡——**直接支持 r=0.7 STEM 重仓的中训练设定**；同时警告 **ClimbMix 400B 内部构成是 from-scratch 优化的（偏均衡），不能直接当中训练配比抄**（池子≠配方）；② **实验设定与用户同构**：62M/350M/1B 基座先 10T tokens 预训练（WSD 调度、decay 段做配比研究）再 40B tokens CPT 搜配比——用户"ClimbLab 基座+1.8B tokens 中训练"= 同构结构预算 1/22，但其配比结论直接迁移有 #124 重复率失配风险；③ **消融数字**：112-run 迭代搜索（4:2:1 分配最优）超 DoReMi/RegMix（1B/40B：60.41 vs 59.37/59.16；350M：54.83 vs 53.78/53.38）；proxy 尺寸收益温和（62M→350M 仅 +0.3——**用户 400M proxy 与主设定同区**）；搜索算力翻倍仍 +0.7。**坑**：预测器位置性（Marin 点名：重聚类即重触发 112-run 扫描，微簇基直方图 h=V·w 修复）；优化目标为 PIQA/ARC_E/HS val 非 STEM accuracy；112 搜索≈5040 GPUh 对 N=1-2 不可及——正确姿势=消费 ClimbMix+自行小规模 r 扫描。已同步：总表 #74 悬空引用注记、README 目录树（123→127）。T1 [高价值] 维持。总计 127 篇论文（标记 38 篇）
- **2026-10-13（README 核心洞察节刷新 + 悬空引用审计清理 + 新录论文规模/机构补核）**：**① 核心洞察节 23→26 小节**（该节已落后近月结论，本次补齐）：新增 §24 重复率失配（外推失败第四解释、唯一可机械修复；三件套 #125 定理/#123 实证几何/#126 AI token 变号；用户 16× 失配数字 1.4 vs 0.09 epoch）、§25 CPT vs from-scratch 配比形状（CLIMB §6：CPT 尖锐/from-scratch 均衡——r=0.7 直接支持+ClimbMix 不能直接抄为中训练配比）、§26 验证目标选择>配比方法选择（MDE ρ=0.844 vs 0.282）；§22 同步更新（"83 篇"→127 篇、失败机制表补第四解释、加 #124 收窄注记——"根本无法解决"收窄为容量/相变部分，预算轴可修复）。**② 悬空引用审计清理**：git 历史确认总表 6 个 `*_深度分析.md` 引用从未建立——#77 Phi-4 改指 `一线大厂annealing阶段数据配比分析.md`（§2 专节）；#75 OLMo 2/#76 Llama 3/#78 DCLM/#80 SmolLM2/#83 MobileLLM-Pro 移除引用（内容已由总表行内 takeaway 覆盖）；统计：引用文档行 78→73、仅索引 49→54，悬空引用清零（46/46 全存在）。**③ 新录论文补核**（arXiv HTML 全文+PDF 首页+S2/OpenAlex 交叉）：#126 尺寸 19.9M-973M（726 拟合 19.9M-268M+74 验证 477M/973M=1.8×/3.6× 外推；**nanochat recipe+Muon 与用户栈同 recipe**，但优化器不同）、机构块 UMD/Pangram Labs（Wieting=UChicago、Iyyer=UMass Amherst）；#127 LLaMA-1B/100B tokens from scratch（一作 Lee）；#123 Chai=FloatAI+ETH Zurich、Xiong=独立研究者（PDF 首页核实——机构层面弱已记入行）；#128 华沙理工大学（Klimaszewski 亦挂 IDEAS Research Institute）。**总表 ⚠️ 待核/未核/未标 全部清零**。**④ README 梯队表补齐（审计新发现）**：逐 ID 对比总表发现两类历史缺口（均为 2026-09-14 批量索引+changelog 录入时漏加表行）——T4 表补 #105（MoE 重复过拟合边界点）/#108（Edu-QuRating 多维教育评分）/#109（Data Scout 定向爬取），25→28；T3 表补 #103（Cohere 多语言推理迁移）/#104（Climate-ModernBERT merge-vs-mix 对照）/#106（Curriculum as Transport，**[有意思]**）/#107（scaling law BO 采集顺序），42→46。**四梯队表现为总表完整镜像**（15/38/46/28=127，含 T4 表 #117 历史乱序修复——原排在 #114 前，移至 #116 后）。总计 127 篇论文（标记 38 篇）
- **2026-10-13（T1 前向引用扫描：13 篇高价值论文的引文全扫 + 准则评估后录 7 篇）**：对 T1 全部 13 篇（#01/#03/#04/#09/#10/#21/#27/#31/#37/#38/#74+#124/#126）做 S2 batch 前向引用扫描（CLIMB 66 引/Apple #03 54/PreSelect 24/#01 16 等；OpenAlex 对 2026 arXiv 引用边≈0 不可用），筛出 ~20 篇未裁决候选；按用户准则（解析前先评估：①实验规模 ②作者机构 ③业界关注度）逐篇评估后经用户批准录入 **7 篇**：① **#130 MergeMix**（T2 **[高价值]**，2601.17858，人大高瓴+蚂蚁）——merge-as-proxy：每域 5B token 专家（试验预算 2.5%）在合并权重空间搜配比，8B/16B MoE 工业级 mid-training（沿 #74 四域）超已上线生产的人工基线、成本降 100×、ρ>0.9，**8B 配比直接迁移 16B 仍超人工**（proxy→target 跨尺度正面证据，与 #73 反例对照）→ 新增核心洞察 §27；② **#131 Optimal Splitting**（T2，2603.19149，Apple **#03 原班**）——scaling law 定通用预训练→专门域 CPT 的最优切换点 t_S，CPT from scratch 显著差（→§25 补强）；③ **#132 SuperValid**（T2，2605.28179，蚂蚁）——capability 级 OOD 验证集构造配方，2T tokens 受控：IID loss 随配比漂移而 SuperValid loss 平稳（→§26 工具化）；④ **#133 Shift is Good**（T3，2510.25108，UChicago+清华+TTIC Srebro/Lyu）——**定理级：训练配比≠测试配比几乎总是更优**，r=0.7 的理论执照（→新增 §28）；⑤ **#134 Validity Threats**（T3 **[有意思]**，2606.05029，Tübingen AI Center）——proxy/观测/单 run 的效度威胁画像，"跨尺度可预测性无法只靠 proxy 自证"；⑥ **#135 RGI**（T3，2609.33016，Yale+Sea AI Lab）——任意两 token val-loss 差跨优化器/中等架构近似不变、换数据流强破坏：**域间相对损失差=数据敏感、优化器不敏感信号**（Layer 4 反例线索，→§28）；⑦ **#136 Mix or Merge**（T4，2601.21115，RPI+IBM）——mix vs merge 尺度边界（7B merge 优/2B mix 优，#104 生成式补集）。**排除 1 篇撞车**：2605.31175（Annealing w/ Principled Sample Selection）=#62 DiReCT 已录（T2/ICML'26，标题缩写导致候选漏配——准则核查抓出）。**核心洞察 26→28 小节**（新增 §27 merge-as-proxy+跨尺度、§28 相对信号+错配理论；§25/§26 各补一行）。统计：总 127→**134**（T2 38→41、T3 46→49、T4 28→29），标记 38→**40**（[高价值]+1、[有意思]+1），仅索引 54→61；快速索引"共 38 篇"→40、"全部 123 篇"→134（陈旧计数修正）。四梯队表已同步（完整镜像校验通过）。**用户裁定（同日）**：#131 三处"r=0.7=通用/专门分配内部解"表述收敛为"结构类比，非其结论"——论文专门化阶段喂纯域数据，未测 CPT 内通用/专门混合比例（总表行/README T2 行/§25 各一处）。**用户裁定（同日第二项）**：**#130 MergeMix 与 #132 SuperValid 升 T1**（用户提议+准则核对通过：MergeMix 8B/16B MoE 工业规模+12 配比真训保真验证 ρ>0.9；SuperValid 2T tokens 受控+16 开源模型观测含 Qwen2.5-1.5B+蚂蚁工业组——先例 GLM-5/Nemotron/MAI 零评审 TR 凭规模+机构进 T1）；SuperValid 同时加 [高价值] 标记；⚠️ 保留（无 venue/MoE vs dense/观测相关性/2T 实验模型尺寸未明）。统计：T1 15→17、T2 41→39、标记 40→41（[高价值] 21→22）、快速索引 40→41。**用户裁定（同日第三项）**：#130/#132 当日深读完成并建深度分析文档——`MergeMix_深度分析.md`（四步机制+Taylor 一阶等价/二阶交叉项理论+证据链四层+Mini-MergeMix 用户版草稿：ClimbLab 基座训 3 短专家×~100-150M token+40 种子点 merge+STEM 直评+LightGBM 学面，在 1.5B 本尊上获得 >40 配比信息量而回避 400M proxy 跨尺度外推）+ `SuperValid_深度分析.md`（四步管线+三层证据+CAP v2 草稿：GSM8K/MATH→知识因子→ClimbLab 检索→改写 2000-5000 样本，回溯检查零成本验证管线是否跑偏）；引用文档行 73→75、仅索引 61→59。**同日讨论补充入库**：① 用户提出的**权重合并盆地风险**（"合并后无训练→推理链断裂"：各自从零训的模型平均=乱码搀混；本文靠同基座+短步长留在同一线性连通盆地——专家必须短的另一层理由）与**短专家看不到相变**（math 是相变嫌疑最大域，#68 同款轨迹外推盲区）记入 MergeMix 风险节；② 证据链逐层挑刺（12 点全位次 ρ 不保证头部排序/人工基线为上一旗舰调优混淆/跨尺度 n=1/受控层选友好域对）+40 种子点=回归训练数据说明（与 RegMix 同构，数据点生产方式从 proxy 训练换成 merge+推理）记入对应节；③ SuperValid"不跳的机制"（OOD-by-construction=尺子钉在能力空间）记入证据链节；④ **落地污染警告**：检索池=训练池（ClimbLab）时验证文本可被字面背下（免疫分布距离≠免疫字面污染）——须剔除命中源文档或用 held-out 切片检索，记入 CAP v2 草稿
- **2026-10-14（仓库可见度工程：全量同步+更名 mixture+topics+英文门面+术语考证）**：① **全量同步推送**：公开仓库原停在 67 篇时代（8c82d9d），本地 134 篇+12 篇深度分析文档全部未推——commit 645a1b3 一次追平（16 文件，+2533/-122）；`.opencode/` agent 工作区加入 .gitignore（敏感扫描干净）；② **仓库更名** `data-mix-insights` → **`data-mixture-insights`**（GitHub API PATCH；旧地址 301 自动重定向，star/fork/issues 保留；本地 remote 已更新）；③ **topics 上线**（10 个）：data-mixture/data-mixing/pretraining-data/llm-pretraining/data-curation/mid-training/scaling-laws/data-selection/paper-notes/llm；④ **README 英文门面**：顶部加 English TL;DR（tier-graded library/140 papers/T1 highlights/notes in Chinese）。⑤ **术语考证（mixture vs recipe，一手 TR 词频实测）**：Nemotron 3 Ultra（data mixture×16；recipe×24 全为量化/训练流程，且"open-sourcing the training recipes, **data**, and RL environments"中 recipe 与 data 为并列项）、Llama 3（data mix×34；recipe×22 全为 Training/SFT Recipe=流程）、Qwen3（data mixture×4+mixture×9+data mix×4；recipe×0）、DeepSeek-V4（mixture×9/composition×2；recipe×0）、GLM-5（mixture×2；recipe×3 全为 Muon/DSA 超参）、MAI-Thinking-1（datamix）、Marin（mixture pipelines/hero-mixture-log.md；recipe=scaling recipe/超参类）——**结论：头部 TR 清一色用 mixture 指数据权重；recipe=训练流程（超参/量化/优化器/SFT 流程）；"data recipe"是方法论文/开源复现层词汇（DataChef/Puro/UltraData）**。库内口径据此定名：名称对齐 mixture（更名动机），收录范围=数据配方五轴（配比/选样/顺序/重复/调度耦合）——mixture 权重轴为主+相邻四轴。⚠️ Phi-4"data recipe"说法撤回（本环境 2412.04905 非该文无法验证，S2 限流）
- **2026-10-14（C/D 组 8 篇裁决：录 6 不录 2，134→140）**：前向引用扫描 C/D 组候选逐篇评估（摘要+全文+机构+引用核实）后按数据配方口径裁决。**录入 6 篇**：① **#137 LR Decay**（T2 **[高价值]**，2511.18903，清华 PACMAN **#84/#125 同组**）——升序质量课程×衰减 LR 冲突的受控机制：恒定 LR 下课程显著赢、标准衰减下优势消失（好数据压轴时 LR 已太低、模型"接不住"）；修复=温和衰减/checkpoint 平均替代衰减，1.5B/30B tokens、多种质量度量 +1.64%；**三方独立收敛**（#84 Puro CMA 同组产品化+#130 MergeMix WSM 独立组）："用权重平均替代 LR 衰减"是 2026 年课程/merge 时代重复出现的配方组件；**即 2026-07-22 移除的旧 #28**（当时口径"训练调度非数据配比"），按数据配方五轴口径（顺序×调度耦合）重录——git 旧档 +2.7% vs 现 v3 +1.64%，数字随版本漂移例证；② **#138 CD Scaling**（T2，2607.25271，Harvard+MIT CSAIL）——η token 有效性函数统一重复/改写定价（14M-600M/Dolma-3 拟合、随语料扩张饱和）、compute/data/model-bound 三区制、**经典 compute-optimal 在 data-bound 区次优**（用户 TPP≈1200 同区的理论根据）；③ **#139 Kinetics**（T3，2607.27281，**独立研究者单作者——当假说源不当证据源**）——驱动成核速率定律：浓度门槛点燃 10/10 vs 0/12（#01 相变的动力学重述）、**val loss 平滑而能力永不点燃**（监控盲区=门3 只能测的机制论证）、precursor 预测点燃时间 5% 中位误差（若复现="相变前信号预测相变后"第一个正面线索，与 §4.1 信息论论证张力最大处）；已录核心问题文档 **§4.5**；④ **#140 Chinchilla A2**（T3，2603.22339，Generation Lab+Columbia+UT Austin+Amazon）——isoFLOP 抛物线拟合系统偏差三源（网格宽度/采样不对中/loss 面不对称）、Llama 3 公开数据上 $1.4M 浪费、Approach 3+VarPro 修复——本项目一切 scaling law 外推的引用层纪律；⑤ **#141 HES**（T3，2605.22389，USTC 冯福利组+阿里+NUS）——top 0.5% 高熵 token 熵和免训练质量信号：SFT top 20%≈全量（31.34 vs 31.83，STEM 反超）、最低 20% 崩到 11.75、**0.6B 打分≈8B 自选**（免训练信号跨尺度稳定，#135 同向）、RL 段 1.5B 模型非对称采样超全批且半量数据——质量-效用张力线正面证据（熵轴>风格轴）；**录入时机构纠错：全文 affiliation 块为 USTC/阿里/NUS，非此前评估的 NTU**；⑥ **#142 WARP**（T4，2607.01686，**UW-Madison**，ICML'26 Workshop on Weight-Space Symmetries，有代码）——merge 插值伪轨迹逆向恢复微调配比（Mimic Score 梯度对齐按域池化→softmax/MLP 投影），BERT/GPT-2 MAE 0.046/0.104、**伪轨迹>真轨迹**；merge 家族分工补全（#130 正向搜索/#04 替代训练/本篇逆向取证）；**录入时机构纠错：UW-Madison 非 CMU**。**不录 2 篇**：LOO gradient matching（2610.00436，川大+电子科大）——微调在线批选择，数据配方口径外（原 T3 评估撤回）；BrowseComp-Plus→ClimbMix（2608.20317，Waterloo Jimmy Lin 组）——agentic 搜索评测语料构建，ClimbMix 仅语料巧合（57/830 存活）。**统计**：总 134→**140**（T2 39→41、T3 49→52、T4 29→30），标记 41→**42**（[高价值] 22→23），仅索引 59→65；快速索引 41→42（+#137）；#84 Puro 行加 #137 互链；相变文档新增 §4.5；总表注释更新（#28 重录说明、140 篇编号 1-142 缺 28/34）。总计 140 篇论文（标记 42 篇）
- **2026-08-27**: 用户反馈 SmolLM2 (2025.02) 已偏旧（18 个月），深搜 2025 下半年-2026 年最新报告。排查 20+ 候选（Gemma 3/Grok 4/GPT-5/Claude 4/Llama 4/ERNIE 5.0/InternLM 3/Baichuan 3/Yi-2/Hunyuan-2 等均无配比方法论或无 arXiv 报告）。新增 2 篇：**#82 MobileLLM-R1** (T2 [高价值], Meta AI, ICLR'26, arXiv:2509.24945, 2025.09)：**与 MAI-Thinking-1 §2.5 可比的最新报告**——§2 配比 3 页+附录，cross-capability self-influence 配比方法（AutoMixer 框架）、LOO ablation、Pareto frontier 分析、2 阶段配比表。核心发现：2T tokens 足够推理涌现（Qwen3 36T 的 11.7%）；FineWeb-Edu 最重要跨域数据源；**StarCoder 对数学收益 > OpenWebMath 对代码**（反转常识）；mid-training influence score 收敛=数据耗尽信号。**#83 MobileLLM-Pro** (T2, Meta Reality Labs, arXiv:2511.06719, 2025.11)：offline data-mix simulation + LOO continued tuning + 配比表。同时 **SmolLM2 降级**：[高价值] → 无标记（已偏旧，参考价值有限）。总计 83 篇论文

## License

MIT
