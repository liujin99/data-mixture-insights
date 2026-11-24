# Data Mixture Insights

LLM 预训练数据配比（data mixture）领域的研究与工业实践洞察库。

> **English**: A tier-graded research-insight library on LLM pretraining **data mixture** — 171 papers tracked and adjudicated (T1–T4) under a published credibility rubric (experiment scale / institution / community traction / open-source auditability + adoption / recency ≤2y), plus an **"industrial playbook" core reference set**: mixture methods actually used by industrial-scale models, evidence-graded across three tiers (TR disclosure / open-asset auditability / multi-vendor convergence) — Nemotron 3 Ultra, Nemotron 3.5+Singhal, Nemotron-CLIMB, MAI-Thinking-1, Marin 535B, Beam. Tiering is anchored to a small-active-MoE mid-training use case with a STEM-accuracy focus. Headline T1 findings: scale-induced rank non-invariance (MAI-Thinking-1), capability-aligned OOD validation (SuperValid), open-auditable mixture engineering (Marin / Nemotron-CLIMB), phase transitions in data mixing. Notes are in Chinese; the tier tables are language-light.

> **免责声明**：本库所有分析、评估和结论均为个人主观研究判断，针对特定场景（3B-A1 MoE 中训练、STEM accuracy 评估）——不代表任何官方立场，也不构成对任何论文或方法的权威评价。梯队划分与价值标记仅作研究参考。

## 项目目标

追踪、评估与归档 LLM 预训练数据配比（data mixture）领域的研究进展与工业实践，为配比方法的研究与工程实现提供参考。三条主线：

- **论文索引**：全部收录论文按可信度准则分四梯队（T1–T4），每篇一句话核心结论；
- **工业实战精选**：生产级模型真实在用的配比方法参考集（席位+证据分档）；
- **核心洞察**：跨论文综合的领域级结论与未解决问题。

**范围**：本库为文献洞察库——只做追踪/分析/记录，不做训练实验、不产出配比方案或训练交付物。

## 核心问题

1. **Proxy 可靠性**：小模型 proxy 的信号是否可靠？如何修复？
2. **Scaling Law 外推**：小模型拟合的 scaling law 能否外推到大模型？
3. **相变现象**：数据配比中的相变如何影响训练效果？
4. **质量信号**：如何定义和量化数据质量？

## 目录结构

```
data-mixture-insights/
├── papers/          # 论文 PDF (35 篇, 本地存储不入库)
├── docs/            # 分析文档 65 个：论文总表.md（全 171 篇主表）+ 可信度评估准则.md + 工业实战操作手册.md + 更新日志.md + 关键Takeaway.md + 深度分析/专题分析文档
└── README.md
```

## 可信度评估准则

对论文/报告的评估遵循证据强度分层：**定理 > 受控机制/反例 > 方向性结论 > 具体数字**——"X 有用"式正结论最易随规模蒸发，反例与算术账最耐用。在此之上核对五项准则（第 1–4 条裁"可信"〔准入级〕，第 5 条裁"价值"〔时效折损〕）；**梯队 = 可信 × 价值 × 场景**：

1. **实验规模**：小规模模型/数据上的结论不必然适用于大规模。与工业主流规模（>100B 参数 / 数十 T tokens）相差 2 个数量级以上默认标 ⚠️ 规模外推风险；有多尺寸扫描或工业规模佐证可放宽。场景锚点：3B-A1 MoE 中训练（3B-A1→7B→15B→500B 梯子首档验证档）。
2. **作者与机构**：作者背景（理论/统计/工程）与机构类型（头部厂商实验室 / 知名学术组 / 初创 / 单作者）作为可信度考量，行内标注。
3. **业界关注度**：被大厂 TR 引用、工业采用、社区讨论度是正信号；缺乏关注度的工作在引用其结论前须回溯一手证据。
4. **开源可审计 + 社区采用**：完全开源（代码 + 数据或权重，足以审计核心主张）且社区采用达标（GitHub star 千级量级 / 被第三方项目或工业管线实际采用 / 发布 ≥6 个月持续维护，满足其一）——认定为可信、高参考价值，可独立支撑 T1 准入。三条边界：背书工程可复现性不自动背书科学结论；不解除规模警示；无开源佐证的"工业规模"自报不得独立支撑准入。新发布（<6 个月）默认只计开源不计采用，达标后重评。
5. **时效**：以首次发布日期起算滚动 24 个月窗口，窗口内全额计值；出窗标 ⚠️时效，"具体数字 / 配比配方 / 管线实践"层默认不再直接引用。三条豁免：定理与受控机制/反例层不折损；开源资产以仓库现状独立评估；被后继修正的旧结论降为"历史锚点"。

> 完整版（含全部边界条款、豁免细则与增补记录）见 [`docs/可信度评估准则.md`](docs/可信度评估准则.md)。

## 论文索引

> 四梯队共 171 篇，每行**一句话核心**：讲清这个工作最核心的结论/指导，或为什么对本项目（3B-A1 MoE 中训练、数据配比）有高参考价值。详细内容、⚠️ 边界与梯队注记以 [`docs/论文总表.md`](docs/论文总表.md) 为准（完整主表：梯队+标记+机构+Venue+文档路径+关键贡献）。
>
> **梯队划分依据**《可信度评估准则》（实验规模 / 作者机构 / 业界关注度 / 开源可审计+社区采用 / 时效）；个别行的升降档及原因见 [`docs/更新日志.md`](docs/更新日志.md)。
>
> **标记说明**：`[高价值/重要]` = 项目理论根基；`[高价值]` = 直接可用方法/数据；`[新发现]` = 新视角启发；`[有意思]` = 有价值但需批判看待；`[独立再发现]` = 不同搜索角度独立验证高相关；`[高价值/轴外]` = 高可信+理论新颖但 T4 场景外——价值认可保留、不作直接实验参考（/轴外 可与其他标记组合，如 #114 [高价值/新发现/轴外]）；`[工业实战/…]` = 工业实战精选成员（判断准则见论文索引「工业实战精选」表、操作手册见 docs/工业实战操作手册.md；与 T1-T4 不同维度，分级不受影响）。约束：无后缀 `[高价值]` 仅限 T1/T2。

### 工业实战精选（6 席 7 篇，全 T1）

> **判断准则：工业级模型真正在使用的配比方法**——先不管创新性，主打生产采用证据；参照它们可能不是最优，但不会差到哪里去，远比"为发 paper 而做的研究"可靠（后者存在美化可能）。与 T1-T4 不同维度（**生产采用过滤** vs 可信度分层），不是"更高梯队"，入选者均为 T1 行（已计入四梯队统计，不另计）。**证据三档**：一档=TR 直接描述生产使用；二档=开源资产可审计（含第三方生产采用）；三档=≥2 家独立头部厂商披露同款实践。**十一环节操作手册、席位明细与边界记录见 [`docs/工业实战操作手册.md`](docs/工业实战操作手册.md)**。

| 席 | # | 论文 | 机构 | Venue | 一句话核心 |
|---|---|------|------|-------|-----------|
| ① | 60 | **Nemotron 3 Ultra** | NVIDIA | Tech Report (2026.6) | 数据配比最透明的大厂 TR：19 类两阶段比例全公开+定向注入价目表+RL blends 开源资产——**分栏唯一二档证据**，披露密度天花板 |
| ② | 160+161 | **Nemotron 3.5 Super/Lightning + Singhal Recipe** | NVIDIA | Tech Report (2026.10, 2610.38095) + arXiv 2610.22019 | 首个量化配方全公开的工业级 agentic 中训练：域数据 ~75%+**PT 回放 20%**（40% 无增益消融）+轨迹封顶 10-15%+SFT 桥评测 |
| ③ | 74 | **定比例=业界事实"手调"** + CLIMB | NVIDIA 等 | NeurIPS 2025 | **无一家头部披露用自动搜索定生产配比**——手调+消融+护栏+桥；CLIMB=唯一开源搜索参考（方法未进自家生产线） |
| ④ | 73 | **MAI-Thinking-1** | Microsoft AI | Tech Report (2026.08) | 千 proxy 实证 rank non-invariance（"proxy 排名不可信"最强工业证据）+memorization-aware 重复上限+完整配比表 |
| ⑤ | 110 | **Marin 535B** | Open Athena（Stanford 起源） | 开源工程项目 (2026.08-09) | 535B/18T 全程可审计：40 域×5 质量桶混合空间+三层选权协议+uncheatable held-out+SFT 回放 20.4% 生产实证 |
| ⑥ | 173 | **Beam** | Reflection AI | arXiv 2611.26358 (2026.11) | 工业级生产 TR（一档）：PT 分域轮次配比+质量分档因果消融+**MT 回放第二家**（10%/35%，剂量阶段依赖）+RL 任务策展；**注记**：分数/效率主张未经独立复现、数据未开源（档位一） |

**深读落点**：`头部厂商TechReport_数据配比深度分析.md` §2/§7 · `OpenMidtrainingRecipe_深度分析.md` · `NemotronCLIMB_深度分析.md` · `MAI_Thinking_1_深度分析.md` · `Marin535B_Harrier_深度分析.md` · `Beam_深度分析.md`

### 第一梯队：高可信度 + 高价值（准入：定理级 / 工业规模 / 强机构+强 venue / 完全开源+社区采用）

| # | 论文 | 机构 | Venue | 一句话核心 |
|---|------|------|-------|-----------|
| 01 | **Phase Transitions in Data Mixing** | 清华 | NeurIPS'25 Spotlight | **[高价值/重要]** 配比存在相变：STEM 知识获取在阈值处突变（r_thres∝M^(-α)），不是平滑曲线能外推的——本库"小 proxy 预测不了涌现"的理论根基 |
| 03 | Scaling Laws for Optimal Data Mixtures | Apple | arXiv (2025.07, 2507.09404) | 把域配比与 loss 的关系做成参数化 scaling law、可外推可跨模态——"用拟合曲线代替扫配比"路线的代表 |
| 04 | DeMix: Model Merging | 小红书+东大 | ICML'26 | 每域单独训专家再参数合并、替代联合配比训练：7 域×30B=211B tokens、1.7B/8B 双尺寸验证、数据集开源；数据集开源 ✓、社区采用未录 |
| 09 | CAMEL: Capacity-Aware Mixture Law | 清华 | arXiv | 把相变理论落成容量感知的配比律——Phase Transitions 的实践版（同系工作） |
| 10 | Proxy Reliability | Princeton/Berkeley/Stanford/VT | ICLR'26（arXiv 2512.24503，2025.12） | 找到 proxy 失效的修复旋钮：调低 proxy 学习率可让小模型信号与大模型对齐——400M proxy 线的超参纪律 |
| 21 | PiKE: Adaptive Data Mixing | Google Research+USC | arXiv | 大规模预训练中任务梯度几乎不冲突，按梯度范数/方差自适应配权即可（冲突缓解方法基本无效）——1.9× 收敛加速 |
| 27 | **Domain-Aware Scaling Laws** | MIT+Microsoft Research | arXiv (2026.7) | **[高价值]** 域间协同/竞争可量化（R²=0.91）且≠语义重叠——配比不能只看单域好坏，pairwise 特征可直接进 LightGBM |
| 31 | PreSelect | HKUST+Vivo AI | **ICML'25**（arXiv 2503.00808，2025.03） | 用 predictive strength（免训练统计量）+fastText 选数据，10× 计算缩减、开源——"不训 proxy 也能选"的轻量可靠线 |
| 37 | **InfoLaw** | ByteDance | **ICML'26** | **[高价值]** 信息论推导的 scaling law 同时建模配比权重与重复递减收益，7B/425B 验证误差 0.15%——配比+重复的统一框架 |
| 38 | **Explaining Data Mixing Scaling Laws** | 北理工+清华 IIIS | **ICML'26** | **[高价值]** 解释配比 scaling law 为何长这样（容量竞争+噪声消减两机制）——理论上背书"高质量该重仓"，也解释 proxy→target 为何失效 |
| 58 | **GLM-5** | 智谱 AI + 清华 | Tech Report (2026.2) | **[高价值]** 头部厂商唯一设专门 Mid-Training 章节的 TR（744B/28.5T）：PT 4K 27T→MT 三阶段窗口课程（32K/1T→128K/500B→200K/50B）+160B SWE——与 NVIDIA 三明治独立同构；**LR 连续衰减 4e-5→1e-5**（vs NVIDIA 重置恒定，连 #145）、无 PT 回放披露（活分歧）、DSA 20B 追平 MLA、OPCD、主动打榜（深读 `GLM5_深度分析.md`） |
| 60 | **Nemotron 3 Ultra** | NVIDIA | Tech Report (2026.6) | **[高价值]** 数据配比最透明的大厂 TR（550B/20T）：19 类别两阶段比例全公开+定向注入消融价目表（100B tokens 合成 Q&A→GPQA +11.1）——目标尺度的参考答案 |
| 160 | **Nemotron 3.5 Super and Lightning** | NVIDIA | Tech Report (2026.10, 2610.38095) | **[高价值]** 首个量化配方全公开的工业级 agentic 中训练：65.4B@256K（Issue-PR 41%+仓库 20%+agent 轨迹 10% loss-mask+**PT 回放 20%**+LC 5%）、LR 重置恒定、9B 恢复段、**SFT 桥评测**（主效应=跨 5 harness 全胜 +8.6）、Lightning 固定预算配比扫描 10/20/30% 单调——3B-A1 MT 设计直接引用 |
| 161 | **Priming Base Models for Agentic Post-Training: An Open Midtraining Recipe** | NVIDIA | arXiv (2026.10.15, 2610.22019) | **[高价值]** 业界第一份消融级 open midtraining recipe——三明治中间层完整配方（自然 SWE 数据 ~75%+轨迹 ≤10-15%+回放 20%）；**base 代理指标单调改善却"不足以指导配方决策"，评测靠 SFT 桥**（#113 工业证词）、工具响应 mask 差 10 分、LR 按尺寸分档；3B-A1 MT 设计头号直接参照（深读 `OpenMidtrainingRecipe_深度分析.md`；T1 准入=工业规模路线） |
| 73 | **MAI-Thinking-1** | Microsoft AI | Tech Report (2026.08) | **[高价值]** 千个 proxy 实证 rank non-invariance（小尺度赢家到大尺度反超）——"proxy 排名不可信"的最强工业证据；另有 memorization-aware 重复上限与完整配比表 |
| 74 | **Nemotron-CLIMB** | NVIDIA | **NeurIPS 2025**（arXiv 2504.13161，2025.04） | **[高价值]** 本库基座的数据来源：ClimbLab 1.2T 语义聚类 20 簇+迭代搜索出 ClimbMix 400B，优于 DoReMi/RegMix——"语义聚类+bandit 搜索"的工业实现；全开源+社区采用+本库一手复现（准则4 命中）——双支柱；⚠️出窗预警（2027-04，ClimbMix 资产+复现按时效豁免不折损） |
| 101 | **UltraData: Tiered Data Management** | OpenBMB/面壁智能 (Yudong Wang 等) | arXiv (2026.02, 2602.09003) | **[高价值]** L0-L4 分层数据管理：L3 格式重构（QA/教科书化）才是数学跃迁来源（+9.24pp MATH500）、纯筛选增益小——"筛"与"改"的价值分界；论文+数据+代码全开源+HF Datasets Trending #1+被 Puro-2B 采用为 P2 最大数学源（第三方工业采用，准则4 命中） |
| 110 | **Marin 535B Hero Run / Harrier 数据管线** | Open Athena（Stanford 起源） | 开源工程项目（2026.08-09） | **[高价值]** 535B/18T 全程可审计的开源训练项目：40 域×5 质量桶混合空间+三层选权协议+uncheatable held-out——配比方法学的开源对照实验室；全开源+~4k★ 社区采用（准则4 命中） |
| 132 | **SuperValid: Capability-Aligned OOD Validation** | 蚂蚁集团 | arXiv (2026.05, 2605.28179) | **[高价值]** IID 验证 loss 在配比变化时会漂、capability 级 OOD 验证集不漂（2T tokens 受控实锤）——扫配比期间用什么当监控信号的答案 |

### 第二梯队：高相关（未全达 T1 准入）

| # | 论文 | 机构 | Venue | 一句话核心 |
|---|------|------|-------|-----------|
| 11 | DataMan | 阿里 Qwen | ICLR'25 | 给数据打"能力说明书"（14 维质量/能力标注、447B tokens）——质量信号的工业化标注体系 |
| 12 | UtiliMax / MEDU | Meta FAIR | arXiv | token 计数当配比基线：200× 加速且不输学习式方法——"先试简单方法"的实证代表 |
| 13 | Repetition-Aware Scaling | Apple | arXiv | 高质量数据可重复 15-20× 而不掉点——重复轴的经典量化锚点 |
| 14 | Synthetic Data in Pretraining | Meta | EMNLP'25 | 合成:自然 ≈ 30:70 是最优区——合成掺入比例的受控结论 |
| 20 | Chameleon | EPFL | **ICML'25** | KRLS 领域权重、换目标时免重训迁移配比——实验最大仅 1.2B，按规模准则降级 |
| 22 | ADAPT: Online Reweighting | Cambridge+Oxford | **ICLR'26** | 在线按训练动态重加权优于离线一次性选子集——"边训边调"路线的受控证据 |
| 26 | **FastMix: Gradient Descent Mixing** | 腾讯混元+港大 | **ICLR'26** | **[高价值]** 可微 bilevel 优化把配比搜索降到 1 个 proxy（550× 加速）、§4.3 工业踩坑经验直接适用——预训练验证最大 1B，按规模准则降级；代码开源 ✓、社区采用未达（研究代码） |
| 30 | **WebGraphMix** | Princeton (Danqi Chen) | arXiv | 网页图中心度当 training-free 质量信号、与 fastText 正交（乘法组合 +1.5%）——零训练信号的正交补充 |
| 36 | **RegMix-D: Dynamic Mixing** | U Tokyo+NII | arXiv (2026.6) | **[高价值]** RegMix 动态化：用 loss 轨迹回归替代终点 loss，128 个 25% 预算 proxy 超 512 个全预算 RegMix——proxy 搜索的 ROI 改进路径 |
| 40 | **HDS: Holistic Data Scheduler** | Dang, Ma, Liao | **KDD'26** (S2 DBLP 确认) | 在线 RL（SAC）按三目标奖励调度数据，Pile 省 44% 迭代——调度轴的强化学习实现 |
| 41 | TANDEM: Twin Networks | Kuaishou (12 作者) | arXiv (2026.6) | twin networks 差值在线度量数据效能做 bilevel 配比（带理论保证）——工业产品线视角 |
| 42 | **Solar Open 2** | Upstage | Tech Report (2026.7) | 250B MoE 业界实践：quality+rarity 策展+mixture-ratio 优化、20T→10T 精炼——"配比优化有效"的厂商侧确认 |
| 61 | **Velocitune** | Microsoft | **ACL'25** | 按学习速度动态调域配比、绕开跨模型外推——但逐域 scaling law 稳定性不透明是隐患 |
| 62 | **DiReCT** | MSRA+HKUST(GZ) | **ICML'26** | annealing 阶段按 Hessian 谱几何选样本（flat 方向梯度大者）——冷却期数据选择的机制化方法 |
| 63 | **Data Mixing Agent** | Microsoft | **ACL'26** | **[高价值]** 用 RL 学配比策略、benchmark 分数当 reward（绕过 val_loss 不靠谱），+3.02% over RegMix——配比搜索的 agent 化 |
| 64 | **BETR: Benchmark-Targeted Ranking** | **Apple** | arXiv (2025.07) | **[高价值]** 按 benchmark 相似度做文档级选择：~2-4 GPUh 全库成本、2.1× CM——最便宜的可用选择器；"最小模型要最狠过滤"的 scaling 结论 |
| 65 | **Task-Agnostic Training Data Influence** | NAIST+NII 等 6 机构 | **COLM'26** | **[有意思]** task-agnostic 数据贡献度量——但有循环论证批判（θ* 内生性），对配比决策指导有限 |
| 66 | **DIDS** | HKUST+MetaX+Alibaba 等 | **EMNLP'25** | **[高价值]** FIM+loss 轨迹动态配比，直接对比赢 Velocitune 3 分、10% 数据超全量——验证规模小（1B tokens）降级，方向价值保留；代码开源 ✓、社区采用未达（1★/3 commits 原型级） |
| 67 | **MIRA** | SJTU+UBC 等 | arXiv (2026.05) | 中训练源感知 rubric 筛选：25B 半量匹配 50B 全量——样本级筛选与域级配比互补 |
| 68 | **LogFloor** | Stanford/KAIST | arXiv (2026.08) | **[高价值]** proxy 学到的"瓶颈顺序"可跨尺度迁移（70M→12B replay 省 30.9%）——但用 loss 绕过了 accuracy 涌现问题 |
| 71 | Scaling Domain Data Repetition | 清华+字节 Seed | arXiv (2026.08) | 域级最优重复次数与 val loss 强负相关、同 TPP 可迁移——数据充裕区（本库场景）帮助有限 |
| 72 | **Cross-Difficulty Optimization Dynamics** | 复旦 | arXiv (2026.08) | 跨难度知识迁移度量+动态采样——课程学习的优化动力学解释（post-training 场景） |
| 75 | **OLMo 2** | Allen AI (Team OLMo) | arXiv (2024.12) | 数据+recipe+中间 checkpoint 全开源+annealing curriculum+checkpoint soups——可复算的中训练参照系；完全开源（准则4 命中）；⚠️出窗预警（2024.12）——活内容已被 #73/#84/#137 继承 |
| 77 | Phi-4 Technical Report | Microsoft | arXiv (2024.12) | 合成数据配比当核心创新（"beyond distillation"）——合成为主路线的代表 TR；⚠️出窗预警（2024.12；合成 30:70 属数字层）；仅权重部分开源不计准则4 |
| 80 | SmolLM2: When Smol Goes Big | Hugging Face (Ben Allal et al.) | arXiv (2025.02) | 4 阶段 evolving mixture 的小模型开源参照——已 18 个月偏旧；⚠️出窗预警（2025.02） |
| 82 | **MobileLLM-R1** | Meta AI | **ICLR'26** (arXiv 2509.24945, 2025.09) | **[高价值]** 2T tokens 足够推理涌现（Qwen3 36T 的 11.7%）+cross-capability influence 配比法+两阶段配比表——中训练配比工业细节最全的论文之一 |
| 83 | **MobileLLM-Pro** | Meta Reality Labs | arXiv (2025.11, 2511.06719) | offline 数据混合仿真+LOO continued tuning+配比表——静态仿真不如 R1 的迭代式 |
| 84 | **Puro-2B** | 清华 PACMAN | arXiv (2026.08, 2608.27370) | **[高价值]** 消费卡复现 2B/1.4T（$6.9K）全开源：课程/CMA 数据配方贡献 2.40×（最大单项）、配比收益穿透 SFT（+1.77~2.02pp）——与本库场景同构的端到端参照；注记：2027 复现第二席（T2 不变）；Apache 2.0 全开源 ✓、发布未满 6 个月暂不计社区采用 |
| 88 | **Capacity-Dependent Data Selection for Reasoning** | Virginia Tech (Ruoxi Jia 组) | **COLM 2026** (arXiv 2026.08, 2608.13721) | **[高价值]** 数据难度是（数据×模型）配对属性：1.5B 学"够不着"的数据只会死记、8B 才反转——STEM 池别灌超难、打分用学生模型自己 |
| 111 | **AutoData: Agentic Search for Pre-training Data Selection** | Weco AI+UvA | arXiv (2026.09.17, 2609.19754，在投) | agent 在可执行算法空间搜选择器（底座与本项目栈同构）；核心发现"精炼池质量轴已死"（最优 vs 随机仅 ~0.1%）——Climb 级语料只有内容构成还有残值；代码开源（MIT）✓、社区采用未达（8★） |
| 115 | **When do data mixtures improve scaling laws?** | ISTA+Yale | arXiv (2026.09, 2609.38011) | 理论：混合优于单数据集 ⟺ 辅助域谱重尾（互补覆盖）且规模适中增长——源选择=选互补覆盖而非堆冗余质量 |
| 122 | **MDE: Optimizing Pre-training Data Mixtures with Mixtures of Data Expert Models** | Google DeepMind | arXiv (2025.02, 2502.15950, preprint) | **[高价值]** 验证目标选择>方法选择：end-task 验证 loss 与下游相关 0.844 vs 域 loss 0.282——换验证目标的收益远大于换方法；⚠️出窗预警（2025.02；ρ=0.844 属测量型机制证据，出窗时重审而非自动降档） |
| 123 | **What Is a Repeated Token Worth?** | FloatAI+ETH Zurich+独立研究者 | arXiv (2026-10, 2610.05591, 无 venue) | 重复代价 ∝ (R-1)N/U 单变量定律：固定语料下更大模型更快到最优 epoch——过训练区的重复理论 |
| 124 | **Repetition Mismatch: Why Data Mixture Experiments Don't Scale** | Imperial College London | **EMNLP 2026 Main**（arXiv 2606.07597） | **[新发现]** 小实验外推失败的元凶=重复率失配：子采样匹配目标重复率后 1/16 tokens 恢复最优配比（误差 0.85→0.10）——proxy 实验设计的必修修复 |
| 125 | **Larger Datasets Can Be Repeated More** | 清华 PACMAN（#84 同组） | **ICLR 2026**（arXiv 2511.13421） | 定理级：大数据集更耐重复（有效复用率饱和值随 N 增长）——修正"≤4 epoch 无差"旧共识 |
| 126 | **How Much Is an AI Token Worth?** | UMD+Pangram Labs 等 | arXiv (2026.09, 2609.40295) | wild AI 文本掺入会变号：数据饥饿模型先受益后受害、高预算模型立即有害——"先重复 human 再扩 AI 语料"；WildAI 语料/800 模型/代码全量开源 ✓（发布未满 6 个月暂不计社区采用） |
| 130 | **MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging** | 人大高瓴+蚂蚁集团 | arXiv (2026.01, 2601.17858，ICML 模板) | **[高价值]** 把模型合并权重当配比搜索的低耗代理（每域 5B token 专家=预算 2.5%）：8B/16B 工业 mid-training 超人工调参、成本降 100×，8B 配比直接用于 16B 仍有效——proxy 跨尺度少数正面证据；降 T2（反向条款：无代码/数据/权重、规模自报不可审计、v1 挂 9 个月无修订）——放出代码/录用/第三方采用即重评 |
| 131 | **Optimal Splitting of Language Models from Mixtures to Specialized Domains** | Apple（#03 原班） | arXiv (2026.03, 2603.19149) | scaling law 定"通用预训练→专门域 CPT"的最优切换点：早切≈全量、CPT from scratch 显著差——两阶段结构的 Apple 侧证据 |
| 137 | **How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining** | 清华 PACMAN（#84/#125 同组） | arXiv 2511.18903（2025.11，v3 2026.05，无 venue） | **[高价值]** 升序质量课程必须配恒定 LR：标准衰减把课程收益吃掉（好数据压轴时 LR 已太低）——修复=温和衰减或 checkpoint 平均 |
| 138 | **Bridging Compute- and Data-Optimal Pretraining** | Harvard+MIT CSAIL | arXiv 2607.25271（2026.07，无 venue） | token 有效性函数 η 给"重复/改写值多少新鲜 token"定价：池≫预算时重复只在局部簇发生——优化精力该花在选择轴 |
| 143 | **TiKMiX: Take Data Influence into Dynamic Mixture for Language Model Pre-training** | 字节（QuaDMix 同团队） | arXiv 2508.17677（2025.08，v1，无 venue） | **[高价值]** 免 proxy 在目标模型上 0.1B tokens 直测 Group Influence 定配比（比 loss 观察 50× 便宜）——QuaDMix 团队自己转向动态路线的标志 |
| 145 | **Data Schedules and Learning Rate Schedules Should Be Designed Together** | 清华 PACMAN+鹏城实验室 | arXiv (2026.10, 2610.29348) | **[高价值]** PT→MT 数据调度×LR 调度必须联设：衰减 LR 下早混入 MT 数据更优、恒定 LR 下反转；LR 衰减应在 MT 进场后开始；MoE 渐进入场避免路由失衡尖峰——#137 同组续作，3B 级 MoE 中训练结构直接参照 |
| 155 | **Olmix: Data Mixing Throughout LM Development** | Allen AI | **ICML 2026**（arXiv 2602.12237，2026.02） | 配比方法设计空间系统实证+域集演化时的 mixture reuse（5 次更新 74% 省算力、+11.6% 下游）——配比结论可继承性问题（AI2 班底框架级 benchmark） |
| 156 | **The Finetuner's Fallacy** | ISTA+CMU+Meta 等 34 人 | arXiv (2026.03, 2603.16177) | 专门域数据从预训练起掺入（SPT）优于留给微调：1.75× token 效率、1B SPT>3B 标准、过拟合 scaling law 定最优重复——“域数据尽早进场”受控证据（调度三连之一） |
| 157 | **Replaying Pre-training Data Improves Fine-tuning** | Stanford | arXiv (2026.03, 2603.04964) | 微调/中训练回放通用数据反而提升目标域（1.87×/2.06× 数据效率，8B 实证）——PT replay 线来源（#145 给出其 LR 边界），30% 非 STEM 陪跑收益的量化参照 |
| 158 | **Nemotron-CC-Math** | NVIDIA | **ICLR 2026**（arXiv 2508.15096，2025.08） | 133B/52B（4+）开源数学语料：超 FineMath-4+ 且体量 5.5×、8B 实证 MATH +4.8~12.6——STEM 池数据量瓶颈的现成扩充资产（代码+数据开源） |
| 165 | **daVinci-Prover: Agent-Native Mid-Training** | GAIR（上交/复旦系，Pengfei Liu 通讯） | arXiv (2026.11, 2611.16433) | **[高价值]** 定理证明的透明 MT 配方：33.7B token 配比表全公开（分解 25%/轨迹 21.3%/自验 16.9%/回放 15.2%/修正 11.4%/搜索 10.2%）；**消融：8.4B agent 轨迹>25.3B 技能先验**；+13.7 系旧 base 数字（新 base 收敛 +1~3）；前 15B 剂量甜点；成本 $15 万级——尺度最接近 3B-A1 的域专项 MT 素材（直接参考权=工业实战精选独占，本文仅作分析素材） |

### 第三梯队：方向参考（综述 / 假说 / 场景错位 / 待验证）

| # | 论文 | 机构 | Venue | 一句话核心 |
|---|------|------|-------|-----------|
| 02 | Mixture Scaling Under Constraints | Apple | arXiv | 数据受限（预算约束）场景下的配比 scaling law |
| 05 | Data Mixing Laws | 复旦 | ICLR'25 | 嵌套 scaling law 两层拟合（域内+域间）建模配比——配比 scaling law 路线的奠基工作 |
| 06 | Multi-fidelity BO | Columbia | arXiv | 大小实验混用的贝叶斯优化搜配比——省算力的搜索框架 |
| 07 | Beyond Language Modeling | Meta FAIR | arXiv | 多模态预训练的配比——配比问题的轴外扩展参照 |
| 08 | CMR Scaling Law | - | EMNLP'24 | 跨模态 retrieval 的 scaling law（跨模态配比参照） |
| 15 | OP-Mix | NYU+DeepMind | arXiv | LoRA 插值当配比 proxy——轻量代理的早期尝试 |
| 16 | Data Organization | Microsoft | ACL'26 | 数据顺序/组织优化——顺序轴的早期系统工作 |
| 17 | LLMSurgeon | MBZUAI+UCL | ACL'26 | 从模型生成行为反推预训练配比——逆向配比 |
| 18 | DataChef | 复旦+上海AI Lab | arXiv | RL 自动生成 adaptation 配方——配方自动化的早期 agentic 尝试 |
| 19 | Group-MATES | CMU+Meta | arXiv | 组级数据选择、显式建模数据交互效应——"组级选择 vs 个体选择差 2×"的量化来源 |
| 23 | NAG: Neuron-Activated Graph | ByteDance+UCSC+UNC | arXiv | 神经元激活图做 target-oriented 数据选择、training-free——零训练选择信号 |
| 24 | CausalMix: Causal Inference | 清华+蚂蚁集团 | arXiv (2026.7) | 因果推断框架做 SFT 配比（DML 正交化+CATE）——因果视角 |
| 25 | HERMES: Multi-Granularity Labeling | Wizard Quant+北大+中科大 | arXiv (2026.7) | 层级 RVQ 标签基底+粒度×质量过滤交互——标签粒度轴 |
| 29 | MixAtlas: GP Surrogate | Apple+UW+VTech | arXiv | GP surrogate+GP-UCB 主动采样搜配比，同预算比 LightGBM 高 4-7 点、0.5B→7B 迁移——代理模型选择 |
| 32 | GRASP: Subset Interaction | Wizard Quant+USTC | arXiv | 子集交互感知的数据归因（二次几何惩罚），rank correlation 翻倍 |
| 33 | DoGraph: Gradient Dynamics | HKUST-GZ+HIT+OpenCSG | arXiv | 梯度动态重定义域（人类域≠模型域）+图约束重加权——域的定义本身可学 |
| 35 | Survey: Data Mixture for LLMs | CUHK-SZ | arXiv (2025.5) | 19 种方法分类的综述；"最优权重仅略优于 uniform"这一争议结论的出处 |
| 39 | **DomainPilot** | 单作者 (He Zhang) | arXiv (2026.7) | **[独立再发现]** 域级 loss 引导两阶段优化、~30 行 patch 工业部署——单作者无 venue，工程思路高参考、结论需验证 |
| 43 | DataOrchestra | SJTU | arXiv (2026.7) | 样本级策展编排（drop/clean/untouched 决策），0.5B-7B from scratch |
| 44 | DecoupleMix | Meituan | arXiv (2026.7) | VLM 配比解耦：类间搜索+类内凸优化，proxy 比例可迁移到更大尺度 |
| 45 | CuraWeb | Meituan | arXiv (2026.7) | 质量/冗余/多样性联合优化，2T 语料 3B 规模 +1.8% |
| 46 | Reasoning Quality Emerges Early | UCLA | **ICML'26** (Poster) | 前 100 token loss+扰动 checkpoint 即可检测推理数据难度，91% 更高效——早期信号 |
| 47 | Online Selection Is Implicit Alignment | 3 作者 | arXiv (2026.7) | 在线数据选择会隐式改变模型行为偏好——选择不是中性操作 |
| 48 | CoDA: Co-Adaptive LoRA | 4 作者 | arXiv (2026.7) | 无标签域参与度控制（competence+affinity 信号），半数据量超 uniform |
| 49 | DataFlex | 北大 | arXiv (2026.3) | 统一数据中心训练框架（selection+mixture+reweighting） |
| 50 | Survey #2: Data Mixing | Tianjin U | Data Intelligence (2026, 期刊同行评审) | 期刊综述：bilevel on simplex 形式化+inverse data mixing 方向 |
| 69 | Bayesian Domain Reweighting | 西安交大 (Zongben Xu) | arXiv (2026.07) | Dirichlet+Gamma 先验直接推断域权重——贝叶斯路线，批评 scaling law 结构假设 |
| 76 | **Llama 3 Herd of Models** | Meta | arXiv (2024.07) | 15T tokens 预训练数据工程的最广泛引用参考：annealing 高质量收尾+质量分类器管线；降 T3（已出窗：数字层〔15T 配比/质量分类器实践〕过期、机制层〔高质量收尾 annealing〕已被在窗 TR 覆盖）——“最广泛引用”历史锚点 |
| 78 | DataComp-LM (DCLM) | 多机构 (Li, Fang et al.) | arXiv (2024.06) | 240T 语料受控实验框架（412M-7B）——数据管线比较的标准基建 |
| 79 | Data Mixing Survey #3 | CUHK-SZ | arXiv (2026.03) | 41 页综述：Static vs Dynamic 分类法，覆盖 DoReMi/RegMix/BiMix/Chameleon/Domain2Vec 等 |
| 81 | Tulu 3 | Allen AI + UW (Lambert et al.) | arXiv (2024.11) | post-training 配比方法论：skill 上界模型+配比逼近；8B 与 70B 最优配比不同 |
| 85 | D³-MOPD | 苏州大学 | arXiv (2026.08, 2608.24987) | **[有意思]** 按域收敛信号（reverse-KL）零开销动态调度域配比——轻量动态法，短预算友好 |
| 89 | GEM: Geometric Entropy Mixing | Yue Min et al. | **ICML 2026 Poster** (arXiv 2026.05, 2605.26121) | 数据策展重构为超球面变分问题+混合平衡正则——几何视角的配比 |
| 90 | Always Learning, Always Mixing | NYU+MIT | arXiv (2026.05, 2605.15220) | pretraining/continual/adaptation 全生命周期统一混料框架——批评单阶段割裂 |
| 91 | Mix, Don't Tune | ENS Paris/INRIA+DTU | arXiv (2026.05, 2605.13225) | 数据受限（重复训练）场景：混入高资源辅助数据 > 调超参 |
| 94 | KD During Mid-Training Favors Reasoning over Factual Recall | Meta+Princeton+CMU | arXiv (2026.09, 2609.01532) | **[新发现]** forward KL 蒸馏在 mid-training 拖慢 factual recall（pre-training 阶段则双赢）——"数据价值依赖训练阶段"的机制证据 |
| 95 | REER-PT | StepFun | arXiv (2026.08, 2608.30627) | **[有意思]** PPL 难数据不丢弃而是修复（插入推理注释，680M +2.07pp）——PPL 过滤线的互补方案 |
| 96 | PLES: Power-Law Entropy Search | Meta Ax/BoTorch | arXiv (2026.09, 2609.01431) | **[新发现]** 不确定性缩减采集函数估 scaling law 本身（<1/10 网格算力）——proxy 实验点该怎么选 |
| 98 | OpenEuroLLM Scaling Laws | OpenEuroLLM 联盟 | arXiv (2026.08, 2608.28308) | **[新发现]** LR/batch+边际 scaling law、overtraining regime 显式建模、全部 run 开源——可当 proxy 拟合的外部验证集 |
| 103 | Building Multilingual Bridges | Cohere Labs | arXiv (2026.09, 2609.10445) | 英语推理主干+多语言非推理数据即可跨语言迁移推理（SFT 层）——"推理是语言无关行为" |
| 104 | Climate-ModernBERT | UZH+McGill/Mila+Princeton 学术组 | arXiv (2026.09, 2609.07798) | 域适应 CPT 的 merge-vs-mix 受控对照：参数空间合并优于联合混合训练（encoder 侧） |
| 106 | **Curriculum as Transport** | Microsoft Research | **COLM'26** (arXiv 2609.09099) | **[有意思]** 课程解耦框架（ordering/exposure/pacing 分离）：无普适最优顺序、课程收益≠累积曝光——顺序实验设计参考 |
| 107 | Amortizing Scaling Law Construction Costs | TU/e+Tübingen/MPI-IS | arXiv workshop (2026.09, 2609.05016) | BO 采集顺序（compute 渐进+surrogate 幻想补全）省 10-100×——scaling law 拟合的省算力采集 |
| 113 | **Conditional Transfer from Controlled Pretraining Mixtures to Code** | 单作者（无机构） | arXiv (2026.10, 2610.11548) | **[有意思]** 可教性≠迁移：最优混比在内部点（75/25）非纯域；按在线 loss 调度会把真正有用的数据砍掉——同尺度 loss 指错方向的受控反例 |
| 127 | **SPOKES: Optimizing for Diverse Pretraining Data Selection** | UW NLP+A*STAR | arXiv (2026.06, 2606.15216) | 质量×多样性联合选择（G-Vendi 分数）超语义去重——多样性是质量的放大器不是替代品 |
| 128 | **Educational or Just Wikipedia-Style?** | Warsaw Univ. of Technology | arXiv (2026.05, 2605.23721) | 维基风格化改写即可翻转学习式质量分类器（~7% 文档）——L2 学习式分级的风格混淆警告 |
| 129 | **Domain Mixture Design via Log-Likelihood Differences** | 京都大学 | arXiv (2026.03, 2603.16622) | LL 空间几何定域权重（蒸馏的配比替代）——需目标模型，与本库场景错位 |
| 133 | **Shift is Good: Mismatched Data Mixing Improves Test Performance** | UChicago+清华+TTIC | arXiv (2025.10, 2510.25108) | 定理级：训练配比≠测试配比"几乎总是更优"——重仓目标域+保留其他数据的理论执照 |
| 134 | **Validity Threats for Foundation Model Research** | Tübingen AI Center+Vienna | arXiv (2026.06, 2606.05029) | **[有意思]** 方法论元批判：跨尺度可预测性无法只靠 proxy 实验自证——proxy 线/单 run 设计的威胁清单 |
| 135 | **Relative Generalization Invariance of LLM Pretraining** | Yale+Sea AI Lab 等 | arXiv (2026.09, 2609.33016) | 域间相对损失差跨优化器/架构近似不变、换数据流强破坏——用相对量（ranking）当 proxy 目标更稳 |
| 139 | **The Kinetics of Training** | 独立研究者（单作者） | arXiv 2607.27281（2026.07，40 页，无 venue） | 能力形成=成核事件（速率定律+浓度门槛点燃）："过临界步 val loss 平滑但能力永不点燃"的监控盲区论证——单作者，当假说源不当证据源 |
| 140 | **Problems with Chinchilla Approach 2** | Columbia+UT Austin+Amazon | arXiv 2603.22339（2026.03，无 venue） | isoFLOP 抛物线拟合有系统性偏差（套 Llama 3 公开数据=参数少投 6.5%）——一切 scaling law 外推的引用纪律 |
| 141 | **Unified Data Selection for LLM Reasoning（HES）** | USTC+阿里+NUS | arXiv 2605.22389（2026.05，无 venue） | 样本 top 0.5% 最高熵 token 的熵和当免训练质量信号：top 20%≈全量、0.6B 打分≈8B 自选——熵轴比风格轴更接近训练效用 |
| 144 | **GRAPE: Group Robust Multi-target Adaptive Pretraining** | EPFL（Jaggi 组） | arXiv 2505.20380（2025.05，无 venue） | 多目标稳健动态配比（group DRO×梯度对齐，0.7B 用 40% fewer tokens 达 uniform 水平）——静态法多目标脆弱的对照 |
| 146 | **LinMix: Data Mixing as Linear Domain Attribution** | 华沙大学+Credible AI+Vienna | NeurIPS'26 Workshop（arXiv 2610.37390，2026.10） | 顶点采样替代 Dirichlet 做 proxy 混合实验：288 域只有顶点支持线性拟合、4B 上 MBPP 相对 RegMix +46%——高维细域的采样警告（**[高价值] 已撤：标记仅限 T1/T2**；完整边界分析见 `docs/Dirichlet采样可用性边界.md`）。**⚠️ 与 CLIMB 一手经验张力**：#74 CLIMB 用 Dirichlet（20-21 簇粒度）配比优化有效——LinMix 自己也承认 24 粗域时"各方法都能拟合"；其"Dirichlet 崩"仅在高维细域（288）+贴 prior 条件下成立，**结论边界=细簇参数化时的采样警告，非普适否定** |
| 147 | **Hinted Scaling Laws** | Stanford | arXiv (2026.11, 2611.02270) | **[新发现]** hint 把 benchmark 变连续难度谱→小模型可外推：1e20 FLOPs 拟合预测 1e23 GSM8K 误差 3pp（标准法差 13pp）——门3"小模型 accuracy 读不出趋势"的测量层工具（T3：评测方法论，非配比本体） |
| 148 | **Advancement or Acceleration? How Data Curation Improves LLM Training** | 疑字节系（待核） | arXiv (2026.10, 2610.36983) | **[有意思]** 策展=提升还是加速的形式化：Δ-test（策展前可算的训练/目标 loss 差）非零→收益持久、Δ≈0→随预算消失——投入策展/配比搜索前的事前判据（连 #111/#122） |
| 149 | **Rewriting Scaling Laws for Synthetic Data** | Stanford | arXiv (2026.10, 2610.31867) | 合成改写最优占比随算力增长（<10%→32% @1e22）、22B 外推 33× 误差 1.1%——#14 合成 30:70 的 scaling-law 版（合成轴暂不聚焦，备查） |
| 150 | **A Scaling Theory of Multi-Epoch Pretraining** | Bordelon 系 | arXiv (2026.11, 2611.03590) | 多 epoch 晚期过拟合理论+集成（logit 平均）延缓最优停点、固定数据下 ensemble 分配律——重复轴 #123/#125/#138 理论补强 |
| 151 | **Scaling Laws for Multimodal Data Mixtures** | EPFL+CMU | **NeurIPS 2026 Spotlight**（arXiv 2610.14423） | 268 实验/457M-8.3B MoE：compute optima 按模态而异、视觉/音频重复>2× 无增益、跨模态干扰不对称——多模态配比轴外参照（对齐 #07/#08） |
| 154 | **Time-Incremental Continued Pretraining of LLMs** | Mila/Concordia+Utrecht 等 | arXiv (2026.09, 2609.23916) | 时间增量 CPT（1B-8B 六模型）：近零遗忘获取新知识、精选 6B≈宽泛 40B（质量>数量）、LR 双最优差 10×、LoRA=全参 CPT |
| 159 | **Beyond Data Schedule: Loss-Reweighting (LARE)** | Jiazheng Li 等 5 人（未核） | arXiv (2026.10, 2610.32141) | 固定数据曝光、按 running loss 域级 loss 重加权（inverse-square-root 规则免搜索）——"配比之外的独立控制"：单 checkpoint 超两阶段各自最佳；1.5B/240B +7.84pp；连 #51 Nexus（目标函数级域平衡）/#113 ADO（loss 驱动调整的迁移风险） |
| 162 | **Beyond Passive Imitation: On-Policy Mid-Training Distillation** | 阿里+复旦/川大/港理工/上交 | arXiv (2026.11, 2611.13855) | **[有意思]** MT 阶段小 LR+KL 在线蒸馏替代 NTP；反例比正结论硬：同语料同配方 Llama-3.2-3B 升/Qwen3-1.7B-Base 大降（跨 base 迁移线）⚠️ LR confound（1e-5 vs 1e-6+KL）、单 seed、代码 404 |
| 163 | **Primitives and Paths** | Anqi Qu 等（未核） | arXiv (2026.11, 2611.11260) | **[新发现]** 二维 post-trainability 状态空间（full-solution support × primitive coverage）三 regime（放大/组合受限/习得受限）——"RL 是放大还是新习得取决于基座状态"理论化，与 3 Ultra MOPD 支撑集机制发现同构 |
| 166 | **Lara Mini Base** | Translated/EU LARA 联盟 | arXiv (2026.11, 2611.16956) | **[有意思]** 多语配比 TR：6T tokens/45 语言/7.2B-A0.6B MoE——训练中非英语占比逐步反超英语（语言份额=课程）、MoE 缓解语言间容量竞争（连 #52）；T3：多语非业界当前重点、数学才是 |
| 167 | **Agentic-OPD** | Tingcheng Bian 等（未核） | arXiv (2026.11, 2611.12799) | **[有意思]** RL checkpoint 动物园当免费教师池：best-val ckpt 会丢早期 ckpt 已解题（并集>>单点），HCN 按学生未解覆盖选师+LAD 局部蒸馏——**3 Ultra MOPD/GLM-5 OPCD"前序 ckpt 当教师"的学术机制版**；4B 学生 5 域 24 基准最高 +7.4 |
| 168 | **SFT Overtraining Predicts Rank Inversion via Entropy Collapse** | Aphale/Liu 双作者（未核） | arXiv (2026.06, 2606.18487) | **[有意思]** 过量 SFT 伤 RL 受控证据：3B 五档 SFT 深度——pre-RL pass@1 升但峰值 GRPO pass@10 0.806→0.481；机制=熵塌缩→组内奖励全同无信号；**pre-RL 熵 ρ=+0.69=事前前兆**；破阈值 p*(g) 才反转（6.7B 未破不反转）——#161 轨迹封顶条的学术机制版 |
| 169 | **Sequential Beats Joint: OPD 与 RLVR 的相互作用** | Boyan Li/Xi Ye 等（未核） | arXiv (2026.09, 2609.04108) | **[新发现]** OPD-then-RL 两阶段>一切联合目标——**OPD 扩张教师支撑解覆盖、RL 在支撑内 sharpen、联合互相干扰**：三明治顺序栈的顺序化证据；与 3 Ultra MOPD（联合式奖励成形）对照面记档；与 #163 支撑集框架同构 |
| 170 | **RL Starts before RL: OPD for Better RL** | Shuai Dong 等（未核） | arXiv (2026.09, 2609.28145) | **[有意思]** 精确化边界：on-policy 蒸馏当 RL 准备期>直接 RL>SFT-then-RL，收益与即时精度无关——**伤 RL 的不是模仿数据，是离策略/过量模仿**：#161 封顶+on-policy 设计的第三方注脚、与 MOPD"sampled-token>全分布"同向 |

### 第四梯队：轴外相关或初步工作

| # | 论文 | 机构 | Venue | 一句话核心 |
|---|------|------|-------|-----------|
| 51 | **Nexus: Same Loss, Better Downstream** | 清华+字节跳动 Seed | arXiv (2026.4) | **[高价值/轴外]** 同一 pretraining loss GSM8K 差 15 分：优化器收敛到 close minimizer 自动平衡欠优化域——val_loss 不充分的终极证明；与配比正交可叠加 |
| 52 | **Why Larger Models Learn More** | U Chicago+Apple+Anthropic | arXiv (2026.5) | **[新发现]** 容量竞争：小模型把容量分给高频域、牺牲稀有复杂域——"为什么小 proxy 学不了 STEM"的机制 |
| 53 | **RISE: Scalable Data Attribution** | Vendor AI | arXiv (2026.4) | **[新发现]** forward-only 数据归因、112× 存储压缩、扩展到 32B |
| 54 | **Kernel Surrogate for Task Attribution** | U Chicago | **ICLR'26** | **[新发现]** 核函数代理>线性代理（LOO 相关+25%、数据选择+40%）——可替代 LightGBM 的代理模型 |
| 55 | **K-EXAONE 2.0** | LG AI Research | Tech Report (2026.8) | "difficulty-focused mid-training"——难度聚焦中训练的厂商数据点 |
| 56 | **Beyond Rephrasing: Book-Level Organization** | - | arXiv (2026.7) | 书级组织的合成教材（686K 教材/32B tokens）+1.09 平均提升——中训练语料形态 |
| 57 | **DeepSeek-V4** | DeepSeek-AI | Tech Report (2026.4) | 32T+33T tokens：提到 mid-training 加 agentic 数据但无配比细节——头部 TR 保密程度的样本 |
| 59 | **Qwen3** | Alibaba Qwen | Tech Report (2025.5) | 三阶段预训练（S1 通用 30T→S2 推理 5T→S3 长上下文）+实例级配比优化超越域级方法 |
| 70 | Data Mixing as Mixture Experiment | Yicheng Mao, Hongru Du | arXiv (2026.08) | **[有意思]** Scheffé 混合实验设计：域间成对交互真实、I-最优设计省 25% proxy runs——无 venue 仅仿真 |
| 86 | Instella-MoE | AMD | Tech Report (2026.09, 2609.00791) | 16B/2.8B MoE 全管线开放（含 mid-training+feedback-driven curation）——AMD 的中训练数据点 |
| 87 | SPT | 无机构（3 作者） | arXiv (2026.08, 2608.26563) | skill 包做中训练语料（agentic 形态）——可信度低 |
| 92 | MidTool | UC Berkeley+UW | arXiv (2026.08, 2608.20314) | 中训练工具使用语料合成管线（agentic 方向） |
| 93 | OctoLong | Würzburg UKP Lab | arXiv (2026.08, 2608.05141) | AST 管线构造跨仓库长程依赖语料——长上下文中训练形态 |
| 97 | Qwen3.8-Next Architecture | Alibaba Qwen | Tech Report (2026.08, 2608.30320) | 架构 TR，配比相关仅一点：loss 单调降而 accuracy 饱和（n-gram 词表）——loss≠accuracy 新数据点 |
| 99 | SPARK: Skeleton-Guided Reasoning Synthesis | MSRA | arXiv (2026.08, 2608.30214) | 科学文献→推理骨架→四类任务的合成管线——明确高难度，1.5B 勿直接灌（#88 容量警示） |
| 100 | Where Induction Runs Out | 单作者（无机构） | arXiv (2026.08, 2608.29411) | MDL 两部分编码当难度信号：OEIS 上实测是记忆不是归纳——与小模型死记同构 |
| 102 | MiniCPM5-2B | OpenBMB/面壁智能 | Release (2026.09.07) | Base→Midtrain→SFT→Final 四 checkpoint 全链开源——可 diff 工业级 2B 中训练增益、为本库期望收益定标 |
| 105 | **Data Scarcity and Model Sparsity: MoEs Overfit More to Repeated Data** | Stanford+UW | arXiv (2026.09, 2609.11917) | MoE 4× 重复即退化 vs dense 8× 近无损——架构转 MoE 时的重复边界 |
| 108 | Edu-QuRating | UCL 背景学术组 | arXiv (2026.09, 2609.09425) | 教育价值评分多维化（QuRating 后续），9 预训练基准超 FineWeb-Edu |
| 109 | Data Scout | IBM Research | arXiv (2026.09, 2609.05766) | 反转"过滤 CommonCrawl"范式：taxonomy 定向爬取——将来 STEM 池缺数据时的现成管线 |
| 112 | **Stress-testing Alignment Midtraining** | Arcadia Impact（UK AISI 合作） | arXiv (2026.09.17, 2609.20412) | **[有意思]** 2% 冲突 SFT（~80k tokens）即可推翻 190M token 中训练动机——中训练收益必须 SFT 后再测 |
| 114 | **Generalization Dynamics of LM Pre-training** | UC Berkeley+Stanford/GDM | arXiv (2026.09, 2609.33150) | **[高价值/新发现/轴外]** mode-hopping：相邻 checkpoint 深层泛化 81%→0%→81.7% 而 loss/常规 benchmark 平稳——中间 checkpoint 必须保存、终点要比较后选 |
| 116 | **Fisher-Guided Submodular Data Selection for CPT** | UIC+Cisco | arXiv (2026.10, 2610.02593) | PPL 选择会系统性磨损基座已承诺方向（Fisher 几何）——"loss 信号在参数空间指错方向"的机制证据；医疗单域场景窄 |
| 117 | **OptiSelect: How does the Optimizer Shape Data Curriculum?** | EPFL | arXiv (2026.10, 2610.03432) | **[有意思]** 边训边挑数据不是免费的（读 2× token、总账反而亏）且取决于优化器——在线选择的边界数据点 |
| 118 | **What Pre/Midtraining Make Learnable from Rewards?** | CityU HK+UNSW | arXiv (2026.09, 2609.38446, 160 页) | 理论：RL 只能在已有计算中选择——中训练语料须供应可被奖励调用的计算形态 |
| 119 | **AutoDataBench** | Ruifeng Yuan 等 | arXiv (2026.09, 2609.40097, 有代码) | 数据智能受控测试床（agent 理解/操纵/改进训练数据能力的隔离评估） |
| 120 | **Multilingual Quality Classifiers** | EPFL | arXiv (2026.10, 2610.11585) | 英文质量分类器的多语言扩展——英文 STEM 场景备查 |
| 121 | **Scaling Down the Scaling Laws** | 单作者 | arXiv (2026.10, 2610.06387, 综述) | 资源受限训练综述——单作者二手综合，仅作索引锚点 |
| 136 | **Multi-task Code LLMs: Data Mix or Model Merge?** | RPI+Cisco+IBM Research | arXiv (2026.01, 2601.21115) | mix vs merge 的尺度边界：7B merge 优、2B mix 优——1.5B 落 mix 侧 |
| 142 | **WARP: Weight-Space Analysis for Recovering Training Data Portfolios** | UW-Madison | arXiv 2607.01686 + **ICML'26 Workshop**（有代码） | 从公开权重反推微调数据配比（merge 插值造伪轨迹+梯度对齐）——逆向配比取证 |
| 152 | **Kolibri: A Sovereign European Model** | Aleph Alpha | Tech Report (2026.10, 2610.31945；Apache 2.0) | 78B/3.46B active MoE 全开放：24T 跨 pre/mid/long-context 三段、德语 >20%+2T 自策展——激活量与 3B-A1 重合的开放 MoE TR |
| 153 | **IQuest-Q1** | IQuest | IQuest Report (2026.10, 2610.30949；开源) | 320B/15B MoE：S2 课程加码 code 配比+细粒度 mid-training 扩域渐进上下文+跨阶段 checkpoint 合并——MoE 中训练结构参照 |
| 164 | **DRIVE** | Qing Liu/Xinrui Chen 等（未核） | arXiv (2026.10, 2610.32501) | 源预训练数据不可得时 SFT 样本加权替代回放（contribution+多样性信号可微加权+先验正则）——SFT 层轴外备查 |

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

**结论**：领域整体向动态迁移，QuaDMix 的静态 LightGBM 回归越来越孤立。最低成本动态化路径：① 记录 proxy loss 轨迹（RegMix-D）→ ② 目标训练加域 loss 监控 patch（DomainPilot）→ ③ 如需更强效果加 model-driven reward（HDS）→ ④ 在目标模型上 0.1B token 免训练测 Group Influence（TiKMiX #143）。**最硬的一条**：#143 TiKMiX 是 QuaDMix 同团队（字节，作者重叠 5 人、遵循 QuaDMix 实验设置）——**发明静态配比搜索的团队自己转向了动态**，1B/1T 上超 RegMix +1.6~2.0% 而搜索开销仅其 20%。

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

- **域间成对交互是真实且重要的**——弱域通过与 web 文本组合变得有利。Scheffé 二阶模型 $\mathcal{L}(\boldsymbol{r}) \approx \sum \beta_k r_k + \sum \beta_{kk'} r_k r_{k'}$ 中的 $\beta_{kk'}$ 就是域间关联 $\sigma_{kk'}$，且可通过稀疏化有效估计
- **稀疏 Scheffé 模型跨尺度保持 mixture ranking**——对 Layer 4（代理不可解性）的重要 nuance：无结构预测器（LightGBM/神经网络）不可迁移（QuaDMix 失败），但**结构化（稀疏低阶多项式）响应面可能跨尺度稳定**。不推翻 Layer 4，只细化——仍是 loss 空间，不解决 Layer 5
- **I-最优设计省 25% proxy runs**——实验设计效率直接降低 proxy 搜索成本
- **⚠️ 可信度存疑**：2 作者无 venue 无机构，经验声明仅仿真验证。但方法学根基（Scheffé 混合实验设计）本身是 60 年经典统计，框架价值独立于论文可信度。**保留为 [有意思]：框架有参考价值，具体经验声明需批判看待**

**Scaling Domain Data Repetition (#71, arXiv:2608.14071, 清华+字节 Seed)** 发现域级最优重复次数与 val loss 强负相关（非数据量决定），同 TPP 可跨尺度迁移。

- 团队可信（清华+字节 Seed），但**对本项目场景帮助有限**：目标场景中训练 TPP≈1.2 vs 论文验证 TPP=20，迁移条件不直接满足；本项目瓶颈是数据量不足（1.26B vs 参考 150B），非重复策略优化

**结论**：域间交互 $\sigma_{kk'}$ 不仅存在，而且可以用经典统计工具（Scheffé 混合实验设计）在 proxy 层面估计。代理不可解性针对的是无结构黑箱——施加正确结构先验（稀疏低阶多项式在单纯形上）可能部分恢复跨尺度信息。但关键限制：这些仍在 loss 空间，Layer 5（loss-accuracy 鸿沟）不受影响。**#70 的框架思路值得参考，但不应作为高可信依据；#71 对本项目场景的战术价值需等数据量问题解决后才有意义。**

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

**对本项目的直接修正**：数学池 1.26B token 在 1.8B 预算下 ~1.4 epoch；400M proxy 线若用 1/16 预算且不子采样则只有 ~0.09 epoch——重复率失配 16 倍。proxy 搜索前必须先按目标重复率子采样。

### 25. CPT 与 from-scratch 的最优配比形状不同：CPT 尖锐、from-scratch 均衡

**Nemotron CLIMB (#74, NVIDIA, T1)** 深读结论（`docs/NemotronCLIMB_深度分析.md`）：

- CLIMB 搜索设定与本项目场景**同构**：10T-token 基座（62M/350M/1B，WSD 训练）上搜 40B-token CPT 配比，proxy+LightGBM+迭代搜索；1B/40B CPT 后 60.41 vs RegMix 59.37 / DoReMi 59.16
- **§6 关键发现**：CPT 的最优配比是**尖锐的**——少数域（C8/C9/C18/C19）主导；from-scratch 的最优配比需要**均衡**。两者搜索结果不可互换
- **直接支持 r=0.7**：本项目的激进 STEM 占比是"CPT 形状"，有 NVIDIA 实验背书
- **警告**：ClimbMix 数据集内部构成是 from-scratch 优化的（偏均衡）——**不能直接抄为中训练配比**，只适合当语料池
- proxy 尺寸收益温和（62M→350M 仅 +0.3）——本项目的 400M proxy 在合理区间
- **坑**：预测器有位置性（Marin 批评：重聚类使已搜配比失效；修复=微簇基直方图 h=V·w）；112 次 proxy 搜索 ≈5040 GPUh（≈0.8× 一次目标训练）——N=1-2 下不可复现，只能借鉴结论而非流程
- **Apple 侧补强（#131）**：Optimal Splitting（#03 原班）用 scaling law 定"通用预训练→专门域 CPT"的最优切换点 t_S——各预算下 CPT from scratch 都显著差于先预训练再专门化，且 t_S 两侧准确率先升后降（内在最优存在，预训练过多也变差）。正面支持"ClimbLab 基座+STEM CPT"结构；**⚠️ 边界**：论文的专门化阶段喂纯域数据，未测 CPT 内通用/专门混合比例——与 r=0.7"CPT 中保留 30% 非 STEM"只是结构类比，非其结论

### 26. 验证目标的选择比配比方法的选择更重要

**MDE (#122, Google DeepMind, [高价值], arXiv 2502.15950)**：

- **Table 1（510M/50K steps）**：优化目标与下游 All 准确率的 Spearman——用 11 个 end-task 验证域的 loss 做 objective **ρ=0.844**，用 7 个 SlimPajama 域 loss（DoGe/DML 常用目标）只有 **0.282**——"换验证目标"的收益远大于"换配比方法"
- **Mixture of Data Experts**：每域训一个小专家模型，loss 按混合权重加权聚合 ≈ 任意混合配比的 val loss（附 Proposition 理论保证）；作为回归特征后配比排名预测 0.81→0.95，仅需 ~25 个真实混合 run 做监督，独立搜配比省 3× 算力
- **对本项目**：CAP v1/OpenHermes 验证集的域覆盖度检查的理论根据——validation mixture 本身是最大的杠杆，先检查它再谈换方法
- **工具化落地（#132 SuperValid，蚂蚁）**：capability 级 OOD 验证集的构造配方——benchmark→知识成分蒸馏→语料检索扩展→LLM 过滤改写成验证样本；2T tokens 受控实验里 IID val loss 随配比变化漂移而 SuperValid loss 平稳（MSE 1.51e-3 跨架构/尺度/数据分布）。CAP v1 若要升级为"capability 级 OOD 验证集"，这是现成管线；与 MDE 互补：MDE 证明"选对验证目标"收益大，SuperValid 给出"怎么造对目标"的工业配方

### 27. 模型合并可以当配比搜索的低耗代理，且工业级跨尺度迁移成功

**MergeMix (#130, 人大高瓴+蚂蚁, [高价值], arXiv 2601.17858)**：

- **方法**：每域只训小 token 专家（K=4，沿 #74 Nemotron 四域划分，5B≈试验预算的 2.5%），在**参数合并权重空间**里对下游 benchmark 搜最优配比——"权重混合≈数据混合"有 Taylor 展开轨迹分解论证
- **工业级验证**：8B/16B MoE（各 1.4B 激活）mid-training（200B 预算，恒定 LR+merge 16 checkpoints 模拟 annealing）：math/code/reasoning 超**已上线生产的人工调参基线**、其余域近无损；对比 Adapted RegMix（#36）也占优
- **两个关键数字**：搜索成本降 **100×**；排序保真 Spearman **ρ>0.9**
- **最重要的证据（跨尺度）**：8B 上搜出的配比**直接用于 16B mid-training 仍超人工基线**——proxy→target 配比迁移在工业尺度成功，是 400M→1.5B 计划的正面外推证据；与 #73 MAI rank non-invariance（反例）对照读：跨尺度迁移"常常成功但非保证"，机制边界在容量差与相变区（#01）
- **与 merge 家族分工**：#04 DeMix=分开训再合并当**最终模型**；MergeMix=合并当**搜索代理**；#104/#136=merge vs mix 的受控对照。本项目 N=1-2 下合并推理成本近零，是 proxy 搜索之外的第二条低耗路径（但 5B 专家成本对本项目仍偏高，结构上可缩）

### 28. 相对信号比绝对信号耐造：跨优化器稳定的损失差 + 错配可有益的理论

**RGI (#135, Yale+Sea AI Lab, arXiv 2609.33016)**：

- **现象**：任意两个 token 的 val-loss 之差，跨优化器（Adam/Lion/Soap/Muon）与中等架构变化**近似不变**——这些选择对 token 损失是"近似均匀平移"；换数据流（FineWeb→CodeParrot）则**强烈破坏**该不变性（0.7B 验证），且有二次模型定理刻画（GD/Adam/Muon 统一）
- **含义**：**域间相对损失差=数据敏感、优化器/架构不敏感的信号**。①用相对量（域 loss 差/ranking）当 proxy 目标，比绝对 loss 更可能跨尺度保持——400M→1.5B 同栈不同尺寸的 proxy 协议应优先输出相对量；②本项目 AdamW vs #126 nanochat Muon 的优化器差异对相对结构影响小（换优化器≈均匀平移）
- **边界**：RGI 验证的是优化器/中等架构变化，**跨容量**（小 proxy→大 target 的容量差）是否保持未明——与 #73 rank non-invariance、#01 相变的张力是 Layer 4（代理不可解性）的正面战场

**Shift is Good (#133, UChicago+清华+TTIC Srebro/Lyu, arXiv 2510.25108)**：

- **定理级结论**：训练配比≠测试配比不但不亏反而"**几乎总是更优**"——只要各子群误差函数非常数，最优训练混合比例 generically 不等于测试分布比例（power law/记忆/组合技能/迁移四情形+Corollary 7.4）
- **含义**：**r=0.7 的理论执照**——训练混合不必匹配评测构成（STEM 评测近 100% STEM，训练留 30% 其他数据可以是 generic 最优而非妥协）；把"为什么不是 r=1.0"的回答从经验直觉（#113 内部点/#14 30% 合成/#01 预算依赖）升级到定理层。实操提醒：错配的方向与幅度依赖各域误差曲线形状——具体偏离多少仍须 proxy 实验定，理论只保证"≠测试分布"这一点

### 29. 数据调度结论是 LR 协议的函数：排序反转 + 延迟衰减 + MoE 渐进入场（#145）

**Data×LR Schedules (#145, 清华 PACMAN+Lyu, arXiv 2610.29348)**（深读：`docs/DataLRSchedules_深度分析.md`）：

- **排序反转**：固定算力与 MT 预算下，"早混入 MT+PT 陪跑 vs 纯 MT 压轴"的胜负被 LR 调度单独反转——全程衰减（FDR）下早混入赢（复现 #157）、恒定 LR 下压轴赢。**一切"数据课程/调度 A 优于 B"的结论都隐含一个未声明的 LR 协议**；r-scan 必须锁死并报告 LR regime，跨 LR regime 的配比排名不可迁移（#137 对顺序实验的警告扩展到 PT/MT 全程）
- **延迟衰减**：19 个 sweep 一致——LR 应在 MT 进场后才开始衰减，只覆盖 MT 窗口末段（dense TS 40%/ZL 20%、MoE 20%/10%）；MoE 下 MDR（进场即衰减）三设置全劣于 Const——衰减起点是一阶变量不是规范选择。SDE+块对角二次理论给出定理版：Const 最优=纯压轴；强衰减下三阶段最优（混合段 MT 占比随 LR 降而升的凸 ramp，ZL 是其线性近似）
- **MoE 渐进入场（3B-A1 直接）**：数据跳变（TS）触发 load-balance loss 尖峰（~80 步）+专家负载排名大洗牌（Spearman 0.30 vs ZL 的 0.74），渐进（ZL）消除且更稳；**MoE 中训练配方默认=ZL 进场+MT 窗口末 10-20% 延迟衰减**
- **路线分歧的自觉**：恒定 LR+checkpoint 合并（#84 CMA/#137 修复/Marin §8.8）与延迟衰减是同一权衡的两个工程解，但**数据调度偏好相反**（Const 理论最优=压轴；衰减路线=早进场渐进升）——3BA1 两臂对照可顺带产出 Const+merge vs 延迟衰减的独立数据点

### 30. 零分以下仍有梯度：hinting 让相变前信号可测（#147）

**Hinted Scaling Laws (#147, Stanford tatsu-lab, arXiv 2611.02270——T3：评测方法论，非配比本体；[新发现] 保留)**（深读：`docs/HintedScalingLaws_深度分析.md`）：

- **原语**：hint（按难度揭示部分解/teacher logit 混合）把 benchmark 变成难度连续谱——小模型 accuracy 随 hint 水平平滑上升，且 (规模, hint) 平面上 iso-accuracy 等高线近似平行对角线（**提示与规模线性互换**）；floored sigmoid 联合律 6 参数，≤1e20 FLOPs 拟合外推 1e23 的 GSM8K 误差 3pp（标准律差 13pp），中位误差 21pp→4.3pp
- **对 §22（相变前信号）的含义**：#01 说阈值下 accuracy 恒零、门3 说小模型读数无梯度——本篇给出**测量层的第一操作方案**：不是小模型没信号，是评测没给信号显形的梯度；Proposition 1 定理化"什么样的 hinting 有信息"（必须按底层能力分离 pre-emergence 模型，"直接给答案"式 hint 零信息）
- **r-scan 直接用法**：①近零配比候选按 hint-prior 排名（单 hinted rollout≈5 个 unhinted 判别力，λ=8 z 加权，代码开源）；②外推预测用联合律+**多种子硬要求**（单种子误差 3×）；③Delphi 开源梯子=外部验证集（先 hinted 复核再上大档）
**边界**：保序≠保距（头部决策需 unhinted 复核）；交换率线性是经验假设；hinting 只解决"测得准"不解决"该配什么"——与 ScalingBench（条件 NLL）是失败模式正交的两条连续读数路

### 31. r-scan 采样协议集齐：顶点定主效应+内部点测交互（#146）

**LinMix (#146, 华沙 Credible AI+维也纳, arXiv 2610.37390, NeurIPS'26 Workshop)**（深读：`docs/LinMix_深度分析.md`；**Dirichlet 可用性边界的完整交叉分析：`docs/Dirichlet采样可用性边界.md`**）：

- **顶点采样**：proxy 混合按 Greedy fill 填到 cap 或置零（+配对反向排列去相关）——D-最优、零超参；Dirichlet 的定理级代价（Theorem 1：贴 prior 欠分散 1/c × 尾部饥饿 d^(p−1)，BPB 变动小 5×=没信号可回归）。**与 #70 Scheffé、#124 重复率子采样三处独立收敛：细簇参数化时该用经典统计设计。⚠️ 按本库 CLIMB 一手经验修正边界**：#74 CLIMB 用 Dirichlet（20-21 簇粒度）配比优化有效、LinMix 自证 24 粗域各方法持平——**"Dirichlet 崩"仅在高维细域（288）+贴 prior 条件下成立，粗簇层（≲24 域）Dirichlet 实战可用；读作"可用性随维度退化的警告"，非普适否定**
- **合成方案（r-scan 采样协议就此集齐）**：顶点群定主效应（哪些域正归因，β 可解释）+ 内部/轴向点测交互（#70）——两段式；LinMix 的顶点最优结构性测不到混合区最优点（#27/#130：交互真实、math 最优点在内部），**内部点扫描不可省**；重复轴双旋钮 κ（硬 cap）+isoelastic r（软曲率）
- **数字边界**：价值全在细域（288 域 Dirichlet ρ≤0.39 vs 顶点+线性照常、ARC +13.8%；**24 粗域各方法差别小**）；4B 上 MBPP +46%/MATH +31.6% vs 最佳基线，且 RegMix 的 GSM8K 混合低于 prior（不分化 vs LinMix 目标定向分化）；BPB 目标+学术规模+workshop venue——**协议全引、配比数字不引**

## License

MIT
