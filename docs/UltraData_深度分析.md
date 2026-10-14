# UltraData: Data Science and Technology Towards AGI Part I — Tiered Data Management

- arXiv 2602.09003（2026.02.09，27 页），Tsinghua（Yudong Wang/Zixuan Fu/Chaojun Xiao/Xu Han/Zhiyuan Liu/Maosong Sun）+ ModelBest（面壁）+ BIT + 华农
- 数据/工具全开源（Apache-2.0）：UltraData-Math L1 170B/L2 33B/L3 88B、Ultra-FineWeb-en L2 1.8T/en-L3 200B、Ultra-FineWeb-zh L2 120B/zh-L3 200B、Math-Parser/Math-Generator/en-zh Classifiers

## 一句话总结

L0-L4 分层数据管理框架（按数据纯度与管理成本分层、跨训练阶段策略性分配），在 MiniCPM-1.2B 上三层验证：数学域 L3（合成精修）比 L1（web 过滤）MATH500 +9.24pp 且跨域迁移（English/Code 均涨）；同配比下 L1→L2→L3 顺序出场（tiered）比全程混合（mix）+1.49pp 且后段增长不饱和——1.2B 规模的"质量层级 × 出场时机"双重证据。

## 0. 实验设置（对用户最重的部分：三种数据验证方法学）

**模型**：MiniCPM-1.2B——非嵌参数 1,247,442,432（≈用户 1.5B 同规模区），52 层，GQA 24Q/8KV，d_model 1536，vocab 73448（MiniCPM3-4B tokenizer）。Megatron-LM 训练，全程 μP（Maximal Update Parameterization）。

**三种验证协议**（都是"30% 待验数据 + 70% 默认分布"的混合续训）：

| 协议 | 预算 | 基座 | 配置 | 用途 |
|---|---|---|---|---|
| Efficient Verification | 10.5B | 1.1T 从零（WSD：1T stable + 0.1T decay） | seq 4096，GBS 512，5000 步，LR 指数衰减 1e-3→5e-5（500 步 warmup） | 快速筛选，迭代效率高但方差大 |
| Pre-train Verification | 125.8B | 从零 | seq 4096，GBS 2048，15000 步，cosine 1e-2→5e-4 | 中等置信 |
| **Decay Verification** | 104.9B | **1.3T 已完成 warmup+stable 的基座** | seq 4096，GBS 1280，20000 步，LR 指数衰减 7.5e-4→3.75e-5 | 最接近真实训练终点，定论级 |

**Decay Verification 与用户中训练结构同构**：已完成的大规模基座 + 目标数据混合续训 + LR 衰减——就是用户的 continual pre-training 范式（ClimbLab 1.2T 基座 + 1.8B tokens 混合续训）。差别仅在方向：UltraData 是 30% 目标 + 70% 默认（防遗忘为主），用户是 70% STEM + 30% 通用（域适应为主，目标占比高一个量级）。

**评测**：OpenCompass；MATH500/GSM8K 4-shot 生成式、MMLU/C-Eval 5-shot PPL、MBPP 3-shot/HumanEval 0-shot。

## 1. L0-L4 框架定义（每层 = 数据属性 + 管理策略 + 训练角色）

| 层 | 定义 | 典型操作 | 指定训练阶段 |
|---|---|---|---|
| L0 Raw | PB 级未策展原始资源，高冗余高噪声 | 爬取/解析/存档 | 不直接训练（存档+可追溯） |
| L1 Filtered | 格式标准化、基本可读 | 启发式清洗+去重 | 大规模预训练 |
| L2 Selected | 主题鲜明、高信息密度 | 分类器/质量打分/语义选择 | **Decay + MidTraining** |
| L3 Refined | 结构化推理、显式教学意图 | 重写/合成/人工精修 | **MidTraining + SFT + RL** |
| L4 Organized | 可信可验证的结构化知识 | 编排+事实验证（KG/DB） | RAG/下游 |

核心主张（论文原话）："strategically deploying valuable data at critical training moments – such as mid-training phases or annealing stages"——数据价值须与出场时机匹配；高质量数据贵，应部署在边际收益最大的关键时刻。**框架显式把 L2/L3 指派给 mid-training**，这是对用户"中训练取什么数据"问题最直接的工业级回答。

## 2. UltraData-Math 三层构造细节

- **L1（170.5B，85.6M docs）**：数学专用 HTML 解析（magic-html 基座 + w3m 布局保留渲染 + 多级 fallback），MathML/KaTeX/AsciiMath 统一标准化为 LaTeX；启发式清洗（格式修复 mapper + 内容 filter）+ 文档级去重。动机：trafilatura/readability 等通用解析器破坏公式结构、论坛类数学讨论提取不全。
- **L2（33.7B，14.98M docs，当前发布为 preview）**：专有大模型对种子数据多维打分 → 蒸馏为轻量 embedding 分类器 → 对 L1 全量打分筛选。保留：详细解题步骤/数学概念讲解/高水平学术讨论；排除：名词堆砌/无意义数字列表/低幼内容/非数学噪声。
- **L3（88B，81.4M docs）**：种子清洗+LaTeX 标准化后，多模型集成（Qwen2.5-72B-Instruct/Qwen3-32B/GLM-4.5）生成五种格式：① 难度分层 QA（小学→本科课程递进）；② 7 种 persona 对的师生多轮对话（追问/纠错/引导）；③ 多风格改写（严谨教科书/竞赛解题/直观科普——把数学核心逻辑与表述风格解耦，防过拟合窄语言分布）；④ 知识点教科书模块（定理/公理→教学讲解+多级练习）；⑤ persona 集成合成。全部输出经 LaTeX 语法错误+逻辑不完整严格过滤。

竞品关系：种子生态来自 Nemotron-CC-Math/MegaMath/FineMath（致谢名单），对标方法为 Nemotron-CC 的 "parsing-then-editing" 范式；差异在分层体系化+五格式多样性。

## 3. 核心结果

### 3.1 四域层级效应（Table 5，10B efficient verification）

L1→L2→L3 平均分：Web-en 52.26→53.36→53.96（+1.70）、Web-zh 49.44→50.66→51.48（+2.04）、**Math 23.78→24.82→30.84（+7.06，四域最大）**、Code 34.49→35.29→36.28（+1.79）。L3>L2>L1 无一例外。**数学域的层级效应比 web/code 大 3-4 倍**——质量精修在数学域的边际收益最高。

### 3.2 数学域 100B Decay Verification（Table 6，对用户最重要的一张表）

| | MATH500 | GSM8K | English avg | Chinese avg | Code avg |
|---|---|---|---|---|---|
| Math-L1 | 27.78 | 54.66 | 51.32 | 51.59 | 37.30 |
| Math-L2 | 29.20 | 52.92 | 51.69 | 50.84 | 38.41 |
| **Math-L3** | **37.02** | **61.79** | **54.77** | **53.48** | **41.10** |

- **L2 vs L1 增益小且混合**（MATH500 +1.42、GSM8K **-1.74**）；**L3 vs L1 才是大头**（MATH500 +9.24、GSM8K +7.13）。结论：对数学能力，纯"筛选"（学习式质量分级）不够，**"格式重构+合成"（QA/对话/教科书化）才是跃迁来源**——web 数学文本多为陈述式，缺结构化推理步骤。
- **跨域迁移**：Math-L3 使 English avg +3.45（OBQA +15.40、BBH +5.87、ARC-E +5.29）、Code +3.80（MBPP +4.56、HumanEval +3.05）、Chinese +1.89——**高质量数学数据是通用推理能力的驱动力**，直接支持用户 r=0.7 STEM 重仓的合理性（STEM 数据收益不局限于 STEM 评测）。
- 与公开基线同协议对比（数据集卡）：MATH500 37.02 比 Nemotron-CC-Math 4plus +3.62、GSM8K +3.34，同时 code/general 不掉。

### 3.3 Tiered vs Mix（Table 7 + Fig.3，出场时机独立于配比）

设置：120B 从零训练，域分布 50% web-en/25% web-zh/8% math/17% code；两种策略**域配比与 L1:L2:L3=1:1:1 完全相同**，仅出场方式不同——mix 全程混合，tiered 分三段各 40B 顺序 L1→L2→L3。

- 总 avg：**tiered 31.66 vs mix 30.17（+1.49pp）**；Math avg 4.60 vs 2.36（近 2×）、MATH500 4.20 vs 1.60、GSM8K 5.00 vs 3.11；Code 9.70 vs 7.25；Chinese 27.54 vs 24.72；English 42.29 vs 41.41。
- 训练曲线：前段（都在吃 L1 为主）两者相近（~24.7→28.3）；后段 tiered +3.31（28.35→31.66）vs mix +1.91（28.26→30.17），**1.7× 增长差**——后段引入高质量数据避免性能饱和，低质量样本会拖慢晚期收敛。
- 轻微回退：HellaSwag/PIQA/SIQA/Winogrande（常识广度任务受益于 L1 大覆盖）——**深度换广度**的权衡信号，tiered 不是全域无损。
- ⚠️ 注意：这是**从零训练**实验，不是 mid-training；可迁移的是机制（出场时机贡献独立增量），幅度不可直接比用户 1.8B。

## 4. 与本项目的系统关联

### 4.1 场景匹配度
- **同规模**：MiniCPM-1.2B（1.25B 非嵌）≈ 用户 1.5B
- **同范式**：Decay Verification = 已完成基座 + 混合续训 + LR 衰减，与用户 continual pre-training（ClimbLab 基座 + 1.8B 混合续训）结构同构
- **同域**：数学域是用户 STEM 主成分；L3 跨域迁移证据直接支撑 STEM 重仓策略
- **同数据生态**：与 ClimbLab/Nemotron-CC 同源竞品关系（#74），Puro-2B（#84）P2 阶段已用 UltraData-Math 118B

### 4.2 可迁移洞察（按价值排序）
1. **T1 - 中训练数据准入门槛应为 L2/L3 级**：框架显式指定 L2→Decay/MidTraining、L3→MidTraining；实证 L3 比纯过滤的 L1 在数学域 MATH500 +9.24pp。用户 STEM 池的建设标准：优先"精修/合成/教科书化"级数据源；Nemotron 系 4plus/3plus 质量档 ≈ L2 级（不够），UltraData-Math-L3（88B，Apache-2.0）可直接作为 STEM 池候选补充源——1.2B 上验证过的现成 L3 级数学数据。
2. **T1 - STEM 重仓的跨域收益证据**：Math-L3 同时提升 English（BBH +5.87）与 Code（MBPP +4.56）——用户 r=0.7 STEM 不只买 STEM accuracy，还买通用推理；这是对"STEM 挤占通用能力"担忧最直接的反证（但注意 3.3 的广度任务轻微回退信号，通用防护仍需监控）。
3. **T2 - 出场时机是配比之外的独立轴**：tiered vs mix 同配比不同顺序 +1.49pp、后段增长 1.7×——与 #84 Puro 质量升序课程、#28 LR Decay 同族。用户静态 70/30 协议**对比实验期间不可动**；正确用法是配比轴定论后的独立小实验（同 70/30 总量，STEM 池内部按质量分层、高质量后段出场，N=1 即可，赢了免费叠加）。尺度警告：UltraData 顺序效应在 120B 验证，用户 1.8B（66× 小）幅度无保证。
4. **T2 - 两级验证方法学模板**：Efficient（10B，快筛高方差）→ Decay（100B，定论）的层次设计，正是用户 400M proxy → 1.5B 主实验结构的工业版；细节可抄：30% 待验+70% 默认混合、WSD 基座、指数衰减终点 LR=稳定段 LR 的 1/20、μP。用户 proxy 线的协议设计有了同规模同范式的公开参照。
5. **T2 - L2 学习式分级与 PPL 分位过滤正交可叠加**：蒸馏分类器捕捉"有无解题步骤/讲解/学术性"（语义质量），PPL 分位捕捉难度/信息量（统计性质）——两个信号维度不同。但 ⚠️ 实证警示：L2 级筛选增益小（MATH500 +1.42、GSM8K -1.74）——**质量分级不是数学域的主增益来源，别高估过滤的价值；大头在格式重构**。
6. **T3 - L0 数学专用解析是数学数据第一道门槛**：公式保真度（MathML/KaTeX/AsciiMath→LaTeX 标准化）决定下游一切质量操作的原料品质；用户 STEM 池若含 HTML 来源，先检查公式解析质量再谈配比。

### 4.3 不可直接迁移
- **混合方向不同**：UltraData 30% 目标+70% 默认（验证协议，防遗忘主导）vs 用户 70% STEM+30% 通用（域适应主导）——目标占比差 2 倍以上，遗忘/干扰动力学不同，+9.24pp 不能当作用户 70% 配比的期望值
- **预算差**：100B/120B 验证 vs 用户 1.8B（55-66× 尺度差），层级效应与顺序效应幅度均无保证
- **L3 合成依赖 72B 级教师**（Qwen2.5-72B/GLM-4.5）：用户无此基建，只能复用现成 L3 数据集，不能自产
- **结论粒度**：全部 benchmark 均值，无 STEM accuracy 相变/涌现维度检验（#01/#88 视角缺席）
- MiniCPM3-4B tokenizer/Megatron/μP 栈与用户 nanochat 环境不同

## 5. 局限性（作者自述 + 本项目视角补充）
1. **L2 仅 preview**（33.7B，全量版本另仓待发）——分级器本身未随论文开源（Ultra-FineWeb en/zh 分类器开源但 Math 分类器未单列）
2. **N=1 per 配置**：无 run 级方差报告；efficient verification 作者自认方差大
3. **tiered vs mix 单点验证**：只在 1:1:1 层级配比 + 50/25/8/17 域分布下做了一次，无层级配比扫描
4. **mid-training 直接证据少**：框架主张 L2/L3 用于 mid-training，但实证主要是 decay verification（≈预训练尾段）与从零训练；真正的 mid-training 落地证据在 MiniCPM5-2B（#102）但配方未公开
5. **position paper 性质**（"Part I"）：体系化框架+实证，非严格受控科学；四范式叙事（符号/监督/自监督/反馈）是叙事框架不是发现
6. **无污染审计细节**：L3 合成用 Qwen/GLM 教师，评测 MATH500/GSM8K 对这些教师模型的训练分布存在潜在泄漏面（作者仅提及基准去重属 FineMath 流程）

## 6. 评估

**相关性：★★★★★**——91+ 篇中与用户场景匹配度最高的数据工作之一：规模重合（1.2B≈1.5B）、范式同构（decay verification=continual pre-training）、域直接命中（数学=STEM 主成分）、数据可直接取用（L3 88B Apache-2.0）、方法学可抄（两级验证协议）。核心增量：①"筛选不够、格式重构才是数学跃迁来源"（L2 vs L3 增益差）是对用户 PPL 过滤线的定位校准；②出场时机独立于配比的受控证据；③STEM 数据跨域迁移的 1.2B 实证。扣分点：mid-training 直接证据间接、N=1。

**交叉引用**：#84 Puro-2B（UltraData-Math 118B 使用者、质量升序课程同族）、#102 MiniCPM5-2B（框架的全栈落地、四 checkpoint diff）、#74 Nemotron-CLIMB（L0-L2 同生态竞品、4plus 质量档）、#88 Capacity-Dependent（L3 数据难度-容量警示：1.2B 验证≠1.5B 必然可迁移，高难度合成数据须过容量检验）、#99 SPARK（高难度 STEM 合成的反例警示）、#01 Phase Transitions（STEM 相变维度的缺席）、#28 LR Decay / #94 KD-During-Mid-Training（训练阶段依赖性同族）。
