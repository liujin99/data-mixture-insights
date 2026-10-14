# MiniCPM5-2B: 四 Checkpoint 全链开源的 2B 级 SOTA（含 Midtrain 单独发布）

- OpenBMB/面壁智能，2026.09.07 发布；MiniCPM5 系列第二弹（继 MiniCPM5-1B，同配方 scale up）
- HF/ModelScope：Final/SFT/Midtrain/Base 四 BF16 checkpoint + GGUF/MLX/GPTQ/DSpark 变体；训练语料全开源（UltraData 家族）
- 本文档基于官方 README/模型卡全文（训练配方、评测表、模型清单）

## 一句话总结

2B dense 达 2B 级开源 SOTA（avg 53.9，超过对比集内全部 4B 级，最高 51.1），其独一无二的价值是 **Base→Midtrain→SFT→Final 四 checkpoint 全链开源（公开生态首次 Midtrain checkpoint 单独发布）**——用户可零训练成本 diff 工业级 2B mid-training 的 STEM 增益，给 1.5B/1.8B 中训练的期望收益定标；配方是 UltraData 分层管理（#101）的全栈实践。

## 0. 模型与发布结构

**架构**：标准 `LlamaForCausalLM`，总参 2,516,756,480 / 非嵌 1,981,982,720（≈用户 1.5B 的 1.3×），42 层，GQA 16Q/2KV，上下文 131,072。

**Checkpoint 链**（全部 BF16 开源）：

| Checkpoint | 阶段 | 对用户的用途 |
|---|---|---|
| Base | 仅预训练（stable+decay 完成） | diff 起点 |
| **Midtrain** | mid-training 完成、SFT 前 | **diff 终点：工业级 2B mid-training 裸增益** |
| SFT | 400B deep-thinking SFT 完成、RL/OPD 前 | 观察增益兑现结构 |
| Final | RL+OPD 后 | 参照上限（含 post-training 增益） |

⚠️ **只有 2B 有 Midtrain checkpoint**；MiniCPM5-1B 仅 Final/SFT/Base。要 diff mid-training 就只有这一个公开对象。

## 1. 训练配方（三阶段，README 原文口径）

1. **Base training**：stable + decay（WSD 两段），建立核心语言能力与训练稳定性。
2. **Mid-training**："strengthen target capabilities and adapt to the target data distribution"（强化目标能力 + 适配目标数据分布）——官方对 mid-training 职能的一句话定义，与用户 continual pre-training 的目标结构（STEM 强化）同构。语料（全家族口径、不分阶段）：Ultra-FineWeb（L2 web）、Ultra-FineWeb-L3、UltraX（高质量 web）、UltraData-Code（L0-L3 分层代码）、UltraData-Math（L1-L3 分层数学）。
3. **Post-training** 三步：① SFT——400B tokens deep-thinking SFT（UltraData-SFT-2605 + UltraData-SFT-Agent-2609），建立深度思考与通用对话；② RL——按域（math/code/agentic/writing 等）训练专门化 RL 教师，critic-based 算法（JustRL II）；③ **OPD（On-Policy Distillation）**——把 RL 产出的 **16 个专家模型（含 5 个 agentic）合并回单一发布模型**：在每个 response 位置计算 student-teacher 全词表 reverse KL 作为 advantage 估计（替代 verification-based advantage），蒸馏数据直接复用各 RL 教师的训练 prompts（无需新建语料）。

**RL+OPD 增益**（官方口径）：推理+通用能力平均 **+10.96**，agentic +6.96。

**关键缺口**：mid-training 的 token 数、域配比、层级分配、LR 调度全部未公开；语料列表是"全家族不分阶段"口径。**diff 实验只能测 mid-training 总增益，不能归因到数据构成**——归因要回 #101 论文（decay verification 协议）找间接证据。

## 2. 评测（Final checkpoint，与 2B/4B 级对比）

avg **53.9**；2B 级对手：LFM2.5-2.6B 33.2 / Qwen3.5-2B 28.0 / Gemma-4-E2B-it 24.6；4B 级对手：Qwen3.5-4B 51.1 / granite-4.2-3B 42.7 / Nemotron-3-Nano-4B 32.6 / Gemma-4-E4B-it 31.2 / LFM2.5-8B-A1B 32.6。

| 域 | 代表成绩（Final） |
|---|---|
| Math | AIME'25 86.5（Qwen3.5-4B 78.8）、AIME'26 86.5（82.7）、HMMT Feb'26 63.8、MATH-500 94.6（99.0） |
| Code | LiveCodeBench v6 69.1、LCB-Pro Easy 68.0、SWE-bench Verified 46.4、SciCode 26.3 |
| General | MMLU-Pro 70.8、MMLU-Redux 84.7、GPQA-D 70.2、HLE 8.9、SuperGPQA 40.8 |
| Long Context | NoLiMa 68.1、AA-LCR 59.0、LongBenchPro 44.8 |
| IF/Agent | IFBench 66.3、IFEval 86.7、τ³-Bench Banking 20.8、GAIA Text-103 88.7 |

⚠️ 全部为 Final（RL+OPD 后）数字——RL+OPD 平均贡献 +10.96，**不能直接当 mid-training 的能力参照**；Midtrain checkpoint 无官方评测表，增益结构须自测（见 §3）。

## 3. 对用户的核心价值：Base vs Midtrain Diff 定标实验

> **⚠️ 用户决策（2026-09-11）：此 diff 实验不做**——工程成本（下载接入两个 5GB checkpoint + 适配评测管线）与收益不匹配。改为零成本静态定标：直接引用 #101 UltraData decay verification 数字（1.2B + 100B L3 数据 → MATH500 37.02、英数码全涨）作为"≈用户规模 + 50× 预算 + 顶层质量数据"的增益上限参照。以下实验设计仅存档备查。

**设计**：下载 MiniCPM5-2B-Base 与 MiniCPM5-2B-Midtrain（BF16 各约 5GB，HF 直连被墙须走镜像站），用用户现有 STEM accuracy 评测管线（nanochat 直评 accuracy，无需 SFT）同协议评测两个 checkpoint。

**输出与解读框架**：
1. **得到工业级 2B mid-training 的 STEM 增益量级 X pp**（math/code/长上下文适配的总效应，数据全开源家族喂出来的）。
2. 用户 1.5B/1.8B（70% STEM）中训练的合理期望应**显著低于 X**：预算差（MiniCPM mid-training token 数未公开，但 2B 模型 base+mid 全程语料为万亿级家族，用户 1.8B）+ 规模差（1.98B vs 1.5B 非嵌）。可结合 #101 decay verification（100B 续训、L3 vs L1 +9.24pp MATH500）校准量级感。
3. **若 diff 很小**：说明 mid-training 的收益主要在后续 SFT/RL 才兑现（能力"播种"假说，与 #94"数据价值依赖训练阶段"机制呼应）→ 直接影响用户对 1.8B 中训练的期望管理与评测协议设计（中训练终点测什么才不漏掉已播种未兑现的增益——accuracy 直测之外可加探针式指标）。
4. 可顺带 diff Midtrain→SFT→Final，观察增益兑现结构（预训练播种多少、post-training 收割多少）——公开生态唯一能做此分解的对象。

**成本**：纯推理评测，零训练成本；唯一工程量是两个 checkpoint 的下载与接入用户评测管线。

**局限**：diff 只测总增益不能归因（配方黑箱）；131k 上下文与用户短上下文协议差异（长上下文适配收益会在 diff 中占比不明）；tokenizer/架构与用户栈不同，绝对分不可比，**只取增益量级（Δpp）不取绝对分**。

## 4. 与本项目的系统关联

### 4.1 场景匹配度
- **同规模区**：非嵌 1.98B ≈ 用户 1.5B（1.3×）
- **同阶段定义**：mid-training 职能=强化目标能力+适配目标分布，与用户 continual pre-training（STEM 强化）同构；base（WSD）→ mid → post 三段式与用户"基座已定、中训练独立"结构对应
- **同数据生态**：UltraData 家族（#101）与 ClimbLab/Nemotron（#74）同源竞品；#84 Puro-2B 已验证 UltraData-Math 在 2B 级的使用
- **同目标域**：math/code 是 mid-training 强化主目标，与用户 STEM 主成分重合

### 4.2 可迁移洞察（按价值排序）
1. **T1 - Diff 定标实验**（§3 详述）：公开生态唯一可精确测量"工业级 2B mid-training 前后 STEM accuracy 增益"的机会；零训练成本，一次评测换来用户 1.8B 中训练的期望收益锚点。
2. **T2 - mid-training 作为独立阶段的工业确认**：OpenBMB 把 Midtrain 单独 checkpoint 化发布，与 #101 框架（L2/L3 指派给 mid-training）互相印证——中训练取高层级数据、独立成阶段是头部团队的工程共识，不是学术虚构。
3. **T2 - mid-training 语料构成参照**：目标域（UltraData-Math/Code，L2/L3 级）+ 通用高质量（Ultra-FineWeb/UltraX，L2 级）混合——与用户"70% STEM 高质量 + 30% 通用高质量"结构同族；间接支持用户 STEM 池取 L2/L3 级、通用池取 L2 级的分层取数策略。
4. **T3 - 评测定标时的扣减意识**：官方公布的 AIME 86.5/MATH-500 94.6 含 RL+OPD（+10.96 avg）；任何用 MiniCPM5 数字给自己定标的场合都必须先扣 post-training 增益，否则严重高估中训练收益。

### 4.3 不可直接迁移
- **RL+OPD 全流程**：用户无 post-training 阶段；OPD 的 reverse-KL 优势估计也与用户 NTP 训练无关
- **131k 长上下文**：mid-training 含长上下文适配成分，用户短上下文场景 diff 会高估"纯 STEM 域适配"增益
- **配方黑箱**：mid-training token 数/配比/分层分配未公开——不能直接抄配方，只能测总效应
- **架构/ tokenizer**：LlamaForCausalLM+131k RoPE 配置与用户栈不同，绝对分不可比

## 5. 局限性（发布口径 + 本项目视角补充）
1. **mid-training 细节全黑箱**（token 数、域配比、层级分配、LR）——最大限制；科学归因只能回 #101 论文间接推
2. **Midtrain checkpoint 无官方评测表**：全部公开数字是 Final；Base/Midtrain/SFT 的能力结构须自测（这正是 §3 实验的存在意义）
3. **评测口径混合**：部分分数（†标记）来自 Artificial Analysis 官方发布，其余内部复现——跨行可比性有噪声
4. **工程发布而非论文**：无受控实验、无方差报告；"2B 级 SOTA"限定于其自选对比集
5. **SFT/RL/OPD 语料与 mid-training 语料的边界未公开**：UltraData-SFT-2605/RL-2609 是 post-training 口径，不能反推 mid-training 用了什么

## 6. 评估

**相关性：★★★★☆**——作为"可测量的工业级 mid-training 参照物"独一无二（四 checkpoint 链 + 全开源语料 + 1.98B 规模重合），diff 定标实验是用户期望管理的关键拼图；扣一星因 mid-training 配方黑箱（只能测总增益不能归因）与长上下文成分混杂。与 #101 互补：UltraData 给机制与协议（为什么 L3、怎么验证），MiniCPM5 给总量级（工业级 2B mid-training 到底涨多少）。

**交叉引用**：#101 UltraData（数据底座+分层框架+验证协议）、#84 Puro-2B（UltraData-Math 118B 的 2B 级使用先例、post-training 存活实验范式）、#74 Nemotron-CLIMB（同生态竞品）、#88 Capacity-Dependent（diff 结果解读时的容量维度：2B 的增益不保证 1.5B 等比）、#94 KD During Mid-Training（训练阶段依赖性机制）、#98 OpenEuroLLM Scaling Laws（overtraining regime 增益量级参照）。
