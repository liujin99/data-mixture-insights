# GLM-5 深度分析（canon 席位 3 深读升级）

> arXiv 2602.15763，2026.2，智谱 AI + 清华。744B/40B MoE（GLM-4.5 的 2×：355B/32B → 744B/40B），256 experts top-8、80 层、MLA+Muon Split、MTP（accept 2.76 > DSv3.2 的 2.55）。
> 总表 #58、核心参考 canon 席位 3。2026-11-13 深读（用户指令"确保 canon 优中选优"；底档 /tmp/opencode/glm5.html，703KB）。

## 0. 核心结论（一段话）

28.5T = **PT 27T（4K 短上下文）+ MT 1.55T（32K/1T → 128K/500B → 200K/50B 三阶段窗口课程）+ DSA 20B（MT 后稀疏注意力适配）**。与 NVIDIA 三明治（#160/#161：PT 8K → MT 256K）**独立同构**：短上下文 PT + MT 扩窗装 agentic/长上下文数据——中训练阶段结构跨厂商成立的又一证。关键差异三条：**MT 学习率=从 PT 终值连续线性衰减（4e-5→1e-5），非 NVIDIA 的重置恒定**（#145 两案的跨厂商实证）；**无 PT 回放披露**（NVIDIA 消融支撑 20% 回放必要，GLM 缺位=活分歧样本）；**主动打榜**（自报 AA.ai/LMArena Text/Code SOTA）vs NVIDIA 从不提交——两家第三方指数的可信度用法截然不同。披露面：token 量全给、比例无、消融无（MT 数据侧）、附录超参全披露。

## 1. 结构与规模

- PT 27T"prioritizing code and reasoning early on"（代码与推理早进场——#156 SPT 的 PT 内变体）；4K 上下文。
- MT 三阶段（§2.3）：32K（1T）→ 128K（500B）→ 200K（50B）；后期阶段上采样长文档+合成 agent 轨迹。
- DSA（§2.1.3，MT 后）：warmup 1000 步（14×202,752 tokens/步，LR 5e-3→2e-4）+ 稀疏适配 20B（恒定 1e-5，**复用 MT 训练数据与超参**）——追平 MLA，预算仅 DSv3.2（943.7B）的 **1/47**。

## 2. 预训练数据（§2.2，27T）——管线级披露、无比例

- Web：GLM-4.5 管线 + 新 DCLM sentence-embedding 分类器（标准分类器之外捞高质量）+ World Knowledge 分类器（Wikipedia+LLM 标注优化，从**中低质量数据里蒸馏长尾知识**——"低质源里的知识"路线，连 #101 L2/L3 分层）。
- Code：快照刷新+代码网页（去重后 unique token +28%）、Software Heritage 元数据修复、低资源语言专用分类器（Scala/Swift/Lua）。
- Math & Science：LLM 打分只留最高教育价值内容、长文档 chunk-and-aggregate 打分；**严格排除合成/AI 生成/模板化数据**（与 #126 wild-AI 轴对照的自然数据保守立场）。
- 无任何比例百分比披露。

## 3. 中训练（§2.3）——窗口课程为组织轴

| 阶段 | 窗口 | tokens | 内容侧重 |
|---|---|---|---|
| Stage 1 | 32K | 1T | 基础长上下文 |
| Stage 2 | 128K | 500B | 长文档+合成 agent 轨迹上采样 |
| Stage 3 | 200K | 50B | +MRCR-like 数据（OpenAI 范式多变体，多轮对话回忆） |

- **SWE 数据（~160B unique tokens）**：repo 级文件+commit diffs+issues+PR+相关源文件拼统一序列；**放宽 repo 级过滤扩池（→10M issue-PR 对）+ 收紧单 issue 级质量过滤**——"池宽 vs 质紧"的旋钮式披露（连 #111 精炼池质量轴）。
- **LC 数据**：自然（书籍/论文/通用语料，PPL+去重+长度多级过滤，知识密集域上采样）+ 合成（NextLong/EntropyLong 启发，高相似文本 **interleaved packing** 拼 序列抗 lost-in-the-middle）。
- 声明级发现两条（**无 ablation 表支撑**）：① 数据多样性渐进提升 LC 性能；② **"200K 阶段还增强了 128K 窗口内性能"**（跨窗口溢出主张——弱旁证=DSA 段 Table 6：150B 联训后 16K/32K/64K 全超 MLA 基线 +0.49~+1.72、仅 128K −0.35，"后段短训反超短窗"同族现象，但非直接证据）。

## 4. 训练协议（附录全披露）——LR 营地对立，canon 关键新证据

- **PT**：warmup 0→2e-4，衰减至 **4e-5**（PT 段末）。
- **MT**：**从 4e-5 线性衰减至 1e-5**——从 PT 终值连续衰减、无重置。
- **DSA**：warmup 5e-3→2e-4；稀疏适配恒定 1e-5。
- **三家 LR×数据调度对照**（canon 内直接可比）：

| 营地 | 代表 | 协议 | 出处 |
|---|---|---|---|
| 重置恒定 | NVIDIA #160/#161 | MT 重置（2× PT 终值/分档值）warmup→恒定 | #161 §2；#160 LR 重置 |
| 延迟衰减 | #145（清华 PACMAN） | LR 在 MT 进场后才开始衰减，只盖 MT 末段 | #145 19 sweep 一致 |
| **连续衰减** | **GLM-5** | **从 PT 终值线性衰减贯穿整个 MT** | 本文附录 |

- 注：GLM-5 用 Muon 优化器（非 Adam），LR 营地差异可能部分系优化器/谱系耦合（#133 引用纪律：跨谱系照抄前自验）；智谱谱系自 GLM-4.5 连续两代沿用此协议。

## 5. DSA 段量化表（唯一 ablation 级数据，9B 级验证）

- Table 6（GLM-4.7-Flash+DSA，RULER 4K-128K）：warmup-only 已保绝大部分（128K 79.21→71.35）；**+150B 联训后 16K +0.86 / 32K +0.49 / 64K +1.72 / 128K −0.35**。
- 效率注意力横评（Table 5）：朴素 SWA 交错 RULER@128K **−30.35**（灾难级）；搜索式层选择（PostNAS 启发、16K 上搜索一次全长度泛化）大幅收窄；线性注意力（GDN）最优也要 −5.69 RULER@128K；**DSA 无损（构造上不丢长程依赖）**——"上下文效率该在注意力结构里买还是数据里买"的 GLM 答案：结构里买。
- Muon Split ablation（Table 1）：MLA+Muon 下 576 维 latent KV 不敌 GQA-8 → 拆头矩阵正交化后追平（Hellaswag/MMLU/C-Eval 等 7 项）——优化器×注意力耦合的工程证据。

## 6. Post-training（GLM 版"SFT/RL 层"，三明治下游对照）

- SFT：multi-task + **Interleaved Thinking**（每次响应/工具调用前思考）+ **Preserved Thinking**（coding agent 跨轮保留全部 thinking 块、复用已有推理不重推——长程任务省算）。
- **错误段保留但 loss-mask**（学纠错不强化错误动作）——与 #161"只训 assistant turn"同哲学的跨厂商平行（GLM 用于 SFT 轨迹，NVIDIA 用于 MT 轨迹）。
- RL 序列：Reasoning RL → Agentic RL → General RL；GRPO+IcePop（显式区分 train/inference policy，**去掉 KL 项加速**）。
- **On-Policy Cross-Stage Distillation（OPCD）**：最终阶段用**前序各阶段 ckpt 当教师**（prompt 从对应教师 RL 训练集采样、按比例混合），在策略蒸馏恢复早期技能防遗忘——与 3 Ultra MOPD 同年平行实现（多教师在线蒸馏合流；差异：GLM 教师是自家历史 ckpt，NVIDIA 教师含别家开源模型）。
- 难度过滤：只留"对 GLM-4.7 也难"的题（自家上一代当难度门——#88"学生模型打分"的工业版）。
- 评测姿态：**主动打榜**——自报 AA Intelligence Index v4.0/LMArena Text/Code SOTA；Table 7 对比集含闭源（Claude Opus 4.5/Gemini 3 Pro/GPT-5.2 xhigh）；自报 SWE-V 77.8 / SWE-Multi 73.3 / TB 2.0 双 harness（Terminus-2 56.2/60.7†；Claude Code 56.2/61.1†）/ LongBench v2 64.5（仅次 Gemini 3 Pro 68.2）；平均比 GLM-4.7 +20%。**canon 用法注记**：GLM 的 AA 高分=主动对齐口径的产物，与 NVIDIA 被动低分（§7.6 伪影）不可同日而语——第三方指数只在"模型主动参加的那场考试"里可比。

## 7. 与本库连点

- **#161/#160（三明治跨厂商同构）**：PT 4K→MT 扩窗+SWE 数据，与 NVIDIA PT 8K→MT 256K 独立同构——中训练阶段结构是行业共识而非一家做法。
- **#145/#137（LR×数据调度）**：连续衰减营地的工业实证（§4 对照表）。
- **#156 SPT（域数据早进场）**：27T PT"prioritizing code and reasoning early on"+ #161 future work"推理/聊天挪进 MT"——同一"早进场"光谱的 PT/MT 两档。
- **#101 UltraData（低质源里的知识）**：World Knowledge 分类器从中低质量数据蒸馏长尾——L2/L3 思路的厂商版。
- **#88（难度=学生配对属性）**：GLM-4.7 当难度门。
- **#112/#84（中训练收益 SFT 后再测）**：GLM 未披露桥协议——其 MT 收益主张全部无 SFT 桥隔离（TR 只报终模型对别家成绩）。
- **3 Ultra MOPD ↔ 本文 OPCD**：多教师在线蒸馏防阶段遗忘的两家平行实现。
- **#133（配方×base）**：LR 营地差异可能系 Muon/谱系耦合——跨谱系引用先自验。

## 8. 席位裁定（canon 守门，2026-11-13）

**深读前六问六答**（2026-11-12 断网期间所列判据，逐条兑现）：

| 问 | 答 |
|---|---|
| ① 三阶段课程有 ablation 表？ | **无**——仅声明级（多样性→LC 性能；200K 溢出） |
| ② 除 token 量外有比例百分比？ | **无**（1T/500B/50B、160B、27T/28.5T 为全部数字） |
| ③ PT 回放？ | **无披露**——vs NVIDIA 20%（"40% 无增益"消融支撑）：**活分歧样本** |
| ④ MT LR 协议？ | **全披露**——连续衰减 4e-5→1e-5，营地对立（§4），canon 级新证据 |
| ⑤ annealing/收尾质量上采样？ | **无披露** |
| ⑥ 200K 溢出有支撑数字？ | **无表**——DSA 段 RULER 反超现象仅弱旁证 |

**裁定：席位保留，角色精确化**——GLM-5 在 canon 里不是"NVIDIA 级配方参考答案"（无比例/无消融/无回放披露），而是四件别家给不了的东西：**跨厂商结构见证**（唯一专门 MT 章节+三明治同构）、**LR 营地另一极**（连续衰减全披露）、**回放分歧活样本**、**打榜姿态对照**（AA 口径的两种用法）。canon 内互补性成立，不与 NVIDIA 两席冗余；无需换席（SuperValid/InfoLaw 仍留第二梯队）。

**引用清单**：窗口课程结构（32K/1T→128K/500B→200K/50B）/ LR 全套数字（PT 2e-4→4e-5；MT 4e-5→1e-5）/ DSA 效率账（20B 追平 MLA，1/47 DSv3.2）/ SWE 池-质旋钮（放宽 repo 收紧 issue）/ 错误段 mask / OPCD。**不引**：配比数值（无）、200K 溢出主张（无表）、自报分数当目标线（AA 系主动对齐口径）。

## 9. 局限与触发器

- MT 数据侧零消融、零比例；200K 溢出断言级；DSA 表在 9B（GLM-4.7-Flash）非 744B 本体；单 run；post-training 数据配比未披露。
- 触发器：① GLM-4.5 TR 回溯（中训练框架出处，看协议是否 4.5 已定）；② GLM 下一代 TR 的 MT 章节演进（是否补比例/消融/回放）；③ 若 GLM 放出中训练数据细节（目前无开源声明）→ 复核。

---
*深读底档：/tmp/opencode/glm5.html（2026-11-13 抓取，703KB）。头部厂商 doc §1 为 2026-08 批量分析，本文为 canon 级深读升级，§1 评级表维持。*
