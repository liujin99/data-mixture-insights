# Singhal《Priming Base Models for Agentic Post-Training: An Open Midtraining Recipe》深度分析

> arXiv 2610.22019，2026-10-15，19 页，NVIDIA（Singhal/Jennings/Shum/Bukharin/Zirui Liu/Kong/Kliegl/Delalleau/Satheesh/Patwary 等 14 人）。
> 总表 #161（伴生 TR #160 Nemotron 3.5）。2026-11-11 深读（用户裁定"潜在高价值，好好研究"）。
> 落点：**3B-A1 中训练数据设计的头号直接参照**——业界第一份带消盲级消融细节的 agentic MT 配方。

## 0. 核心结论（一段话）

三明治中间层的完整工业配方：**PT（8K 短上下文、无 agent 数据、无 loss-mask）→ MT（256K、数百 B token、自然 SWE 数据为主 + 10-15% agent 轨迹）→ 短 SFT（合成数据）→ RL**。MT 的存在意义 = 把"自然 SWE 数据 + 少量轨迹"的学习从 SFT 挪出来（超长 SFT 会伤 RL 可训练性，引 Liu 2026b/Kang 2025/Zhang 2026），让 SFT 保持短。**评测协议不靠 base 模型代理指标，靠同款 SFT 桥**——因为他们实测 BPB/LoCoDiff 等 base 代理"与 post-training 结果不可靠相关"。两个尺寸验证：Lightning-30B-A3B（500B token）+ Super-120B-A12B（280B token）。

## 1. 配方全表（Table 1，500B run 口径）

| 来源 | 唯一 token | 混合占比 | 实际 token | epoch |
|---|---|---|---|---|
| Issue-PR | 216.9B | 40.8% | 205.2B | 0.95 |
| 仓库 | 104.1B | 20.0% | 100.7B | 0.97 |
| Agent 轨迹 | 39.4B | 10.0% | 50.3B | **1.28** |
| Diff 数据 | 16.0B | 3.0% | 15.1B | 0.95 |
| 构建/CI 日志 | 6.5B | 1.2% | 6.1B | 0.95 |
| **PT 回放** | ~2.0T 池 | 20.0% | 100.7B | 0.05 |
| 长上下文数据 | 67.4B | 5.0% | 25.2B | 0.37 |

- 全源纪律：仅 permissive 许可证；对 SWE-bench/Terminal-Bench 去污染；密钥脱敏；重复分隔符/哈希/时间戳 loss-mask。
- 混合总量 ~380B 唯一 token（500B run 基本单 epoch）。
- 轨迹占比封顶 10-15%：全是轨迹 ≈ 变相加长 SFT → 伤 RL 可训练性。

### 1.1 Issue-PR（占比最大，40.8%）
- 合并 PR 构建：仓库快照 + PR 描述 + 关联 issue + review 讨论 + patch 演化；≥10 星仓库；按 repo+PR id 去重；12.1M PR / 200B-2T token（按上下文包含量浮动）。
- **LLM judge 检 issue-PR 匹配**：7 类失败模式（revert/偶然链接/错误引用/部分修复/issue 无行动/欠规格/scope 超出）→ 修复策略（重链接/缩范围/降级 PR-only/拆分）；PR-only 用确定性规则（剔除 README-only 或 ≤3 行）。
- 布局：仓库元数据、日期、文件树、代码上下文 → 条件化重建提交变更；无 chat 模板。

### 1.2 仓库（20.0%）
- 选库：Nemotron 3 PT 语料内的仓库、14 语言、**import 图连通率 ≥20%（Ruby 10%——Rails autoload 遮蔽静态解析）**；排除 Shell/R/Swift/标记语言。
- 序列化：README 先行 → import 图深度优先 → 相连文件相邻；超长仓库按连通分量切分而非截断。
- **依赖排序 vs 字母序的分选效应**：强模型 BPB 降、弱模型升——暴露弱模型尚未学会的跨文件结构（难度校准观察，连库内"排序即难度"线）。
- 语言配比：自然份额起步，**upweight Python（23.5→31%）、downweight Dart/Scala/Lua**；7.38M 记录/104.12B token。
- 过滤：路径内去重、Python 2 整库剔除、服务端密钥弃库/客户端脱敏、剥许可证与生成头、>2M token 弃库；去污染剔 SWE-bench 仓库+fork+HumanEval/MBPP/LiveCodeBench 解。

### 1.3 Diff 数据（3.0%）
- 两型：**multi-file edits**（733,705 例：改前文件 → 预测改后；只保留 import 图相连文件的改动=改动手臂耦合）+ **code-evolution 文档**（377,618 例：初态 + commit 序列（旧→新，中位 3 个、上限 59）+ 终态；训练长程状态跟踪，即 LoCoDiff 所测）。
- 纪律：patch 必须 byte-for-byte 可重放；去污染剔 SWE-V/Pro/Multi-SWE/TB/LoCoDiff 全部用库；剔 GPL/密钥/lockfile/生成文件。loss 覆盖：原文件+diff+改动行（两行上下文）；commit 哈希/作者/日期 mask。

### 1.4 构建/CI 日志（1.2%）
- LogHub 全档案（BGL/Thunderbird/HPC/HDFS/Spark/Windows）：120 行上下文窗 + 16 行目标窗；HDFS 在 mask 块 id/主机/时间戳后去重；单档案 ≤28%、超算日志 ≥30%、CI/CD（GHALogs）≥12%。
- 632 例诊断样本：以警告/错误/严重事件为中心开窗，前沿模型写诊断。

### 1.5 Agent 轨迹（10.0%，39.4B）
- >1M 条轨迹、~500K 去重 query、30M 工具调用；含 SWE-Zero/SWE-Hero 轨迹（**Ludwig et al. 2026 = NVIDIA 内部〔Ahmad/Majumdar/Ginsburg〕，非 Marin，2026-11-11 引文核实**）；Perplexity（Nemotron Coalition）贡献一部分。
- **12 个 scaffold**（codex 19.4%/openhands 14%/generic_code_cli 12.5%/gemini_cli 12.2%/terminus 11.6%/taubench_react 5.9%/swe_agent 5.4%/opencode 5.2%/mini_swe_agent 4.1%/search_browse 3%/chat 2.4%/claude_code 1.3%）——**单一 scaffold 轨迹迁移差**（引 Qwen3-Coder-Next）。
- 过滤针对：捷径（Git 历史/联网查答案泄漏）、未完成轨迹、畸形工具调用、重复文件巡检循环（快照 8.9 万条剔 8,935 ≈1%：未完成 6,960 + 循环失败 1,959）。
- 渲染于 Nemotron chat 模板；**只训 assistant turn（含工具调用）**。
- 轨迹语料已扩至 ~90B token（future work）。

## 2. 训练协议（§3.1 + Table 17）

- 256K 上下文；从收敛 base 出发。Lightning 早期实验从 8K ckpt 起；正式 run + Super 从 1M LC ckpt 起（"成熟管线 MT 应在最终 LC 段**之前**"——顺序注记）。
- **LR：warmup 后恒定**——Lightning 5e-5（偏 agentic 增益）vs Super 1e-5（偏保通用）；**按模型尺寸分档**。
- PT 回放 20%（默认值；40% 无增量收益）；MTP 双层保留（loss 权重 0.1/0.3）；inter-doc attention mask；Megatron-LM @GB300；Adam β=(0.9,0.95) wd 0.1 clip 1.0 BF16；batch 64 序列。

## 3. 评测协议（§4.1）——本库最关心的一节

- **SFT 桥**：同款 agentic SFT 两档（短 33B / 全量 100B），MT vs base 各做一次再比——**短桥结论可外推到全桥**（Table 2 双档验证）。
- **原话**："base-model proxy metrics such as bits-per-byte and LoCoDiff **did not reliably predict post-training results**"——四个 base 可训练性代理（LoCoDiff/RepoBench/仓库 BPB/**轨迹决定性 turn BPB**）全随 MT 单调改善，但"**不足以指导配方决策**"（跟踪≠排序）。
- 轨迹 BPB 用 DeepSWE 决定性 turn（对结果影响最大的 turn）= **hinted/conditional NLL 家族（连 #147）**——四代理中与 post-training 相关最好的一个，仍不够。
- pass@k 一并报告：**pass@k 高 = RL 提升空间大**（引 Yue 2025）。
- 回归护栏：MT 全程跟 PT 通用 benchmark 四组（知识/数学/代码/常识）。
- RL 持久性（附录 D）：20 步 GRPO×14 可验证环境；**配对式 prompt 过滤**（对两个 ckpt 都非全对非全错才保留）；MT 优势 SWE-V ~2 分全程保持；TB pass@8 base 46.1→39.3 而 MT 持平 45-46；单 seed，只能证"优势不在 RL 起步期消失"。

## 4. 主结果

- **Lightning-LC 500B（全量 SFT 后）**：SWE-V 60.7→64.5（100B 已拿大头）、SWE-Multi 55.8→62.8（涨到 500B 不饱和）、Pro 32.0→35.5、TB 噪声大 300B 峰值；pass@k 四项全涨 2.3-4.7——"提的是会解的上限，不只是稳定性"。
- **Super 280B**：DeepSWE（最难、从零写长程任务）11.7→15.7（短 SFT 6.4→12.4）；SWE-V 72.9→74.3；BrowseComp +3；SWE-Multi/Pro 全量 SFT 后 -1.1/-2.1（作者判为噪声内不一致效应）。
- **Harness 稳健性（§4.5）**：全 harness 胜（Super 短 SFT 平均 +8.6 SWE-V / +6.8 TB；Lightning 全 SFT 六 harness 平均 +11.6）；**base 最弱的 harness 增益最大**（Codex +13.4/+19.0）；无轨迹 MT 已在 4/6 harness 胜 base（最高 +4.7）。
- 上下文经济性（E.1）：256K 塞满失败率 SWE-Pro 4.1%→0.6%；200-turn 上限触顶 11 次→1 次——"MT 让 agent 更省上下文和 turn，不只是更准"。
- LoCoDiff：base 逐字节重建 0.0% → 100B 21.5% → 500B 26.8%；**10B 受控对比：无 diff 数据 1.3% vs 50% diff 29.8%**——diff 数据驱动该指标。

## 5. 消融五连（§5 + 附录）

1. **MT 即半个 LC 段（§5.1）**：MT@256K 使 8K 模型 RULER 逼近 LC base 至 128K（差 ~5 分），256K 差 14；**MT→LC 顺序在 64K-256K 全长度超直上 LC**；E.2：Super MT 后 RULER-1M 79.9→33.0，~9B 纯 LC 段恢复至 81.0（超过起点）——TR #160 的"恢复段"出处。
2. **工具响应 mask（§5.2）**：不 mask 27.5 vs mask 37.8（SWE-V，差 ~10 分）——"预测环境反应"的世界建模路线（FAIR CodeGen/daVinci-Dev）**开箱即翻车**；长而噪的响应（文件转储/测试日志/报错栈）抢容量。
3. **回放 + LR（§5.3）**：20% 回放下 Super@1e-5 走 65B 四组通用全在 1.2 分内（Table 10；**论文无 Super 高 LR 数据**）；40% 无增益。**LR 三档扫描在 Lightning-8K 上做（Table 11）**：1e-5 通用几乎无伤（MATH-500 78.8≈base 79.7）但"新东西学不进"（SFT 后增益小）、3e-5 −5.4、5e-5 −18 分（79.7→61.7）——**Lightning 生产仍选 5e-5 = 产品定位上故意拿 18 分数学换 agentic 增益**（专才线），Super 旗舰选 1e-5 保通用。即"取舍方向分档"的准确含义 = **产品定位决定站激进-保守光谱哪端**（非尺寸定律；尺寸效应——同 LR 对大模型等效更猛、μP ~1/宽度——系通识推论，论文未测 Super 高 LR）。建议：温和上调 + 调 LR 时全程盯四组通用 benchmark 当护栏（摔过线退档）。
4. **轨迹是增益主carry但封顶**：无轨迹 MT 全量 SFT 后 SWE-V <1 分增益；全配方 +3.0/+2.5/+6.6——**post-SFT 增益大头来自轨迹**；但自然数据的收益体现在 LoCoDiff/RepoBench/harness 稳健性（不在 pass@1 的那条路）。⚠️ **2026-11-11 用户质询复核（"很经验主义"——成立）**：10-15% 封顶**无剂量-响应扫描支撑**——证据仅为二元消融（有/无轨迹）+外部引证（Liu 2026b/Kang 2025/Zhang 2026"超长 SFT 伤 RL"）+机制论证（轨迹=对话格式=SFT 型数据，喂多≈提前做长 SFT、行为提前定型、RL 没得塑形）；对照 #160 Indic CPT 的 10/20/30% 扫描（自然域轴有真扫描、**轨迹轴没有**）。对 3B-A1：轨迹占比应列为 r-scan 轴自行定值（本文值作先验）；其后续动作（轨迹语料 39.4B→90B 扩容）指向旋钮是"多样性"而非"占比"——两个独立旋钮。
5. **RL 起步持久性（附录 D）**：见 §3——优势不消失即可，单 seed 20 步不承诺 RL 增量。

## 6. 与本库连点

- **#113（loss 最优≠迁移最优）的工业级证词**：四代理单调改善却"不足以指导配方决策"——三轨制①（loss/代理=装填信号）与③（SFT 桥=裁决）分层的直接工业实证；短桥（33B）→全桥（100B）外推性也被验证（桥量程可缩小的背书）。
- **#147（hinted 工具/条件 NLL）精确定位**：他们的"决定性 turn BPB"= 同家族，四代理中相关最好、仍不够——hinted 线的工业位置：最好的 base 代理，还不是裁决。
- **#112/#84（SFT 桥翻案线）的完整工业版**：配对桥 + 回归护栏 + RL 起步检查三层全齐。
- **#145（LR 锚）**：warmup→恒定（非衰减），且**按模型尺寸分档**（30B-A3B 5e-5 / 120B-A12B 1e-5）——与 #145 衰减锚构成对照点对；"PT 以小衰减 LR 结束，MT 需更高 LR 学新数据"同句给出。
- **#133/#159（配方×base 交互）**：他们只测 Nemotron 家（自认局限）；同期 OPMD 的 Llama↑/Qwen↓ 现象互补——配方跨 base 迁移仍是开放风险。
- **#73/#74（研究线/产品线分裂的实证，2026-11-11 用户质询引出）**：NVIDIA 内部同构"两个世界"——研究线发 CLIMB 搜索方法（62M-1B proxy、40B CPT、112 次搜索 ≈5040 GPUh），产品线配比全是手设计（Nemotron 3 Ultra 19 类两阶段课程 TR 通篇无一句称配比来自 CLIMB；本文 MT 配比明说手调）；**Nemotron-CC 语料资产进生产**（Nemotron-CC-Code 为 PT 类别、Indic 语料译自其英文源）**而 CLIMB 搜索方法未见于自家生产**——资产被生态消费（Puro-2B/AutoData/nanochat/用户基座），方法停在研究演示。原因结构：proxy 排名跨尺度不可信（#73）+ SFT 桥裁决太贵付不起搜索 → 生产配方=手设计。即"loss 搜索失效的地盘=手调+桥消融的地盘"在同一公司内部成立。
- **#156 SPT/#84 课程线的反向注记**：future work "把更多 post-training 数据（推理/聊天）挪进 MT"——SPT"域数据尽早进场"的 MT 版延伸。
- **#123/#125 重复上限**：轨迹 39.4B 唯一 token 跑 1.28 epoch（>1）——小语料多 epoch 的重复定价问题的又一处工业数据点。
- **#160**：TR="是什么"，本文="怎么来的+为什么"（消盲级消融出处）。

## 7. 局限与风险

- 单 run 对比（1-2 分噪声、TB 尤甚）；无"同 token 换算成更长 PT/SFT"对照（作者自认：MT 是补 SFT 不是替 SFT）。
- 仅 Nemotron 家、仅 agentic coding；SFT 桥为 agentic 聚焦混合（非通用 post-training 混合）。
- **准则 4 未兑现**：数据处理脚本与训练集元数据"未来版本放出"（截至 2026-11-11 未放出）——新文期 <6 个月只计承诺不计兑现，**兑现日为 watch 触发器**。
- RL 证据单 seed 20 步（自认局限）；SWE-Multi/Pro 在 Super 上全量 SFT 后小幅负向（判噪声）。

## 8. 3B-A1 草图直接引用清单（并入实验设计）

1. 配比结构：域数据 ~75% + PT 回放 20% + LC 数据 5%；轨迹类 ≤10-15% 封顶。
2. 轨迹处理：多 scaffold 来源、只训 assistant turn、捷径/未完成/循环三类过滤、chat 模板渲染。
3. 训练协议：LR warmup→恒定（**分档=产品定位取舍**：专才小模型 5e-5 故意吃通用损失换 agentic 增益 / 旗舰大模型 1e-5 保通用；+ #145 衰减锚两案对照）；回放 20%（40% 无增益）；MTP 层保留。机制注记（2026-11-11 用户质询二轮修正）：**论文 LR 扫描只在 Lightning-8K**（Table 11：1e-5 无伤但增益小/3e-5 −5.4/5e-5 −18 MATH），无 Super 高 LR 数据点；尺寸效应（同 LR 对大模型等效更猛、μP ~1/宽度）系通识推论；损伤机制=大步子×新旧分布差的灾难性遗忘（20% 回放只部分顶住）；论文只报现象、机制为通行解释。
4. 轨迹占比**不定死 10-15%**：列为 3B-A1 r-scan 轴（固定预算扫 5/10/20%），本文值作先验——原文仅二元消融无剂量扫描（见 §5.4 ⚠️）。
5. 评测：短 SFT 桥（可外推）+ 四组通用回归护栏 + pass@k 作 RL 余量 + RL 起步持久性检查（配对 prompt 过滤设计可抄）。
6. 顺序：MT 在最终 LC 段之前（或从 LC ckpt 起步 + 事后恢复段）。
7. 效应读法：域内 pass@1 可能不动，增益显形处 = 最难 benchmark + 跨 harness + 上下文经济性。

### 8.1 可迁移性分界表（2026-11-12 用户质询"很多优化和经验是跟 SWE 数据相关的"——成立）

| SWE/agent 域绑定（不直接迁移） | 域无关（3B-A1 数学 MT 直接有效） |
|---|---|
| Issue-PR/仓库/diff/日志构建细节（import 图、LLM judge 七类失败、LogHub 切窗） | 三明治定位（阶段结构判断） |
| 轨迹数据处理（scaffold 多样性、捷径/循环过滤、chat 模板渲染） | PT 回放 20% + 四组回归护栏（40% 无增益；防遗忘机制与域无关） |
| 工具响应 mask（数学数据无"工具返回"；同构问题=CoT 数据该对哪些 token 算 loss，本文未测） | LR 协议（warmup→恒定、护栏下调档、按产品定位分档） |
| 跨 harness 稳健性作为主效应（数学 benchmark 无 harness 概念；弱对应物=prompt 格式/答案抽取器） | MT→LC 顺序 + 恢复段（阶段结构与域无关） |
| LoCoDiff/RepoBench 等 base 代理（SWE 特有指标） | SFT 桥评测协议（短桥外推性验证与域无关——**最值钱**） |
| 上下文经济性（256K 塞满率，长上下文 agent 工作流特有） | ≤10-15% 对话式数据原则（数学 MT 掺 CoT/对话式讲解数据的占比问题同构；见 §5.4 r-scan 轴） |
| | 去污染/脱敏/无规律内容 mask 纪律 |
| | 短 SFT 保 RL 可训练性（数学 RL 主流，同样适用） |

- **SWE 绑定半边的数学域对应物库内已有**：#158 Nemotron-CC-Math 语料构建（超 FineMath-4+ 的管线）、#101 UltraData L3 格式重构、合成数学数据过滤线——**两分支互补拼图**：本文供域无关半边（协议+方法论），数学线供域绑定半边（语料工程）。
- **代理指标可靠性的域差异注记**：数学域 loss→capability 的 gap 可能小于 agentic 域（数学能力更接近 loss 可测物），但 #113 受控反例恰在数学/推理域（loss 指错方向）——SFT 桥在数学 MT 仍必要，代理只作装填信号。

## 9. 后续触发器

- 脚本/元数据放出（准则 4 兑现日）→ 复核。
- ~90B 轨迹语料扩容的后续论文 → sweep 关键词（Nemotron agentic traces）。
- "超长 SFT 伤 RL 可训练性"文献线（Liu 2026b/Kang 2025/Zhang 2026）→ 下轮 sweep 候选。

---
*深读底档：/tmp/opencode/singhal.html（2026-11-11 抓取，266KB 全文存档）。*
