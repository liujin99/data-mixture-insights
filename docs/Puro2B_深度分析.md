# Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090

| 项目 | 信息 |
|---|---|
| arXiv | 2608.27370 (2026.08.27, 62 页 20 图 24 表) |
| 团队 | 清华 PACMAN 实验室（Wenguang Chen 陈文光教授，通讯）+ Pengcheng Lab；Kaifeng Lyu（UChicago，scaling law 学者，通讯）；11 作者，4 人同等贡献 |
| 开源 | Apache 2.0 全开源：模型权重（10 个 checkpoint 版本）、数据 manifests + 物化组件、训练代码（Puro-Megatron）、数据处理代码（Kaiyuan-SpaRK） |
| 可信度 | **高**。全管线可复现（数据/顺序/训练状态完整），成本账目透明（RMB/USD 双账本），局限性章节诚实（污染审计缺失、中文非目标、700 TPP 过训练区非 compute-optimal） |

## 一句话总结

用消费级 RTX 5090（8 卡节点）+ FP8 + MuonH 优化器 + **数据课程/课程模型平均（CMA）**，从零训练 2B 稠密模型（Qwen2-1.5B 同款架构），1.4T tokens 总成本 $6.9K（17.6 天），数学/代码均值超 Qwen2-1.5B、逼近 Qwen2.5-1.5B；并给出 recipe 级成本缩放律 P = a + b·log₂(C − C_P1)（$4.4K 过 Qwen2-1.5B）。

## 0. 管线全貌

```
开源候选语料池（带样本级质量分数的源 + 无分数源）
  ↓ Kaiyuan-SpaRK（Spark）：源内去重（不做跨源去重）→ proxy 基准测试 → 启发式过滤 → 物化为固定 shards（无在线混合）
Phase 1（438.8B tokens）：广覆盖打底（English 73.2%），开式 power LR 衰减
  ↓ 过渡段 43.9B（21.9B Phase 1 replay + 21.9B Phase 2 最早 2.34% 课程范围）
Phase 2（960B = 938B 组件池 + 21.9B replay）：目标域倾斜（Math 18.3%）+ 升序质量课程（376 桶）+ 长线性 LR 衰减
  ↓ 尾段：step 218k 起恒定 LR 续训 + 6 checkpoint 等权平均（CMA）
Post-training（SFT 三设置）：聚焦数学 / 放大数学 / 广泛指令（Tulu-3）
```

## 1. 五大效率组件与 2.40× 的来源

| 组件 | 相对收益 | 说明 |
|---|---|---|
| RTX 5090 硬件 | 2.77× | FP8 算力/$ 是 H200 的 ~2.7×；魔改开源 NVIDIA 驱动开 PCIe P2P（31.5→56 GB/s 单向），400Gbps IB 跨节点 |
| Blockwise FP8 | 1.34× | 精度损失可接受 |
| MuonH 优化器 | 1.19× | Muon + 参数/更新的超球约束；有效 LR 高达 5e-2（Hyperball weight LR） |
| **课程/CMA 数据配方** | **1.65×→2.40×** | **算法侧最大单项**：uniform 等效成本 $11.36K vs 生产 $6.89K（1.65×）；加恒定 LR 续训+平均后 $16.55K（2.40×） |

关键启示：在 2B/1.4T 这个尺度上，**数据配方（课程+顺序）的算法收益（2.40×）超过了换硬件的收益**。且作者强调这不是独立优化叠加，而是联合设计系统（LR 调度 ↔ 课程顺序 ↔ checkpoint 平均三者耦合）。

## 2. 数据配比（Table 3，两阶段域级配比）

| 域 | Phase 1 (438.8B) | Phase 2 (938.1B 池) | P1→P2 变化 |
|---|---|---|---|
| English | 73.2% | 59.4% | ↓ 但绝对量 321B→558B 仍增 |
| **Math** | **7.2%** | **18.3%** | **2.5× 相对提升，最大倾斜项** |
| Chinese | 11.7% | 9.4% | 微降 |
| Code | 7.9% | 11.5% | 相对提升 |
| SFT/指令 | 0% | 1.3% | 新引入 |

- **与 MobileLLM-R1 同构的 P1→P2 模式**（广覆盖→目标域倾斜→尾段引入指令数据），但 Puro 的目标域是 Math（用户是 STEM 整体）。
- 组件级（Table 20 摘要）：English 主力是 **Nemotron-CC-v2 HQ（P1 107.5B→P2 448B，最大单一组件）**——与用户 ClimbLab 同属 NVIDIA Nemotron 数据生态；Math 主力是 **UltraData-Math（P2 118.3B）**+ MegaMath-Web-Pro 13.5B + OpenWebMath 13.2B；Code 主力 MegaMath-Code 42.8B。
- 2B 模型 1.4T tokens = **700 TPP（每参数 token 数），过训练数据充裕区**（与用户 ClimbLab TPP≈800 同区）；Phase 2 数据基本单遍（TPP≈1），与用户中训练 TPP≈1.2 同构。

## 3. Proxy 基准协议（§3.6.1 + 附录 I，proxy 可靠性主题的一手素材）

**要解决的问题**：几十个候选数据源，想知道每个源"喂给模型会怎样"，但每个源从头训一个 proxy 太贵。Puro 的方案是一把"尺子"反复用：共享基座只训一次，每个候选只做 2000 步续训测"指纹"，大源再加源内分位窗做"CT 扫描"。类比：不是每个配方都烤一个完整蛋糕，而是先做一个白坯蛋糕，每种原料抹一点尝味道记下来，最后设计好配方再认真烤一次。**先用便宜的指纹测量把候选摸完 → 人工设计配比 → 把预算花在 1-2 次真跑上**（N=1-2 友好的核心）。

**协议**：共享 Qwen3-0.6B 基座（先在统一广域混合上训 86B tokens）→ 对每个候选源/切片续训 2000 步（≈8.4B tokens，seq 4096，全局 batch 1024）→ 15 benchmark 全套评估 → 聚合为 Math/Code/Chinese/General 四轴 capability vector。

**候选切片构造**：
- >50B tokens 且带分数标签：取 q00/q25/q50/q75 四个分位窗口（各 ~4B）
- 5-50B 或无分数：随机 4B
- <5B：跳过

**候选占比调度**：r(t) = min(0.8, 0.8·t/1600)，前 1600 步从 0 爬到 80%，最后 400 步保持 80%（base replay 相应降至 20%）——控制分布突变，避免测量被分布冲击污染。

**关键设计洞察**：
1. **百分位粒度**：样本级分数只在源内可比 → 分位切片把"一个源级分数"变成"源内细粒度决策信号"。跨源无可比分数（回应了全局排名式方法的前提缺陷）。
2. **PCA 审计**（43 条件 × 15 benchmark）：PC1（34.5%）= General vs Code 对抗；PC2（28.1%）= Math vs Chinese 对抗；General-Math 相关仅 0.136。**明确声明 PCA 坐标不可作优化目标**（符号任意、方差最大旋转≠效用定义），只用于描述能力冲突——源级决策在原始聚合轴上做。这是对"代理指标当目标"的方法论自觉。
3. **过滤实例**：MegaMath-Web-Pro proxy 得分最高（Math/General 双高）→ Phase 2 **整体剔除 FineMath**（Phase 1 用过）、分配 13.5B；Nemotron-CC-Math 降为仅保留 quality score ≥4 切片；轴领先者补位（中文 FineWeb-Edu-CN / 通用 DCLM / 代码 MegaMath-Code）。
4. **承认局限**：proxy 特征向量"条件于 proxy 规模、续训调度、候选占比、评估套件"——不声称跨设置可迁移（与本项目 LogFloor/#51 Nexus 的 proxy 可靠性议题直接对话）。

**注意**：proxy 是 0.6B 而最终模型是 2B——容量差异下的选择信号是否成立未验证（可对照 #88 Capacity-Dependent 的容量依赖发现，见交叉引用）。

## 4. 课程模型平均 CMA（§3.4 + 附录 H，2.40× 的核心机制）

**课程构造**：
- 每个组件内按源内分数排序（低质量→高质量升序）；无分数组件用固定随机序
- 以 **token 质量**（非文档数）计归一化排名 p(x)∈[0,1)，切成 **376 桶**（每桶 ~2.5B tokens）
- 第 k 桶从**每个组件**取其第 k 个分位区间 → **跨组件配比在每个桶内近似保持**（这是关键：课程改变的是"每个源内部什么时候看到高质量数据"，不是全局质量排名）
- 实现上用 Spark 近似分位数（256 分数层 × 64 hash 子分）避免全量文档级排序

**尾段优化**：Phase 2 长线性衰减至 step 218k → 恒定 LR（base 4.08e-5 / Hyperball 4.08e-4）续训 → 6 个 checkpoint（间隔 100 步 ≈ 0.63B tokens）等权平均。

**⚠️ 血统与业界实况（2026-09-02 原文逐篇核实，修正此前三处错误引用）**：权重平均不是新东西（SWA, Izmailov 2018；Model Soups, Wortsman ICML'22 best paper），但**"主流大模型是否在用"此前我引证有误，现按原文核实如下**：

| 模型 | 预训练/中训练权重平均 | 核实结果 |
|---|---|---|
| **DeepSeek-V3** | ✅ **预训练 EMA** | 论文明确：CPU 中异步维护参数 EMA（每步更新，零额外显存/时间开销），用于"早期估计 LR 衰减后的模型性能" |
| **OLMo 2** | ✅✅ **中训练 Checkpoint Soups（§4.5）** | 7B：最终高质量混合独立 anneal **3 次×50B tokens**（各不同随机数据顺序）→ 平均 3 个模型；13B/32B：3×100B+1×300B → 平均 4 个。动机原文："get the most out of this high-quality data, and to find a better local minimum"。**恰在 mid-training 阶段，与用户场景直接对口** |
| Gemma 3 | 仅后训练 ✅ | weight-averaged reward models/policies（WARP，Ramé 等）；预训练无 |
| Llama 3 | ❌ | 92 页全文检索无权重平均（有**数据 annealing**——高质量数据收尾，数据侧技巧非权重侧） |
| Qwen2.5 | ❌ | 全文检索无 |
| ~~Gemma 1/2 EMA~~ | ❌ 误引撤回 | "EMA"命中全是 math**ema**tics 子串；Gemma 1/2 预训练无 EMA 记录 |
| ~~MiniCPM model soup~~ | ❌ 误引撤回 | arXiv 版（2404.06345）检索不到 soup/anneal（可能记忆混淆自社区博客） |

**结论**：不是"没人干"也非"人手一份"——DeepSeek-V3（EMA）与 OLMo 2（soups，恰在中训练）两家头部/主流开源在用，Llama 3/Qwen2.5 不用。Puro 的消融恰好解释了不普适的原因：**衰减 LR 轨迹上直接平均无效甚至有害**（uniform −0.42），多数家的收尾正是 decay-to-zero，试了没效果自然放弃；有效条件是恒定 LR 续训（SWA 原始洞见）。另注意 OLMo 2 soups 与 Puro CMA 是同族不同风味：前者平均**独立 run**（不同数据顺序、各自衰减到零），后者平均**单 run 内 checkpoint**（恒定 LR 续训）——两者都印证"数据顺序 × 权重平均"的耦合价值。

**消融（六端点对照）**：
| 组件 | 效果 |
|---|---|
| 课程 vs uniform 顺序 | +1.18 pts（57.17 vs 55.99，无平均时）；+1.61（有平均时） |
| 模型平均（单独） | **无独立收益**：uniform −0.42，curriculum +0.01 |
| 恒定 LR 续训 + 平均 | +0.63（57.81 vs 57.18）；215k 续训点反而差 1.01 |

**机制解释**：升序课程与常规终端 LR 衰减存在冲突——课程把高质量数据排在训练末尾，恰逢参数更新量变小；恒定 LR 续训 + checkpoint 平均解除这个冲突。**课程与 LR 调度必须联合设计**（与本项目 #28 LR Decay、MobileLLM-R1 的 LR confound 洞察同一主题）。

**⚠️ Contamination 混杂（作者自己承认）**：课程后段数据可能含 benchmark 污染，部分优势可能源于此；但 post-training 后优势仍在（+1.77/+2.02/+1.59 pp），"仅靠污染难以解释持续改进"；语料级污染审计缺失，留作未完成工作。

## 5. Post-Training Case Study（§5，对用户最有价值的实验设计）

**问题**：两个预训练 endpoint（uniform vs curriculum/CMA）最终 loss 相近、架构数据源相同，但内部状态不同——**差异是否在相同 SFT 后仍持续？**

| 设置 | Uniform | Curriculum | 差值 |
|---|---|---|---|
| 聚焦数学 SFT（GSM8K, step 172） | 66.89% | 68.66% | **+1.77 pp**（3 次重复运行一致领先） |
| 放大数学 SFT（step 2431，更大混合+replay） | 74.10% | 76.12% | **+2.02 pp**（每次运行都领先） |
| 广泛指令（Tulu-3, step 300, 15 任务宏平均） | 54.99% | 56.58% | **+1.59 pp**（13/15 任务改进；+IFEval/BBH/MMLU-Pro 共 15/18） |

分析协议亮点：GSM8K 用 IoU 分解（共享解集 vs 单侧独有）区分真实能力差与格式效应；显式答案抽取失败单独检查。

**对用户场景的意义**：这是"**预训练数据配方的好处能否穿透 post-training 存活**"的受控实验——大多数配比论文只报预训练终点，Puro 证明了课程优势在匹配的 SFT 后仍持续且跨域迁移。用户的中训练配方决策同样应以此为评估标准（SFT 后 STEM accuracy，而非中训练终点 loss）。

## 6. 成本缩放律（Puro Cost Scaling Law）

- 形式：**P = a + b·log₂(C − C_P1)**，C_P1 = $1.8K 固定成本（Phase 1 底座），从同一 Phase 1 checkpoint 起不同 Phase 2 预算的 uniform 系列拟合，RMSE 0.209
- 外推：$4.4K（< $5,090 主题数）过 Qwen2-1.5B（55.14）；CMA 再省 1.65×
- **适用边界（作者明确）**：固定 2B、recipe 特定的 scale-down 曲线，不是模型尺寸缩放律；服务于"预算有限时能到什么水平"的参考
- 成本口径：仅计最终两次生产 run 的 GPU 时间（$0.31/h RTX 5090 摊销价），**不含**数据策展、proxy 实验、消融、失败 run、后训练、人力——与 MobileLLM-R1 的 15% 数据策展开销对照，Puro 的 $6.9K 显著低估全成本

## 7. 与本项目的系统关联

### 7.1 场景匹配度
- **同规模**：2B 稠密（Qwen2-1.5B 架构）≈ 用户 1.5B
- **同数据生态**：主力组件 Nemotron-CC-v2（English）/UltraData-Math（Math）与用户 ClimbLab（Nemotron-CLIMB）同源
- **同 TPP 区**：700 TPP 过训练区（≈ClimbLab 800）；Phase 2 单遍 ≈ 用户中训练 TPP 1.2
- **同目标结构**：P1 广覆盖 → P2 目标域倾斜（Math 18.3%）→ 尾段指令数据，与 MobileLLM-R1 P1→P2、用户 70% STEM 的结构可互为参照

### 7.2 可迁移洞察（按价值排序）
1. **T1 - 评估指标：accuracy 直测即可，警惕的是 loss**：Puro 的"post-training 存活"实验否定的是**终点 loss**（两个配方 loss 相近但 SFT 后差 2 pp），不是"必须经过 SFT"。用户若评估/交付的就是中训练后模型，直接测 mid-training accuracy 即是正确目标指标（离最终用途最近、少一个混杂）；仅当最终管线还含 SFT/对齐时，建议**一次性**验证配方差异能穿透 SFT（Puro 证明 +1.77~2.02 pp 可穿透，非格式效应）。
2. **T1 - 升序质量课程 + 恒定 LR 续训 + checkpoint 平均**：用户 1.8B 中训练若数据源有质量分数，可低成本套用（376 桶实现细节开源）；注意课程与 LR 调度耦合设计。大白话：课程**不改变配比只改变出场顺序**（每个源内部坏的先喂好的压轴，跨源配比在每个桶内保持）；checkpoint 平均 = 恒定 LR 下模型在极小值附近游走，6 个点平均逼近谷底中心，**必须先恒定 LR 续训再平均**（衰减轨迹上直接平均无用甚至有害）。
3. **T2 - Proxy 分位切片协议（测量工具，非训练配置）**：q00/q25/q50/q75 窗口 + 80% 候选占比爬坡 + 统一 86B-token 基座——仅在需要**自评候选数据源**（往 ClimbMix 加/换源）时使用；直接复用现成配方（如 ClimbMix）训练时不需要。比 RegMix 式小模型从头训便宜一个量级。
4. **T2 - 渐进分布过渡（爬坡思想向真跑的迁移）**：Puro 在 P1→P2 之间插 43.9B 过渡段（21.9B 旧数据 replay + 21.9B 新数据早期段），仅占全程 ~3-5%。用户从 ClimbLab 基座切 70% STEM 中训练时同理：开头混入部分基座分布 replay 再升 STEM 占比，避免分布突变（与 D³-MOPD 域信号调度同族思想）。**⚠️ 用户实际约束（2026-09-02 确认）**：用户当前协议是静态 70/30 全程均匀混合，且对比基线同协议——**对比实验期间不可改动**（引入混杂破坏公平性）。正确用法：把"顺序/调度"当作配比轴定论之后的独立小实验（同 70/30 总量只改时间分布，N=1 即可，赢了免费叠加输了不伤主结论）。**尺度警告**：Puro 课程收益在 960B Phase 2 验证，用户中训练仅 1.8B（500× 小），顺序效应幅度无保证。
5. **T2 - 模型平均单独无益、须配恒定 LR 续训**：低成本消融结论，防止用户盲目套 checkpoint 平均。
6. **T3 - PCA 只做描述不做目标**：proxy 能力轴冲突的可视化审计方法（General-Code 对抗、Math-Chinese 对抗）。
7. **T3 - Math 7.2%→18.3% 的倾斜幅度**：1.4T 总量的 2.5 倍相对倾斜是"够用"的参照点（但注意 Puro 无 STEM accuracy 相变维度的检验）。

### 7.3 不可直接迁移
- **硬件/精度/优化器栈**（5090/FP8/MuonH）：与用户 NPU 环境正交；且 MuonH 与本项目记录的 Nexus 集成"Moun 不兼容"约束冲突——若用户走 Nexus 路线则不能同时用 MuonH
- **700 TPP 大规模预训练**：用户是 continual pre-training（基座已 TPP=800），Puro 是从零
- **成本缩放律**：固定 2B recipe 特定，不能外推到其他设置
- **中英文双语**：中文 9-12% 占比是 Puro 特有约束

## 8. 局限性（作者自述 + 本项目视角补充）

1. **污染审计缺失**（作者自述）：无语料级 exact/near-duplicate 审计，课程 vs uniform 比较的证据强度受影响
2. **proxy 规模差**（本项目视角）：0.6B proxy 决定 2B 模型的数据——#88 证明数据选择价值容量依赖，proxy 容量 ≠ 学生容量时的选择信号可靠性未检验
3. **成本口径窄**：$6.9K 仅计生产 run；数据策展/proxy/消融全排除（对照 MobileLLM-R1 策展占 15%）
4. **无 accuracy 相变维度**：全部用 benchmark 均值，未触及本项目关心的 STEM accuracy 涌现/相变问题
5. **N=1 的配方比较**：主比较是单 run 对单 run（post-training 有 3 次重复但预训练无）；bootstrap 依赖 checkpoint 间方差而非 run 间方差

## 9. 评估

**相关性：★★★★☆**——规模、数据生态、TPP 区、P1→P2 结构四重匹配；扣一星因为其目标域是 Math（非 STEM 整体）且无相变维度。与 MobileLLM-R1 互补：R1 提供 influence 方法论与中训练配比表，Puro 提供 proxy 协议模板、课程机制与 post-training 存活实验范式。开源完整度（数据 manifests + 物化组件 + 全部代码 + 10 个 checkpoint）为 91 篇中最高之一，可作用户 1.8B 中训练的工程参照系。

**交叉引用**：#82 MobileLLM-R1（P1→P2 模式、LR confound、污染混入）、#88 Capacity-Dependent（proxy 容量 vs 学生容量）、#51 Nexus（Muon 不兼容）、#74 Nemotron-CLIMB（数据生态同源）、#70 MixtureExperiment（proxy 实验设计）、#28 LR Decay（课程-LR 耦合）。
