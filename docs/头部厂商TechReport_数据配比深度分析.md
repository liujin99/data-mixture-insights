# 头部厂商 Tech Report 数据配比深度分析

> 聚焦 DeepSeek-V4/V4.1 / GLM-5 / Qwen3 / Nemotron 3 Ultra 头部厂商 frontier 规模技术报告，提取中训练数据配比的可操作洞察。

---

## 论文总览

| 厂商 | 论文 | 规模 | 预训练 tokens | 时间 | arXiv |
|------|------|------|-------------|------|-------|
| **DeepSeek-AI** | DeepSeek-V4 | 1.6T/49B + 284B/13B MoE | 32T + 33T | 2026.4 | 2606.19348 |
| **DeepSeek-AI** | DeepSeek-V4.1-Flash（补录 §4.6） | 552B/8-16B 多模态 MoE | 45T | 2026.9-10 | 无 arXiv（HF PDF） |
| **智谱 AI + 清华** | GLM-5 | 744B/40B MoE | 28.5T | 2026.2 | 2602.15763 |
| **Alibaba Qwen** | Qwen3 | 235B/22B MoE (最大) | 36T | 2025.5 | 2505.09388 |
| **NVIDIA** | Nemotron 3 Ultra | 550B/55B MoE | 20T | 2026.6 | 2606.15007 |

**筛选标准**：业界公认 AI 头部厂商 + frontier 规模 + 2025-2026 新发布。

---

## 1. GLM-5（智谱 AI + 清华, 744B/40B, 28.5T tokens）

> arXiv: 2602.15763v2 | 2026.2（v1→v2 系 2026-11-20 增量复读时发现此前标 v1 有误，v2=2026-02-24） | Zhipu AI & Tsinghua University。深读见 `GLM5_深度分析.md`（canon 级 2026-11-13 + 增量复读 §10 2026-11-20）。

### 1.1 架构

- 744B total / 40B active MoE（GLM-4.5 的 2x：355B/32B → 744B/40B）
- 80 层, 256 experts, top-8 + 无 shared expert
- MLA (Multi-Latent Attention) + Muon Split optimizer
- MTP with parameter sharing（3 层共享参数，accept length 2.76 > DeepSeek-V3.2 的 2.55）
- DSA (DeepSeek Sparse Attention) 用于持续预训练，减少长上下文计算

### 1.2 预训练数据（27T, Section 2.2）

**Web 数据：**
- 基于 GLM-4.5 数据管线，精细化筛选标准
- 新增 DCLM classifier（基于 sentence embeddings）识别标准分类器之外的高质量数据
- World Knowledge classifier（Wikipedia entries + LLM-labeled data 优化）从中等偏低质量数据中蒸馏长尾知识

**Code 数据：**
- 扩展代码语料：刷新代码托管平台快照 + 更多代码网页
- 去重后 unique tokens 增加 28%
- 修复 Software Heritage 元数据对齐问题
- 更精确的语言分类管线
- 沿用 GLM-4.5 quality-aware 采样策略
- 为低资源编程语言（Scala, Swift, Lua 等）训练专用 classifier

**Math & Science 数据：**
- 来源：网页、书籍、论文
- 改进 PDF 解析机制提高数据质量
- LLM 打分候选文档，只保留最有教育价值的内容
- 长文档采用 chunk-and-aggregate scoring 算法提高打分准确性
- **严格过滤**：排除合成/AI 生成/模板化数据

### 1.3 中训练（Section 2.3）— **头部厂商唯一的专门 Mid-Training 章节**

基于 GLM-4.5 的中训练框架，扩大训练量和上下文长度。

#### 三阶段上下文递进

| 阶段 | 上下文窗口 | tokens | 重点 |
|------|-----------|--------|------|
| Stage 1 | 32K | 1T | 基础长上下文能力 |
| Stage 2 | 128K | 500B | 长文档+合成 agent 轨迹上采样 |
| Stage 3 | **200K** | 50B | MRCR-like 数据增强多轮对话回忆 |

> "We progressively extend the context window across three stages: 32K (1T tokens), 128K (500B tokens), and 200K (50B tokens)."

**关键发现**：
> "a subsequent 200K mid-training stage, building upon the initial 128K phase, further bolstered the model's performance even within the 128K context window."

即 200K 阶段不仅提升 200K 窗口性能，**还增强了 128K 窗口内的性能**。

#### 软件工程数据

- 范式：repo 级代码文件 + commit diffs + GitHub issues + pull requests + 相关源文件 → 拼接为统一训练序列
- 放宽 repo 级过滤标准以扩大候选池 → ~10M issue-PR pairs
- 加强单个 issue 级质量过滤以减少噪声
- 为每个 issue-PR pair 检索更多相关文件 → 更丰富的开发上下文
- 过滤后：**~160B unique tokens**

#### 长上下文数据

- **自然数据**：书籍、学术论文、通用预训练语料文档
  - 多阶段过滤：PPL + 去重 + 长度过滤
  - 知识密集域上采样
- **合成数据**：受 NextLong 和 EntropyLong 启发
  - 多种技术构建长程依赖
  - 高相似度文本通过 interleaved packing 拼接为序列
  - 目标：缓解 lost-in-the-middle 现象
- 200K 阶段额外加入 MRCR-like 数据（OpenAI 范式的多变种）

### 1.4 评级

| 维度 | 评分 | 说明 |
|------|------|------|
| 规模 | ★★★★★ | 744B/40B frontier MoE |
| 中训练相关性 | ★★★★★ | 头部厂商唯一专门 Mid-Training 章节 |
| 数据配比透明度 | ★★★☆☆ | 有 token 量但无比例百分比 |
| 可操作性 | ★★★★☆ | 三阶段递进 + 160B SWE + 合成长上下文方法可复现 |

### 1.5 canon 级深读升级（2026-11-13，详见 `GLM5_深度分析.md`）

- **LR 协议全披露（附录）**：PT warmup→2e-4 衰减至 4e-5；**MT 从 PT 终值连续线性衰减 4e-5→1e-5**（非 NVIDIA 重置恒定——LR×数据调度营地的跨厂商对照，连 #145/#137）；DSA warmup 5e-3→2e-4、稀疏适配恒定 1e-5。
- **无 PT 回放披露**（vs NVIDIA 20%+"40% 无增益"消融）——回放必要性的活分歧样本。
- **DSA 段**：MT 后 20B token 稀疏注意力适配（复用 MT 数据+超参）追平 MLA，仅 DSv3.2（943.7B）的 1/47；RULER 表：+150B 联训后 16K/32K/64K 反超基线 +0.49~+1.72。
- **Post-training**：错误段保留但 loss-mask（连 #161 轨迹处理）；GRPO+IcePop 去 KL；**OPCD**（前序各阶段 ckpt 当教师的在策略蒸馏防阶段遗忘——3 Ultra MOPD 的平行实现）；自家上一代 GLM-4.7 当难度门。
- **评测姿态**：主动打榜（自报 AA/LMArena SOTA；SWE-V 77.8 自报）vs NVIDIA 从不提交——canon 内两家第三方指数用法对照。
- **六问六答席位裁定**：MT 数据侧零消融/零比例/零回放披露，席位保留但角色精确化=跨厂商结构见证+LR 另一极+回放分歧样本（详见深读文档 §8）。

---

## 2. Nemotron 3 Ultra（NVIDIA, 550B/55B, 20T tokens）

> arXiv: 2606.15007v1 | 2026.6 | NVIDIA

### 2.1 架构

- 550B total / 55B active MoE, Hybrid Mamba-Attention
- 108 层, 512 experts, top-22 activated
- LatentMoE (优化 accuracy per FLOP + accuracy per parameter)
- MTP (2 heads, shared parameters) for speculative decoding
- NVFP4 pretraining (largest-scale FP4 training to date)

### 2.2 两阶段预训练课程

| 阶段 | tokens | 占比 | 重点 |
|------|--------|------|------|
| Phase 1 | 15T | ~75% | **多样性** (diversity) |
| Phase 2 | 5T | ~25% | **质量** (quality) |

> "We design our data mixtures to balance diversity and quality. We adopt the proposed two-phase curriculum and transition from a data mixture which biases dataset diversity (phase 1) to a data mixture which biases dataset quality (phase 2). This transition occurs after ~15 trillion tokens corresponding to ~75% of pretraining."

### 2.3 数据配比（19 个高级类别，Figure 4）

**Phase 1 vs Phase 2 对比（关键类别）：**

| 类别 | Phase 1 | Phase 2 | 变化 |
|------|---------|---------|------|
| Web crawl (crawl-medium/high + syn-crawl) | ~49% | ~38% | ↓ 质量提升 |
| FinePDFs | 基础 | **上采样** | ↑ 质量聚焦 |
| Math | 基础 | 调整 | |
| Code (Nemotron-CC-Code) | 基础 | 调整 | |
| Wikipedia | 基础 | 调整 | |
| Academic texts | 基础 | 调整 | |
| Legal | 基础 | 调整 | |
| Multilingual (11 语言) | 基础 | 调整 | |
| Crawl++ (OpenWebText+BigScience+Reddit) | 基础 | 调整 | |
| **SFT-style (sft-code/sft-stem/sft-general)** | 基础 | 调整 | ↑ frontloading |

**SFT-style 数据进预训练**是 Nemotron 3 的独特实践，引用 Akter 2026 "frontloading" 证明其有效。

### 2.4 预训练消融实验

| 新增数据 | 验证模型 | tokens | 效果 |
|----------|---------|--------|------|
| Benchmark-oriented 合成 Q&A | Nemotron-family checkpoint | 100B | MMLU-Pro +1.8, code +1.9, commonsense +1.6, **GPQA +11.1** |
| Fact-seeking (Finewiki 生成) | Nemotron 3 Nano | 100B | **SimpleQA +9.92** (40.2→50.2) |
| Legal 数据集 | Nemotron 3 Nano | 100B | **LegalBench 64.6→74.7** |

### 2.5 长上下文扩展（33B tokens）

- **46% long-context + 54% Phase 2 数据**
- 92% 迭代用 1M 上下文，8% 用 4K
  - 4K 迭代只放 math + code SFT-style 数据（维持短 benchmark）
  - 1M 和 4K 不在同一迭代内混合
- 每迭代 25,165,824 tokens
- 未使用 RULER-style 数据
- 学习率恒定 2.5×10⁻⁶

### 2.6 超参数

- WSD (Warmup-Stable-Decay) 学习率
- Warmup: 200B tokens, peak 2.5×10⁻⁴
- Decay: 最后 5T tokens, minus-sqrt 到 2.5×10⁻⁶
- MTP loss scaling: 0.1
- Offline checkpoint merging: 500B token 滑动窗口, 25B 间隔

### 2.7 评级

| 维度 | 评分 | 说明 |
|------|------|------|
| 规模 | ★★★★★ | 550B/55B frontier MoE |
| 中训练相关性 | ★★★★☆ | 两阶段课程 + 长上下文扩展 |
| 数据配比透明度 | ★★★★★ | **19 类别比例公开**，最透明 |
| 可操作性 | ★★★★★ | 两阶段切换 + SFT-in-pretrain + 消融数据丰富 |

### 2.8 同侪对位 + MOPD 后训练 + 第三方口径核查（2026-11-12 复读正文，应用户"看看 Nemotron Ultra"）

**七列同侪对位表**（Ultra vs MiniMax-2.7-230B-A10B / GLM-5.1-744B-A40B / Kimi-K2.6-1T-A32B / Qwen-3.5-397B-17B / DS-v4-Pro / DS-v4-Flash，自家 harness 单协议）：

- **中游偏下、agentic 最弱**：SWE-V 70.7 与 SWE-Multi 67.7 **双垫底**（GLM-5.1 76.2 / Kimi 75.7 / DSv4-Pro 74.5）；BrowseComp 44.4、HLE 26.7、GDPVal 46.7、OmniScience-acc 24.1 皆倒数两名内；TB 2.1 56.4（第 3）、MMLU-Pro 86.8（第 4）、LCB v6 89.0（第 4）、IOI 570（第 3/5 有值）、GPQA 87.0（第 5，同侪 86.1-91.0 挤成一团）。Kimi K2.6 与 DSv4 系霸榜。
- **赢的两列全是纪律型维度**：OmniScience 非幻觉 **78.7 全场第一**（第二名 MiniMax 74.4，DSv4 系 2.8-5.7 对照惨烈）；IFBench 81.7 第二（仅次 DSv4-Flash 82.0）——反幻觉/指令遵循强、榜单应试弱，与"不对榜单优化"自洽。
- **真实卖点=吞吐非精度**：摘要明卖 vs GLM-5.1 **5.9×** / Kimi **4.8×** / Qwen-3.5 **1.6×** 推理吞吐（8K in/64K out）"on-par accuracy"——hybrid Mamba-Attention 省 KV cache + NVFP4；模型=推理效率展示品（卖卡卖栈的延伸）。输掉的行照登=透明度实至名归。
- **修正 §7.5 推广语**："TR 只跟自己比"系 3.5 TR 特征；本 TR 七列同侪表公开且自曝短板——单协议同侪表的可信度 > 第三方指数。

**MOPD 后训练机器**（2026-08 初析未覆盖段，本轮补录）：SFT + 多环境 RLVR + **MOPD（Multi-teacher On-Policy Distillation）**——RLVR 学生全域 rollout、域教师（SWE/终端/GDPval/聊天/通用推理）对 rollout 密集打分、异步流水线；两轮迭代、第二轮教师从 MOPD1 学生再分支（师生共同进化）、无教师覆盖域由 RLVR 学生自教。**数据引擎大量用别家开源模型**（DeepSeek-V3.2/V4/R1、GPT-OSS-120B、Minimax-M2.5、Qwen3-Coder-480B 当教师/轨迹生成器；通用推理教师自报"追平 DS-v4-Pro (High)"）。机制发现两条：① **教师优势必须落在学生已能采样的轨迹支撑集内**——MOPD 在 agentic 域恢复强（部分 benchmark 反超教师）、HLE 恢复弱（教师增益来自学生没见过的离策略数据时，在线蒸馏失灵）；② sampled-token 目标 > 全分布匹配（离支撑前缀上教师 logit 失准）。SWE 教师 RL 反作弊纪律：容器仓库重写为 base commit 等价新克隆（未来 commit **物理删除**）+ 运行时命令过滤封 git 恢复/网络拉取；未完成轨迹 loss-mask、畸形推理/工具调用负 advantage。**与 #161 三明治的连接**：NVIDIA 版"SFT/RL 层"的旗舰级实现——#161 的 MT 喂进的就是这套。开源声明**已核验**（2026-11-16，HF 经 hf-mirror 通道；github 当日仍不通）：**三项兑现**——①三档 ckpt（`Ultra-550B-A55B-Base-BF16`/`-BF16`/`-NVFP4`，另有 GenRM）；②数据 226 个数据集全披露，PT 集合级 token 数公开（CC-v2/v2.1 9.1T、Pretraining-Code v1-v3 1.7T、CC-Code-v1 427.9B、Specialized 660B、CC-Math 133.3B、Legal 4.3B、多语 CC 812.7B、GitHub crawl 747.4+172.7B——公开集合对账 ~14T/20T，"major portions" 表述诚实，余量=私有/第三方合成部分〔如实列名未放〕）；③recipe 指向 NeMo-NeMo/Nemotron repo（链在、内容未目检——低优先级尾巴）。**#60 准则 4 强命中**（一档 TR + 二档开源资产双证据；BF16 主 ckpt 月下载 1M+）。**超预期发现：RL/MOPD 配比表直接公开**（`Nemotron-RL-Ultra-Training-Blends`，2026-05-29 建）——七份 blend 各带精确百分比：**rlvr1/rlvr2**（agentic 工具对话 ~20.3%、SWE-rebench-V2+SWE-Gym ~14.1%、IF ~12%、数学 ~10.9%、竞赛码 ~8%）、**ifbench**（QA 弃答 43% 单一目的）、**rlhf**（GenRM 84.6%/安全 15.4%）、**reasoning**（100% 外部 virtuoussy/Multi-subject-RLVR）、**swe**（97.4% 外部 nebius/SWE-rebench-V2）、**mopd**。两条方法学信号：① **rlvr1→rlvr2 轮间迭代可审计**——成分稳定、饱和项微降（function-calling 4.08→2.69）+新增 ~1.4% 级定向补丁（litmus/citation/free-form formatting）= RL 端"定向注入"实操；② 单一目的小 blend 补丁模式（ifbench/reasoning/swe）与 PT 端价目表同构；且旗舰 RL 混合头部成分直接用第三方开源数据集。

**AA 11.8 的测量学教训（修订 §7.6 ③）**：自报 GPQA 87.0 / MMLU-Pro 86.8（同侪第 5/第 4）vs AA intelligence 11.8（#45/57）——第三方指数把同侪中游模型测成垫底，**开放权重第三方测量的系统性风险坐实**（NVFP4 原生权重出自家栈的 serving 量化/模板/推理模式触发）；AA 数字降级"仅趋势参考"。水平终判以同侪表为准：**开源旗舰中游偏下、agentic 垫底、纪律型第一**。四层拆解补注：第 1-2 层（基座×后训练）相对同侪未达顶是自家协议下的事实，第 4 层（不应试）只解释榜单缺席、不解释同侪排名。

---

### 2.8.1. recipe repo 目检（2026-11-16 尾巴清仓——低优先级尾巴升级为超预期收获）

`NVIDIA-NeMo/Nemotron`（"Developer Asset Hub"，2349★，11-12 仍活跃）训练配方在 `docs/nemotron/ultra3/`（pretrain/sft/mopd/quantization 四阶段，产品线另有 lightning35/super35/nano3/omni3 全套同款）：
- **PT 配比可执行化（二档证据升级）**：两相配比编码为 `config/data_prep/data_blend_raw_{phase1,phase2}.json`（权重对齐 TR Figure 4）+ Ray tokenize CLI（`nemotron ultra3 data prep pretrain`）——19 类两相配比从"TR 图"升级为"可执行 JSON+CLI"；新数据用 Specialized-v1.2/Legal-v1 已放 repo。
- **诚实边界三条**：① **LC 阶段不含**（数据非开源），给复刻配方：从 Phase 2 ckpt 起 CPT，~46% LC+54% Phase 2、**恒 LR 2.5e-6**、~33B tokens、**92% 迭代@1M+8%@4K（仅数学/码 SFT 式）**、不用 RULER 式数据（LC 阶段恒定低 LR 与 Singhal 恒定哲学同族）；② **MOPD 全链不可复现**——每教师 ckpt 与迭代-1 MOPD ckpt 未开源，`mopd.md` 只给"单趟代表性参考"（Student RLVR→小教师面→一趟 MOPD）；③ 未开源类目如实列 `_missing_categories`（code/CC-Code/academic/crawl++；**Code-v3 只放 repo manifest 不放 tokenized text**）。
- **后训练管线序全披露**：Base → SFT → RLVR → MOPD warmup → MOPD（×N）→ **MTP boosting**（新环节：MTP 增强收尾）。
- **mopd.md 操作细节（RL/MOPD 环节二档升级）**：**general teacher=Student RLVR 自身**、其余教师槽缺省回退 general（"无教师覆盖域学生自教"的操作化）；**agent→teacher 路由表**（math judge/equivalence/mcqa/ns_tools/code_gen→reasoning；swe_pivot/terminal harnesses/droid/structured outputs/freeform/citation→SWE；GenRM→RLHF；IF/abstention/multichallenge→IFBench）；**blend 第三方数学行为占位符+`fill_placeholders.py` 从原始 HF 数据集恢复**（合规重分发模式）；**验证集协议=每 blend 末 100 行**；Stage 1 RLVR 两阶段（49k/128 步→65k/~50 步，rlvr1→rlvr2 换 blend，长度惩罚 0.15→0.08，GBS 8192=512×16，advantage clip ±20）；**NL2BASH 裁判=Qwen3-235B-A22B-FP8**（"别家模型当裁判/数据引擎"再+1）；SWE 环境构建带 verify gate（gold patch 应用+FAIL_TO_PASS/PASS_TO_PASS 过滤才发布镜像）；默认 224 节点拓扑（64 训练+128 vLLM+12 Gym+20 教师）。
- **Singhal（#161）脚本触发器未兑现维持**：repo 无 MT 阶段文档（各产品线只有 pretrain/sft/rl/quantize；data/ 下仅通用 curation/sdg）、arXiv 2610.22019 仍 v1（10-15）——"未来版本放出脚本+元数据"未兑现，触发器=arXiv v2 / repo 新增 MT 文档 / Megatron-Bridge 新资产。


## 3. Qwen3（Alibaba, 235B/22B, 36T tokens）

> arXiv: 2505.09388v1 | 2025.5 | Qwen Team

### 3.1 架构

- 235B total / 22B active MoE (flagship)
- 94 层, 128 experts, top-8, 无 shared expert
- GQA + SwiGLU + RoPE + RMSNorm + QK-Norm
- 移除 QKV-bias（区别于 Qwen2）
- Global-batch load balancing loss

### 3.2 三阶段预训练

| 阶段 | tokens | 序列长度 | 重点 |
|------|--------|---------|------|
| S1 (General) | ~30T | 4K | 通用知识 + 119 语言 |
| **S2 (Reasoning)** | **~5T** | 4K | **增加 STEM/代码/推理/合成数据比例** |
| S3 (Long Context) | 数百B | 32K | 75% 文本 16K-32K + 25% 文本 4K-16K |

> S2 "increasing the proportion of STEM, coding, reasoning, and synthetic data" + "accelerate the learning rate decay during this stage"

S2 本质上是中训练——在通用预训练后聚焦推理能力增强。LR decay 加速暗示 S2 末尾接近 annealing。

### 3.3 预训练数据（36T, Section 3.1）

- 总量 36T tokens，是 Qwen2.5 的 2 倍
- 119 语言和方言（Qwen2.5 仅 29 种）
- 领域：coding, STEM, reasoning, books, multilingual, synthetic

**数据扩展方法：**
1. **PDF 文本提取**：Qwen2.5-VL 识别 → Qwen2.5 精炼 → 数万亿高质量 tokens
2. **合成数据**：Qwen2.5-Math + Qwen2.5-Coder 生成数万亿 tokens
   - 格式：textbooks, QA, instructions, code snippets
   - 覆盖数十个领域
3. **多语言扩展**：29 → 119 语言

### 3.4 关键创新——实例级配比优化

> "Unlike previous studies (DoReMi, DoGE, RegMix) that optimize the data mixture at the data source or domain level, our method optimizes the data mixture at the **instance-level** through extensive ablation experiments on small proxy models with the fine-grained data labels."

- 多语言数据标注系统：30T+ tokens 标注
- 标注维度：educational value, fields, domains, safety
- 用小 proxy 模型做大量消融实验确定实例级最优混合
- **超越域级配比方法**（DoReMi/DoGE/RegMix）

### 3.5 评级

| 维度 | 评分 | 说明 |
|------|------|------|
| 规模 | ★★★★☆ | 235B/22B，MoE 但小于 GLM-5/Nemotron |
| 中训练相关性 | ★★★★☆ | S2 推理阶段本质是中训练 |
| 数据配比透明度 | ★★★☆☆ | 描述方法但未公开 S2 具体比例 |
| 可操作性 | ★★★★☆ | 实例级配比优化思路 + 三阶段课程 |

---

## 4. DeepSeek-V4（DeepSeek-AI, 1.6T/49B, 32T+33T tokens）

> arXiv: 2606.19348v1 | 2026.4 | DeepSeek-AI

### 4.1 架构

两个模型：

| 模型 | 总参数 | 激活参数 | 层数 | Hidden | Experts |
|------|--------|---------|------|--------|---------|
| V4-Flash | 284B | 13B | 43 | 4096 | 256 routed (6 active) |
| V4-Pro | 1.6T | 49B | 61 | 7168 | 384 routed (6 active) |

- Manifold-Constrained Hyper-Connections (mHC)
- Hybrid Attention: CSA (Compressed Sparse Attention) + HCA (Heavily Compressed Attention)
- Muon optimizer
- MTP depth=1
- 1M token context

### 4.2 预训练课程

**序列长度递进**：4K → 16K → 64K → 1M

**注意力密度分阶段**：
- 前 1T tokens: dense attention warmup
- 64K 起: sparse attention (CSA)
- 先 warmup lightning indexer, 再全量 sparse training
- V4-Pro 有更长的 dense attention 阶段

**Batch size scheduling**：小 → max → 恒定
- Flash max: 75.5M tokens
- Pro max: 94.4M tokens

**学习率**：
- Linear warmup 2000 steps → peak → 恒定 → cosine decay → end
- Flash: peak 2.7×10⁻⁴, end 2.7×10⁻⁵
- Pro: peak 2.0×10⁻⁴, end 2.0×10⁻⁵

**MTP loss weight curriculum**：0.3 (大部分训练) → 0.1 (LR decay 开始时) — 功能等价于 annealing

### 4.3 数据构造（Section 4.1）

**数据类别（仅定性描述）：**
- Web 数据（过滤批量自动生成/模板化内容）
- 数学语料
- 代码语料
- 多语言语料（更大语料库，改善跨文化长尾知识）
- 长文档（科学论文、技术报告等学术价值材料）
- Agentic 数据（**在中训练阶段加入**）
- 其他高质量类别

> "we further enhance the coding capabilities of DeepSeek-V4 series by incorporating **agentic data during the mid-training phase**."

**这是全文唯一提到"mid-training"的地方。** 无 token 数、无比例、无阶段设计。

**预处理**：
- 沿用 DeepSeek-V3 预处理策略
- Tokenizer: 基于 V3, 128K vocab, 新增少量 special tokens
- 继承 token-splitting + Fill-in-Middle (FIM)
- Document packing: 跨源打包减少截断
- **新增**：sample-level attention masking（V3 无此功能）

### 4.4 训练稳定性

**Anticipatory Routing**：
- Step t 用 θ_{t-Δt} 计算 routing indices（解耦 backbone 和 router 更新）
- 预计算 routing 只需一次 forward pass
- 动态激活：loss spike 时触发 rollback + 开启，恢复后切回
- 额外开销 ~20%

**SwiGLU Clamping**：
- Linear component: clamp to [-10, 10]
- Gate component: upper bound 10
- 有效消除 outlier，不影响性能

### 4.5 评级

| 维度 | 评分 | 说明 |
|------|------|------|
| 规模 | ★★★★★ | 1.6T/49B，最大规模 |
| 中训练相关性 | ★★☆☆☆ | 仅一句话提到，无细节 |
| 数据配比透明度 | ★☆☆☆☆ | **最不透明**，无任何比例数字 |
| 可操作性 | ★★☆☆☆ | 训练稳定性技术可参考，数据配比不可用 |

### 4.6. DeepSeek-V4.1-Flash TR 补录（2026-11-16 五厂盯哨触发器命中）

> **来源**：`DeepSeek_V41_Tech_Report.pdf`（51 页，挂 HF `deepseek-ai/DeepSeek-V4.1-Flash` 仓库；**无 arXiv**）。主库 2026-09-10 建库即挂、1017 构建 10-16——修正 11-16 sweep"五厂 TR 全未出"对 DS 的漏检。总表 #171（T2）。主题=KV cache 压缩+原生多模态+agentic：552B backbone（prefill 8B/decode 16B 激活）、45T 多模态语料、稀疏注意力 64K 从零（免 dense warmup）→34T 处扩 1M。**无 MT 章节**（LC 在 PT 内完成）。

**4.6.1. 数据构造（TR §4.1）**：
- **文本**：从"样本级质量"转向"**语料间整体交互/独特信息增益**"视角；systematic pipeline 优化 mixture；**scaling ladder（参数×数据）经更全面评估设计以指导大规模 run**——又一家披露"管线+评估驱动设计"而无任何搜索算法（席位③+1）。
- **"隐式重复"**：过滤低信息增益的模型生成内容（弱模型输出+低质机翻）=**隐式重复，长训练 horizon 有害**——#126"先重复 human 再扩 AI 语料"的工业印证、MAI memorization 重复上限（手册"重复控制"环节）同族；另披露 model-in-the-loop 数据迭代为未来大规模合成数据打底。
- **代码**：换新（新仓库/commits/库/框架，更多语言，贴近当代软件工程场景）。
- **多模态**：**反大规模合成立场**——"原生数据清洗优先，最直接可扩展的视觉知识压缩"；爬虫文本偏置→从 Common Crawl 重新 bootstrap；图文对=alt text+相关性阈值+图像语义去重；交错数据=网页+PDF，**渐进提价管线**（先启发式/统计过滤+去重+质量模型，幸存者才取图组装，再做图像感知过滤+SmolVLM 严格打分）；被滤文档回收重组为图文对；领域特定数据补 grounding/OCR/长尾；**image-code 对+computer-use 轨迹进 PT 语料**——多模态轴的"轨迹进 PT"，与 NVIDIA/Singhal"轨迹进 MT"路线的**轴分歧**（如实记）。
- **整合**：文本∪多模态；重叠样本以多模态版替换文本版+取更大 epoch 数；**文本:多模态=7:1 token 比**（唯一轴级配比数字）；联合预取分配最小化样本重叠；超长文档确定性预切分；best-fit packing padding≤1e-4。

**4.6.2. 训练设置（TR §4.2）**：Muon（线性层）+AdamW（RMSNorm 等）+Sinkhorn 均衡更新（embedding/prediction head）；batch 100.6M 恒定；**LR：warmup 2000 步→2.6e-4 恒定至 28T→28-40T cosine 衰至 1/10（2.6e-5）→40-45T 恒定低温平台收尾**——"先衰减后平台"第三形态（#145 衰减营/#161 恒定营之外的混合式）。

**4.6.3. Post-Training（TR §5）**：
- **头条证词**："固定且平凡的优化程序下，合成数据与环境的规模/多样性/可验证性的系统改进贡献了基本上全部收益……**后训练中数据与环境管线的边际回报远超算法创新**"——与 #161"base 代理不足以指导配方决策"同族的工程证词、库核心命题（生产在管线不在搜索）再+1。
- **任务合成**：任务=（问题,环境,验证系统）三元组；**难度+正确性当 reward** 迭代训练模型造任务；**任务全生命周期监控**（每个新 RL run 的轨迹=任务质量再审计证据）。通用 agent：员工/伙伴真实工作流回流→**mock 工具复刻真实 SaaS/企业接口**→失败回放+定向 RL 补弱点。编码 agent：真实会话滤复杂/低性能任务+星标仓；**多专职 agent 建环境流水线**（可构建性判定→任务设计 fail-to-pass/pass-to-pass→容器自测+**清除解题痕迹**→多 solver→**独立质检 agent** 查环境/事实/eval 错配/hackability→修复 agent→复审循环）。
- **RL 配比的执行机制**：异步 RL 下 **per-dataset 并发上限=稳态 batch 中各数据集占比的旋钮**——Nemotron RL blend 百分比表"如何被执行"的工程侧写（两家互为表里）；长度偏置双护栏（并发上限+丢弃早期返回短样本）。
- **Off-policy 双护栏**：调度逻辑控制最大 off-policy 比例+**staleness loss masking**（过陈旧样本的 token 不进梯度）——#168-170"伤 RL 的是离策略/过量模仿"的头部厂工程对策。
- **RL 算力扩展**：跨 run checkpoint **合并再初始化**续训（异构 scaffold 间聚合并行 RL 算力）——"合并当 RL 续算力"（与 #130 MergeMix"合并当配比搜索代理"不同轴）。
- **OPD 终段**：全词表 on-policy distillation，**40+ 异构教师**、每域最优教师可来自不同发展阶段、**训练中动态调整 mixture/每数据集并发/活跃教师池**（异步下配置切换一致性有工程方案）——MOPD/OPCD 家族**第二家工业印证**（Nemotron 域教师体系+#167 RL ckpt 池；引 Thinking Machines OPD 博客）。
- **评测反作弊**：断网+剥 Git 历史+清构建缓存（go mod/node_modules/.jar/pycache）；实测仍见 exploit（反编译 Ubuntu 核心包找 CyberGym 漏洞）——"标准评测基建随模型能力增值而贬值"的实证警示（daVinci 反作弊纪律同族）。

**4.6.4. 评级**

| 维度 | 评分 | 说明 |
|------|------|------|
| 规模 | ★★★★★ | 552B/45T，多模态从零 |
| 中训练相关性 | ★☆☆☆☆ | 无 MT 章节；LC 在 PT 内扩展；agentic 轨迹走 PT 多模态轴 |
| 数据配比透明度 | ★★★☆☆ | 7:1 轴级+管线细节全披露；无逐类配比表 |
| 可操作性 | ★★★★☆ | RL 数据管线/任务合成/反作弊纪律可直接抄；PT 侧偏多模态 |

---

## 5. 横向对比与中训练洞察

### 5.1 中训练数据配比策略对比

| 厂商 | 中训练阶段 | tokens | 核心策略 | 配比透明度 |
|------|-----------|--------|---------|-----------|
| **GLM-5** | 3 阶段上下文递进 | 1T+500B+50B | 长文档+SWE+合成长上下文 | 中 |
| **Nemotron 3** | 2 阶段课程 + 长上下文 | 15T+5T+33B | diversity→quality + SFT-in-pretrain | **高** |
| **Qwen3** | S2 推理阶段 | ~5T | STEM/代码/推理上采样 | 中 |
| **DeepSeek-V4** | 提到一次 | 未披露 | 仅说 agentic 数据 | **极低** |

### 5.2 对 QuaDMix 中训练项目的可操作洞察

#### 洞察 1：两阶段 diversity→quality 切换（Nemotron 3）
- Phase 1 (75%): 多样性优先
- Phase 2 (25%): 质量优先
- **直接验证 quality-first 设计**：QuaDMix 的 quality-top-k 思路被 NVIDIA 在 550B 规模验证
- **可操作**：在训练 75% 处切换数据配比，从多样→高质量

#### 洞察 2：三阶段上下文递进（GLM-5）
- 32K → 128K → 200K，每阶段增加难度数据
- **关键发现**：200K 阶段不仅提升 200K 性能，还增强 128K 窗口内性能
- **可操作**：中训练可分阶段递进，后期加入更难/更高质量数据

#### 洞察 3：S2 增加 STEM/代码比例（Qwen3）
- S2 = 5T tokens 聚焦 STEM/coding/reasoning/synthetic
- 与用户当前 3:7（通用:STEM）配比思路一致
- **可操作**：在通用预训练后增加 STEM 专项阶段

#### 洞察 4：SFT-style 数据进预训练（Nemotron 3）
- sft-code/sft-stem/sft-general 在 Phase 1/2 均有纳入
- 引用 Akter 2026 "frontloading" 证明有效
- **可操作**：考虑在中训练中加入 SFT-style 高质量格式数据

#### 洞察 5：SWE 数据实践（GLM-5）
- repo 级代码文件 + commit diffs + GitHub issues + PRs → 统一序列
- ~10M issue-PR pairs → ~160B unique tokens
- **可操作**：如需加入代码数据，这是头部厂商的实践范式

#### 洞察 6：实例级配比优化（Qwen3）
- 超越域级配比（DoReMi/DoGE/RegMix）
- 用细粒度标签做实例级消融
- **与 QuaDMix 的关系**：QuaDMix 是域级配比，Qwen3 表明实例级更优但需要更精细标签

#### 洞察 7：合成数据策略差异
- **GLM-5 Math**：严格排除合成/AI 生成/模板数据
- **Qwen3**：大量使用 Qwen2.5-Math/Coder 合成数据（数万亿 token）
- **Nemotron 3**：benchmark-oriented 合成 Q&A 有效（GPQA +11.1）
- **DeepSeek-V4**：过滤批量自动生成/模板化内容
- **结论**：头部厂商对合成数据态度分化，但 benchmark-oriented 合成数据有明确消融验证

### 5.3 数据配比透明度排名

1. **Nemotron 3 Ultra**（NVIDIA）：19 类别 + Phase 1/2 比例 + 消融数据 ★★★★★
2. **GLM-5**（智谱）：三阶段 token 数 + SWE 数据量 + 方法描述 ★★★☆☆
3. **Qwen3**（阿里）：三阶段 token 数 + 方法描述，无 S2 比例 ★★★☆☆
4. **DeepSeek-V4**（DeepSeek）：仅定性类别列表，无任何数字 ★☆☆☆☆

### 5.4 头部厂商 vs 学术论文的互补关系

| 需求 | 头部厂商 Tech Report | 学术方法论论文 |
|------|---------------------|--------------|
| 配比数字和策略 | ✅ Nemotron 3/GLM-5 | ❌ 规模太小 |
| 可复现算法 | ❌ 不透明 | ✅ InfoLaw/DoReMi |
| 中训练专项 | ✅ GLM-5/Qwen3 | ✅ CAMEL/ADAPT |
| 前沿规模验证 | ✅ 235B-1.6T | ❌ 最大 55B(CAMEL) |

**建议**：以头部厂商 tech report 的配比策略为**设计框架**，以学术论文的可复现方法为**优化工具**。

---

## 6. 与用户当前中训练设置的对照

用户当前设置：3:7（通用:STEM），无 code 数据，198 域 → 需分组。

| 头部厂商实践 | 用户当前 | 建议 |
|-------------|---------|------|
| Nemotron 3: 75% diversity → 25% quality | 全程 3:7 固定 | 考虑分两阶段：前 75% 多样性，后 25% 质量聚焦 |
| GLM-5: 三阶段上下文递进 | 未分阶段 | 考虑分阶段递进上下文+难度 |
| Qwen3: S2 增加 STEM/代码 | 3:7 通用:STEM | ✅ 思路一致，但头部厂商都含 code |
| Nemotron 3: SFT-in-pretrain | 无 SFT-style 数据 | 考虑加入 sft-stem 类型数据 |
| GLM-5: 160B SWE 数据 | 无 code 数据 | 头部厂商全部包含 code，建议补充 |
| Qwen3: 实例级配比 | 域级配比 | 长期目标：从域级升级到实例级 |

**核心差距**：所有头部厂商的中训练/预训练都包含**大量代码数据**（≥15%）。用户当前无 code 数据是与头部厂商实践的最大差异。

---

## 7. Nemotron 3.5 Super/Lightning（NVIDIA, 2026-10-28 回看定向抽取）

> 回看背景：11-09 sweep 曾以"家族由 #60 代表"搁置本 TR；11-11 用户裁定执行回看。抽取范围=§2.5 中训练 + §2.2 蒸馏刷新 + §2.3 Indic CPT 配比消融；其余（MTP/量化/推理部署）轴外。arXiv 2610.38095，Super 235B / Lightning 30B-A3B（Mamba-Transformer 混合 MoE）。

### 7.1 中训练配方（§2.5——本 TR 最高价值段）

**定位哲学（三明治逻辑）**：PT 在短上下文跑、几乎见不到 agent 交互，而软件工程自然数据（完整仓库、PR+issue、commit 史、构建日志）太长进不去 PT；SFT 几乎全是合成数据且太长会损害后续 RL 可训练性。**中训练卡在两者之间：长上下文+自然数据+少量 agent 轨迹，行为塑造留给 post-training**——对 3B-A1 的直接启示：MT 装"知识与格式"，不装"行为"。

**数据配比（Table 4，训练 token 份额）**：

| 数据源 | 说明 | 份额 |
|---|---|---|
| Issue–PR | GitHub issues 带仓库上下文+合并补丁 | 41% |
| Repositories | 完整仓库（import 相关联文件放一起） | 20% |
| Agentic traces | 多 scaffold 轨迹，**工具响应 loss-mask** | 10% |
| Commit diffs | 多文件提交+逐文件编辑史 | 3% |
| Build/CI logs | 原始构建与持续集成日志 | 1% |
| **预训练回放** | Phase-2 预训练混合 | **20%** |
| 长上下文数据 | LC 阶段数据 | 5% |

- 自然数据保留原始格式、不加 chat 模板；agent 轨迹是唯一 SFT 式数据，压小份额、只训模型自己的轮次
- 全部数据对 SWE-bench/Terminal-Bench 去污染+去密钥
- **PT 回放 20%=回放线第三处工业实例**（#157 微调/中训练 + Marin SFT 20.35% + 此处），跨三家收敛

**训练协议**：从 LC checkpoint 出发 65.4B tokens @ 256K；**LR 重置+线性 warmup 至 1e-5（≈2× PT 终值）后恒定**（无衰减）；256K 训练损伤 RULER-1M → 追加 ~9B token 长上下文恢复段（恢复段预算 ≈ MT 的 14%）——**MT 可损伤相邻能力，恢复段是标配动作**。

**效果评测（SFT 桥协议）**：同一份实验性 agent SFT（33B tokens/2K 步）分别套在 base 与 MT 后 checkpoint——"可靠预测完整 post-training 后表现"，**工业级 SFT 桥实践**（与本项目三轨制讨论直接互证）。结果：DeepSWE 6.4→10.4、BrowseComp 微升、Multilingual/Pro 在 run-to-run 波动（~1 分）内；**主效应=跨 harness 稳健性**：5 个非标准 harness 全胜（SWE-bench Verified 平均 +8.6、Terminal-Bench 2.1 平均 +6.8、最大 Codex 48.4→61.8）——**MT 的价值主要不在标准 benchmark 分数，在"换 harness 不塌"**。

**完整配方在伴生论文**：Singhal et al. 2026《Priming Base Models for Agentic Post-Training: An Open Midtraining Recipe》（arXiv 2610.22019，10-15 提交）——逐数据源构建/过滤/消融全公开，Lightning 上 500B token MT 实验。**该文曾漏网**：落在 10-10 与 10-17 两轮 sweep 之间的窗口缝隙，且标题用无连字符 "midtraining"（11-11 缝隙补扫确认缝隙内仅此一篇在轴）。已录 #161，**2026-11-11 深读完成**：`OpenMidtrainingRecipe_深度分析.md`（五要点：SFT 桥评测协议/工具响应 mask 差 10 分/LR 尺寸分档+回放 20%/轨迹增益大头但封顶/配比全表+逐源纪律；升 T1 建议待裁）。

### 7.2 蒸馏刷新（§2.2）
3.5 Base 不从头训：从 3 系 Base 初始化后**蒸馏 1T tokens**（Lightning←Super 3 Base：top-128 logits、forward KL 0.9+CE 0.1+sink token；Super←Ultra 3 Base：Skew Reverse KL α=0.1；LR 1e-5→1e-4（10% peak）→线性回 1e-5；MoE load-balance 系数 1e-5）——"换代=蒸馏"的低成本模型更新路径，蒸馏轴备查。**2026-11-16 recipe repo 佐证补录**：① Lightning 出身细节补全（Nano 3 Base 初始化+Super 3 MTP 头〔LatentMoE 投影融合进专家 up/down 后剪枝：hidden 4096→2688、专家中间维 2688→1856、路由专家 512→128〕+**100B MTP 热身〔backbone 冻结只训 MTP 头〕**+1T 蒸馏）；② **模型卡与 TR 冲突**——卡称">20T NVFP4+MTP 层 CPT、无蒸馏"、TR 称蒸馏——NVIDIA 自己的 recipe 文档表态"follow the tech report"（一档/二档冲突时信 TR 的活样本）；③ 开源参考配方 `lightning35 pretrain`=从零随机初始化+**全 1.0 均匀权重 blend（23 子集×4 家族）+明说"不代表生产配比"**——开源参考配方≠生产配方，"定比例=手调"再添二档注脚。

### 7.3 Indic CPT 与 Lightning 配比消融（§2.3）
- 语料：Nemotron-Pretraining-Indic-v1 ≈**2.83T 生成 token**（Gemma-4-31B-it 翻译 Nemotron-CC 英文源、22 语言 34 种配置；语料+生成管线将开源）
- **Lightning 3.5 固定预算配比扫描**：200B token 预算，10%/20%/30% Indic 份额 → 四基准均值（MILU/MMLU-Pro/ARC-C/GSM8K）63.1/64.6/65.6（base 45.5），单调升至 30% 且明示"未确立最优份额"——**目标尺度固定预算 r-scan 的工业模板**（30B-A3B 用户量级）
- 伴生报告《Scaling Synthetic Pre-training Data and Evaluation for Indic LLMs》将出（翻译模型对比+Nano 3 数据源消融+AI4Bharat 人评套件）

### 7.4 对 3B-A1 的可直接引用清单
1. MT 定位：装知识/格式，不塑造行为（行为归 SFT/RL）
2. 配比结构：域自然数据 ~65% + **PT 回放 20%** + LC 数据 5% + SFT 式数据 ≤10%（工具响应 loss-mask）
3. LR 锚点：重置+2× PT 终值+恒定（与 #145"衰减应在 MT 进场后"并存为两个对照锚点——场景不同〔256K 域注入 vs 数学 MT〕，实验做对照臂）
4. 恢复段：MT 后补短程相邻能力恢复
5. SFT 桥：固定实验 SFT（2K 步量级）当完整 post-training 表现的早期信号
6. 评测口径：标准 benchmark 可能不动（run-to-run ~1 分），**跨 harness 稳健性才是 MT 主效应**——评测设计要含"换 harness 不塌"维度

### 7.5 榜单对位核查（2026-11-12，SWE-bench Verified 官方提交榜）

- **官方榜上无名**：swebench.com Verified 榜 75 个模型条目中**零 Nemotron 3.5 条目**——NVIDIA 从未提交该系列（历史唯一 NVIDIA 相关提交=2025-04 Nemotron-CORTEXA agent 脚手架，挂别家底模，非模型本体）；Terminal-Bench 榜 JS 渲染无法静态核查。印证"TR 只跟自己比"的不打榜定位（**2026-11-12 修正**：此特征属 3.5 TR；3 Ultra TR 自带七列开源同侪对位表且输多赢少照登，见 §2.8——"只跟自己比"应收窄为"不对外提交榜单"）。
- **纸面对位**（⚠️ 协议不同仅可粗看）：榜头 Claude 4.5 Opus 79.2 / Doubao-Seed-Code 78.8 / Gemini 3 Pro 77.4 / GPT-5 74.4；最强开源提交 Kimi K2 71.2 / Qwen3-Coder-480B 69.6 / GLM-4.6 68.2。Super 74.3（自家 OpenHands 3-rollout 口径）纸面 ≈GPT-5 档、高于全部开源提交；Lightning 64.5 ≈GLM-4.5（64.2）档，**同规格唯一对位条目 Qwen3-Coder-30B-A3B 60.4**（纸面 +4.1，但系 2025-02 老提交不可直接比）。
- **打折理由**：榜单各条目 harness/尝试次数不同（头部基本 Attempts 2+）；跨 harness 差可达数分（本文 Table 7 自显最大 +13.4）——纸面对位只可读作"不落后于开源第一梯队"的弱证据。**画像结论：中上水平+最透明披露，非榜上王者；参考答案价值与此排位无关。**

### 7.6 第三方评测核查（2026-11-12，Artificial Analysis 三指数，经 OpenRouter 板块）

- **可得性**：LMArena / Terminal-Bench 均 JS 渲染无法静态抓取；OpenRouter rankings 页内嵌 AA 三面板（intelligence 57 模型 / coding 153 / agentic 110）。
- **Nemotron 全家三指数全线中下-垫底**：intelligence——3.5 Super VL #41（13.6）、3 Ultra #45（11.8）、3.5 Lightning #51/57（5.5），头部 Claude Fable 5.2 64.6；coding——3 Ultra #65（49.3）、3 Super #86（37.7）、3.5 Lightning #100（26.8），头部 Fable 5.1 81.6；agentic——3 Ultra #49（20.1）、3.5 Lightning #74（3.5）、3 Super #79（1.7），头部 Fable 5.1 57.9。**同规格对位（30B-A3B 级，coding）：Qwen3.5-35B-A3B 37.0 / Qwen3.6-35B-A3B 41.9 vs Lightning 26.8——输同档 Qwen 两位数**。
- **与 SWE-bench 纸面对位的尖锐反差及三个调和点**：① OR 上 Lightning 为 `20260807` 版（早于 10 月 MT 论文），论文 64.5 系"MT 配方+自家 agentic SFT 后"数字，OR 所测可能非同版；**3.5 Super 纯文本版（论文主角）完全不在 AA 覆盖**（仅 VL 变体）；② 口径不同（AA coding=非 agentic 代码基准、AA agentic=自家套件固定 harness，跨 harness 差 +13 自证）；③ 3 Ultra intelligence 11.8 连 Kimi K3 Low 17.4 都不如——无法全赖口径，真实画像=**不做榜单优化 post-training、不提交榜单，第三方指数自然难看**（与卖卡+卖栈的商业模式自洽）。**11-12 复读 3 Ultra TR 后修订**：其同侪表自报 GPQA 87.0/MMLU-Pro 86.8（§2.8），AA 11.8 主要是**开放权重第三方测量的伪影**（NVFP4 权重出栈的 serving 量化/模板/推理触发）——AA 指数对开放权重降级为仅趋势参考；真实水平以同侪表为准=中游偏下、非幻觉第一。
- **引用规则（本轮最重要的操作性结论）**：Nemotron **配方结论可抄**（消融=同协议相对比较，不受影响）**、效果数字不引用**（绝对分数=自家协议+特定版本，第三方口径下同款家族明显更弱）。
- **"配方垃圾？"质询（2026-11-12，成立与否）**：不成立——第三方分数 = 基座 × 配方 × post-training × 榜单优化四层相乘，Nemotron 分数差主要打击第 1/4 层（3 Ultra 无 MT 配方时 intelligence 已 11.8；不提交榜单不做调优），配方层外部零验证（主角模型不在 AA 覆盖、OR 版本早于配方）。**反证**：配方骨架（自然 SWE 数据+issue-PR+轨迹+长上下文+回放）系全行业收敛做法（Qwen3-Coder-Next 256K 数 T token MT/Kimi-Dev 150B/GLM-5 160B SWE），同骨架他家第三方分数不差（Kimi K2 SWE-V 71.2/GLM-5.3 agentic 53.1/Qwen3.8 Max 56.0）——骨架不是问题。**真内核三条**：① 从未被证明最优（手调+无搜索+无剂量扫描）；② 跨 base/跨域迁移未验证（#133：30B agentic 域≠3B-A1 数学域，照抄前必自验）；③ 绝对分数不可当目标线。**定位=设计空间的一个人手调样本点+方法论工具箱，非标准答案**（结构/LR/mask/桥协议直接抄，配比数值当先验，效果自测）。

## 附录：论文 URL

- DeepSeek-V4: https://arxiv.org/abs/2606.19348
- GLM-5: https://arxiv.org/abs/2602.15763
- Qwen3: https://arxiv.org/abs/2505.09388
- Nemotron 3 Ultra: https://arxiv.org/abs/2606.15007
- Nemotron 3 Super: https://arxiv.org/abs/2604.12374
- Nemotron 3.5 Super/Lightning: https://arxiv.org/abs/2610.38095
- Singhal et al.（MT 完整配方伴生论文，#161）: https://arxiv.org/abs/2610.22019
