# 头部厂商 Tech Report 数据配比深度分析

> 聚焦 DeepSeek-V4 / GLM-5 / Qwen3 / Nemotron 3 Ultra 四篇头部厂商 frontier 规模技术报告，提取中训练数据配比的可操作洞察。

---

## 论文总览

| 厂商 | 论文 | 规模 | 预训练 tokens | 时间 | arXiv |
|------|------|------|-------------|------|-------|
| **DeepSeek-AI** | DeepSeek-V4 | 1.6T/49B + 284B/13B MoE | 32T + 33T | 2026.4 | 2606.19348 |
| **智谱 AI + 清华** | GLM-5 | 744B/40B MoE | 28.5T | 2026.2 | 2602.15763 |
| **Alibaba Qwen** | Qwen3 | 235B/22B MoE (最大) | 36T | 2025.5 | 2505.09388 |
| **NVIDIA** | Nemotron 3 Ultra | 550B/55B MoE | 20T | 2026.6 | 2606.15007 |

**筛选标准**：业界公认 AI 头部厂商 + frontier 规模 + 2025-2026 新发布。

---

## 1. GLM-5（智谱 AI + 清华, 744B/40B, 28.5T tokens）

> arXiv: 2602.15763v1 | 2026.2 | Zhipu AI & Tsinghua University

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

---

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
3.5 Base 不从头训：从 3 系 Base 初始化后**蒸馏 1T tokens**（Lightning←Super 3 Base：top-128 logits、forward KL 0.9+CE 0.1+sink token；Super←Ultra 3 Base：Skew Reverse KL α=0.1；LR 1e-5→1e-4（10% peak）→线性回 1e-5；MoE load-balance 系数 1e-5）——"换代=蒸馏"的低成本模型更新路径，蒸馏轴备查。

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

- **官方榜上无名**：swebench.com Verified 榜 75 个模型条目中**零 Nemotron 3.5 条目**——NVIDIA 从未提交该系列（历史唯一 NVIDIA 相关提交=2025-04 Nemotron-CORTEXA agent 脚手架，挂别家底模，非模型本体）；Terminal-Bench 榜 JS 渲染无法静态核查。印证"TR 只跟自己比"的不打榜定位。
- **纸面对位**（⚠️ 协议不同仅可粗看）：榜头 Claude 4.5 Opus 79.2 / Doubao-Seed-Code 78.8 / Gemini 3 Pro 77.4 / GPT-5 74.4；最强开源提交 Kimi K2 71.2 / Qwen3-Coder-480B 69.6 / GLM-4.6 68.2。Super 74.3（自家 OpenHands 3-rollout 口径）纸面 ≈GPT-5 档、高于全部开源提交；Lightning 64.5 ≈GLM-4.5（64.2）档，**同规格唯一对位条目 Qwen3-Coder-30B-A3B 60.4**（纸面 +4.1，但系 2025-02 老提交不可直接比）。
- **打折理由**：榜单各条目 harness/尝试次数不同（头部基本 Attempts 2+）；跨 harness 差可达数分（本文 Table 7 自显最大 +13.4）——纸面对位只可读作"不落后于开源第一梯队"的弱证据。**画像结论：中上水平+最透明披露，非榜上王者；参考答案价值与此排位无关。**

### 7.6 第三方评测核查（2026-11-12，Artificial Analysis 三指数，经 OpenRouter 板块）

- **可得性**：LMArena / Terminal-Bench 均 JS 渲染无法静态抓取；OpenRouter rankings 页内嵌 AA 三面板（intelligence 57 模型 / coding 153 / agentic 110）。
- **Nemotron 全家三指数全线中下-垫底**：intelligence——3.5 Super VL #41（13.6）、3 Ultra #45（11.8）、3.5 Lightning #51/57（5.5），头部 Claude Fable 5.2 64.6；coding——3 Ultra #65（49.3）、3 Super #86（37.7）、3.5 Lightning #100（26.8），头部 Fable 5.1 81.6；agentic——3 Ultra #49（20.1）、3.5 Lightning #74（3.5）、3 Super #79（1.7），头部 Fable 5.1 57.9。**同规格对位（30B-A3B 级，coding）：Qwen3.5-35B-A3B 37.0 / Qwen3.6-35B-A3B 41.9 vs Lightning 26.8——输同档 Qwen 两位数**。
- **与 SWE-bench 纸面对位的尖锐反差及三个调和点**：① OR 上 Lightning 为 `20260807` 版（早于 10 月 MT 论文），论文 64.5 系"MT 配方+自家 agentic SFT 后"数字，OR 所测可能非同版；**3.5 Super 纯文本版（论文主角）完全不在 AA 覆盖**（仅 VL 变体）；② 口径不同（AA coding=非 agentic 代码基准、AA agentic=自家套件固定 harness，跨 harness 差 +13 自证）；③ 3 Ultra intelligence 11.8 连 Kimi K3 Low 17.4 都不如——无法全赖口径，真实画像=**不做榜单优化 post-training、不提交榜单，第三方指数自然难看**（与卖卡+卖栈的商业模式自洽）。
- **引用规则（本轮最重要的操作性结论）**：Nemotron **配方结论可抄**（消融=同协议相对比较，不受影响）**、效果数字不引用**（绝对分数=自家协议+特定版本，第三方口径下同款家族明显更弱）。

## 附录：论文 URL

- DeepSeek-V4: https://arxiv.org/abs/2606.19348
- GLM-5: https://arxiv.org/abs/2602.15763
- Qwen3: https://arxiv.org/abs/2505.09388
- Nemotron 3 Ultra: https://arxiv.org/abs/2606.15007
- Nemotron 3 Super: https://arxiv.org/abs/2604.12374
- Nemotron 3.5 Super/Lightning: https://arxiv.org/abs/2610.38095
- Singhal et al.（MT 完整配方伴生论文，#161）: https://arxiv.org/abs/2610.22019
