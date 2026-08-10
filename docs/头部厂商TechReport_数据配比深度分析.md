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

## 附录：论文 URL

- DeepSeek-V4: https://arxiv.org/abs/2606.19348
- GLM-5: https://arxiv.org/abs/2602.15763
- Qwen3: https://arxiv.org/abs/2505.09388
- Nemotron 3 Ultra: https://arxiv.org/abs/2606.15007
- Nemotron 3 Super: https://arxiv.org/abs/2604.12374
