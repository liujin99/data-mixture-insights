# PreSelect 深度分析

> Predictive Data Selection: The Data That Predicts Is the Data That Teaches
> ICML 2025 | HKUST + Vivo AI | arXiv:2503.00808

---

## 1. 论文核心思想

**一句话总结**：如果某些数据上的模型 loss 能预测下游任务表现，那这些数据就是好的训练数据。

### 1.1 核心假设

来自 Huang et al. (2024) "Compression Represents Intelligence Linearly" 的发现：
- 不同模型在某些原始文本上的归一化 loss（压缩效率）与下游 benchmark 表现**强相关**
- 例如：GitHub 代码上的 loss ↔ 代码任务表现；Common Crawl 上的 loss ↔ 知识任务表现
- 但这种相关性**不是普遍存在的**——CC loss 与非知识任务的相关性很弱

**PreSelect 的推论**：压缩能代表智能的数据 → 适合学习该智能的数据。

### 1.2 Predictive Strength 定义

给定 N 个开源模型 {M₁, ..., Mₙ}，按下游平均分排序 S₁ < S₂ < ... < Sₙ。

对任意文档 d，计算每个模型的归一化 loss（bits per character）{C₁, ..., Cₙ}。

Predictive Strength：

```
S = Σ_{i<j} I{Cᵢ > Cⱼ} / Z
```

其中 Z = N(N-1)/2 是归一化因子，S ∈ [0, 1]。

**直觉**：如果弱模型的 loss 总是大于强模型的 loss（即 loss 排序与能力排序完全反向对齐），则 S = 1，说明该文档的 loss 完美预测模型能力。

### 1.3 为什么用 Rank 而非 Pearson

- 单文档 loss 计算噪声大（尤其短文档）
- Pearson 对数值变化敏感，不稳定
- Rank-based 更鲁棒：只关心排序，不关心绝对值
- 实验验证：PPL Correlation (DD) 用域级 Pearson 比 PreSelect 差 4.8%

---

## 2. 完整方法流程

### 2.1 Seed Set 构建

| 参数 | 值 |
|------|-----|
| 域数量 | 3,000（最高频域） |
| 每域样本数 | 300 |
| 总 seed 大小 | **900K 文档** |
| 来源 | RefinedWeb（与训练数据分离） |

### 2.2 模型选择

| 模型 | 参数量 |
|------|--------|
| Llama-1 7B | 7B |
| Llama-1 13B | 13B |
| Llama-1 33B | 33B |
| Llama-1 65B | 65B |
| Llama-2 7B | 7B |
| Llama-2 13B | 13B |

**共 6 个模型，全部来自 Llama 族**。

**关键设计决策**：只用同族模型！
- 跨族模型引入大量噪声（不同 prompt 敏感性、不同 tokenizer）
- 初步实验发现跨族模型选的 data 打不过 random baseline
- 这与 Thrush et al. (Perplexity Correlation) 用 ~90 个不同族模型形成对比

### 2.3 下游 Score 来源

使用 Huang et al. (2024) 提供的 **12 个 benchmark 平均分**作为模型能力排序。

也探索了用单一任务（如 HellaSwag）作为目标，见 Appendix A.7.2。

### 2.4 FastText 分类器训练

| 参数 | 值 |
|------|-----|
| 正样本 | ~200K（最高 predictive strength 的文档） |
| 负样本 | ~200K（最低 predictive strength 的文档） |
| 特征 | bi-gram |
| 部署阈值 | 0.5（保留 score > 0.5 的文档） |

### 2.5 部署

只需 fastText 分类器，零 GPU 开销：

```python
from datatrove.pipeline.filters import FastTextClassifierFilter
FastTextClassifierFilter("PreSelect-classifier.bin", keep_labels=[("1", 0.5)])
```

---

## 3. 实验结果详解

### 3.1 RefinedWeb 主实验

| 规模 | 方法 | Tokens | Avg (7 tasks) | Math BPC ↓ | Code BPC ↓ |
|------|------|--------|---------------|------------|------------|
| 400M | Random | 8B | 24.2 | 1.224 | 1.111 |
| 400M | DCLM | 8B | 25.7 | 1.031 | 0.929 |
| 400M | **PreSelect** | **8B** | **27.0** | **0.995** | **0.885** |
| 1B | Random | 30B | 28.1 | 1.023 | 0.901 |
| 1B | DCLM | 30B | 31.2 | 0.857 | 0.773 |
| 1B | **PreSelect** | **30B** | **33.4** | **0.830** | **0.744** |
| 1B | Random | 300B | 31.3 | 0.892 | 0.804 |
| 3B | Random | 100B | 34.7 | 0.818 | 0.726 |
| 3B | DCLM | 100B | 37.8 | 0.712 | 0.664 |
| 3B | **PreSelect** | **100B** | **39.5** | **0.694** | **0.648** |

**关键发现**：
- 1B/30B (PreSelect) > 1B/300B (Random) → **10x 计算缩减**
- 1B/30B (PreSelect) > 1B/300B (Vanilla) → 选完数据后比训全量还好
- 3B 上优势持续：+1.7 vs DCLM, +4.8 vs Random

### 3.2 1B 全 Baseline 对比（30B tokens, 15 benchmarks 平均）

| 方法 | Avg | vs Random |
|------|-----|-----------|
| Random | 28.1 | — |
| PPL Filtering (CCNet) | 29.1 | +1.0 |
| PPL Correlation (DD) | 28.6 | +0.5 |
| PPL Correlation (DP) | 29.3 | +1.2 |
| FineWeb-Edu | 31.1 | +3.0 |
| DCLM | 31.2 | +3.1 |
| **PreSelect** | **33.4** | **+5.3** |

### 3.3 C4 + Pythia 跨架构验证

| 规模 | 方法 | Avg |
|------|------|-----|
| 410M/25B | Random | 42.7 |
| 410M/25B | DSIR | 42.5 |
| 410M/25B | DsDm | 43.4 |
| 410M/25B | QuRating | 43.5 |
| 410M/25B | MATES | 44.0 |
| 410M/25B | **PreSelect** | **44.9** |
| 1B/25B | Random | 44.7 |
| 1B/25B | MATES | 45.8 |
| 1B/25B | **PreSelect** | **46.8** |

### 3.4 选择比例消融

| 比例 | 方法 | Tokens | Avg |
|------|------|--------|-----|
| 10% | DCLM | 30B | 31.2 |
| 10% | PreSelect | 30B | 33.4 |
| 30% | DCLM | 90B | 32.6 |
| 30% | PreSelect | 90B | 34.0 |

10% 选择比例下 PreSelect 优势更大（+2.2 vs DCLM），30% 时差距缩小（+1.4）。

---

## 4. FastText 特征分析

### 4.1 PreSelect 学到的正/负特征

**正特征（高质量数据标志）**：
```
ˆ (337.7), Likes (241.4), share|improve (241.3), MIT (240.6),
Retrieved (233.0), Received (214.9), Photo: (205.8), Azure (200.5),
Tolkien (195.0), Math (193.7), answer (187.7), Contributed (185.5),
Cite (184.7), MATLAB (177.0)
```

→ 学术/技术内容：Stack Overflow、学术论文、技术文档、百科全书

**负特征（低质量数据标志）**：
```
Register (-375.4), Details (-335.2), Comments (-320.8), Updated: (-310.8),
COVID-19 (-303.9), tho (-281.7), Print (-271.0), Profile (-262.1),
Leave (-258.9), Published: (-235.2), Product (-228.6), coronavirus (-227.2),
2020 (-223.7), By: (-222.9), Shipping (-219.5)
```

→ 网页噪声：注册页、评论区、产品页、新闻时效性内容

### 4.2 与 DCLM 特征对比

**DCLM 正特征**：
```
' (968.2), Why (671.6), Answer: (620.4), Additionally (599.6),
assistant. (567.3), Edit: (565.8), Generate (537.1), answer. (502.9),
Given (474.2), A: (457.4), Question: (447.8), Write (440.7),
task. (423.9), ELI5 (419.0), AI (403.3)
```

→ **SFT 风格**：问答对、指令-回复格式、对话标记

**关键差异**：
- DCLM 用 SFT 数据做正样本 → 偏好"像 SFT 数据"的预训练文本
- PreSelect 用 predictive strength → 偏好"能反映模型能力"的文本
- PreSelect 选出的数据更"自然"（学术、技术），DCLM 选出的更"结构化"（问答格式）

### 4.3 PPL Correlation (DP) 特征对比

**PPL Correlation 正特征**：
```
Archives (347.5), Navigation (299.7), Commenting (288.5),
Section (280.3), E-mail (276.1), Played: (263.9), Lyrics (240.4)
```

→ **网页结构标记**，而非内容质量标记。域级选择导致 fastText 学到了网页结构而非内容质量。

---

## 5. 与 QuaDMix 的系统对比

### 5.1 方法层次对比

| 维度 | PreSelect | QuaDMix |
|------|-----------|---------|
| **选择粒度** | 文档级（每个文档独立打分） | 域级（22 个 Dewey Decimal 域的配比） |
| **质量维度** | 1 维（predictive strength） | 5 维（dclm, fineweb_edu, english, math_general, math_openweb） |
| **优化目标** | 选 top-k 文档 | 优化域配比 + 质量权重 + 采样曲线 |
| **搜索空间** | 1 维阈值 | 198 维（22 域 × 9 参数） |
| **代理模型** | 6 个开源 Llama（7B-65B） | 14M proxy（2-layer LLaMA） |
| **部署工具** | fastText 分类器 | LightGBM surrogate + 搜索 |
| **是否训练 proxy** | ❌ 只用现成模型 | ✅ 训练 proxy 模型 |

### 5.2 质量信号对比

| 质量信号 | PreSelect | QuaDMix | 重叠？ |
|----------|-----------|---------|--------|
| DCLM | ❌ 不用 | ✅ `qs_dclm` | — |
| FineWeb-Edu | ❌ 不用 | ✅ `qs_fineweb_edu_approx` | — |
| English | ❌ 不用 | ✅ `qs_english` | — |
| Math (general) | ❌ 不用 | ✅ `qs_eai_general_math` | — |
| Math (open web) | ❌ 不用 | ✅ `qs_eai_open_web_math` | — |
| **Predictive Strength** | ✅ 核心 | ❌ 没有 | **新增维度** |

**结论**：PreSelect 的 predictive strength 与 QuaDMix 的 5 个质量信号**完全不重叠**，是独立的第 6 个质量维度。

### 5.3 质量信号分布对比

QuaDMix 的 5 个 FastText 分类器分布特征（来自 `analyze_quality_positive.py`）：

| 分类器 | 分布 | 区分度 |
|--------|------|--------|
| dclm | 95% 近 0，5% 高 | 弱 |
| fineweb_edu | 均匀 [0, 3.94] | **强** |
| english | 集中 ~0.89 | 弱 |
| math_general | 95% 近 0 | 弱 |
| math_openweb | 中等分散 | 中 |

所有两两相关性 < 0.37 → 5 个信号基本独立。

**Predictive strength 预期特征**：
- 基于 6 个模型的 loss 排序一致性
- 应该与 fineweb_edu 有一定相关性（都偏好高质量内容）
- 但机制不同：fineweb_edu 偏好教育性，predictive strength 偏好"能区分模型能力"的文本
- 预期与现有 5 个信号相关性也 < 0.5

### 5.4 互补性分析

```
PreSelect 解决：哪些文档值得训？（文档级质量筛选）
QuaDMix 解决：各域训多少？怎么采样？（域级配比优化）
```

两者是**不同层次的问题**，可以叠加：
1. 先用 PreSelect 做文档级筛选（或作为第 6 个质量特征）
2. 再用 QuaDMix 做域级配比优化

### 5.5 QuaDMix 集成方案

**方案 A：作为第 6 个质量特征（推荐）**

```python
# constants.py 修改
QUALITY_NAMES = ["dclm", "fineweb_edu", "english", "math_general", "math_openweb", "preselect"]
QUALITY_COLUMNS = [..., "qs_preselect"]
FASTTEXT_FIELDS = [..., "preselect"]
```

- 下载 PreSelect 的 fastText 分类器（已开源：`hkust-nlp/preselect-fasttext-classifier`）
- 对全量数据跑一遍 fastText 打分
- 加入 QuaDMix 的 quality_matrix，从 [N, 5] 变为 [N, 6]
- LightGBM 自动学习第 6 维的权重

**成本**：
- fastText 推理：CPU only，~900K 文档/小时（单核），275M 文档约 300 CPU-hours
- 搜索空间：从 198 维变为 220 维（22 域 × 10 参数），增加 10%
- 预期收益：+1-2%（基于 PreSelect 在 1B/3B 上的优势）

**方案 B：作为预过滤步骤**

- 先用 PreSelect 过滤掉低质量文档（阈值 0.5）
- 在过滤后的数据池上运行 QuaDMix
- 优点：减少 QuaDMix 处理的数据量
- 缺点：丢失了 QuaDMix 对"低 PreSelect 分但高其他质量分"文档的利用

**方案 C：作为验证集替代**

- PreSelect 的 predictive strength 本质是"loss 与下游表现的相关性"
- 可以用来构建更好的验证集：选择 predictive strength 高的文档作为验证集
- 替代 OpenHermes-10k（SFT 数据，与预训练分布不匹配）
- 这与 Google MDE 的发现一致：验证集选择比方法选择更重要（ρ=0.844 vs 0.282）

---

## 6. 关键实验洞察

### 6.1 同族模型至关重要

- 用 6 个 Llama 模型 → 效果好
- 用跨族模型（不同家族）→ 打不过 random
- 原因：不同族模型对 prompt 敏感性不同，base model 尤其敏感
- **对 QuaDMix 的启示**：如果要用类似方法，proxy 模型应与目标模型同架构族

### 6.2 文档级 vs 域级

| 方法 | 粒度 | 1B Avg |
|------|------|--------|
| PPL Correlation (DD) | 域级 | 28.6 |
| PPL Correlation (DP) | 域级→页级 | 29.3 |
| **PreSelect** | **文档级** | **33.4** |

域级选择的问题：
- 高相关域中也包含低质量文档
- 低相关域中也包含高质量文档
- 整域保留/丢弃 → 多样性损失（HellaSwag 显著下降）

### 6.3 正/负数据分析

论文分析了 PreSelect 选出的正/负数据分布：
- 正数据：更多学术网站、技术文档、百科全书
- 负数据：更多社交媒体、新闻、产品页、论坛水贴
- 长度分布：正数据更长（中位数 ~500 tokens），负数据更短（~200 tokens）

### 6.4 在高质量数据池上仍有效

Perplexity Correlation 的更新版本发现：在高质量预过滤数据池上，域级相关性方法**无法提升** random baseline。

但 PreSelect 在 RefinedWeb（已经过启发式过滤和去重）上仍然显著优于 random（+5.3%）和 DCLM（+2.2%）。

**原因**：文档级粒度避免了域级方法的多样性损失问题。

---

## 7. 局限性

### 7.1 前置成本

- 需要 6 个 Llama 模型（7B-65B）对 900K 文档做推理
- 估算：65B 模型推理 900K 文档 ≈ 数十 GPU-hours（A100）
- 但这是一次性成本，之后只需 fastText

### 7.2 模型族依赖

- 只用 Llama 1&2 计算 predictive strength
- 如果目标模型是不同架构（如 Mamba、RWKV），predictive strength 可能不适用
- 论文验证了 Pythia（GPT-NeoX 架构）上有效，但更多架构未验证

### 7.3 下游 benchmark 依赖

- 使用 12 个 benchmark 平均分作为模型能力排序
- 如果目标场景是特定任务（如代码），平均分会稀释信号
- Appendix A.7.2 探索了用单一任务（HellaSwag），效果略差

### 7.4 规模验证

- 最大验证到 3B 模型 / 100B tokens
- 未在 7B+ 模型上验证
- 10x 计算缩减的结论在 1B 上成立，更大规模未确认

---

## 8. 开源资源

| 资源 | 链接 |
|------|------|
| 代码 | https://github.com/hkust-nlp/PreSelect |
| FastText 分类器 | https://huggingface.co/hkust-nlp/preselect-fasttext-classifier |
| 筛选后数据集 (100B) | https://huggingface.co/datasets/hkust-nlp/PreSelect-100B |
| GitHub Stars | 66 |
| Forks | 10 |

---

## 9. 对 QuaDMix 的行动建议

### 9.1 立即可做（方案 A）

1. 下载 PreSelect fastText 分类器
2. 对 Essential-Web 全量数据跑 fastText 打分
3. 作为第 6 个质量特征加入 `quality_matrix`
4. 在 LightGBM surrogate 中自动学习权重

**预期收益**：+1-2%（基于 PreSelect 在 1B/3B 上 vs DCLM 的 +2.2% 优势）
**成本**：~300 CPU-hours（fastText 推理）+ 搜索空间增加 10%

### 9.2 值得探索（方案 C）

用 predictive strength 高的文档构建新验证集，替代 OpenHermes-10k：
- OpenHermes 是 SFT 数据，与预训练分布不匹配
- PreSelect 验证集来自预训练数据本身，分布一致
- 结合 Google MDE 的发现（验证集选择 ρ=0.844 vs 域 loss ρ=0.282），这可能是最高 ROI 的改进

### 9.3 不建议

- 直接用 PreSelect 替代 QuaDMix：PreSelect 只做文档级选择，不做域级配比优化
- 用 PreSelect 做预过滤（方案 B）：会丢失 QuaDMix 对低 PreSelect 分文档的利用能力
