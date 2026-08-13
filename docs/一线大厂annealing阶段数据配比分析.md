# 一线大厂 Annealing 阶段数据配比分析

> 分析日期：2026-08-10  
> 筛选标准：仅含一线大厂（Google、Microsoft、Alibaba、DeepSeek、Moonshot）2025+ 报告 + 学术界广泛认可的方法  
> 不含：中小团队（OLMo、SmolLM2 等）、2024 年及更早的报告

---

## 一、一线大厂 Annealing 阶段做法汇总

### 1. Qwen3（Alibaba, 2025-05, arXiv:2505.09388）

**三阶段课程**，S2 功能上等同于 annealing：

| 阶段 | tokens | 重点 | 序列长度 |
|---|---|---|---|
| S1 通用 | >30T | 语言能力 + 世界知识，119 语言 | 4K |
| **S2 推理** | **~5T** | **提升 STEM/代码/推理/合成数据比例 + 加速 LR decay** | 4K |
| S3 长上下文 | 数百B | 高质量长文档，75% 16K-32K + 25% 4K-16K | 32K |

**数据选择方法**：
- 多语言标注系统：30T+ tokens，维度含 educational value / fields / domains / safety
- **实例级数据混合优化**（对比 DoReMi/DoGe/RegMix 的域级优化）
- 用 Qwen2.5/Math/Coder 生成 trillions of tokens 合成数据（textbooks, QA, instructions, code）

**局限**：无具体配比百分比，无消融数字

---

### 2. Phi-4（Microsoft, 2024-12, arXiv:2412.08905）

**两阶段策略**：Phase 1（web）→ Phase 2（synthetic + ultra-filtered web）

**最终整体配比**（§3.2, Table 5）：

| 数据源 | 占比 | 唯一 tokens | epoch |
|---|---|---|---|
| Synthetic | **40%** | 290B | **13.8** |
| Code | **20%** | 820B | 2.4 |
| Web | **15%** | 1.3T | 1.2 |
| Web rewrites | **15%** | 290B | 5.2 |
| Acquired sources | **10%** | 580B | 1.7 |

**关键消融发现**：
- **合成数据重复 12 epoch > 4 epoch + 更多新 web tokens**（"despite many epochs, we do not see overfitting"）
- 纯合成数据：MMLU +0.8, MATH +4.9, HumanEval +12.1，但 **TQA -14.8**（知识基准下降）
- **有机问题 > 合成问题**（用于播种）
- 合成数据 50 种类型，~400B unweighted tokens

**Midtraining**：250B tokens，**30% 长上下文 + 70% 回忆 tokens**，RoPE 250K，LR 降低 10 倍

**"Textbook" 哲学**：合成数据作为"spoonfeeding"——结构化、渐进式、与推理上下文对齐

---

### 3. DeepSeek V3（DeepSeek, 2024-12, arXiv:2412.19437）

**训练配方**：
- 预训练 14.8T tokens
- **500B annealing 阶段**，constant 低 LR

**数据方向**：相比 V2，提升数学和编程样本比例 + 扩展多语言覆盖

**局限**：无数据配比细节，仅定性描述

---

### 4. Kimi K2（Moonshot, 2025-07, arXiv:2507.20534）

**Cooldown 阶段 + 复述增强**：

| 策略 | SimpleQA Acc |
|---|---|
| 原始 wiki 重复 10 轮 | 23.76 |
| 复述1次 × 重复10轮 | 27.39 |
| **复述10次 × 训练1轮** | **28.94** |

**结论**：多样复述 > 简单重复，每个语料最多复述2次

---

### 5. Kimi K3（Moonshot, 2026-07）

**Cooldown 阶段（256K → 1M）**：
- 上采样长文档
- 合成长上下文数据：排列/拼接多模态文档与子任务
- 渐进式上下文扩展：8K → 64K → 256K → 1M

---

### 6. Gemma 3（Google DeepMind, 2025-03, arXiv:2503.19786）

**关键发现：无 annealing 阶段**

- 14T tokens（27B），无 annealing/cooldown/curriculum 描述
- 唯一的"后期"操作：32K → 128K 上下文扩展（**架构调整，非数据策略**）
- 无配比百分比，无命名数据源
- 质量加权参考 Sachdeva et al. [2024]（"How to train data-efficient LLMs"）
- 语言平衡策略：Unimax（Chung et al. 2023）
- 知识蒸馏：256 logits/token 从 teacher 采样

**结论**：Google 对数据策略高度保密

---

### 7. Phi-4-reasoning（Microsoft, 2025-04, arXiv:2504.21318）

**关键发现：后训练报告，无 annealing 内容**

但有一个重要的数据混合方法论发现：

**"数据混合的加法属性"**：
> "mixtures could be optimized independently for each domain—such as math and code—and then combined by simply concatenating their respective weights. The resulting composite mixture preserved the domain-specific gains achieved during isolated tuning."

→ **各领域可以独立优化配比，然后直接拼接**，不需要联合优化所有领域。这大幅简化了搜索空间。

**"Teachable" 数据选择**：
- 筛选处于模型能力边缘的样本（不是太简单也不是太难）
- 用弱模型与强模型的 agreement rate 估计难度
- 只保留"有改进空间"的种子

---

## 二、跨大厂交叉验证

### 被多家一线大厂独立采用的做法（= 广泛认可）

| 做法 | Qwen3 | Phi-4 | DeepSeek V3 | Kimi K2/K3 | Gemma 3 |
|---|:-:|:-:|:-:|:-:|:-:|
| **两阶段训练（bulk → annealing）** | ✅ S1→S2 | ✅ Phase1→2 | ✅ 500B annealing | ✅ cooldown | ❌ |
| **后期提升 STEM/代码/推理比例** | ✅ S2 | ✅ Phase2 synthetic | ✅ "提升数学/编程" | — | ❌ |
| **后期使用合成数据** | ✅ S2 | ✅ Phase2 40% synthetic | — | ✅ 复述增强 | ❌ |
| **长上下文数据在最后阶段** | ✅ S3 32K | ✅ Midtraining 16K | — | ✅ 256K→1M | ✅ 32K→128K |
| **加速/降低 LR in annealing** | ✅ 加速 decay | ✅ LR 降 10x | ✅ constant 低 LR | — | ❌ |
| **小模型消融定配比** | ✅ instance-level | ✅ 7B→14B rank | — | ✅ 小模型消融 | ❌ |
| **LLM 标注 → 分类器 → 筛选** | ✅ 多语言标注 | ✅ 10^6 标注 | — | — | ✅ 质量加权 |

### 仅一家大厂做（参考价值较低）

| 做法 | 来源 | 说明 |
|---|---|---|
| 复述增强（10x1 > 1x10） | Kimi K2 | 提升约 22% SimpleQA |
| 70% 回忆 tokens in midtraining | Phi-4 | 防遗忘 |
| 实例级配比优化 | Qwen3 | 对比 DoReMi/RegMix 的域级 |
| Unimax 语言采样 | Gemma 3 | 参考 Chung et al. 2023 |
| 知识蒸馏（256 logits/token） | Gemma 3 | 预训练 + 后训练 |

---

## 三、学术界广泛认可的方法

### 1. 两阶段训练 + annealing（被全部大厂采用）

理论来源：Chinchilla（Hoffman et al. 2022）证明数据质量在超越一定规模后比数量更重要。MiniCPM（2024）提出 WSD（Warmup-Stable-Decay）调度器，在 decay 阶段使用高质量数据。

**状态**：业界标准做法，被 Qwen3、Phi-4、DeepSeek V3、Kimi K2/K3 全部采用。

### 2. LLM 标注 → 轻量分类器 → 大规模筛选

被 Qwen3（多语言标注系统）、Phi-4（10^6 LLM 标注）、Gemma 3（质量加权，参考 Sachdeva et al.）采用。学术上源自 DCLM（Stanford/Berkeley, arXiv:2406.11794）的 FastText 分类器方法，被多个模型采用。

**状态**：业界标准数据选择方法，学术+工业双重背书。

### 3. 小模型 proxy 做配比消融

被 Qwen3（"extensive ablation experiments on small proxy models"）、Phi-4（7B→14B rank correlation）采用。

**学术警告**：
- Phase Transitions（NeurIPS'25 Spotlight, 清华）证明：proxy 低于相变阈值时完全无信号
- Proxy Reliability（ICLR'26, Princeton/Berkeley/Stanford）证明：需要极低 LR 才有 rank correlation > 0.95
- CAMEL（ICML'26, 清华）证明：不同能力的容量分配比 R 不同，proxy 最优配比 ≠ target 最优配比

**状态**：业界采用但学术界有重大保留。Phi-4 的 rank correlation 证据（7B→14B）是最强的正面证据，但前提是 proxy 足够大（7B 级别，不是 1M）。

### 4. 数据混合的加法属性（Phi-4-reasoning）

> "mixtures could be optimized independently for each domain and then combined by concatenating their respective weights"

**意义**：如果此属性在预训练 annealing 阶段也成立，则 198 维搜索空间可分解为独立的 per-domain 优化，大幅降低搜索复杂度。

**状态**：Microsoft 在 SFT 阶段验证。是否适用于预训练 annealing 需要验证，但这是一个高价值的假设。

### 5. Scaling law 预测超参（Qwen3 + DeepSeek LLM）

Qwen3 用 scaling law 预测每个阶段的 LR 和 batch size。DeepSeek LLM 发现数据质量影响 scaling law 的 a/b 指数分配。

**状态**：学术+工业双重背书。

### 6. 高质量数据留到 annealing 阶段

理论支撑：Chinchilla 的数据质量 > 数量结论 + MiniCPM 的 WSD 调度器。

**状态**：业界标准做法。

---

## 四、对 QuaDMix 在 Annealing 阶段的启发

### 核心矛盾

| 维度 | QuaDMix 当前 | 一线大厂 annealing 做法 |
|---|---|---|
| Proxy 规模 | 1M（已知不可用） | Phi-4: 7B；Qwen3: 小模型消融 |
| 信号源 | val_loss（已知不预测推理性能） | 直接评估 benchmark |
| 搜索空间 | 198 维（22域×5质量） | Phi-4: ~5 数据源；"加法属性"可分解 |
| 训练阶段 | 预训练配比算法 | annealing 阶段配比 |
| 数据选择 | 域级配比 | Qwen3: 实例级；Phi-4: "teachable" 筛选 |

### 可操作建议（按优先级）

**1. 验证"加法属性"是否适用于 annealing（最高 ROI）**

Phi-4-reasoning 发现各领域可以独立优化配比然后拼接。如果这在 annealing 阶段也成立：
- 198 维搜索 → 分解为 ~22 个独立的一维搜索
- 每个 domain 独立做 micro-annealing（10B tokens 级别）
- 成本大幅降低

**验证方法**：取 2 个域（如 math + code），分别做独立 annealing + 联合 annealing，比较效果。

**2. 用目标模型直接做 micro-annealing 替代 1M proxy**

一线大厂的做法是直接用目标模型（Phi-4 用 7B→14B rank correlation，Qwen3 用小模型消验）。不用 1M proxy。

**具体做法**：
- 从当前预训练 checkpoint 出发
- 每个候选数据源做 10B token 微退火（LR 线性衰减到 0）
- 直接评估 benchmark（GSM8K、MMLU 等）
- 信号好的加入最终 annealing mix

**3. 后期提升 STEM/代码/合成数据比例（跨大厂共识）**

Qwen3、Phi-4、DeepSeek V3 都在 annealing 阶段提升 STEM/代码/合成数据比例。

**4. 高质量数据留到最后（跨大厂共识 + 学术背书）**

全部一线大厂（除 Gemma 3）都将最高质量数据放在 annealing 阶段。

**5. 合成数据可以高重复（Phi-4 发现）**

Phi-4 的 synthetic 数据重复 13.8 epoch 不过拟合，且 12 epoch > 4 epoch + 更多 web tokens。如果使用合成数据，不必担心重复。

**6. 复述增强（Kimi K2 发现）**

复述 10 次 × 训练 1 轮 > 原始重复 10 轮（+22% SimpleQA）。如果数据量不足，复述比简单重复有效。

---

## 五、一线大厂 annealing 配比数据总结

| 模型 | annealing tokens | 数据方向 | 有百分比吗 |
|---|---|---|---|
| Qwen3 S2 | ~5T | STEM/代码/推理/合成 ↑ | ❌ |
| Phi-4 Phase2 | ~整体 10T | synthetic 40% / code 20% | ⚠️ 整体配比，非 Phase2 专属 |
| DeepSeek V3 | 500B | 数学/编程 ↑ | ❌ |
| Kimi K2 | — | 复述增强 | ✅ 复述消融 |
| Kimi K3 | — | 长上下文合成 | ❌ |
| Gemma 3 | — | 无 annealing | — |

**结论**：一线大厂在 annealing 阶段的数据配比细节普遍不披露。最详细的是 Phi-4 的整体配比（40% synthetic / 20% code / 15% web / 15% web-rewrites / 10% acquired）和 Kimi K2 的复述消融。其余只有定性描述。

---

## 六、相关文件索引

| 文件 | 说明 |
|---|---|
| `docs/业界模型技术报告_数据配比分析.md` | Kimi K3 + DeepSeek 全系列分析（含 DeepSeekMath） |
| `docs/小Proxy不可用_多论文交叉验证总结.md` | 6 篇论文验证 1M proxy 不可用 |
| `docs/PhaseTransitions_深度分析_v2.md` | 相变现象详细分析 |
| `docs/Proxy_Reliability_论文总结.md` | Proxy 可靠性总结 |
| `docs/DomainPilot_深度分析.md` | 域级 loss 轨迹方法 |
| `/home/liujin99/quadmix/` | QuaDMix 代码仓库 |
