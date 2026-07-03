# NAG: Target-Oriented Pretraining Data Selection via Neuron-Activated Graph 深度分析

**分析日期**: 2026-07-03

---

## 1. 论文概览

| 维度 | 信息 |
|------|------|
| arXiv | 2604.15706 (2026.04) |
| 机构 | **华东师范大学 + 商汤科技 + UC Santa Cruz** |
| 作者 | Zijun Wang, Haoqin Tu, Weidong Zhou, Yiyang Zhou, Xiaohuan Zhou, Bingni Zhang, Weiguo Feng, Taifeng Wang, Cihang Xie, Fengze Liu |
| 方向 | 基于神经元激活图的 target-oriented 预训练数据选择 |
| 代码 | https://github.com/asillycat/NAG |
| 与 QuaDMix 相关性 | **中** |

### 作者背景

| 作者 | 机构 | 分量 |
|------|------|------|
| **Cihang Xie** | UC Santa Cruz | ★★★★☆ 对抗鲁棒性/可解释性方向知名学者 |
| **Taifeng Wang** | 商汤科技 | ★★★☆☆ 业界资深 |
| **Fengze Liu** (通讯) | 华东师范大学 | ★★★☆☆ |
| Zijun Wang (一作) | 华东师范大学 | ★★☆☆☆ 博士生 |
| Haoqin Tu | UC Santa Cruz | ★★☆☆☆ |

---

## 2. 核心问题

**Target-oriented pretraining**：给定一个目标任务（如 HellaSwag），从大规模数据池中选择最有助于提升该任务性能的预训练数据。

与 QuaDMix 的区别：
- **QuaDMix**：领域级数据配比优化（多领域混合比例）
- **NAG**：样本级数据选择（从池中选择 top 样本）
- 两者解决的是**不同层次的问题**，但可以互补

---

## 3. 方法

### 3.1 核心思想

不用黑盒 embedding 或 loss 信号，而是直接用 LLM 内部的**神经元激活模式**来表征每个文本输入。如果两个输入激活了相似的神经元集合，说明它们在模型内部触发了相似的计算过程，因此训练数据应该对目标任务有帮助。

### 3.2 Neuron Impact（神经元影响力度量）

对于投影矩阵 W ∈ R^{d_in × d_out}，每列视为一个"神经元"。给定输入 h_in，神经元 N_k 的影响力：

```
Imp(N_k | h_in) = |h_in^T · W_{:,k}|
```

即：zeroing 该神经元后，层输出的变化量。**计算极其简单**——一次 forward pass 即可得到所有神经元的影响力。

### 3.3 Neuron-Activated Graph (NAG)

对每一层，选 top-K 个高影响力神经元，组成跨层的图结构：

```
NAG(c) = {(ℓ, k) | ℓ ∈ {1,...,L}, k ∈ TopK(Imp scores at layer ℓ)}
```

关键参数：
- **神经元比例 r_k = K/d_ℓ = 0.3%**（极稀疏，每层只选 0.3% 的神经元）
- **最佳神经元类型**：FFN UP projection（高维扩展层，信号最丰富）
- **必须跨层**：只用最后一层 → 平均性能下降 4.1%

### 3.4 数据选择

1. 对 target examples 提取 NAG，聚合为 target profile（每个神经元被激活的频率）
2. 对候选数据 c 提取 NAG
3. 计算 c 与 target profile 的相似度（Sørensen-Dice 系数的变体）
4. 按相似度排序，选 top r_f = 20%

**关键特性**：
- **Training-free**：不需要训任何额外模型
- **Target-independent extraction**：NAG 特征提取一次，可用于任意 target
- **可解释**：选出的神经元集合可以可视化、分析

---

## 4. 实验结果

### 4.1 主实验（Single-Target，1.2B 模型，30B tokens）

| 方法 | ARC-C | HellaSwag | TriviaQA | MMLU | XStoryCloze | XWinograd | Avg. |
|------|-------|-----------|----------|------|-------------|-----------|------|
| Random | 28.5% | 51.6% | 15.6% | 30.2% | 67.1% | 76.5% | 44.9% |
| FineWeb-Edu | 34.3% | 55.3% | 20.1% | 32.8% | 65.9% | 76.2% | 47.4% |
| BETR | 32.3% | 57.5% | 20.2% | 31.1% | 71.0% | 80.7% | 48.8% |
| **NAG (Qwen3-1.7B)** | **34.0%** | **60.6%** | **22.3%** | **32.2%** | 70.0% | 80.1% | **49.8%** |

- vs Random: **+4.9%**
- vs FineWeb-Edu: **+2.4%**
- vs BETR: **+1.0%**
- 在 HellaSwag 上 vs BETR: **+3.1%** (60.6% vs 57.5%)

### 4.2 Multi-Target 设置

6 个 benchmark 同时作为 target，各选 r_f/6 的数据后直接混合：

| 方法 | Avg. |
|------|------|
| Random | 44.9% |
| FineWeb-Edu | 47.4% |
| BETR | 44.4% (**-0.5%**，多目标下退化) |
| **NAG (Qwen3-1.7B)** | **48.5%** (+3.6%) |

**关键发现**：BETR 在多目标下退化（embedding 相似度在多目标时冲突），NAG 仍稳定提升。

### 4.3 与质量信号结合

NAG + FineWeb-Edu 联合排序：

| 方法 | Avg. |
|------|------|
| FineWeb-Edu alone | 47.4% |
| + NAG (SmolLM3-3B) | **49.6%** (+2.2%) |

**NAG 与质量信号正交**，可叠加使用。

### 4.4 Scaling 实验

| 设置 | HellaSwag | 提升 |
|------|-----------|------|
| 1.2B / 30B + NAG (Qwen3-1.7B) | 60.6% | +9.0% |
| **7B / 100B** + NAG (Qwen3-1.7B) | **71.4%** | **+8.4%** |
| 1.2B / 30B + NAG (Qwen3-0.6B) | 59.9% | +8.3% |

- 7B 规模上效果保持（+8.4% vs +9.0%）
- **提取模型可以比训练模型小**（0.6B 提取 → 1.2B 训练，仍优于所有 baseline）

### 4.5 计算效率

| 阶段 | 成本 |
|------|------|
| NAG 提取（150B tokens, Qwen3-1.7B） | **192 GPU-hours (H100)** |
| 排序 | CPU-only，O(N)，可忽略 |
| 下游训练效率提升 | **1.27-2.42x compute multiplier** |

**一次性成本**：NAG 提取后可复用于任意 target。BETR 每个新 target 需要重新训 classifier + 重新 forward 整个候选池。

---

## 5. 关键分析

### 5.1 为什么 NAG 有效

**发现 1：NAG 捕获了关键神经元**

| 操作 | 神经元比例 | 性能下降 |
|------|-----------|---------|
| Deactivate random neurons | 0.12% | -0.2% |
| **Deactivate NAG neurons** | **0.12%** | **-23.5%** |

仅 0.12% 的神经元被 NAG 选中，但关掉它们导致 23.5% 的性能崩塌。这证明 NAG 找到了"功能骨架"。

**发现 2：NAG 是任务判别性表征**

t-SNE 可视化显示 NAG 表征自然聚类为不同任务（MathQA 和 GSM8K 相近，与 XNLI 远离）。

**发现 3：NAG 排序与下游 utility 对齐**

随着过滤率 r_f 降低（选更少数据），NAG 性能持续上升（r_f=5% 时 +1.8%），而 Random 和 BETR 下降。说明 NAG 的排序真正反映了数据的任务价值。

### 5.2 NAG 如何运作

| 设计选择 | 效果 |
|----------|------|
| FFN UP neurons | 最佳（60.6%），高于 down_proj (58.0%) 和 k_proj (56.7%) |
| 全层 vs 最后一层 | 全层 +4.1%，最后一层在 TriviaQA/MMLU 上甚至不如 Random |
| 稀疏度 r_k = 0.3% | 最优，更大比例反而下降（→100% 时退化为 Random） |
| 更大提取模型 | 更好（1.7B < 4B < 8B，固定 r_k 下） |

---

## 6. 对 QuaDMix 的启示

### 6.1 问题层次不同

| 维度 | QuaDMix | NAG |
|------|---------|-----|
| 操作粒度 | 领域级配比 | 样本级选择 |
| 目标 | 多领域平衡 | 特定任务优化 |
| 需要 proxy | 是（14M） | 否（用 off-the-shelf LLM） |
| 需要训练 | 是（proxy + LightGBM） | **否（training-free）** |
| 可解释性 | 中（LightGBM 特征重要性） | **高（神经元可视化）** |

### 6.2 潜在互补

1. **NAG 可作为 QuaDMix 的质量信号**：NAG 论文已证明 NAG + FineWeb-Edu 叠加有效。QuaDMix 的 5 个 FastText 质量分类器可以考虑加入 NAG 信号
2. **NAG 的 multi-target 退化指出了方向**：NAG 在多目标下用简单等权混合，论文自己提到 RegMix 和 QuaDMix 可以改进多目标设置 → **QuaDMix 的领域配比优化正好填补 NAG 的短板**
3. **NAG 的跨模型迁移性验证了 proxy 思路**：0.6B 提取 → 1.2B 训练仍有效，支持"小模型信号可迁移到大模型"的假设

### 6.3 对 QuaDMix proxy 的启示

NAG 证明了：
- **FFN UP 层的神经元信号最丰富** → QuaDMix 的 proxy 如果也用 Transformer 架构，应重点关注 FFN UP 层的输出
- **跨层聚合比单层好** → proxy 信号应利用多层信息，而非只看最终 loss
- **0.3% 的稀疏度就够** → 不需要完整的模型输出，极稀疏的信号就足够

### 6.4 QuaDMix 的独特优势

NAG 不做领域配比优化。在 multi-target 设置下，NAG 用简单等权混合，而 QuaDMix 可以搜索最优配比。两者结合可能是最优方案：
1. 用 NAG 在每个领域内选择高质量样本
2. 用 QuaDMix 优化领域间的混合比例

---

## 7. 可信度评估

| 维度 | 评分 | 理由 |
|------|------|------|
| 机构 | ★★★☆☆ | 华东师范大学 + 商汤 + UCSC，非顶级 LLM 实验室 |
| 理论 | ★★★☆☆ | 无收敛保证或理论 bound，纯实证 |
| 实验 | ★★★★☆ | 6 个 benchmark × 3 个 backbone × 2 个 target 设置 + 7B scaling |
| 创新性 | ★★★★☆ | 神经元级数据选择是新颖视角，可解释性强 |
| 实用性 | ★★★★☆ | training-free，一次性提取可复用，但需要 off-the-shelf LLM |
| Baseline 对比 | ★★★☆☆ | 只比了 Random、FineWeb-Edu、BETR，缺少 DoReMi/RegMix/QuaDMix 等 |

### Baseline 对比弱点

| 缺失的 Baseline | 为什么应该比 |
|----------------|------------|
| DoReMi | data mixing 标准 baseline |
| RegMix | 论文自己提到了 |
| QuaDMix | 论文自己引用了 |
| QuRating | 同类 quality-based 方法 |
| DSIR | 同类 target-oriented 方法 |
| DCLM | 同类 data selection 方法 |

---

## 8. 局限

1. **主实验只到 1.2B**：7B 只有 HellaSwag 一个 benchmark 的 scaling 实验
2. **数据池只有 RefinedWeb**：未验证多语言、领域特定数据
3. **Multi-target 用简单等权混合**：论文自己承认这是 lower bound
4. **Baseline 较弱**：只比了 3 个 baseline，缺少主流 data mixing 方法
5. **无理论保证**：纯实证工作
6. **NAG 提取需要 off-the-shelf LLM**：192 GPU-hours (H100) 对于 150B token 池，如果池更大成本线性增长
7. **未考虑数据交互效应**：NAG 独立评分每个样本，不考虑样本间的组合效应（Group-MATES 已证明交互效应重要）

---

## 9. 与已有论文的关联

### 方法谱系定位

```
                    样本级选择                    领域级配比
                    ←————————————————————————————→
    Training-free:  NAG, FineWeb-Edu, QuRating
    Proxy-based:    BETR, SHUM, Thrush         QuaDMix, RegMix, DoReMi, OP-Mix
    Online:         ADAPT, PiKE
```

### 关键引用关系

- NAG 引用了 **QuaDMix** 和 **RegMix** 作为未来多目标改进方向
- NAG 引用了 **BETR** 作为主要 target-oriented baseline
- NAG **未引用** DoReMi、DoGE、Chameleon、Group-MATES
- **AttentionInfluence** (Hua et al., 2025) 被提及为同类工作

### 与 Group-MATES 的对比

| 维度 | NAG | Group-MATES |
|------|-----|-------------|
| 选择粒度 | 样本级（独立评分） | 组级（考虑交互） |
| 是否考虑数据交互 | ❌ | ✅ |
| 计算成本 | 192 GPU-hours | 2.7 GPU-hours |
| 最大验证规模 | 7B (1 benchmark) | 3B |

**潜在组合**：先用 NAG 独立评分，再用 Group-MATES 的 cosine similarity 方法考虑交互效应。

---

*分析完成。PDF 存储在 `papers/23_NAG_NeuronActivatedGraph_DataSelection.pdf`。*
