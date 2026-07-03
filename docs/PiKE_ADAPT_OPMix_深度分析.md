# PiKE / ADAPT / OP-Mix 深度分析

**分析日期**: 2026-07-03

---

## 总览

| # | 论文 | 机构 | 会议 | 核心方法 | QuaDMix 相关性 |
|---|------|------|------|----------|----------------|
| 21 | PiKE | Google Research + USC | arXiv'25 | 梯度交互自适应权重 | **高** |
| 22 | ADAPT | Cambridge + Oxford + Samsung | **ICLR'26** | 在线样本级 reweighting | **中** |
| 23 | OP-Mix | NYU + CMU + Microsoft | arXiv'26 | LoRA 插值模拟配比 | **高** |

**注 1**: 用户指定的第 4 篇 "Group-Level Data Selection" 即 Group-MATES (paper 19)，已在 `GroupMATES_深度分析.md` 中分析。

**注 2**: OP-Mix (paper 23) 与 paper 15 为同一篇论文（已确认文件完全一致），以下分析基于全文精读，比 paper 15 的初始分析更为深入。

---

# 一、PiKE: Adaptive Data Mixing for Large-Scale Multi-Task Learning Under Low Gradient Conflicts

## 1.1 论文概览

| 维度 | 信息 |
|------|------|
| arXiv | 2502.06244 (2025.02, v2: 2025.06) |
| 机构 | **Google Research + USC** |
| 作者 | Zeman Li, Yuan Deng, Peilin Zhong, Meisam Razaviyayn, Vahab Mirrokni |
| 方向 | 基于梯度交互的自适应数据配比 |
| 与 QuaDMix 相关性 | **高** |

## 1.2 核心发现

### 发现 1: LLM 预训练中梯度冲突极少

这是论文最重要的实证发现。在大规模语言模型预训练中：

- **多语言场景**（en/hi/es 等）：任务梯度大多**正相关或近乎正交**
- **多领域场景**（GLaM 6 领域）：梯度余弦相似度普遍 ≥ 0
- 这与 CV 领域的 MTL 问题**截然不同**

**含义**：PCGrad、AdaTask、NashMTL 等解决梯度冲突的方法在 LLM 预训练中**基本无用**。

### 发现 2: Mix 策略优于 Random 和 Round-Robin

每个 batch 包含所有领域的数据（Mix）> 每个 batch 只含一个领域（Random）> 轮转领域（Round-Robin）。

### 发现 3: 静态权重始终劣于动态权重

即使找到最优静态权重，也不如动态调整。最优权重取决于：
1. **每个任务的梯度范数平方** ‖∇L_k‖²
2. **每个任务的梯度方差** σ²_k

## 1.3 方法

### 核心公式

PiKE 的最优权重（简化版）：

```
w_k* ∝ ‖∇L_k(θ)‖² / (‖∇L_k(θ)‖² + σ_k² / b_k)
```

直觉：
- **梯度大**的任务 → 还没学好 → 应该多训 → 权重高
- **方差大**的任务 → 噪声大 → 应该少训 → 权重低
- 两者平衡

### 计算效率

- **不需要**存储每个任务的完整梯度（O(Kd) 内存）
- 只需估计每个任务的**梯度范数**和**方差**（标量）
- 额外开销：**可忽略**（线性于模型大小和任务数）

### Balanced-PiKE

引入 Tilted ERM 防止某些任务被过度忽略：

```
min_θ (1/α) · log(Σ_k exp(α · L_k(θ)))
```

当 α → 0 时退化为普通 ERM；α > 0 时更关注表现差的任务。

## 1.4 实验结果

### 多语言预训练（C4 en/hi, 1B 模型）

| 方法 | 平均下游准确率 | 收敛速度 |
|------|---------------|----------|
| ERM (Uniform) | baseline | 1x |
| DoReMi | +2.1% | 1.2x |
| PiKE | **+7.1%** | **1.9x** |

### GLaM 多领域预训练（750M 模型）

| 方法 | 准确率提升 |
|------|-----------|
| vs DoReMi | **+3.4%** |
| vs Uniform | **+5.2%** |

### 规模扩展

在 110M → 1B 参数规模上**一致优于**其他方法。

## 1.5 对 QuaDMix 的启示

### 直接可用

1. **梯度范数 + 方差作为权重信号**：QuaDMix 的 proxy 可以直接输出每个领域的梯度范数和方差，作为配比依据
2. **Mix 策略确认**：每个 batch 应包含所有领域的数据（QuaDMix 已采用）
3. **动态权重必要性**：静态权重始终劣于动态调整，支持 QuaDMix 的迭代优化思路

### 需要注意

1. **低梯度冲突假设**：如果 QuaDMix 的领域之间确实存在梯度冲突（如代码 vs 文学），PiKE 的简化版可能不适用
2. **在线方法**：PiKE 需要在训练过程中实时计算梯度统计量，与 QuaDMix 的 proxy-then-train 范式有差异
3. **无 proxy 信号问题**：PiKE 直接在目标模型上计算梯度，不涉及 proxy → target 的迁移问题

### 可信度评估

| 维度 | 评分 | 理由 |
|------|------|------|
| 机构 | ★★★★★ | Google Research + USC |
| 理论 | ★★★★☆ | 有收敛保证，tight bound |
| 实验 | ★★★★☆ | 110M-1B，多场景，但最大只到 1B |
| 实用性 | ★★★★☆ | 开销极低，但需要在线计算梯度 |

---

# 二、ADAPT: Rethinking Data Curation in LLM Training — Online Reweighting Offers Better Generalization than Offline Methods

## 2.1 论文概览

| 维度 | 信息 |
|------|------|
| arXiv | 2605.05227 (2026.04) |
| 机构 | **Cambridge + Oxford + Samsung AI Center** |
| 会议 | **ICLR 2026** |
| 作者 | Wanru Zhao, Yihong Chen, Yuzhi Tang 等 |
| 方向 | 在线样本级 reweighting（统一数据策展框架） |
| 与 QuaDMix 相关性 | **中** |

## 2.2 核心贡献

### 贡献 1: 统一数据策展框架

将所有数据策展方法统一为 **reweighting** 的特例：

| 方法 | 权重形式 | 操作时机 |
|------|---------|----------|
| Data Selection | w(x) ∈ {0, 1}（二值） | 离线 |
| Data Mixing | w_d（领域级） | 离线 |
| Data Reweighting | w(x) ∈ [0, 1]（连续） | 在线 |

### 贡献 2: 离线方法的根本缺陷

**核心观察**：离线数据选择在 in-domain 评估上看起来好，但 **out-of-domain 泛化严重不足**。

- LESS 在 MMLU→MMLU 上表现好，但 MMLU→BBH 上**严重退化**
- 原因：离线硬过滤**牺牲数据多样性**，用记忆替代泛化
- 全量数据训练在两个 benchmark 上都**稳定**

### 贡献 3: ADAPT 方法

**核心思想**：不改变训练数据量，而是动态调整每个样本的**学习率**。

```
θ_{t+1} = θ_t - η · w_t(i) · ∇_θ ℓ(θ_t; x_i)
```

权重计算：
1. **ADAPT-BM25**：基于 BM25 文本相似度（模型无关）
2. **ADAPT**：基于当前模型的 hidden state 余弦相似度（模型感知）

关键设计：
- **Sigmoid 全局权重**：权重不依赖 batch 内其他样本（vs softmax 归一化）
- **在线 embedding 更新**：每 R 步刷新验证集 embedding
- **Position-weighted pooling**：后部 token 权重更高（抵消 causal mask 偏差）

## 2.3 实验结果

### Instruction Tuning（Llama-2-7B + LoRA）

| 方法 | MMLU(val)→MMLU | MMLU(val)→BBH | 泛化差距 |
|------|----------------|---------------|----------|
| LESS | 50.2 | **38.7** | -11.5 |
| Full SFT | 49.7 | 44.4 | -5.3 |
| ADAPT-BM25 | 50.9 | 43.7 | -7.2 |
| **ADAPT** | **50.7** | **44.8** | **-5.9** |

关键：LESS 的 in-domain 分数高但 OOD 严重退化；ADAPT 的泛化差距最小。

### Pretraining（TinyLlama-120M, 50B tokens）

| 方法 | Avg (All) | Avg (Unseen) | FLOPs 开销 |
|------|-----------|-------------|-----------|
| Uniform | 37.81 | 31.98 | 0 |
| DoReMi | 37.32 | 31.95 | 4.9×10¹⁹ |
| RegMix | 37.97 | 32.46 | 3.1×10¹⁸ |
| ADAPT-BM25 | 38.05 | 33.49 | ≪10¹⁴ |
| **ADAPT** | **38.00** | **33.73** | ≪10¹⁵ |

### 100B tokens 扩展

ADAPT 在 15 个 benchmark 中 11 个优于 Uniform，平均提升 0.38%。

## 2.4 对 QuaDMix 的启示

### 重要警示

1. **离线方法的泛化风险**：QuaDMix 作为离线配比方法，需要特别关注配比的**泛化性**，不能只在 in-domain 上评估
2. **数据多样性 > 数据质量**：硬过滤/过度选择高质量数据可能**损害泛化**
3. **FLOPs 公平比较**：评估时应计入 proxy 训练的计算成本

### 可借鉴

1. **样本级 reweighting 思想**：QuaDMix 可以考虑在领域配比基础上加入样本级权重
2. **Sigmoid 全局权重**：比 softmax 归一化更稳定，不受 batch 组成影响
3. **在线 embedding 更新**：如果 QuaDMix 迭代运行，可以用当前模型的 embedding 而非 proxy 的

### 局限

1. **实验规模小**：预训练只测了 120M，instruction tuning 用 LoRA on 7B
2. **需要验证集**：ADAPT 需要一个代表目标分布的验证集，这在通用预训练中可能不可用
3. **与 QuaDMix 解决不同层次的问题**：ADAPT 是样本级 reweighting，QuaDMix 是领域级配比

### 可信度评估

| 维度 | 评分 | 理由 |
|------|------|------|
| 机构 | ★★★★☆ | Cambridge + Oxford，ICLR'26 接收 |
| 理论 | ★★★☆☆ | 主要是框架统一 + 实证，理论贡献有限 |
| 实验 | ★★★☆☆ | 预训练只到 120M，instruction tuning 用 LoRA |
| 实用性 | ★★★★☆ | 几乎零额外开销，实现简单 |

---

# 三、OP-Mix: Always Learning, Always Mixing — Efficient and Simple Data Mixing All The Time

## 3.1 论文概览

| 维度 | 信息 |
|------|------|
| arXiv | 2605.15220 (2026.05) |
| 机构 | **NYU + CMU + Microsoft** |
| 作者 | Michael Y. Hu, Apurva Gandhi, Kyunghyun Cho, Tal Linzen, Pratyusha Sharma |
| 方向 | 跨训练生命周期的统一数据配比 |
| 与 QuaDMix 相关性 | **高** |

## 3.2 核心问题

现有数据配比方法的两个根本限制：

1. **不能扩展到新领域**：DoReMi、RegMix、DoGE 等都假设领域集合固定
2. **依赖 off-policy proxy**：小模型的配比信号不一定迁移到大模型

OP-Mix 的解决方案：**on-policy proxy + LoRA 插值**

## 3.3 方法

### 核心流程

```
1. 对每个领域 D_k，从当前模型出发训练一个 LoRA adapter
2. 线性插值不同 LoRA 的权重 → 模拟不同配比
3. 在插值空间中评估 → 拟合 log-linear 回归模型
4. 优化得到最优配比 α*
5. 用 α* 在基础模型上做最终训练
```

### 关键创新

**1. On-Policy Proxy**

不是训一个独立的小模型，而是从**当前模型**出发训 LoRA：
- 保证 proxy 与目标模型的**学习动态一致**
- 不需要单独初始化小模型
- 在 continual learning 中特别重要（没有匹配的小模型可用）

**2. LoRA 插值模拟配比**

```
θ^LoRA_α = α_old · θ^LoRA_old + Σ_k α_k · θ^LoRA_{D_k}
```

- 训 K 个 LoRA → 模拟 O(∞) 种配比
- 只需 forward pass，不需要额外训练
- 基于 **Linear Mode Connectivity** 的理论支持

**3. Mixture Expansion（持续学习）**

新领域到来时：
- 不重新为每个旧领域训 LoRA
- 只训一个 "old" adapter + K 个新领域 adapter
- 保持旧领域的相对比例不变

### 理论保证

OP-Mix 的性能差距上界：

```
J(α^) - J(α*) ≤ 2(ε_merge + ε_LoRA)
```

其中：
- **ε_merge**：线性插值 vs 真实混合的误差（由 Linear Mode Connectivity 保证小）
- **ε_LoRA**：LoRA vs 全参数微调的误差

## 3.4 实验结果

### 预训练（OLMo 150M/300M/530M）

| 方法 | PPL 改进 | 计算效率 |
|------|---------|----------|
| ERM (no mixing) | baseline | 1x |
| OLMix (off-policy proxy) | +1.0% | 0.9x |
| MergeMix (on-policy, full FT) | +6.3% | 0.86x |
| **OP-Mix** | **+6.3%** | **1.0x** (比 MergeMix 省 14%) |

关键：OP-Mix 和 MergeMix 都是 on-policy，远优于 off-policy 的 OLMix。

### 持续中间训练（Continual Midtraining）

| 方法 | 遗忘缓解 | 计算量 |
|------|---------|--------|
| Continual SFT | 0% (严重遗忘) | 1x |
| WSD-S | baseline | 1x |
| 10% Data Replay | +15% | 1x |
| **OP-Mix** | **+27% avg, +71% Reddit** | **0.34x** (省 66%) |
| Retraining (skyline) | 100% | 3-5x |

### 持续指令微调（Qwen2.5-7B-Instruct）

| 方法 | 准确率 | 计算量 |
|------|--------|--------|
| Continual SFT | 55.3% | 1x |
| SDFT | 60.2% | 20x |
| OP-Mix + SFT | **60.0%** | **1x** (省 95%) |
| **OP-Mix + SDFT** | **61.9%** | 20x |

关键发现：
- OP-Mix + SFT 匹配 SDFT 的性能，但**省 95% 计算**
- OP-Mix + SDFT 进一步提升，说明**数据配比和训练目标是正交的改进轴**

### 损失面拟合质量

OP-Mix 估计的损失面与真实损失面**高度吻合**（Figure 7），平均 loss 增加仅 0.9%（vs 10% replay 的 2.9%）。

## 3.5 对 QuaDMix 的启示

### 直接竞品分析

OP-Mix 与 QuaDMix 解决**同一层次的问题**（领域级数据配比），但方法路线不同：

| 维度 | QuaDMix | OP-Mix |
|------|---------|--------|
| Proxy 类型 | 独立小模型 (1M) | LoRA on 当前模型 |
| 配比搜索 | 训练多个 proxy | 训 K 个 LoRA + 插值 |
| On/Off-Policy | Off-policy | **On-policy** |
| 持续学习 | 不支持 | **原生支持** |
| 最大验证规模 | ? | 530M (预训练), 7B (指令微调) |

### 关键启示

1. **On-policy 远优于 off-policy**：OP-Mix 和 MergeMix 都 on-policy，一致远优于 off-policy 的 OLMix。这直接质疑 QuaDMix 用独立 1M proxy 的做法。

2. **LoRA 作为 proxy 的可行性**：
   - 训 K 个 LoRA 的成本 << 训 K 个独立 proxy 模型
   - LoRA 插值可以**后验**模拟任意配比，不需要预先决定搜索空间
   - 但 QuaDMix 的 1M proxy 可能比 LoRA 更便宜（取决于 LoRA rank 和基座模型大小）

3. **持续学习能力**：新领域到来时只需训 1 个新 LoRA + 重新插值，不需要重跑整个 pipeline

4. **20% ERM warmup**：OP-Mix 在预训练中先用均匀权重训 20% tokens，再调整配比。这个 warmup 策略 QuaDMix 可以参考。

### 局限

1. **预训练只验证到 530M**：LoRA 插值在更大规模上是否仍然有效未知
2. **领域数量未扩展**：只测了 5 个领域，10+ 领域时 LoRA 插值的误差可能增大
3. **LoRA rank 选择**：论文未深入讨论 rank 对配比质量的影响

### 可信度评估

| 维度 | 评分 | 理由 |
|------|------|------|
| 机构 | ★★★★★ | NYU (Kyunghyun Cho, Tal Linzen) + CMU + Microsoft |
| 理论 | ★★★★☆ | 有性能差距上界，基于 Linear Mode Connectivity |
| 实验 | ★★★★☆ | 3 个场景 × 3 个规模，最大 7B (指令微调) |
| 创新性 | ★★★★★ | 首个跨生命周期的统一数据配比算法 |
| 实用性 | ★★★★★ | 实现简单，开销低，自然支持新领域 |

---

# 四、三篇论文横向对比

## 4.1 方法路线对比

| 维度 | PiKE | ADAPT | OP-Mix |
|------|------|-------|--------|
| **操作粒度** | 领域级 | 样本级 | 领域级 |
| **在线/离线** | 在线（训练中调整） | 在线（训练中调整） | 离线（训前决定） |
| **信号来源** | 梯度范数 + 方差 | 样本-验证集相似度 | LoRA 插值 loss |
| **需要 proxy** | 不需要 | 不需要 | 需要（LoRA） |
| **理论保证** | 收敛保证 | 无 | 性能差距上界 |
| **持续学习** | 不原生支持 | 不原生支持 | **原生支持** |
| **最大验证** | 1B | 120M (PT) / 7B (IT) | 530M (PT) / 7B (IT) |

## 4.2 对 QuaDMix 的综合启示

### 高优先级行动

1. **考虑 On-Policy Proxy**（来自 OP-Mix）
   - QuaDMix 的 1M off-policy proxy 是最大风险点
   - 替代方案：从当前模型出发训 LoRA 作为 proxy
   - 或至少降低 proxy LR（来自 ICLR'26 Proxy Reliability 论文）

2. **梯度范数 + 方差作为配比信号**（来自 PiKE）
   - 比纯 loss 信号更稳定
   - 实现简单：在 proxy 训练过程中记录每个领域的梯度统计量

3. **评估泛化性而非 in-domain 性能**（来自 ADAPT）
   - QuaDMix 的配比评估必须包含 OOD benchmark
   - 避免过度优化特定领域而损害整体泛化

### 中优先级行动

4. **Mix 策略确认**（来自 PiKE）：每个 batch 包含所有领域
5. **Sigmoid 全局权重**（来自 ADAPT）：如果用样本级 reweighting，避免 softmax 归一化
6. **20% ERM warmup**（来自 OP-Mix）：配比优化前先训 20% tokens

### 长期方向

7. **持续学习能力**（来自 OP-Mix）：支持新领域到来时增量更新配比
8. **在线 + 离线结合**：PiKE/ADAPT 的在线调整 + OP-Mix 的离线搜索可能是最优组合

---

# 五、可信度与优先级排序

## 5.1 综合评分

| 论文 | 机构 | 会议 | 理论 | 实验规模 | QuaDMix 相关性 | 总分 |
|------|------|------|------|---------|---------------|------|
| **OP-Mix** | ★★★★★ | arXiv | ★★★★ | ★★★★ | ★★★★★ | **4.6** |
| **PiKE** | ★★★★★ | arXiv | ★★★★ | ★★★★ | ★★★★☆ | **4.4** |
| **ADAPT** | ★★★★ | ICLR'26 | ★★★ | ★★★ | ★★★☆☆ | **3.4** |

## 5.2 推荐阅读顺序

1. **OP-Mix**（最高优先级）：直接竞品，on-policy + LoRA 插值的思路极具启发性
2. **PiKE**（高优先级）：梯度交互分析提供了新的配比信号维度
3. **ADAPT**（中优先级）：统一框架和泛化性分析有参考价值，但实验规模偏小

---

# 六、与已有论文的关联

## 6.1 方法谱系

```
                    离线 ←————————————————→ 在线
                     |                        |
    领域级    RegMix → OP-Mix → QuaDMix    PiKE → DoReMi/DoGE
                     |                        |
    样本级         LESS                     ADAPT → LinUpper
```

## 6.2 Proxy 可靠性谱系

```
    Off-policy (风险高)                    On-policy (风险低)
    ←————————————————————————————————————————→
    QuaDMix(1M)    Phase Transitions    CAMEL    OP-Mix(LoRA)    PiKE(无proxy)
```

## 6.3 关键引用关系

- OP-Mix 引用了 **RegMix** (log-linear 回归)、**MergeMix** (LoRA 合并)、**OLMix** (持续配比)
- PiKE 引用了 **DoReMi** (DRO-based mixing)、**FAMO** (scalable MTL)
- ADAPT 引用了 **LESS** (gradient influence)、**DoReMi**、**RegMix**
- 三篇都引用了 **DoReMi**，说明它仍是 data mixing 的核心 baseline

---

*分析完成。所有 3 篇论文的 PDF 已存储在 `papers/` 目录。*
