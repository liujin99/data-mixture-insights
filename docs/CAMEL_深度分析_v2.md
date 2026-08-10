# CAMEL 深度分析

**论文**: Capacity-Aware Mixture Law Enables Efficient LLM Data Optimization
**arXiv**: 2603.08022v2 (2026.05.06)
**作者**: Jingwei Li, Xinran Gu, Jingzhao Zhang (清华)
**Venue**: ICML 2026
**架构**: DeepSeek V3 MoE, 590M-A12M ~ 7B-A150M → 55B-A1.2B 外推

---

## 一、核心方法

### 1.1 问题定位：Mid-Training

论文**明确针对 mid-training 场景**（非 from-scratch pretraining）：

> "Mid-training is a crucial period when the model quickly builds essential skills such as abstract reasoning, math problem solving, coding, and planning. In cases like mid-training, data quality is prioritized over data quantity, as model ability can be easily weakened when mixed with low-quality, high-volume web text."

**与用户场景完全一致**：用户做中训练数据配比，3:7 通用:STEM。

### 1.2 Capacity-Aware Mixture Law (CAMEL)

**核心公式 (Theorem 2.2)**：

```
L_val(r, M) = C + Σ_i K_i / (<t_i, r>^α_i · M^β_i)
```

| 符号 | 含义 |
|------|------|
| r | 混合配比向量 |
| M | 模型大小（容量） |
| <t_i, r> | 内禀域 i 在配比 r 下的有效权重 |
| α_i | 配比对 loss 的指数影响 |
| β_i | 模型大小对 loss 的指数影响 |
| K_i | 域特定常数 |
| C | 高阶项（常数） |

**关键区别**：CAMEL **联合建模**配比和模型大小。之前的 DML (Ye et al. 2025) 和 SODM (Shukor et al. 2025) 将两者分开建模。

### 1.3 容量分配视角

**核心假设**：预训练是一个隐式的容量分配优化问题：

```
min_{m̃}  L_train(r; m̃) = Σ_i η_i(r) · ℓ_i(m̃_i)
s.t.     Σ_i m̃_i ≤ M
```

模型在总容量 M 约束下，自然地将容量分配到各内禀域。**不同域的容量分配不成比例**：

- R_math ≈ 2.4（从小→中 vs 小→大的 loss 改善比）
- R_know ≈ 4.3

**这直接解释了为什么小模型 proxy 配比不能迁移到大模型** — 不同域的容量分配随模型大小变化的速率不同！

### 1.4 Loss-to-Benchmark 预测定律

**核心公式**：

```
Acc_b(L) = C_b + A_b / (1 + exp(k_b^T · L + B_b))
```

其中 **L = (L_1, ..., L_N)** 是来自**多个验证集**的 loss 向量。

**关键区别**：
- 之前的方法：单个 val_loss → benchmark（不可靠，Lourie et al. 2025 证实）
- CAMEL：**多个 val_loss → benchmark**（多维度信号）

**理由**：benchmarks（如 MMLU）需要综合多种能力（知识+推理+语言理解），单一 val_loss 不足以预测。

### 1.5 端到端流水线

```
配比 r → L_b(r, M) [benchmark-proxy loss] → Acc_b(r, M) [benchmark accuracy]
```

1. 先用多个验证集 loss 拟合 loss-to-benchmark 映射
2. 定义 benchmark-proxy loss: L_b = k_b^T · L
3. 用 CAMEL 定律拟合 L_b(r, M)
4. 外推到目标模型大小 M_target，搜索最优 r*

### 1.6 Hourglass 采样策略

在固定计算预算下，**沙漏型采样**（两端多、中间少）最小化外推误差：

- 最小模型：多采样（成本低）
- 最大模型：适当采样（锚定外推）
- 中间模型：少采样（信息量低）

**对比 5 种策略**（Figure 7）：hourglass < diamond < triangle < inverted triangle < rectangle

---

## 二、实验结果

### 2.1 Benchmark 结果（Table 1, 55B-A1.2B 目标模型）

| 方法 | MMLU | ARC-C | BBH | GSM8K | MATH | HumanEval | C-Eval | 加权平均 |
|------|------|-------|-----|-------|------|-----------|-------|---------|
| Human | 0.573 | 0.535 | 0.429 | 0.652 | 0.325 | 0.354 | 0.637 | 0.496 |
| DoReMi | 0.569 | 0.529 | 0.437 | 0.674 | 0.336 | 0.450 | 0.627 | 0.514 |
| DML | 0.579 | 0.552 | 0.454 | 0.696 | 0.393 | 0.353 | 0.631 | 0.519 |
| SODM | 0.566 | 0.541 | 0.453 | 0.705 | 0.399 | 0.421 | 0.628 | 0.526 |
| **CAMEL** | 0.577 | **0.553** | **0.455** | 0.693 | 0.372 | **0.451** | **0.653** | **0.532** |

- CAMEL 加权平均最高（0.532）
- 比次优 SODM 高 +0.6%（绝对）
- 比人类设计高 +3.6%（绝对）
- 比模型大小无关方法（DoReMi）高 +1.8%

### 2.2 Held-out Benchmark 泛化（Figure 8）

7 个未见 benchmark 上 CAMEL 仍然最高 → 不是过拟合到优化目标。

### 2.3 成本

- 比基线减少 **50% 混合优化成本**
- 用不到目标模型一次完整训练的成本即可找到高质量混合
- +3% downstream benchmark 性能

---

## 三、与 QuaDMix 的关联分析

### 3.1 信号对比

| 维度 | QuaDMix | CAMEL |
|------|---------|-------|
| 信号 | 1M proxy val_loss（单点） | 多验证集 loss 向量 → benchmark |
| proxy→target 迁移 | 直接迁移（已知有问题） | 显式建模 M，CAMEL 定律外推 |
| benchmark 预测 | 无（用 val_loss 替代） | loss-to-benchmark logistic 映射 |
| 容量建模 | 无 | 容量分配优化问题 |
| 计算预算分配 | 均匀分配 336 实验 | hourglass 策略（两端多中间少） |
| mid-training 适用性 | 通用 | **明确针对 mid-training** |

### 3.2 CAMEL 如何解决 QuaDMix 的核心问题

#### 问题 1: val_loss 不预测推理性能
**CAMEL 的解法**：不是用单个 val_loss，而是用**多个验证集 loss 向量**通过 logistic 映射预测 benchmark。不同 benchmark 需要不同能力组合，多维信号可以捕捉。

**对 QuaDMix 的启示**：不要用单一 STEM val_loss 做信号，应该用**多个验证集**（通用、STEM、数学、代码等各自 val_loss）组合预测 benchmark。

#### 问题 2: proxy→target 迁移失败
**CAMEL 的解法**：显式建模模型大小 M。CAMEL 定律中 L_val 依赖于 M^β_i，不同域的 β_i 不同，因此小模型最优配比 ≠ 大模型最优配比。

**对 QuaDMix 的启示**：QuaDMix 的 1M proxy 实验结果需要通过 CAMEL 定律外推到目标模型大小，而不是直接迁移。

#### 问题 3: 搜索空间过大（198 维）
**CAMEL 的解法**：使用**内禀域**（intrinsic domains）概念。k 个内禀域（k << 198）可以通过数据组合权重 <t_i, r> 来表示。论文中只用 5 个域。

**对 QuaDMix 的启示**：198 域需要先投影到 5-10 个内禀域。这与之前的"域分组"建议一致，但 CAMEL 提供了理论基础——内禀域是验证 loss 的隐变量。

### 3.3 可操作建议

**立即可做**：
1. **收集多个验证集 loss**：QuaDMix 当前只有 STEM val set（6 tasks, 16455 docs）。应增加通用、数学、代码等验证集，构成多维 loss 向量。

2. **拟合 loss-to-benchmark 映射**：在已有 336 个 proxy 实验上，每个实验有 val_loss 和（如果能跑 benchmark 的话）benchmark accuracy。拟合 logistic 映射，验证多维 loss 是否能预测 benchmark。

3. **检查容量分配不成比例**：CAMEL 的 Figure 3 实验可以直接在 QuaDMix 的 proxy 实验上复现——比较不同模型大小（如果有的话）对各域 loss 的改善率 R_i。如果 R_i 不同，说明 proxy→target 迁移必然失败。

**短期可做**：
4. **用 CAMEL 定律重新拟合 proxy 数据**：将 336 个实验的 (r, M_proxy, L_val) 数据用 CAMEL 公式拟合，然后外推到 M_target，看预测的最优 r* 是否不同于 LightGBM 回归的结果。

5. **实现 hourglass 采样策略**：如果做新一轮 proxy 实验，用 hourglass 策略分配计算预算（小模型多采、大模型适当采、中间少采）。

### 3.4 关键限制

1. **假设 2.1 要求域间指数接近**（α_i ≈ ā + ε_i）：如果 198 域中某些域的 α_i 差异很大，CAMEL 定律可能不适用。
2. **需要多个验证集**：CAMEL 的 loss-to-benchmark 映射需要 >200 个验证集来拟合。QuaDMix 目前只有 1 个 STEM 验证集。
3. **5 个域 vs 198 域**：CAMEL 只用 5 个域，QuaDMix 有 198 个。需要先分组。
4. **MoE 架构**：CAMEL 用 DeepSeek V3 MoE，用户模型架构可能不同。
5. **没有开源代码**。

---

## 四、可信度评估

### 强信号
- **ICML 2026** 接收（venue 确认）
- 清华大学 Jingzhao Zhang 组（知名 ML 理论组）
- **55B-A1.2B 目标模型**验证——索引中最大规模
- 理论推导完整（Theorem 2.2 + Appendix D 证明）
- 明确针对 mid-training
- 7+7 benchmark 验证（含 held-out）
- 与 4 个基线全面对比
- 引用了 Gu et al. 2025 (NeurIPS Spotlight)、Lourie et al. 2025 等关键相关工作

### 弱信号
- 假设 2.1 可能不满足（域间指数差异大）
- 无代码开源
- 5 个域 vs 用户 198 域
- +0.6% 绝对提升（vs SODM）不算大
- 部分 benchmark CAMEL 不是最优（MMLU: 0.577 vs DML 0.579, MATH: 0.372 vs SODM 0.399）

### 判断
- **理论贡献高**：容量分配视角解释 proxy→target 失效机制
- **loss-to-benchmark 映射实用**：直接解决 val_loss→benchmark gap
- **55B 验证可信**：大规模验证非 toy experiment
- **mid-training 明确**：与用户场景完全匹配
- **整体可信度：高**

---

## 五、对破局路线的定位

CAMEL 对三条破局路线的影响：

### Route A（预训练小模型 + benchmark 评估）
- CAMEL 的 loss-to-benchmark 映射可以替代直接 benchmark 评估——用多个 val_loss 预测 benchmark
- 如果验证成功，Route A 的成本可以大幅降低（不需要在每个 mixture 上跑 benchmark，只需跑 val_loss）
- CAMEL 的 hourglass 采样策略可以指导 Route A 的计算预算分配

### Route B（Perplexity Correlations 式域级加权）
- CAMEL 的多维 loss 信号是 Route B 单维 loss-benchmark 相关性的自然推广
- CAMEL 提供了显式的函数形式（logistic），而 Route B 只是经验观察相关性
- 可以将 CAMEL 的 loss-to-benchmark 映射视为 Route B 的理论版

### Route C（Nexus 优化器）
- CAMEL 不直接影响 Route C
- 但 CAMEL 的容量分配视角可以用于评估 Nexus 是否真的平衡了各域容量

**结论**：CAMEL 同时增强了 Route A 和 Route B 的理论基础，是连接两者的桥梁。

---

## 六、总结

CAMEL 是目前索引中**最直接对应用户场景**的论文：

1. **明确 mid-training** — 与用户完全一致
2. **loss-to-benchmark 映射** — 直击 val_loss→benchmark gap 痛点
3. **容量分配解释 proxy 失效** — R_math ≠ R_know，不同域容量分配不成比例
4. **55B-A1.2B 验证** — 最大规模验证
5. **ICML 2026** — 高可信度

**最高 ROI 行动**：
1. 在 QuaDMix 现有 336 个 proxy 实验上拟合 CAMEL 定律，检查是否能预测最优配比
2. 收集多个验证集 loss（通用/STEM/数学/代码），拟合 loss-to-benchmark 映射
3. 检查不同模型大小对各域 loss 的改善率 R_i，验证容量分配不成比例
