# Scaling Domain Data Repetition in LLM Pretraining

**论文**: arXiv:2608.14071 (2026.8.14 提交)
**作者**: Jingwei Li, Xinran Gu, Rui Dai, Xintong Hao, Chengyin Xu, Yan Wu, Shuran Zheng, Jingzhao Zhang
**机构**: 清华大学 + 字节跳动 Seed
**代码**: 未提及
**状态**: 新论文，无 venue

---

## 一句话总结

**高质量域数据重复多少次最优？取决于该域的最终 loss，不取决于数据量。** 在 compute-optimal scaling（TPP 固定）下，最优重复次数与域的 val loss 强负相关——loss 低的域可以多重复，loss 高的域少重复。同 TPP 小 proxy 上调的重复次数可迁移到大模型。

---

## 1. 核心问题

### 1.1 高质量数据的稀缺困境

随着模型规模增大，训练 token 预算按 Chinchilla 比例（TPP ≈ 20）同步增长。但高质量域数据（如数学、代码、学术论文）的增长远慢于一般 web 数据。

结果：**高质量域在训练混合中的比例自然稀释**。如果不重复，大模型的训练数据中高质量占比反而比小模型低。

### 1.2 重复 vs 过拟合的权衡

重复高质量数据是应对稀释的自然方案，但过度重复导致过拟合：
- 重复太少 → 高质量信号不足
- 重复太多 → 模型记忆而非泛化

**核心问题**：每个域的最优重复次数是多少？它由什么决定？能否在小模型上调好后迁移到大模型？

### 1.3 与已有工作的区别

| 论文 | 关注点 | 本文差异 |
|------|--------|---------|
| #13 Repetition-Aware Scaling (Apple) | 高质量数据整体可重复 15-20x | **域级**精细分析，非整体 |
| Chinchilla (DeepMind) | TPP=20，不重复 | 重复场景下 TPP 的含义 |
| DataConstrained (Muennighoff) | 数据受限时 epoch+参数缩放 | 聚焦域级重复，非整体 epoch |

---

## 2. 实验设置

### 2.1 Compute-Optimal Scaling

关键前提：训练 token 预算 $T$ 与模型参数量 $M$ 成比例增长：

$$T = \text{TPP} \times M$$

其中 TPP（tokens-per-parameter）保持固定。这是 Chinchilla 最优缩放的设定。

在此设定下，当 $M$ 增大时 $T$ 同步增大，各域的可用数据是否够用取决于该域的数据量 $|D_k|$：
- 如果 $r_k \times T > |D_k|$，需要重复 $\lceil r_k T / |D_k| \rceil$ 次
- 如果 $r_k \times T \leq |D_k|$，无需重复

### 2.2 域级重复实验

对每个域 $k$，独立研究重复次数 $n_k$ 对模型性能的影响：
- 固定模型大小和总 token 数
- 变化域 $k$ 的重复次数 $n_k$（同时调整其他域填充剩余预算）
- 记录域 $k$ 的 val loss

---

## 3. 关键发现

### 3.1 固定 TPP 下，最优重复次数随模型规模温和增长

> "at a fixed TPP, the optimal repetition count mildly increases with model size"

**反直觉**：预期大模型更容易过拟合（参数多→记忆能力强），应该少重复。实际相反——大模型的最优重复次数略高。

**可能机制**：大模型有更多容量，能更好地从重复数据中提取泛化特征而非记忆。容量增长速度超过过拟合风险增长速度。

**含义**：在 compute-optimal 设定下，大模型对重复的容忍度更高。不需要因为模型变大就减少重复。

### 3.2 最优重复次数与域最终 val loss 强负相关

> "the optimal repetition count is strongly negatively correlated with the final validation loss of a domain: domains with lower loss can generally benefit from more repetitions"

**这是最重要的发现。** 

- **loss 低的域**（模型已经学得好）→ 可以多重复（继续巩固）
- **loss 高的域**（模型还在挣扎）→ 少重复（多重复 = 过拟合噪声）

**直观理解**：
- loss 低 = 模型已找到该域的泛化模式 → 重复强化模式，不过拟合
- loss 高 = 模型未充分理解该域 → 重复 = 反复看不懂的内容 → 记忆噪声而非学习

**操作含义**：不需要知道域的数据量、难度、知识复杂度——只需要测量域的 val loss，就能估计最优重复次数。

### 3.3 唯一数据量与最优重复次数仅弱相关

> "the amount of unique domain data is only weakly related to the optimal repetition count"

**反直觉**：预期"数据少的域需要多重复"。实际不成立——决定因素是 loss，不是数据量。

**解释**：数据量影响的是"是否需要重复"（不够就必须重复），但"重复多少次最优"取决于模型对数据的吸收效率（loss 反映）。

区分两个问题：
1. **是否需要重复**：由 $r_k T$ vs $|D_k|$ 决定（数据量问题）
2. **重复多少次最优**：由域 val loss 决定（学习效率问题）

### 3.4 同 TPP 小 proxy 上的重复次数可迁移到大模型

> "repetition counts tuned on smaller proxy models with the same TPP can provide a practical estimate for larger models"

**跨尺度可迁移量**：重复次数。与 LogFloor (#68) 的"瓶颈顺序可迁移"同类——不同的可迁移量，但都说明**不是所有东西都不可迁移**。

**关键条件**：同 TPP。proxy 和 target 的 TPP 必须一致。如果 proxy 用 TPP=20，target 用 TPP=1.2（用户的中训练场景），不一定迁移。

---

## 4. 与本项目的系统关联

### 4.1 对用户场景的直接含义

用户场景：1.5B 模型，1.8B tokens，TPP ≈ 1.2，3:7 (web:math)

**重复策略分析**：

| 域 | 比例 | token 数 | 可能数据量 | 需要重复？ | 重复策略 |
|----|------|---------|-----------|----------|---------|
| Math/STEM | 70% | 1.26B | 取决于数据源 | 如果高质量 math < 1.26B，需要 | 按 loss 决定次数 |
| Web/General (ClimbMix) | 30% | 0.54B | ClimbMix 400B（充足） | 不需要 | n=1 |

**Math 域的关键问题**：
- 用户用的 math 数据池有多大？如果 < 1.26B，必须重复
- 如果必须重复，重复多少？→ 测量 math 域的 val loss
- 如果 math loss 仍高（模型在挣扎）→ 少重复（避免过拟合噪声）
- 如果 math loss 已低（模型学得好）→ 多重复（巩固）

**中训练的特殊性**：基座已在 ClimbLab 1.2T 上预训练（TPP=800，充分）。中训练是 continual pre-training，基座对 math 已有一定基础。中训练开始时的 math val loss 可能中等（非零非极低），此时重复策略需要谨慎。

### 4.2 TPP 不匹配的迁移风险

论文验证的迁移条件是**同 TPP**。用户场景：
- 预训练：TPP = 800（ClimbLab 1.2T / 1.5B）
- 中训练：TPP ≈ 1.2（1.8B / 1.5B）

两者 TPP 差 600 倍。论文的迁移结论**不直接适用**。

但可以推断：
- TPP=1.2 是极度数据受限场景（远低于 Chinchilla 的 TPP=20）
- 数据极度受限时，重复几乎是必然的（1.8B tokens 不足以训练 1.5B 模型到收敛）
- 此时最优重复次数可能更高（数据少→必须充分利用）

### 4.3 与 Phase Transitions 的交互

Phase Transitions (#01) 的相变阈值 $r_{\text{thres}} \propto S/M^{1.283}$：

- 如果 $r_{\text{math}} > r_{\text{thres}}$（在相变以上），重复 math 数据在"线性增长区"——每次重复都有效
- 如果 $r_{\text{math}} < r_{\text{thres}}$（在相变以下），重复无用——模型还没跨过门槛，重复不跨

用户 $r_{\text{math}} = 0.7 \gg r_{\text{thres}} \approx 0.06\text{-}0.08$，在相变以上。此时重复应该有效（在论文的 loss-correlated 框架下）。

**但数据量是瓶颈**：1.26B math token 仅是 DeepSeekMath 参考线 150B 的 ~1%。即使重复到 5x（6.3B），仍远低于参考线。重复到 10x（12.6B）开始接近但 TPP 严重失衡。

### 4.4 与已有论文的关联

| 论文 | 关联 |
|------|------|
| **#13 Repetition-Aware Scaling (Apple)** | Apple 说整体 15-20x；本文域级精细化，且发现 loss 是决定因素而非数据量 |
| **Phase Transitions (#01)** | 相变以上重复有效，相变以下重复无用——本文未涉及相变，但 loss-correlated 发现可能与相变有关（loss 高=可能近相变阈值？） |
| **LogFloor (#68)** | 另一个跨尺度可迁移量（瓶颈顺序）；本文的重复次数是同类发现 |
| **Nexus (#51)** | 重复不改优化器，不影响 close/distant minimizer |
| **CAMEL (#09)** | CAMEL 的 $R_{\text{math}} \neq R_{\text{know}}$——重复数学知识可能降低 $R_{\text{know}}$ 的 loss 但不影响 $R_{\text{math}}$ 的相变 |
| **Mixture Experiment (#70)** | 域间交互可能影响重复策略：如果 web×math 有正交互，web 数据的重复也可能间接帮助 math |

### 4.5 可操作建议

1. **测量域 val loss**：在中训练开始前（或早期），用基座模型测量各域 val loss
2. **低 loss 域多重复，高 loss 域少重复**：按论文的 loss-correlated 规则
3. **proxy 实验**：用小模型（如 70M）+ 同 TPP=1.2，测试不同重复次数，找到最优后迁移到 1.5B
4. **注意 TPP 不匹配**：预训练 TPP=800 vs 中训练 TPP=1.2，迁移不直接。但 proxy 和 target 都用 TPP=1.2 可以
5. **与 Phase Transitions 结合**：确认 $r_{\text{math}} = 0.7 \gg r_{\text{thres}}$，重复在有效区

---

## 5. 局限性

1. **仅在 loss 空间**：所有分析基于 val loss，不涉及下游 accuracy。Layer 5（loss-accuracy 鸿沟）未触及
2. **Compute-optimal 前提**：TPP 固定的 Chinchilla 缩放设定。用户中训练 TPP=1.2 严重偏离 Chinchilla 最优 TPP=20，结论可能不直接适用
3. **域级独立分析**：每次只变一个域的重复次数。域间交互（如 web 重复影响 math 学习）未建模
4. **无相变建模**：loss-correlated 规则在相变附近的行为未分析。loss 高可能因为近相变阈值（数据量不够），也可能因为域本身难——两种情况最优重复策略不同
5. **无 peer review**：新论文，无 venue
6. **迁移条件严格**：同 TPP 要求。不同训练阶段（预训练 vs 中训练）的 TPP 差异巨大

---

## 6. 总结评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 理论深度 | ★★★☆☆ | 实验驱动，无新理论。但 loss-correlated 发现简洁实用 |
| 实验质量 | ★★★★☆ | 清华+字节 Seed 团队，域级精细分析 |
| 可信度 | ★★★★☆ | 8 作者，清华+字节 Seed（同 Nexus 团队），可信度高 |
| 与配比项目相关性 | ★★☆☆☆ | 对用户当前场景帮助有限：TPP≈1.2 vs 验证 TPP=20 差 17 倍；用户瓶颈是数据量（1.26B vs 参考 150B）非重复策略。战术级参考，非战略级帮助 |
| 可操作性 | ★★★☆☆ | 测域 loss → 定重复次数，规则简单。但迁移条件（同 TPP）不满足，需自行验证 |
| 风险 | 低 | 方法简单，最坏情况 = 回退到均匀重复 |

**核心价值**：将"该重复多少次"从直觉/经验决策转化为**loss-driven 的可测量规则**。"同 TPP 可迁移"是又一个跨尺度可迁移量的实证，丰富了我们对"什么能迁移、什么不能"的理解图谱。

**对用户场景的诚实评估**：论文团队可信、发现简洁，但**对当前问题帮助有限**。原因：① 用户中训练 TPP≈1.2 远低于 Chinchilla TPP=20，迁移条件不满足；② 用户瓶颈是数据量不足（1.26B 数学 token 仅参考线 1%），重复策略是锦上添花而非雪中送炭；③ 论文研究从头预训练，用户是 continual pre-training，重复语义不同。**等数据量问题解决后（如放大到 3B+ / 100B token 级），此论文的规则才有战术价值。**
