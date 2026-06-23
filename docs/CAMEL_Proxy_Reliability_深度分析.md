# CAMEL & Proxy Reliability 深度分析

**分析日期**: 2026-06-23  
**论文**: 
1. CAMEL: Capacity-Aware Mixture Law (arXiv:2603.08022, 清华)
2. Proxy Reliability: Can Small Training Runs Reliably Guide Data Curation? (arXiv:2512.24503, ICLR'26)

---

## 执行摘要

这两篇论文从不同角度解决了 QuaDMix 面临的核心问题：**如何让小模型 proxy 的信号可靠地指导大模型训练**。

- **CAMEL** 提供了理论框架：capacity-aware mixture law + loss-to-benchmark prediction
- **Proxy Reliability** 提供了实践修复：降低 proxy learning rate 即可大幅提升信号可靠性

**核心结论**：这两篇论文应该立即应用到 QuaDMix 中，可以显著改善 1M proxy 的可靠性问题。

---

## 论文 1: CAMEL (Capacity-Aware Mixture Law)

### 基本信息

- **标题**: Capacity-Aware Mixture Law Enables Efficient LLM Data Optimization
- **作者**: Jingwei Li, Xinran Gu, Jingzhao Zhang (清华大学)
- **发表时间**: 2026-03-09 (v2: 2026-05-06)
- **arXiv**: 2603.08022
- **关键词**: data mixture, scaling laws, capacity allocation, MoE models

### 核心问题

现有的 mixture scaling laws 存在两个关键问题：

1. **模型大小和配比的交互效应被忽略**：之前的方法（如 DML, SODM）将模型大小和配比效应分离建模，但实际上它们存在非线性交互
2. **只优化 validation loss，不直接优化下游性能**：validation loss 和 benchmark accuracy 并不总是对齐

### 核心贡献

#### 1. Capacity-Aware Mixture Law

**关键洞察**：将数据配比优化视为**容量分配问题**（capacity allocation problem）

**数学形式**：

```
L_val(r, M) = C + Σ_{i=1}^k K_i / (⟨t_i, r⟩^α_i * M^β_i)
```

其中：
- `r` = 数据配比向量
- `M` = 模型大小（参数量）
- `t_i` = 第 i 个 intrinsic domain 的 profile vector
- `⟨t_i, r⟩` = 配比 r 诱导的第 i 个 domain 的有效权重
- `K_i, α_i, β_i` = 可学习参数

**与之前方法的区别**：

| 方法 | 公式形式 | 问题 |
|------|---------|------|
| DML (Ye et al.) | L = Σ S_i [C_i + K_i exp(Σ t_ij r_j)] | 不包含模型大小维度 |
| SODM (Shukor et al.) | L = f(r) * g(M) | 分离建模，忽略交互 |
| **CAMEL** | L = C + Σ K_i / (⟨t_i, r⟩^α_i * M^β_i) | **联合建模 r 和 M** |

**理论依据**：

论文证明了一个关键观察：**有效参数分配不随模型大小线性增长**

实验观察（Figure 3）：
- 在 Math + Knowledge 混合训练上
- R_math ≈ 2.4, R_know ≈ 4.3（不同 domain 的 loss 改进比例不同）
- 这说明模型在不同 domain 上的容量分配是**动态调整**的，而非线性增长

#### 2. Loss-to-Benchmark Prediction Law

**问题**：validation loss 不一定与下游 benchmark accuracy 对齐

**解决方案**：引入 logistic 形式的预测 law

```
Acc_b(L) = C_b + A_b / (1 + exp(k_b^T L + B_b))
```

其中：
- `L = (L_1, ..., L_N)^T` = 多个 validation set 上的 loss
- `Acc_b` = benchmark b 的准确率
- `A_b, B_b, C_b, k_b` = 可学习参数

**关键创新**：使用**多个 validation loss** 而非单个 loss 来预测 benchmark

**理由**：benchmark（如 MMLU）通常需要多种能力（知识保留 + 推理），单个 loss 信号不够

**实验结果**（Figure 4）：
- 在 14 个 benchmark 上拟合，prediction error 很低
- 证明 validation losses 提供了足够的信号来预测下游性能

#### 3. Hourglass Sampling Strategy

**问题**：在有限 compute budget 下，如何在不同模型大小上分配采样点？

**五种策略对比**（Figure 6）：
1. **Rectangle**: 均匀分配（baseline）
2. **Triangle**: 更多小模型，更少大模型
3. **Inverted Triangle**: 更多大模型，更少小模型
4. **Diamond**: 从中间向两端扩展
5. **Hourglass**: 优先采样最小和最大模型，减少中间尺度

**实验结果**（Figure 7）：
- Hourglass 策略在所有 compute budget 下都达到最低 prediction error
- 直觉：最小模型提供趋势信息，最大模型提供外推锚点，中间尺度信息冗余

### 实验验证

#### Setup

- **模型架构**: Deepseek V3 (MoE)
- **拟合 law 的模型**: 590M-A12M → 7B-A150M (8 个尺度)
- **目标模型**: 55B-A1.2B
- **数据域**: English, Chinese, Code, Math, Knowledge (5 个域)
- **Benchmarks**: MMLU, ARC-C, BBH, GSM8K, MATH, HumanEval, C-Eval (7 个优化目标) + 7 个 held-out

#### 主要结果（Table 1）

| 方法 | MMLU | ARC-C | BBH | GSM8K | MATH | HumanEval | C-Eval | Weighted Avg |
|------|------|-------|-----|-------|------|-----------|--------|--------------|
| Human Designed | 0.573 | 0.535 | 0.429 | 0.652 | 0.325 | 0.354 | 0.637 | 0.496 |
| DoReMi (model-size agnostic) | 0.569 | 0.529 | 0.437 | 0.674 | 0.336 | 0.450 | 0.627 | 0.514 |
| DML | 0.579 | 0.552 | 0.454 | 0.696 | 0.393 | 0.353 | 0.631 | 0.519 |
| SODM | 0.566 | 0.541 | 0.453 | 0.705 | 0.399 | 0.421 | 0.628 | 0.526 |
| **CAMEL** | **0.577** | **0.553** | **0.455** | 0.693 | 0.372 | **0.451** | **0.653** | **0.532** |

**关键发现**：
- CAMEL 在 weighted average score 上达到最高（0.532 vs 0.526）
- 在大多数 benchmark 上表现最好或接近最好
- **成本降低 50%**：使用不到一次完整目标模型训练的成本

#### 泛化性（Figure 8）

- 在 7 个 held-out benchmarks 上（未用于优化），CAMEL 仍达到最高平均准确率
- 说明优化的 mixture 具有良好的泛化能力

### 对 QuaDMix 的启示

#### 直接应用

1. **在 LightGBM 回归中加入 capacity-aware 特征**
   - 当前 QuaDMix Eq.1: `w_i = f(quality_i, diversity_i, ...)`
   - 改进: `w_i = f(quality_i, diversity_i, model_size, interaction_terms)`
   - 具体: 加入 `quality_i * model_size`, `diversity_i * model_size` 等交互项

2. **实现 loss-to-benchmark prediction**
   - 当前 QuaDMix 只优化 validation loss
   - 改进: 训练一个 logistic 模型，从多个 validation loss 预测目标 benchmark
   - 需要: 收集多个 validation set 的 loss（不仅仅是 Pile）

3. **采用 Hourglass sampling**
   - 当前 QuaDMix 在 1M proxy 上训练
   - 改进: 如果有多个 proxy 尺度可用，优先使用最小和最大的

#### 间接启示

4. **验证 capacity allocation 假设**
   - 实验: 在不同大小的 proxy 上训练相同 mixture，观察各 domain 的 loss 改进比例
   - 如果比例差异大（如 CAMEL Figure 3），说明需要 capacity-aware 建模

5. **考虑 MoE 架构**
   - CAMEL 在 MoE 模型上验证（Deepseek V3）
   - 如果 QuaDMix 的目标模型是 MoE，需要特别考虑 expert 的容量分配

---

## 论文 2: Proxy Reliability (ICLR'26)

### 基本信息

- **标题**: Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice
- **作者**: Jiachen T. Wang, Tong Wu (Princeton), Kaifeng Lyu (Tsinghua/Berkeley), James Zou (Stanford), Dawn Song (Berkeley), Ruoxi Jia (Virginia Tech), Prateek Mittal (Princeton)
- **发表时间**: 2025-12-30 (v2: 2026-04-12)
- **arXiv**: 2512.24503
- **Venue**: **ICLR 2026**
- **关键词**: proxy models, data curation, learning rate, hyperparameter sensitivity

### 核心问题

**Proxy 模型的脆弱性**：使用相同超参数训练不同 data recipe 的 proxy 模型，可能导致**结论翻转**

这是一个**隐蔽但关键**的问题：
- 表面上看，使用相同超参数是"公平比较"
- 实际上，每个 data recipe 的最优超参数是不同的
- 固定超参数的结论可能 overfit 到特定设置

### 核心发现

#### 1. Learning Rate Sensitivity（Figure 3）

**实验**：在 GPT2-Small 上用两个相近的 learning rate 训练 DCLM 和 DCLM-dedup-GS

**结果**：
- LR = 2.5e-4: DCLM 优于 DCLM-dedup-GS
- LR = 3.5e-4: DCLM-dedup-GS 优于 DCLM

**结论**：即使是相近的 learning rate，也能翻转 data recipe 的排名

**原因分析**（一阶 vs 二阶效应）：

```
ΔL_val(θ) ≈ -η ∇L_val(θ) · ∇L(θ) + (η²/2) ∇L(θ)^T H ∇L(θ)
```

- **小 η**: 一阶梯度对齐项主导 → 排名由 train-val 分布相似度决定
- **中等 η**: 二阶曲率项变得重要 → 可能翻转排名

#### 2. Tiny Learning Rate 修复（Section 5）

**核心洞察**：使用**极小的 learning rate**（10^-5 ~ 10^-6）训练 proxy 模型

**两个关键发现**：

**Finding I**: 同一模型，tiny LR 性能与最优调参性能强相关（Figure 4a）
- GPT2-Small 在 tiny LR (1e-6) 下的 loss 与 full hyperparameter sweep 的最优 loss 强相关
- Spearman correlation > 0.95

**Finding II**: 跨尺度，tiny LR 保持排名一致（Figure 4b）
- 当 proxy 和 target 都用 tiny LR 时，dataset 排名几乎完美相关
- Spearman correlation > 0.95

**理论证明**（Theorem 1, Informal）：

对于 random feature models，如果模型宽度足够大，使用足够小的 learning rate 训练后，dataset 的相对顺序与 infinite-width limit 一致。

**直觉**：
- Tiny LR → 一阶梯度对齐主导 → 排名由 train-val 分布相似度决定
- 分布相似度是 dataset 的内在属性，不依赖于模型大小或超参数

#### 3. 实践建议

**如何选择 "tiny" learning rate**：

**理论指导**：
- η_tiny ≪ 1/λ_max，其中 λ_max 是 validation loss Hessian 的最大特征值
- 可以通过短 warmup 训练后的 checkpoint 估计

**经验法则**（更实用）：
- 使用比标准 LR **小 1-2 个数量级**的 learning rate
- 对于 LLM pretraining，标准 LR 通常是 1e-4 ~ 3e-4
- 因此 tiny LR ≈ **1e-5 ~ 1e-6**

**实验验证**（Section 6）：
- 23 个 data recipes
- 3 个模型家族（GPT2, Pythia, OPT）
- 70M ~ 1B 参数
- 4 个数据维度（domain composition, quality filtering, deduplication, major corpora）

**结果**：
- Tiny LR (1e-5) 在所有设置下都保持高 transferability
- Spearman rank correlation > 0.9 between proxy (125M) and target (1B)

### 对 QuaDMix 的启示

#### 立即行动：修复 Proxy LR

**当前问题**：
- QuaDMix 使用 1M proxy 模型
- 可能使用了标准的 learning rate（如 1e-4）
- 这可能导致 proxy 信号不可靠

**修复方案**：

```python
# 当前（可能的问题）
proxy_lr = 1e-4  # 标准 LR

# 修复后
proxy_lr = 1e-5  # 或 1e-6，比标准小 1-2 个数量级
```

**验证步骤**：
1. 在几个代表性 mixture 上，用不同 LR (1e-4, 1e-5, 1e-6) 训练 proxy
2. 比较 proxy 的 ranking 与 7B 模型的 ranking
3. 计算 Spearman rank correlation
4. 选择 correlation 最高的 LR

#### 短期行动：验证 Tiny LR 假设

**实验设计**：

1. **选择 5-10 个多样化的 mixture**
   - 覆盖不同的 quality/diversity 组合
   - 包括一些"反直觉"的 mixture（如高质量但低多样性）

2. **在 proxy 上用多个 LR 训练**
   - LR = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
   - 记录每个 mixture 的 validation loss

3. **在 7B 模型上验证**
   - 用标准 LR 训练（需要 hyperparameter tuning）
   - 记录 validation loss 和 benchmark accuracy

4. **分析**
   - 计算每个 LR 下 proxy ranking 与 7B ranking 的 correlation
   - 绘制 heatmap（类似 Figure 4b）
   - 确认 tiny LR 是否真的更好

#### 长期行动：重新定义 Proxy 评估协议

**当前协议**（可能有问题）：
```
For each mixture:
    Train proxy with fixed LR
    Record validation loss
Rank mixtures by validation loss
```

**改进协议**：
```
For each mixture:
    Train proxy with tiny LR (1e-5 or 1e-6)
    Record validation loss
Rank mixtures by validation loss
```

**理由**：
- Tiny LR 下的 ranking 更能反映 dataset 的内在质量
- 避免了 hyperparameter sensitivity 导致的 ranking flip
- 与大模型调参后的 ranking 更一致

---

## 两篇论文的协同效应

### 问题分解

QuaDMix 面临的 proxy 可靠性问题可以分解为：

1. **信号提取**：如何从小模型提取可靠的信号？
   - **Proxy Reliability 的回答**：用 tiny LR 训练

2. **信号外推**：如何将小模型信号外推到大模型？
   - **CAMEL 的回答**：用 capacity-aware mixture law

### 组合方案

**Step 1**: 用 tiny LR 训练多个尺度的 proxy（如 1M, 10M, 100M）

**Step 2**: 收集 (mixture, model_size, validation_loss) 三元组

**Step 3**: 拟合 CAMEL law
```
L_val(r, M) = C + Σ K_i / (⟨t_i, r⟩^α_i * M^β_i)
```

**Step 4**: 外推到目标模型大小（如 7B），预测最优 mixture

**Step 5**: （可选）拟合 loss-to-benchmark law，直接优化目标 benchmark

### 预期收益

| 维度 | 当前 QuaDMix | 改进后 | 提升 |
|------|-------------|--------|------|
| Proxy 可靠性 | 可能不可靠（标准 LR） | 高可靠性（tiny LR） | Spearman corr > 0.9 |
| 外推准确性 | 无显式外推 | Capacity-aware 外推 | 减少 extrapolation error |
| 优化目标 | Validation loss | Benchmark accuracy | 直接优化目标性能 |
| 计算成本 | 1M proxy | 1M + 10M + 100M proxy | 增加 ~2x，但仍远小于 7B |

---

## 实施路线图

### Phase 1: 立即修复（1-2 天）

**目标**：验证并应用 tiny LR 修复

**任务**：
1. 修改 proxy 训练代码，支持自定义 LR
2. 在 3-5 个 mixture 上测试不同 LR
3. 选择最佳 LR（预期 1e-5 或 1e-6）
4. 重新运行 QuaDMix pipeline

**成功标准**：
- Proxy ranking 与 7B ranking 的 Spearman correlation > 0.8

### Phase 2: Capacity-Aware 建模（1 周）

**目标**：实现 CAMEL law

**任务**：
1. 收集多个尺度（1M, 10M, 100M）的 proxy 数据
2. 实现 CAMEL 公式的拟合代码
3. 验证拟合误差
4. 用拟合的 law 预测 7B 的最优 mixture

**成功标准**：
- CAMEL 的 fitting error < baseline (DML, SODM)
- 预测的 mixture 在 7B 上优于 baseline

### Phase 3: Loss-to-Benchmark Prediction（2 周）

**目标**：直接优化目标 benchmark

**任务**：
1. 收集多个 validation set 的 loss
2. 收集对应的 benchmark accuracy
3. 拟合 logistic 形式的 prediction law
4. 用 prediction law 优化 mixture

**成功标准**：
- Prediction error < 5%
- 优化的 mixture 在目标 benchmark 上提升 > 1%

---

## 关键风险与缓解

### 风险 1: Tiny LR 导致训练不充分

**问题**：LR 太小，模型可能没有充分学习

**缓解**：
- 监控 training loss，确保在下降
- 增加 training steps（如 2x 或 3x）
- 验证 final loss 是否合理

### 风险 2: CAMEL 参数过多，过拟合

**问题**：CAMEL 有多个可学习参数，可能在小样本上过拟合

**缓解**：
- 使用 regularization（如 L2）
- Cross-validation 选择超参数
- 对比 baseline 的泛化性能

### 风险 3: Loss-to-benchmark 需要大量数据

**问题**：拟合 logistic 模型需要多个 validation set 和 benchmark

**缓解**：
- 先用少量 benchmark 验证可行性
- 逐步扩展 validation set 数量
- 考虑使用 pre-trained 的 prediction law（如果可用）

---

## 总结

### 核心洞察

1. **Proxy 可靠性是可修复的**：降低 LR 即可大幅提升信号可靠性
2. **Capacity-aware 建模是必要的**：模型大小和配比的交互效应不可忽略
3. **直接优化 benchmark 是可行的**：通过 loss-to-benchmark prediction

### 行动优先级

| 优先级 | 行动 | 预期收益 | 成本 |
|--------|------|---------|------|
| **P0** | 降低 proxy LR 到 1e-5 | 高（修复核心问题） | 低（改一行代码） |
| **P1** | 实现 CAMEL law | 高（改善外推） | 中（需要多尺度数据） |
| **P2** | Loss-to-benchmark prediction | 中（直接优化目标） | 高（需要大量数据） |

### 下一步

1. **立即**：修改 proxy LR，验证修复效果
2. **本周**：收集多尺度 proxy 数据
3. **下周**：实现并验证 CAMEL law
4. **下月**：探索 loss-to-benchmark prediction

---

## 附录：关键公式汇总

### CAMEL Law

```
L_val(r, M) = C + Σ_{i=1}^k K_i / (⟨t_i, r⟩^α_i * M^β_i)
```

### Loss-to-Benchmark Prediction

```
Acc_b(L) = C_b + A_b / (1 + exp(k_b^T L + B_b))
```

### Tiny Learning Rate

```
η_tiny ≈ 1e-5 ~ 1e-6  # 比标准 LR 小 1-2 个数量级
```

### Hourglass Sampling

```
优先采样最小和最大模型，减少中间尺度
```

---

**文档版本**: v1.0  
**最后更新**: 2026-06-23
