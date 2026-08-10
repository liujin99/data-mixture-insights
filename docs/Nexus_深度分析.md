# Nexus: Same Pretraining Loss, Better Downstream Generalization via Common Minima

**论文**: arXiv:2604.09258v2 (2026.4 提交, 2026.5 修订)
**机构**: 清华大学 (Jun Zhu 组) + 字节跳动 Seed
**作者**: Huanran Chen, Huaqing Zhang, Xiao Li, Yinpeng Dong, Ke Shen, Jun Zhu
**模型规模**: 130M - 3B (Llama 架构)
**代码**: 未开源（论文 Algorithm 3 给出完整伪代码）

---

## 一句话总结

**同一个 pretraining loss，GSM8K 差 15 分。** 标准优化器（AdamW）收敛到各域 minima 几何远离的点，Nexus 通过最大化梯度相似性收敛到各域 minima 接近的点，在 loss 不变的前提下大幅提升下游性能。

---

## 1. 核心问题

预训练数据是异构混合体（general/math/code/...），标准做法是最小化平均 loss：

```
L_train(θ) = (1/K) Σ L_k(θ)
```

**几何问题**：模型收敛到哪里？
- **Average-type minimizer**：最小化了总 loss，但各域 minima `θ_k*` 彼此远离
- **Intersection-type minimizer**：不仅最小化总 loss，还靠近各域各自的 minima

两种收敛点训练 loss 相同，但下游性能差异巨大。

---

## 2. 理论

### 2.1 Closeness 是二阶泛化指标

二阶 Taylor 展开（Eq.4）：

```
L_下游(θ_train*) = L_下游(θ_下游*) + ½ × Closeness × Directional Sharpness
```

- **Closeness** = `‖θ_train* - θ_下游*‖²`（参数距离）
- **Directional Sharpness** = Hessian 方向曲率

结论：更 close + 更 flat = 更好下游泛化。Flatness 已被充分研究，本文聚焦 closeness。

### 2.2 梯度相似性上界界定 Closeness（Theorem 3.1）

```
(1/K) Σ ‖θ - θ_k*‖² ≤ (G² / Kλ²) Σ_{i≠j} (1 - CosSim(∇L_i, ∇L_j))
```

直观理解：如果各域梯度方向一致，它们各自的 minima 必然相同。

### 2.3 二次型下的严格证明（Theorem 2.1）

假设各域 loss 局部二次，minima 服从 `P(μ, σ²I)`：

```
E[L_下游(θ_train*)] - C_train = (a/K) × σ²
```

**核心 trade-off**：同一训练 loss C_train 下，minima 方差 σ² 越小（越 close），泛化 gap 越小。代价是 basin intrinsic loss c_B 更高（更深的地势）。但下游泛化只依赖 σ²，不依赖 c_B。

### 2.4 推广到非二次型（Theorem 2.2）

只要 loss landscape 局部方向强凸，closeness → 更好 OOD 泛化的结论成立：

```
E[L_下游] - C_train ≤ (λ_max / 2K) × ((λ_max/λ_min)² + 1) × σ²
```

---

## 3. Nexus 算法

### 3.1 标准算法（Algorithm 1）

双循环结构：
- **内循环**（K 步）：随机抽域 s_m，做 Normalized SGD 步 `θ_{t,m} = θ_{t,m-1} - γ · g/‖g‖`
- **Pseudo-gradient**：`ĝ_t = θ_{t,0} - θ_{t,K}`（内循环总位移）
- **外循环**：用 ĝ_t 更新 θ（AdamW / SGDM 等）

**关键**：内循环必须用 NSGD（无 memory），否则破坏梯度相似性最大化的数学性质。

### 3.2 理论保证（Theorem 3.2）

Nexus 的期望更新方向 = 标准梯度 + 梯度相似性正则化项 + O(γ³) 误差：

```
E[ĝ_t] = γ Σ ∇L_i/‖∇L_i‖ - γ²(K-1)/4K × ∇(Σ CosSim(∇L_i, ∇L_j)) + O(γ³)
```

即 Nexus 近似最大化梯度余弦相似性，误差为 O(γ³)。

### 3.3 工程适配（Algorithm 3 — 实际使用的版本）

问题：Algorithm 1 要求每个外循环步骤遍历所有 K 个域，K>50 时 effective batch size 与标准设置差异过大。

解决方案：利用 gradient accumulation 结构近似：

```
# 标准训练 vs Nexus 对比
# 标准：accumulate gradients → optimizer step → zero grad
# Nexus：inner_model 逐步 NSGD 更新 → 累积位移 → outer optimizer step → 重置 inner_model

for batch in loader:
    loss = inner_model(batch)
    loss.backward()
    opt_inner.step()           # NSGD 立即更新 inner_model
    
    if accumulation_boundary:
        g_hat = inner_model - model   # 位移 = pseudo-gradient
        opt_outer.step(grad=-g_hat)   # AdamW 更新主模型
        inner_model = model.clone()    # 重置
```

**成本**：
- 计算开销：几乎为零（forward/backward 次数不变）
- 内存开销：多一个模型副本（可 CPU offload + 异步）
- 归一化方式：per-param（per-matrix），非全局向量归一化（受 μP 启发）

---

## 4. 实验结果

### 4.1 主实验（Table 1）

3B 模型，2× Chinchilla tokens：

| 指标 | AdamW | Nexus | 提升 |
|------|-------|-------|------|
| **Pretrain Loss** | 1.606 | 1.602 | **+0.004 (可忽略)** |
| OOD Loss | 1.302 | 1.290 | +0.012 |
| MMLU | 47.8 | 48.9 | +1.1% |
| GSM8k | 44.0 | 59.0 | **+15.0%** |
| MATH500 | 32.0 | 40.0 | +8.0% |
| HumanEval | 43.0 | 47.0 | +4.0% |
| MBPP | 38.0 | 38.0 | 0.0% |
| **Avg Acc** | **37.1** | **40.3** | **+3.2%** |

1B 模型同样模式：loss 差 0.000，Avg +1.7%，GSM8k +2.0%。

### 4.2 模型规模 scaling（Fig.5）

Nexus 增益随模型变大单调放大：
- 130M: +0.8%
- 300M: +1.5%
- 520M: +1.7%
- 1.2B: +2.6%
- 2.3B: +3.2%

GSM8k 增益从 130M 几乎为 0 扩大到 3B 的 +15%。

### 4.3 训练 token scaling（Table 2）

3B 模型从 2×→4× Chinchilla：
- AdamW: 37.1 → 38.0（自然提升）
- Nexus: 40.3 → 41.7（保持领先）
- 优势不随训练变长而缩小

### 4.4 数据混合鲁棒性（Section 4.5 — 最关键）

3B 模型，math 比例 10%/40%/70%：

| 切换 | 域 | AdamW 波动 | Nexus 波动 |
|------|----|-----------|-----------|
| Math10→70 | GSM8k | 33 分 (77→44) | 17 分 (76→59) |
| Math10→70 | MMLU | 3.8 分 | <1 分 (49.8→48.9) |

**核心发现**：
- Math 稀疏(10%) → Nexus 大幅提升 math（GSM8k +15），"挖掘"稀疏信号
- Math 密集(70%) → Nexus 自动转向提升 General（MMLU +5.8）和 Reasoning（GPQA +3.1）
- 论文原文："Nexus reduces sensitivity to manual data mixing ratios, acting as an automatic balancer that prioritizes representations for the most under-optimized tasks"

**Nexus = 隐式数据混合器**：自动补偿数据少的域，降低对人工配比的敏感度。

### 4.5 学习率调度鲁棒性（Table 3）

WSD 和 Cosine 两种调度下 Nexus 均有效：
- Cosine + Nexus: GSM8k 59.0 vs 60.0(AdamW)，Avg 43.4 vs 41.0

### 4.6 SFT 后保持优势（Table 11, Section 14.6）

Pre-SFT → Post-SFT：
- Nexus Avg 41.4 → 42.7
- AdamW Avg 40.7 → 40.7
- Nexus 在 MATH +7.0%, HumanEval +6.0%, MBPP +6.0%

**结论**：Nexus 的 pretraining 优势不因 SFT 消失，甚至放大。

### 4.7 公开数据集（Section 14.3）

在含污染的公开数据集上，Nexus 有效抵抗 shortcut over-memorization，将容量重分配到未污染任务。

### 4.8 Muon 对比（Section 5.2）

- Muon 的增益来自更低 pretraining loss（-0.029），不是更好的 implicit bias
- Nexus 增益来自 implicit bias（loss 不变 +0.004）
- Muon + AdamW 曲线完全重叠 → Muon 没有更好的 implicit bias

### 4.9 消融（Section 5.3）

- **NSGD（K=1）= 标准 Adam**：确认增益来自内循环 K>1 产生的相似性正则化
- **Dot product 版本（Nexus-Dot）失败**：梯度范数 shortcut 破坏了 loss 最小化，cosine 必须用归一化梯度

---

## 5. 关键限制

1. **不兼容 Muon**：Nexus+Muon 不如 Nexus+AdamW，可能因 Newton-Schulz 迭代的数值敏感性
2. **内循环必须 NSGD**：不能用 momentum（Adam/SGDM 做内循环都不行），否则破坏相似性最大化
3. **K>50 工程适配**：通过 mini-batch 级别近似（连续 batch 来自不同域概率高），但这是近似不是精确
4. **未开源代码**：但 Algorithm 3 伪代码仅 11 行，可自行实现

---

## 6. 与 QuaDMix 的关联

### 6.1 val_loss 不充分的终极证明

QuaDMix 诊断结论是"1M proxy val_loss 奖励表面特征，不奖励推理"。Nexus 提供了更根本的证据：

**不是 proxy 太小的问题，是 val_loss 这个信号本身缺失了一个完整维度。** 即使 3B 模型，同样的 val_loss，GSM8k 可以差 15 分。val_loss 衡量的是"Average-type minimizer 质量"，完全无法区分"Intersection-type minimizer"。

### 6.2 第三条破局路线（Route C）

| 路线 | 方法 | 成本 | 与 QuaDMix 的关系 |
|------|------|------|------------------|
| A | 预训练小模型 + benchmark 评估 | 50-200 GPU-h | 替代 val_loss 信号 |
| B | Perplexity Correlations 域级加权 | 极低 | 替代整个 QuaDMix 框架 |
| **C** | **Nexus 优化器** | **代码改动小** | **正交，可叠加** |

Route C 不改数据配比，改优化器，自动平衡欠优化域。与 Route A/B 正交，可叠加使用。

### 6.3 对 3:7 配比的直接含义

用户当前 3:7（通用:STEM）配比下：
- Nexus 可能自动补偿 STEM 侧（70% 时 STEM 是多数域，Nexus 会自动提升 30% 的通用侧）
- 降低了精确配比的重要性
- 如果 Nexus 真有效，QuaDMix 配比搜索的价值打折扣

### 6.4 中训练适用性

Nexus 原理（梯度相似性 → closeness → downstream 泛化）不依赖 pretraining 特有机制：
- SFT 实验证明 gains 在后续训练阶段保持
- mid-training 同样是多域混合训练场景
- Algorithm 3 的 gradient accumulation 适配直接适用于 mid-training

### 6.5 诊断工具

在 QuaDMix proxy 训练中可测量：
- `CosSim(∇L_general, ∇L_STEM)` — 各域梯度一致性
- 如果很低（<0.1），说明 AdamW 确实收敛到 distant minimizer，Nexus 有空间

---

## 7. 可操作建议

### 7.1 最快验证路径

在 nanochat 训练脚本中实现 Algorithm 3：

1. 训练循环中新增 `inner_model = model.clone()`
2. 每个 mini-batch 对 inner_model 做 NSGD 更新
3. accumulation boundary 时 `g_hat = inner_model - model` 喂给 AdamW
4. 重置 `inner_model = model.clone()`

代码量约 20-30 行。额外内存 = 一个模型副本（可 CPU offload）。

### 7.2 验证实验设计

- **基线**：3:7 配比 + AdamW（当前设置）
- **实验**：3:7 配比 + Nexus+AdamW
- **评估**：STEM benchmark（GSM8K, MATH, etc.）+ 通用 benchmark（MMLU, HellaSwag）
- **关键观察**：Nexus 是否在通用域提升更大（因为 STEM 是 70% 多数域）

### 7.3 诊断先行

在现有 QuaDMix 实验数据中：
1. 记录各域梯度 `∇L_k`（需 per-domain batch forward/backward）
2. 计算 `CosSim(∇L_general, ∇L_STEM)` 沿训练步数的轨迹
3. 如果始终很低，验证了 AdamW 收敛到 distant minimizer

---

## 8. 总结评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 理论深度 | ★★★★★ | 完整理论链：closeness → gradient sim → downstream gen |
| 实验质量 | ★★★★★ | 130M-3B, 多数据混合, 多调度, SFT 验证 |
| 机构可信度 | ★★★★★ | 清华 Jun Zhu 组 + 字节跳动 Seed |
| 与 QuaDMix 相关性 | ★★★★☆ | 非数据配比方法但直接影响配比价值判断 |
| 可操作性 | ★★★★★ | 11 行伪代码, 几乎零计算开销, 可立即验证 |
| 风险 | 低 | 方法简单，最坏情况 = 回退到 AdamW |

**总体判断**：Nexus 是一个低成本、低风险、高潜力的正交方法。它不替代数据配比优化，但可能显著降低对精确配比的依赖。建议立即在 nanochat 中实现并验证。
