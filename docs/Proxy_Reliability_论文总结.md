# Proxy Reliability 论文总结

**论文**: Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice  
**来源**: ICLR 2026  
**作者**: Princeton, Berkeley, Stanford, Tsinghua, Virginia Tech

---

## 核心论点

**用相同超参数比较不同 data recipe 是不公平的，因为每个 recipe 的最优超参数不同。**

业界标准做法是用固定超参数（如 LR=3e-4）训练 proxy 模型来评估不同数据配方，认为这样"公平"。但论文证明：
- 不同 data recipe 的最优 LR 不同
- 固定 LR 下的排名可能在 LR 微调后翻转
- 应该比较"各自调到最优后谁赢"，而非"在这个固定 LR 下谁赢"

---

## 关键发现

### 1. Learning Rate 敏感性

**实验**：DCLM vs DCLM-dedup-GS，在 GPT2-Small 上用两个相近 LR 训练

| Learning Rate | 获胜 Recipe |
|---------------|-------------|
| 2.5e-4 | DCLM |
| 3.5e-4 | DCLM-dedup-GS |

**结论**：LR 只差 1e-4，排名就完全翻转。

### 2. Tiny Learning Rate 修复

**建议**：把 proxy LR 降到 **1e-5 ~ 1e-6**（比标准小 1-2 个数量级）

**效果**：
- Standard LR (3e-4): Spearman ρ < 0.75（proxy vs target 排名不一致）
- **Tiny LR (1e-5)**: Spearman ρ > 0.95（排名高度一致）

---

## 数学原理

### Taylor 展开推导

一步梯度下降：`θ_new = θ - η·∇ℓ(θ)`

Validation loss 变化：
```
Δℓ_val = ℓ_val(θ_new) - ℓ_val(θ)
       = ℓ_val(θ - η·∇ℓ(θ)) - ℓ_val(θ)
```

对 ℓ_val 在 θ 处做 Taylor 展开：
```
f(x + δ) ≈ f(x) + ∇f(x)ᵀ·δ + ½·δᵀ·H·δ
```

令 `δ = -η·∇ℓ(θ)`，代入：

```
Δℓ_val ≈ ∇ℓ_valᵀ·(-η·∇ℓ) + ½·(-η·∇ℓ)ᵀ·H·(-η·∇ℓ)
       = -η·[∇ℓ_val·∇ℓ]  +  ½η²·[∇ℓᵀ·H·∇ℓ]
         ─────────────       ─────────────────
           一阶项                二阶项
```

### 两项的含义

**一阶项**：`-η · ∇ℓ_val · ∇ℓ`
- 训练梯度和验证梯度的**内积**（梯度对齐）
- 衡量 train-val 分布相似度
- **只取决于数据本身**，跟 LR、模型大小无关
- 是数据的**内在属性**

**二阶项**：`½η² · ∇ℓᵀ · H · ∇ℓ`
- 训练梯度在验证 Hessian 上的二次型
- 涉及 loss 地形的曲率
- **受训练配置影响**（模型大小、训练进度、batch size）
- 不同 recipe 的 Hessian 不同

### 为什么 Tiny LR 有效

当 η 很小时：
```
η = 1e-5  →  一阶项 ∝ 1e-5,  二阶项 ∝ 1e-10
```
二阶项比一阶项小 **10 万倍**，可忽略。排名完全由一阶项（数据分布相似度）决定。

当 η 中等时：
```
η = 3e-4  →  一阶项 ∝ 3e-4,  二阶项 ∝ 9e-8
```
二阶项只小 3000 倍，如果不同 recipe 的 Hessian 差异大，就可能翻转排名。

### 数字例子

两个 recipe A 和 B：

| | 一阶项 | 二阶项 |
|---|---|---|
| A | 0.8 | 10000 |
| B | 0.6 | 10 |

**η = 1e-5**：
```
ΔL_A ≈ -8e-6 + 5e-7 ≈ -7.5e-6
ΔL_B ≈ -6e-6 + 5e-9 ≈ -6e-6
→ A 赢（一阶项主导）
```

**η = 3e-4**：
```
ΔL_A ≈ -2.4e-4 + 4.5e-4 = +2.1e-4  ← loss 上升！
ΔL_B ≈ -1.8e-4 + 4.5e-9 ≈ -1.8e-4
→ B 赢（二阶项主导，排名翻转）
```

---

## 直觉理解

### 类比 1：称苹果

- **Tiny LR** = 精密天平，称出真实重量差
- **Standard LR** = 弹簧秤，不同苹果的弹性不同，弹簧反弹力（二阶效应）可能让轻的苹果显示更重

### 类比 2：下山

- **Tiny LR** = 迈一小步，坡度完全由脚下地形决定 → 测的是"这条路起点陡不陡"
- **Standard LR** = 迈一大步，受路上坑和弯道影响 → 测的不只是路好不好，还有弯弯绕绕

### 类比 3：运动员

- **一阶项** = 运动员的**天赋**（到哪都强）
- **二阶项** = 运动员和**这双鞋的适配度**（换双鞋就不行了）

Proxy 的目的是选天赋最好的，不是选穿这双鞋最快的。因为大模型一定会重新调鞋（调超参）。

---

## 实验设置

### 模型规模

| 角色 | 模型 | 参数量 |
|------|------|--------|
| Proxy | GPT2-Small | 125M |
| Proxy | Pythia-70M | 70M |
| Proxy | OPT-125M | 125M |
| Target | GPT2-Large | 774M |
| Target | Pythia-1B | 1B |

**最大 target 只有 1B**，未验证 7B+。

### 训练配置

- Token-per-parameter ratio: 20（Chinchilla optimal）
- Context length: 1024
- Optimizer: AdamW (betas 0.9/0.999)
- LR scheduler: Warmup-Stable-Decay
- 每个 recipe 3 个 random seed
- **总共 >20,000 次训练**

### Data Recipes

**23 个 recipe**，覆盖 4 个维度：
1. **Domain composition** (10): Pile 子集的不同配比
2. **Established datasets** (3): C4, DCLM, RefinedWeb
3. **Quality filtering** (6): 不同 perplexity 阈值
4. **Deduplication** (4): 不同去重严格度

---

## Tiny LR 的推导

### 理论指导

η_tiny ≪ 1/λ_max，其中 λ_max 是 validation loss Hessian 的最大特征值。

可通过短 warmup 训练后的 checkpoint 估计。

### 经验法则

使用比标准 LR **小 1-2 个数量级**的 learning rate。

对于 LLM pretraining：
- 标准 LR: 1e-4 ~ 3e-4
- Tiny LR: **1e-5 ~ 1e-6**

### 下限

不能低于 ~1e-7（32-bit 浮点精度限制，参数更新 < 1e-7 会被 flush 到 0）。

---

## 对 QuaDMix 的启示

### 当前设置

QuaDMix 1M proxy 的 LR = **4e-4**，比论文建议的 1e-5 大 **40 倍**。

### 行动建议

**P0（立即）**：
```python
# 修改 src/quadmix/pipeline/essential_proxy_runner.py:77
learning_rate: float = 1e-5  # 从 4e-4 改为 1e-5
```

**验证步骤**：
1. 选 5-10 个多样化 mixture
2. 用不同 LR (4e-4, 1e-4, 1e-5, 1e-6) 训练 proxy
3. 在 7B 模型上验证排名
4. 计算 Spearman correlation，选最高的 LR

### 适用性评估

| 维度 | 论文设置 | QuaDMix 设置 | 差距 |
|------|---------|-------------|------|
| Proxy 大小 | 70M~125M | **1M** | 小 70-125x |
| Target 大小 | 774M~1B | 可能 7B+ | 未验证 |
| Proxy LR | 3e-4 → 1e-5 | 4e-4 | 比 standard 还高 |
| Tokens | 2.5B (proxy) | 1B | 更少 |
| 模型架构 | GPT2/Pythia/OPT | TinyLlama | 不同 |

**关键问题**：
- 论文的结论基于 70M-125M proxy → 1B target
- QuaDMix 用的是 **1M proxy**，比论文验证范围小两个数量级
- Tiny LR 的建议可能仍然适用，但**外推到 7B+ target 的可靠性未被论文验证**

---

## 局限性

1. **最大 target 只有 1B**，7B+ 没验证
2. **Proxy 最小 70M**，1M proxy 没验证
3. **理论证明只在 random feature model 上成立**，不是 transformer
4. **只做了 single-epoch**
5. **作者自己说这是 patch，不是根本解决方案**——长期应该联合优化数据和超参数

### Random Feature Model vs Transformer

**Random Feature Model (RFM)**：
```
f(x) = W · φ(x)
```
- φ(x) = 固定的随机特征映射（不学习）
- W = 唯一可学习的线性层
- 本质是随机初始化特征 + 线性回归

**为什么用 RFM**：
- 数学上可解，能推导出严格证明
- 和 Neural Tangent Kernel (NTK) 有对应关系

**和 Transformer 的差距**：

| 维度 | RFM | Transformer |
|------|-----|-------------|
| 特征 | 随机固定 | 端到端学习 |
| 可学习参数 | 一层线性 | 几十层非线性 |
| 训练动态 | 凸优化 | 非凸优化 |
| 二阶效应 | 简单可控 | 复杂不可控 |

**意味着什么**：
- 理论上证明了简化版
- 实际效果靠实验验证（在 GPT2/Pythia/OPT 上有效）
- 但没人能保证在 Transformer 上一定成立

---

## 总结

### 论文核心贡献

1. **发现问题**：固定超参数比较 data recipe 是不公平的
2. **证明问题**：LR 微调就能翻转排名（实验 + 理论）
3. **给出 patch**：降低 LR 到 1e-5 ~ 1e-6
4. **解释原理**：小 LR 下一阶效应主导，排名由数据分布相似度决定

### 对 QuaDMix 的价值

- **立即行动**：降低 proxy LR（改一行代码，成本极低）
- **需要验证**：1M proxy 和 7B+ target 是否适用
- **长期方向**：联合优化数据和超参数（不是 patch）

### 关键洞察

**Proxy 模型的任务不是"学好"，而是"测出数据好不好"。**

Tiny LR 让 proxy 变成一个**数据质量探测器**，而不是一个真正在学习的模型。

---

**文档版本**: v1.0  
**创建日期**: 2026-06-23  
**最后更新**: 2026-06-23
