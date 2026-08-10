# 小 Proxy 不可用：多论文交叉验证总结

**创建日期**: 2026-08-10  
**结论**: QuaDMix 的 1M 随机初始化 proxy + val_loss 信号路径存在根本性缺陷，不可通过调参修复，需更换信号源或模型规模。

---

## 一、问题陈述

QuaDMix 当前流程：
1. 用 1M 参数随机初始化的 TinyLlama proxy（2 层, 8 heads, 256 hidden）
2. 对 336 种配比各训练一轮
3. 取 validation loss 作为信号
4. LightGBM 回归 + LCB 搜索找最优配比

**现象**：选出的配比比 Random 还差。

**根因诊断**：1M proxy 的 val_loss 不预测推理性能。

本文档汇总 6 篇论文从不同角度独立验证这一结论，并解释其深层原因。

---

## 二、六篇论文的独立证据

### 证据 1：Phase Transitions（NeurIPS'25 Spotlight，清华 IIIS）

**论文**: 2505.18091v3  
**模型**: Pythia 14M - 6.9B

**核心发现**：存在"相变"现象——

- 固定配比增大模型：低于临界大小 M_thres 时准确率 ≈ 0%，跨过阈值后突然跳到 60%+
- 固定模型增大配比：低于临界配比 r_thres 时，即使训练 512B tokens 也学不到任何知识
- 临界配比与模型大小满足幂律：`r_thres ∝ M^(-α)`

**论文直接点名**：
> "用小 proxy 模型做数据策展可能完全无信号，因为小模型可能低于相变阈值，根本学不到知识。"

**对 QuaDMix 的解释**：1M proxy 极可能远低于 STEM 知识获取的相变阈值 → proxy 在 STEM 域的 loss 几乎是随机噪声 → 优化器在拟合噪声。

详见：`docs/PhaseTransitions_深度分析_v2.md`

---

### 证据 2：CAMEL（ICML'26，清华，明确针对 mid-training）

**论文**: 2603.08022v2  
**模型**: 55B-A1.2B（索引中最大规模验证）

**核心发现**：不同能力的"容量分配比" R 不同——

- R_math ≈ 2.4，R_know ≈ 4.3
- 意味着 proxy 和 target 在不同能力上的容量分配**不成比例**
- proxy 的最优配比 ≠ target 的最优配比

**对 QuaDMix 的解释**：1M proxy 和目标大模型在数学/知识上的容量竞争格局完全不同，proxy 的最优配比对 target 无预测力。

**正面贡献**：发现 loss→benchmark 的映射可以用 logistic 函数拟合（多 val_loss → benchmark），为 Route A 提供降成本方案。

详见：`docs/CAMEL_深度分析_v2.md`

---

### 证据 3：Explaining DMSL（ICML'26）

**论文**: 2606.08167v3

**核心发现**：数据混合影响模型有两个机制——

1. **Capacity Competition**（容量竞争）：不同域竞争模型容量
2. **Noise Reduction**（噪声降低）：更多数据降低方差

- 纯容量分配下，最优配比 = 目标分布（h* = w），偏差完全来自噪声
- **"Shared Head, Disjoint Tail"**：头部共享容量，尾部竞争

**对 QuaDMix 的解释**：1M proxy 容量极小，容量竞争机制与目标模型根本不同 → proxy 的容量分配格局对 target 无参考价值。

详见：`docs/ExplainingDMSL_深度分析_v2.md`

---

### 证据 4：ADAPT（ICLR'26，Cambridge+Oxford）

**论文**: 2605.05227v1

**核心发现**：离线配比选择不如在线 reweighting——

- LESS（离线选择）在 OOD 上退化 -11.5 分（50.2→38.7）
- ADAPT（在线 reweighting）只退化 -5.9 分（50.7→44.8）
- 硬过滤牺牲数据多样性，用记忆替代泛化

**对 QuaDMix 的解释**：QuaDMix 的离线选择模式本身就有 OOD 退化风险——即使 proxy 信号可靠，离线选择也会损害泛化。

详见：`docs/ADAPT_深度分析_v2.md`

---

### 证据 5：Proxy Reliability（ICLR'26，Princeton+Berkeley+Stanford）

**论文**: 2409.05816v2

**核心发现**：固定超参数比较不同 data recipe 是不公平的——

- 不同 recipe 的最优 LR 不同，固定 LR 下排名可能翻转
- Standard LR (3e-4): Spearman ρ < 0.75
- Tiny LR (1e-5): Spearman ρ > 0.95

**对 QuaDMix 的解释**：QuaDMix 1M proxy 的 LR = 4e-4，比论文建议的 1e-5 大 40 倍。即使降 LR 也只是 patch——论文作者自己说"这不是根本解决方案"。

**关键局限**：论文验证范围是 70M-125M proxy → 1B target，QuaDMix 的 1M proxy 比验证范围小两个数量级，7B+ target 未验证。

详见：`docs/Proxy_Reliability_论文总结.md`

---

### 证据 6：DomainPilot（arXiv 2026.7，清华）

**论文**: 2607.22769v1

**核心发现**：与 QuaDMix 同类方法，但用域级 loss 轨迹而非单点 val_loss——

- 三因子 reward score：收敛难度 × 学习速度 × 剩余 headroom
- 域级 loss 轨迹 > 样本级 loss 终点
- 单点 val_loss 信号太弱

**对 QuaDMix 的解释**：即使保留 1M proxy，信号源也应从单点 val_loss 改为域级 loss 轨迹。DomainPilot 从另一个角度独立发现单点 val_loss 信号不足。

详见：`docs/DomainPilot_深度分析.md`

---

### 旁证：Nexus（arXiv 2026.4，清华+字节 Seed）

**论文**: 2604.09258v2

**核心发现**：同一 pretraining loss（差 0.004），GSM8K 差 15 分（44.0 vs 59.0, 3B 模型）。

**对 QuaDMix 的解释**：即使两个配比在 val_loss 上几乎相同，下游性能可能差距巨大。val_loss 作为信号在推理域上几乎是平的，无法区分好坏配比。

详见：`docs/Nexus_深度分析.md`

---

## 三、交叉验证矩阵

| 结论 | Phase Transitions | CAMEL | Explaining DMSL | ADAPT | Proxy Reliability | DomainPilot | Nexus |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 小 proxy 配比信号不可靠 | ✓（相变） | ✓（容量比不同） | ✓（容量竞争） | — | ✓（超参翻转） | ✓（信号太弱） | ✓（loss≠bench） |
| val_loss ≠ benchmark | ✓ | ✓（映射） | ✓（噪声偏差） | ✓（LESS 退化） | — | ✓ | ✓（15 分差距） |
| 离线选择有 OOD 风险 | — | — | — | ✓ | — | — | — |

**六篇论文从六个不同角度独立指向同一结论**：QuaDMix 的 1M proxy + val_loss 路径存在根本缺陷。

---

## 四、深层原因链路

```
1M proxy 远低于 STEM 相变阈值（Phase Transitions）
        ↓
proxy 在 STEM 域学不到任何知识（r_thres ∝ M^(-α)）
        ↓
proxy 的 STEM val_loss ≈ 随机噪声
        ↓
容量分配格局完全不同（CAMEL: R_math≠R_know, Explaining DMSL: 容量竞争）
        ↓
val_loss 不能区分好坏配比（Nexus: 同 loss 差 15 分）
        ↓
离线选择还有 OOD 退化（ADAPT: -11.5 分）
        ↓
LightGBM 拟合噪声 → 选出比 Random 差的配比
```

**这不是调参能修复的**——Phase Transitions 证明训练再长也没用，Proxy Reliability 证明降 LR 只是 patch，CAMEL 证明容量分配根本不同。

---

## 五、不可修复的论证

### 为什么不能通过以下方式修复：

| 修复方案 | 为什么不行 | 证据 |
|---------|-----------|------|
| 降低 LR | 只是 patch，1M 仍低于相变阈值 | Proxy Reliability + Phase Transitions |
| 训练更久 | 低于相变阈值时训练 16x 也没用 | Phase Transitions |
| 用 7M proxy | 已试过且失败（equal_weight 模式） | QuaDMix 实验 |
| 用预训练初始化 | 不解决推理域问题，1M 仍低于相变阈值 | Phase Transitions |
| 调整 val_loss 计算 | val_loss 本身不预测推理性能 | Nexus + HE-SNR |

### 结论

**1M proxy + val_loss 路径不可修复，必须更换信号源或模型规模。**

---

## 六、破局路线

### Route A：预训练小模型 + 直接评估 benchmark
- 用 Qwen2.5-0.5B（Pythia 410M 已跨过相变阈值，0.5B 应该也行）
- 直接评估 benchmark 而非 val_loss
- CAMEL 的 loss→benchmark 映射可降低成本
- **前提**：198 域需分组到 5-10 大类（Explaining DMSL 的 Disjoint Tail 原则）
- **成本**：50-200 GPU 小时

### Route B：Perplexity Correlations 式域级加权
- 4-6 个开源模型 forward pass only
- 算 loss-benchmark 相关性 → 域级权重
- 完全改变 QuaDMix 逻辑，成本极低
- Explaining DMSL 的 Noise Reduction 理论提供依据

### Route C：Nexus 优化器
- 不改配比，改优化器
- 双循环（内循环 NSGD + 外循环 AdamW），~20 行代码
- 自动平衡欠优化域
- 与 Route A/B 正交，可叠加

### Route D：ADAPT 在线 reweighting
- 用模型自身 hidden 表示做信号，不需要 proxy
- ~50 行代码，代码开源
- 但只在 120M 验证过

### 零成本验证：DomainPilot 式后验分析
- 对 336 个已有 proxy 实验做后验 scaling law 拟合
- 检查域级 loss 轨迹信号是否比单点 val_loss 更好
- 如果更好，说明信号源可以改进（不换模型，换信号）

---

## 七、建议优先级

1. **零成本先做**：DomainPilot 式后验分析（已有 336 个实验数据，只需重处理）
2. **低成本高价值**：Route B（Perplexity Correlations，只需 forward pass）
3. **中等成本根治**：Route A（0.5B 模型 + benchmark，CAMEL 映射降成本）
4. **可叠加**：Route C（Nexus，~20 行代码改动）

---

## 八、相关文档

| 文档 | 内容 |
|------|------|
| `docs/PhaseTransitions_深度分析_v2.md` | 相变现象详细分析 |
| `docs/CAMEL_深度分析_v2.md` | 容量分配 + loss→benchmark 映射 |
| `docs/ExplainingDMSL_深度分析_v2.md` | Capacity Competition + Noise Reduction |
| `docs/ADAPT_深度分析_v2.md` | 在线 reweighting vs 离线选择 |
| `docs/Proxy_Reliability_论文总结.md` | Tiny LR patch + Taylor 展开推导 |
| `docs/DomainPilot_深度分析.md` | 域级 loss 轨迹 + 三因子 reward |
| `docs/Nexus_深度分析.md` | 同 loss 差 15 分 + Nexus 优化器 |

---

**文档版本**: v1.0  
**创建日期**: 2026-08-10
