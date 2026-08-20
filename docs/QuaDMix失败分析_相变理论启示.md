# QuaDMix 1.5B 验证失败分析与 Phase Transitions 理论启示

> 核心问题：QuaDMix（14M proxy + 198 维搜索 + LightGBM）在 1.5B + 1.8B token 中训练验证中，大部分 STEM benchmark 得分**比 Random 还差**。本文从 Phase Transitions（NeurIPS'25 Spotlight, #01）理论出发，分析失败根因，并推导域级相变对数据配比的直接启示。

---

## 一、QuaDMix 在 1.5B + 1.8B 失败的根因

### 1.1 实验结果回顾

| Benchmark | QuadMix | Random | Manual (3:7) | QuadMix vs Random |
|---|---|---|---|---|
| GSM8K | 0.0697 | 0.0804 | 0.0834 | ❌ 差 |
| MMLU STEM | 0.2849 | 0.2922 | 0.2860 | ❌ 差 |
| ARC Challenge | 0.4445 | 0.4548 | 0.4369 | ❌ 差 |
| ARC Easy | 差 | 好 | — | ❌ 差 |
| GPQA | 0.2475 | 0.2273 | 0.2424 | ✅ 赢 |
| MATH | 好 | 差 | — | ✅ 赢 |

- QuadMix **Final Loss**（training loss）= 0.9166（最低），Random = 1.4508，Manual = 1.3583
- QuadMix **6 任务中 4 个落后于 Random**，仅 GPQA 和 MATH 领先
- 14M proxy 信号 R²：math=0.844, gsm8k=0.751, gpqa=0.661, arc_challenge=0.402, arc_easy=0.394, mmlu=0.302

### 1.2 根因：proxy gap（14M → 1.5B）

QuaDMix 失败的根因是 **14M proxy 的最优配比方向与 1.5B 相反**。Phase Transitions 论文从两个层面解释为什么不同模型的最优配比可以方向相反：

#### 原因 1：相变阈值差异 → 不同 regime

Phase Transitions 核心公式：

```
r_thres ∝ S / M^(α+1)     α ≈ 0.283, α+1 = 1.283
```

- S = 该域事实总数，M = 参数量
- 不同模型大小 M → 不同 r_thres → 不同的最优配比

| | 14M proxy | 1.5B target |
|---|---|---|
| r_thres（估） | >0.30-0.40（很高） | ~0.06-0.08 |
| r=0.7 时 | 可能刚跨阈值或在阈值附近 | 深入阈值以上 10 倍 |
| val_loss 信号 | R²=0.30-0.84（有信号但不稳定） | 应有更强信号 |
| accuracy | 0%（14M 学不到 STEM） | ~7-8%（在学） |

14M 的 val_loss **有信号**（R²=0.30-0.84，不是随机噪声——修正之前"小Proxy不可用"文档的说法），但这个信号来自与 1.5B **不同的 regime**：
- 14M 可能在"刚跨阈值"或"阈值以下靠 log-likelihood 分化"的状态
- 1.5B 在"深入阈值以上、线性学习区"
- 两个 regime 的最优配比方向可以**相反**

#### 原因 2：容量分配不同（背包问题）→ 最优解方向相反

Phase Transitions 的核心理论（§4）：模型 = 最优有界容量学习器，在容量 M 约束下最小化 test loss = **分数背包问题**。模型按**边际价值**（每单位容量的 test loss 下降量）分配容量。

- **14M**（容量极小）：某些 STEM 子域的边际价值 < web 数据 → 不值得学 → 最优配比 = **少给**这些子域
- **1.5B**（容量大）：同样子域的边际价值 > web 数据 → 值得学 → 最优配比 = **多给**这些子域
- 14M 的"少给"对 1.5B 是**反方向** → 比 Random（均匀）还差

**这就是 QuadMix 比 Random 还差的机制**：不是"信号无意义"，而是"信号方向相反"。

模型学习知识密集数据的条件（论文 Eq. 11-12）：

```
不学（随机猜）:  r/(1-r) < D⁻F_P2(M) / D⁺F_P1(0)
完全学:          r/(1-r) > D⁺F_P2(M-β) / D⁻F_P1(β)
```

其中 D⁻F 和 D⁺F 是 web 数据和知识数据的 loss 对容量的导数（边际价值）。不同 M 下这些导数不同 → 不同的分配决策。

#### 原因 3：R² 高 ≠ 传递好

R²=0.751（gsm8k）说明 LightGBM 能很好预测 **14M 自己**的 gsm8k val_loss。但：
- R² 是 **within-model**（14M → 14M），不是 **cross-model**（14M → 1.5B）
- 14M 的 val_loss 最优配比 ≠ 1.5B 的 accuracy 最优配比
- **证据**：gsm8k R²=0.751（高信号），但 QuadMix 在 gsm8k 上反而最差（0.0697 vs Random 0.0804）→ 高 R² 也没用，方向反了

### 1.3 直接证据链

| 证据 | 说明什么 |
|---|---|
| QuadMix Final Loss=0.9166（最低） | 14M 优化的配比确实在 1.5B 上降低了 training loss |
| QuadMix accuracy 4/6 任务最差 | 降低 training loss ≠ 提高 accuracy |
| QuadMix 2/6 任务最好（GPQA, MATH） | 14M 信号部分正确（R² 高的子域） |
| QuadMix 在 gsm8k 最差（R²=0.751） | 即使 R² 高，方向也可反 → proxy gap |
| Random（均匀）> QuadMix（4/6） | 14M 最优是**反方向**，不是"仅次优" |

### 1.4 贡献因素（加剧 proxy gap，非独立原因）

| 因素 | 说明 | 来源 |
|---|---|---|
| **loss ≠ value** | 即使 1.5B 直接搜索，val_loss 最优 ≠ accuracy 最优。Phase Transitions Fig 10 显示 val_loss 也有相变 → 有一定信号，但阈值效应可能让最优不同。**1.5B 未验证。** | Schaeffer 2023 + Phase Transitions |
| **198 维 × 336 实验 → 过拟合** | 搜索空间过大，LightGBM 预测精度随维度指数下降 → 过拟合 14M 噪声 → 加剧 proxy gap | Survey_RegMixD §C.3 |
| **< 0.5B proxy 不稳定** | FastMix 论文自己承认 <0.5B proxy 在工业数据上不稳定。14M << 0.5B → 信号不可靠 | FastMix doc line 326-327 |
| **验证集分布不匹配** | QuaDMix 用 OpenHermes-10k（SFT 数据）做验证集，与预训练分布不匹配 → 优化的 val_loss 方向进一步偏离 | FastMix doc line 338, PreSelect line 404 |

### 1.5 不是原因

| 非原因 | 为什么不是 |
|---|---|
| token 预算不够（1.26B vs 150B 参考） | r=0.7 >> r_thres → 模型在学 STEM。token 不够让所有策略 accuracy 低，但不会让 QuadMix **比 Random 差**。这是噪声底问题，不是方向反问题。 |
| training loss 混淆 | QuaDMix 搜索的 reward 是 benchmark Q&A val_loss（不是 training loss）。Final Loss 低是**结果**（14M 优化方向碰巧降低了 training loss），不是**原因**。 |

### 1.6 根因总结

```
根因 = proxy gap（14M → 1.5B）
  ├─ 相变阈值差异：14M 阈值附近 vs 1.5B 深入阈值以上 → 不同 regime
  ├─ 容量分配不同：14M 最优"少给某些子域" vs 1.5B 最优"多给" → 方向相反
  └─ R² within-model ≠ cross-model transfer

贡献因素（加剧）：
  ├─ loss ≠ value（未验证）
  ├─ 198 维过拟合
  ├─ < 0.5B proxy 不稳定（FastMix 警告）
  └─ 验证集分布不匹配（OpenHermes SFT vs 预训练）

结果：14M 最优配比方向与 1.5B 相反 → 比 Random 还差
```

---

## 二、Phase Transitions 论文核心信息

**论文**: Data Mixing Can Induce Phase Transitions in Knowledge Acquisition
**arXiv**: 2505.18091v3 (2026.05.10)
**作者**: Xinran Gu, Kaifeng Lyu, Jiazheng Li, Jingzhao Zhang（清华 IIIS + 上海期智）
**Venue**: NeurIPS 2025 Spotlight
**模型**: Pythia 14M - 6.9B

### 2.1 两个相变现象

#### 相变 1：模型大小相变
- 固定 r，增大 M → M < M_thres 时 accuracy ≈ 0%，M > M_thres 突跳到 >60%
- 转变是**尖锐**的，不是平滑过渡

#### 相变 2：配比相变
- 固定 M，增大 r → r < r_thres 时即使训 512B token（每条 biography 出现 ~3000 次）accuracy 仍 ≈ 0%
- r > r_thres 后快速开始记忆

### 2.2 临界配比幂律

```
r_thres ∝ S / M^(α+1)     α ≈ 0.283, α+1 = 1.283
```

**含义**：
- 模型越大 → r_thres 越低（所需 STEM 配比越低）
- 事实集越大 → r_thres 越高（需要更多 token 覆盖每个事实）
- 单事实阈值频率：`f_thres ∝ 1/M^1.283`

**关键数字**：

| 模型 | 数据集 | r_thres（实验观测） |
|---|---|---|
| 70M | SynBio-320k | ≈ 0.27-0.30 |
| 410M | SynBio-1.28M | ≈ 0.35-0.40 |
| 2.8B | SynBio-20k | 相变持续（Fig 9） |
| 6.9B | SynBio-10k | 相变持续（Fig 9） |

**1.5B 推算**（假设 S 不变）：
- (410M/1.5B)^1.283 = 0.273^1.283 ≈ 0.19
- r_thres(1.5B) ≈ 0.19 × r_thres(410M) ≈ **0.06-0.08**
- 用户 r=0.7 是 r_thres 的 **~10 倍** → 深入阈值以上区间

**70M + SynBio-320k 详细数据**（Table 1, 3 seed）：

| r | accuracy | |
|---|---|---|
| 0.10-0.25 | 0.4-0.8% | 阈值以下 |
| 0.30 | 7.3% | 刚跨阈值 |
| 0.35 | 27.5% | 快速增长 |
| 0.40 | 40.4% | |
| 0.45 | 58.1% | |

### 2.3 训练步数随 1/r 的增长

- r > 0.4：**线性增长**（1/r）
- r < 0.4：**指数甚至超指数增长**
- r = 0.25：实际步数 > 幂律预测 2.9×，> 指数预测 2×
- r = 0.2：实际步数 > 幂律预测 1.9×

**拟合**：
- 指数 T(r) = A·exp(B/r)，B ≈ 1.514，R² = 0.998
- 幂律 T(r) = C·r^(-3.839)，R² = 0.985

**关键实验**：r=0.2, 70M, 训到 256B token（每条 biography 出现 ~3000 次）→ accuracy 仍 ≈ 0%

### 2.4 背包理论 + 边际价值

模型 = 最优有界容量学习器（Definition 4.1），在容量 M 约束下最小化期望交叉熵 test loss = 分数背包问题。

- 给每个域分配 1 单位容量的"边际价值" = 该域 test loss 的下降量
- **知识密集数据"值得学"的条件**：`r/(1-r) > D⁻F_P2(M) / D⁺F_P1(0)`
- 低于阈值时，模型把**所有容量分给 web 数据**（STEM 边际价值太低）
- 高于阈值时，STEM 边际价值 > web → 模型主动学 STEM

**关键定理（Theorem 4.3）**：当 web data loss 幂律 F_P2(M) = C + A·M^(-α) 时：
- 临界模型大小：`M_thres ~ (1/(r·p))^(1/(α+1))`
- 临界配比：`r_thres ~ 1/(p·M^(α+1))`
- 阈值频率：`f_thres ~ 1/M^(α+1)`

### 2.5 两个增强策略

#### 策略 1：随机子采样（4x 提升）
- 减少 web 数据 → 提高有效 STEM 比例 → 每个事实频率上升
- WikiBio continual pre-training：子采样到 50% → 学到的事实数 **4 倍提升**
- 1B + SynBio-2.56M continual pre-training：子采样到 50% (r=0.2) → accuracy 32.94% vs 不子采样 r=0.4 (early stop) 4.66% → **7 倍提升**

#### 策略 2：Compact Knowledge Mixing (CKM)（20x 提升）
- 将知识重写为**紧凑形式**（元组、QA 对、知识列表），加入原始数据集，保持配比不变
- WikiBio 元组格式：`Bio: N {name} B {birth date} O {occupation}`
- CKM ratio τ = 紧凑形式 token / 原始 token
- τ = 0.1（~2x 频率）→ **4 倍提升**
- τ = 0.3（~3x 频率）→ **20 倍提升**
- 下游通用能力损失仅 ~1.8%
- **原理**：紧凑形式减少每个事实的平均 token 数 → 提高事实频率 f → f > f_thres

### 2.6 推理任务也有相变

- **slope calculation**（两点求斜率）：Figure 5 显示与事实知识**相同的相变模式**
- **Max-over-N**（30 个数找最大值，输入空间 10^60）：70M + r=0.5 → 60% accuracy（仅 3500 训练样本，证明泛化）
- "memorization is foundational to reasoning" — 不记住定理/定义就无法推理

### 2.7 proxy 警告（原文引用）

> "Naively measuring the performance of small models on a small data domain may provide little to no predictive signal on how well larger models perform, revealing a potential limitation of using small proxy models for data curation."

引用 5 篇论文佐证：Jiang 2024 (ADO), Kang 2024 (AutoScale), Ye 2024 (Data Mixing Laws), Magnusson 2025 (DataDecide), Mizrahi 2025。

### 2.8 continual pre-training 验证

论文在 continual pre-training 设定下验证了策略：
- 410M + 1B 模型，从 100k step checkpoint 继续训
- Early stopping 标准：Pile val_loss 上升 **0.05**（410M）或 **0.03**（1B）→ 对应 ~2% 下游损失
- Catastrophic forgetting：r 大时 Pile val_loss 上升

### 2.9 计算成本参考

| 模型 | 每 B token 耗时（A100） |
|---|---|
| 70M | 0.25h |
| 410M | 1.0h |
| 1B | 2.0h |
| 2.8B | 5.8h |
| 6.9B | 16.89h |

---

## 三、域级相变与 STEM trade-off 分析

### 3.1 每个域有自己的 r_thres

Phase Transitions 公式 `r_thres ∝ S / M^(α+1)` 中，S 是该域事实总数。不同 STEM 子域 S 不同 → r_thres 不同：

| 域 | S（事实数） | r_thres(1.5B) 估 | r=0.7 时状态 |
|---|---|---|---|
| 数学（基础算术/代数） | 较小 | 低 → 70% 远超阈值 | 正在学 ✅ |
| 物理/化学/生物（分散知识点） | 较大 | 可能更高 | 70% 中数学占比高时其他子域可能不够 ⚠️ |

### 3.2 高数学配比损害其他 STEM 任务的机制

背包理论（§4）的直接推论：
- r_math = 70% → 数学在 STEM 数据中占比高 → 数学获得更多容量 → 数学边际价值高（正在学）→ 模型多分配给数学
- 其他 STEM 子域在 70% STEM 数据中占比低 → 可能低于自己的 r_thres → 边际价值 < web → **模型把容量分给 web 而不是这些子域**
- 结果：**数学得分在涨，其他 STEM 得分停滞或下降**

这是**不是平滑 trade-off，而是相变 trade-off**：
- 某子域低于阈值 → 给再多数据也没用（accuracy=0%）→ 不如给数学
- 某子域高于阈值 → 有边际价值 → 值得分配容量

### 3.3 数学得分提升需要足够 token 走完相变

Phase Transitions 的三个层次：

#### 层次 1：跨过阈值（ratio 维度）
- r_math > r_thres(math, 1.5B) ≈ 0.06 → 70% 远超 → ✅ 已跨过
- 如果 r_math < r_thres → 训再久也是 0%（超指数增长）

#### 层次 2：走完相变（token 维度）
- 跨过阈值后，accuracy 不是立刻跳到 60%+，而是**随训练步数增长**
- r > 0.4 → 线性增长（每多一倍 token，accuracy 线性提升）
- r = 0.7 >> 0.4 → 在线性区，每多 1B token 都有收益

**用户的情况**：
- r_math = 0.7 → 在线性区（高效学习）
- 但 1.26B 数学 token 是 DeepSeekMath 参考（150B）的 ~1%
- GSM8K ~7-8% = "**刚跨阈值、刚开始学、还没走完**" 的状态
- 如果给 10B+ 数学 token → GSM8K 可能到 ~15-20%（线性增长推算）

#### 层次 3：推理任务也有相变
- slope calculation 和 Max-over-N 都验证了相同相变模式
- 数学推理（GSM8K）需要先记住基础算术规则 → memorization → reasoning

### 3.4 用户 1.5B 实验数据印证

| Benchmark | QuadMix | Random | Manual | 解读 |
|---|---|---|---|---|
| GSM8K | 0.0697 | 0.0804 | 0.0834 | ~7-8% → 刚跨阈值，还没走完 |
| MMLU STEM | 0.2849 | 0.2922 | 0.2860 | ~28-29% → 选择题有 25% 随机基线 |
| GPQA | 0.2475 | 0.2273 | 0.2424 | ~22-25% → 接近随机（25%）→ 可能低于阈值 |

**QuadMix（14M 优化的高数学配比）在 GPQA 上赢但在 GSM8K 上输**：
- GPQA：QuadMix 0.2475 > Random 0.2273 → 更多数学数据可能刚好把 GPQA 推过阈值
- GSM8K：QuadMix 0.0697 < Random 0.0804 → 14M 优化的配比方向反了（proxy gap），不是数学不够

### 3.5 域状态分类与最优操作

| 域状态 | 特征 | 识别方法 | 最优操作 |
|---|---|---|---|
| **远超阈值 + 有 headroom** | 正在学，loss 还在降 | L_curr >> b_i, α 中等 | 多给数据（线性收益） |
| **刚跨阈值** | accuracy 低但非零，loss 快速降 | L_curr >> b_i, α 高 | 多给数据（最高边际价值） |
| **低于阈值** | accuracy=0%，loss 平坦 | L_curr ≈ b_i, α 低 | 大幅加比例推过阈值，或放弃 |
| **已收敛** | loss 接近 b_i（渐近线） | L_curr ≈ b_i, α 高 | 减少数据（无 headroom，浪费） |

**DomainPilot 的三因子 score 与此对应**：
- `L_curr - b_i`（headroom）→ 高 = 正在学 → 多给
- `α_i`（收敛速度）→ 低 = 学得慢 → 可能低于阈值
- `b_i`（难度）→ 高 = 难域 → 需要更多数据

**但 DomainPilot 无法区分"已收敛"和"低于阈值"** — 两者都是低 headroom。需要看 α_i：
- 高 α + 低 headroom → 已收敛 → 减数据
- 低 α + 低 headroom → 可能低于阈值 → 加数据推过阈值 or 放弃

---

## 四、结论与行动方向

### 4.1 失败根因总结

QuaDMix 在 1.5B + 1.8B 验证不好的根因是 **proxy gap（14M → 1.5B）**，Phase Transitions 论文从理论层面解释：

1. **相变阈值差异**：14M 的 r_thres 很高（>0.30-0.40），1.5B 的 r_thres ≈ 0.06-0.08。14M 可能在阈值附近，1.5B 深入阈值以上。不同 regime 的最优配比方向可以相反。
2. **容量分配不同**：背包问题对不同 M 有不同解。14M 容量小 → 某些 STEM 子域边际价值低 → 最优 = 少给。1.5B 容量大 → 同样子域边际价值高 → 最优 = 多给。14M 的最优 ≠ 1.5B 的最优，可以**方向相反** → 比 Random 还差。
3. **R² within-model ≠ cross-model**：R²=0.751（gsm8k）只说明 predictor 能拟合 14M 的 val_loss，不说明 14M 最优 = 1.5B 最优。

**一句话**：QuaDMix 失败不是因为没有信号（R²=0.30-0.84 证明有信号），而是 14M 的信号来自不同的相变 regime + 不同的容量分配格局，导致最优配比方向与 1.5B **相反**。

### 4.2 修复方向：14M → 1.5B 直接搜索

| | QuaDMix 原版（已失败） | 修复版 |
|---|---|---|
| 搜索模型 | 14M proxy | **1.5B 直接** |
| 信号 | val_loss on benchmark Q&A | **同（但 1.5B 上的 val_loss）** |
| 框架 | 198 维 + LightGBM | **22 维简化**（quality-top-k 作预处理） |
| 每实验 token | ? | <1B |
| 实验数 | 336 | **~30-50**（CLIMB 式迭代 30→10→3） |
| proxy gap | ❌ 14M ≠ 1.5B | ✅ 无 |

**搜索成本**：30-50 实验 × ~1-2 小时 = 30-100 小时 ≈ 1.3-4 天

### 4.3 Step 0 验证（30 min 推理）

在 3 个已有 1.5B checkpoint（QuadMix, Random, Manual）上算 benchmark Q&A val_loss（answer token 交叉熵）：

| val_loss 排名 | accuracy 排名 | 结论 |
|---|---|---|
| Random < QuadMix（跟 accuracy 一致） | Random > QuadMix（已知） | ✅ loss → value 成立 → 修 proxy 就够 |
| QuadMix < Random（跟 accuracy 相反） | Random > QuadMix（已知） | ❌ loss ≠ value → 换信号源 |

这是 30 分钟决定整个方案走向的关键测试。

### 4.4 DomainPilot 作为兜底

| | QuaDMix at 1.5B | DomainPilot |
|---|---|---|
| proxy gap | ✅ fixed（1.5B 直接） | ✅ 无 proxy |
| loss≠value 风险 | ❌ 需 Step 0 验证 | ✅ **无**（用 training loss 轨迹，不用 val_loss） |
| 搜索成本 | 43 实验（1.3-2.5 天） | 0（在线，1 个 run） |
| 信号强度 | 更强（直接预测 accuracy，如果成立） | 更弱（学习效率 ≠ accuracy 贡献） |
| 预期提升 | ~1-2%（如果 loss→value） | ~1-2%（简单启发式） |
| 失败风险 | 30-40%（loss≠value） | 低（但提升幅度不确定） |

**DomainPilot 可信度**：低（单作者、无 venue、仅 SFT 验证），但方法思路合理（域级 loss 轨迹 > 样本级 loss 终点，~30 行 patch，<1% 开销）。

### 4.5 两条腿走路建议

1. **先做 Step 0**（30 min 推理）→ 判断 val_loss → accuracy 是否成立
2. **同时实现 DomainPilot patch**（~30 行，几小时代码）→ 作为训练基础设施
3. **如果 Step 0 通过** → QuaDMix at 1.5B 搜索（高 ceiling）+ DomainPilot 做训练监控
4. **如果 Step 0 不通过** → 直接用 DomainPilot 在线调权（安全兜底）

### 4.6 不含 CKM 的结论

Phase Transitions 论文提供的两个新杠杆中，CKM（20x 提升）改动太大暂不考虑。不含 CKM 时，论文主要提供**确认**而非新杠杆：
- 确认 14M proxy 不可靠（→ 支持 14M→1.5B 修复）
- 确认 r=0.7 在阈值以上（→ 策略差异应真实）
- 确认 continual pre-training 适用（→ 场景验证）
- 确认 early stopping 标准（→ val_loss +0.05）

新杠杆只有随机子采样（4x，QuaDMix 的 sampling 参数可能已等效）和 CKM（20x，暂不做）。

---

## 五、相互参照

- `docs/PhaseTransitions_深度分析_v2.md` — 相变理论详细分析（r_thres ∝ M^(-α), 背包模型）
- `docs/STEM能力涌现分析.md` — STEM 涌现可行性判断（1.5B + 1.26B 数学 token ≈ 参考 1%）
- `docs/小Proxy不可用_多论文交叉验证总结.md` — 6 篇论文交叉验证 proxy 不可靠（注意：本文修正了其中"proxy val_loss ≈ 随机噪声"的说法——14M 有信号 R²=0.30-0.84，但信号不传递到 1.5B accuracy）
- `docs/FastMix_DomainAwareScalingLaws_深度分析.md` — FastMix <0.5B proxy 不稳定警告（line 326-327）
- `docs/Survey_RegMixD_深度分析.md` — 198 维过拟合分析（§C.3）
- `docs/DomainPilot_深度分析.md` — 域级 loss 轨迹方法（兜底方案）
- `docs/业界模型技术报告_数据配比分析.md` — DeepSeekMath 数据锚点（150B 数学 token → 21% GSM8K）
- `docs/Nexus_深度分析.md` — 规模门控数据（GSM8K 130M≈0 → 3B+15）
