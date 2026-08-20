# STEM 能力涌现分析

> 核心问题：STEM（数学/推理）是"规模涌现"的能力吗？需要多大模型、多少数据才会涌现？应用于本项目 nanochat-npu 1.5B 中等训练的可行性判断。

---

## 一、业界共识演变（三层）

### 1.1 经典"涌现"叙事（Wei et al. 2022, Emergent Abilities of LLMs）

- GSM8K 等数学推理被列为 **emergent ability**：小模型上 ≈ 随机，到某个规模才"跳"出来。
- 原始图中，GSM8K 涌现出现在 ~10²² FLOPs 量级（大模型区，~几十 B 参数）。
- 这奠定了"STEM 是规模涌现能力"的业界初印象。

### 1.2 指标假象修正（Schaeffer et al. 2023, Are Emergent Abilities a Mirage?）

- 指出"涌现"部分是 **accuracy 阈值指标造成的假象**：换成 log-likelihood / 连续指标，增益是渐进平滑的。
- 但 **对下游 accuracy 目标，阶跃在实践中是真的**——用户关心的是 GSM8K accuracy，不是 log-likelihood，所以阶跃效应仍成立。

### 1.3 相变理论（Phase Transitions, NeurIPS'25 Spotlight，本项目 #01，理论根基）

本仓库已深度分析（见 `docs/PhaseTransitions_深度分析_v2.md`）。最精确的刻画：

- 相变是 **尖锐**的，不是平滑过渡：M < M_thres → 准确率 ≈ 0%；M > M_thres → 跳到 >60%。
- 相变是 **(模型大小 M, STEM 配比 r) 的二元联合曲面**，不是单一数字：
  - **模型大小相变**：固定 r，增大 M，跨过 M_thres 后突跳。
  - **配比相变**：固定 M，增大 r，跨过 r_thres 后才开始记忆；**r < r_thres 时训练再久也 = 0%**（实验：70M + r=0.2 训到 256B tokens，每条 biography 出现 ~3000 次，准确率仍 ≈ 0%）。
- **临界配比幂律**：`r_thres ∝ M^(-α)` —— 模型越大，所需 STEM 配比越低；模型越小，所需配比越高。
  - 推论：1M proxy 可能需要 >30% STEM 才能跨过相变点（这正是 QuaDMix 14M/1M proxy 失效的根本原因）。
  - Pythia 410M 已能跨过相变点。
- **容量分配 = 背包问题**：模型在容量 M 约束下最小化测试 loss；只有 r 或 M 超过阈值时，知识密集数据才"值得学"，否则容量全分给 web 数据。

**共识结论**：STEM 不是单纯"大小涌现"，而是 **(M, r, STEM token 绝对量, 数据质量) 的联合曲面**；阈值随 STEM 配比/质量/数据量下移。

---

## 二、门槛数据点（业界实证锚点）

| 来源 | 模型 | STEM 数据 | GSM8K（基座/少样本，SFT 前） | 备注 |
|---|---|---|---|---|
| 经典涌现 | GPT-3 家族 | 通用混合 | ~10²² FLOPs 才跳出 | Wei 2022 |
| Llama 2 base | 7B / 70B | 通用混合 | 7B ~14-15%（8-shot）/ 70B ~56% | 业界 |
| DeepSeekMath | 7B（续 DeepSeek-Coder） | 500B 总，56% 数学（≈280B 数学 token 含重复） | ~24% base | 本仓库 `docs/业界模型技术报告_数据配比分析.md` §4 |
| DeepSeekMath 消融 | 1.3B | 150B 数学 | ~21% base（Code→Math 两阶段） | 同上 §4.4 |
| Phi-3-mini | 3.8B | "教科书"高质量数学 + SFT | ~85%（SFT 后） | 高质量数据 + SFT 把 STEM 拉到小模型 |
| Nexus 规模门控 | 130M / 1B / 1.2B / 3B | 默认混合 | GSM8K 增益(Nexus−AdamW): 130M −2 / 1B +2 / 1.2B +7 / 3B +15 | `docs/Nexus_深度分析.md` §4.2 |

**关键锚点**（本仓库已验证）：
- **DeepSeekMath 1.3B + 150B 数学 token ≈ 21% GSM8K** —— 1.3B 能出 STEM，但需要 ~150B 数学 token。
- **DeepSeekMath 7B + 280B 数学 token（含重复）≈ 24% GSM8K** —— 7B 也只到 24% base。
- **Nexus 规模门控**：即使是优化器层面的"同 loss 更好下游"修复，STEM 增益也是 130M≈0 → 3B+15，**STEM 收益在 1B 以下几乎不存在，3B 才显著**。

---

## 三、应用于本项目：nanochat-npu 1.5B

### 3.1 基座

- **基座模型**：1.5B，预训练数据为 **ClimbLab**（CLIMB 论文 arXiv:2504.13161 发布的 1.2T 语料，20 个语义聚类）。
- **训练参考**：nanochat-npu（训练代码基）。
- **基座预训练充分度**：1.2T tokens / 1.5B 参数 ≈ 800 tokens/param，**远超 Chinchilla 最优（20 tokens/param）**，基座本身预训练充分。
- **基座 STEM 状况未知**：ClimbLab 的 20 个聚类是语义聚类（K-means on embeddings），**非领域标注**，无法直接得出 STEM 占比。基座可能已有一定 STEM 能力（若 ClimbLab 含数学/代码聚类），也可能很弱。**建议先评测基座在 GSM8K/MATH 上的零样本表现**，作为中训练前基线。

### 3.2 中训练 token 预算（决定性约束）

- **中训练总量**：1.5B × 1.2 = **1.8B tokens**（1.2 倍参数量的数据）。
- **STEM 部分**：3:7（通用:STEM）→ 70% × 1.8B = **1.26B 数学 token**。
- **对比 DeepSeekMath 参考**：

| | 数学 token | 相对参考 |
|---|---|---|
| 本项目中训练 | 1.26B | **1%** |
| DeepSeekMath 1.3B（~21% GSM8K） | 150B | 100% |
| DeepSeekMath 7B（~24% GSM8K） | 280B（含重复） | 100% |

- **结论：数学 token 仅 ~1% of 参考线。这是涌现的生死瓶颈，不是配比问题。**

### 3.3 配比合理性（3:7 = STEM 70%）

- 由 `r_thres ∝ M^(-α)`：模型越小，所需 STEM 配比越高。1.5B 属小模型，**高 STEM 配比方向正确**——70% 有助于跨过 (M, r) 相变阈值。
- 即：**配比选对了，问题不在配比**。Phase Transitions 支持小模型用高 r。
- 但配比正确 ≠ 数据量足够。Phase Transitions 的"低于阈值=0%"是 (M, r) 阈值；跨过阈值后仍需足够 token 走完相变。1.26B 数学 token 太可能"刚跨阈值但没走完"。

### 3.4 涌现可行性判断

| 维度 | 状态 | 评估 |
|---|---|---|
| 模型大小 M = 1.5B | 下沿 | DeepSeekMath 1.3B 先例存在（~21%），但需 150B 数学 token；1.5B 处于 STEM 涌现下沿 |
| STEM 配比 r = 70% | 充足 | 由 r_thres ∝ M^(-α)，70% 对 1.5B 足够高，方向正确 |
| STEM token 绝对量 = 1.26B | **严重不足** | 参考线 120-280B，本项目 ~1%，**大概率走不完相变** |
| 数据质量 | 未知 | 若已用 quality-top-k / 复述增强，可下移阈值；需确认 |
| 基座 | 通用（ClimbLab） | 非代码/数学专用基座，涌现阈值比 DeepSeek-Coder 基座高 |

**总体判断**：1.5B + 1.8B 总量（1.26B 数学 token）下，**STEM 涌现基本不可能**。即使配比正确、模型在下沿可涌现区间，数据量比参考线少 ~100×。预期 GSM8K 停留在个位数（接近基座水平）。**本次中训练应定位为"基座风味调整 / 数据管线验证 / 配比框架 dry-run"，而非 STEM 涌现实验。**

---

## 四、结论与行动项

### 4.1 现阶段定位

1. **1.5B + 1.8B 中训练不指望 STEM 涌现**。数学 token 量（1.26B）是 DeepSeekMath 参考（120-280B）的 ~1%，走不完相变。
2. **真正的 STEM 涌现要等**：① 模型放大到 3B+（Nexus 数据：3B 才有 +15 GSM8K 空间）；② 数学 token 到 ~100B 量级。
3. **当前运行的价值**：验证数据管线、配比框架、评测流程，为后续放大做准备。

### 4.2 行动项

1. **先评测基座**：在 GSM8K/MATH 上测 ClimbLab 基座的零样本表现，确定中训练起点。基座可能已有一定 STEM（ClimbLab 1.2T 含未知比例 STEM 聚类）。
2. **记录基座→中训练的 STEM delta**：即使绝对值低，delta 能反映 1.8B 中训练的边际收益。
3. **若要强化 STEM（当前 1.5B 预算内）**：
   - 提高 STEM 配比至更高（如 80-90%），用 `r_thres ∝ M^(-α)` 压低相变阈值——但数据量瓶颈仍在。
   - 用 Compact Knowledge Mixing（Phase Transitions §1.5）：把数学知识重写为更紧凑形式，提高每个事实的边际价值（Kimi K2/K3 的 SwallowMath 复述增强同思路）。
   - 代码数据补充：DeepSeekMath §4.4 证明 Code→Math 两阶段优于纯数学（1.3B 上 21.9% vs 20.5%）；本项目当前无 code 数据，是最大差异（见 README 行动项 #28）。
4. **长期 STEM 涌现路径**：
   - 模型放大到 3B+（Nexus 规模门控 + DeepSeekMath 7B 参考均指向 3-7B）。
   - 数学 token 提到 ~100B 量级（DeepSeekMath 参考）。
   - 基座可考虑代码/数学专用预训练（DeepSeek-Coder 基座降低涌现阈值）。

### 4.3 相互参照

- `docs/PhaseTransitions_深度分析_v2.md` —— 相变理论与 r_thres ∝ M^(-α) 定律（本文理论根基）。
- `docs/业界模型技术报告_数据配比分析.md` §4 —— DeepSeekMath 数据点与配比。
- `docs/Nexus_深度分析.md` §4.2 —— 规模门控数据（GSM8K 130M→3B 增益曲线）。
- `docs/小Proxy不可用_多论文交叉验证总结.md` —— 小 proxy 在 STEM 上失效的交叉验证。
