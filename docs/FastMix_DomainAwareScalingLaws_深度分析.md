# FastMix & Domain-Aware Scaling Laws 深度分析

**分析日期**: 2026-07-20

---

## 总览

| # | 论文 | 机构 | 会议 | 核心方法 | QuaDMix 相关性 |
|---|------|------|------|----------|----------------|
| P0 | FastMix | 腾讯混元 + 港大 + 中大 | **ICLR 2026** | 可微 bilevel 优化，1 个 proxy | **★★★★★** |
| P1 | Domain-Aware Scaling Laws | MIT CSAIL + Microsoft Research | arXiv (2026.7) | 域间协同量化，观测拟合 | **★★★★☆** |

---

# 一、FastMix: Fast Data Mixture Optimization via Gradient Descent

## 1.1 论文概览

| 维度 | 信息 |
|------|------|
| arXiv | 2606.14971 (2026.06.12) |
| 机构 | **腾讯混元 LLM + 港大 (Xiaojuan Qi) + 中大** |
| 会议 | **ICLR 2026（已接收）** |
| 代码 | github.com/hrtan/fastmix（已开源，2 commits） |
| 作者 | Haoru Tan, Sitong Wu, Yanfeng Chen, Jun Xia, Ruobing Xie, Bin Xia, Xingwu Sun, Xiaojuan Qi |
| 方向 | 可微 bilevel 优化数据配比 |

### 作者可信度

| 作者 | 背景 |
|------|------|
| Haoru Tan | 港大 EEE，第一作者 |
| Sitong Wu | 通讯作者之一 |
| Yanfeng Chen | 腾讯混元，† 通讯 |
| Xiaojuan Qi | 港大，† 通讯，知名 CV/ML 研究者 |
| Ruobing Xie | 腾讯混元，清华 NLP 背景 |

**可信度评估**: ★★★★☆ — ICLR 2026 接收是强信号，腾讯混元是工业级 LLM 团队，有代码。但论文仅 2 commits，star=3，社区验证尚浅。

## 1.2 核心方法

### 1.2.1 问题重参数化：从采样概率到 loss 权重

**原始问题（不可微）**：

```
min_α L_target(w*(α))
s.t. w*(α) = argmin_w L_train(D, w | α)   # α 控制采样概率
```

α 作为采样概率是不可微的（离散采样操作）。

**FastMix 重参数化（可微）**：

```
min_α L_target(w*(α))
s.t. w*(α) = argmin_w Σ_i α_i · L_train(D_i, w)   # α 作为 loss 权重
```

**等价性证明**：

```
L_train(D, w | α) = E_{i~Cat(α)} E_{x~D_i}[ℓ(x,w)]
                   = Σ_i α_i · L_train(D_i, w)
```

即：按 α 采样后求期望的 loss = 按均匀采样后对每个域的 loss 加权 α_i。

**关键洞察**：这个等价性使得 α 变成连续可微参数，可以用梯度下降优化。

### 1.2.2 双层优化算法

**内层循环（模型参数更新）**：

```
w^{t+1} ← w^t - η_w · ∂[Σ_i α_i^t · L_train(D_i, w^t)] / ∂w^t
```

每 n₁ 步更新一次模型参数。

**外层循环（配比权重更新）**：

```
α^{t+1} ← α^t - η_α · ∂L_target(w^{t+n₂}) / ∂α^t
```

每 n₁ 步触发一次配比更新。

### 1.2.3 闭式梯度（n₂=1 时）

当 n₂=1 且用 SGD 时，梯度有闭式解：

```
∂L_target(w^{t+1}) / ∂α_i^t = -η_w · ∇_w ℓ_val(V, w^{t+1}) · ∇_w L_train(D_i, w^t)
```

**直觉**：
- 梯度 = **验证集梯度** 与 **领域 i 训练梯度** 的**内积**
- 内积 > 0（梯度对齐）→ 增加 α_i（该域对验证有益）
- 内积 < 0（梯度冲突）→ 减少 α_i（该域对验证有害）
- 内积 ≈ 0（梯度正交）→ 不调整

### 1.2.4 泛化增强

**熵正则化**：

```
R_entropy = Σ_i α_i · log(α_i)
```

防止配比坍缩到少数域。

**辅助训练 loss 目标**：

```
L_target(w) = ℓ_val(w) + β · L_train(w) + λ · Σ_i α_i · log(α_i)
```

β=0.1, λ=1e-5 为推荐值。

## 1.3 代码实现细节（来自 GitHub 源码精读）

### 1.3.1 Proxy 模型架构

```python
config = LlamaConfig(
    vocab_size=50432,
    hidden_size=256,          # n_embd = 256
    intermediate_size=512,    # FFN hidden dim = 512
    num_hidden_layers=2,      # n_layer = 2
    num_attention_heads=8,    # n_head = 8
    max_position_embeddings=2048,
)
model = LlamaForCausalLM(config)
```

| 参数 | FastMix | QuaDMix |
|------|---------|---------|
| 层数 | 2 | 2 |
| Hidden dim | 256 | 256 |
| Attention heads | 8 | 8 |
| FFN dim | 512 | 512 |
| Vocab size | 50,432 | 50,432 |
| Block size | 2048 | 2048 |
| 总参数 | ~14M | ~14M |
| 非 embedding 参数 | ~1.3M | ~1.3M |

**惊人发现**：FastMix 和 QuaDMix 的 proxy 模型**架构完全相同**（都是 2 层 256d LLaMA）。

### 1.3.2 梯度计算实现

代码中只使用 **lm_head 参数**计算梯度（不是全模型）：

```python
# Step 1: 验证集梯度（仅 lm_head）
val_head_grads = [p.grad.detach().clone() for p in model.lm_head.parameters()]

# Step 2: 每个领域的训练梯度（仅 lm_head）
group_head_grad = [p.grad.detach().clone() for p in model.lm_head.parameters()]

# Step 3: 内积
inner_prod = sum([(val_grad * group_g).sum()
                  for val_grad, group_g in zip(val_head_grad, group_head_grad)])
```

**为什么只用 lm_head？**
- lm_head 是模型输出层，维度 = vocab_size × hidden_dim = 50432 × 256 ≈ 12.9M 参数
- 相比全模型梯度，lm_head 梯度更稳定、计算更便宜
- 论文未讨论此选择，但代码中硬编码

### 1.3.3 配比权重更新

```python
# 初始化：全 1（均匀 pre-softmax logits）
dataset_probs = torch.nn.Parameter(torch.ones(K, requires_grad=True))

# Softmax 映射到概率
probs = F.softmax(dataset_probs, dim=0)

# 优化器：AdamW，lr=1e-4
dataset_optimizer = torch.optim.AdamW(
    [probs_module.probs], lr=0.0001, weight_decay=0, betas=(0.9, 0.95)
)

# 手动梯度赋值
grad = compute_dataset_probs_gradient(probs, inner_products, lr, lambda_reg)
probs_module.probs.grad = grad
dataset_optimizer.step()
```

### 1.3.4 内层/外层循环交替

| 参数 | 值 |
|------|-----|
| 内层步数 n₁ | 1000（eval_step_interval） |
| 外层步数 n₂ | 1（固定） |
| 总训练步数 | 25,000 steps |
| 外层更新次数 | ~25 次 |
| Global batch size | 512 |
| Micro batch size | 4 |
| Gradient accumulation | 16 |
| 总 token 数 | ~26.2B（3 epochs） |

### 1.3.5 熵正则化状态

```python
lambda_reg = 0.00000000000   # 实际为 0，已禁用
```

代码中完整实现了熵正则化梯度，但系数设为 0。**论文声称使用 λ=1e-5，但代码中未启用。**

### 1.3.6 训练 loss 辅助目标

代码中**未实现** β 参数或训练 loss 辅助目标。论文声称 β=0.1 有效，但 `_val` 变体中未包含。可能在 `_sft` 变体中实现。

## 1.4 实验结果

### 1.4.1 Pre-training（Pile, 17 domains）

| 方法 | Avg Score (14 benchmarks) | Avg Rank | Search Cost (GPU-hours) |
|------|--------------------------|----------|------------------------|
| Human Heuristic | 45.8 | 5.2 | 0 |
| DoReMi | 46.1 | 4.5 | ~50 |
| ODM | 45.5 | 5.8 | ~10 |
| IDEAL | 46.3 | 4.2 | ~30 |
| RegMix | 47.2 | 2.8 | **720.5** |
| CLIMB | 47.5 | 2.3 | **71.9** |
| **FastMix** | **48.2** | **1.0** | **1.3** |

**关键数字**：
- FastMix 比 RegMix 快 **550x**，比 CLIMB 快 **55x**
- 14 个 benchmark 中 **9 个第一**
- 只用 **1 个 proxy**（RegMix 512 个，CLIMB 64 个）

### 1.4.2 Post-training / SFT（Qwen2.5-Math-Instruct 7B）

| 方法 | Avg Score (4 benchmarks) | Search Cost (GPU-hours) |
|------|-------------------------|------------------------|
| Human Heuristic | 55.2 | 0 |
| DoReMi | 56.8 | ~100 |
| RegMix (64 proxies) | 58.1 | 115.9 |
| CLIMB (64 proxies) | 59.9 | 117.4 |
| **FastMix** | **65.4** | **2.2** |

**关键发现**：
- SFT 场景优势更大：+5.5 over CLIMB
- 数学优化的 mix 泛化到 coding (LiveCodeBench) 和 STEM (GPQA-Diamond)
- 但 SFT proxy 必须用 ~1B 模型（Qwen2.5-1.5B-Instruct），不能用 1M

### 1.4.3 论文自述的局限性（Section 4.3，"痛苦教训"）

**1. 不可微目标**：
- 离散指标（accuracy）不能直接优化
- 解决方案：用 SFT loss 作为可微代理
- **警告**：有限差分和 SPSA 在工业数据上**几乎不收敛**

**2. n₂ > 1 的问题**：
- 闭式梯度只在 n₂=1 时成立
- n₂ > 1 需要 BPTT（内存爆炸）或有限差分（不稳定）
- **建议**：尽量保持 n₂=1

**3. 正则化在工业数据上无效**：
- 学术数据上熵正则化有效
- 工业数据上**最稳健的方案是限制过采样比例**（如最多 3x）

**4. 小 proxy 模型的局限**：
- **< 0.5B 的 proxy 在工业数据上表现不稳定**
- 收敛不稳定 → 高噪声的配比结果
- 模型容量和架构差异导致对不同域的偏好不同
- **这与 QuaDMix 的核心问题直接相关**

**5. 搜索目标数据**：
- Pre-training 序列通常很长，SFT 数据显著更短
- 这种结构差异导致梯度**严重分歧**，FastMix 失败
- **解决方案**：拼接多个 SFT 序列以对齐 pre-training 长度

## 1.5 与 QuaDMix 的系统对比

### 1.5.1 架构对比

| 维度 | FastMix | QuaDMix |
|------|---------|---------|
| **Proxy 架构** | 2-layer LLaMA 256d (~14M) | 2-layer LLaMA 256d (~14M) |
| **Proxy 数量** | **1 个** | **~100-500 个** |
| **搜索空间** | K 维（K = 域数） | **198 维**（22 域 × 9 参数） |
| **优化方法** | 梯度下降（AdamW on α） | LightGBM surrogate + 随机搜索 |
| **信号来源** | 梯度内积（val vs train） | Proxy validation loss |
| **质量过滤** | 无 | 5 个 FastText 分类器 |
| **采样曲线** | 无（直接权重） | Sigmoid 参数化（λ, ω, η, ε） |
| **验证集** | Pile-cc val split | OpenHermes-10k / CAP v1 |
| **搜索成本** | **1.3 GPU-hours** | ~100+ GPU-hours |
| **代码** | 已开源 | 内部 |

### 1.5.2 核心差异分析

**1. 搜索空间维度**

FastMix 只优化 K 个域权重（Pile 上 K=17）。QuaDMix 优化 198 维空间（22 域 × (5 质量权重 + 4 采样参数)）。

**含义**：FastMix 假设质量过滤和采样策略是**预先确定**的，只优化域配比。QuaDMix 联合优化质量过滤 + 采样策略 + 域配比。

**2. 优化范式**

FastMix 是**在线**方法（配比在训练过程中动态调整）。QuaDMix 是**离线**方法（先搜索，再用搜索结果训练最终模型）。

**含义**：FastMix 的配比是训练过程中的"轨迹"，最终输出的是一个时间平均的配比。QuaDMix 输出的是一个静态配比。

**3. Proxy 使用方式**

FastMix 在**同一个 proxy 训练过程中**同时优化模型参数和配比。QuaDMix 训练**多个 proxy**，每个 proxy 用不同的配比，然后用 LightGBM 拟合配比→性能映射。

**含义**：FastMix 的 proxy 信号来自训练动态（梯度），QuaDMix 的 proxy 信号来自训练结果（validation loss）。

### 1.5.3 FastMix 对 QuaDMix 的启示

**高优先级**：

1. **梯度内积作为配比信号**：
   - QuaDMix 的 proxy 可以记录每个域的梯度范数和方差（PiKE 也建议了这一点）
   - 计算验证集梯度与每个域训练梯度的内积
   - 这比纯 validation loss 更稳定，因为它捕捉了**方向**而非仅**幅度**

2. **1 proxy 可行性验证**：
   - FastMix 证明 1 个 proxy 就够（在学术数据上）
   - 但 FastMix 自己承认 < 0.5B proxy 在工业数据上不稳定
   - QuaDMix 的 14M proxy 远小于 0.5B → **这个警告直接适用**

3. **可微目标的重要性**：
   - FastMix 的核心 trick 是让 α 可微
   - QuaDMix 的 LightGBM surrogate 本质是非参数回归，不需要可微
   - 但 FastMix 的梯度信号比 LightGBM 的 loss 预测**信息量更大**（包含方向信息）

**中优先级**：

4. **熵正则化**：论文声称有效但代码未启用 → 需要自己验证
5. **n₂=1 限制**：FastMix 的经验表明多步外层更新不可靠 → QuaDMix 的 LightGBM 方法天然避免了这个问题
6. **搜索目标对齐**：FastMix 发现 pre-training 和 SFT 序列长度差异导致梯度分歧 → QuaDMix 用 OpenHermes-10k（SFT 数据）作为验证集可能存在同样的问题

**低优先级**：

7. **SFT 场景泛化**：FastMix 在 SFT 上优势更大，但 QuaDMix 主要面向 pre-training

## 1.6 可信度与风险评估

### 1.6.1 优势

| 维度 | 评分 | 理由 |
|------|------|------|
| 机构 | ★★★★☆ | 腾讯混元（工业级 LLM 团队）+ 港大 |
| 会议 | ★★★★★ | ICLR 2026 接收 |
| 代码 | ★★★★☆ | 已开源，但仅 2 commits |
| 理论 | ★★★★☆ | 闭式梯度推导清晰，bilevel 优化有理论保证 |
| 实验 | ★★★★☆ | Pre-training + SFT，14 benchmarks，但最大 1B |

### 1.6.2 风险

| 风险 | 严重度 | 说明 |
|------|--------|------|
| 工业数据不稳定 | **高** | 论文自己承认 < 0.5B proxy 在工业数据上不可靠 |
| 仅 lm_head 梯度 | **中** | 代码只用 lm_head 梯度，论文未讨论此选择的影响 |
| 正则化未启用 | **中** | 论文声称 λ=1e-5，代码中 λ=0 |
| Pile 数据集 | **中** | 只在 Pile（17 域）上验证，域数远少于 QuaDMix 的 22 |
| 无大规模验证 | **中** | Pre-training 最大 1B，SFT 最大 7B |

### 1.6.3 综合评分

| 维度 | 评分 |
|------|------|
| 与 QuaDMix 相关性 | **★★★★★**（proxy 架构相同，解决同一问题） |
| 可借鉴性 | **★★★★☆**（核心思想可借鉴，但工业数据风险需注意） |
| 可信度 | **★★★★☆**（ICLR 接收 + 有代码 + 腾讯混元） |
| 实施难度 | **★★★☆☆**（需要改造 QuaDMix pipeline 为在线模式） |

---

# 二、Domain-Aware Scaling Laws Uncover Data Synergy

## 2.1 论文概览

| 维度 | 信息 |
|------|------|
| arXiv | 2607.11052 (2026.07.13，1 周前) |
| 机构 | **MIT CSAIL + Microsoft Research + Harvard** |
| 作者 | Kimia Hamidieh (MIT), Lester Mackey (MSR), David Alvarez-Melis (MSR/Harvard) |
| 方向 | 量化域间协同效应，改进 scaling law |
| 代码 | 无 |

### 作者可信度

| 作者 | 背景 |
|------|------|
| Kimia Hamidieh | MIT CSAIL，第一作者 |
| Lester Mackey | Microsoft Research，资深研究员，h-index ~30+ |
| David Alvarez-Melis | Microsoft Research + Harvard，OT/ML 方向 |

**可信度评估**: ★★★★☆ — Microsoft Research + MIT CSAIL 组合，理论功底强。但无代码、无会议接收（仅 1 周前提交），验证实验规模小（30M/150M）。

## 2.2 核心方法

### 2.2.1 问题定义：两种协同效应

**一阶协同（Domain→Benchmark）**：

γ_{j,k} 衡量域 k 的训练数据对 benchmark j 的贡献**超出总 token 数预期的部分**。

- γ > 0：协同（该域数据对 benchmark 有超比例贡献）
- γ < 0：干扰（该域数据对 benchmark 有负面影响）

**二阶协同（Domain×Domain）**：

σ_{kk'} 衡量两个域**共同出现**时产生的非加性效应。

- σ > 0：互补（两个域一起训练 > 各自训练之和）
- σ < 0：稀释（两个域一起训练 < 各自训练之和）

### 2.2.2 Scaling Law 形式

**Domain-agnostic baseline（Chinchilla）**：

```
L(N, D) = L_∞ + A·N^(-α) + B·D^(-β)
```

**一阶模型（FO）**：

```
L_j(N_i, d_i, u_i) = L_{∞,j} + A_j·N_i^(-α_j) + B_j·d_i^(-β_j) · Π_k (u_{i,k}·d_i)^(-γ_{j,k}·u_{i,k})
```

每个域 k 修改 benchmark j 的数据 scaling 指数：β_j → β_j + γ_{j,k}

**二阶模型（SO）**：

在 FO 基础上加入 pairwise 交互项：

```
Φ_{i,j} = b_j - [β_j·1 + γ_{j,·}]^T·z_i - β_j·H(u_i) - β_j·Σ_{k<k'} (γ_{j,k}+γ_{j,k'})·σ_{kk'}·softmin_τ(z̄_{i,k}, z̄_{i,k'})
```

其中 softmin 确保交互项在任一域缺失时消失。

### 2.2.3 有效 token 解释

二阶协同可以理解为产生"bonus tokens"：

```
log D^{eff}_{i,j} = log d_i + Σ_{k<k'} (γ_{j,k}+γ_{j,k'})·σ_{kk'}·s_{i,kk'}
```

当两个协同域共同出现时，有效数据量 D^{eff} > D（模型表现得像用了更多数据）。

## 2.3 实验细节

### 2.3.1 观测数据

| 模型组 | 规模范围 | 数量 | 预训练数据 |
|--------|---------|------|-----------|
| GPT-Neo/J/NeoX | 125M-20B | 5 | The Pile（~30% web, ~27% science, ~15% books） |
| Pythia | 70M-12B | 8 | The Pile |
| DataDecide | 150M-1B | 30 | Dolma variants（no-code, no-math-code, no-flan, no-reddit） |
| OLMo | 1B-7B | 2 | Dolma（69-97% web） |
| OpenLLaMA | 3B-13B | 5 | RedPajama（mostly web） |
| RedPajama-INCITE | 3B-7B | 2 | RedPajama |
| **总计** | **70M-20B** | **52** | |

### 2.3.2 8 域分类

| 域 | 预训练来源 |
|----|-----------|
| Books | Books3, Project Gutenberg, BookCorpus2 |
| Code | GitHub (Pile), The Stack, StarCoder |
| Encyclopedia | Wikipedia, Wikibooks |
| Legal | FreeLaw, USPTO Backgrounds |
| Math | DM Mathematics, OpenWebMath, Algebraic Stack |
| Q&A | StackExchange |
| Science | ArXiv, PubMed Central, PubMed Abstracts, PhilPapers, NIH ExPorter, peS2o |
| Web | Pile-CC, OpenWebText2, CommonCrawl, C4, RefinedWeb |

### 2.3.3 拟合方法

- **两阶段拟合**：先拟合 per-task Chinchilla baseline，再拟合完整协同模型
- **β prior**：以 Chinchilla 值 β̂_j^{chin} 为中心的先验，保持数据 scaling 指数正
- **Huber loss**（δ=0.5）+ L1/L2 正则化 → 稀疏 γ
- **L-BFGS** 优化
- **5-fold CV** 评估
- **B=50 bootstrap** 不确定性量化（80% 子采样 + 有放回抽样）

## 2.4 核心结果

### 2.4.1 一阶协同 γ（Domain→Benchmark）

**Top 正协同（synergy）**：

| Domain → Benchmark | γ | 含义 |
|-------------------|---|------|
| Math → HumanEval | **+1.34** | Math 数据对代码 benchmark 贡献超比例 |
| Math → MBPP | **+1.44** | 同上 |
| Code → IFEval | **+0.88** | Code 数据对指令遵循有超比例贡献 |
| Code → MBPP | **+0.59** | Code 数据对自身 benchmark 的增强 |
| Code → HumanEval | **+0.47** | 同上 |

**Top 负协同（interference）**：

| Domain → Benchmark | γ | 含义 |
|-------------------|---|------|
| Books → HumanEval | **-0.90** | Books 数据严重干扰代码能力 |
| Books → GSM8K | **-0.67** | Books 数据干扰数学推理 |
| Encyclopedia → HellaSwag | **-0.76** | Encyclopedia 干扰常识推理 |
| Encyclopedia → HumanEval | **-0.69** | 干扰代码能力 |
| Science → ARC-Challenge | **-0.54** | Science 干扰科学问答 |
| Web → HumanEval | **-0.45** | Web 数据干扰代码能力 |

**关键发现**：
- HumanEval 的数据 scaling 指数 β 从 0.45（完整 dolma1.7）降到 0.18（去掉 code）
- 这说明**组成决定了 scaling 的斜率**，而不仅仅是截距

### 2.4.2 二阶协同 σ（Domain×Domain）

**Top 正协同**：

| Domain Pair | σ | 含义 |
|-------------|---|------|
| Code × Science | **+2.55** | Code 和 Science 共同出现时效果 > 各自之和 |
| Code × Math | **+2.40** | Code 和 Math 互补最强 |
| Q&A × Web | **+2.39** | Q&A 和 Web 互补 |

**Top 负协同**：

| Domain Pair | σ | 含义 |
|-------------|---|------|
| Math × Books | **负值** | Math 和 Books 共同出现时互相稀释 |

**关键发现**：
- Code 是"超级协作者"：与 Math、Science 都有强正协同
- Books 是"干扰源"：与 Math 有负协同，对多个 benchmark 有一阶负协同
- **协同 ≠ 重叠**：Code 和 Science 的 embedding centroid cosine = 0.31（低重叠），但协同强（σ=+2.55）

### 2.4.3 R² 对比

| 模型 | 中位 R² | 平均 R² | 最差 R² |
|------|---------|---------|---------|
| **SO (本文)** | **0.912** | **0.903** | **0.866** |
| FO (本文) | 0.906 | 0.906 | 0.874 |
| Data Mixing Laws (扩展) | 0.910 | 0.871 | 0.554 |
| AutoScale (扩展) | 0.891 | 0.852 | 0.598 |
| Scaling Laws for Opt. Data Mix. | 0.847 | 0.791 | 0.395 |
| RegMix (扩展) | 0.790 | 0.747 | 0.511 |
| **Chinchilla baseline** | **远低于所有** | | |

**关键**：
- HumanEval 上 Chinchilla R²=0.41 → FO R²=**0.92**（+0.51）
- MBPP 上 Chinchilla R²=0.20 → FO R²=**0.88**（+0.68）
- 代码 benchmark 上 domain-agnostic law 几乎无效

### 2.4.4 验证实验

**30M 参数，3B tokens**：

| Task | Balanced | Optimal | Anti-optimal | Gap |
|------|----------|---------|-------------|-----|
| HumanEval | 1.268 | 1.110 (-12.4%) | 1.503 (+18.6%) | **0.393** |
| GSM8K | 1.818 | 1.779 (-2.1%) | 1.935 (+6.5%) | **0.156** |
| IFEval | 1.451 | 1.513 (+4.3%) | 1.551 (+6.9%) | **0.038** |

**150M 参数，5B tokens**：

| Task | Balanced | Optimal | Anti-optimal | Gap |
|------|----------|---------|-------------|-----|
| HumanEval | 0.868 | 0.721 (-16.9%) | 1.058 (+21.9%) | **0.337** |
| GSM8K | 1.643 | 1.402 (-14.7%) | 1.591 (-3.2%) | **0.189** |
| IFEval | 1.366 | 1.324 (-3.0%) | 1.365 (-0.0%) | **0.041** |

**关键**：
- 方向性预测全部正确（optimal > balanced > anti-optimal）
- HumanEval 上差距最大（0.337 BPB at 150M）
- GSM8K 在 150M 上差距显著增大（2.1% → 14.7%）
- IFEval 最难预测（gap 小且方向不稳定）

## 2.5 稳健性检验

### 2.5.1 组成结构

- 8 个域但**只有 2 个有效 PCA 方向**（95% 方差解释）
- 第一主成分：code 比例
- 第二主成分：books/science/legal 相关

### 2.5.2 可辨识性

- log D 和 log N 相关（r=0.62），α 和 β 不可辨识
- **但 γ 与 log N 几乎正交**（R²<0.09），所以 γ 是可辨识的
- γ 的稳定性远高于 α 和 β

### 2.5.3 跨规模稳定性

- 在 >1B 子集上重新拟合 FO
- 13 个最大显著 γ 值**全部保持符号**
- 9/13 保持 90% CI 排除零

### 2.5.4 协同 ≠ 语义重叠

- 用 all-MiniLM-L6-v2 嵌入 5000 文档/域
- Code×Science centroid cosine = 0.31（低重叠），但 σ=+2.55（强协同）
- Web×Books centroid cosine = 0.50（高重叠），但无显著协同
- **结论**：协同来自功能互补，不是语义相似

## 2.6 局限性（作者承认）

1. **观测设计**：γ 和 σ 是**关联**而非因果
2. **低维混合空间**：8 域但只有 2 个有效方向
3. **小规模验证**：只在 30M/150M 上验证
4. **仅 pre-training**：SFT/RL 场景未验证
5. **域定义敏感**：不同域划分会得到不同协同
6. **scale-independent γ**：假设混合偏好不随规模变化

## 2.7 与 QuaDMix 的系统对比

### 2.7.1 方法对比

| 维度 | Domain-Aware Scaling Laws | QuaDMix |
|------|--------------------------|---------|
| **目标** | 理解域间协同 | 优化数据配比 |
| **方法** | 观测拟合 scaling law | Proxy + LightGBM + 搜索 |
| **域数** | 8 | 22 |
| **数据来源** | 52 个公开模型 | 自建 proxy 实验 |
| **验证规模** | 30M/150M | 14M proxy → 7B+ target |
| **输出** | γ 矩阵 + σ 矩阵 | 最优参数 θ* |
| **可操作性** | 方向性指导 | 直接可用的配比 |

### 2.7.2 对 QuaDMix 的启示

**高优先级**：

1. **验证集选择**（来自 Google MDE 的交叉验证）：
   - 一阶协同 γ 表明不同域对不同 benchmark 的贡献差异巨大
   - QuaDMix 用 OpenHermes-10k（通用 SFT）作为验证集 → 可能无法捕捉特定域的协同效应
   - **建议**：用多域验证集（类似 CAP v1 的 5 个 capability cluster），或直接用 end-task validation

2. **域间交互建模**：
   - QuaDMix 的 LightGBM surrogate 隐式捕捉了域间交互（树模型自动学习特征交叉）
   - 但 198 维搜索空间中，域间交互是隐式的、不可解释的
   - **建议**：在 LightGBM 特征中显式加入 pairwise 域比例特征（u_k × u_{k'}），让 surrogate 更容易学习二阶协同

3. **Code × Math 协同的实操意义**：
   - σ_{code,math} = +2.40 是最强的二阶正协同之一
   - QuaDMix 的 22 域中，Computers (00) 和 Mathematics (51) 对应 Code 和 Math
   - **建议**：确保这两个域在最终配比中同时保持足够比例

**中优先级**：

4. **Books 干扰的警示**：
   - γ_{Books→HumanEval} = -0.90，γ_{Books→GSM8K} = -0.67
   - σ_{Math×Books} < 0
   - QuaDMix 的 Books_and_Literature (80-89) 域可能需要控制比例上限

5. **有效混合空间的低维性**：
   - 论文发现 8 域只有 2 个有效 PCA 方向
   - QuaDMix 的 22 域可能也存在类似的低维结构
   - **建议**：对 QuaDMix 的 proxy 实验数据做 PCA，检查有效维度

**低优先级**：

6. **Scaling law 作为 surrogate 的替代**：
   - 如果 QuaDMix 的 LightGBM surrogate R² 不够高，可以考虑用 domain-aware scaling law 作为参数化 surrogate
   - 但需要先在 QuaDMix 的 proxy 数据上验证拟合质量

## 2.8 可信度与风险评估

### 2.8.1 优势

| 维度 | 评分 | 理由 |
|------|------|------|
| 机构 | ★★★★★ | MIT CSAIL + Microsoft Research + Harvard |
| 理论 | ★★★★★ | 数学推导严谨，scaling law 形式优美 |
| 实验设计 | ★★★★☆ | 52 个模型，5-fold CV，50 bootstrap |
| 可解释性 | ★★★★★ | γ 和 σ 直接可解释 |

### 2.8.2 风险

| 风险 | 严重度 | 说明 |
|------|--------|------|
| 观测 ≠ 因果 | **高** | γ 和 σ 是关联，不是因果效应 |
| 低维混合空间 | **高** | 8 域只有 2 个有效方向，泛化性存疑 |
| 小规模验证 | **中** | 只在 30M/150M 上验证，7B+ 未验证 |
| 无代码 | **中** | 无法复现和扩展 |
| 域定义粗 | **中** | 8 域远粗于 QuaDMix 的 22 域 |

### 2.8.3 综合评分

| 维度 | 评分 |
|------|------|
| 与 QuaDMix 相关性 | **★★★★☆**（提供域间协同的定量理解） |
| 可借鉴性 | **★★★☆☆**（方向性指导为主，非直接可用方法） |
| 可信度 | **★★★★☆**（MSR + MIT，但无代码无会议） |
| 实施难度 | **★★☆☆☆**（需要在 QuaDMix 数据上重新拟合） |

---

# 三、两篇论文的交叉分析

## 3.1 方法谱系定位

```
                    离线搜索 ←————————————→ 在线优化
                         |                      |
    参数化模型     RegMix → QuaDMix → MergeMix    FastMix → PiKE
                         |                      |
    非参数模型     Data Mixing Laws              ADAPT → DoReMi
                         |                      |
    观测拟合       Domain-Aware SL              (无)
```

FastMix 和 QuaDMix 在谱系上处于**不同端**：
- QuaDMix 是离线搜索（先搜索后训练）
- FastMix 是在线优化（边训练边搜索）

Domain-Aware Scaling Laws 不在谱系中（它是理解工具，不是优化方法）。

## 3.2 Proxy 可靠性视角

| 方法 | Proxy 策略 | 风险 |
|------|-----------|------|
| QuaDMix | 14M off-policy, 多个 | 高（proxy→target 迁移无保证） |
| FastMix | 14M off-policy, 1 个 | 高（论文自己承认 < 0.5B 不稳定） |
| Domain-Aware SL | 无 proxy（观测拟合） | 中（关联 ≠ 因果） |
| OP-Mix | LoRA on-policy | 低 |
| PiKE | 无 proxy（在线梯度） | 低 |

**关键发现**：FastMix 和 QuaDMix 使用**完全相同的 proxy 架构**（2-layer LLaMA 256d），但 FastMix 只训 1 个而 QuaDMix 训多个。FastMix 论文自己承认这个规模的 proxy 在工业数据上不稳定——这**直接质疑了两者的共同基础**。

## 3.3 对 QuaDMix 的综合行动建议

### P0：立即可做（1-2 周）

| # | 行动 | 来源 | 预期收益 | 成本 |
|---|------|------|---------|------|
| 1 | 在 proxy 训练中记录每个域的梯度范数和方差 | FastMix + PiKE | 新的配比信号维度 | 低 |
| 2 | 在 LightGBM 特征中加入 pairwise 域比例特征 | Domain-Aware SL | 显式捕捉二阶协同 | 低 |
| 3 | 检查 CAP v1 验证集的域覆盖度 | Domain-Aware SL + Google MDE | 确保验证集不偏向特定域 | 低 |

### P1：短期可做（2-4 周）

| # | 行动 | 来源 | 预期收益 | 成本 |
|---|------|------|---------|------|
| 4 | 实现梯度内积作为辅助配比信号 | FastMix | 更稳定的配比信号 | 中 |
| 5 | 对 proxy 实验数据做 PCA，检查有效维度 | Domain-Aware SL | 理解搜索空间的真实维度 | 低 |
| 6 | 在 QuaDMix proxy 数据上拟合 domain-aware scaling law | Domain-Aware SL | 验证 γ/σ 是否与 LightGBM 学到的交互一致 | 中 |

### P2：中期探索（1-2 月）

| # | 行动 | 来源 | 预期收益 | 成本 |
|---|------|------|---------|------|
| 7 | 实现 FastMix 风格的在线优化作为 QuaDMix 的替代模式 | FastMix | 550x 搜索加速 | 高 |
| 8 | 用 domain-aware scaling law 作为参数化 surrogate（替代 LightGBM） | Domain-Aware SL | 可解释的 surrogate | 高 |

### 关键警示

1. **FastMix 的工业数据警告**：论文自己承认 < 0.5B proxy 在工业数据上不稳定。QuaDMix 的 14M proxy 远小于 0.5B。这不是说 QuaDMix 一定不行，而是说**需要严格验证 proxy 信号的可靠性**。

2. **Domain-Aware SL 的观测局限**：γ 和 σ 是从 52 个公开模型拟合的关联，不是因果效应。在 QuaDMix 的 22 域上，协同模式可能完全不同。

3. **两者的共同盲区**：都没有在 >1B 的 pre-training 上验证过。QuaDMix 的 7B+ target 仍然是一个未验证的跳跃。

---

# 四、与已有论文的关联

## 4.1 引用关系

- FastMix 引用了 **RegMix**（实验设置）、**CLIMB**（主要 baseline）、**DoReMi**、**IDEAL**、**ODM**
- FastMix **未引用** QuaDMix、PiKE、OP-Mix、Chameleon
- Domain-Aware SL 引用了 **RegMix**、**Data Mixing Laws**、**AutoScale**、**Scaling Laws for Opt. Data Mix.**
- Domain-Aware SL **未引用** FastMix、QuaDMix、CLIMB

## 4.2 方法演进

```
Chinchilla (domain-agnostic)
    ↓
Data Mixing Laws (per-domain power law, no scale)
    ↓
RegMix (nonparametric regression, no scale)
    ↓
Domain-Aware SL (composition in scaling exponent, observational)
    ↓
FastMix (bilevel optimization, differentiable, single proxy)
```

## 4.3 Proxy 可靠性谱系更新

```
Off-policy (风险高)                    On-policy (风险低)
←————————————————————————————————————————→
QuaDMix(14M) = FastMix(14M)    Phase Transitions    CAMEL    OP-Mix(LoRA)    PiKE(无proxy)
```

FastMix 和 QuaDMix 在 proxy 可靠性谱系上处于**完全相同的位置**。

---

*分析完成。两篇论文的 PDF 已存储在 `papers/` 目录。*
