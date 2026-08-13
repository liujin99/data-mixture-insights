# ADAPT 深度分析

**论文**: Rethinking Data Curation in LLM Training: Online Reweighting Offers Better Generalization than Offline Methods
**arXiv**: 2605.05227v1 (2026.04.19)
**作者**: Wanru Zhao (Cambridge), Yihong Chen (Oxford OATML), et al.
**Venue**: ICLR 2026
**代码**: https://github.com/Ryan0v0/ADAPT
**模型**: https://huggingface.co/DATA-ADAPT

---

## 一、核心论点

### 1.1 离线方法的致命缺陷

**离线数据选择（LESS, DSIR, QuaRating）的三大问题**：
1. **忽视训练动态**：样本价值不是静态的，随模型学习而变化。离线选择在 proxy 上冻结价值，与训练模型的演化需求脱节。
2. **牺牲多样性**：硬过滤丢弃数据分布，用记忆替代泛化。
3. **过拟合验证任务**：LESS 用 MMLU 做验证集选数据 → MMLU 50.2 分但 BBH 只有 38.7 分（OOD 严重退化）。

**对比**：
| 方法 | in-domain (MMLU) | OOD (BBH) | OOD 退化 |
|------|-----------------|-----------|---------|
| LESS | 50.2 | 38.7 | -11.5 |
| Full Dataset SFT | 49.7 | 44.4 | -5.3 |
| ADAPT | 50.7 | 44.8 | -5.9 |

**关键洞察**：全量数据 SFT 的 OOD 泛化比离线选择更好——离线选择用记忆替代了泛化。

### 1.2 统一形式化

论文将数据策展统一为**数据重加权**的特例：
- 数据选择 = 二值权重 (0/1)
- 数据混合 = 域级分数权重
- 数据重加权 = 样本级分数权重

**FLOPs 公平比较**：
- 离线：F_prep（评分）+ F_train（子集训练）
- 在线：F_train（全集训练）+ F_metrics（即时计算，极小）

---

## 二、ADAPT 方法

### 2.1 Per-Sample Learning Rate

```
θ_{t+1} = θ_t - η * w_t(i) * ∇l(θ_t; x_i)
```

w_t(i) 充当 per-sample learning rate 倍率：大权重放大有效步长，小权重抑制无用样本。

### 2.2 相似性评分

**ADAPT（核心方法）**：使用模型自身的 hidden representation

```
s_ADAPT(x) = (1/|D_val|) * Σ_{v∈D_val} cos(φ(x), φ(v))
```

其中 φ(x) = position-weighted mean pooling of last-layer hidden states：
```
φ(x) = Σ_i w_i * h_i,  w_i = i / Σ_j j
```

后位置的 token 权重更高（补偿 causal mask 偏差）。

**权重转换**：sigmoid + 温度
```
w_t(i) = σ(s_ADAPT(x_i) / max(τ, ε)) = 1 / (1 + exp(-s/τ))
```

**关键设计**：sigmoid（非 softmax）→ 绝对权重，不依赖 batch 内其他样本。

### 2.3 在线验证集 embedding 更新

每 R 步刷新验证集 embedding → 跟踪模型表示空间的演化。

### 2.4 为什么 embedding > gradient

- 梯度在训练早期波动大，计算昂贵
- embedding 提供更平滑、更稳定的信号
- embedding 随模型表示空间一致演化

---

## 三、实验结果

### 3.1 指令调优（Llama-2-7B + LoRA）

| 方法 | MMLU(in-domain) | BBH(OOD) |
|------|-----------------|----------|
| LESS | 50.2 | 38.7 |
| BM25(offline) | 48.7 | 42.3 |
| Full SFT | 49.7 | 44.4 |
| ADAPT-BM25 | 50.9 | 43.7 |
| ADAPT | 50.7 | 44.8 |

- ADAPT in-domain 与 LESS 相当
- ADAPT OOD 比 LESS 高 +6.1 分
- ADAPT-BM25 > 离线 BM25（同信号，在线 > 离线）

### 3.2 预训练（TinyLlama-120M, 50B tokens, SlimPajama）

比 Uniform/DoReMi/RegMix 在 15 个 benchmark 上：
- **7.2% 更高的 per-FLOP 准确率**
- **6.1% 更强的 OOD 泛化**

---

## 四、与 QuaDMix 的关联分析

### 4.1 直接对比

| 维度 | QuaDMix | ADAPT |
|------|---------|-------|
| 范式 | 离线（proxy → 回归 → 搜索） | 在线（训练中动态调整） |
| 信号 | 1M proxy val_loss | 模型自身 hidden × 验证集 hidden cosine |
| 数据处理 | 硬过滤/重采样（改变数据量） | 重加权（保留全部数据） |
| 多样性 | 牺牲（丢弃"低质量"数据） | 保留（只调 learning rate） |
| OOD 泛化 | 风险高（离线过拟合） | 强（全量数据 + 动态权重） |
| proxy 依赖 | 是（1M 模型） | 否（用目标模型自身） |
| 计算开销 | 高（336 实验 + LightGBM） | 极低（cosine 计算） |

### 4.2 ADAPT 揭示的 QuaDMix 风险

1. **OOD 泛化风险**：QuaDMix 的 quality-top-k 是离线硬过滤 → 可能像 LESS 一样过拟合到验证任务
2. **多样性损失**：丢弃"低质量"数据可能丢弃了泛化所需的多样性
3. **静态信号**：proxy 在训练前计算一次，但样本价值随训练变化

### 4.3 可操作建议

**立即可做**：
1. **评估 QuaDMix 的 OOD 泛化**：在 QuaDMix 选出的配比上训练，检查 in-domain 和 OOD benchmark 差距。如果差距大（如 >5%），说明离线过拟合问题严重。

2. **在 nanochat 中实现 ADAPT 式在线重加权**：
   - 收集 STEM 验证集（已有 6 tasks, 16455 docs）
   - 训练中每 R 步计算训练样本的 hidden 表示
   - 计算 cos(φ(train_sample), φ(val_sample))
   - 用 sigmoid 转换为 per-sample weight
   - 乘到梯度上

3. **对比 ADAPT vs quality-top-k**：同一训练预算下，ADAPT（全量+重加权）vs quality-top-k（子集），比较 in-domain 和 OOD 性能。

**短期可做**：
4. **用 ADAPT 替代 QuaDMix 做在线配比调整**：不需要 proxy 模型，不需要 LightGBM 回归，不需要 336 实验。直接在目标训练中动态调整。

### 4.4 关键限制

1. **预训练实验只在 TinyLlama-120M**：极小规模，外推到大模型未验证
2. **验证集选择**：ADAPT 的 anchor set 从 evaluation benchmarks 采样 → 可能是循环论证
3. **sigmoid 温度 τ**：超参数，默认 1.0，但最优值可能随任务变化
4. **embedding 刷新开销**：每 R 步需要前向传播验证集
5. **SFT vs pretraining**：ADAPT 在 SFT 上效果更显著，pretraining 上提升较小

---

## 五、可信度评估

### 强信号
- **ICLR 2026** 接收
- Cambridge + Oxford OATML + Samsung AI Center
- **代码+模型开源**
- 统一形式化框架清晰
- 与 LESS/DoReMi/RegMix 全面对比
- OOD 泛化实验设计合理

### 弱信号
- 预训练只在 120M 模型验证（极小）
- 15 个 benchmark 提升幅度不大（多数 <1%）
- 验证集从 evaluation benchmarks 采样（循环论证风险）
- SFT 结果更显著，但用户做 mid-training

### 判断
- **理论贡献高**：统一形式化 + 在线 vs 离线的系统对比
- **实用性中等**：方法简单可复现，但大规模验证不足
- **对 QuaDMix 的警示价值高**：揭示了离线方法的 OOD 泛化风险
- **整体可信度：中高**（ICLR + 开源 + 理论好，但规模小）

---

## 六、对破局路线的影响

### Route A（预训练小模型 + benchmark）
- ADAPT 的在线 reweighting 可以在 Route A 的训练中直接使用
- 不需要 proxy 模型，用训练模型自身的 hidden 表示

### Route B（Perplexity Correlations）
- ADAPT 用 cosine 相似性替代 loss 相关性
- 可以看作 Route B 的在线版本

### Route C（Nexus 优化器）
- ADAPT 调 per-sample learning rate，Nexus 调 optimizer trajectory
- 两者正交，可以叠加

**结论**：ADAPT 提供了一个绕过 QuaDMix proxy 瓶颈的完整替代方案——不需要 proxy、不需要回归、不需要搜索。直接在训练中用模型自身的表示空间做在线重加权。但预训练验证规模太小，需要自行验证。

---

## 七、总结

ADAPT 的核心价值：

1. **揭示离线方法的 OOD 泛化缺陷**：LESS 在 OOD 上 -11.5 分，这是对 QuaDMix 的直接警示
2. **在线重加权 > 离线选择**：保留全部数据 + 动态权重 > 硬过滤子集
3. **模型自身 hidden 表示是好信号**：不需要 proxy，不需要 gradient
4. **统一形式化**：选择/混合/重加权都是重加权的特例

**对 QuaDMix 的最高价值**：
1. 检查 QuaDMix 选出配比的 OOD 泛化（是否像 LESS 一样过拟合）
2. 在 nanochat 中实现 ADAPT 式在线重加权（~50 行代码，开源参考）
3. 如果 OOD 泛化差，考虑从离线 quality-top-k 切换到在线 reweighting

---

## 八、与 Velocitune / DiReCT / Data Mixing Agent 的对比

### 四篇联合对比表

| 维度 | ADAPT | Velocitune | DiReCT | Data Mixing Agent |
|------|-------|-----------|--------|-------------------|
| **Venue** | ICLR 2026 | ACL 2025 | ICML 2026 | ACL 2026 |
| **机构** | Cambridge+Oxford | Microsoft | MSRA+HKUST(GZ) | Microsoft |
| **阶段** | SFT + 预训练 | Continual pre-training | Annealing | Continual pre-training |
| **粒度** | 样本级 | **域级** | 样本级 | **域级** |
| **信号** | hidden 表示相似性 | 学习速度 (loss 轨迹) | Hessian 谱几何 | benchmark 分数 (RL reward) |
| **Proxy** | 不需要 | 目标模型自身 | 不需要 | 50M proxy (一次性 384 轨迹) |
| **动态** | 在线 | 在线 | 离线预计算 | 在线 |
| **代码** | ✅ GitHub | ❌ 无代码 | ✅ GitHub | ❌ 无代码 |
| **模型规模** | 120M/7B | 7B-8B | 355M-1.1B | 3B |
| **GitHub Stars** | 3 | — | 2 | — |
| **引用数** | 未获取 | 11 (3 有影响力) | 0 (太新) | 未获取 |
| **头部厂商引用** | ❌ 无 | ❌ 无 | ❌ 无 | ❌ 无 |

### 用户场景适用性排序

| 排名 | 方法 | 适用性 | 关键优势 | 关键限制 |
|------|------|--------|---------|---------|
| 1 | **Data Mixing Agent** | ★★★★ | 域级 + benchmark reward + 跨模型泛化 | 1996 GPU 小时成本 + 无代码 |
| 2 | **Velocitune** | ★★★☆ | 域级 + 不用 1M proxy + 在线 | scaling law 不透明 + 无代码 |
| 3 | **ADAPT** | ★★★☆ | 开源 + 在线 + OOD 泛化 | 样本级非域级 + 120M 规模 |
| 4 | **DiReCT** | ★★☆☆ | 开源 + 理论好 | annealing only + 100B 不可行 |

### 组合使用

- **Data Mixing Agent** 做域级权重（主训练阶段）
- **ADAPT** 做域内样本级权重（叠加）
- **DiReCT** 做冷却阶段样本选择（如果冷却数据是小子集）

三者正交，理论上可叠加。
