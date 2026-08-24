# LogFloor 深度分析：Proxy 动态瓶颈顺序的跨尺度迁移

> arXiv:2608.14936 | Small Models Scout Bottleneck Order for Large-Model Data Control
> 作者: Seungmin Choi, Jiwon Sung, Muhammad Umer, Abhiram Rao Gorle, Guijin Son, Youngjae Yu, John M. Cioffi
> 状态: Submitted to AAAI 2027（2026-08-14 提交）
> 分类: cs.AI

---

## 一、核心问题与方法

### 1.1 问题定位：first-passage skill training

LogFloor 不优化"固定预算下的平均分数"，而优化"所有技能到达 floor 的最少 token 数"：

$$\text{minimize } T \text{ s.t. } s_{T,k} \geq \tau_k \text{ for all } k$$

其中 $s_{r,k} = \ell_{0,k} / \ell_{r,k}$ 是技能 $k$ 在第 $r$ 轮的 **log-loss 减少倍数**（不是 accuracy），$\tau_k$ 是目标 floor。

**关键设计决策：用 loss 不用 accuracy。** 这是 LogFloor 绕过 phase transition 问题的核心策略——即使 70M 模型在 GSM8K 上 accuracy=随机水平，loss 仍会因为训练而下降。

### 1.2 LogFloor 控制器

每轮计算各技能的 deficit $d_{r,k} = [\tau_k - s_{r,k}]^+$，用 softmax 分配下一轮配比：

$$\tilde{w}_{r,k} = \frac{q_k \exp(d_{r,k}/\eta)}{\sum_j q_j \exp(d_{r,j}/\eta)}$$

$$w_r \leftarrow (1-\alpha) w_{r-1} + \alpha \cdot \text{Bound}(\tilde{w}_r)$$

- $\eta$ 控制瓶颈聚焦程度（越小越激进）
- $\alpha$ 控制配比变化速度
- Bound 施加上下界约束

输出：有序配比轨迹 $\gamma = \{w_0, w_1, \ldots, w_{T-1}\}$（瓶颈解决顺序）

---

## 二、关键实验结果

### 2.1 同尺度效果（Qwen2.5-1.5B）

5 个 bAbI 技能切片，6 个 floor 级别（2.8~3.3），每级别 5 种子：

| 方法 | 池化节省 | 命中率 |
|------|---------|--------|
| LogFloor | 56.2% | 30/30 |
| Greedy LogFloor | 51.1% | 30/30 |
| Skill-It | -7.8% | — |
| ODM | -0.9% | — |

### 2.2 跨尺度迁移（70M→12B）

核心实验：70M 模型产生瓶颈顺序，在 12B 上 replay。

| 方法 | 命中 | 胜/平/负 | 配对均值节省 | 池化节省 | 含 scout 成本 |
|------|------|---------|------------|---------|-------------|
| 12B uniform | 8/8 | — | 0.0% | 0.0% | 0.0% |
| 12B online LogFloor | 8/8 | 8/8 | 34.4% | 39.4% | 39.4% |
| **70M target-probe replay** | **8/8** | **7/8** | **30.9%** | **39.4%** | **37.6%** |
| Probe + frozen target mix | 7/8 | — | — | — | — |
| 70M phase replay (oracle) | 8/8 | 7/8 | 30.1% | 35.1% | 33.4% |

**关键发现**：70M scout 轨迹 replay 在 12B 上几乎和 12B 在线控制一样好（39.4% 池化节省一致），且即使计入 scout 成本仍省 37.6%。

### 2.3 顺序消解实验（最核心证据）

MMLU-control 5 个学术域（数学/科学/历史/法律社科/人文），冻结 70M seed 2 scout path 在 8 个 12B target 上 replay：

| 调度方式 | 保留信号 | 命中 | 配对均值 | 池化轮次节省 |
|---------|---------|------|---------|------------|
| **正确顺序 replay** | 完整轨迹 | 8/8 | 46.8% | **60.3%** |
| 静态塌缩 | marginal mixture only | 8/8 | -2.9% | 10.3% |
| 逆转顺序 | 同配比逆转 | 8/8 | -13.6% | 6.0% |
| 随机打乱 | 同配比 shuffled | 6/8 | — | — |
| 瓶颈标签（正确顺序） | 仅瓶颈 ID 序列 | 8/8 | 27.9% | 25.0% |
| 瓶颈标签逆转 | 同标签逆转 | 8/8 | -66.7% | -11.2% |

**结论**：顺序携带了大部分迁移收益。静态配比塌缩从 60.3% 降到 10.3%，逆转降到 6.0%——"顺序 > 配比"。

### 2.4 共享瓶颈几何

将 70M 和 12B 轨迹按归一化进度 $\phi \in [0,1]$ 对齐，比较 deficit 向量：

| 对比 | deficit 余弦 | top-2 瓶颈重叠 |
|------|------------|--------------|
| 同种子 (i→i) | 0.807 | 0.930 |
| 跨种子 (i→j) | 0.814 | 0.935 |

跨种子甚至比同种子更相似——瓶颈模式不是某个种子的偶然，而是结构性的。

---

## 三、跨尺度迁移的论证结构

LogFloor 声称 70M 的瓶颈顺序可迁移到 12B。论证有三层，但没有一层正面回答"容量不同为什么顺序一样"。

### 第一层：实证——"确实一样"

观察 70M 和 12B 对齐后的 deficit 余弦 0.80-0.81。这是观察，不是解释。

### 第二层：几何——"deficit 相似则配比相似"

数学上界：

$$\|\tilde{w}_S - \tilde{w}_L\|_1 \leq \frac{1}{\eta} \|d_S - d_L\|_\infty$$

softmax 对 deficit 变化的敏感度有界。如果 deficit 相似，最优配比就相似。但这是局部连续性论证——它说明"如果 deficit 恰好相似"，不解释"为什么相似"。

### 第三层：隐含假设——"难度层级跨尺度不变"

论文未明说，但整个论证依赖：

> **技能间的相对难度层级在不同模型尺寸间保持不变。**

数学比代码难在 70M 上成立，在 12B 上也成立。模型大了所有技能都学得更快，但**谁更难的排序不变**。

这在 bAbI/MMLU 上成立，因为这些技能：
1. 在 70M 就可学（loss 能下降）
2. 难度差异由数据分布决定，不由模型容量决定
3. **没有相变**

---

## 四、与 Phase Transition 的关系——LogFloor 没有解决的问题

### 4.1 LogFloor 如何"绕过" phase transition

| 策略 | 效果 |
|------|------|
| 用 loss 不用 accuracy | 70M 即使 accuracy=随机，loss 仍下降，所以 floor 可达 |
| 用 bAbI/MMLU 任务 | 简单任务无相变，难度层级线性保持 |
| 不声称预测能力涌现 | 只声称预测瓶颈顺序（loss 下降的难易排序） |

### 4.2 为什么这个绕行策略在 STEM 场景可能失败

| 条件 | bAbI/MMLU（LogFloor 验证的） | STEM accuracy（我们关心的） |
|------|------------------------------|---------------------------|
| 70M 能学吗？ | 能（loss 下降） | loss 可能下降，但 accuracy=随机 |
| 有相变吗？ | 无（简单任务） | 有（GSM8K 在 ~1.5B 涌现） |
| 难度排序跨尺度不变？ | 成立（无相变 → 线性进步） | **可能不成立**（相变前后难度重排） |

**核心矛盾**：如果某个技能有相变，它在小模型上是"慢速进步型"（loss 线性下降，看起来是瓶颈），但在大模型上可能突然跃迁（loss 快速下降，不再是瓶颈）。**难度排序在相变点前后会重排**，破坏尺度不变性。

### 4.3 LogFloor 自己知道的局限

论文引用了 Gu et al. 2026（数据配比引发相变，NeurIPS'25），说明作者知道相变问题。论文 Limitations 节明确承认：

> "Replay across architectures is not guaranteed. For instance, a replay from Qwen to Mamba transfers well, whereas a replay from Mamba to Qwen gives mixed results."
> 
> "The controller also treats monitored skills separately and does not explicitly model transfer or interference among them."

### 4.4 loss 顺序 ≠ accuracy 涌现顺序——未验证的桥

LogFloor 传递的是 **log-loss 下降的瓶颈顺序**。我们关心的是 **accuracy 涌现的顺序**。这两者是否一致？

- 简单任务（bAbI）：loss 下降和 accuracy 提升高度相关 → 一致
- 有相变的任务（STEM）：loss 下降可能领先 accuracy 涌现很久 → 不一定一致

**这个桥论文没有搭，恰好是我们 QuaDMix 失败时撞到的同一面墙。**

---

## 五、与 QuaDMix 失败的关系

### 5.1 一致的发现：静态配比不可迁移

QuaDMix 失败：14M proxy 搜出的静态最优配比传递到 1.5B 时失真（R²=0.30-0.84 但信号不传递到 accuracy）。

LogFloor 的消解实验独立验证了同样的发现：

> "Collapsing a path to its static marginal mixture removes most benefits"（60.3% → 10.3%）

两篇论文从完全不同的角度得出同一结论：**proxy 的价值不在静态配比。**

### 5.2 LogFloor 的新贡献：顺序可迁移（但仅限无相变任务）

LogFloor 的独到之处是证明了 proxy 的价值在**动态顺序**而非静态配比。70M→12B 顺序迁移成功（bAbI），且消解实验极干净。

但 QuaDMix 失败的根因不全是"用了静态而非动态"——更深的原因是**有相变的技能在 proxy 上和 target 上表现完全不同**。LogFloor 用无相变任务避开了这个问题。

### 5.3 对 QuaDMix 失败归因的修正

| 失败原因 | QuaDMix 失败分析文档的归因 | LogFloor 的新视角 |
|---------|--------------------------|-----------------|
| 静态 vs 动态 | ✅ 部分原因 | ✅ 独立验证（静态塌缩消除收益） |
| 相变阈值差异 | ✅ 核心原因 | ❌ 未涉及（用无相变任务绕过） |
| proxy→target gap | ✅ 核心原因 | ⚠️ 部分反驳（顺序在无相变时可迁移） |

---

## 六、价值评估

### 优点

1. **消解实验极干净**：60.3%→10.3%→6.0%（正确→静态→逆转）是"顺序 > 配比"目前最强的实验证据
2. **70M→12B 迁移**：170x 尺度跨度，bAbI 30.9% 节省，MMLU 60.3% 节省
3. **用 Qwen2.5-1.5B**：和我们的场景同尺寸
4. **loss 代替 accuracy 的思路有实操价值**：域级 loss 轨迹可以在训练中实时获取，不需要等 accuracy 涌现

### 不足

1. **没验证 loss 顺序 = accuracy 涌现顺序**——恰好是 STEM 场景的桥
2. bAbI/MMLU 域太简单，70M 都能取得 loss 进展；对 GSM8K 这种 70M accuracy=随机的任务未测试
3. 没有跨训练阶段（pretrain→mid-train）的验证
4. "Submitted to AAAI 2027"——还在审，不是已发表论文
5. 跨架构迁移不保证（Qwen→Mamba 行，Mamba→Qwen 不行）

### 评级

- **梯队**: T2（大厂经验级——实证扎实但场景不完全对口）
- **相关性**: ★★★☆☆（概念相关，实证有参考价值，但核心 STEM 相变问题未解决）
- **对项目的核心价值**: 提供了 QuaDMix 失败的新解释角度（"proxy 的价值在顺序不在配比"），同时暴露了新的未验证假设（"loss 顺序能否代替 accuracy 涌现顺序"）

---

## 七、与现有论文的关联

| 关联论文 | 关系 |
|---------|------|
| Phase Transitions #01 (Gu et al., NeurIPS'25) | LogFloor 引用但绕过；我们用它解释为什么 LogFloor 在 STEM 场景可能失败 |
| QuaDMix 失败 | LogFloor 独立验证"静态配比不可迁移"（60.3%→10.3%）；新发现"顺序可迁移"（但仅限无相变任务） |
| Why Larger Models Learn More #52 | 容量竞争解释为什么 70M 的难度层级可能和 12B 不同 |
| RegMix-D #36 | 同为 proxy 轨迹利用，但 RegMix-D 回归 loss 轨迹搜静态配比，LogFloor 直接用顺序 |
| ADAPT #22 / HDS #40 | 同为"动态 > 静态"证据，LogFloor 的消解更干净 |
| DomainPilot #39 | 同为 loss 引导方法，但 DomainPilot 在 target scale 操作不用 proxy |

---

## 八、核心结论

> **LogFloor 证明了 proxy 的价值在"顺序"不在"配比"，用 loss 绕过了 accuracy 涌现问题。但 loss 下降顺序是否等于 accuracy 涌现顺序——这个桥没搭，也正是我们 QuaDMix 失败时撞到的墙。**
