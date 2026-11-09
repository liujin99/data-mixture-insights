# LinMix 深度分析（#146，T3 [高价值]——采样协议层精读）

> 2026-11-09 深读（深读队列 3/3）。用户场景：r-scan proxy 实验的**采样协议设计**（3BA1 实验设计的直接输入）。本文价值不在最终配比数字（BPB 目标+学术规模），而在"proxy 混合实验该怎么采样"——与 #70 Scheffé、#124 重复率子采样构成三处独立收敛。全文 HTML 已核（arXiv 2610.37390v1）。

## 基本信息

| 项目 | 内容 |
|------|------|
| **标题** | LinMix: Data Mixing as Linear Domain Attribution |
| **arXiv** | 2610.37390（2026-10-27 v1）；**NeurIPS 2026 Workshop on Attributing Model Behavior at Scale** |
| **机构** | 华沙大学+Centre for Credible AI（Baniecki/Biecek）+维也纳大学（Pawelczyk）——可解释 AI/数据归因方向 |
| **代码** | CredibleAI/linmix（MIT，0★ 新文） |
| **梯队** | T3 [高价值]（2026-11-09 用户裁定降 T3：workshop venue+学术规模+BPB 目标；标记保留=协议价值） |

## 一句话

**【一句话】**proxy 配比搜索的采样该用"顶点采样"而不是 Dirichlet：把每个域推到 cap 或置零（Greedy fill+配对反向排列），设计矩阵近正交、线性模型就够——288 细域时 Dirichlet 全崩（ρ≤0.39）、顶点+线性照常工作，4B 上 MBPP 相对 RegMix +46%。

## 方法（三个设计选择全部经典化）

把 regression-based data mixing（RegMix/Olmix 式）形式化为**域级 datamodel**：proxy 结果对混合线性，系数=域归因（边际贡献，Prop 2 从 Ilyas datamodels 严格化；Remark 1 连 Shapley/Banzhaf 值）。三个设计选择：

1. **采样 Π（核心创新）**：**顶点采样**——均匀随机排列域、按 Greedy fill 逐域填到 cap（式 3），每个混合至多一个内点坐标、其余钉在 cap 或 0；**配对反向排列**（run 2i 复用 run 2i−1 的逆排列——一个填空的域另一个先填满）进一步去相关。D-最优性（max det ΠᵀΠ），**零采样超参**
2. **代理模型**：线性 f(π)=βᵀπ，OLS；β 中心化后=相对平均域的边际价值——**可解释系数直接就是归因**
3. **目标混合**：LP 顶点解（按 β 降序贪心填 cap）；重复递减用 **isoelastic 效用** Σ β_k·π_k^(1−r)/(1−r)（r→0 线性、r=1 对数）——**Prop 1**：正归因域决定支持集、曲率 r 只在内部配预算；**Corollary 1** 闭式 π*_k=min(u_k,(β_k/ω)^(1/r))

**Cap 约束**（重复轴的接口）：π_k ≤ u_k=min(1, κ·π_prior,k)；RegMix 硬编码 κ=15、Olmix κ=4n/b_target、LinMix/Wettig κ≈n/b_target（目标尺度不重复）——**κ 是"上推=重复"的硬旋钮，r 是软旋钮**。

## 理论（定理层）

- **Theorem 1（Dirichlet 的代价）**：归因误差被 Dirichlet 放大——浓度 γ 大（贴 prior）时因子 1/c（欠分散），且 prior 比 τ 能烫平的更偏时再乘 d^(p−1)（尾部饥饿，p=Zipf 尾指数/τ）；对比单纯形最优 σ²d²/m
- **Prop 3（经典）**：无 cap 时平衡顶点设计最优——单域测量使估计不相关、等频测量等方差
- **Theorem 2**：等 cap 下顶点采样期望意义最优（不等 cap 留作 future work；等 cap 下一切设计都要付 ~d/(κ−1) 因子）
- **经验注脚**：贴 prior 的 Dirichlet 混合使 HellaSwag BPB 的变动比顶点设计**小 5×**——"没信号可回归"是 Dirichlet 失败的直接原因

## 实验证据（数字全录）

**1B OLMo2 梯（23× 1B target、32B tokens、WebOrganizer 200B）**：

| 设置 | 结果 |
|---|---|
| 24 域（512×30M proxy） | 各方法都能拟合（两种采样下所有 surrogate 都行）；LinMix ARC .510（+2.8% vs RegMix .496）、HS +0.9%、CPile −0.6%（略输 RegMix）——**粗域时差别小** |
| 288 域（1024×60M proxy） | **Dirichlet 默认超参（τ=2, γ∈[0.1,5]）不收敛**（cap 违约拒绝循环），按条件数调 γ 到 [40,4000] 后仍 ρ≤0.39；顶点+线性与最佳非线性持平；**ARC +13.8%/HS +11.8%/HS+PIQA +7.6%/CPile +1.9%** |
| 跨尺度迁移 | 30M 群线性拟合给 60M/100M 排名 **ρ≥0.95**；Dirichlet ≤0.42 |
| 半群效率 | 一半顶点群比全量 Dirichlet 群拟合更好 |

**4B Qwen3.5 梯（10× 4B target、80B tokens=20× TPP、416B=WebOrganizer+Nemotron-CC-Math〔#158〕+Stack-Edu、40 域、512×100M proxy）**：

| 目标 | LinMix vs 最佳基线 |
|---|---|
| MBPP（code） | **.444 vs RegMix .304/Olmix .302（+46%）**；HumanEval +15% |
| GSM8K（math） | **.337 vs Olmix .294（+14.4%）**；MATH .266 vs .202（**+31.6%**） |
| HellaSwag | .696（+4.4%）；CPile +0.5% |

**关键对照**：RegMix 的 GSM8K .135 **低于 prior .214**（Fig.4：RegMix 三目标给出几乎相同的混合——不分化）；Olmix 部分分化；LinMix 分化最强（语言目标降 math/code、code 目标升编程语言/软件域、math 目标升 math/science 域）——**目标定向能力本身是差异化来源**。

## 证据链与挑刺

1. **线性假设的边界（对 r-scan 最重要）**：顶点设计识别的是**主效应**——每个混合由 1-few 个钉 cap 的域主导，**域间交互项（共训于中等比例时）恰恰被顶点设计最少行使**。而库内证据（#27 域间协同可量化、#70 Scheffé 成对交互真实、#130 math 最优点在混合区不在纯域）一致表明交互真实存在。**LinMix 的最优解永远是顶点（贪心填 cap）**——它结构上回答不了"最优 r 是否在内部"的问题；isoelastic r 是唯一的软化旋钮（r=1 对数效用把预算摊给全部正归因域）
2. **BPB 目标**：proxy 指标=benchmark 训练_split_ 的 BPB（比通用 val loss 更贴 end-task，#122 方向正确），但门3 未测——BPB→accuracy 的选择收益在 4B 上验证了，机制上仍是"排序对了"
3. **规模**：30M-100M proxy/1B-4B dense target；**MoE 未测**；但实验量大（23+10+4 个 target run、2000+ proxy run）
4. **venue**：NeurIPS 2026 **Workshop**（AtScale）——非主会；0★ 新代码
5. **诚实处**：自曝 Dirichlet 调参协议（条件数最小化）是自己发明的补丁、不等 cap 的最优性留白、标准误缩放系数的 ablation 建议都来自附录
6. **⚠️ 与 CLIMB 一手经验的张力（2026-11-09 用户提出，重要边界——已扩展为专门分析 `docs/Dirichlet采样可用性边界.md`）**：#74 CLIMB（NVIDIA，NeurIPS'25，T1）用 **Dirichlet 采样**（按簇 token 数初始化；其消融还显示优于均匀采样）在 **20-21 语义簇**粒度上做配比搜索，**优化有效**（超 DoReMi/RegMix 基线，用户一手复现同向）——与"Dirichlet 出局"表面相反。实为**边界互补**：LinMix 自己承认 24 粗域时"各方法都能拟合"（ARC 仅 +2.8%、CPile 略输），其"崩"仅在高维细域（288 域、贴 prior 采样）成立。**正确读法：LinMix 的主张是"Dirichlet 可用性随维度退化"，不是"Dirichlet 普适无效"——粗簇层（≲24 域）Dirichlet 可用（CLIMB 一手证据 + LinMix 自证双确认），细簇参数化时须换顶点/经典设计**

## 对用户场景的落地推演（r-scan 采样协议——本篇对本项目的全部意义）

**（a）Dirichlet 的可用边界（2026-11-09 按 CLIMB 一手经验修正）**：r-scan 采样协议**不盲从"Dirichlet 出局"**——粗簇层（≲24 域，CLIMB 的 20-21 簇粒度）Dirichlet 实战有效（#74 一手证据+LinMix 自证 24 域各方法持平）；**LinMix 的警告只在细簇参数化（100+ 域）时生效**（贴 prior 欠分散+尾部饥饿的定理级代价）。若 r-scan 域数多（ClimbLab 226 微簇级），再切顶点/经典设计
**（b）r-scan 设计的直接合成（顶点+内部点）**：LinMix 顶点定主效应（哪些域值得喂）+ #70 Scheffé 内部/边点测交互（最优 r 是否在内部）——**两段式设计**：先顶点群出归因 β（哪些簇正贡献、排序如何），再在正贡献簇的张成内部放轴向+中心点扫 r；配对反向排列技巧直接抄（两 run 一对，支持集互补）
**（c）两个重复旋钮的分工**：κ（硬 cap，LinMix 设 κ≈n/b_target=目标尺度不重复）与 isoelastic r（软曲率，ln π 摊预算）——3BA1 的 STEM 池若小于目标占比，κ 由池子大小物理决定，r 则是"第 2 个 epoch 值多少"的先验（连 #123 定价几何/#125 大数据集耐重复——r 应随池子变大而调小）
**（d）归因系数的用法**：β_k 是**目标定向的可解释输出**（哪个簇喂哪个能力）——比 LightGBM 黑箱多一层"配比理由"；r-scan 结束时可输出"STEM 目标的正归因簇清单"作为后续细化的搜索空间
**（e）跨尺度证据**：30M→60M/100M 线性归因排名 ρ≥0.95（同栈近邻尺度）——**归因排序在小尺度邻域内稳**（连 #135 RGI 相对信号耐造），但 1B/4B target 的增益证明的是"用归因选混合有效"，不是"30M 归因=1B 归因"（那一步仍靠 3BA1 中间验证点）

## 库内连线

| 论文 | 关系 |
|------|------|
| #70 Mixture Experiment | **同层三处收敛之三**：#70 用 Scheffé I-最优+成对交互模型；#146 用 D-最优顶点+纯线性——**互补而非替代**（主效应 vs 交互）；Greedy fill+配对反向排列与 #70 的设计点选择可拼装 |
| #124 Repetition Mismatch | 同层收敛之二：#124 治"重复率失配"（proxy 与 target 的 epoch 结构对齐）；#146 治"采样欠分散"——r-scan 协议须两者都过 |
| #36 RegMix | 被击败的基线（288 域 ρ≤0.39、GSM8K 低于 prior）；其 LightGBM+Top-K 平均在粗域仍可用 |
| #155 Olmix | 被击败的基线（指数律 f(π)=exp(Σθπ)+θ₀，凸可精确最小化）；其 mixture reuse 主题与采样协议正交 |
| #158 Nemotron-CC-Math | **被 LinMix 用作 4B 语料成分**（416B 之一）——库内资产被第三方采用的又一数据点 |
| #27/#130 | 交互真实性的反方证据：LinMix 顶点解结构性测不到混合区最优点——r-scan 不能只抄顶点 |
| #123/#125/#105 | 重复轴三件套：κ/r 两旋钮的定量参照 |
| #10 Proxy Reliability | LinMix 引用同线；顶点设计=在"proxy 可靠性"问题上先修测量（设计矩阵条件数），再谈外推 |
| #135 RGI | 30M→100M ρ≥0.95 是"相对信号近邻尺度耐造"的又一实例 |

## 可信度评估（准则）

- **①规模**：30M-100M proxy/1B-4B dense target/32-80B tokens——学术规模（用户裁定降 T3 主因）；MoE 未测；但 2000+ proxy run+37 target run 的实验量在学术工作里属大
- **②机构**：华沙 Credible AI+维也纳（可解释 AI 方向组，非头部 LLM 组）——中等
- **③关注度**：workshop venue、新文引用未起
- **④开源**：代码 MIT（0★ 新文）+模型/数据链接；按新文条款暂不计采用
- **⑤时效**：2026-10 在窗
- **证据纪律**：定理+反例+跨尺度迁移 ablation+诚实自曝调参补丁——上乘；主缺口=BPB 目标、MoE/大规模未测、顶点解与交互的盲区自认不足

## 深读结论

**r-scan 采样协议的三个组件就此集齐**：①Dirichlet **按维度条件化**——粗簇层（≲24 域）可用（CLIMB 一手证据+LinMix 自证）、细簇层（100+ 域）定理级崩，届时换顶点设计；②顶点群定主效应（Greedy fill+配对反向排列，零超参）+内部点测交互（#70 Scheffé）——两段式设计；③重复轴 κ（硬）+r（软）双旋钮。**边界**：LinMix 的顶点最优结构性测不到混合区最优点（交互盲区）——r-scan 的内部点扫描不可省；BPB 目标+学术规模+workshop venue，最终配比数字不引，协议层引。
