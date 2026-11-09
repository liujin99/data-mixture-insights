# Hinted Scaling Laws 深度分析（#147，T2 [新发现]）

> 2026-11-09 深读（深读队列 2/3，T2 优先）。用户场景：门3（loss→accuracy）测量层——3B-A1 验证档的 STEM accuracy 读数 + r-scan 配比候选排名。本文不改训练、只改**评测协议**：把 benchmark 变成难度连续谱，让小模型打出非零、可外推的分。全文 HTML 已核（arXiv 2611.02270v1）。

## 基本信息

| 项目 | 内容 |
|------|------|
| **标题** | Hinted Scaling Laws |
| **arXiv** | 2611.02270（2026-11-01 v1，无 venue；**有代码+数据** `tatsu-lab/hinted-scaling-laws`，9★、2026-10-31 建、无 license 标注——总表原行"无代码"已修正） |
| **机构** | Stanford（van Adrichem†/Bailey†/Kuditipudi†/Ye/Band/Ahmed/**Duchi**/**Liang**/**Hashimoto**——tatsu-lab=Hashimoto 组，Marin/Delphi 生态班底） |
| **关注度** | v1 一周；新文条款下暂只计开源不计采用（代码已放） |
| **梯队** | T2 [新发现]（2026-11-09 录入裁定：测量层新原语） |

## 一句话

**【一句话】**给模型一点"提示"（hint），把 benchmark 变成难度连续谱：小模型也能打出随 hint 水平平滑上升的分，且 hint 水平与模型规模在 iso-accuracy 等高线上**近似线性互换**——于是"≤1e20 FLOPs 模型 + 全 hint 阶梯"能拟合出外推到 1e23 的 unhinted accuracy（3pp 内），标准 scaling law 同数据差 13pp。

## 方法：三种 hinting 实例化（H=0 恒等于原 benchmark）

| 方法 | 机制 | 需要什么 |
|---|---|---|
| **Truncation** | 把参考解的前 H 比例（词边界截断）放进 prompt | 仅参考解 |
| **Masking** | 参考解整段在 context，随机遮掉 (1−H) 比例的词 | 仅参考解 |
| **Logit mixing** | 与更强 teacher 联合解码：s=z_model+H·z_teacher，H→∞ 收敛到 teacher 贪婪采样 | teacher（**须同 tokenizer**）+ logits 访问 |

统一框架（§6）：与 pass@k（多_rollout）、finetune-data scaling（Snell 2024）同属"沿第二参数变化评测、仍预测原 benchmark accuracy"的家族——hinting 是其中每题成本不变、只改信息量的一支。

## 两个核心现象

1. **Hint scaling（平滑性）**：每个模型的 accuracy 随 H 平滑上升（自由渐近线 sigmoid 拟合很好），且**家族排序随 H 增加基本保持**（大模型曲线在小模型上方，很少交叉）。形状是方法属性：masking 全程稳步爬升、truncation 在解快显完时才动
2. **联合结构（交换率）**：(C,H) 平面上 iso-accuracy 等高线≈**平行对角线**——固定精度下 hint 水平与模型规模（log10 FLOPs/参数）**线性互换**，"提示一分≈规模若干"。例：H=0 时 Delphi 梯子在 1e22 FLOPs 前趴在噪声地板；masking H=0.5 时 9e18 起全部 >7%

## 预测结果（floored sigmoid 联合律）

联合律 `L(C,H)=max(F, L+(A−L)·σ(β₁C+β₂H+β₃))`——6 参数；地板 F 容"涌现点前趴平"、上限 A 容"饱和不足 1"。

- **拟合误差**：联合律 MAE 3.1（masking）/3.3（truncation）/1.7（logit mixing）pp，对比逐片单独拟合 1.8/1.6/1.7——用少 36-96 个参数几乎不损拟合
- **外推（核心数字）**：Delphi 梯子 GSM8K——≤1e20 FLOPs 拟合→**1e23 FLOP 模型（25.9%）误差 3pp**；标准律同数据低报 13pp，放进 1e22 仍差 12pp。**拟合模型算力仅为目标 1/5600、H=0 得分近零**
- **失败案例对照**：Qwen2.5 用最小两个模型拟合——基线在 32B 预测 ~100%（真值 12%，误差 88.2pp，不可辨识）；hinted 同两模型误差 3.0pp。19 面板中 hinted 集成在 17 个上击败基线
- **中位误差 21pp→4.3pp**（跨 Delphi+后训练家族、GSM8K/AIME/HLE）
- **Logit mixing（teacher=Llama-3.1-8B-Instruct 带全解在 context）**：全家族共享 teacher 把所有 hint 切片抬到共同天花板；MAE 1.7、拟合地板 F=2.5% 恰在 1e22 以下模型 0.5-2% 的 unhinted 分之上；9 个训练集平均误差 3.3pp，从目标算力 0.03% 外推最差 6.6pp，**~0.2% 处预测收敛**
- **多种子重要性**：单种子 10.2pp vs 多种子 3.3pp（作者为 Delphi 7 个最小预算补训 2 种子+2 个新预算点 3 种子）——**评测噪声是外推第一杀手，与 Marin 预测赛的 packing 口径教训同族**

## 排名应用（对 r-scan 最直接）

近零模型集（AIME 0.06-0.81%、HLE 1.7-3.0% 的 7 模型组；真值排名来自 4096/1024 rollouts）：

- **hint-prior 排名**：每题 1 个混合 hint 水平 H̄ 的 rollout + (k−1) 个 unhinted rollout，按 z 分数加权 α_k=(k−1)/(k−1+λ)、λ=8（hinted 分算 8 个 unhinted 的信息量）
- 结果：**单 hinted rollout/题 的排名 ≈ 5 个 unhinted rollout/题**（Kendall's τ 相对 unhinted 排名 +0.33 AIME/+0.27 HLE）；最优 H 因 benchmark 而异（AIME 峰值 H=0.7 且 H=1 骤降、HLE 升到 H=1）——hint-prior 用全水平混合避开调 H
- 对比基线：teacher-forced log-loss 排名（零 rollout）与 pass@k 均被击败

## 理论边界（Proposition 1——什么让 hinting 有信息）

**反例定理化**："unhinted-determined" 的 hinting 方法（hint 切片只依赖 Acc(M,0)——如"以概率 H 直接给正确答案"）在 Acc(M,0)=0 的模型集上产生**完全重叠的 hinted 曲线→相同预测→零信息**（Fig.18 实证惨败）。**好的 hinting 必须能按底层能力分离 pre-emergence 模型**——truncation/masking/logit mixing 满足（hint 是部分解，能否用上看模型真本事）。这是设计任何 hinted 协议的第一原理，也是本库"门3"问题的精确化：**不是小模型没信号，是评测没给信号显形的梯度**。

## 证据链与挑刺

1. **外推验证的可回环性**：17/19 面板、多家族多 benchmark——但验证用的后训练家族（Qwen/Gemma）的"能力轴"与预训练梯子的配比外推场景不同构；Delphi 是最贴的场景且数字最好（3pp）
2. **联合律形式风险**：floored sigmoid 的线性交换假设（等高线平行直线）对 masking/truncation 只是"roughly"成立（作者自示部分家族/方法对不平行）；真涌现若比 (C,H) 联合 sigmoid 更尖锐，外推会系统性偏——**3pp 是这些 benchmark 的经验数字，非保证**
3. **排名"近似保序"**：τ≠1，近交叉处仍会换位——r-scan 头部候选（top-2/3）决策仍需 unhinted 复核（与 #130 保真层同款问题：全位次相关≠局部排序正确）
4. **成本结构**：预测用法需小模型全 H 阶梯×多 rollout（小模型便宜，可接受）；排名用法定义上省 5×；但 hint 需参考解（GSM8K 现成、AIME/HLE 需生成+防泄漏——附录 B 两段式制作）
5. **门3 定位清醒**：hinting 不解决"loss→该配什么数据"，只解决"accuracy 在小尺度可测"——是测量层原语，配比决策仍需 3BA1 式受控实验

## 对用户场景的落地推演

**（a）r-scan 的 accuracy 读数协议（最直接）**：3B-A1 验证档扫 r 时，小/中模型在 MATH500/AIME 级 benchmark 上近零——现在有两条读数路线：① **hinted 读数**：GSM8K 级用现成参考解做 truncation/masking 阶梯（每 r 臂扫 H∈{0,...,1}，10 rollouts/题），读"hinted accuracy 曲线族"做配比间的**排名**（hint-prior，单 rollout≈5 rollout 判别力）；② **外推读数**：若要预测更大档（7B/15B）的 unhinted 分，用 floored sigmoid 联合律在梯子小档+全 H 上拟合——Delphi 的 0.2% 算力收敛点是"多小的验证档能支撑多远外推"的定量参照
- **协议细节**：λ=8、混合 H̄、z 分数加权可直接抄（代码开源）；参考解须做防答案泄漏改写（附录 B 协议）
**（b）Delphi 梯子作为外部资产（新用途）**：开源权重梯子（447M→25B、12 预算、多种子、含作者补训的种子）= 我们的**外部验证集**：r-scan 若在自建梯子上发现某配比信号，可先在 Delphi 对应尺度上做 hinted 复核（零训练成本）再上 7B/15B；也是"loss→accuracy 台阶"（Held 原方法）与 hinted 律的 A/B 试验场
**（c）评测算力预算表**：agentic/安全评测单 rollout $3k-5k（UK AISI cyber-range）的量级下，hinting 排名的 5× 节省是真金白银——若 3BA1 后续加长程 agentic 评测臂，hint-prior 是默认排名协议；hint 可回放成功轨迹前段（作者提的 agentic 扩展，未验证）
**（d）与 ScalingBench 的分工**：连续读数两条路——hinting 改**考卷**（信息量轴），条件 NLL 改**读数口径**（似然轴）；r-scan 可双读数交叉验证（两者失败模式不同：hinting 依赖交换率线性、条件 NLL 依赖 CoT 轨迹质量）

## 库内连线

| 论文 | 关系 |
|------|------|
| #10 Proxy Reliability | 同一问题（小模型信号缺失）的**测量层解**：#10 证明 proxy 排名不可靠（Layer 4），#147 提供让信号显形的评测原语——两者可叠加（hinted 读数下重测 proxy 可靠性是显然的后续实验） |
| #01 Phase Transitions | 互为镜像：#01 说相变阈值下 accuracy 恒零；#147 说 hinted 轴上"零下仍有梯度"——**相变前信号**（本库核心未解决问题）的第一个可操作读数方案，Proposition 1 恰好形式化了"什么样的相变前信号有信息" |
| #110 Marin / Delphi | Delphi 是 Marin 生态的开源梯子资产；本文为其补训多种子（种子噪声 10.2→3.3pp 的教训对 Marin 预测赛同构） |
| #122 MDE | "验证目标选择>方法选择"的测量层延伸：目标本身现在可以按信息量参数化 |
| #88 Capacity-Dependent | 互补的容量视角：#88 说小模型学不了超难数据（数据侧），#147 说小模型在 hint 下能显形部分能力（评测侧）——"难度"在两侧被同一变量（学生与任务的差距）控制 |
| #135 RGI | 相对信号耐造的另一实例：hinted 分差=数据敏感的相对量，且跨 H 保序 |
| ScalingBench 线 | 连续读数两条路（改考卷 vs 改口径），失败模式正交 |

## 可信度评估（准则）

- **①规模**：Delphi 梯子（447M→25B、3e18→1e23 FLOPs、12 预算多种子）+四个后训练家族+三个 benchmark——**评测方法论文里规模充分**；不涉训练侧
- **②机构**：Stanford Duchi/Liang/Hashimoto（tatsu-lab）+Marin 生态班底——**过线**
- **③关注度**：v1 一周引用未起；代码+数据已开源（9★ 新文）
- **④开源**：代码+数据（hinted 评测结果）放 tatsu-lab/hinted-scaling-laws——按新文条款暂计开源不计采用
- **⑤时效**：2026-11 在窗
- **证据纪律**：19 面板外推验证+反例基线（trivial hinting、pass@k、log-loss）+失败案例展示+理论边界形式化——方法论文的上乘

## 深读结论

**可立即用于 3BA1 的三条**：① r-scan 的 STEM 读数默认加 hinted 阶梯（GSM8K 级、truncation+masking 双法、参考解防泄漏），近零模型按 hint-prior 排名（省 5× 评测）；② Delphi 梯子作外部验证集（hinted 复核→再上大档）；③ 外推预测用 floored sigmoid 联合律，多种子是硬要求（单种子误差 3×）。**边界**：保序≠保距（头部决策需 unhinted 复核）、交换率线性是经验假设、hinting 不回答"该配什么"只回答"测得更准"。
