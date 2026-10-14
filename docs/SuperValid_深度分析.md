# SuperValid 深度分析（#132，T1 [高价值]）

> 2026-10-13 录入当日深读。用户场景：1.5B dense / ClimbLab 基座 CPT / 1.8B tokens / r=0.7 / STEM accuracy 直评 / N=1-2。**重要前提修正**：用户现有做法已经是 benchmark loss（MDE 的 end-task 路线，0.844 侧），本文增量是"降格式伪影+免评测成本"，不是"救偏心"。

## 基本信息

| 项目 | 内容 |
|------|------|
| **标题** | SuperValid: Capability-Aligned OOD Validation for Generalizable Downstream Scaling |
| **arXiv** | 2605.28179（v1 2026-05-27，v2 2026-09-03） |
| **机构** | 蚂蚁集团（Quanen Sun, Changxin Tian, Ke Shi, Cai Chen, Cunyin Peng, Jia Liu, Kunlong Chen, Zhiqiang Zhang, Jun Zhou；实习工作） |
| **关注度** | 引用 1/高影响 1——**4.5 个月新文，时间因素非质量信号** |
| **梯队** | T1 [高价值]（2026-10-13 用户裁定升 T1：2T 受控+16 模型观测+蚂蚁工业组） |

## 一句话

**【一句话】**把 benchmark 蒸馏成知识因子、去语料检索真实文本、LLM 改写扩展成"知识密集的普通文本"当验证集——训练中算 loss 一样便宜，但对配比变化免疫（IID loss 会漂）且对格式伪影免疫（benchmark 题面 loss 会虚高）。

## 问题定位（三种验证信号各自的病）

| 信号 | 便宜？ | 病 |
|------|--------|-----|
| IID val loss（训练分布切子集） | 是 | 配比一换就漂——loss 变化混入"训练数据离验证集远近"的污染项，跨 run 不可比（MDE 的 0.282 侧） |
| benchmark 直评 | 否 | 贵；scenario 伪影（格式怪癖/小样本噪声） |
| benchmark 题面 loss（**用户当前做法**） | 中 | 混"题面格式熟悉度"——数学语料含大量题目式文本时高 r run 虚高（0.844 侧但带伪影） |
| **SuperValid loss** | 是 | 对上述两类免疫；代价=一次性合成管线+仍是能力的代理非能力本身 |

## 算法（四步，离线一次，全 run 复用）

对每个能力域 d_k（benchmark 集 B_k）：
1. **知识抽取**：LLM 把每题拆成"解这道题必须懂的知识因子"（概念/定理/方法名）
2. **检索**：每个因子去语料（FineWeb+OLMOTrace）检索真实文本——找"这个知识的常见叙述场景"
3. **过滤**：LLM judge 删弱相关
4. **场景扩展**：LLM 把检索文本**改写扩展**成知识密集、上下文多样的验证样本

产出：6 域 86,102 样本（Knowledge 19.8k/Basic Math 12.6k/Intermediate Math 21.7k/Code Gen 3.9k/Code Comp 3.2k/Reasoning 24.9k），均长 1.3k-4.8k tokens。生成器=Qwen3-Next-80B-A3B-Instruct。

**指标用法**：验证文本是普通叙述文本（非问答对），算 next-token loss，不判对错——考"懂不懂这个知识"，不考"见没见过这种格式"。

## 三层证据链

1. **温度计准吗（观测）**：16 个开源基座（Qwen3 1.7B-30B、Qwen2.5 0.5B-3B、GLM-4.5-Air/4.7-Flash、DeepSeek-V3/V3.1、Kimi-K2、Ling-Mini-2.0——**Qwen2.5-1.5B 在列=用户锚点尺寸**）上 SuperValid loss 与能力 sigmoid 拟合，平均 MSE 1.51e-3（跨架构/尺度/数据分布）
2. **哪个零件不可少（消融）**：挖空题目 7.29e-3 → 只检索不改写 2.31e-3 → 完整管线 **6.73e-4**——**改写扩展贡献最大**（检索锚定真实性，改写换覆盖与多样性，二者缺一不可）
3. **换配比后还准吗（受控）**：2T tokens from-scratch 两阶段（1T 配比 A → 1T 配比 B 加大合成比例），每 25B tokens 采一次信号——**切换瞬间 IID loss 大跳（破坏 scaling law 追踪），SuperValid loss 平稳**。**不跳的机制（2026-10-13 讨论记录）**：IID 验证集靠近旧配比，切换后模型分布离验证集变远——读数混入"尺子跟着旧配比挪动"；SuperValid 文本构造上就没靠近过任何训练分布（检索自通用语料、按知识概念组织），不存在"变远"——**尺子钉在能力空间的墙上**，读数只随能力本身变化（构造逻辑+实证，非定理）

## 两个配套性质（白话）

- **域级聚合 > 单 benchmark**：单 benchmark 自带格式怪癖和小样本噪声；域内多 benchmark 合起来造一个验证集，信号更稳，且能预测造集时没用的新 benchmark（cross-domain 泛化）。含义：验证集按 capability 粒度造，不用每个考试一套
- **服从 log-linear scaling law**：193M-6.17B MoE 同数据 from-scratch，SuperValid loss 随算力 log-linear 下降（与普通 val loss 同款性质）。含义：不止当监控仪表，还能**当 scaling law 拟合的目标函数**——compute-optimal 分析可建立在能力对齐信号上

## 对用户场景的落地推演（CAP v2 草稿）

1. GSM8K+MATH 题目 → 强 LLM 抽知识因子（任何强模型，不必 Qwen3-Next）
2. 从 **ClimbLab 池**检索（池 1.2T 规模不大，BM25/嵌入式检索即可）→ 过滤 → 改写成叙述性数学文本，~2000-5000 样本。**污染警告（2026-10-13 讨论）**：检索池=训练池时，验证文本可能字面混入训练数据被背下（loss 虚低）——SuperValid 构造上免疫"分布距离"伪影，**不免疫字面污染**；检索命中的源文档必须从训练侧剔除，或改用 ClimbLab 的 held-out 切片做检索池
3. **回溯验证（零训练成本）**：拿已有的不同 r 检查点，算 SuperValid loss 和 benchmark loss，看哪个与最终 STEM accuracy 排序更一致——一致则管线成立，不一致则改写跑偏、重建
4. 用途分层：IID val（run 内监控）→ SuperValid loss（**跨 run 挑配比**，免疫格式虚高）→ STEM accuracy 直评（最终裁决）

**改写跑偏的风险与保险**：真正的风险不是"答错题"（没有题）而是文本偏离目标知识。三道保险：检索锚定（改写真实语料非自由生成）、过滤步、实证相关检验（16 模型相关性=管线整体没跑偏的证明；用户版=第 3 步回溯检查）。

## 库内连线

| 论文 | 关系 |
|------|------|
| #122 MDE (GDM) | 前驱：证明验证目标选择>方法选择（ρ=0.844 vs 0.282）；SuperValid 给"怎么造对的目标"的工业配方——两篇合起来是完整验证集设计论 |
| #51 Nexus | loss≠accuracy 鸿沟：SuperValid loss 仍是代理（代理的代理），STEM accuracy 直评不可替代 |
| #130 MergeMix | 互补工作流：merge 搜索（试错）+ SuperValid 信号（挑选）+ benchmark 直评（裁决），全部塞得进 N=1-2 |
| #101 UltraData | tiered vs mix 证据同族：信号质量分层设计 |

## 可信度评估（准则）

- **①规模**：2T tokens 受控训练（token 轴工业级）+16 模型观测（覆盖 Qwen2.5-1.5B 至 DeepSeek-V3/Kimi-K2 级）；**注意 total vs active**：自训系列 193M-6.17B MoE（total 轴大）但 active 10M-324M（计算量轴小于用户 1.5B dense）——**过线（边界案）**，靠观测广度补
- **②机构**：蚂蚁工业组——**过线**
- **③关注度**：无 venue（v1 2026-05 提交，可能正在评审周期）、引用 1——**时间因素非质量信号，未知待观察**
- **核心保留**：观测证据是相关性非因果；2T 切换实验的模型尺寸正文未明说；域划分（6 域）手工；中文评测烙印（CEval/中高考）——方法可平移，数字勿直引
