# 关键 Takeaway（[高价值] 论文家族）

> 从 `论文总表.md` 表头迁出独立成档（2026-11-16 用户裁定；原板块 2026-09-02 设立）。
> **对用户场景**：3B-A1 MoE 中训练（2026-11-06 锚点更新）/STEM accuracy/N=1-2 runs。
> **收录规则**：[高价值] 家族全覆盖（纯 [高价值] / [工业实战/高价值] / [高价值/重要]；另含 #51/#114 两行改标前遗留的轴外变体——takeaway 仍有效）。
> **记录规范**（2026-10-12）：一句话白话版关键结论置前，细节与数字置后。
> 新增 [高价值] 论文时同步在本档补行。

- **#01 Phase Transitions**：STEM 知识获取存在尖锐相变（r_thres ∝ M^(-α)），1.5B 已跨 STEM 阈值 → 最优配比"多给 STEM"，与 14M proxy 结论方向相反——这是 QuaDMix 失败的理论根源
- **#26 FastMix**：可微 bilevel 优化 1 个 proxy 实现 550× 配比搜索加速；但 proxy 架构与 QuaDMix 相同 → 同样带容量错位风险，§4.3 工业踩坑经验直接适用
- **#27 Domain-Aware Scaling Laws**：域间协同（γ）与竞争（σ）可量化，R²=0.91，协同 ≠ 语义重叠——pairwise 特征可直接加入 LightGBM
- **#36 RegMix-D**：用训练全程 loss 轨迹（而非终点 loss）做回归，128 个 proxy 超过 512 个——信号在轨迹形状里，不在终点
- **#37 InfoLaw**：信息论 scaling law 同时建模配比权重+重复递减收益，7B/425B 验证，0.15% loss 误差
- **#38 Explaining DM Scaling Laws**：容量竞争理论——有限容量在域间分配导致域 loss 全局耦合，proxy 与 target 容量竞争机制不同（小 proxy 不可用的机制解释）
- **#51 Nexus**：同 val_loss 差 15 分下游——优化器本身就是隐式配比器，val_loss 不充分的最硬证据
- **#58 GLM-5**：唯一设专门 Mid-Training 章节的大厂 TR——中训练已是标配阶段
- **#60 Nemotron 3 Ultra**：19 类别数据比例全公开，数据配比最透明的大厂参照
- **#63 Data Mixing Agent**：RL 学配比策略、benchmark 直接作 reward（绕过 loss≠accuracy），但 1996 GPUh 一次性成本不适合 N=1-2
- **#64 BETR**：benchmark 相似度做文档级选择，成本最低（~2-4 GPUh），精度导向而非 loss 导向
- **#66 DIDS**：FIM+loss 轨迹动态配比，赢 Velocitune 3 分，域内在线方法代表
- **#68 LogFloor**：proxy 动态瓶颈顺序可跨尺度迁移（70M→12B，顺序消解 60.3%→10.3%），但用 loss 绕过 accuracy 涌现——STEM 相变场景未验证
- **#73 MAI-Thinking-1**：rank non-invariance 工业实证（proxy 排名随规模反转，23B×20T 才暴露）→ 400M proxy 结论上 1.5B 前须中间验证点；memorization-aware epoch capping（NLL<0.01 占比定重复预算）；中训练不引入新源只重加权；TPP 500-1000 与用户同区
- **#74 Nemotron-CLIMB**：用户基座的数据来源——聚类迭代搜索优于 DoReMi/RegMix；ClimbMix 400B 已策展，直接消费无需再过滤
- **#82 MobileLLM-R1**：2T tokens 足够推理涌现（Qwen3 36T 的 11.7%）；StarCoder 对数学收益 > OpenWebMath 对代码（跨域收益反转常识）；influence score 收敛=数据耗尽信号
- **#84 Puro-2B**：数据配方是最大杠杆（2.40× > 架构/优化器）——质量课程+恒定 LR+checkpoint 平均，且模型平均单独无益；配比收益穿透 SFT 才算数（+1.77~2.02 pp）；P2 Math 18.3% 配比可当 1.5B STEM 中训练初始候选；⚠️ 0.6B proxy 选 2B 数据的容量错位未检验
- **#88 Capacity-Dependent**：数据难度是（数据×模型）配对属性；1.5B 从"够不着"的数据只学到死记、5 epochs 追不上（No Catch-up 最深区）→ STEM 池别灌超难、打分用 1.5B 自己（400M 系统性高估难度）；边界：+9.4 pp 是蒸馏配对实验数字，文档池过滤收益无人量化过
- **#101 UltraData**：L0-L4 分层数据管理 + 全套开源；UltraData-Math 290B（L1 web 170.5B→L2 模型分级 33.7B→L3 合成精修 88B）在 **MiniCPM-1.2B** 上 MATH500 +3.62pp vs Nemotron-CC——与用户 1.5B 规模重合的最直接公开参照；**深读核心**：L2 vs L1 增益小（+1.42）而 L3 vs L1 **+9.24pp**——纯筛选不够、格式重构（QA/对话/教科书化）才是数学跃迁来源（PPL 过滤=必要非充分）；Math-L3 跨域迁移（English +3.45/BBH +5.87、Code +3.80）=STEM 重仓买通用推理的直接实证；tiered vs mix 同配比 +1.49pp=出场时机独立于配比；Decay Verification（1.3T 基座+30/70 混合续训）=用户 CPT 同构协议模板；中训练应取 L2/L3 级数据（分层→跨阶段分配）；配套 MiniCPM5-2B Base→Midtrain→SFT→Final 四 checkpoint 全开源（#102），可 diff 工业级 2B mid-training 增益定标
- **#110 Marin 535B Hero Run / Harrier**（工程项目非论文）：当前公开可见最完整的"语义聚类+配比搜索+分阶段调度"端到端实现——40 语义域×5 质量桶=200 单元格混合空间，BO+多种子 swarm+缩放梯子三层验证选权重，三阶段中途换混（initial/main/cooldown，Q4 质量桶冷却期 22.9→28.0%）。**配比轴拿走**：共享前缀 checkpoint 恢复+多种子+训练至终点比 final（60M@22% 进度 Spearman 0.298=早期检查点不可靠的又一独立证据）、7 个零权重 uncheatable held-out 评测、60-80% 检查点线性外推+小宽度回测残差校正的中期预测、桶级曝光上限 2.96 epochs。**聚类轴拿走（ClimbMix 复现直接相关）**：5k 微簇→40 域两级聚类（源均衡+每域≥1% 人口下限）、质量分桶按内容类型校准截断、**冻结微簇基直方图特征 h=V·w 解决重聚类搁浅（CLIMB 被点名反面教材：位置性预测器重聚类即重触发 112-run 扫描）**。⚠️ 数值不迁移：目标函数是 macro BPB 非 STEM accuracy（#68 LogFloor 同构陷阱未解），535B/18T 差 4 个数量级；math 簇全程降权 4.3→1.4% 属目标函数差异不构成对 r=0.7 的反证
- **#114 Generalization Dynamics**：预训练全程 "mode-hopping"——相邻 checkpoint 在泛化轴上可完全翻转（81%→0%→81.7%）而 loss 与常规 benchmark 平稳；**"训更久 ≠ 更聪明"（2026-10-12 用户指定关键结论）：继续预训练/中训练只改善常规（in-distribution）成绩，不改善泛化/对齐甚至变差**（4.5T vs 4.9T 经相同 post-training：GPQA 36.3% vs 29.8%、prefilling 鲁棒 53% vs 21%）→ **单 checkpoint 泛化读数不可信、中间 checkpoint 必须保存**；数据子集选择可定向控制泛化动力学；跨数据集泛化行为相关低（单探针不可外推）。⚠️ 32B 尺度+泛化敏感指标（常规 benchmark 平稳→对 STEM accuracy 直接威胁有限）
- **#122 MDE（Google DeepMind，补档）**：**验证目标选择 > 方法选择**——end-task 验证 loss 与下游 accuracy 的 Spearman ρ=0.844，域数据 loss（DoGe/DML 常用 avg-sp 目标）仅 0.282（510M 受控测量）——本项目 6 处引用的"Google MDE"即此文，loss≠accuracy 轴最硬量化之一；MDE 特征（k 个域专家小模型 loss 加权聚合近似混合 loss，理论保证）作回归附加特征 → mixture 排名 Spearman 0.81→0.95（GBM），仅训 k 个专家+~25 混合样本（3× 省算力）。对用户：CAP v1 域覆盖度检查依据；域专家 loss 可作 proxy 线廉价特征（专家=小模型无容量错位）。⚠️ SlimPajama/生成排序基准非 STEM accuracy；Marin #6326 反例（0.86 vs DSP 0.914）警示特征工程上限；2025.02 补档
- **#130 MergeMix**：每域只花 2.5% 预算训个小专家、在"合并权重"空间搜配比——比 proxy 便宜 100×，且 **8B 搜出的配比直接给 16B 用仍然有效**（proxy 跨尺度少数正面证据）；⚠️ 无代码/数据/权重，规模自报不可审计（降 T2 留动态复评触发器）
- **#132 SuperValid**：IID val loss 会随配比"漂"而 capability 级 OOD 验证集不漂——把 benchmark 蒸馏成知识文本构造验证集，配比变化下仍稳定预测下游；动手调参前先修尺子
- **#137 LR Decay Wastes Best Data**：升序质量课程必须配恒定 LR——标准衰减会把"压轴好数据"的收益吃掉（后期 LR 太低模型接不住）；修复=温和衰减或用 checkpoint 加权平均替代衰减（1.5B 上 +1.64%）
- **#143 TiKMiX**：不用 proxy 训练——在目标模型上花 0.1B token 直测各域"实时影响"就定下阶段配比，超 RegMix +1.6~2.0% 而开销仅其 20%；**发明静态配比搜索的团队（字节 QuaDMix 班底）自己转向了动态**
- **#145 Data×LR Schedules**：中训练的数据调度与 LR 调度是一个设计问题不是两个——“MT 数据压轴 vs 早混入”的胜负由 LR 调度决定（全程衰减 LR 下早混入+PT 陪跑赢、恒定 LR 下纯压轴赢）；最优=MT 进场后才开始衰减 LR（收尾段 20-40%）；MoE 下 STEM 重仓配方应渐进入场（ZL）而非跳变（TS）——跳变触发 load-balance loss 尖峰+专家负载排名洗牌
- **#160 Nemotron 3.5 TR（工业实战精选②）**：首个量化配方全公开的工业级 agentic 中训练——65.4B tokens @256K（Issue-PR 41%/仓库 20%/agent 轨迹 10%（工具响应 loss-mask）/commit 3%/CI 1%/**PT 回放 20%**/LC 5%）；效果须用**固定 SFT 桥**评（主效应=跨 5 个非标准 harness 全胜 +8.6/+6.8，标准 benchmark 在波动内）；三明治：MT 装知识/格式，行为归 post-training
- **#161 Singhal Open Midtraining Recipe（工业实战精选②）**：**中训练喂 agent 轨迹要封顶 10-15%——喂太多=变相超长 SFT，会把 RL 阶段练废；该结论 2026-11-16 压力测试通过**（#168/#169/#170 三篇独立学术证据一致、无反例：伤的机制=过量+离策略模仿压缩探索熵→RL 组内奖励全同无信号；**pre-RL 熵 ρ=+0.69 可当事前前兆**；on-policy/适度模仿不伤反益）——其余细则：回放 20%（40% 无增益）、工具响应不 mask（差 10 分）、评测用同款 SFT 桥（base 代理不足以指导配方决策）、LR warmup→恒定按尺寸分档（与 #145 对照）
- **#165 daVinci-Prover**：域专项 agent-native MT 的最佳分析样本（配比六类分类法/**轨迹按 token 计价**（8.4B 轨迹>25.3B 技能）/15B 剂量甜点/$15 万成本账）；**2026-11-16 用户裁定：不直接参考（新文未经广泛验证）——百分比=池绑定不可迁移，可迁移的只有分类法+验证程序**；读数标尺：“+13.7 是旧 base 的数字，新 base 上只剩 +1~3”
