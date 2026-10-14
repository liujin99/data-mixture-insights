# MAI-Thinking-1: Building a Hill-Climbing Machine（#73 深读）

- **论文**: MAI-Thinking-1: Building a Hill-Climbing Machine
- **作者/机构**: The Microsoft AI Team（官方 Tech Report，2026.08，通讯 mai-technical-report@microsoft.com）
- **模型**: MAI-Base-1 = 35B active / 1T total MoE（LatentMoE，L78）；预训练 30T tokens@16K → 中训练 3.4T@64K → 150B@256K；GB200 × 8192 卡
- **定位**: 不是数据配比论文，而是**全流程技术报告**；§2.5（约 6 页）是微软级预训练+中训练数据配比实操记录——**业界顶级配比方法的一手证据**
- **梯队**: T2 [高价值]

## 一句话总结
微软用 183 个 proxy 模型的 Pareto frontier 做配比搜索，过程中**实证推翻了自己依赖的 rank invariance 假设**（RegMix 的理论根基），随后用 hierarchical local/global search + scaling ladder 补救；中训练配比 STEM:Code:Other = 35:55:10，并用 **memorization-aware epoch capping**（NLL<0.01 占比）动态限制各源重复次数——这是对"高质量小数据集多 epoch 重复"问题的目前最精细的业界解法。

## 1. 配比目标函数（Eq.3，§2.3）——先定义"什么算好"
```
Target = 0.5×Coding + 0.175×STEM + 0.175×Math + 0.1×GeneralKnowledge + 0.05×Multilingual
```
- **约 40 个内部 NLL 基准**，5 类；全部与训练数据 20-gram 模糊去重（阈值 80%）+ 移除全部 huggingface.co 及镜像域名
- **Math 单列**（与宽泛 STEM 分开），因为数学需要显式优先级
- NLL 打分先经固定内部参考模型归一化，再类内平均——**不同模型同 tokenizer 才直接可比**
- 价值：配比优化的第一件事是**把多目标压成单目标**，且权重显式编码业务优先级（Coding 0.5 一家独大）

## 2. 核心发现：Rank Non-Invariance（§2.5.2，Figure 6）★★★
**故事线**（对 proxy 可靠性主题最重要的一手证据）：
1. 微软采用 RegMix 式 forecasting：几千个小模型（760M-4B active），基于基线混合采样，拟合配比→性能预测模型
2. 早期某代模型造了两个候选混合：**code-heavy-mix**（约 50% code）vs **stem-heavy-mix**（STEM 大幅上调）
3. 小规模结果符合预期：stem-heavy-mix 在 STEM NLL 上领先
4. **23B active × 20T tokens 中途交叉**：训练中段曲线反转，code-heavy-mix 最终反超
5. 归因：两个高质量 STEM 源（stem-heavy-mix 中占 11.8%，code-heavy-mix 仅 0.3%）**模糊重复多、多样性差**——小模型阶段很香，大规模下多样性不足成为致命伤

**含义**：
- RegMix 的 rank invariance 假设（小模型排名 ≈ 大模型排名）**被工业级实验否定**——且是在 23B/20T 这种量级才暴露
- 与本项目"小 proxy 不可用"证据链一致（`小Proxy不可用_多论文交叉验证总结.md`、CAMEL proxy reliability 系列），但这条是**反向模式**：不是 proxy 信号噪声大，而是**信号本身随规模改变符号**（scale-dependent，引 Ye et al. 2025）
- 微软的应对不是放弃 proxy，而是**升级协议**：所有消融在 scaling ladder 上跑（恒定 TPP 的多尺度曲线），重点看候选混合的**scaling 行为**而非固定尺度成绩；最终候选在 2.8× 计算量上验证后"最优候选不再随规模变化"

## 3. Hierarchical Local/Global Search（§2.5.3）
数据分约 10 个大类（coding/STEM/PDFs/general web...），交替：
- **Local search**：冻结所有大类权重，只动某大类内部子源权重（如 code files vs PRs vs commits）
- **Global search**：冻结各大类内部构成，只动大类间相对权重
- **8× 重复上限**：任何数据集最多重复 8 epochs（防过拟合+边际递减，引 Chen et al. 2026a）
- Figure 7 方法论：采样不同 datamix → 各训一个模型 → 看**最终 NLL 与混合成分的相关性**（例：研究生物理 NLL 与 PDF/数学/STEM web 数据量正相关，general web 中性，**加 code 无益于物理**）——相关性分析当配比信号用

## 4. 最终配比（Table 5，预训练 30T）
| 源族 | Unique (T) | Training (T) | 占比 % | 平均 epochs |
|---|---|---|---|---|
| Code | 7.4 | 16.4 | **54.6** | 2.22× |
| STEM | 2.2 | 4.7 | **15.8** | 2.17× |
| Math | 0.3 | 1.6 | **5.4** | **5.28×** |
| Books & journals | 0.6 | 0.9 | 3.1 | 1.65× |
| PDFs | 2.7 | 1.4 | 4.7 | 0.53× |
| Web text | 8.1 | 4.5 | 14.9 | 0.55× |
| Multilingual (other) | 8.1 | 0.5 | 1.6 | **0.06×** |

- **Math 最稀缺 → 重复最多（5.28×）**；约 300B 高质量数学 token 被反复用足
- Web/PDF 各只消耗一半库存（0.55×/0.53×）——**不稀缺的源不重复，宁可换着用**
- 多语言被最激进降采样（8.1T 库存只用 0.5T）
- 总体 1.03×：30T 训练量下整体近似单 epoch，靠稀缺源定向多 epoch 撑起有效数据量

## 5. 中训练配比（§2.5.4）★★★（用户阶段直接对标）
**STEM:Code:Other = 35:55:10**，且：
- **数据全部来自预训练语料库**，不引入新源/合成数据——只做"过滤+重加权+长序列重打包"
- **single-source microanneals**（OLMo 2 方法，#75）做类内权重微调，目标函数沿用预训练 NLL 套件+长上下文 NLL 任务
- **STEM reasoning 过滤**：从 PDF 语料用 Essential AI 内容质量 taxonomy 过滤——Bloom Analyze 启发式（技术正确性高 + 推理深度至少中级 + Bloom 认知层级 ≥ Analyze），剔除提取伪影/缺失正文，**保留结构化推理文档而非事实陈述**
- **Code 两步**：① 按质量分层的文件扩展名过滤（HTML/CSS/SVG 在顶级仓库保留、在低级仓库剔除——同名文件的价值取决于宿主仓库质量）；② 文件级+仓库级双格式打包
- **Memorization-aware epoch capping**：用两个预训练 checkpoint 之间验证损失改善中，来自 NLL<0.01（近确定预测）token 的**占比**估计记忆化程度 → 记忆化重的源在中训练里给更严的 epoch 上限
  - 本质：**重复预算按源的"可记忆性"动态分配**，而不是统一 8×
- **长上下文两阶段扩展**（附录 C.1）：试过加权重长文档/调域配比，**全部无效**——最优解 = 沿用中训练同配比，仅以目标长度重打包。简单方案赢了
- 中训练超参：Phase1 cosine 2e-5→1e-5 无 warmup；Phase2 恒定 1e-6

## 6. Scaling Ladder（§2.2.1）——所有消融的载体
- 恒定 TPP（tokens per active parameter）训多个尺寸；架构实验用 100-200 TPP（近 Chinchilla），主 run 用 **500-1000 TPP**（过度训练产紧凑模型）
- **用户 1.5B×1.8B tokens ≈ 1200 TPP**，恰在微软主 run 区间（500-1000+）——**用户处于"过度训练紧凑模型"区，配比行为应参考高 TPP 协议，而非 Chinchilla 区**
- EG（efficiency gain）指标：候选与 baseline 的 scaling law 反解等价成本比，且 **FLOPs-EG 与 Time-EG 分开**（解耦实现优化红利）

## 7. 与本项目的系统关联

### 7.1 场景匹配度
| 维度 | MAI-Thinking-1 | 用户管线 |
|---|---|---|
| 阶段 | 预训练配比+中训练配比 | 中训练配比 |
| 规模 | 35B active/30T | 1.5B/1.8B（TPP≈1200，同属高 TPP 区） |
| 目标 | Eq.3 加权 NLL（Coding 0.5） | STEM accuracy |
| Proxy | 760M-4B×几千个 | 400M×N 个 |
| 数据 | 全自研、零合成、零开源数据集 | ClimbLab 1.2T 生态（ClimbMix 复现） |

### 7.2 可迁移洞察（按价值排序）
1. **Rank non-invariance 直接威胁用户的 proxy 线**：400M 代理模型给出的配比排名在 1.5B 上可能反转——尤其当某源"高质量但低多样性"时，小模型高估其价值。微软在 23B/20T 才看到交叉，用户 1.5B/1.8B 的尺度差（400M→1.5B ≈ 3.75×）小于微软（5B→23B ≈ 4.6×）但同数量级。**缓解**：proxy 结论必须在 1.5B 上跑 ladder 式多点验证（用户现有 N=1-2 预算下的最低配版：至少一个中间尺寸/中间 token 点确认趋势不反转）
2. **Memorization-aware epoch capping 可低成本移植**：NLL<0.01 占比只需要两次前向（两个 checkpoint 对同一验证集），是中训练里决定"哪些源该重复、重复几次"的 principled 信号。与 Capacity-Dependent（#88）的"低似然=已掌握"三区间互补：MAI 管**重复上限**，#88 管**入选门槛**
3. **中训练不引入新源**：只在既有语料内过滤+重加权。支持用户"复现 ClimbMix 后再做数据侧增强"的路线——先不动源，动权重与过滤
4. **稀缺源多重复、富源不重复**（Math 5.28× vs Web 0.55×）：用户 30% 通用侧若含高价值小源，重复 2-5× 优于均匀摊薄
5. **Bloom Analyze 层级过滤**：STEM reasoning 数据保留"推理深度≥中级"文档——与 #88 student-grounded 难度互补（MAI 按内容认知层级选，#88 按学生似然选）
6. **上下文扩展调配比无效**：微软花真金白银试出来的负结果——扩展阶段保持配比、只重打包，别折腾

### 7.3 不可直接迁移
- Eq.3 权重（Coding 0.5）是微软业务优先级，用户目标是 STEM accuracy——需重设权重，但"显式加权压成单目标"的做法本身可抄
- 35B MoE/30T 的绝对数字、GB200 基建、几千 proxy 的算力
- 零合成数据原则与用户 ClimbMix（含合成数学）路线冲突——微软是"从零攀爬"哲学的立场声明，非普适结论

## 8. 局限性（作者自述 + 本项目视角补充）
- 作者自述：配比目标函数/NLL 评估/搜索算法每个组件对最终混合影响都很大，端到端 scaling 行为与"哪些指标可靠预测下游性能"仍是**开放问题**（§2.5.3 结尾原话承认 headroom 巨大）
- Rank non-invariance 的归因（两源 11.8%→0.3%）是**事后假设**，无受控验证
- NLL<0.01 阈值未见敏感性分析；"近确定预测"也可能是高频模板而非真记忆
- 本项目视角：无 1.5B 量级数据点，小模型段行为靠外推；Tech Report 无同行评审（但 Microsoft AI 官方出品、细节粒度业界罕见，可信度按厂商 TR 惯例 T2 处理）
- 污染防御（20-gram/80%）自认 imperfect——但比 Puro（无审计）严格

## 9. 评估
- **证据等级**: 一手工业级实验（183 proxy 模型、23B 验证 run、完整 Table 5）；核心负结果（rank non-invariance）有 Figure 6 双尺度曲线支撑，可信
- **与本项目主题的接口**:
  - *proxy 可靠性*: ★★★ 最强业界反例之一——proxy 不仅噪声大，信号可随规模反转；直接喂"核心未解决问题"文档
  - *重复/epoch 策略*: ★★★ memorization-aware capping 是该主题目前最精细方案
  - *中训练配比*: ★★★ 35:55:10 + microanneals + 不引入新源，与 OLMo 2（#75）、一线厂商 annealing 分析互相印证
  - *数据质量过滤*: ★★ Bloom Analyze 层级是可操作的内容侧信号
- **后续动作**: rank non-invariance 实例并入 `核心未解决问题_相变与代理迁移.md`；与 CAMEL/MobileLLM-R1 的 proxy 证据链交叉引用
