---
title: 数据集
tags:
  - 索引
updated: 2026-11-02
created: 2026-11-23
---

> 论文里那些名字对应的是什么题、多大规模、怎么判分。看到不认识的先来这里对一遍，再判断那个分数可不可信。
>
> 链接一律给 arXiv abs 页或官方榜单页，失效了自己换成官方页。指标口径看 [[指标]]。

## 后训练

| 数据集 | 考什么 | 规模 | 判分 | 坑 |
| --- | --- | --- | --- | --- |
| [SearchQA](https://arxiv.org/abs/2610.39508v1) | 开放域搜索问答中的信息检索与答案生成 | 本文 300 train / 24 val / 1,400 test | exact match | 此链接为 SkillCompiler 使用口径；搜索后端、证据时效和答案归一化未固定时不可横比 |
| [OfficeQA](https://arxiv.org/abs/2610.39508v1) | 办公文档与流程相关的问答、信息抽取和操作 | 本文 120 train / 24 val / 102 test | exact match | 小测试集且操作可编译比例高；不能把收益直接外推到开放办公软件环境 |
| [DocVQA](https://arxiv.org/abs/2007.00398) | 文档图像中的视觉文字理解与问答 | 本文 180 train / 22 val / 201 test，属于论文使用子集 | exact match | OCR、页面渲染和答案规范化影响大；本文小子集不等于官方完整 leaderboard |
| [LiveMath](https://arxiv.org/abs/2610.39508v1) | 可验证数学推理与重复执行稳定性 | 本文 60 train / 17 val / 100 test | EM，同时报告 EIR@3 | 此链接为 SkillCompiler 使用口径；题量小，程序化步骤密度高，结果不代表所有数学题 |
| [Reasoning Gym](https://arxiv.org/abs/2610.41073v1) | 可程序生成并验证的逻辑、规则与游戏推理任务族 | 本文使用 knights-and-knaves、zebra puzzles、list functions、string manipulation 等；多项 held-out 为 512 | 各 task checker 的 exact correctness / held-out accuracy | 是任务生成框架而非单一固定测试集；难度、生成 seed、token budget 与 train/held-out 去重决定可比性 |
| [CriticBench](https://huggingface.co/datasets/opencompass/CriticBench) | 给定问题与候选回答，判断回答正确或错误并生成 critique | 本文固定 500 例，250 correct / 250 incorrect；每例采 5 次 | verdict accuracy、incorrect-class F1 | 本文评的是 verdict，不是 critique 文本质量；thinking 关闭，parser 与 subset 固定 |
| [LabBench2](https://arxiv.org/abs/2604.09554) | 有证据条件下的生物研究推理与实验问题 | 本文用 500 个 unique question families 的纯文本子集 | 各 task tag 用 exact-match 或 LLM judge，本文报 accuracy | 本文子集排除缺失 frozen evidence，非完整多模态 benchmark；judge 与 evidence snapshot 会影响分数 |
| [IHEval](https://arxiv.org/abs/2610.41296v1) | 指令层级下 reference、aligned、conflicting instructions 的遵循 | 本文实际 case 数见原文 Table 9，主文未逐项列出 | 官方 task-specific composite scores，三类分开报告 | aggregate 会掩盖 conflict 场景；与普通 instruction-following accuracy 不可直接混比 |
| [IFBench](https://arxiv.org/abs/2610.41296v1) | 可程序验证的复杂指令遵循与跨约束泛化 | 本文使用官方 prepared inputs；实际 case 数见 Table 9 | prompt-level loose / strict accuracy | loose 与 strict 差异大；thinking 关闭，prompt 解析和 verifier 版本必须固定 |
| [ScienceWorld](https://arxiv.org/abs/2203.07540) | 交互式文本环境中的多步科学推理与操作 | KEEPER 沿 STEP-HRL 使用 1,483 train、194 seen eval、211 unseen eval | average task score，包含部分完成，越高越好 | task score 不是纯成功率；seen/unseen、step budget 和 observation processing 必须一致 |
| [TREC Deep Learning Track](https://microsoft.github.io/msmarco/TREC-Deep-Learning.html) | 基于 MS MARCO passage collection 的高相关文档重排 | DecisionRank 用 DL19–DL23 五年评测；实际 query 数依年份而异 | nDCG@10；先逐年按 query 平均，再对五年非加权平均 | DL19–20 用 collection v1、DL21–23 用 v2；年份、候选生成器和 passage 截断长度必须一起报告 |
| [BEIR](https://arxiv.org/abs/2104.08663) | 跨领域零样本信息检索与重排 | DecisionRank 使用 7 个子集，非完整 BEIR 全套 | BM25 top-100 上的 nDCG@10，7 集非加权平均 | 子集、qrels 与候选检索器差异大；只报 BEIR 平均会掩盖单域退步 |
| [BRIGHT](https://arxiv.org/abs/2407.12883) | 需要领域知识和推理的检索任务 | 12 个任务；DecisionRank 重排 ReasonIR-8B top-100 | nDCG@10，12 任务非加权平均 | 检索阶段使用 GPT-4 reasoning queries，reranker 接收 original query；两种 query 不能混写 |
| [R2MED](https://aclanthology.org/2026.emnlp-main.2138/) | 推理驱动的医学信息检索 | 8 个医学数据集；DecisionRank 重排 E5-Mistral-7B-Instruct top-100 | nDCG@10，8 集非加权平均 | 本文的 out-of-domain 指超出蒸馏训练数据，不等于所有医学场景；候选器和 384-token 截断影响结果 |
| [MS MARCO](https://arxiv.org/abs/1611.09268) | 真实搜索 query 的 passage retrieval 与 reranking | DecisionRank 从 Rank-DistiLLM 使用 10,000 queries，不是原全集规模 | 本文用教师排序构造 RankNet supervision，评测另在 TREC DL | collection v1/v2、passage 与 document 任务不可混用；训练 query 数不能当原数据集规模 |
| [MiniGrid](https://arxiv.org/abs/2610.40030v1) | 部分可观察网格中的导航、线索记忆与长期决策 | 本文先用 Memory-S9，再 fine-tune Memory-S17Random；完整规模原文未给 | mean return、达到阈值所需 environment steps | 此链接为 pilot 使用口径；短 context 可被等待时间、位置编码和路径旁路绕过，须检查机制 |
| [Tulu 3 preference mixture](https://arxiv.org/abs/2411.15124) | 多领域偏好与后训练 prompt 来源 | RACE 去重后 271K prompts，非原数据全集规模 | RACE 只用 prompt，由冻结 judge 与生成 rubric 给 reward | preference mixture 的题目与偏好标签分开；未用标签不等于模型无预训练知识 |
| [ZebraLogic](https://arxiv.org/abs/2610.40096v1) | 约束满足与逻辑推理 | RACE 实际测试规模原文未给 | 依 OLMo3-Adapt suite 对应任务 evaluator | 链接为使用口径；需注明难度、split 和生成预算 |
| [AGI-Eval](https://arxiv.org/abs/2610.40096v1) | 综合考试式推理 | RACE 实际测试子集和题数原文未给 | 依 OLMo3-Adapt suite 对应 evaluator | 简称 AGI-E；不能把不同子集、语言和回答抽取方式混比 |
| [JudgeBench](https://arxiv.org/abs/2610.40096v1) | 用回答对检验 judge/rubric 区分能力 | RACE 实际题数原文未给 | 四域平均另于 Overall；分别用冻结 Qwen3-1.7B、GPT-5.4 | 链接为本文使用口径；不同 judge 的绝对分数不可直接归因于 rubric |
| [BrowseComp (original)](https://arxiv.org/abs/2504.12516) | 开放网页上的多跳浏览与难检索问答 | CORAL 实测前 150 题；此处不列原全集规模 | 本文 accuracy，LLM-as-a-Judge | 与已有 BrowseComp-Plus 离线语料版不同；前 150 题不是全榜单，搜索与浏览工具须一致 |
| [NuminaMath](https://arxiv.org/abs/2610.40350v1) | 数学推理训练示范 | CORAL 与 OpenMathReasoning 合取 241 trajectories；各自数量原文未给 | 此处作 SFT 训练数据，非独立评测集 | 链接为使用口径；不能把合计 241 当单集规模 |
| [OpenMathReasoning](https://arxiv.org/abs/2504.16891) | 数学推理训练数据 | CORAL 与 NuminaMath 合取 241 trajectories；原始全集规模此处未核实 | 用作数学 SFT 示范，下游按任务答案判分 | 数学示范不含 web 工具和记忆动作；勿把全集与抽样数量混用 |
| [SynLogic](https://arxiv.org/abs/2610.40576v1) | 可程序验证的逻辑游戏；PIVOT 用 Survo、Minesweeper、Kakurasu hard setting | PIVOT 每类训练 2k、测试 500；不是原全集规模 | SynLogic verifier；报告 avg@8、pass@8 | 此链接为本文使用口径；难度由网格、未知格数和约束密度控制，不能跨难度混比 |
| [Polaris-53K](https://arxiv.org/abs/2610.40576v1) | 数学推理训练题池 | PIVOT 与 Skywork math 合并筛选后取训练 30k、验证 1k、测试 2k；原全集精确规模原文未给 | 基模每题 8 次、成功至多 3 次后保留，Math-Verify 验证 | 此链接为本文使用口径；合并后 split 不能当 Polaris 原始划分；需对 AIME、MATH500 去重 |
| [CounterFact](https://huggingface.co/datasets/azhx/counterfact) | 事实关联编辑、真实对象与反事实替换 | 本文筛 train 前 3,000 records，主学生识别其中 2,443；不是全集规模 | true / counterfactual 概率排序及 held-out retention | counterfactual 是人为重写，不等同真实流行谣言；排序不等于自由问答 |
| [Skywork-OR1-RL-Data](https://arxiv.org/abs/2610.38681v1) | 数学推理训练 prompts | WISD 每域取 3,200 prompts、每题 4 条缓存回答；全集规模原文未给 | 该方法训练不使用参考解答或 reward | 此链接为使用口径来源；不能把训练子集规模当全集规模 |
| [KlearReasoner-CodeSub-15K](https://arxiv.org/abs/2610.38681v1) | 代码推理训练 prompts | WISD 取 3,200 prompts；全集实际规模原文未给 | 该方法训练仅用模型 contrast | 名称含 15K 不替代正式规模核实；与数学学生分开训练 |
| [HMMT25](https://www.hmmt.org/www/archive/282) | 2025 年 HMMT 数学竞赛推理 | WISD 评测 30 题 | Avg@32；原文 A.3 另写 64 samples，设置有冲突 | 年份与月份须明确；不能和 HMMT26 直接视为同一测试集 |
| [DeepMath-103K](https://arxiv.org/abs/2610.39020v1) | 竞赛数学训练；本文按教师单次答题对错分组 | 本文 both-correct、both-wrong 各 7,000 prompts | 按最终答案正确性筛选 | 单次答错不等于教师不能解题；此链接为使用和分组口径来源 |
| [BeyondAIME](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME) | 高中奥数之外的数学推理评测 | 本文实际测试题数原文未给 | avg@4；与另五竞赛取非加权平均 | 需注明版本、样本数和生成预算，不与 pass@4 混用 |
| [Winogrande](https://arxiv.org/abs/1907.10641) | 常识消歧与代词指代；OPST 用作未定向训练的 OOD 评测 | OPST 未列实际测试题数 | OPST 不开思考时按候选选项 log-likelihood 判分 | 需注明具体 split、归一化与提示协议；OOD 均分不能代替单项遗忘检查 |
| [OpenImages V4](https://arxiv.org/abs/1811.00982) | 开放图像集合；Visual OPST 仅使用无标注图片生成视觉问答 | OPST 未列实际抽取图片数 | 不是该文评分任务，效果由 VQAv2、POPE 评估 | 无标注不等于没有外部数据；需注明图片抽样与训练测试去重 |
| [VQAv2](https://arxiv.org/abs/1612.00837) | 对图像内容回答自然语言问题 | OPST 未列测试 split 与题数 | OPST 报 Acc.，越高越好；具体 evaluator 未明确 | 官方共识式评分与简单 exact match 不能默认等价；需注明答案归一化 |
| [POPE](https://arxiv.org/abs/2305.10355) | 用物体是否存在的问答检测视觉幻觉 | OPST 未列图像来源、采样子集与实际题数 | OPST 报 F1、Acc.、Rec.，越高越好 | 需注明采样策略与正负例口径；高 Recall 不能单独证明幻觉减少 |
| [ARPO-10k](https://arxiv.org/abs/2507.19849) | Agentic Reinforced Policy Optimization 相关训练集，本文用于 knowledge QA 蒸馏 | 本文称 ARPO-10k、使用 full split，精确样本数未列 | 按下游 GPQA-Diamond、ARC-Challenge、MMLU-Pro 评分 | 训练集不是测试榜单；应说明是否使用工具轨迹及预处理 |
| [EnvScaler](https://aclanthology.org/2026.findings-acl.407/) | 程序化合成的工具交互环境训练池 | EnvSched 使用 51 环境、2550 任务子池；原系统报告 191 环境、约 7000 scenarios | 各环境 terminal verifier，EnvSched reward 为 [0,1] | 全量池与论文实际子池不同；训练环境数量不是测试泛化范围 |
| [EvoAgentBench](https://arxiv.org/abs/2607.05202) | 多域 Agent self-evolution 与向未见任务迁移 | RuHarness 五域子划分见该笔记 Table 3，非原基准全量 | 各原 benchmark evaluator，报告 Pass@1 与五域 macro-average | evolution、validation、test 不能混用；运行失败的剔除需记录 |
| [GDP-Val / GDPval](https://arxiv.org/abs/2510.04374) | 真实经济价值任务与知识工作产物 | RuHarness 使用 evolution 60、validation 36、test 50 子集 | 按 EvoAgentBench 对应 evaluator，本文记为 Pass@1 | 子集结果不能当作 GDPval 官方全量成绩；评价与裁判口径须注明 |
| [ACEBench-Agent](https://aclanthology.org/2025.findings-emnlp.697/) | 多步、多轮 Agent 工具使用 | EnvSched 未列测试题数，报告 Multi-Step / Multi-Turn | 分子任务与 weighted Overall；用户模拟器须注明 | Agent 子集不能与 ACEBench 全套混比，模拟器版本影响结果 |
| [PubMedQA](https://arxiv.org/abs/1909.06146) | 生物医学研究问答；MPOPD 原文简称 PubMed | MPOPD 未列实际测试题数 | 本文报告 accuracy、avg@1 | 原始数据分区与实际使用子集需分清，不能把自动构建部分和人工标注测试混用 |
| [MedQA](https://doi.org/10.3390/app11146421) | 医学考试问答 | MPOPD 未列实际测试规模 | 本文按 MCQA 选项 accuracy | 需注明语言、版本、选项数与答案抽取规则 |
| [MedX / MedXpertQA](https://arxiv.org/abs/2501.18362) | 专家级医学推理与理解；MPOPD 简称 MedX | MPOPD 未列测试子集规模 | 本文报告 accuracy、avg@1 | 原文简称不等于另一套基准；模态和子集口径要核对 |
| [MMLU](https://arxiv.org/abs/2009.03300) | 57 个学科的知识选择题 | 约 1.4 万题，四选一 | 准确率 | 已饱和；原版有公认错题；5-shot 与 0-shot 不能混比 |
| [MMLU-Pro](https://arxiv.org/abs/2406.01574) | 同上去掉简单题，选项扩到 10 个 | 约 1.2 万题 | 准确率 | 和 MMLU 分数不可直接比较 |
| [MMLU-Redux](https://arxiv.org/abs/2406.04127) | MMLU 的人工清洗版 | 约 5.7k 题 | 准确率 | 题目更少，方差比 MMLU 大 |
| [C-Eval](https://arxiv.org/abs/2305.08322) | 中文多学科知识 | 约 1.4 万题 | 准确率 | 中文场景别直接套英文榜单结论 |
| [CMMLU](https://arxiv.org/abs/2306.09212) | 中文多任务语言理解 | 约 1.1 万题 | 准确率 | 和 C-Eval 有重叠，选一个即可 |
| [GSM8K](https://arxiv.org/abs/2110.14168) | 小学数学应用题，2 到 8 步 | 约 8.5k 题 | 精确匹配 | 已饱和；涨点常来自格式；答案抽取容易出错 |
| [MATH](https://arxiv.org/abs/2103.03874) | 竞赛数学，5 个难度档 | 约 1.25 万题 | 精确匹配 | 难度分布不均，只报总分会被简单题拉高 |
| [AIME](https://artofproblemsolving.com/wiki/index.php/AIME) | 美国数学邀请赛真题 | 每年 30 题，答案 0–999 | 准确率 | 一题 3.3 个百分点，方差极大；必须报 avg@k 或置信区间 |
| [GPQA](https://arxiv.org/abs/2311.12022) | 研究生级物理化学生物 | 约 450 题，Diamond 198 | 准确率 | 必须写清是不是 Diamond；选项位置敏感 |
| [MedMCQA](https://arxiv.org/abs/2203.14371) | 印度医学入学考试风格的多选医学题 | 约 19.4 万题；本文用 30 训练 / 30 测试 | 选项准确率（四选一） | 本文训练取 Pathology、测试取 Dental，跨子集比较不能当同分布；印度课程体系，与 MedQA 语料不同 |
| [SQuAD](https://arxiv.org/abs/1606.05250) | 给定段落做抽取式问答 | v1.1 约 10 万问句、536 段；本文取 70 题对 31 段（测试 25 题） | 答案匹配的 0/1 效用 | 本文用「先归档成 LoRA」的知识注入设置，段落不在作答上下文里，与标准开卷抽取不可比；测试 25 题，颗粒度粗 |
| [BBH](https://arxiv.org/abs/2210.09261) | BIG-Bench 里模型最差的 23 类任务 | 约 6.5k 条 | 准确率 | few-shot 排布能差十几分；不是生成任务 |
| [ARC](https://arxiv.org/abs/1803.05457) | 小学科学，Challenge 需组合常识 | Challenge 约 1.2k 题 | 准确率 | 接近饱和；Easy 子集几乎没有区分度 |
| [ARC（Abstraction and Reasoning Corpus）](https://arxiv.org/abs/1911.01547) | 抽象推理：从几个输入-输出网格示例归纳变换规则，再套到测试网格 | 原版训练 400 / 公开测试 400 / 隐藏测试 400；本文用 19 训练 / 6 dev / 14 测试 | 输出网格与标准答案逐格一致才记 1 分 | 与上一条 AI2 的 ARC 同名不同集，本文 Grid Puzzles 指的是这一个；规则要现推，检索帮不上 |
| [LSAT-AR](https://arxiv.org/abs/2304.06364) | AGIEval 里的 LSAT 分析推理题：分组、排序、条件约束 | 本文用 30 训练 / 30 测试，各含 16 条收集 + 14 条复用 | 选项匹配准确率 | 隶属 AGIEval 子集，报数须写明是 LSAT-AR 还是整套 AGIEval；30 题连续流为 TTRL 批量适应专门构造，不是原切分 |
| [HellaSwag](https://arxiv.org/abs/1905.07830) | 常识续写，四选一 | 约 1 万题 | 准确率 | 老基准，前沿模型 95 分以上，只作回归检查 |
| [TruthfulQA](https://arxiv.org/abs/2109.07958) | 模型会不会复述常见误解 | 817 题 | 准确率（需结合信息量） | 单看准确率会奖励「我不知道」式回避 |
| [HaluEval](https://arxiv.org/abs/2305.11747) | 幻觉识别 | 约 3.5 万条 | 准确率 | 判别式任务，不等于生成时不幻觉 |
| [IFEval](https://arxiv.org/abs/2311.07911) | 可程序化验证的指令约束 | 约 500 条，25 类 | 严格通过率 | 不测有用性；可以靠废话填充骗过字数约束 |
| [MT-Bench](https://arxiv.org/abs/2306.05685) | 多轮对话综合质量 | 80 题，8 类 | 裁判打 1–10 分 | 裁判偏好直接影响分数；换裁判排名会变 |
| [AlpacaEval 2.0](https://arxiv.org/abs/2404.04475) | 通用指令跟随 | 约 800 条 | 胜率，有 LC 版 | 不控长度可以靠写长刷分，务必看 LC |
| [Arena-Hard](https://arxiv.org/abs/2406.11939) | 高难度 prompt | 500 条 | 对基线胜率 | 仍是裁判打分，与人类偏好只是相关 |
| [LMArena](https://lmarena.ai) | 真人盲选开放式对话 | 私有，规模不公开 | Elo | 用户群体有偏；分数不可复现；风格影响大 |
| [HumanEval](https://arxiv.org/abs/2107.03374) | 函数级代码补全 | 164 题 | pass@1 | 污染严重；题目太短，测不出工程能力 |
| [MBPP](https://arxiv.org/abs/2108.07732) | 入门级代码题 | 约 1k 题 | pass@1 | 同污染；要用 sanitized 子集 |
| [LiveCodeBench](https://arxiv.org/abs/2403.07974) | 按发布时间切片的竞赛题 | 持续更新 | pass@1 | 必须写明时间窗口，否则不可比 |
| [RewardBench](https://arxiv.org/abs/2403.13787) | 奖励模型的偏好判别 | 数千对比对，四类 | 准确率 | 测判别不测生成；RM 分高不等于 RL 效果好 |
| [HELM](https://arxiv.org/abs/2211.09110) | 多维度统一评测框架 | 覆盖几十个场景 | 各场景分别计分 | 框架而非单一基准，报「HELM 分」没有意义 |
| [SimpleQA](https://arxiv.org/abs/2411.04368) | 短答案事实性问答 | 约 4k 题 | 正确率（含拒答） | 专为抗幻觉设计，高分依赖肯说不知道 |
| [MLE-bench](https://arxiv.org/abs/2410.07095) | 机器学习工程任务 | 75 个 Kaggle 竞赛 | 相对奖牌线 | 成本极高，单次运行动辄几小时 |
| [Super-Natural Instructions](https://arxiv.org/abs/2204.07705) | 一千多个 NLP 任务的指令遵循，76 类 | 1,616 个任务 | 各任务自带指标，多为准确率或精确匹配 | 难度极不均，大量原始 NLP 任务已饱和；论文常只取子集（有用 875 个），报数要看清 |
| [Omni-MATH-2](https://arxiv.org/abs/2601.19532) | 奥数级数学题，比 MATH 更难更杂 | 原文未给 | 精确匹配 | 原文引的是 [3] 那篇讲「基准被裁判拖垮」的文章，不是数据集本体页；规模别拿 Omni-MATH 的数照搬 |
| [Apex Shortlist](https://openreview.net/forum?id=DmPE4byHuN) | MathArena 里的专家级竞赛短名单 | 原文未给 | 精确匹配 | 只是 MathArena 的一个子集，成绩不能和完整 MATH 混比；题量小，方差大 |
| [StrategyQA](https://arxiv.org/abs/2101.02235) | 需要多跳隐含推理的 yes/no 问题 | 约 2,780 题 | 准确率 | 二分类，瞎猜就有 50%，分差被压扁，看差值别看绝对值 |
| [Social IQA](https://aclanthology.org/D19-1454/) | 社交场景下的常识三选一 | 测试集约 2,055 题 | 准确率 | 三选一，随机基线 33%；句子短，模型容易靠模式匹配刷分 |
| [SuperGPQA](https://arxiv.org/abs/2502.12148) | 285 个研究生学科的知识选择题 | 约 2.65 万题 | 准确率 | 学科分布极不均衡，只报总分会掩盖长尾学科 |
| [NQ](https://ai.google.com/research/NaturalQuestions) | 真实搜索问题上的开放域问答 | 约 30 万条 | 精确匹配 | 原始集没有短答案，开放域版本要自己抽答案，抽取规则不同分数差很大 |
| [TriviaQA](https://aclanthology.org/P17-1147/) | 冷知识问答，带证据文档 | 约 9.5 万题 | 精确匹配 | 检索版和开放书版分数差一大截，报分必须写明是哪个设置 |
| [HotpotQA](https://aclanthology.org/D18-1259/) | 需要两跳推理的多跳问答 | 约 11.3 万题 | 精确匹配 / F1 | 干扰段落版与全维基版不可比；别名和缩写会被判错 |
| [2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.580/) | 维基多跳，强调推理步骤可解释 | 约 19 万题 | 精确匹配 / F1 | 模板生成，实体一换名字性能就掉，容易高估真实推理能力 |
| [PopQA](https://aclanthology.org/2023.acl-long.546/) | 按实体流行度分层的开放域问答 | 约 1.4 万题 | 准确率 | 长尾实体占大头，只看热门实体会虚高；分层结果比总平均更有信息 |
| [MuSiQue](https://arxiv.org/abs/2108.00573) | 组合式多跳问答，2 到 4 跳 | 约 2.5 万题 | 精确匹配 / F1 | 每题短但跳数多，单跳检索型模型掉得厉害 |
| [Bamboogle](https://arxiv.org/abs/2210.03350) | 需要两跳组合推理的问答，专挑模型必须串联两条事实的题 | 125 题 | 精确匹配 | 题量极小，一题约合 0.8 个百分点，单次结果方差大；句子是人造的组合题，分布和 HotpotQA 的真实语料不同 |
| [ChartQAPro](https://aclanthology.org/2025.findings-acl.978/) | 真实图表上的问答 | 约 1,950 题 | 准确率 | 必须看图，纯文本模型只能拿到占位符；图表类型分布和 ChartQA 不同 |
| [MMMU](https://arxiv.org/abs/2311.16502) | 大学级多学科多模态理解 | 约 1.15 万题 | 准确率 | 30 个学科不均衡；不支持图像的模型只能瞎猜，和视觉模型不可直接比 |
| [MMMU-Pro](https://arxiv.org/abs/2409.02813) | MMMU 的加固版，选项更接近更干扰 | 原文未给 | 准确率 | 和 MMMU 不可直接比；纯文本版与视觉版是两个不同设置，别混报 |
| [AMC23](https://artofproblemsolving.com/wiki/index.php/2023_AMC_12A_Problems) | 2023 年 AMC 12A / 12B 竞赛题 | 40 题 | 答案匹配（整数） | 题量极小，一题 2.5 个百分点，必须报 avg@k；和 AIME 难度差一档，别混着讲 |
| [OlympiadBench](https://aclanthology.org/2024.acl-long.211.pdf) | 奥赛级数学与物理题，中英双语，含多模态 | 8,476 题 | 答案匹配；证明题需裁判 | 中英双语两套，报分要写清用哪一版；证明题的判分口径各论文不统一 |
| [Minerva](https://arxiv.org/abs/2206.10555) | 从 MATH 里抽出的竞赛数学，要求写出推理过程 | 约 270 题 | 规则匹配 + 数值/符号等价判断 | 数值近似和等价表达式让规则判分不稳，S3AC 这篇专门改用 LLM judge，分数与规则判分不可比 |
| [IMO-AnswerBench](https://arxiv.org/abs/2511.01846) | 从各国与国际奥赛里挑的题，只要短答案 | 400 题（代数、组合、几何、数论四类） | 可验证的短答案匹配 | 只考答案不考证明，测不到证明能力；题源跨年，难度分布不均 |
| [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision) | 竞赛级多模态数学题 | 约 3k 题 | 答案匹配 | 图里的公式靠 OCR，抽取错就判错；难度方差大 |
| [MathVista](https://arxiv.org/abs/2310.02255) | 视觉场景下的数学推理 | 约 6.1k 题，testmini 约 1k | 准确率 | testmini 与全集不可混比；题型杂，单项分掩盖分布 |
| [HMMT26](https://www.hmmt.org/) | HMMT 高中数学竞赛 2026 年场次 | 约 30 题 | 答案匹配 | 每年题目不同，跨年份分数不可比；30 题量方差极大 |
| [DAPO-Math 7.5k](https://arxiv.org/abs/2503.14476) | 竞赛数学训练集，**训练用不是评测用** | 7.5k prompts（DAPO-Math 的子集） | 规则匹配数学答案 | 从 17k 里怎么切出 7.5k，原文未给；拿它训出来的分不能跨论文直接比 |
| [DeonticBench](https://arxiv.org/abs/2604.04443) | 从真实法条与案情推出义务、许可、禁止；给定一条断言判断其成立与否 | 6,232 题，覆盖美国联邦税、航空行李政策、移民行政、州住房法四域 | 提交标签与参考标签一致的准确率（附录 G 的 trace 里标签形如 Entailment / Contradiction） | 本文只用 AIRLINE、SARA-NUMERIC、SARA-BINARY 三个子集；held-out 只有 150 / 50 / 138 题，Hard 71 题与三者重叠，子集分数方差大 |
| [LocBench](https://arxiv.org/abs/2503.09089) | 只给 GitHub issue，让 agent 找出 gold patch 动过的文件与函数 | 560 个真实 issue（242 bug、150 feature、139 performance、29 security），来自 transformers、scikit-learn、vllm、ultralytics 等仓库 | file 级与 function 级 Acc@k，全部 gold 位置都进前 k 才算 1（新增函数不计） | 「全中才算 1」不是命中率；file 级与 function 级是两套分；本文 held-out 仅 87 个配对 query，波动明显 |
| [MMLU-ProX](https://arxiv.org/abs/2503.10497) | 多选题知识问答，跨语言跨学科 | 14 学科 × 29 语言 | 准确率，也有工作改用 NLL | 语言间难度差极大，必须写清留出了哪些语言与学科；与 MMLU 分数不可比 |
| [SkillsBench](https://arxiv.org/abs/2602.12670) | 任务型 agent 技能基准：每题配一份人写技能，衡量技能到底帮不帮得上忙 | 本文用 83 题、8 个域（软件工程 14、网络安全 7、自然科学 14、金融经济 9、办公 14、媒体 9、工业与物理系统 11、数学与形式推理 5） | 每题由 verifier 给 reward，5 次试次里有正 reward 记为过，报 pass rate（avg@5） | 域间题量极不均，数学只有 5 题，一题约合 20 个点；人写技能的成绩随模型浮动（GPT-6 Luna 56.4、GPT-5.6 Luna 61.2），换模型名次可能翻 |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | 给 PyTorch 参考实现，要求写出结果相同但更快的 GPU kernel | 本文只取 3 个 level-3 问题（deep narrow MLP、ShuffleNet、一个 GRU），三题共用一份技能 | correctness-weighted speedup：实现不正确记 0，正确才算相对 PyTorch 的加速比 | 只 3 题，加速比绑 A100 实测运行时，换 GPU 不可比；GRU 一档三种方法都停在 1.0 附近，说明参考实现已经够紧 |
| [HiMath](https://arxiv.org/abs/2610.39934) | 新定义题（New Definition questions）：题干先给新概念，再连问三个递进小问，考从第一性原理推理 | 151 题；训练 105 / 测试 46（约 7:3，题目级切分，同题三小问不跨集） | Q3 多为证明题，由 LLM 裁判（主裁判 o3-mini）对照人写参考解判 $c_i$，另给 1–5 的过程分 $p_i$；不用字符串匹配 | 作者自建，代码与数据集声明「录用后公开」，目前没有独立下载页；46 题测试集一题约 2.17 个百分点，方差大；分数依赖裁判，换裁判不可直接比 |
| [ZeroBench](https://arxiv.org/abs/2502.09696) | 当代多模态模型基本做不出的极端视觉题，定位是 frontier-unsolvable 压力测试 | 本文未列题数与子集划分 | accuracy（%）；本文未给答案抽取与判分器口径 | 分数接近地板，绝对差几个点意义有限；本文 TACE 该项 18.86，低于不训练的 Base VLM 19.76 |
| [WeMath](https://arxiv.org/abs/2407.01284) | 细粒度数学推理，按知识点与解题步骤分层诊断 | 本文未列全量；敏感性实验用的是 262-example WeMath choice-CoT pilot | accuracy（%）；本文按选项式（choice-CoT）作答 | choice 口径与自由作答不可混比；本文 TACE 该项 45.62，比 Base VLM 的 54.10 低 8.48 个点，是全文最大回归 |
| [ScienceBoard](https://openreview.net/forum?id=bJvwJahJeF) | 科学软件里的多步桌面任务 | 本文在线评测 169 个 episode | 官方 evaluator 检查科学应用终态 | 只在在线设置评测；actor success 23.7%，样本偏难；具体子集与官方全量口径需核对 |
| [OSReward](https://arxiv.org/abs/2607.28609) | 跨平台 CUA 轨迹完成度，带人工标签 | 1,019 条轨迹 | 人工标签，success 为正类 | 标签是人工判断，不是程序真值；本文另用其 hard 子集 |
| [OSReward-Hard](https://arxiv.org/abs/2607.28609) | OSReward 中完成度难判的 hard 子集 | 284 条 | 人工标签 | 是 OSReward 的子集，不能与全量成绩混报；只覆盖难例 |
| [Bi-TSP](https://arxiv.org/abs/2610.40023) | 双目标旅行商；两个目标都是路径长度，用不同边权 | 测试 n ∈ {20, 50, 100}，每尺寸 50 个固定实例；设计用 20 城 10 个实例 | HV↑、IGD↓、Time↓；HV 用固定参考点 (20,20)/(35,35)/(65,65) | 自建合成实例，非 TSPLIB；设计尺寸与测试尺寸不同，跨尺寸结果不能当同分布比较 |
| [Tri-TSP](https://arxiv.org/abs/2610.40023) | 三目标旅行商；三条路径长度 | 测试 n ∈ {20, 50, 100}，每尺寸 50 个实例；设计同 Bi-TSP | HV↑、IGD↓；参考点 (20,20,20) 等 | 只作为 Bi-TSP 的同族迁移目标，不在迁移前参与 harness 自进化 |
| [Bi-CVRP](https://arxiv.org/abs/2610.40023) | 双目标容量约束车辆路径；总距离与最长单条路线 | 测试 n ∈ {20, 50, 100}，每尺寸 50 个实例；设计 50 客户 10 个实例 | HV↑、IGD↓；参考点 (30,8)/(45,8)/(80,8) | 目标含 makespan 型分量，与纯距离口径的 CVRP 不可直接横比 |
| [Bi-SPDP](https://arxiv.org/abs/2610.40023) | 双目标选择性取送货；总距离与所需车辆容量 | 测试 n ∈ {20, 50, 100}，每尺寸 50 个实例；设计 20 节点（含 depot）20 个实例 | HV↑、IGD↓；参考点随尺寸放大 | 是 Bi-CVRP 的迁移目标；容量目标口径与 CVRP 不同，别混用同一参考点 |
| [Bi-KP](https://arxiv.org/abs/2610.40023) | 双目标 0-1 背包；两个目标各是一组收益 | 测试 n ∈ {50, 100, 200}，每尺寸 50 个实例；设计 50 项 10 个实例 | HV↑、IGD↓；参考点 (-5,-5)/(-20,-20)/(-30,-30) | 两个骨干的系数尺度不同：SEMO 用 Unif[0,1) 连续值，NSGA-II 用 Unif{1..100} 整数 |
| [Bi-MDKP](https://arxiv.org/abs/2610.40023) | 双目标多维背包；5 个资源约束，松紧系数 τ = 0.5 | 测试 n ∈ {50, 100, 200}，每尺寸 50 个实例；设计 50 项 10 个实例 | HV↑、IGD↓ | 是 Bi-KP 的迁移目标；约束数 5 与 τ 必须注明，改了就不可比 |
| [Bi-FJSP](https://arxiv.org/abs/2610.40023) | 双目标柔性作业车间调度；makespan 与最大机器负载 | 测试 Mk01–Mk15；设计 NSGA-II 用 Mk15 一个实例、SEMO 用 15 个合成实例 | HV↑、IGD↓；目标先除以实例尺度 $S_I$ 归一化，参考点 (1.1,1.1) | 目标值经实例专属尺度归一化，绝对 HV 不能跨实例比；ID 与 OOD 要分开报 |
| [Tri-FJSP](https://arxiv.org/abs/2610.40023) | 三目标 FJSP；再加总机器负载 | 测试 Mk01–Mk15 | HV↑、IGD↓；参考点 (1.1,1.1,1.1) | 是 Bi-FJSP 的迁移目标；维数从 2 变 3，HV 体积口径随之变，别与 Bi-FJSP 横比 |
| [Brandimarte Mk01–Mk15](https://arxiv.org/abs/2610.40023) | 柔性作业车间调度的经典 15 个实例，本文用作 FJSP 的 OOD 测试 | 15 个实例；Bi-FJSP 与 Tri-FJSP 共用 | HV↑、IGD↓ | 仅 Mk15 参与算子设计，其余 14 个只在测试出现；引用时须写明「ID=Mk15，OOD=Mk01–Mk14」 |
| [模运算任务族（modular arithmetic）](https://arxiv.org/abs/2610.31642) | 学模意义下的二元运算规则，看网络先记住训练集、很久之后才泛化的延迟跃变 | 本文用 $a+b \bmod 31$、$a+b \bmod 113$、$a^2+b^2 \bmod 97$、$a^3+b \bmod 97$、$a+b \bmod 53$；只有模 31 与模 53 明确写了 50% 训练切分 | 训练后全测试集的 test accuracy；训练一律 full-batch | 模数不同就是不同任务，不可混比；横轴有的按 epoch 有的按 step；任务简单不等于好复现，训练预算与种子数要一起报 |

## Agent

| 数据集 | 考什么 | 规模 | 判分 | 坑 |
| --- | --- | --- | --- | --- |
| [SWE-bench](https://arxiv.org/abs/2310.06770) | 真实仓库的 issue 修复 | 原集约 2.3k，Verified 500 | 隐藏测试全通过 | 测试通过 ≠ 补丁质量好；环境与成本差异大；有污染风险 |
| [SWE-rebench](https://arxiv.org/abs/2505.20411) | 自动流水线采集的 SWE 任务，含去污染评测口径 | 本文与 SWE-rebench V2、Scale-SWE 合并训练，三源合计 2,484 题；各源贡献原文未给 | 可执行测试判 resolve rate，作为训练数据时不单独评分 | 合并总数不能当单集规模；去污染口径随时间更新，旧版任务可能与新基准重叠 |
| [SWE-rebench V2](https://arxiv.org/abs/2602.23866) | 语言无关的大规模 SWE 任务采集 | 同上，三源合计 2,484 题，本文未拆分各源题数 | 可执行测试判 resolve rate | 与 V1 的口径差异原文未展开；不要把 V1/V2 混成同一个集 |
| [Scale-SWE](https://arxiv.org/abs/2602.09892) | 面向 GitHub 规模化的编码 agent 任务池 | 同上，三源合计 2,484 题 | 可执行测试判 resolve rate | 与另两源如何配比未报；任务难度分布可能偏斜 |
| [WebArena](https://arxiv.org/abs/2307.13854) | 自建真实网站上的浏览器任务 | 812 个任务 | 终态程序化检查 | 给 DOM 还是给截图，成绩差一截；站点是静态快照 |
| [Mind2Web](https://arxiv.org/abs/2306.06070) | 跨网站的一般化网页操作 | 2k+ 任务，上百网站 | 元素定位 / 步骤成功率 | 定位对了不等于任务完成 |
| [WebVoyager](https://arxiv.org/abs/2401.13919) | 真实公网网站上的端到端任务 | 600+ 任务 | 结果判定（含人工） | 用真实网站，不可复现；页面改版就失效 |
| [OSWorld](https://arxiv.org/abs/2404.07972) | 真实桌面跨应用操作 | 数百任务 | 脚本验收最终状态 | 纯视觉输入极难；一条任务几分钟，全量很贵 |
| [AndroidWorld](https://arxiv.org/abs/2405.14573) | 安卓设备上的 GUI 操作 | 116 个任务，20 个应用 | 终态检查 | 模拟器版本敏感；跨应用任务少 |
| [GAIA](https://arxiv.org/abs/2311.12983) | 需要检索与工具组合的现实问题 | 400 多题，三档 | 精确字符串匹配 | 格式错就判错；依赖实时网页，结果会漂移 |
| [τ-bench](https://arxiv.org/abs/2406.12045) | 客服对话，遵守政策并调工具 | 零售、航空两域 | pass^k（k 次全过） | 用户由模型模拟，换模拟器成绩就变 |
| [AgentBench](https://arxiv.org/abs/2308.03688) | 8 个异构环境 | 各几十到几百 | 各环境成功率再归一 | 不同环境难度不可比，总分意义有限 |
| [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) | 函数调用的选择与参数填写 | 数千用例，多语言签名 | 分类准确率 | 版本迭代快；格式对但语义错可能仍判对 |
| [ALFWorld](https://arxiv.org/abs/2010.03768) | 文本家居环境的长程家务，六类（Pick / Look / Clean / Heat / Cool / Pick2） | seen 140 题，unseen 134 题（换房间布局） | 任务完成率（exact success），50 回合上限 | Heat / Cool 两族极难，base 只有 0–8；回合上限一改成功率就变；unseen 只换布局不换任务模板 |
| [WebShop](https://arxiv.org/abs/2207.01206) | 模拟网店里按描述搜索并买下匹配商品 | 常见 500 题一条流 | Strict-SR（任务分 = 1）/ Mean-Score（任务分均值） | 任务分是 0–1 连续值，Strict 口径很严；商品页是离线快照，页面改版不影响但也不反映真实网页 |
| [AppWorld](https://arxiv.org/abs/2407.18901) | 跨模拟 App 的交互式代码生成 | Test-Normal 168 题 / 56 场景，Test-Challenge 417 题 / 139 场景 | TGC（通过全部状态测试）/ SGC（场景内全过） | 要真的执行代码；SGC 比 TGC 严得多，Challenge 上普遍接近 0；30 次交互预算 |

## 具身操作

| 数据集 | 考什么 | 规模 | 判分 | 坑 |
| --- | --- | --- | --- | --- |
| [LIBERO](https://arxiv.org/abs/2306.03310) | 机械臂桌面操作，分 Spatial / Object / Goal / Long 四个套件，考知识迁移 | 130 条任务分四套；长程子集 LIBERO-LONG 只有 10 条 | 成功率，一题多次试验取比例（AtomicRL 每任务 50 次） | 纯仿真不测真机；LIBERO-LONG 上强策略已到 95% 以上，区分度变小；任务少，单任务一题约合 2 个百分点 |
| [CALVIN](https://arxiv.org/abs/2112.03227) | 语言指令引导的长程操作，看能连续完成几个子任务 | ABC-D 协议：A/B/C 训练、D 测试；常见评估 1000 条指令链 | 连续完成 1–5 个的成功率 + 平均完成长度 Avg. Len | 只看仿真；ABC-D 是固定切分，换协议数字不可比；长链分数对初始策略极敏感 |

## 长上下文与长推理

| 数据集                                                     | 考什么              | 规模                    | 判分       | 坑                                         |
| ------------------------------------------------------- | ---------------- | --------------------- | -------- | ----------------------------------------- |
| [S-NIAH](https://arxiv.org/abs/2404.06654)（RULER 单针）    | 在一大堆无关文本里找一句指定的话 | 论文用 50 个任务，长度可扩到 1M   | 精确匹配     | 信息量恒定，前沿模型能刷满，测不出稠密长上下文能力                 |
| [∞Bench（LongBook）](https://arxiv.org/abs/2402.13718)      | 书级长文问答，要跨章节找细节并综合 | 原版含多种长文任务；本文只取 LongBook，29 题对 25 本书（测试 10 题） | 答案匹配的 0/1 效用 | 与 Needle 式检索不同，LongBook 要读整本书；本文测试仅 10 题，一题就是 10 个百分点 |
| [BrowseComp-Plus](https://arxiv.org/abs/2508.06600)     | 离线语料上的多跳深研究问答    | 1000 篇文档，常用 100–150 题 | 答案正确率    | 输入 6M–11M token，原始模型根本塞不进；给几篇文档、给多少预算都要写清 |
| [OOLONG](https://arxiv.org/abs/2511.02817)              | 逐条语义打标签再全局聚合     | trec_coarse 50 题，131K | 正确率      | 答案依赖几乎每一行，模型硬读在 131K 就开始崩                 |
| OOLONG-Pairs                                            | 列出所有满足条件的用户 ID 对 | 20 题，32K–1M           | F1（按对儿算） | 复杂度平方，输出上万对，输出长度本身就是瓶颈                    |
| [LongBench-v2](https://arxiv.org/abs/2412.15204) CodeQA | 多文件代码仓库理解，多选题    | 23K–4.2M token        | 准确率      | 单个文件就可能超窗口，不切分无法做                         |
| [LongMemEval-S](https://arxiv.org/abs/2410.10813)       | 多 session 长对话里的长期记忆问答，6 个子任务 | 500 题，每题约 40 个 session、约 115K token | 逐子任务 accuracy（官方协议） | Avg. 按题加权，不是子任务算术平均；Single-session Preference 只有 30 题，一题就是 3.33 pp；子任务间难度差极大（约 3% 到约 78%） |
| [LoCoMo](https://arxiv.org/abs/2402.17753)              | 超长期对话的记忆与推理，5 类查询 | 公开十段对话，约 1,980 题（附录写 1,986），19–32 个 session  | 按 query type 报 overall score | Avg. 按题加权；判分口径（裁判模型与规则）多数论文不写，跨论文比分会踩口径；Adversarial 类常接近 0 分，拉低总体 |
| MemoryAgentBench | 增量多轮交互下的记忆智能体能力：Accurate Retrieval、Test-Time Learning、Long-Range Understanding | 常用九任务子集（SH-Doc、MH-Doc、LME(S)、TREC-C、NLU、TREC-F、CLINC、Banking、InfBench）；各任务题数原文未给 | 每任务用自带指标，另有九任务非加权平均（74.7 即九格算术平均） | 九任务难度极不均衡（InfBench 二十几分、SH-Doc 九十），平均分被个别任务拖；常用子集不等于全量榜单，跨论文需核对选了哪九个 |
| [PersonaMem](https://arxiv.org/abs/2504.14225)          | 长期交互中的动态用户画像：回忆事实、泛化新场景、追踪偏好演化、解释偏好变化原因、按当前画像推荐 | 五类查询（Rec.、Gen.、Evol.、Reason、Reco.），实际题数原文未给 | 五个维度分 + 总体分，按题加权 | 总体分不是五维算术平均（本文 71.9/68.4/62.6/82.8/54.6 算平均是 68.06，报的是 69.2），权重未公布；Reco. 类普遍最低，单看总体会掩盖 |
| [LongCite-45k](https://arxiv.org/abs/2409.02897)        | 长上下文问答 + 句子级引用标注，常被拿来当持续预训练语料 | 约 44,600 条，最长 128K token | 本身不是评测集，按下游基准评 | 中英混合，用之前必须说自己滤了哪些（语言、长度区间）；滤完的实际 token 量和原表对不上 |

## 选数据集的三条经验

1. **先看饱和没**：MMLU 88 分以上、GSM8K 95 分以上，再差 1 分基本是噪声，别拿它证明后训练有效
2. **先看判分口径**：同名的 success rate 有四种判法（隐藏测试、终态检查、脚本验收、字符串匹配），跨基准的数字不能直接比
3. **先看污染**：代码和数学的经典题在训练语料里到处都是，LiveCodeBench 这种按时间切片的才有抗污染性

## 维护规则

- 一行一个数据集，不按论文分；新基准直接加行，链接放名字上
- 规模不确定写「约」，判分口径不明就写「口径未统一」
- 改完更新 frontmatter 的 `updated`
