# Slide Outline

## Slide 1
### Title
论文信息与问题背景
### Bullets
- 论文：Exact Combinatorial Optimization with Graph Convolutional Neural Networks
- 作者：Maxime Gasse 等，NeurIPS 2019
- 任务场景：MILP 的 branch-and-bound 变量选择
- 痛点：strong branching 决策质量高，但每一步代价很大
### Suggested Figure
- 论文首页标题截图，或 branch-and-bound 搜索树示意图
### Speaker Notes
- 这篇论文不直接“用神经网络解 MILP”，而是学习求解器里的一个关键决策环节：branching variable selection。
- 先把问题框定清楚，后面就不容易把它误解成端到端优化器。

## Slide 2
### Title
研究问题
### Bullets
- 能否学习一个 branching policy，近似 strong branching 的决策质量
- 同时减少人工特征工程，直接利用 MILP 的图结构
- 还要能泛化到比训练时更大的实例
- 最终评价不是分类精度本身，而是 solve time 和 B&B nodes
### Suggested Figure
- 一个“高质量但慢”的 strong branching 与“快速但弱”的 heuristic 对比图
### Speaker Notes
- 作者想解决的是 exact solver 里的效率问题，而不是近似求解质量问题。
- 所以评估指标要始终盯住 solver performance，而不是只看 ML accuracy。

## Slide 3
### Title
核心思路：把 branching 变成图上的动作分类
### Bullets
- 将当前 B&B 节点状态编码为变量-约束二部图
- 约束节点、变量节点、边特征共同描述当前 LP relaxation
- 用 GCNN 读取图状态，输出候选 branching variables 上的概率分布
- 用 imitation learning 直接学习专家动作，而不是回归分数或排序
### Suggested Figure
- Figure 2 左右合并展示：二部图状态表示 + GCNN 架构
### Speaker Notes
- 这一页讲最核心的一句话：state 是图，action 是“选哪个变量分支”。
- 他们把以往 ranking/regression 风格的问题，改成了 action classification。

## Slide 4
### Title
训练标签与数据构造
### Bullets
- 标签来自 strong branching 专家的离散动作
- 数据是 branch-and-bound 过程中采样的 state-action pairs
- 四类 benchmark：set covering、combinatorial auction、facility location、maximum independent set
- 每个 benchmark 约 100k 训练样本、20k 验证样本
- 为避免 SCIP strong branching 的副作用，作者重写了 `vanillafullstrong`
### Suggested Figure
- 数据采集流程图：实例生成 -> SCIP 运行 -> 记录状态与专家动作
### Speaker Notes
- 这里要强调：label 不是最终最优解，也不是 tree size，而是“当前节点该 branch 哪个变量”。
- `vanillafullstrong` 这个细节很关键，说明作者认真处理了 oracle side-effects。

## Slide 5
### Title
网络结构
### Bullets
- 输入是二部图 `(G, C, E, V)`：图结构、约束特征、边特征、变量特征
- 采用一次 graph convolution，但分成 variable->constraint 和 constraint->variable 两个 half-convolutions
- 每个消息/更新函数是 2-layer MLP
- 最后只保留变量节点表示，用 2-layer perceptron 输出 branching scores
- 主模型采用 sum aggregation + prenorm
### Suggested Figure
- Figure 2 架构图，特别标出两段 half-convolution 与 final MLP
### Speaker Notes
- 这页别讲太多公式，重点让听众知道信息是在变量和约束之间传递的。
- `sum + prenorm` 后面会在 ablation 和 takeaway 里回收。

## Slide 6
### Title
Output / Loss / Inference 是怎么对齐的
### Bullets
- Output：masked softmax over candidate branching variables
- Loss：cross-entropy，最大化专家动作的概率
- Inference：在 SCIP 中用模型分数选择当前节点的 branching variable
- 局部动作空间是对齐的：学什么，推理时就用什么
- 但全局目标仍有偏差：训练学 imitation，评估看 solve time / nodes
### Suggested Figure
- 一个三段式流程图：state -> action distribution -> branch decision -> solver continues
### Speaker Notes
- 这是整篇论文最容易讲清楚、也最值得讲清楚的一页。
- 本地 action-level alignment 很好，但系统级 objective 仍然是 surrogate learning。

## Slide 7
### Title
实验结果与真正的新意
### Bullets
- GCNN 在四类问题上的 imitation accuracy 普遍优于 TREES、SVMRANK、LMART
- 在多数设置下，solve time 和 B&B nodes 也优于 SCIP 默认 branching rule
- 能泛化到比训练更大的实例，这是作者最强调的实验结论
- 真正的新意不只是“模仿 strong branching”，而是图表示 + GCNN + action classification + full solver evaluation
### Suggested Figure
- Table 1 和 Table 2 的裁剪版，突出 GCNN 与 RPB/LMART 的比较
### Speaker Notes
- 建议把“不是 first imitation-learning paper”这句话明确说出来，能显得判断更稳。
- 创新点要落在表示、训练 formulation 和实验 setting 上，而不是笼统说“用了 GNN”。

## Slide 8
### Title
优点、局限与我的 takeaway
### Bullets
- 优点：减少特征工程，decision quality / inference cost tradeoff 做得合理
- 优点：在完整 SCIP 流程中验证，而不是简化 solver 环境
- 局限：只观察到部分 solver state，问题本质上是 POMDP
- 局限：泛化不是无限的，实例继续变大时性能会下滑
- Takeaway：这是“learned branching in exact solvers”的代表作，不是端到端 neural optimizer
### Suggested Figure
- 一张 summary 表：Strengths / Weaknesses / What to remember
### Speaker Notes
- 结束时把听众的认知钉住：网络学的是 brancher，不是整个 solver。
- 如果后续要延展阅读，可以顺着 reinforcement learning branching 或更强的 downstream objectives 往后看。
