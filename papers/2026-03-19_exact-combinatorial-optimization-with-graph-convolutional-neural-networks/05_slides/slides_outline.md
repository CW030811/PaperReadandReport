# Slide Outline

## Slide 1
### Title
论文定位与问题背景
### Bullets
- 论文：Exact Combinatorial Optimization with Graph Convolutional Neural Networks，NeurIPS 2019。
- 问题场景：MILP 求解中的 branch-and-bound variable selection。
- 核心背景：strong branching 决策质量高，但每个节点在线计算代价很大。
- 论文定位：不是端到端 neural solver，而是把 learned brancher 嵌回 exact MILP solver。
### Suggested Figure
- 一张定位图：MILP -> B&B solver -> branching module -> learned GCNN brancher。
### Speaker Notes
- 开场先把边界讲清，避免听众把论文误听成“GNN 直接解组合优化”。
- 强调这是 exact solver 中一个局部决策的学习问题，不是替代整个 SCIP。

## Slide 2
### Title
Research Question
### Bullets
- 能否学出一个 branching policy，逼近 strong branching 的决策质量，同时显著降低每步代价？
- 能否避免大量 hand-crafted solver features，直接利用 MILP 的变量-约束结构？
- 能否在只看较小训练实例的情况下，泛化到更大规模的 MILP 实例？
- 真正评价目标不是 imitation accuracy 本身，而是 solve time、B&B nodes 和真实 solver 中的 wins。
### Suggested Figure
- 一个“质量 vs 代价”二维图：FSB、默认 heuristic、learned GCNN policy。
### Speaker Notes
- 这一页要把 research question 说完整：近似 strong branching、减少特征工程、跨规模泛化、最终提升 solver performance。
- 可以顺带点出 strong branching 是老师，GCNN 是想学一个更快的近似策略。

## Slide 3
### Title
High-Level Algorithmic Idea
### Bullets
- 把当前 B&B 节点状态编码成 variable-constraint bipartite graph $s_t = (G, C, E, V)$。
- 用 GCNN 在图上做 message passing，输出当前 candidate branching variables 上的动作分布。
- 训练范式不是 RL，而是 imitation learning / behavioral cloning from strong branching。
- 保留 SCIP 其它模块不变，只替换 variable selection policy。
### Suggested Figure
- Figure 2 的简化流程版：当前 LP state -> bipartite graph -> GCNN -> action distribution -> branch。
### Speaker Notes
- 这一页用一句话收住方法：state 是图，action 是“选哪个变量 branch”。
- 还要强调本文的主线不是“用了 GNN”，而是“结构化表示 + imitation learning + 嵌回 exact solver”。

## Slide 4
### Title
Training Label and Dataset
### Bullets
- 标签是 expert action，不是最终最优解、不是 tree size、也不是 continuous score。
- 训练样本是 B&B 过程中的 state-action pairs：当前状态 $s_t$ + strong branching 选择的变量 $a_t^\star$。
- 四类 benchmark：set covering、combinatorial auction、capacitated facility location、maximum independent set。
- 每个 benchmark 收集约 100,000 个训练样本、20,000 个验证样本；测试按 easy / medium / hard 分层。
- 为避免 SCIP 自带 strong branching 的 side-effects，作者重写了 `vanillafullstrong` 作为干净 oracle。
### Suggested Figure
- 数据采集流程图：随机实例 -> SCIP 运行 -> 记录 node state + expert action -> 形成 dataset。
### Speaker Notes
- 这一页一定要讲清：监督信号是局部 branching action，不是全局求解标签。
- `vanillafullstrong` 是很有分量的实现细节，说明作者认真处理了 oracle 污染问题。

## Slide 5
### Title
Network Structure
### Bullets
- 输入图由 constraint nodes、variable nodes 和 sparse edge features 组成，保留 MILP 的矩阵结构。
- GCNN 做一轮 graph convolution，但拆成两个 half-convolutions：variable-to-constraint，再 constraint-to-variable。
- $f_C, f_V, g_C, g_V$ 都是 2-layer MLP，用于 message 和 update。
- 最终丢弃 constraint side，只在 variable side 上接 final perceptron 输出分数。
- 架构关键选择是 `sum aggregation + prenorm`，而不是常见的 mean aggregation。
### Suggested Figure
- Figure 2 右图，重点标出 initial embedding、C-side convolution、V-side convolution、final embedding。
### Speaker Notes
- 讲结构时别只列层，要点出“为什么必须先变量到约束，再约束回变量”。
- `sum + prenorm` 是后面 ablation 会回收的重点。

## Slide 6
### Title
Network Output, Loss, and Inference
### Bullets
- Output：masked softmax over candidate branching variables，也就是当前合法可 branch 的变量集合。
- Loss：cross-entropy / negative log-likelihood，提升 expert action $a^\star$ 的概率。
- Inference：在同一候选集合内按模型分数排序，选择 top candidate 接回 SCIP 执行 branching。
- 这篇论文的一个优点是 local output-loss-inference alignment 很干净。
- 但全局仍有 surrogate gap：训练学 imitation，最终评估看 solve time / nodes。
### Suggested Figure
- 一条三段式链路图：state -> masked action distribution -> cross-entropy on expert action -> branch in solver。
### Speaker Notes
- 这页是全报告最值得讲透的一页之一。
- 要明确说：本文不是 regression to strong branching scores，而是 action classification。

## Slide 7
### Title
Novel Contribution and Main Results
### Bullets
- 真正创新不是“第一次做 learned branching”，而是：bipartite graph state + GCNN policy + action-classification imitation。
- Table 1：GCNN 在四类问题上 imitation accuracy 全面优于 TREES、SVMRANK、LMART。
- Table 2：GCNN 整体上在 nodes 和 solve time 上最强，并在几乎所有配置下优于 SCIP 默认 RPB。
- 训练只在 easy 尺度进行，但在 medium / hard 上仍表现强，支持跨规模泛化主张。
- 需要同时看 `Nodes` 和 `Time`：树更小不自动等于总时间更快。
### Suggested Figure
- Table 1 和 Table 2 的裁剪版，突出 GCNN、RPB、FSB 的关键对比格子。
### Speaker Notes
- 强调现实分量：这是在 essentially full-fledged SCIP solver 中比较，不是简化环境。
- 也顺手点一句边界：maximum independent set 的泛化更困难。

## Slide 8
### Title
Strengths, Limitations, and Takeaways
### Bullets
- Ablation 支持两个判断：sum convolution 比 mean 更适合 branching，prenorm 对训练稳定和大规模泛化有帮助。
- Strength：减少特征工程、局部 output-loss-inference 对齐好、在完整 solver 中验证有效。
- Limitation：模型只看到 solver state 的一个子集，严格说更像 POMDP；泛化强但不是无限强。
- Limitation：更深更大的模型可能略降 nodes，但会增加 inference cost，最终 time 未必更优。
- Takeaway：这是 learned branching in exact solvers 的代表作，不是 standalone neural optimizer。
### Suggested Figure
- 一页总结图：Strengths / Limits / Final Takeaway，附上 Table 3 的小结论。
### Speaker Notes
- 结尾建议用一句话收住：这篇论文证明了“结构化 state + GCNN + imitation learning”是 MILP branching 的一条很强路线。
- 如果还有时间，可以补一句 future work：broader benchmarks、RL fine-tuning、hybrid branching integration。
