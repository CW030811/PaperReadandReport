# Chapter 07 - GCNN Architecture

## 1. 本章范围
- Status: active
- Source: Section 4.3 Policy parametrization; Figure 2 right
- Why this chapter: 上一章已经讲清“输入长什么样”，这一章要讲“模型如何处理这张图并输出 branching 分数”。
  这部分术语密度高，但它其实围绕一条很清楚的主线展开：先做图上的信息传播，再只在变量侧读出 policy。

## 2. 阅读前你先抓住这三个问题
1. 作者为什么认为 graph convolution 很适合 MILP 的二部图输入？
2. 这篇论文的“单层图卷积”为什么又被拆成了 variable-to-constraint 和 constraint-to-variable 两个 half-convolutions？
3. `prenorm`、`un-normalized sum aggregation`、`masked softmax` 在整个架构里分别起什么作用？

## 3. 英文原文分段
### Passage 1
We parametrize our variable selection policy $\pi_\theta(a \mid s_t)$ as a graph convolutional neural network. Such models, also known as message-passing neural networks, are extensions of convolutional neural networks from grid-structured data to arbitrary graphs.

### Passage 2
Graph convolutions exhibit many properties which make them a natural choice for graph-structured data in general, and MILP problems in particular: 1) they are well-defined no matter the input graph size; 2) their computational complexity is directly related to the density of the graph, which makes it an ideal choice for processing typically sparse MILP problems; and 3) they are permutation-invariant, that is they will always produce the same output no matter the order in which the nodes are presented.

### Passage 3
Our model takes as input our bipartite state representation $s_t = (G, C, V, E)$ and performs a single graph convolution, in the form of two interleaved half-convolutions. In detail, because of the bipartite structure of the input graph, our graph convolution can be broken down into two successive passes, one from variables to constraints and one from constraints to variables.

### Passage 4

$$
c_i \leftarrow f_C\left(c_i,\ \sum_{(i,j)\in E} g_C(c_i, v_j, e_{i,j})\right), \qquad
v_j \leftarrow f_V\left(v_j,\ \sum_{(i,j)\in E} g_V(c_i, v_j, e_{i,j})\right)
$$

for all $i \in C, j \in V$, where $f_C, f_V, g_C$ and $g_V$ are 2-layer perceptrons with ReLU activation functions.

### Passage 5
Following this graph-convolution layer, we obtain a bipartite graph with the same topology as the input, but with potentially different node features, so that each node now contains information from its neighbors. We obtain our policy by discarding the constraint nodes and applying a final 2-layer perceptron on variable nodes, combined with a masked softmax activation to produce a probability distribution over the candidate branching variables.

### Passage 6
In the literature of GCNN, it is common to normalize each convolution operation by the number of neighbours. As noted by Xu et al. this might result in a loss of expressiveness, as the model then becomes unable to perform a simple counting operation. Therefore we opt for un-normalized convolutions.

### Passage 7
However, this introduces a weight initialization issue. To overcome this issue and stabilize the learning procedure, we adopt a simple affine transformation $x \leftarrow (x-\beta)/\sigma$, which we call a prenorm layer, applied right after the summation. The $\beta$ and $\sigma$ parameters are initialized with respectively the empirical mean and standard deviation of $x$ on the training dataset, and fixed once and for all before the actual training.

### Passage 8
Adopting both un-normalized convolutions and this pre-training procedure improves our generalization performance on larger problems, as will be shown in Section 5.

## 4. 重点概念与术语
- policy parametrization:
  用一个具体神经网络结构来表示 policy $\pi_\theta(a \mid s_t)$。
- graph convolution / message passing:
  节点沿着图边聚合邻居信息，再更新自身表示的运算。
- permutation-invariant:
  节点输入顺序打乱，不应改变模型输出。这对图数据非常关键。
- sparse graph:
  MILP 的变量-约束关系通常是稀疏的，因此图卷积的计算量也比较可控。
- half-convolution:
  作者把一次完整的二部图卷积拆成两个方向的传播：变量到约束、约束到变量。
- 2-layer perceptron:
  文中 $f_C, f_V, g_C, g_V$ 都是 2 层 MLP，带 ReLU 激活。
- masked softmax:
  只在合法 candidate branching variables 上做 softmax，得到动作分布。
- un-normalized sum aggregation:
  聚合邻居信息时直接求和，而不是按邻居数量做平均归一化。
- prenorm layer:
  在求和之后加的仿射标准化层 $x \leftarrow (x-\beta)/\sigma$，用于稳定训练。
- expressive power:
  模型表达能力。作者担心 mean-style normalization 会削弱“计数”这类能力。

## 5. 本章核心内容
### 5.1 用中文讲清楚
这一章真正回答的是：上一章那张二部图，到了网络里以后到底怎么变成“该 branch 哪个变量”的分数。

作者先说明为什么 GCNN 很合适。理由其实很贴合 MILP：

- 它天然支持变大小的图输入，所以不用把不同规模实例硬压成固定维度。
- 计算复杂度更多取决于图的稀疏度，而 MILP 通常本来就是稀疏的。
- 它对节点顺序不敏感，也就是 permutation-invariant，这保证了同一个问题不会因为约束或变量重排就得到不同结果。

接着是这章最核心的结构设计：作者说自己的模型只做“一次 graph convolution”，但这一次并不是简单一层黑箱，而是被拆成了两个 interleaved half-convolutions。

第一步是 variable-to-constraint：
每条约束先从与自己相连的变量那里收信息。

第二步是 constraint-to-variable：
每个变量再从与自己相连的约束那里收信息。

为什么要这么拆？因为输入本来就是 bipartite graph，没有变量到变量、约束到约束的直接边。要让一个变量知道“周围整体约束环境”如何，最自然的路径就是先经过约束节点，再把信息传回来。

公式里可以这样直觉化理解：

- $g_C$ 和 $g_V$ 负责产生 message，也就是“从邻居边上送过来的信息”；
- $f_C$ 和 $f_V$ 负责 update，也就是把原节点状态和聚合后的邻居信息合起来，更新节点表示；
- 四个函数都用 2-layer perceptrons 实现。

做完这轮传播后，图的拓扑不变，变的是节点表示。此时每个变量节点已经不再只知道自己原始特征，而是也混入了与自己相连的约束信息。作者随后把 constraint nodes 丢掉，只在 variable nodes 上再接一个 2 层 perceptron，并用 masked softmax 只对合法候选变量输出概率分布。

这一点非常重要：虽然模型中间读了约束节点，但最后 policy 的动作空间仍然只落在 variable side，因为 branching 动作本来就是“选哪个变量”。

然后是一个很有意思的架构选择：作者没有用常见的按邻居数归一化的 graph convolution，而是故意用 un-normalized sum aggregation。原因是他们担心做平均之后，会损失“计数能力”。比如，一个变量到底出现在多少条约束里，这种信息在 branching 里可能是有意义的；如果全都平均掉，模型就更难显式感受到这种差别。

但直接求和会带来训练稳定性问题，因为不同节点的邻居数不同，求和后的尺度会飘。于是作者加了 `prenorm`：在求和之后做一个简单的仿射标准化 $x \leftarrow (x-\beta)/\sigma$，其中 $\beta$ 和 $\sigma$ 先用训练集上的经验均值和标准差初始化，再固定下来。作者后面会用实验说明，这个 “sum + prenorm” 组合对更大规模实例的泛化更有帮助。

### 5.2 这段在全文中的作用
这一章是从“表示”走向“可执行 policy network”的关键一步。

- Chapter06 告诉你输入是什么。
- Chapter07 告诉你模型怎样在这张图上传播信息，并生成变量侧分数。
- Chapter08 则会继续把 output、loss、inference 三件事彻底对齐。

所以这一章如果没吃透，后面你会知道模型输出了一个分布，但不知道这个分布是如何从变量-约束结构中被算出来的。

### 5.3 容易误解的点
- 作者说“single graph convolution”并不意味着结构非常浅陋；在二部图上，这一次卷积本身就包含两个方向的传播。
- constraint nodes 在最后被丢弃，不代表它们没用；它们在中间承担的是信息中转和上下文聚合作用。
- `masked softmax` 是输出层动作约束的一部分，但真正的输出/损失/推理对齐，我们下一章再系统讲。
- `prenorm` 不是常见的 batch norm 替代物，而是作者为 un-normalized sum aggregation 专门设计的稳定化步骤。

## 6. 证据指针
- Section 4.3 opening paragraph: GCNN 为什么适合 MILP 图数据，尤其是 size-invariance、sparsity-friendly、permutation-invariance。
- Section 4.3 middle paragraph: 单层卷积被拆成 variable-to-constraint 和 constraint-to-variable 两个 half-convolutions。
- Section 4.3 Equation (4): $f_C, f_V, g_C, g_V$ 的角色，以及消息聚合和节点更新形式。
- Section 4.3 policy output paragraph: 丢弃 constraint nodes，只在 variable nodes 上接 final perceptron 和 masked softmax。
- Section 4.3 prenorm paragraph: 为什么不用 neighbor normalization、为什么采用 un-normalized sum、以及 prenorm 的定义。
- Section 4.3 last sentence: sum + prenorm 对大规模泛化有帮助，这一点会在实验里验证。

## 7. 一分钟回顾
- 作者用 GCNN 来参数化 branching policy，因为它天然适合变大小、稀疏、顺序不敏感的图输入。
- 模型核心是一轮二部图卷积：先变量到约束，再约束到变量。
- 最后只在变量侧读出分数，并通过 masked softmax 得到候选 branching variables 上的分布。
- 架构上的一个关键选择是 `un-normalized sum + prenorm`，作者认为这能保留表达能力并改善大规模泛化。

## 8. 你的问答区
- 你可以直接问：“为什么 mean aggregation 会损失计数能力？”
- 你也可以问：“公式里 $f$ 和 $g$ 到底谁负责 message，谁负责 update？”
- 如果你已经吃透这一章，直接写“这章吃透了”或“进入下一章”。
- Q1：这里介绍了他们GCNN的核心架构，有哪些设计是该Paper独创的/创新的，有哪些是他沿用或迁移或拼接了其他人的解决办法的？
- Q2：f和g函数的表达式是什么？展开是一个累计聚合叠加Relu非线性激活的函数吗？
- Q3：`sum aggregation`、`prenorm`、`masked softmax` 三者各自解决什么问题？

## 9. Codex 补充讲解
- Round 0: 已初始化，等待你的问题。
- Round 1: 回答你在 Chapter 7 留下的三个问题。
- Q1 回答：这套 GCNN 架构里，哪些更接近本文创新，哪些更接近已有方法的迁移或组合？
  - 先说结论：
    这篇 paper 的强项不是“发明了一种全新的通用 GCNN 层”，而是把几种已有图学习/消息传递思想，和 MILP branching 这个任务做了非常贴切的结合。
  - 更偏“沿用/迁移”的部分：
    - 用 GCNN / message-passing neural network 来处理图结构输入，这个大方向本身不是本文首创。
    - 用 MLP 做 message function 和 update function，这也是很常见的 message-passing 配方。
    - 最后接 softmax 得到动作分布，这也是标准 policy parametrization 思路。
  - 更偏“本文任务化创新”的部分：
    - 把 MILP branching state 写成 variable-constraint bipartite graph，并让 policy 直接工作在这个结构上。
    - 根据二部图结构，把一次卷积明确拆成 variable-to-constraint 和 constraint-to-variable 两个 half-convolutions。
    - 明确强调 `un-normalized sum aggregation`，因为作者认为 branching 任务里“计数信息”重要。
    - 针对这个 sum aggregation 带来的尺度/初始化问题，再配一个 `prenorm` 稳定训练。
    - 最后只在 variable side 读出 policy，这和 branching 的动作空间完全贴合。
  - 所以更准确地说：
    本文的创新点在“为 MILP branching 量身定制地组织表示、消息传递方向、输出空间和训练 formulation”，而不是单独某一个神经网络算子本身特别前所未有。
- 如果要压成一句话：
  - 通用图学习组件，大多是借来的；
  - 但把它们拼成“适合 exact solver 里 learned branching”的这一套，是本文真正有价值的地方。
- Q2 回答：$f$ 和 $g$ 的表达式是什么？它们是不是“累计聚合再过 ReLU”？
  - 论文在主文里没有把 $f_C, f_V, g_C, g_V$ 的每一层权重矩阵完整展开成大公式。
  - 它只明确说：这四个函数都是 2-layer perceptrons with ReLU activation functions。
- 所以你可以把它们理解成：
  - $g_C(c_i, v_j, e_{i,j})$：
    先读取“约束节点 $i$、变量节点 $j$、边 $(i,j)$”三者的信息，
    再通过一个两层 MLP 产生一条 message。
  - 然后把所有邻居 message 做求和：
    $$
    \sum_{(i,j)\in E} g_C(c_i, v_j, e_{i,j})
    $$
  - 最后 $f_C$ 再把“节点自己原来的表示 + 聚合后的邻居信息”合起来，输出更新后的约束节点表示。
- 变量侧的 $g_V, f_V$ 完全同理。
- 所以你说的“累计聚合叠加 ReLU 非线性激活”这个理解方向是对的，但更精确一点应该拆成两段：
  1. 先用 $g$ 生成每条边上的 message；
  2. 对 message 求和；
  3. 再用 $f$ 把节点本身和聚合结果合并更新。
- 最短记忆法：
  - $g$ 更像 message function；
  - $f$ 更像 update function。
- Q3 回答：`sum aggregation`、`prenorm`、`masked softmax` 三者各自解决什么问题？
  - `sum aggregation` 解决的是“邻居信息怎么聚合”。
    作者不用 mean，而用 sum，是因为他们希望模型保留“邻居数量本身”这类计数信息。
    在 branching 里，变量出现在多少条约束中，可能本来就有意义。
  - `prenorm` 解决的是“sum 之后尺度不稳定、训练难收敛”的问题。
    因为不同节点邻居数不同，直接求和后数值范围会飘；
    作者就用 $x \leftarrow (x-\beta)/\sigma$ 先把聚合结果稳定下来。
  - `masked softmax` 解决的是“动作空间约束”问题。
    branching 不是所有变量都能选，只能在当前合法 candidate variables 上选；
    所以 softmax 之前要先把非法动作 mask 掉。
- 你可以把它们记成一条流水线：
  - `sum aggregation`：把邻居信息收上来；
  - `prenorm`：把收上来的量纲稳住；
  - `masked softmax`：把最后的变量分数限制到合法动作集合上。
- 三者分工其实很清楚：
  - 一个管中间怎么“汇总信息”；
  - 一个管中间怎么“稳定训练”；
  - 一个管最后怎么“输出合法动作分布”。
- 如果你提问，我会优先补三类内容：
  - Equation (4) 的逐项拆解
  - 为什么要先 variable-to-constraint 再 constraint-to-variable
  - `sum aggregation`、`prenorm`、`masked softmax` 三者各自解决什么问题
