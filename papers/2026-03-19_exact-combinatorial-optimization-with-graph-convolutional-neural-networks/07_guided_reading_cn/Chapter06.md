# Chapter 06 - State Representation

## 1. 本章范围
- Status: active
- Source: Section 4.2 State encoding; Figure 2 left
- Why this chapter: 这一章讲的是本文最关键的表示层设计。
  前面我们已经知道作者要学 branching policy，但真正让 GCNN 能接手这个任务的前提，是先把 solver 的局部状态写成一个既结构化、又能跨实例泛化的图表示。

## 2. 阅读前你先抓住这三个问题
1. 作者到底把 B&B 当前时刻的状态编码成了什么图？图里的两类节点和边分别代表什么？
2. 为什么这个图表示能减少手工特征工程，并且更适合跨规模泛化？
3. 作者为什么主动承认“这只是 solver state 的一个子集”，这会带来什么含义？

## 3. 英文原文分段
### Passage 1
We encode the state $s_t$ of the B&B process at time $t$ as a bipartite graph with node and edge features $(G, C, E, V)$, described in Figure 2 (Left).

### Passage 2
On one side of the graph are nodes corresponding to the constraints in the MILP, one per row in the current node’s LP relaxation, with $C \in \mathbb{R}^{m \times c}$ their feature matrix. On the other side are nodes corresponding to the variables in the MILP, one per LP column, with $V \in \mathbb{R}^{n \times d}$ their feature matrix.

### Passage 3
An edge $(i, j) \in E$ connects a constraint node $i$ and a variable node $j$ if the latter is involved in the former, that is if $A_{ij} \neq 0$, and $E \in \mathbb{R}^{m \times n \times e}$ represents the sparse tensor of edge features.

### Passage 4
Note that under mere restrictions in the B&B solver, namely by enabling cuts only at the root node, the graph structure is the same for all LPs in the B&B tree, which reduces the cost of feature extraction.

### Passage 5
The exact features attached to the graph are described in the supplementary materials. We note that this is really only a subset of the solver state, which technically turns the process into a partially-observable Markov decision process, but also that excellent variable selection policies such as strong branching are able to do well despite relying only on a subset of the solver state as well.

## 4. 重点概念与术语
- state encoding:
  把当前 solver 决策时刻的状态，转成模型可以读取的输入表示。
- bipartite graph:
  二部图。本文把状态拆成两类节点：constraint nodes 和 variable nodes，中间通过非零系数关系连边。
- constraint node:
  每个当前 LP relaxation 的约束行对应一个节点。
- variable node:
  每个 LP 列，也就是每个变量，对应一个节点。
- node features:
  附在约束节点和变量节点上的属性向量，分别写成 $C$ 和 $V$。
- edge features:
  附在 constraint-variable 连边上的属性，写成稀疏张量 $E$。
- sparse tensor:
  大部分位置都为零，因此实现上只需要存储非零边及其特征。
- root cuts only:
  只在根节点启用 cuts。作者指出在这个设置下，整棵 B&B 树里 LP 的图结构保持不变，这能降低特征提取代价。
- subset of the solver state:
  模型并没有观察 solver 的全部信息，而是只取了一个结构化子集。
- partially-observable MDP, POMDP:
  如果模型只能看到完整环境状态的一部分，那么严格说这不再是 fully observed MDP，而是部分可观测问题。

## 5. 本章核心内容
### 5.1 用中文讲清楚
这一章真正回答的是：作者拿什么喂给 GCNN？答案不是一串手工拼出来的扁平特征，而是一个 bipartite graph。

这个图的左边是约束节点，右边是变量节点。你可以把它想成“把 MILP 的矩阵结构直接摊开”。每一行约束变成一个 constraint node，每一列变量变成一个 variable node；如果某个变量出现在某条约束里，也就是对应系数 $A_{ij} \neq 0$，那就在这两个节点之间连一条边。

这样一来，MILP 本身最自然的结构关系就被保留下来了：不是把所有信息压成一个固定长度向量，而是明确保留“哪个变量和哪条约束发生关系”。这正是图表示比大量手工 feature engineering 更有吸引力的地方，因为它让模型自己沿着变量-约束关系去传播和整合信息。

作者还给这张图配了三类特征：

- 约束节点特征，写成 $C \in \mathbb{R}^{m \times c}$；
- 变量节点特征，写成 $V \in \mathbb{R}^{n \times d}$；
- 边特征，写成稀疏张量 $E \in \mathbb{R}^{m \times n \times e}$。

这里你暂时不用死记每个 feature 的具体维度含义，因为作者也把精确 feature 列表放到了 supplementary materials。当前这一章最重要的是抓住“信息是按图组织的，而不是按手工平铺的表组织的”。

还有一个非常关键、也很诚实的细节：作者明确说，这个图表示其实只是 solver state 的一个子集。也就是说，前一章 MDP 里定义的完整状态很大，而现在真正喂给模型的，只是其中一个可计算、可泛化、结构化的观察窗口。严格来说，这会让问题更像一个 POMDP。

但作者紧接着又给了一个很重要的辩护：即便是 strong branching 这样的高质量规则，在实践里也不是靠“看见 solver 的全部状态”才有效，它同样依赖一个有限的信息子集。换句话说，“不是全状态”并不会自动毁掉 variable selection；关键是你选出的这个子集是否保留了足够有用的结构信息。

最后还有一个偏工程但很重要的点：如果只在 root node 启用 cuts，那么整棵 B&B 树里的图结构其实保持不变。这样作者在不同节点上重复提取特征时，就不需要反复重建整张图的连边结构，只需要更新与当前状态相关的特征值，代价会低很多。这一点解释了为什么这种图表示不仅概念上自然，而且实现上也可承受。

### 5.2 这段在全文中的作用
这一章是全文从“问题 formalization”走向“模型可实现输入”的关键转折点。

- 前几章告诉你：我们要学 branching，而且问题可以写成 MDP。
- 这一章进一步告诉你：真正给模型看的，不是抽象的全状态 $s_t$，而是它的一个 bipartite graph view。
- 再下一章，作者才会在这张图上定义 GCNN 如何做 message passing 和打分。

所以如果这一章没吃透，后面你会知道“GCNN 在图上跑”，但不知道这张图为什么长这样，也不知道它和 solver state 之间到底是什么关系。

### 5.3 容易误解的点
- 作者这里编码的是“当前节点的 LP relaxation 相关结构”，不是整棵 B&B 树的完整可视化图。
- 图表示保留了 MILP 的关系结构，但不代表它包含 solver 的全部内部信息；作者自己就明确承认这是 subset of the solver state。
- “technically a POMDP” 不等于方法就站不住。作者的意思是：严格建模上不是 fully observed MDP，但这种有限观察在 branching 里依然可以很有用。
- root-only cuts 这一点不是无关紧要的实现细节，它直接帮助图结构在整棵树里保持稳定，从而降低特征提取成本。

## 6. 证据指针
- Section 4.2 opening sentence: 当前状态 $s_t$ 被编码为带节点特征和边特征的二部图 $(G, C, E, V)$。
- Section 4.2 first half: constraint nodes、variable nodes，以及 $C \in \mathbb{R}^{m \times c}$、$V \in \mathbb{R}^{n \times d}$ 的定义。
- Section 4.2 edge paragraph: 连边条件 $A_{ij} \neq 0$ 与边特征张量 $E \in \mathbb{R}^{m \times n \times e}$。
- Figure 2 left: bipartite state representation 的直观图示。
- Section 4.2 solver restriction note: root-only cuts 让整棵树上的图结构保持一致，减少特征提取成本。
- Section 4.2 final sentences: 这只是 solver state 的子集，因此技术上是 POMDP；同时 strong branching 也能在信息子集上工作。

## 7. 一分钟回顾
- 作者把当前决策状态编码成一张 variable-constraint bipartite graph。
- 约束是左边节点，变量是右边节点，非零系数关系决定连边。
- 模型真正看到的不是完整 solver state，而是一个结构化子集，因此严格说更像 POMDP。
- 这个表示的价值在于：它保留了 MILP 结构关系，同时又能控制特征提取成本。

## 8. 你的问答区
- 你可以直接问：“为什么这种图表示比手工扁平特征更自然？”
- 你也可以问：“这里的 $G, C, E, V$ 四个量我该怎么直观记忆？”
- 如果你已经吃透这一章，直接写“这章吃透了”或“进入下一章”。
- Q1：root cut是什么？作者提到这一点的目的是什么？
- Q2：如何理解"subset of the solver state""当前节点的 LP relaxation 相关结构",这里的subset和当前节点具体是什么？是选择某个branching variable后的LP relaxation还是选branching variable前的状态空间？还是什么别的？
- Q3：C，E，V他们携带的Feature 向量是什么？是one-hot 0/1编码还是什么？
- Q4：如何理解Figure2的右图所展示的信息传递逻辑？
- Q5：cut的理解我需要一个具体例子。它是如何导致不同节点的 LP 约束集合变化的。
- Q6：作者是否有解释清楚为什么在 `root-only cuts` 的设置下，整棵树里图的连边结构基本保持一致。
- Q&：只取了与当前 LP relaxation 相关的一部分结构信息，把它编码成 variable-constraint bipartite graph。后文是否有具体介绍？
## 9. Codex 补充讲解
- Round 0: 已初始化，等待你的问题。
- Round 1: 回答你在 Chapter 6 留下的四个问题。
- Q1 回答：`root cut` 是什么？作者提到这一点的目的是什么？
  - 这里更准确的说法是：`cuts only at the root node`，也就是 cutting planes 只在根节点启用，往下的 B&B 子节点不再继续加新的 cuts。
  - 你可以把 cut 粗略理解成：solver 为了收紧 LP relaxation，额外加入的一些有效约束。
  - 如果每个子节点都不断加入新 cuts，那么不同节点的 LP 约束集合会变化，图里的 constraint side 结构也可能跟着变。
  - 作者专门提这一点，是因为在 `root-only cuts` 的设置下，整棵树里图的连边结构基本保持一致。
  - 这样做的直接好处是：特征提取更便宜。
    不用在每个 B&B 节点都重新搭一张新图，只需要在同一张结构图上更新当前节点相关的 feature 值即可。
- 所以这句话的作用不是定义方法创新，而是解释“这个图表示在工程上为什么跑得动”。
- Q2 回答：怎么理解 `subset of the solver state`、`当前节点的 LP relaxation 相关结构`，以及这里的“当前节点”到底指什么？
  - 先说“当前节点”。
    这里指的是 solver 在当前时刻准备展开、并且即将做 branching decision 的那个 B&B leaf node。
  - 所以这不是“已经选完 branching variable 之后”的 child LP，也不是整棵树的全局状态快照。
  - 更准确地说，它是：
    在做当前这一步 branching 之前，
    以当前聚焦 leaf node 对应的 LP relaxation 为中心，
    抽取出来的一份结构化输入表示。
  - 再说 `subset of the solver state`。
    前一章 MDP 里，完整状态 $s_t$ 理论上包括很多东西：
    整棵 B&B 树、历史分支、最好整数解、每个节点的 LP 解、各种 solver 统计量等等。
  - 但这里真正喂给模型的，并不是这些全部内容。
    作者只取了与当前 LP relaxation 相关的一部分结构信息，把它编码成 variable-constraint bipartite graph。
  - 所以这里的 `subset` 指的是：
    从“完整 solver 状态”里裁出一个模型实际可见的观察窗口。
- 你可以把它记成一句话：
  - MDP 里的 $s_t$ 是理论上的完整状态；
  - 4.2 里的图表示，是模型对这个状态的一个局部观察。
- Q3 回答：$C$、$E$、$V$ 里到底装了什么 feature？是 one-hot 还是别的？
  - 不是全都 one-hot。
  - 作者在 supplementary Table 2 里给了具体特征表，结论是：
    有些是数值型并做了 normalization，
    有些是 indicator，
    还有少数类别型特征用 one-hot encoding。
- 具体来说：
  - 约束特征 $C$ 包括：
    `obj_cos_sim`，约束与目标的 cosine similarity；
    `bias`，约束右端项，做过归一化；
    `is_tight`，当前 LP 解下该约束是否紧；
    `dualsol_val`，对偶解值，做过归一化；
    `age`，LP age，也做过归一化。
  - 边特征 $E$ 主要就是：
    `coef`，也就是约束矩阵里的系数 $A_{ij}$，按约束做过归一化。
  - 变量特征 $V$ 更丰富，包括：
    `type`，变量类型，用 one-hot 表示（binary / integer / implicit integer / continuous）；
    `coef`，目标系数，归一化；
    `has_lb` / `has_ub`，是否有上下界；
    `sol_is_at_lb` / `sol_is_at_ub`，当前解是否卡在上下界；
    `sol_frac`，分数性；
    `basis_status`，单纯形基状态，用 one-hot 表示；
    `reduced_cost`，归一化；
    `age`，归一化；
    `sol_val`，当前解值；
    `inc_val`，当前 incumbent 中的值；
    `avg_inc_val`，历史 incumbents 的平均值。
- 所以更准确地说：
  - 这个表示不是“纯结构无特征”；
  - 也不是“只有 one-hot 编码”；
  - 而是结构图 + 数值特征 + indicator + 少量 one-hot 类型特征的组合。
- Q4 回答：怎么理解 Figure 2 右图的信息传递逻辑？
  - 这个问题其实已经踩到下一章 Chapter07 的主题了，不过我先给你一个够用版，不把细节一次讲太满。
  - Figure 2 右图展示的是：模型拿到左图的 bipartite graph 以后，如何在 constraint nodes 和 variable nodes 之间来回传信息，最后只对 variable nodes 输出 branch 分数。
  - 你可以先把它理解成三步：
    1. 先给约束节点、变量节点、边特征各自做初始 embedding；
    2. 再让信息先从变量传到约束，再从约束传回变量；
    3. 最后只保留变量侧表示，用它给每个 candidate variable 打分。
  - 为什么一定要“变量 -> 约束 -> 变量”？
    因为 branch 的对象是变量，但变量是否重要，取决于它和哪些约束相连、在这些约束里扮演什么角色。
  - 所以右图本质上是在做：
    “让变量通过共享约束彼此间接交流信息”。
- 一个直觉版类比是：
  - 左图定义了谁和谁有关系；
  - 右图则定义了信息怎么沿这些关系流动，最后让每个变量形成一个“我现在值不值得被 branch”的表示。
- 到下一章你重点就会看到：
  - 每一步 message passing 具体怎么写；
  - 为什么作者把一次卷积拆成 constraint-side 和 variable-side 两个 half-convolutions；
  - 最终分数是怎样从变量表示上读出来的。
- Round 2: 回答你后来补充的三个问题。
- Q5 回答：我想要一个更具体的 cut 例子。cut 是怎么让不同节点的 LP 约束集合变化的？
  - 可以用一个很简化的整数规划例子来理解：
    原问题里有变量 $x_1, x_2 \in \{0,1\}$，
    原始线性约束只有
    $$
    x_1 + 2x_2 \le 2.
    $$
  - 如果先做 LP relaxation，就允许 $x_1, x_2$ 取连续值，可能会出现一个分数解，比如
    $$
    x_1 = 1,\quad x_2 = 0.5.
    $$
  - solver 这时可能发现：虽然这个点满足原始线性约束，但它不符合整数解的凸包结构，于是可以加一条新的有效不等式，也就是 cut，例如
    $$
    x_1 + x_2 \le 1.
    $$
  - 这条 cut 不会删掉任何合法整数解，但会把刚才那个分数点切掉，因为
    $$
    1 + 0.5 > 1.
    $$
- 现在看为什么不同节点的 LP 约束集合会变化：
  - 如果 solver 在 root node 加了这条 cut，那么 root 的 LP 就变成“原始约束 + 这条新 cut”。
  - 如果 solver 在某个子节点又根据该子节点的局部 LP 状态再生成另一条 cut，那么那个子节点的约束集合就会比别的节点更多或不同。
  - 一旦不同节点拥有不同数量或不同内容的约束，图里的 constraint nodes 和相关连边结构就可能改变。
- 所以 cut 带来的关键变化是：
  - 不只是约束数值变了；
  - 而是“约束集合本身”可能变了；
  - 这会直接影响二部图左侧的节点集合和边集合。
- Q6 回答：作者是否有解释清楚为什么 `root-only cuts` 下，整棵树里图的连边结构基本保持一致？
  - 论文这里给的是一个简洁结论，而不是展开证明。
  - 原文说的是：在一个 solver restriction 下，也就是只在 root node 开 cuts，the graph structure is the same for all LPs in the B&B tree。
- 这句话背后的直觉是：
  - root 之后不再往后续子节点添加新的 cuts，
  - branching 本身主要是在已有变量上修改 bound，
  - 因而后续子节点通常不会新增新的约束行或新的变量列。
  - 这样一来，矩阵的“谁和谁相连”关系，即 $A_{ij} \neq 0$ 的稀疏结构，就基本保持不变。
- 更精确一点说：
  - branch 会改的是 bound，例如 $x_i \le \lfloor x_i^\star \rfloor$ 或 $x_i \ge \lceil x_i^\star \rceil$；
  - 它主要影响节点特征或变量状态；
  - 不一定改变原始约束矩阵的稀疏模式。
- 所以这里“连边结构基本保持一致”的核心原因不是 branching 不影响 LP，
  而是 branching 主要改 bound，而不是反复增删约束行。
- 这部分论文没有做长篇技术论证，但从 LP/B&B 机制上看，这个解释是合理且和原文一致的。
- Q7 回答：只取当前 LP relaxation 的一部分结构信息，后文是否还有更具体介绍？
  - 有，但分散在后文两块：
  - 第一块是 supplementary Table 2。
    它把 $C$、$E$、$V$ 里的具体 feature 列表列得更细，也就是我们前面补充过的那些 constraint / edge / variable features。
  - 第二块是后面的模型章节。
    后文会继续说明：这些图特征如何经过初始 embedding、两次 half-convolution、再变成变量侧分数。
- 但要注意，后文更偏“如何用这些信息”，而不是再重新论证“为什么选这组信息就一定最优”。
- 所以如果你问的是：
  - “作者有没有把具体特征列出来？”
    有，在 supplementary。
  - “作者有没有系统解释为什么完整 solver state 只取这一子集就足够？”
    只有部分解释，没有完全严格证明。
- 论文在这一点上的论证方式更像：
  - 给出结构化设计理由；
  - 承认这是 subset of solver state；
  - 再用实验结果去支持“这个子集已经足够有效”。
- 如果你提问，我会优先补三类内容：
  - Figure 2 左图里每个元素分别对应什么
  - 为什么 subset of solver state 仍然可能够用
  - root-only cuts 和特征提取成本之间的关系
