# Chapter 02 - Related Work

## 1. 本章范围
- Status: active
- Source: Section 2 Related work
- Why this chapter: 这一章不是让你背文献名，而是帮你划清这篇论文和已有工作的边界。
  如果这章没吃透，你后面很容易把作者的创新说成“第一次把学习方法用于 branching”或者“第一次把 GNN 用到组合优化”。

## 2. 阅读前你先抓住这三个问题
1. 离本文最近的 prior work 到底是什么？它们已经在做 imitation of strong branching 了吗？
2. 作者为什么特别强调自己做的是 `classification`，而不是 `ranking` 或 `regression`？
3. 这一章为什么还要提 approximate combinatorial optimization、SAT/SMT、node selection/primal heuristics 这些看起来“不是同一个任务”的工作？

## 3. 英文原文分段
### Passage 1
First steps towards statistical learning of branching rules in B&B were taken by Khalil et al. [30], who learn a branching rule customized to a single instance during the B&B process, as well as Alvarez et al. [4] and Hansknecht et al. [24] who learn a branching rule offline on a collection of similar instances, in a fashion similar to us.

### Passage 2
In each case a branching policy is learned by imitation of the strong branching expert, although with a differently formulated learning problem. Namely, Khalil et al. [30] and Hansknecht et al. [24] treat it as a ranking problem and learn a partial ordering of the candidates produced by the expert, while Alvarez et al. [4] treat it as a regression problem and learn directly the strong branching scores of the candidates. In contrast, we treat it as a classification problem and simply learn from the expert decisions, which allows imitation from experts that don’t rely on branching scores or orderings.

### Passage 3
These works also differ from ours in three other key aspects. First, they rely on extensive feature engineering, which is reduced by our graph convolutional neural network approach. Second, they do not evaluate generalization ability to instances larger than seen during training, which we propose to do. Finally, in each case performance was evaluated on a simplified solver, whereas we compare, for the first time and favorably, against a full-fledged solver with primal heuristics, cuts and presolving activated. We compare against these approaches in Section 5.

### Passage 4
Other works have considered using graph convolutional neural networks in the context of approximate combinatorial optimization, where the objective is to find good solutions quickly, without seeking any optimality guarantees. The first work of this nature was by Khalil et al. [31], who proposed a GCNN model for learning greedy heuristics on several collections of combinatorial optimization problems defined on graphs. This was followed by Selsam et al. [47], who proposed a recurrent GCNN model, NeuroSAT, which can be interpreted as an approximate SAT solver when trained to predict satisfiability. Such works provide additional evidence that GCNNs can effectively capture structural characteristics of combinatorial optimization problems.

### Passage 5
Other works consider using machine learning to improve variable selection in branch-and-bound, without directly learning a branching policy. Di Liberto et al. [15] learn a clustering-based classifier to pick a variable selection rule at every branching decision up to a certain depth, while Balcan et al. [8] use the fact that many variable selection rules in B&B explicitly score the candidate variables, and propose to learn a weighting of different existing scores to combine their strengths.

### Passage 6
Other works learn variable selection policies, but for algorithms less general than B&B. Liang et al. [39] learn a variable selection policy for SAT solvers using a bandit approach, and Lederman et al. [36] extend their work by taking a reinforcement learning approach with graph convolutional neural networks. Unlike our approach, these works are restricted to conflict-driven clause learning methods in SAT solvers, and cannot be readily extended to B&B methods for arbitrary mixed-integer linear programs. In the same vein, Balunovic et al. [9] learn by imitation learning a variable selection procedure for SMT solvers that exploits specific aspects of this type of solver.

### Passage 7
Finally, researchers have also focused on learning other aspects of B&B algorithms than variable selection. He et al. [25] learn a node selection heuristic by imitation learning of the oracle procedure that expands the node whose feasible set contains the optimal solution, while Song et al. [48] learn node selection and pruning heuristics by imitation learning of shortest paths to good feasible solutions, and Khalil et al. [32] learn primal heuristics for B&B algorithms. Those approaches are complementary with our work, and could in principle be combined to further improve solver performance. More generally, many authors have proposed machine learning approaches to fine-tune exact optimization algorithms, not necessarily for MILPs in general. A recent survey is provided by Bengio et al. [10].

## 4. 重点概念与术语
- related work:
  在这篇论文里，它的作用不是“做全景综述”，而是精确划出本文相对最近前作的差异点。
- imitation of strong branching:
  模仿 strong branching 不是本文独有。作者明确承认已有工作已经在这么做。
- ranking problem:
  学的是“候选变量之间谁排前谁排后”的相对顺序。
- regression problem:
  学的是 strong branching 给每个候选变量打出的分数。
- classification problem:
  学的是“专家最后选了哪个动作”。本文把 branching 直接写成动作分类。
- feature engineering:
  手工设计输入特征。作者认为自己用 bipartite graph + GCNN 后，可以减少这部分依赖。
- full-fledged solver:
  完整求解器环境，而不是删减很多组件后的简化版 solver。这里作者特别强调 primal heuristics、cuts、presolving 都开着。
- approximate combinatorial optimization:
  目标是尽快找到好解，但不追求 optimality guarantee。它和本文的 exact optimization 目标不同。
- complementary:
  指这些方法和本文不是互斥替代关系，而是未来可能组合在同一个 solver 里。

## 5. 本章核心内容
### 5.1 用中文讲清楚
这一章最关键的任务，是帮你建立一个很稳的判断：这篇论文并不是“第一个学 branching”的工作，甚至也不是“第一个模仿 strong branching”的工作。离它最近的前作，已经在用 imitation learning 学 branching 了，只不过它们常把任务写成 `ranking` 或 `regression`，并且更依赖手工特征。

所以作者在这里要占住的位置，不是“我第一个提出 learned branching”，而是“我把这个学习问题重新 formulation 了”。具体来说，有四个层面的差异：

- 第一，任务形式不同。别人常学 score 或 ranking，本文直接学 expert action，也就是 `classification over candidate variables`。
- 第二，状态表示不同。别人更依赖人工特征，本文主张用 MILP 的自然二部图结构交给 GCNN 编码。
- 第三，泛化主张不同。作者特别强调自己会看能不能泛化到比训练更大的实例。
- 第四，评测环境不同。作者强调自己不是只在简化 solver 里做比较，而是放回一个更接近真实使用场景的完整 SCIP 环境。

这一章后半段之所以提到 approximate combinatorial optimization、SAT/SMT、node selection、primal heuristics，不是因为它们和本文完全同类，而是为了告诉你两件事。第一，GCNN 确实已经在别的组合优化或逻辑推理任务里表现出对结构信息的建模能力，所以“拿 GCNN 来读组合优化状态”不是凭空冒出来的。第二，machine learning 介入 exact solvers 的位置很多，不只 variable selection 一个点。本文只占其中一个局部环节。

### 5.2 这段在全文中的作用
这部分相当于全文的“创新边界声明”。

- 它先对最接近的 prior work 做切分，防止你把本文创新说大。
- 它再把本文放进更大的 ML-for-optimization 版图里，说明自己既借了别处的建模直觉，又没有越界声称解决了更大的问题。
- 它还提前给实验章埋了一个伏笔：后面实验不是随便做，而是专门去验证这里提前声明的几个区别点，尤其是泛化到更大实例、以及在完整 solver 中比较。

换句话说，Section 2 不是“可有可无的综述开场”，而是在替后文的 novelty claim 和 experimental claim 先做合法性铺垫。

### 5.3 容易误解的点
- 作者说自己是 `classification`，并不意味着后面最终评价指标就是 classification accuracy。真正系统级评价仍然是 solve time、B&B nodes 等 solver performance。
- 本章提到 GCNN 在 approximate optimization 和 SAT 中成功，不等于这些任务和本文完全等价。作者只是借它们来证明“图网络适合处理结构化离散问题”。
- “for the first time and favorably, against a full-fledged solver” 是作者的实验性主张。你现在可以先记住这是他们想强调的差异点，但真正是否站得住，还要到实验章节再验证。
- 本章里提到的 node selection、pruning、primal heuristics 都不是本文要学的对象。它们只是提醒你：solver 里还有很多别的可学习模块。

## 6. 证据指针
- Section 2 first paragraph: 最接近的 learned branching prior work，以及 ranking / regression / classification 三种任务 formulation 的区别。
- Section 2 first paragraph later half: 本文与 prior work 的三项额外差异，分别是 feature engineering、跨规模泛化、full-fledged solver evaluation。
- Section 2 second paragraph: GCNN 在 approximate combinatorial optimization 与 NeuroSAT 中的背景作用。
- Section 2 third paragraph: 不直接学习 branching policy 的 variable-selection 工作，以及 SAT / SMT 上的变量选择学习。
- Section 2 final paragraph: B&B 其他可学习模块，如 node selection、pruning、primal heuristics。

## 7. 一分钟回顾
- 这篇论文不是第一个做 learned branching，也不是第一个模仿 strong branching。
- 它真正强调的新意是：图结构表示、GCNN policy、动作分类式 imitation learning。
- Section 2 还在替后文实验埋钩子：更大规模泛化、以及在完整 solver 里的比较。
- 这章读完后，你对本文的定位应该是“learned branching in exact solvers 的一个更强 formulation”，而不是“ML 取代传统 combinatorial solvers”。

## 8. 你的问答区
- 你可以直接问：“ranking / regression / classification 在这篇论文里到底差在哪？”
- 你也可以让我帮你把本章的 related work 画成一张中文分类树。
- 如果你已经吃透这一章，直接写“这章吃透了”或“进入下一章”。

## 9. Codex 补充讲解
- Round 0: 已初始化，等待你的问题。
- 如果你提问，我会优先补三类内容：
  - 某篇 prior work 在本章里被作者拿来对比什么
  - 作者的 novelty claim 到底落在哪一层
  - 哪些句子是在做“定位”，哪些句子是在做“证据承诺”
