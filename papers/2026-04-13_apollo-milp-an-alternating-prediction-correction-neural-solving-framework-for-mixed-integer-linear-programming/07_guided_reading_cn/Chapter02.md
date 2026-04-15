# Chapter 02 - Section 2 Related Works

## 1. 本章范围
- Status: active
- Source: Section 2 Related Works
- Why this chapter: 这一章不是可有可无的文献综述，而是在明确作者到底把自己放在什么问题线上。读完后你要能区分：Apollo-MILP 不是一般的 “ML for MILP” 工作，也不只是 “更强的 predictor”，而是卡在 `solution prediction` 这条支线里，专门回应 ND / PS 这类方法的局限。

## 2. 阅读前你先抓住这三个问题
1. Section 2 为什么先讲 `ML-Enhanced Branch-and-Bound Solver`，再讲 `ML for Solution Prediction`？这两个方向的任务边界有什么不同？
2. 在作者的叙述里，ND 和 PS 为什么被当成 Apollo-MILP 最直接的前置工作，而不是 branch / cut / node selection 那些更大的 ML4CO 方向？
3. 这章看似在“列文献”，但实际上它在帮作者完成什么论证铺垫？

## 3. 英文原文分段
### Passage 1
- Source: Section 2, opening framing
ML-Enhanced Branch-and-Bound Solver In practice, typical MILP solvers, such as SCIP and Gurobi, are primarily based on the Branch-and-Bound (B&B) algorithm. ML has been successfully integrated to enhance the solving efficiency of these B&B solvers.

### Passage 2
- Source: Section 2, examples of ML-enhanced B&B modules
Specifically, many researchers have leveraged advanced techniques from imitation and reinforcement learning to improve key heuristic modules. A significant portion of this work aims to learn heuristic policies for selecting variables to branch on, selecting cutting planes, and determining which nodes to explore next.

### Passage 3
- Source: Section 2, broader B&B ecosystem
Additionally, extensive research has been dedicated to boosting other critical modules in the B&B algorithm, such as separation, scheduling of primal heuristics, presolving, data generation and large neighborhood search. Beyond practical applications, theoretical advancements have also emerged to analyze the expressiveness of GNNs for MILPs and LPs, as well as to develop landscape surrogates for ML-based solvers.

### Passage 4
- Source: Section 2, switch to solution prediction line
ML for Solution Prediction Another line of research leverages ML models to directly predict solutions. Neural Diving (ND) is a pioneering approach in this field. Specifically, ND predicts a partial solution based on coverage rates and utilizes SelectiveNet to determine which predicted variables to fix.

### Passage 5
- Source: Section 2, ND to PS transition
To enhance the quality of the final solution, subsequent methods incorporate search mechanisms, such as trust-region search (PS) and large neighborhood search with sophisticated neighborhood optimization techniques.

### Passage 6
- Source: Section 2, author narrowing the scope
In this paper, we focus on ND and PS, both of which have gained significant popularity in recent years.

## 4. 重点概念与术语
- `ML-Enhanced Branch-and-Bound Solver`: 用 ML 去增强传统 MILP solver 内部的某个模块，比如 branching、cutting、node selection，而不是直接预测最终解。
- `Branch-and-Bound (B&B)`: 传统 MILP solver 的主干搜索框架。Section 2 前半段列的大量工作都属于“给这个框架装更好的启发式模块”。
- `branching / cutting / node selection`: 三类最经典的 solver-internal 决策点，分别对应“分哪个变量”“加哪些 cutting planes”“扩展哪个搜索节点”。
- `solution prediction`: 直接让模型去猜一个完整或部分解，再围绕这份预测开展 fixing 或搜索。
- `Neural Diving (ND)`: solution prediction 方向的代表方法，核心是预测 partial solution 并固定其中一部分变量。
- `Predict-and-Search (PS)`: 在 ND 基础上不再硬固定，而是围绕预测 partial solution 做 trust-region neighborhood search。
- `large neighborhood search (LNS)`: 在一个给定解附近放宽一部分变量，重新优化局部邻域的搜索思路。Section 2 把它同时放在 B&B 生态和 solution-refinement 脉络里。

## 5. 本章核心内容
### 5.1 用中文讲清楚
这一章的关键，不是记住一长串文献名，而是搞清作者在“切问题”。Section 2 先把整个 ML for MILP 的大盘子摆出来：第一类工作是 `ML-enhanced B&B solvers`。这类方法并不试图直接产出一个 primal solution，而是把学习器嵌进 SCIP、Gurobi 这类 solver 的内部模块里，去改进 branching、cutting、node selection、presolving、primal heuristics scheduling 等步骤。也就是说，它们优化的是“solver 怎么搜”。

然后作者马上切到第二条线：`ML for solution prediction`。这条线才是 Apollo-MILP 真正要接续的脉络。这里的目标不再是提升某个局部 heuristic，而是直接预测一个解，或者至少预测一个高质量 `partial solution`，再用它去缩小搜索空间。ND 是这条线的代表起点：先预测，再 fix 一部分变量。后续工作觉得这样太脆，于是引入 search 机制，比如 PS 的 trust-region search，或者更一般的 large neighborhood search。

所以你可以把 Section 2 理解成一个两层筛选过程。第一层，作者先说：“我不是在 branch-and-bound 内部某个模块上做学习增强。” 第二层，作者再说：“在直接预测解这条线里，我也不是泛泛而谈，我主要对接的是 ND 和 PS 这两个最直接的基线。” 这就是为什么这一章最后一句很重要：`In this paper, we focus on ND and PS.`

这也解释了 Apollo-MILP 在 Chapter01 里的 framing 为什么那么紧。它不是想跟所有 ML4CO 工作竞争，而是想在 `solution prediction` 这条更窄、但更直接相关的路线里回答一个具体问题：如何同时保留 ND 的强 reduction 能力和 PS 的更好 feasibility / search flexibility。换句话说，Section 2 的作用是把 Apollo-MILP 的“对手名单”和“贡献边界”都圈定清楚。

### 5.2 这段在全文中的作用
这一章在全文里承担的是“定位工作坐标系”的功能。

- 前半段把 Apollo-MILP 放到更大的 ML4CO 背景里，说明作者知道还有一整套 B&B-enhancement 工作，但本文不打算在那个层面展开。
- 后半段缩到 solution prediction 这条线，明确 ND 和 PS 才是本文真正要继承、对比和超越的前置方法。
- 这一步做完之后，Section 3 和 Section 4 才能理直气壮地直接使用 `partial solution`、`trust-region search`、`predictor` 这些 PS/ND 语境下的概念，而不用再反复解释“为什么不是 branching policy”。

如果这章没读透，后面你很容易把 Apollo-MILP 的贡献边界搞混，误以为它是在和所有 ML-guided MILP solver 全面竞争。其实作者更精准的主张是：在 `solution prediction -> partial solution -> reduced problem / neighborhood search` 这条链上，Apollo-MILP 比 ND 和 PS 更进一步。

### 5.3 容易误解的点
- Section 2 前半段列了很多 B&B 相关工作，但这不代表 Apollo-MILP 属于同一类方法；作者恰恰是在先划清边界。
- `ML for Solution Prediction` 不等于“直接输出完整可行解”。很多方法其实只输出 partial solution，后续仍要依赖 solver。
- `large neighborhood search` 在这里是被当作后续增强 search 的例子，不是本文真正要详细复现的核心基线。
- 最后一行只点名 ND 和 PS，不是随手举例，而是在正式声明本文方法比较的主线对象。

## 6. 证据指针
- Section 2, first paragraph: B&B solver 是传统 MILP 求解主干，ML 已经大量进入其内部模块。
- Section 2, middle paragraphs: branching、cutting、node selection、presolving、primal heuristics、LNS 等相关方向。
- Section 2, paragraph beginning with `ML for Solution Prediction`: 明确切换到 solution prediction 路线。
- Section 2, sentences on ND and PS: 本文真正要承接的直接前置方法。
- Chapter01 Figure 1: 回看 ND / PS 的差异，会更容易理解为什么 Section 2 只聚焦这两类基线。

## 7. 一分钟回顾
- Apollo-MILP 不是一般意义上的 “ML-enhanced solver module”，而是站在 `solution prediction` 这条路线上的方法。
- Section 2 先把大背景划出来，再把焦点收缩到 ND 和 PS，这样后文的方法和实验比较对象才成立。
- 如果只记一句话：`这章不是在堆文献，而是在给 Apollo-MILP 画边界、选对手、定语境。`

## 8. 你的问答区
- 你可以直接问：为什么 branch-and-bound 增强方法不算 Apollo-MILP 的直接同类？
- 你也可以问：Section 2 里提到的 ND、PS、LNS 三者到底是什么关系？
- 如果你想自测，可以先试着回答：作者为什么不把自己的工作归到 “ML-enhanced B&B solver” 里？
- 如果你觉得这一章已经吃透了，直接回我“进入下一章”，我们就继续按原文顺序读 Section 3 Preliminaries。

## 9. Codex 补充讲解
- Round 0：第二章已初始化。这里先不展开新公式，而是把 Apollo-MILP 的文献坐标系摆正。
- 这一章有个很重要的阅读动作：不要平均用力地记每条 related work，而要看作者如何从“大领域”收束到“我真正比较的那两篇工作”。
- 你可以把 Section 2 的结构压成两句话：
  `句子 1：ML 已经能增强 B&B 里的很多模块。`
  `句子 2：但本文真正所在的是 solution prediction 这条线，而且重点对接 ND 和 PS。`
- 这章吃透之后，下一章 `Section 3 Preliminaries` 就会更顺，因为你会知道为什么作者一上来就讲 MILP 表示、bipartite graph 和 PS，而不是继续展开 branching / cutting 那些内容。
