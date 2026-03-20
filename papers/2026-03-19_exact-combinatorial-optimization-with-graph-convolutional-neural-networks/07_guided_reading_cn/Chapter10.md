# Chapter 10 - Main Experimental Results

## 1. 本章范围
- Status: active
- Source: Section 5.2 Comparative experiment; Tables 1-2
- Why this chapter: 这一章专门回答“实验到底证明了什么”。
  前面的方法设计如果都成立，那最关键的检验就是：它到底有没有比已有学习方法更强，能不能泛化到更大实例，以及是否真的能在完整 solver 里带来更好的时间表现。

## 2. 阅读前你先抓住这三个问题
1. Table 1 的 imitation accuracy 到底在证明什么，它和真正 solver performance 是什么关系？
2. Table 2 里为什么要同时看 `Time`、`Nodes`、`Wins`，而不是只看一个指标？
3. 作者关于“泛化到更大实例”和“优于 SCIP 默认 branching rule”的主张，到底证据强到什么程度？

## 3. 英文原文分段
### Passage 1
In terms of prediction accuracy (Table 1), GCNN clearly outperforms the baseline competitors on all four problems, while SVMRANK and LMART are on par with each other and the performance of TREES is the lowest.

### Passage 2
At solving time (Table 2), the accuracy of each method is clearly reflected in the number of nodes required to solve the instances. Interestingly, the best method in terms of nodes is not necessarily the best in terms of total solving time, which also takes into account the computational cost of each branching policy, i.e., the feature extraction and inference time.

### Passage 3
The SVMRANK approach, despite being slightly better than LMART in terms of number of nodes, is also slower due to a worse running time / number of nodes trade-off. Our GCNN model clearly dominates overall, except on combinatorial auction (Easy) and maximum independent set (Medium) instances, where LMART and RPB are respectively faster.

### Passage 4
Our GCNN model generalizes well to instances of size larger than seen during training, and outperforms SCIP’s default branching rule RPB in terms of running time in almost every configuration. In particular and strikingly, it significantly outperforms RPB in terms of nodes on medium and hard instances for set covering and combinatorial auction problems.

### Passage 5
As expected, the FSB expert brancher is not competitive in terms of running time, despite producing very small search trees. The maximum independent set problem seems particularly challenging for generalization, as all machine learning approaches report a lower number of solved instances than the default RPB brancher.

### Passage 6
For the first time in the literature a machine-learning-based approach is compared with an essentially full-fledged MILP solver. For this reason, the results are particularly impressive, and indicate that GCNN is a very serious candidate to be implemented within a MILP solver, as an additional tool to speed up mixed-integer linear programming solvers.

## 4. 重点概念与术语
- acc@1 / acc@5 / acc@10:
  imitation accuracy 指标，衡量模型把 expert 的最佳动作排在前 1、前 5、前 10 的频率。
- Time:
  1-shifted geometric mean of solving times。是最终最接近用户真实体验的指标之一。
- Nodes:
  B&B 节点数，反映 branching policy 对搜索树大小的影响。
- Wins:
  某方法成为最快方法的次数 / 被成功求解的实例数。
- trade-off between nodes and time:
  节点更少通常是好事，但如果单次决策本身太慢，总时间仍可能更差。
- RPB:
  reliability pseudocost branching，SCIP 默认的 hybrid branching 规则，也是本文最重要的现实基线。
- FSB:
  full strong branching，本文的慢 expert。树通常更小，但时间很难有竞争力。
- generalization to larger instances:
  训练只在 easy 尺度上做，但测试看 medium / hard 时是否仍保持强表现。

## 5. 本章核心内容
### 5.1 用中文讲清楚
这一章最重要的阅读顺序，不是逐格看表，而是顺着作者的证据链走：

第一步，看 imitation accuracy。Table 1 显示 GCNN 在四类问题上都明显优于 TREES、SVMRANK、LMART。这个结果说明：如果标准是“模仿 strong branching 的动作”，GCNN 在局部决策层面确实学得更准。

但作者没有停在这里，因为 imitation accuracy 不是最终目的。真正的问题是：学得更准，是否真的能让 solver 变快、树变小？

于是第二步，看 Table 2 里的 `Nodes`。作者说 accuracy 的提升在节点数上有清楚反映，也就是局部动作更接近 expert，通常会带来更小的搜索树。这一步很关键，因为它说明 local imitation objective 至少和一个重要下游指标是正相关的。

不过第三步，作者又立刻提醒你不能只看 `Nodes`。因为最终 `Time` 还受另一件事影响：每次 branching 决策本身要花多久，包括特征提取和模型推理时间。于是就出现一个很经典但很重要的现象：节点更少，不一定总时间更快。

文中给了一个具体例子：SVMRANK 的节点数略优于 LMART，但总时间更慢，因为它在运行时的 `running time / number of nodes trade-off` 更差。这个例子非常值得记住，因为它说明这篇论文的实验判断并不幼稚，作者不是只盯着树小不小，而是在认真看“决策质量”和“决策开销”之间的平衡。

在这个平衡下，GCNN 的整体表现最强。作者明确说，除 combinatorial auction 的 Easy 配置和 maximum independent set 的 Medium 配置外，GCNN 总体上 dominate。也就是说，它不是每一个格子都赢，但全局看是最稳的最好方法。

然后来到作者最强调的实验结论：泛化到更大实例。训练是在 easy 尺度上做的，但在 medium / hard 上，GCNN 仍然整体表现很好，并且在几乎所有配置下都优于 SCIP 默认的 RPB。特别是 set covering 和 combinatorial auction 的 medium / hard 实例上，GCNN 在节点数上显著优于 RPB，这正是作者最想证明的“不是只在训练尺度内有效”。

作者也没有回避负面结果。maximum independent set 明显更难泛化，所有机器学习方法在 solved instances 数量上都不如默认的 RPB，GCNN 虽然仍是机器学习方法里整体最强的，但 time 和 nodes 的波动都更大。这一点提醒我们：本文的泛化结论是强的，但不是无限强，更不是全问题家族无条件成立。

最后，FSB 也扮演了一个很有教育意义的对照组。它经常能给出非常小的搜索树，但由于每一步都太贵，最终时间并不竞争。这正好再次呼应全文主线：strong branching 是高质量老师，但不适合直接全程在线用；learned brancher 的价值就在于尽量保住决策质量，同时把开销压下来。

### 5.2 这段在全文中的作用
这一章相当于全文主张的集中验收。

- 它先验证“学 expert 动作”在局部 accuracy 上成立。
- 再验证这种局部优势能否迁移到节点数和求解时间。
- 最后用 medium / hard 实例检验跨规模泛化，以及与完整 SCIP 默认 brancher 的真实对抗结果。

所以 Section 5.2 不是简单展示几张表，而是在逐步回答全文最重要的三个 claim：

- 学得比其他 ML branching 方法更好；
- 在真实 solver 指标上也更强；
- 而且不仅限于训练规模。

### 5.3 容易误解的点
- Table 1 的 accuracy 很重要，但它不是最终胜负标准；真正更关键的是 Table 2 的 time / nodes / wins。
- 节点数更少不等于时间一定更快，这篇论文反而非常强调这个 trade-off。
- “generalizes well” 是实验结论，不是理论保证，而且 maximum independent set 就展示了明显的边界情况。
- 作者说自己“for the first time”是在 essentially full-fledged MILP solver 中做比较，这个 claim 的分量很大，也正因此他们特别强调结果的现实意义。

## 6. 证据指针
- Section 5.2 first sentence: GCNN 在四类问题上的 imitation accuracy 全面领先。
- Section 5.2 second sentence: accuracy 提升会反映到 nodes 上，但 best nodes 不等于 best time。
- Section 5.2 SVMRANK vs LMART sentence: `running time / number of nodes trade-off` 的具体例子。
- Section 5.2 generalization sentence: GCNN 对更大实例泛化良好，并在几乎所有配置下优于 RPB。
- Section 5.2 set covering / auction sentence: medium / hard 上对 RPB 的显著 node 优势。
- Section 5.2 MIS sentence: maximum independent set 对泛化特别困难，所有 ML 方法 solved instances 更少。
- Section 5.2 final paragraph: 与 essentially full-fledged MILP solver 的比较意义。

## 7. 一分钟回顾
- GCNN 在 imitation accuracy 上全面优于其他学习基线。
- 这种局部优势大体能转化成更少节点和更快求解，但时间还取决于推理开销。
- 整体上 GCNN 表现最强，并且在 medium / hard 上显示出明显跨规模泛化。
- 但泛化不是没有边界，maximum independent set 就是明显更难的一类。

## 8. 你的问答区
- 你可以直接问：“为什么作者说 `Nodes` 和 `Time` 必须一起看？”
- 你也可以问：“maximum independent set 为什么看起来特别难泛化？”
- 如果你已经吃透这一章，直接写“这章吃透了”或“进入下一章”。 
- Q1：SCIP是什么？

## 9. Codex 补充讲解
- Round 0: 已初始化，等待你的问题。
- Round 1: 回答你在 Chapter 10 留下的问题。
- Q1 回答：`SCIP` 是什么？
  - `SCIP` 是一个现代的求解器框架，主要用于 mixed-integer linear programming / mixed-integer nonlinear programming 等组合优化问题。
  - 在这篇论文里，你可以先把它理解成：
    一个真实可用、功能很完整的 MILP solver。
  - 它不只是“解一个 LP”那么简单，而是内部包含了 branch-and-bound、branching rules、cutting planes、heuristics、presolving、node selection 等很多模块。
- 这也是为什么作者反复强调：
  - 他们不是在一个简化 toy environment 里比较；
  - 而是把 learned branching policy 接回了 `SCIP 6.0.1` 这种 essentially full-fledged solver 里。
- 所以在本文语境里，`SCIP` 的意义主要有两层：
  - 第一，它是实验 backend。
    作者所有 policy evaluation 都是在 SCIP 里跑出来的，而不是脱离 solver 单独做离线分类测试。
  - 第二，它是现实基线来源。
    例如文中的默认 branching rule `RPB` 就是 SCIP 里的默认/代表性 expert-designed branching 规则之一。
- 为什么这点重要？
  - 因为如果只在简化环境里比较，模型可能只是“在一个被阉割的问题上好用”；
  - 但如果能在 SCIP 这种完整求解器里带来 time / nodes 改善，结论就更接近真实部署价值。
- 所以最短版记法是：
  - `SCIP` 不是本文的方法；
  - `SCIP` 是本文把 learned brancher 嵌进去进行真实评测的 solver 平台。
- 如果你提问，我会优先补三类内容：
  - Table 1 和 Table 2 应该怎样联动解读
  - 为什么 `FSB` 树小却时间差
  - “generalizes well” 到底强到什么程度、边界又在哪里
