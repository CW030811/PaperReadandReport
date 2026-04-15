# Chapter 01 - Abstract & Section 1 Introduction

## 1. 本章范围
- Status: mastered
- Source: Abstract; Section 1 Introduction
- Why this chapter: 这一章决定你后面是把 Apollo-MILP 看成“更强的 predictor”，还是看成“先预测、再让 solver 校正、最后只固定可靠变量”的安全 problem reduction 框架。

## 2. 阅读前你先抓住这三个问题
1. 为什么作者认为现有 ND / PS 各自只解决了问题的一半，既没有同时做到“强 reduction”又没有做到“高可靠 fixing”？
2. Apollo-MILP 里的 `correction` 究竟比“把 predictor 训练得更准一点”多带来了什么信息？
3. 这篇论文真正声称的新意是更好的预测器、更新的 fixing criterion，还是一整套 prediction-correction 闭环？

## 3. 英文原文分段
### Passage 1
- Source: Abstract, opening motivation
Leveraging machine learning (ML) to predict an initial solution for mixed-integer linear programming (MILP) has gained considerable popularity in recent years. These methods predict a solution and fix a subset of variables to reduce the problem dimension. Then, they solve the reduced problem to obtain the final solutions. However, directly fixing variable values can lead to low-quality solutions or even infeasible reduced problems if the predicted solution is not accurate enough.

### Passage 2
- Source: Abstract, framework overview
To address this challenge, we propose an Alternating prediction-correction neural solving framework (Apollo-MILP) that can identify and select accurate and reliable predicted values to fix. In each iteration, Apollo-MILP conducts a prediction step for the unfixed variables, followed by a correction step to obtain an improved solution (called reference solution) through a trust-region search.

### Passage 3
- Source: Abstract, UEBO and headline result
By incorporating the predicted and reference solutions, we introduce a novel Uncertainty-based Error upper BOund (UEBO) to evaluate the uncertainty of the predicted values and fix those with high confidence. A notable feature of Apollo-MILP is the superior ability for problem reduction while preserving optimality, leading to high-quality final solutions. Experiments on commonly used benchmarks demonstrate that our proposed Apollo-MILP significantly outperforms other ML-based approaches in terms of solution quality, achieving over a 50% reduction in the solution gap.

### Passage 4
- Source: Section 1 Introduction, background
Mixed-integer linear programming (MILP) is one of the most fundamental models for combinatorial optimization with broad applications in operations research, engineering, and daily scheduling or planning. However, solving large-size MILPs remains time-consuming and computationally expensive, as many are NP-hard and have exponential expansion of search spaces as instance sizes grow. To mitigate this challenge, researchers have explored a wide suite of machine learning (ML) methods.

### Passage 5
- Source: Section 1 Introduction, ND and PS trade-off
Recently, extensive research has focused on using ML models to predict solutions for MILPs. Notable approaches include Neural Diving (ND) and Predict-and-Search (PS), as illustrated in Figure 1. ND fixes a subset of variables based on the prediction and solves the reduced problem, but incorrect fixing can misguide the search and even make the reduced problem infeasible. PS offers a more effective search strategy for a pre-defined neighborhood of the predicted partial solution, leading to better feasibility and higher-quality final solutions, but it is less effective than the fixing strategy in terms of problem dimension reduction.

### Passage 6
- Source: Section 1 Introduction, key intuition behind correction
To address the aforementioned challenges, a natural idea is to refine the predicted solutions before fixing them. We observe that the search process in PS provides valuable feedback to enhance solution quality for prediction, an aspect that has been overlooked in existing research. Specifically, the solver guides the searching direction toward the optimal solution while correcting variable values that are inappropriately fixed. Theoretically, incorporating this correction yields higher precision for predicted solutions.

### Passage 7
- Source: Section 1 Introduction, Apollo-MILP summary
In light of this, we propose a novel MILP optimization approach, called the Alternating Prediction-Correction Neural Solving Framework (Apollo-MILP), that can effectively identify the correct and reliable predicted variable values to fix. In each iteration, Apollo-MILP conducts a prediction step for the unfixed variables, followed by a correction step to obtain an improved solution through a trust-region search. By incorporating both predicted and reference solutions, we introduce a novel Uncertainty-based Error upper BOund (UEBO) to evaluate the uncertainty of the predicted values and fix those with high confidence.

### Passage 8
- Source: Section 1 Introduction, contribution list
We highlight our main contributions as follows. (1) A Novel Prediction-Correction MILP Solving Framework. Apollo-MILP is the first framework to incorporate a correction mechanism to enhance the precision of solution predictions, enabling effective problem reduction while preserving optimality. (2) Investigating Effective Problem Reduction Techniques. We rethink the existing problem-reduction techniques for MILPs and establish a comprehensive criterion for selecting an appropriate subset of variable values to fix, combining the advantages of existing search and fixing strategies. (3) High Performance across Various Benchmarks.

## 4. 重点概念与术语
- `Neural Diving (ND)`: 先预测，再直接固定一部分变量，靠 reduced MILP 来换速度和规模。
- `Predict-and-Search (PS)`: 仍先预测，但不直接硬固定，而是在预测 partial solution 周围做 trust-region search。
- `partial solution`: 当前只对一部分变量给出并尝试固定的解片段，不是最终完整解。
- `reference solution`: 在 Apollo-MILP 里由 solver 的 trust-region search 给出的“校正后”解，用来判断预测值靠不靠谱。
- `prediction-correction`: 不是单纯“预测一次再求解”，而是让 predictor 和 solver 交替配合，逐步扩大可安全固定的变量集合。
- `UEBO`: `Uncertainty-based Error upper BOund`，作者用来评估预测值不确定性与预测-校正分歧的上界型指标。
- `problem reduction`: 通过固定一部分高置信变量，缩小后续 MILP 的搜索空间和问题规模。

## 5. 本章核心内容
### 5.1 用中文讲清楚
这一章的任务不是展开公式，而是先把整篇论文的矛盾搭起来。作者首先指出，近年来很多 ML for MILP 的工作都在做 solution prediction：先让模型猜一个初始解，再把其中一部分变量固定住，最后把 reduced problem 交给传统 solver。这条思路很自然，因为一旦固定变量成功，问题维度会明显下降，搜索会更快。

但问题在于，“固定”是一种很强的操作。只要 predictor 给错了几个关键变量，reduced problem 就可能被强行推离最优区域，甚至直接失去可行性。Figure 1 里其实已经把两条已有路线的取舍画得很清楚：ND 的 reduction 强，但风险也高；PS 的 search 更稳，但它为了保留可行性，需要把 neighborhood 留得更大，因此 reduction 没那么彻底。

Apollo-MILP 的切入点正好卡在这条矛盾线上。作者不是单纯说“我要把 predictor 训得更准”，而是说：既然 PS 的 search 过程本身会暴露哪些预测值靠谱、哪些不靠谱，那就不要浪费这个 solver feedback。于是他们把框架改成一个交替闭环：先预测，再让 solver 在 trust region 里校正出一个 `reference solution`，然后只固定那些在 prediction 和 correction 里都站得住的变量。

所以这一章你至少要先建立一个非常稳的心智模型：Apollo-MILP 的目标不是直接替代 solver，而是把 solver 反馈引进 reduction 决策里。真正被优化的对象，不只是 `prediction accuracy`，而是“哪些变量值得被安全地 fix”，也就是更高质量的 problem reduction。

### 5.2 这段在全文中的作用
这一章相当于整篇论文的“任务定义 + 方法承诺”。

- Abstract 负责给出最浓缩的答案：问题是什么，Apollo-MILP 做什么，为什么 UEBO 是必要的，最后实验声称赢了什么。
- Introduction 负责把已有方法的 trade-off 说透：为什么 direct fixing 危险，为什么 pure search 又不够 aggressive，以及 correction 这一步为什么值得引入。
- 最后的 contribution list 是后文的检查清单。Section 4 必须兑现“prediction-correction + criterion + theory”，Section 5 必须兑现“benchmark performance”。

如果这一章没吃透，后面你很容易把 Apollo-MILP 误读成“PS 加一个新 loss”或者“ND 加一个更强的 confidence score”。其实作者想强调的是一个更高层的 solving loop 设计。

### 5.3 容易误解的点
- 作者批评的不是“所有 ML predictor 都没用”，而是“在 predictor 有误差时，直接 fixing 太脆弱”。
- `correction` 在这里不是简单的后处理 polish，它是后续 deciding what to fix 的关键信号来源。
- `UEBO` 在这一章里还是一个先导概念，严格定义和近似实现要到 Section 4.2 和 4.3 才会真正落地。
- Introduction 中关于 “preserving optimality” 和 “over 50% reduction in the solution gap” 目前都还是作者主张，正式证据要去后面的 theorem 和 experiments 里核对。

## 6. 证据指针
- Abstract: 直接给出问题、方法总览、UEBO 动机和 headline result。
- Section 1 Introduction: ND / PS 的局限、correction intuition、Apollo-MILP 的 framing。
- Figure 1: ND 和 PS 的对比，是整篇论文动机最直观的图。
- Section 1, paragraph on “natural idea is to refine the predicted solutions before fixing them”: correction 这一步为什么不是可有可无。
- Section 1 contribution list: 后文需要逐项兑现的承诺。

## 7. 一分钟回顾
- 现有 ML-based MILP solution prediction 最大的问题，不是不会预测，而是“错了还硬 fix”会把 reduced problem 搞坏。
- Apollo-MILP 的核心不是只强化 predictor，而是引入 solver-based correction 来判断哪些变量真的可以安全固定。
- 如果只记一句话：`prediction 提议，correction 校验，fixing 只作用在双方一致且更可信的变量上。`

## 8. 你的问答区
- 你可以直接贴你看不懂的英文句子，我会按原文顺序帮你拆。
- 你也可以只问一个术语，比如 `reference solution`、`trust-region search`、`UEBO`。
- 如果你想确认自己有没有读懂，也可以先试着回答：Apollo-MILP 到底想同时继承 ND 和 PS 的哪两种优点？
- 如果你觉得这一章已经吃透了，直接回我“进入下一章”就可以，我会继续按原文顺序推进。
- Q1：如何理解PS的机制？ND方法是固定一部分Variable，在剩余空间里求解其他Variable。PS的trust-region search如何理解？Figure1里的红字是ND和PS的目标吗（x[P]=\hat{x}[P]&...）


## 9. Codex 补充讲解
- Round 0：首章已初始化。本周这篇 paper 的领读已经改成“原文顺序版”，不会再额外插出跳序的公式总览章。
- 先给你一个最短记忆版：`ND reduction 强但 fix 风险高；PS search 更稳但 reduction 弱；Apollo-MILP 让 solver 先校正，再决定能 fix 谁。`
- 下一章会接着原文进入 `Section 2 Related Works`，目的不是复习背景，而是确认作者把自己放在哪一条已有工作线上。
- Round 1：回答你这次关于 PS 机制的追问。这里最关键的是把 `硬固定` 和 `带半径的局部搜索` 区分开。
- 先看 ND。它的思想是：模型先挑出一批最有把握的变量，然后直接把这些变量钉死。写成约束就是你提到的这种形式：
  `x[P] = \hat{x}[P]`
  这里 `P` 是被选中的变量下标集合，`\hat{x}[P]` 是模型给这批变量的预测值。这个约束一旦加上，solver 在后续求解时不能再改这些变量，只能在剩余自由变量上搜索。
- 再看 Han et al. 的原始 PS。它先做的前半步其实和 ND 很像：先由 GNN 预测每个二元变量取 1 的边缘概率，再选出 `k_1` 个“最像 1”的变量和 `k_0` 个“最像 0”的变量，构成一个 `partial solution`。Han 文中的记号是 `I_1`、`I_0` 和 `\hat{x}_I`；Apollo 里把同一件事写成 `P` 和 `\hat{x}[P]`。两篇论文在这一步的本质是一致的。
- 真正分叉发生在后半步。PS 不再要求 `x[P] = \hat{x}[P]` 必须逐位成立，而是只要求最终解不要离这份 partial solution 太远。Han 的原始写法是：
  `x \in D \cap B(\hat{x}_I, \Delta)`
  其中
  `B(\hat{x}_I, \Delta) = \{x : \|x_I - \hat{x}_I\|_1 \le \Delta\}`
  Apollo 在 Section 3.3 里把它改写成同样含义的局部约束：
  `x[P] \in B_P(\hat{x}[P], \Delta)`
- 这条约束的直觉是：`P` 里这些变量仍然是“重点参考对象”，但 solver 被允许推翻其中一部分。因为这里主要看的是二元变量，`\|x[P]-\hat{x}[P]\|_1` 实际上就等于“有多少个被选中的变量和预测值不一致”。所以：
  - 如果 `\Delta = 0`，那就退化成一个都不能改，也就是硬固定，等价于 ND 式约束。
  - 如果 `\Delta = 3`，那就表示在被选中的这批变量里，最多允许 3 个位置和预测值不同。
- 所以 PS 的 `trust-region search` 你可以把它想成：
  “不是在整个原始可行域里无约束乱搜，也不是把预测结果当成绝对真理；而是在 `预测 partial solution 附近` 的一个 Hamming / `\ell_1` 球里，让 solver 做一次受控的局部搜索。”
- 这也解释了为什么 Apollo 说 PS 比 ND 更稳。因为 ND 一旦某个关键变量预测错了，但它又被 fix 住，solver 连纠错机会都没有；PS 则给 solver 留了一个“可翻案”的窗口，只要错误数量没有超过 `\Delta` 所允许的范围，solver 仍有机会把这些错位变量改回来。
- 再回答你关于 Figure 1 红字的疑问：它们不是 ND/PS 的“监督目标”，也不是“最终一定要达到的解标签”。更准确地说，它们是 `搜索锚点` 或 `局部搜索中心`。
  - 在 ND 里，这个锚点会被直接变成硬约束，所以看起来像目标，但本质上它是“被强行信任的预测”。
  - 在 PS 里，这个锚点只是 neighborhood center。最终解只需要“离它不太远”，而不是“和它完全一样”。
- 如果对应到 Apollo Figure 1 里的那几类标注：
  - `Predicted variables in the partial solutions` 表示被 predictor 选出来、进入 partial solution 的那批变量。
  - `Variables with changed values after search` 表示 solver 在 trust-region search 后把其中一部分值改掉了。
  - 真正不变的优化目标始终是原 MILP 的目标函数 `c^\top x`，不是“尽量拟合红字预测值”。
- 你还可以再记一个特别有用的判断句：PS 不是 “predict-and-fix”，而是 “predict-and-search-around-the-prediction”。Huang 这篇 ConPaS 论文在 Section 4.3 里也明确写了，它测试时还是“the same way as Predict-and-Search (Han et al., 2022)”；也就是说，它主要改的是训练信号，不是把 PS 的 trust-region 机制换掉。
- 结合 Apollo 来看，作者正是抓住了 PS 的这一点：既然 trust-region search 会告诉我们“哪些预测值经得住 solver 校正，哪些经不住”，那这个搜索结果就不该只拿来产出一个更好的解，还应该反过来参与下一轮 fixing 决策。这就是 Apollo 里 `reference solution` 的来源。
- 如果你愿意，我下一轮可以继续把 `Han_PS` 里的 Problem (8) 和 Problem (9) 逐项对照成“ND 子问题 vs PS 子问题”的中文版，并顺手解释 Apollo 为什么说 `\Delta = 0` 时 PS 会退化回 ND。
