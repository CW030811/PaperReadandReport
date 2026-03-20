# Chapter 08 - Output, Loss, and Inference Alignment

## 1. 本章范围
- Status: active
- Source: Section 4.1 Imitation learning; Section 4.3 Policy parametrization; Equation (3)
- Why this chapter: 这一章专门把本文最重要的一条链讲透。
  也就是：模型输出的是什么，训练损失逼它学什么，推理时又拿这个输出怎样在 solver 里做动作。很多论文这三者并不完全对齐，而这篇 paper 在这一点上相对很干净。

## 2. 阅读前你先抓住这三个问题
1. 这篇论文的 network output 到底是什么，是 score、ranking、还是动作分布？
2. 公式 (3) 的 cross-entropy loss 真正在优化什么，它的 supervision signal 是什么？
3. 模型在推理时到底怎么接回 SCIP，输出分布最后怎样变成“选哪个变量 branch”？

## 3. 英文原文分段
### Passage 1
We train by behavioral cloning using the strong branching rule, which suffers a high computational cost but usually produces the smallest B&B trees. We first run the expert on a collection of training instances of interest, record a dataset of expert state-action pairs

$$
\mathcal{D} = \{(s_i, a_i^\star)\}_{i=1}^N,
$$

and then learn our policy by minimizing the cross-entropy loss

$$
L(\theta) = -\frac{1}{N}\sum_{(s,a^\star)\in\mathcal{D}} \log \pi_\theta(a^\star \mid s). \tag{3}
$$

### Passage 2
Following the graph-convolution layer, we obtain a bipartite graph with the same topology as the input, but with potentially different node features, so that each node now contains information from its neighbors.

### Passage 3
We obtain our policy by discarding the constraint nodes and applying a final 2-layer perceptron on variable nodes, combined with a masked softmax activation to produce a probability distribution over the candidate branching variables (i.e., the non-fixed LP variables).

## 4. 重点概念与术语
- behavioral cloning:
  一种 imitation learning 方式，直接把 expert 在每个状态下采取的动作当监督标签。
- expert state-action pairs:
  数据集里的基本样本，每条样本都是“某个 solver 状态 + expert 在该状态下选的 branching action”。
- action label $a^\star$:
  expert 选择的正确动作。这里不是最终解，也不是连续分数，而是“该 branch 哪个变量”。
- cross-entropy loss:
  让模型把 expert 动作的概率尽量推高的分类损失。
- $\pi_\theta(a^\star \mid s)$:
  在状态 $s$ 下，模型给 expert 动作 $a^\star$ 分配的概率。
- probability distribution over candidates:
  模型输出的是当前候选 branching variables 上的概率分布。
- masked softmax:
  softmax 只在合法候选变量上做，不合法变量不会进入动作分布。
- inference policy:
  推理时如何把模型输出接回 solver。这里本质上是按候选变量分数/概率排序，再选最优那个去 branch。

## 5. 本章核心内容
### 5.1 用中文讲清楚
这一章最值得你记住的一件事，就是这篇论文在 `output / loss / inference` 这条链上相对非常一致。

先看 output。模型最终不是输出一个全局求解结果，也不是输出 strong branching score 的回归值，而是输出“当前候选 branching variables 上的一个概率分布”。这个分布来自 variable side 的 final perceptron 再接 `masked softmax`。也就是说，动作空间从头到尾都被限定在“当前哪些变量可以 branch”这件事上。

再看 supervision signal。Section 4.1 很明确：数据集是 expert state-action pairs

$$
\mathcal{D} = \{(s_i, a_i^\star)\}_{i=1}^N.
$$

这里的标签 $a_i^\star$ 不是 subtree size，不是最终 solve time，也不是每个变量的连续分数，而是 expert 在状态 $s_i$ 下最终选了哪个动作。

于是公式 (3) 的 loss 也就很自然：

$$
L(\theta) = -\frac{1}{N}\sum_{(s,a^\star)\in\mathcal{D}} \log \pi_\theta(a^\star \mid s).
$$

这行式子的含义非常直白：如果 expert 在状态 $s$ 下选了动作 $a^\star$，那模型就应该尽量把 $\pi_\theta(a^\star \mid s)$ 变大；如果它给 expert 动作分配的概率太小，loss 就会变大。换句话说，模型训练的目标不是“模仿专家打分”，而是“模仿专家最终的动作选择”。

这一点正是本文相对 ranking / regression 路线的一个清晰区别。它把 learning target 压成了最直接的动作分类问题。

最后看 inference。模型在推理时仍然面对同样的动作空间，也就是当前 candidate branching variables。它先输出这些候选变量上的 masked softmax 分布，然后 solver 按这个分布对应的分数/概率做排序，最终选出最优候选去 branch。也就是说，训练时学的是“哪个动作应该被选中”，推理时用的也还是“哪个动作被选中”。

所以这条链可以压成一句话：

- output：候选变量上的动作分布；
- loss：提高 expert 动作的概率；
- inference：从同一个动作空间里选出最优变量去 branch。

这就是我前面一直强调的“局部对齐很好”。真正存在的 mismatch 不在这条局部链内部，而在更高一层：训练时优化的是 imitation accuracy，最终评价却看 solve time 和 B&B nodes。也就是 local objective 和 global solver objective 之间仍然是 surrogate relationship。

### 5.2 这段在全文中的作用
这一章在全文里相当于方法定义的“语义闭环”。

- Chapter06 讲清了 state 是什么。
- Chapter07 讲清了这个 state 如何经过 GCNN 变成变量侧表示。
- Chapter08 则把这些表示最终翻译成动作分布、监督目标和 solver 里的实际 branch 行为。

如果没有这一章，你很容易把模型误读成“在预测某种 branching score”；但读完这一章后，你应该很清楚：它本质上是在做 candidate-variable action classification。

### 5.3 容易误解的点
- `masked softmax` 输出的是动作分布，不等于 calibrated probability，也不等于真实 strong branching score。
- 公式 (3) 优化的是 expert 动作的 log-probability，不是直接优化 solve time 或树大小。
- 训练 supervision 是单步动作标签，但推理效果最后要通过整个 solver 运行来检验，这里仍有局部目标和全局目标之间的间接性。
- 这篇论文的亮点之一恰恰在于：虽然全局目标没有被直接优化，但局部动作空间在 output、loss、inference 三者之间是对齐的。

## 6. 证据指针
- Section 4.1 first sentence: 训练方式是 behavioral cloning，expert 是 strong branching。
- Section 4.1 dataset sentence: 数据集由 expert state-action pairs 组成。
- Equation (3): cross-entropy / negative log-likelihood 目标，直接提高 expert 动作的概率。
- Section 4.3 output sentence: final perceptron + masked softmax，输出 candidate branching variables 上的概率分布。
- Section 4.3 candidate clarification: 动作空间是当前 non-fixed LP variables。
- Section 5.1 / paper brief synthesis: 推理时实际接回 SCIP，按模型分数选变量 branch。

## 7. 一分钟回顾
- 这篇论文的输出不是全局解，也不是回归分数，而是候选 branching variables 上的动作分布。
- loss 用 cross-entropy 直接逼模型提高 expert 动作的概率。
- inference 时仍在同一个动作空间里选变量，所以 local output/loss/inference 对齐得很好。
- 真正的 objective mismatch 在更高层：训练学 imitation，最终评估看 solver performance。

## 8. 你的问答区
- 你可以直接问：“为什么说这篇 paper 是 action classification，而不是 score regression？”
- 你也可以问：“masked softmax 和最后实际 branch 之间差几步？”
- 如果你已经吃透这一章，直接写“这章吃透了”或“进入下一章”。
- Q1：文章提到的合法变量不合法变量是如何判断的？他们合法不合法的判断在哪一环节影响了模型结构或决策？
- Q2：所以这里变量是否合法的信息是有个Feature存在Variable Node上吗？不合法的变量只在图上占位，但不参与Cross Entropy等训练以及Output输出吗？

## 9. Codex 补充讲解
- Round 0: 已初始化，等待你的问题。
- Round 1: 回答你在 Chapter 8 留下的问题。
- Q1 回答：文章提到的“合法变量 / 不合法变量”是怎么判断的？这种判断在哪一环节影响模型结构或决策？
  - 先说最核心的结论：
    “合法不合法”不是让网络自己学出来的，而是由当前 solver 状态先定义好的 candidate action set。
  - 从前文 MDP 定义看，动作集合是
    $$
    A(s_t),
    $$
    也就是当前聚焦节点下可供 branching 的候选变量集合。
  - 在 Section 3.3 里，作者把它描述为当前节点的 fractional variables。
    也就是说，至少要满足“现在这个节点上它是可 branch 的候选变量”。
  - 在 Section 4.3 里，作者又用更实现化的表述写成：
    candidate branching variables，也就是当前 non-fixed LP variables。
- 这两种说法合起来，实操上你可以这样理解：
  - solver 先根据当前节点状态筛出“当前允许被 branch 的变量集合”；
  - 这些变量才进入模型最后的合法动作空间；
  - 其余变量即使在图里有节点，也不会在最终 softmax 里作为可选动作参与竞争。
- 所以“合法/不合法”的判断主要影响的是输出层，而不是前面的图卷积结构本身。
  - 前面的 GCNN 仍然可以读取整张当前状态图；
  - 但到最后输出 policy 时，要通过 `masked softmax` 把不合法动作屏蔽掉。
- 这意味着 mask 生效的环节有两个层面的意义：
  - 第一，结构意义：
    模型的 action space 被强制限制在 solver 当前真正允许的变量集合里。
  - 第二，训练意义：
    cross-entropy 只会要求模型在合法候选中把 expert 动作概率推高，而不会把概率浪费在非法变量上。
  - 第三，推理意义：
    solver 最终选 branch 变量时，也只会在这些合法候选中排序和选最大值。
- 所以这件事正是本章“局部对齐很好”的一个关键支点：
  - output 在合法动作集合上定义；
  - loss 在合法动作集合上的 expert label 上定义；
  - inference 也在同一个合法动作集合里做最终选择。
- 如果要再压成一句话：
  - 合法变量集合由 solver/当前节点状态先给定；
  - 网络不负责决定“谁有资格进入动作空间”，网络只负责在这个空间里决定“谁最该被选中”。
- Q2 回答：所以“变量是否合法”是某个 feature 存在 variable node 上吗？不合法变量是不是只在图上占位，但不参与 cross-entropy 和输出？
  - 更稳的理解是：
    “是否合法”在这篇论文里主要不是一个单独要预测的 node feature，
    而是由 solver 当前状态导出的 candidate mask。
  - 也就是说，合法性更像“动作空间约束”，而不是“模型要从特征里推断的标签”。
- 这和 variable node feature 是两回事：
  - variable node 上当然会有很多特征，比如 type、bounds、solution value、fractionality、basis status 等；
  - 这些特征会帮助模型判断“哪个变量更值得 branch”；
  - 但“这个变量此刻有没有资格进入最终动作集合”这件事，主要还是由 solver 侧的 candidate filtering 决定。
- 所以你可以把它拆成两层：
  - 图表示层：
    变量节点可以仍然存在于图里，参与 message passing，给别的节点提供上下文。
  - 动作输出层：
    最终 softmax 只在当前合法 candidate variables 上做。
- 这正是 `masked softmax` 的作用：
  - 它不是把这些变量从图里删除；
  - 而是把它们从“最终可选动作集合”里排除掉。
- 因而你的后半句理解基本是对的，但我帮你收紧一下：
  - 不合法变量可以在图里继续占位、继续参与中间表示计算；
  - 但它们不会进入最终 output distribution 的支持集；
  - 也不会在 cross-entropy 里作为合法竞争动作去分配概率质量。
- 更具体地说，训练时：
  - expert label $a^\star$ 一定来自当前合法动作集合；
  - cross-entropy 关注的是模型在这个合法集合里有没有把 $a^\star$ 的概率抬高。
- 推理时：
  - solver 同样只会在 mask 后剩下的合法候选里选最大值去 branch。
- 所以如果要压成一句实现直觉：
  - 图上的 variable nodes 是“状态表示对象”；
  - mask 后的 candidates 才是“动作决策对象”。
- 如果你提问，我会优先补三类内容：
  - Equation (3) 每一项到底是什么意思
  - 为什么这篇论文的 output / loss / inference 被我说成“局部对齐很好”
  - local imitation objective 和 global solver objective 的差别
