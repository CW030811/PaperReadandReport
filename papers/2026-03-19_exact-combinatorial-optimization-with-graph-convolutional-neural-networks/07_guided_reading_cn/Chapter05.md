# Chapter 05 - Why Imitation Learning Instead of RL

## 1. 本章范围
- Status: active
- Source: Section 4 opening paragraphs
- Why this chapter: 这一章专门解释作者在训练范式上的关键选择。
  前一章已经把 branching 写成了 MDP，所以这章要回答一个很自然的问题：既然形式上像 RL，为什么作者最后不用 reinforcement learning，而改用 imitation learning？

## 2. 阅读前你先抓住这三个问题
1. 在这个 branching problem 里，为什么 reinforcement learning 看起来很自然？
2. 作者具体认为 RL 会卡在哪些地方，为什么这些问题在这里特别严重？
3. imitation learning 在这篇论文里承担的到底是什么角色，它学的“老师”是谁？

## 3. 英文原文分段
### Passage 1
We now describe our approach for tackling the B&B variable selection problem in MILPs, where we use imitation learning and a dedicated graph convolutional neural network model.

### Passage 2
As the B&B variable selection problem can be formulated as a Markov decision process, a natural way of training a policy would be reinforcement learning. However, this approach runs into many issues.

### Passage 3
Notably, as episode length is proportional to performance, and randomly initialized policies perform poorly, standard reinforcement learning algorithms are usually so slow early in training as to make total training time prohibitively long.

### Passage 4
Moreover, once the initial state corresponding to an instance is selected, the rest of the process is instance-specific, and so the Markov decision processes tend to be extremely large.

### Passage 5
In this work we choose instead to learn directly from an expert branching rule, an approach usually referred to as imitation learning.

## 4. 重点概念与术语
- reinforcement learning, RL:
  通过与环境交互、依靠长期回报来学习 policy 的方法。这里如果采用 RL，目标会更接近“最终让 solver 更快、更省节点”。
- imitation learning:
  不直接靠长期回报摸索，而是先向 expert 学“在当前状态下应该怎么做”。
- expert branching rule:
  本文里被模仿的高质量老师。后面会明确是 `strong branching`。
- episode length:
  一个 episode 持续多久。这里一次 episode 就是解完整个 MILP，而树越差、求解越慢，episode 往往越长。
- randomly initialized policy:
  刚开始还没学会的 policy。作者认为这种初始 policy 在 B&B 里会非常差，导致训练前期代价极高。
- instance-specific MDP:
  一旦实例固定，后续状态转移几乎都围绕这个实例展开，因此单个 MDP 会非常大、非常复杂。
- prohibitively long:
  不是“慢一点”，而是训练长到在实践中基本不可接受。

## 5. 本章核心内容
### 5.1 用中文讲清楚
这一章虽然短，但它在整篇论文里非常关键，因为它决定了作者后面整套训练路线。逻辑起点其实很自然：上一章已经把 branching 写成了 MDP，那么最顺手的想法就是直接用 RL 去学一个 policy，毕竟 RL 本来就是为这种 sequential decision making 问题准备的。

但作者没有这么做。他们的理由不是“RL 在理论上不对”，而是“RL 在这个任务上训练代价太高”。文中点了两个主要问题。

第一个问题是 episode length 和 policy quality 强相关。这里的一次 episode 不是几步小游戏，而是“完整解完一个 MILP 实例”的全过程。初始 policy 如果很差，就会做出很多低质量 branching 决策，搜索树会迅速变大，求解过程会拖得很长。结果就是：训练一开始，RL 连收集一个完整 episode 都非常慢，总训练时间会被前期糟糕 policy 拖到难以承受。

第二个问题是 instance-specific MDP 太大。对这个任务来说，一旦你固定了某个 MILP instance，后面整个搜索过程都围绕这个实例的搜索树展开。换句话说，虽然形式上都叫 MDP，但每个实例都会诱导出一个非常巨大、结构复杂、强依赖具体实例的决策过程。作者认为在这种条件下，直接做标准 RL 很不现实。

于是他们选了 imitation learning。它的核心想法是：既然已经有一个高质量但昂贵的 branching expert，那不如先直接向这个 expert 学动作，而不是让模型从零开始靠 trial-and-error 自己摸索。这样做的好处是，训练信号变得局部、直接、稳定得多。你不需要等整棵树跑完才知道“这一步是不是好决策”，而是可以直接拿 expert 在当前状态下的动作当监督信号。

这也是这篇论文很重要的一条方法论路线：不是端到端优化整个 solver，而是在 exact solver 的局部关键决策上，先用 imitation 学一个高质量近似策略。

### 5.2 这段在全文中的作用
这一段其实是全文 method 设计的“总开关”。

- 它先承认：从问题形式上看，RL 很自然。
- 然后它再说明：从训练可行性看，RL 不划算。
- 最后它把全文正式切到 imitation learning 轨道上，给后面的 expert label、cross-entropy、behavioral cloning 铺路。

所以这一章不是在泛泛谈训练范式，而是在回答一个任何读者都会追问的关键问题：既然你都已经把问题 formalize 成 MDP 了，为什么不直接做 RL？

### 5.3 容易误解的点
- 作者不是在说 RL 永远不适合 branching；他们是在说对这篇论文的设定和资源约束而言，RL 代价太高。
- imitation learning 不是“更高级”的选择，而是“更实际”的选择。这里的优势主要来自高质量 expert 已经存在。
- 本章说的 imitation learning 还是训练范式层面的决定；真正的标签、损失函数、输出空间细节，要到后面章节再正式展开。
- 这章虽然还没点名 `strong branching`，但“expert branching rule” 的位置已经埋好了，后文会把它具体化。

## 6. 证据指针
- Section 4 opening sentence: 作者的方法由 imitation learning 和 dedicated GCNN model 组成。
- Section 4 RL motivation sentence: branching 问题既然可写成 MDP，RL 就是自然候选。
- Section 4 first RL difficulty: episode length 与 policy quality 耦合，导致 early training 极慢。
- Section 4 second RL difficulty: fixed instance 后 MDP 变得 instance-specific 且极大。
- Section 4 closing sentence before 4.1: 作者因此选择直接从 expert branching rule 学习，也就是 imitation learning。

## 7. 一分钟回顾
- 把 branching 写成 MDP 后，RL 在形式上很自然。
- 但在这个任务里，坏 policy 会把 episode 拉得极长，导致 RL 训练前期极慢。
- 同时每个实例诱导出的搜索过程都很大、很复杂，使标准 RL 更难落地。
- 作者因此选 imitation learning：直接学 expert 的局部 branching 动作。

## 8. 你的问答区
- 你可以直接问：“为什么 episode length 和 performance 成正比？”
- 你也可以问：“这里的 imitation learning 到底是不是 behavioral cloning？”
- 如果你已经吃透这一章，直接写“这章吃透了”或“进入下一章”。

## 9. Codex 补充讲解
- Round 0: 已初始化，等待你的问题。
- 如果你提问，我会优先补三类内容：
  - 为什么这个任务上的 RL 特别慢
  - instance-specific MDP 到底是什么意思
  - imitation learning 在本文里和后面 loss / labels 的关系
