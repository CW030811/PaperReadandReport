# Chapter 09 - Dataset Collection and Training Details

## 1. 本章范围
- Status: active
- Source: Section 5.1 Setup; Supplementary Section 1 Dataset collection details; Supplementary Section 2.1 GCNN training details
- Why this chapter: 这一章讲的是方法“怎么真正落地成可训练数据和可执行训练流程”。
  对这篇论文来说，标签不是天然给定的，而是要在 solver 运行过程中自己采出来；所以数据采集流程和训练细节本身就是方法可信度的一部分。

## 2. 阅读前你先抓住这三个问题
1. 这篇论文的训练样本到底是什么，标签从哪里来，为什么不是现成监督数据？
2. 作者为什么要重写 `vanillafullstrong`，原版 strong branching 有什么问题？
3. GCNN 训练时具体怎么做优化，数据规模、batch、learning rate、prenorm 预处理分别是什么设置？

## 3. 英文原文分段
### Passage 1
For each benchmark, we generate random instances and solve them with SCIP. During the branch-and-bound process, we record new node states and strong branching decisions to obtain datasets of state-action pairs.

### Passage 2
For each benchmark problem, we generate 10,000 random instances for training, 2,000 for validation, and $3 \times 20$ for testing (20 easy, 20 medium, 20 hard). We continue processing instances sampled with replacement until the desired number of node samples is reached, namely 100,000 samples for training and 20,000 for validation.

### Passage 3
The strong branching rule implemented in SCIP triggers side-effects which change the solver state itself. In order to use strong branching as an oracle only when generating training samples, we re-implemented a vanilla version of the full strong branching rule in SCIP, named `vanillafullstrong`.

### Passage 4
As described in the main paper, for our GCNN model we record strong branching decisions $(a)$ and extract bipartite state representations $(s)$ during branch-and-bound on a collection of training instances. This yields a training dataset of state-action pairs $\{(s_t, a_t)\}$.

### Passage 5
We first pretrain the prenorm layers as described in the main paper. We then minimize a cross-entropy loss using Adam with minibatches of size 32 and an initial learning rate of $10^{-3}$. We divide the learning rate by 5 when the validation loss does not improve for 10 epochs, and stop training if it does not improve for 20.

### Passage 6
Throughout all experiments we use SCIP 6.0.1 as the backend solver, allow cutting planes at the root node only, deactivate restarts, and otherwise keep SCIP parameters at default values.

## 4. 重点概念与术语
- branching sample:
  一条训练样本，对应某个 B&B 节点时刻的状态和 expert 动作。
- state-action pair:
  本文监督学习的基本单位。状态是当前节点表示，动作是 expert 选的 branching variable。
- sampled with replacement:
  采样实例时允许重复抽到同一个实例，因此并不是每个生成的实例都会被用上。
- `vanillafullstrong`:
  作者为数据采集重写的无副作用 full strong branching 版本，用来作为干净的 oracle。
- side-effects:
  原版 strong branching 在给决策的同时，还会改变 solver 内部状态；这会污染数据采集过程。
- prenorm pretraining:
  在真正训练 GCNN 之前，先用训练集统计量初始化 prenorm 所需的均值和方差参数。
- Adam:
  本文用于优化 cross-entropy 目标的优化器。
- validation plateau:
  验证集损失长期不下降时触发学习率衰减或提前停止。

## 5. 本章核心内容
### 5.1 用中文讲清楚
这一章最重要的事实是：本文的监督数据不是一个现成标注集，而是作者自己在 solver 运行过程中采出来的。

每条训练样本的形式其实很简单，就是一个 state-action pair：

- `state`：当前 B&B 节点的状态表示，也就是前面几章讲过的 bipartite graph；
- `action`：strong branching expert 在这个状态下最终选的 branching variable。

所以标签并不是人工标注，也不是最终最优解，而是“专家在这一时刻会怎么 branch”。这和很多普通分类任务非常不同。

作者为四个 benchmark family 都分别生成随机实例。按 supplementary 的说法，每类问题会先准备 10,000 个训练实例、2,000 个验证实例，以及测试用的 easy / medium / hard 三组各 20 个实例。然后他们并不是固定每个实例只跑一次、每次采固定几个节点，而是反复从这些实例集合里有放回抽样，用 SCIP 去跑，持续记录新的节点状态和 strong branching 决策，直到凑够目标样本量。

这个目标样本量是：

- 100,000 个训练 branching samples；
- 20,000 个验证 branching samples。

这里“sample with replacement” 很关键，因为它说明：作者关注的是凑够足够多的高质量 branching decisions，而不是机械地把每个实例平均用一遍。也因此 supplementary 里还专门报告了实际被 SCIP 用来生成样本的实例数。

然后是一个特别重要的实现细节：作者发现 SCIP 自带的 strong branching 规则不仅会给出决策，还会改动 solver 自身状态。对数据采集来说，这很麻烦，因为你想要的是一个“只给标签、不额外扰动环境”的 oracle。于是他们专门重写了一个 `vanillafullstrong` 版本，把 strong branching 当成纯 oracle 用。这一步不只是工程细节，它直接关系到训练标签是否干净、是否可复现。

训练端则相对标准但也很清楚。GCNN 先做 prenorm 预处理，然后用 Adam 去优化 cross-entropy。具体设置是：

- minibatch size 32；
- 初始学习率 $10^{-3}$；
- 验证损失 10 个 epoch 不改善，就把学习率除以 5；
- 验证损失 20 个 epoch 不改善，就提前停止。

最后，作者还强调实验环境的一致性：backend solver 用的是 SCIP 6.0.1，cutting planes 只在 root node 启用，solver restart 关闭，其它参数尽量保持默认。这是为了让后续比较更公平，也和前面 state representation 里“root-only cuts” 的结构假设保持一致。

### 5.2 这段在全文中的作用
这一章在全文里承担的是“方法可操作性证明”。

- 前面几章已经定义了 state、policy、loss。
- 这一章回答的是：这些东西在真实 solver 里到底怎么收集成数据、怎么训练起来。
- 它同时也解释了为什么本文的标签构造是可信的，因为 label 来源、oracle 实现、采样策略和训练超参数都被交代清楚了。

如果没有这一章，方法会显得很干净，但你会不知道数据是怎么来的，也无法判断实验结果有多可复现。

### 5.3 容易误解的点
- 样本数 100,000 / 20,000 指的是 branching decisions 的数量，不是 MILP 实例的数量。
- 标签不是最终最优解，也不是全局树大小，而是局部动作标签：当前节点 expert 选哪个变量。
- `vanillafullstrong` 的作用不是提升性能，而是避免用 SCIP 内建 strong branching 采样时产生副作用污染。
- 训练细节看起来普通，但和前面的方法设计强相关，尤其是 prenorm 预处理和 root-only cuts 这两个点。

## 6. 证据指针
- Section 5.1 training paragraph: 每个 benchmark 训练 100,000 个 branching samples、验证 20,000 个 branching samples。
- Supplementary Section 1: 10,000 训练实例、2,000 验证实例、测试 3×20 实例，以及 sampled with replacement 的数据采集流程。
- Supplementary Section 1 final paragraph: 原版 strong branching 有 side-effects，因此作者重写 `vanillafullstrong`。
- Supplementary Section 2.1: 先做 prenorm 预处理，再用 Adam、batch size 32、初始学习率 $10^{-3}$ 训练，并用验证损失调 learning rate / early stopping。
- Section 5.1 setup paragraph: backend solver 是 SCIP 6.0.1，root-only cuts，restarts deactivated。

## 7. 一分钟回顾
- 本文的监督数据是作者自己在 SCIP 运行过程中采出来的 state-action pairs。
- 每个 benchmark family 目标收集 100,000 个训练样本和 20,000 个验证样本。
- 为了得到“干净标签”，作者重写了无副作用 oracle：`vanillafullstrong`。
- GCNN 训练采用 prenorm 预处理 + Adam + batch size 32 + learning rate schedule + early stopping。

## 8. 你的问答区
- 你可以直接问：“为什么 strong branching 的 side-effects 会污染训练数据？”
- 你也可以问：“100,000 个 branching samples 和 10,000 个 training instances 是什么关系？”
- 如果你已经吃透这一章，直接写“这章吃透了”或“进入下一章”。

## 9. Codex 补充讲解
- Round 0: 已初始化，等待你的问题。
- 如果你提问，我会优先补三类内容：
  - 数据采集流程到底是“先生成实例”还是“先凑样本数”
  - `vanillafullstrong` 为什么是这篇论文里很关键的一个实现细节
  - 训练超参数和前面方法设计之间的关系
