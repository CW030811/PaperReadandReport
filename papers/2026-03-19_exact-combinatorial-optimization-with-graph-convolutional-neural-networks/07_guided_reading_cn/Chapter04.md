# Chapter 04 - Branching Rules and MDP Framing

## 1. 本章范围
- Status: active
- Source: Sections 3.2-3.3; Figure 1
- Why this chapter: 这一章把“branching 为什么重要”与“为什么作者把它写成 sequential decision making”正式接上。
  你会在这里第一次看到本文如何把 solver、brancher、state、action、transition 放进同一个抽象框架里。

## 2. 阅读前你先抓住这三个问题
1. `strong branching` 为什么通常能产生更小的搜索树，但又不能在每个节点都直接用？
2. `hybrid branching` 想解决的到底是什么问题，它和 `conflict score`、`pseudo-cost` 的关系是什么？
3. 作者把 branching 写成 MDP 时，`state`、`action`、`policy`、`transition` 在 solver 里分别对应什么？

## 3. 英文原文分段
### Passage 1
A key step in the B&B algorithm is selecting a fractional variable to branch on in (2), which can have a very significant impact on the size of the resulting search tree. As such, branching rules are at the core of modern combinatorial optimization solvers, and have been the focus of extensive research.

### Passage 2
So far, the branching strategy consistently resulting in the smallest B&B trees is strong branching. It does so by computing the expected bound improvement for each candidate variable before branching, which unfortunately requires the solution of two LPs for every candidate.

### Passage 3
In practice, running strong branching at every node is prohibitive, and modern B&B solvers instead rely on hybrid branching, which computes strong branching scores only at the beginning of the solving process and gradually switches to simpler heuristics such as the conflict score, the pseudo-cost, or a hand-crafted combination of the two.

### Passage 4
As remarked by He et al., the sequential decisions made during B&B can be assimilated to a Markov decision process. Consider the solver to be the environment, and the brancher the agent. At the $t$-th decision the solver is in a state $s_t$, which comprises the B&B tree with all past branching decisions, the best integer solution found so far, the LP solution of each node, the currently focused leaf node, as well as any other solver statistics.

### Passage 5
The brancher then selects a variable $a_t$ among all fractional variables $A(s_t) \subseteq \{1, \ldots, p\}$ at the currently focused node, according to a policy $\pi(a_t \mid s_t)$. The solver in turn extends the B&B tree, solves the two child LP relaxations, runs any internal heuristic, prunes the tree if warranted, and finally selects the next leaf node to split. We are then in a new state $s_{t+1}$, and the brancher is called again to take the next branching decision. This process, illustrated in Figure 1, continues until the instance is solved.

### Passage 6
As a Markov decision process, B&B is episodic, where each episode amounts to solving a MILP instance. Initial states correspond to an instance being sampled among a group of interest, while final states mark the end of the optimization process. The probability of a trajectory $\tau = (s_0, \ldots, s_T) \in \mathcal{T}$ then depends on both the branching policy $\pi$ and the remaining components of the solver,

$$
p_\pi(\tau) = p(s_0)\prod_{t=0}^{T-1}\sum_{a \in A(s_t)} \pi(a \mid s_t)\, p(s_{t+1} \mid s_t, a).
$$

### Passage 7
A natural approach to find good branching policies is reinforcement learning, with a carefully designed reward function. However, this raises several key issues which we circumvent by adopting an imitation learning scheme, as discussed next.

## 4. 重点概念与术语
- branching rule:
  在当前 B&B 节点上，从多个 fractional variables 中决定“branch 哪一个”的规则。
- strong branching:
  在真正分支前，先为每个候选变量做更昂贵的试探性评估，估计它会带来多大的 bound improvement。
- hybrid branching:
  一种工程上更可用的折中方案。前期多借助 strong branching，后期逐渐切到更便宜的启发式。
- conflict score:
  一类利用冲突信息的启发式评分。在本文里它只是 hybrid branching 可能用到的简单替代分数之一。
- pseudo-cost:
  根据历史 branching 效果累计出来的经验性分数，用来便宜地估计未来 branch 的收益。
- Markov decision process, MDP:
  用状态、动作、转移、策略来描述连续决策的抽象框架。作者用它来定义 branching 问题，而不是在这一节就真的开始做 RL。
- state $s_t$:
  第 $t$ 次 branching 决策时 solver 的状态，包括当前搜索树、历史分支、当前聚焦叶节点、已知最好整数解、LP 解以及其他统计量。
- action $a_t$:
  当前时刻选择的 branching variable。
- action set $A(s_t)$:
  状态 $s_t$ 下所有可选的 fractional variables。
- policy $\pi(a_t \mid s_t)$:
  在状态 $s_t$ 下选择动作 $a_t$ 的规则或概率分布。后文学习的就是这个 policy。
- episodic:
  一个 episode 有明确开始和结束。这里一次完整求解一个 MILP instance，就是一个 episode。

## 5. 本章核心内容
### 5.1 用中文讲清楚
这一章先回答一个非常现实的问题：为什么 branching 这么值得学？作者给出的理由很直接。branching variable 选得好不好，会显著影响最后搜索树有多大；而搜索树大小又直接牵动求解时间。所以 branching rules 一直是 MILP solver 里的核心部件。

然后作者把最强基线摆出来：`strong branching`。它之所以强，不是靠神秘技巧，而是因为它在“真正 branch 前”会为每个 candidate variable 先做代价很高的试探，估计如果沿这个变量切下去，bound 会改善多少。这个前瞻动作通常能带来更小的树，但代价也很明显：每个候选变量都要额外解两个 child LP。候选变量一多，在线开销就迅速爆炸。

所以工程上大家不会在整棵树的每一步都跑 full strong branching，而是采用 `hybrid branching`。它的直觉很朴素：求解早期多用一些贵但准的 strong branching 信息，等积累出更多历史统计后，再逐渐切到便宜的启发式，比如 `conflict score`、`pseudo-cost`，或者二者的组合。

接着作者做了一个对全文非常关键的抽象升级：把 branching 看成一个 MDP。这里你可以先完全抛开“RL”三个字的压力，只把它当成一个记账框架。

- environment 是 solver。
- agent 是 brancher。
- state $s_t$ 是第 $t$ 次 branching 决策时 solver 的整体状态。
- action $a_t$ 是这一步选哪个变量 branch。
- action set $A(s_t)$ 是当前节点所有 fractional variables。
- transition 指的是：brancher 选完变量后，solver 去扩展树、解两个 child LP、运行内部 heuristics、做 pruning、再选下一个要展开的叶节点，于是系统进入 $s_{t+1}$。

Figure 1 就是在图上把这件事画出来。左边是当前状态 $s_t$，粉色节点表示当前 solver 选中准备展开的叶节点；此时可行动作集合是若干个候选变量。brancher 选出一个动作，比如 $a_t = 4$，右边就得到新的状态 $s_{t+1}$，搜索树也因此长出新的分支。

这一节最值得你记住的地方，是作者并不是为了炫技才把 B&B 写成 MDP，而是为了把“学一个 branching policy”这件事写得足够标准化。后面不管是 imitation learning 还是讨论 RL 为什么不合适，都是建立在这套 state-action-transition 语言之上的。

### 5.2 这段在全文中的作用
这一章在全文里承担两层作用。

- 第一层是任务价值说明。Section 3.2 先告诉你：branching 不是边角料，它是 solver 的核心瓶颈之一；strong branching 是高质量专家，但太贵。
- 第二层是学习问题的正式建模。Section 3.3 把“选哪个变量 branch”写成 policy learning 问题，为后面 method 和 training objective 铺路。

如果没有这章，后面你会知道作者“在学 policy”，但不知道这个 policy 到底嵌在 solver 的哪一步，也不知道为什么 imitation strong branching 是一个合理的学习目标。

### 5.3 容易误解的点
- `strong branching` 强，不等于它总能带来最短 wall-clock time。它主要以更小的搜索树著称，但在线计算代价非常高。
- 作者在这里把问题写成 MDP，不等于他们接下来就一定用 reinforcement learning。相反，这一节的结尾已经在为“为什么不用 RL”做过渡。
- 这里定义的 `state` 很完整，接近“solver 理论上的全部状态”；但后面真正喂给模型的观测并不是这个全量状态，而是一个可计算、可泛化的结构化表示。这一点要到后续状态表示章节再看。
- Figure 1 里 solver 和 brancher 是分工关系，不是说神经网络接管了整个 solver。网络未来只替换 brancher 这一小块。

## 6. 证据指针
- Section 3.2 first sentence: branching variable 对搜索树大小的重要性。
- Section 3.2 strong branching paragraph: strong branching 为什么强，以及为什么每个候选变量都要额外解两个 LP。
- Section 3.2 hybrid branching paragraph: hybrid branching、conflict score、pseudo-cost 的定位。
- Figure 1 caption: 当前状态、动作选择、以及得到下一状态的图示。
- Section 3.3 first paragraph: solver 作为 environment、brancher 作为 agent 的 MDP 对应关系。
- Section 3.3 middle paragraph: $A(s_t)$、$\pi(a_t \mid s_t)$、$s_{t+1}$ 的定义。
- Section 3.3 trajectory paragraph: episodic MDP、trajectory probability，以及为何这为后续 imitation learning / RL 讨论提供形式化基础。

## 7. 一分钟回顾
- branching rule 决定当前节点 branch 哪个变量，对搜索树大小影响很大。
- `strong branching` 质量高，但因为要为每个候选变量额外解 LP，所以太贵。
- 真实 solver 常用 `hybrid branching`，在 early stage 借 strong branching，后面切到更便宜的启发式。
- 作者把 branching 写成 MDP，是为了标准化定义 policy learning 问题，而不是在这里直接做 RL。

## 8. 你的问答区
- 你可以直接问：“`pseudo-cost` 和 `strong branching score` 的区别到底是什么？”
- 你也可以让我把 Figure 1 用纯中文重新画成一步一步的流程。
- 如果你已经吃透这一章，直接写“这章吃透了”或“进入下一章”。
- Q1：作者这里引入MDP的数学公式有什么意义？
- Q2：Hybrid Branching的Pseudo Cost和Conflict Score如何理解？
- Q3：为什么引入MDP有利于理解Learning？并且和Reinforcement Learning强相关？
- Q4：帮我简单解释一下MDP数学公式里派和P的含义
- Q5：解释一下RL和imitation learning的区别

## 9. Codex 补充讲解
- Round 0: 已初始化，等待你的问题。
- Round 1: 回答你在 Chapter 4 留下的三个问题。
- Q1 回答：作者这里引入 MDP 的数学公式，主要不是为了让你真的去手算轨迹概率，而是为了把“branching 是一个连续决策过程”写得形式化、可学习。
  - 公式

$$
p_\pi(\tau) = p(s_0)\prod_{t=0}^{T-1}\sum_{a \in A(s_t)} \pi(a \mid s_t)\, p(s_{t+1} \mid s_t, a)
$$

  表达的是：一条完整求解轨迹 $\tau$ 会如何同时受到 branching policy $\pi$ 和 solver 其余机制的共同影响。
  - 它的第一层意义是“记账”。
    也就是说明一次 MILP 求解不是单步分类，而是一串状态-动作-转移连起来的 trajectory。
  - 它的第二层意义是“界定学习对象”。
    作者真正要学的是 $\pi(a_t \mid s_t)$，而不是整个 solver 的转移机制 $p(s_{t+1} \mid s_t, a)$。
  - 它的第三层意义是“为后文解释为什么不用 RL 铺路”。
    因为一旦你把问题正式写成 MDP，读者自然会问“那为什么不直接 reinforcement learning？”作者下一节正是顺着这个问题回答。
- 所以更直白地说，这个公式的价值不在数值计算，而在概念定位：它把 branching policy 放回了一个长期序列决策系统里。
- Q2 回答：`pseudo-cost` 和 `conflict score` 都是 hybrid branching 里更便宜的启发式信号，但它们看问题的角度不同。
  - `pseudo-cost` 可以理解成“历史经验账本”。
    某个变量以前 branch 过以后，objective bound 通常改善多少，solver 会把这种历史效果积累起来。以后再遇到这个变量或类似情况，就用这个经验值快速估计“现在 branch 它大概值不值”。
  - 它和 strong branching 的关系是：
    strong branching 是“现在就真的去试一遍，再做决定”，`pseudo-cost` 是“根据过去的试验经验，便宜地猜一遍”。
  - `conflict score` 可以理解成“这个变量和搜索冲突/推理困难的关联度有多高”。
    它更像是从求解过程中积累的冲突信息里判断：branch 这个变量，可能更有助于尽快触发传播、剪枝或暴露矛盾。
  - 在这一章里，你不用把这两个 heuristic 的实现细节啃太深。更重要的是抓住它们在作者论证中的角色：
    它们都是比 full strong branching 便宜得多的替代信号，所以 modern solver 才会采用 hybrid branching，而不是全程跑 strong branching。
- 你可以先把三者记成一条精简对比线：
  - strong branching：最前瞻、最贵，像“现算现看”。
  - pseudo-cost：经验估计、便宜，像“看历史账本”。
  - conflict score：冲突导向、便宜，像“看哪里更容易逼出矛盾和剪枝”。
- Q3 回答：引入 MDP 之所以有利于理解 learning，是因为它把“学习分支规则”从一句口号拆成了标准的 learning problem。
  - 没有 MDP 语言时，你只能模糊地说“模型根据当前节点选变量”。
  - 有了 MDP 语言后，你就可以明确地区分：
    state 是什么，
    action 是什么，
    action set 是什么，
    policy 学的是什么，
    一次决策如何影响后续整个求解轨迹。
- 这正是它和 reinforcement learning 强相关的原因。
  - RL 天生就是为 MDP 里的 policy optimization 设计的。
  - 一旦把 branching 写成 MDP，就很自然会想到：能不能直接优化长期回报，比如更短 solve time、更少 B&B nodes、更小 tree？
- 但“强相关”不等于“必须使用”。
  - 这篇论文正好是在说：是的，问题形式上非常像 RL；
  - 但实际做起来，episode 太长、早期 policy 太差、instance-specific MDP 太大，所以直接 RL 很难训；
  - 因而作者退回到 imitation learning，先学一个高质量 expert 的局部动作。
- 这里你可以把作者的逻辑记成四步：
  1. branching 是 sequential decision making；
  2. sequential decision making 可以 formalize 成 MDP；
  3. formalize 成 MDP 后，RL 是自然候选；
  4. 但本文基于训练可行性，选择 imitation learning 而不是 RL。
- 一个很重要的小提醒：
  - MDP framing 帮你理解的是“全局过程”；
  - 本文真正训练的 supervision 仍然是“局部动作标签”，也就是 expert 在某个状态下选了哪个 branching variable。
- Round 2: 回答你后来补充的两个问题。
- Q4 回答：MDP 数学公式里最容易混淆的就是两个 `p` 和一个 `\pi`，你可以这样记。
  - $\pi(a \mid s)$ 读作“pi”。
    它表示 policy，也就是在状态 $s$ 下选择动作 $a$ 的规则或概率。
    在这篇论文里，它对应“brancher 看到当前 solver 状态后，会选哪个变量 branch”。
  - $p(s_0)$ 里的 $p$ 表示 probability distribution。
    它是初始状态分布，也就是一次 episode 一开始抽到什么样 MILP instance、进入什么初始求解状态。
  - $p(s_{t+1} \mid s_t, a)$ 里的 $p$ 还是 probability distribution。
    它表示 transition，也就是在状态 $s_t$ 下采取动作 $a$ 后，solver 如何转移到下一个状态 $s_{t+1}$。
- 所以最短版记法是：
  - $\pi$：你要学的“决策规则”；
  - $p(\cdot)$：环境本身的“概率规律”。
- 放回本文语境里看：
  - $\pi$ 是 learned brancher 或 expert brancher；
  - $p$ 则包含 solver 的其余机制，比如扩树、解子 LP、做 heuristics、剪枝、再选下一叶节点。
- 这也是为什么作者说自己学习的是 policy，而不是整个 solver dynamics。
- Q5 回答：`reinforcement learning` 和 `imitation learning` 的最大区别，在于“老师从哪里来”和“优化信号是什么”。
  - imitation learning：
    先有一个 expert teacher，模型直接学“老师在这个状态下会怎么做”。
    这篇论文里，teacher 就是 `strong branching`。
  - reinforcement learning：
    不直接抄老师动作，而是让模型通过和环境反复交互，自己试出来什么动作序列能带来更高长期回报。
    这里的长期回报可以是更少 nodes、更短 solve time、或更好的 bound 改善。
- 你可以把两者的区别记成一句话：
  - imitation learning 学“像老师”；
  - reinforcement learning 学“拿高回报”。
- 放到这篇论文里，差别更具体：
  - imitation learning 的标签是局部动作标签：
    在状态 $s_t$ 下，expert 选了哪个 branching variable。
  - reinforcement learning 的信号是全局结果信号：
    这整条 branching trajectory 最后到底让求解变快了多少、树变小了多少。
- 为什么本文选 imitation learning 而不是 RL？
  - 因为这里有一个高质量 expert，`strong branching` 虽慢但决策质量高，很适合拿来当老师。
  - 同时作者也明确说了 RL 的训练难点：episode 很长、初始 policy 太差、instance-specific MDP 很大，导致训练成本很高。
- 但两者也不是完全对立的。
  - imitation learning 往往更容易起步，能先学到一个靠谱 policy；
  - RL 理论上更贴近最终目标，因为它能直接优化长期 solver performance；
  - 所以后来很多 follow-up work 会考虑 imitation 初始化、再做 RL 微调，或者直接探索更强的 downstream objective。
- 就本文这一章的阅读重点而言，你只要先抓住：
  - authors 用 MDP 语言定义问题；
  - 他们承认 RL 很自然；
  - 但他们最终选择 imitation learning，因为它在这个任务上更可训练、更现实。
- 如果你提问，我会优先补三类内容：
  - Figure 1 里每个符号在 solver 里的具体对应
  - `strong branching -> hybrid branching -> learned policy` 这条过渡线
  - 为什么 MDP framing 不等于必须用 RL
