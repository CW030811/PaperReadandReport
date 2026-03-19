# Chapter 03 - MILP Problem Definition

## 1. 本章范围
- Status: active
- Source: Section 3.1 Problem definition
- Why this chapter: 从这一章开始，论文进入“正式定义对象”的阶段。
  你现在要先吃透 MILP、LP relaxation、上下界、以及一次 branching 在数学上到底做了什么，后面 Chapter04 才能顺畅读 B&B 和 MDP framing。

## 2. 阅读前你先抓住这三个问题
1. 公式 (1) 里的 MILP 到底在约束什么，哪些变量必须是整数，哪些可以连续？
2. 为什么把整数约束放松成 LP relaxation 后，得到的是一个 `lower bound`？
3. 公式 (2) 的左右两个子问题，本质上为什么只是在当前变量上加 tighter bounds？

## 3. 英文原文分段
### Passage 1
A mixed-integer linear program is an optimization problem of the form

$$
\arg\min_x \left\{ c^\top x \mid Ax \le b,\ l \le x \le u,\ x \in \mathbb{Z}^p \times \mathbb{R}^{n-p} \right\}, \tag{1}
$$

where $c \in \mathbb{R}^n$ is called the objective coefficient vector, $A \in \mathbb{R}^{m \times n}$ the constraint coefficient matrix, $b \in \mathbb{R}^m$ the constraint right-hand-side vector, $l, u \in \mathbb{R}^n$ respectively the lower and upper variable bound vectors, and $p \le n$ the number of integer variables. Under this representation, the size of a MILP is typically measured by the number of rows ($m$) and columns ($n$) of the constraint matrix.

### Passage 2
By relaxing the integrality constraint, one obtains a continuous linear program (LP) whose solution provides a lower bound to (1), and can be solved efficiently using, for example, the simplex algorithm. If a solution to the LP relaxation respects the original integrality constraint, then it is also a solution to (1).

### Passage 3
If not, then one may decompose the LP relaxation into two sub-problems, by splitting the feasible region according to a variable that does not respect integrality in the current LP solution $x^\star$,

$$
x_i \le \lfloor x_i^\star \rfloor \;\lor\; x_i \ge \lceil x_i^\star \rceil,\quad \exists i \le p \mid x_i^\star \notin \mathbb{Z}, \tag{2}
$$

where $\lfloor \cdot \rfloor$ and $\lceil \cdot \rceil$ respectively denote the floor and ceil functions. In practice, the two sub-problems will only differ from the parent LP in the variable bounds for $x_i$, which get updated to $u_i = \lfloor x_i^\star \rfloor$ in the left child and $l_i = \lceil x_i^\star \rceil$ in the right child.

### Passage 4
The branch-and-bound algorithm, in its simplest formulation, repeatedly performs this binary decomposition, giving rise to a search tree. By design, the best LP solution in the leaf nodes of the tree provides a lower bound to the original MILP, whereas the best integral LP solution (if any) provides an upper bound. The solving process stops whenever both the upper and lower bounds are equal or when the feasible regions do not decompose anymore, thereby providing a certificate of optimality or infeasibility, respectively.

## 4. 重点概念与术语
- mixed-integer linear program, MILP:
  一部分变量必须取整数，另一部分变量可以连续的线性优化问题。这里的 `mixed` 指“整数变量 + 连续变量”混合存在。
- objective coefficient vector `c`:
  目标函数里的系数，决定每个变量对目标值的贡献。
- constraint matrix `A` / right-hand side `b`:
  线性约束 $Ax \le b$ 的主体。`A` 决定系数结构，`b` 决定约束右端。
- lower bound / upper bound:
  对最优目标值的下界和上界。对最小化问题，lower bound 越高越接近最优值，upper bound 来自当前已找到的可行整数解。
- integrality constraint:
  整数约束。也就是前 `p` 个变量不能随便取实数，必须落在整数点上。
- LP relaxation:
  把整数约束先去掉，只保留线性约束和变量上下界，得到一个更容易解的 LP。
- fractional variable:
  在当前 LP 解里取到了非整数值、但本来应该取整数的变量。它是 branch 的候选对象。
- floor / ceil split:
  对一个 fractional variable 做二分切割。左边强制它不大于下取整，右边强制它不小于上取整。
- certificate of optimality / infeasibility:
  exact solver 的关键价值之一。不是只给一个答案，而是能在终止时给“已经最优”或“不可行”的证明性结论。

## 5. 本章核心内容
### 5.1 用中文讲清楚
这一章在做一件很基础但很关键的事：把“论文到底在解什么对象”写严谨。公式 (1) 告诉你，这篇论文的基本对象是 MILP，也就是在线性约束 $Ax \le b$ 和变量上下界 $l \le x \le u$ 之下，最小化 $c^\top x$，同时要求前 $p$ 个变量必须取整数。

你可以先把它理解成两层结构。第一层是普通线性规划的壳子：目标函数和约束都是线性的。第二层是困难真正来自哪里：有些变量不能取连续实数，只能取整数。正是这层 integrality constraint 让问题通常变难，也让后面的 branch-and-bound 变得有必要。

接着作者引入 `LP relaxation`。它的意思不是“换了一个问题”，而是“先临时忽略整数要求，求一个更容易的连续版本”。因为对最小化问题来说，去掉整数限制只会让可行域变大，不会变小，所以 relaxed LP 的最优值不可能比原始 MILP 更差；因此它给出的就是原问题最优值的一个 `lower bound`。

这里有一个非常重要的分叉：

- 如果 LP relaxation 的最优解碰巧已经满足整数约束，那么它不仅是 LP 的最优解，也是原 MILP 的一个可行解，而且由于 LP 已经给出了 lower bound，这时这个解实际上就已经把原问题解掉了。
- 如果 LP relaxation 的解里有某个该为整数的变量却取了小数值，比如 $x_i = 2.7$，那就不能直接接受。此时 B&B 的基本动作就是选这个变量，把原问题拆成两个子问题：一个要求 $x_i \le 2$，另一个要求 $x_i \ge 3$。

这就是公式 (2) 的意思。它不是在做复杂的新建模，而是在当前 LP 节点上沿着某个 fractional variable 加两条互斥的 bound 约束，把原来包含 `2.7` 的连续可行域切成左右两块。之后分别求这两个 child problem，就形成了搜索树。

作者还特别强调了一个 solver 视角的重要点：在实践里，这两个子问题通常并不需要重写整套模型，它们相对父节点只是在这个变量的 bound 上更紧了。左子节点把上界改成 $\lfloor x_i^\star \rfloor$，右子节点把下界改成 $\lceil x_i^\star \rceil$。这也是为什么后面 solver 可以高效地在树上持续推进。

### 5.2 这段在全文中的作用
Section 3.1 的功能是给后面所有“branching policy”讨论打地基。

- Chapter04 讲 branching rule 时，默认你已经知道“branch 一次”在数学上就是公式 (2) 这种 split。
- 后面讲 strong branching 时，所谓“评估一个候选变量值不值得 branch”，本质上就是在比较如果沿不同 fractional variable 去切，子问题的 bound 改善会怎样。
- 再后面讲图表示时，很多节点和边的含义也都建立在这里定义的 MILP 对象之上。

所以这一章虽然暂时没有机器学习内容，但它在全文里非常关键。没有这章，后面的 learned branching 很容易只剩口号，不知道 network 到底是在对什么对象做决策。

### 5.3 容易误解的点
- `LP relaxation provides a lower bound` 这句话只是在最小化问题语境下这么说；本论文这里确实写的是最小化形式，所以这样理解是对的。
- “LP 解满足整数约束”时，它就已经是 MILP 解，这个结论不是经验性的，而是由 relax 后可行域包含原可行域这一关系直接推出的。
- 公式 (2) 不是说一定只存在一个 fractional variable，而是说只要存在至少一个，就可以挑其中一个来 branch。到底挑哪一个，正是下一章 branching rule 要解决的问题。
- 本章里的 B&B 还是“simplest formulation”。真实 solver 里还会有 cuts、heuristics、presolving 等额外机制，但这些不影响这里的核心定义。

## 6. 证据指针
- Section 3.1 Equation (1): MILP 的正式定义、变量类型、目标和约束。
- Section 3.1 first paragraph after Equation (1): `c, A, b, l, u, p, m, n` 的符号解释，以及用 rows/columns 衡量 MILP 尺寸。
- Section 3.1 LP-relaxation paragraph: 为什么 LP relaxation 给出 lower bound，以及何时 LP 解直接成为 MILP 解。
- Section 3.1 Equation (2): 对 fractional variable 的左右分支切分。
- Section 3.1 final paragraph: 搜索树、lower/upper bounds，以及 optimality / infeasibility certificate。

## 7. 一分钟回顾
- MILP 是“线性壳子 + 部分变量必须取整数”的优化问题。
- LP relaxation 就是先去掉整数约束；对本文的最小化问题，它给出 lower bound。
- 如果 LP 解已经整数可行，就直接解决了当前问题；否则就沿某个 fractional variable 做左右切分。
- B&B 的搜索树，就是不断重复这种 split，再结合上下界逐步逼近最优解。

## 8. 你的问答区
- 你可以直接问：“为什么 relaxed LP 一定是 lower bound？”
- 你也可以贴一个具体例子，让我带你手算一次 $x_i = 2.7$ 时左右子问题是怎么来的。
- 如果你已经吃透这一章，直接写“这章吃透了”或“进入下一章”。

## 9. Codex 补充讲解
- Round 0: 已初始化，等待你的问题。
- 如果你提问，我会优先补三类内容：
  - 公式 (1) 和公式 (2) 的逐符号解释
  - lower bound / upper bound 在最小化问题里的直觉图像
  - “一个 branch 到底做了什么” 的小例子拆解
