# PPT Content

## 使用说明
- 本文件是基于 `07_guided_reading_cn/` 全部章节整理出的科研汇报内容库。
- 适用场景：`10-15 分钟` 论文汇报。
- 组织方式：沿用 `slides_outline.md` 的 `8 页` 结构，但每页提供更完整的可上屏内容。
- 数学公式全部使用 Preview 可渲染的 LaTeX 写法。
- 使用建议：
  - 如果想做更精简的 PPT：每页从“核心内容”里保留 3-5 条即可。
  - 如果想做讲稿版 PPT：保留“补充说明”内容作为演讲口播。

---

## Slide 1
### 标题
论文定位与问题背景

### 核心内容
- 论文题目：**Exact Combinatorial Optimization with Graph Convolutional Neural Networks**
- 作者：Maxime Gasse, Didier Chetelat, Nicola Ferroni, Laurent Charlin, Andrea Lodi
- 会议：NeurIPS 2019
- 研究对象不是“神经网络端到端求解 MILP”，而是 **MILP exact solver 中的 branching variable selection**
- 论文所处的问题链条是：
  - 组合优化问题
  - 可统一建模为 MILP
  - 精确求解依赖 branch-and-bound
  - branch-and-bound 中的 branching rule 对搜索树规模和求解时间影响很大
- strong branching 是传统上非常强的 branching rule，但每一步都很贵，无法在整棵树上放心全程使用

### 建议上屏文字
> 这篇论文不是在做 neural combinatorial optimizer。  
> 它研究的是：能否在 **exact MILP solver** 内部，用一个 learned branching policy 去近似 strong branching 的决策质量，同时大幅降低每步决策成本。

### 必放定义/公式
MILP 的基本定义可以在背景页先给出：

$$
\arg\min_x \left\{ c^\top x \mid Ax \le b,\ l \le x \le u,\ x \in \mathbb{Z}^p \times \mathbb{R}^{n-p} \right\}. \tag{1}
$$

### 补充说明
- 这页的主要任务是“把论文边界讲清”，防止听众误解成“GNN 直接替代 B&B”。
- 如果想做图，可以画成：
  `MILP -> SCIP / B&B -> Branching module -> Learned GCNN brancher`

---

## Slide 2
### 标题
Research Question

### 核心内容
- 核心研究问题可以压成一句话：
  > 能否学习一个 branch-and-bound variable selection policy，使其逼近 strong branching 的决策质量，同时减少特征工程，并泛化到比训练时更大的 MILP 实例？
- 具体拆成三个子问题：
  - 能否用学习方法近似 strong branching，而不承担它的高在线代价？
  - 能否把 solver state 编码成一种更自然、更少人工特征工程的结构表示？
  - 能否在 small / easy 实例上训练后，泛化到 medium / hard 实例？
- 评价目标不是 imitation accuracy 本身，而是：
  - `solve time`
  - `B&B nodes`
  - `wins over solved instances`

### 建议上屏文字
> 论文不是在问“能不能把 strong branching 学出来”。  
> 它更强的提问是：  
> **能否在真实 solver 里学到一个既快又能泛化的 branching policy？**

### 必放定义/公式
branching 的局部动作对象可以在这一页配上：

$$
x_i \le \lfloor x_i^\star \rfloor \;\lor\; x_i \ge \lceil x_i^\star \rceil,
\qquad \exists i \le p \mid x_i^\star \notin \mathbb{Z}. \tag{2}
$$

它说明 branch 的本质就是：对某个当前取非整数值的变量做二分切割。

### 补充说明
- 这一页要强调：本文最终想优化的是 **solver performance**，不是单纯分类指标。
- 这会为后面“局部 imitation 目标 vs 全局 solver 指标”的差异埋伏笔。

---

## Slide 3
### 标题
High-Level Algorithmic Idea

### 核心内容
- 方法主线可以概括为四步：
  1. 把当前 B&B 节点状态编码成一张 variable-constraint bipartite graph
  2. 用 GCNN 读取这张图
  3. 输出当前 candidate branching variables 上的动作分布
  4. 用 imitation learning 模仿 strong branching 的动作选择
- 输入不是手工拼接的大量 solver features，而是 MILP 的自然图结构
- 训练不是 RL，而是 behavioral cloning
- 方法只替换 variable selection，solver 其它部分保持不变

### 建议上屏文字
> **State 是图，Action 是“选哪个变量 branch”，Learning 是 imitation of strong branching。**

### 必放定义/公式
当前状态编码为：

$$
s_t = (G, C, E, V),
$$

其中：
- $G$：二部图结构
- $C$：constraint node features
- $V$：variable node features
- $E$：edge features

### 补充说明
- 这一页建议配一张 Figure 2 的简化版流程图。
- 如果口头展开，可强调本文的主线不是“用了 GNN”，而是“结构化状态表示 + imitation learning + exact solver integration”。

---

## Slide 4
### 标题
Training Label and Dataset

### 核心内容
- 本文监督数据不是现成标注集，而是作者自己在 SCIP 运行过程中采出来的
- 每条训练样本是一个 **state-action pair**：
  - `state`：当前 B&B 节点的图表示
  - `action`：strong branching expert 选择的 branching variable
- 标签不是：
  - 最终最优解
  - tree size
  - continuous strong branching score
- 四类 benchmark family：
  - set covering
  - combinatorial auction
  - capacitated facility location
  - maximum independent set
- 每个 benchmark：
  - 约 `100,000` 个训练 branching samples
  - 约 `20,000` 个验证 branching samples

### 建议上屏文字
> 这篇论文学的不是“解是什么”，而是“当前节点 expert 会选哪个变量 branch”。

### 必放定义/公式
训练集可写为：

$$
\mathcal{D} = \{(s_i, a_i^\star)\}_{i=1}^{N},
$$

其中：
- $s_i$：第 $i$ 个节点状态
- $a_i^\star$：strong branching 在该状态下的 expert action

### 关键实现细节
- 作者不能直接把 SCIP 内建 strong branching 当成纯标签器，因为它会带来 solver side-effects
- 因此他们重写了一个 side-effect-free 的 oracle：
  - `vanillafullstrong`

### 补充说明
- “100,000 / 20,000” 指的是 branching samples 数，不是 instance 数。
- 数据采集方式是从实例集合里 **有放回采样**，不断运行直到凑够样本数。

---

## Slide 5
### 标题
Network Structure

### 核心内容
- 图结构输入：
  - 左边是 constraint nodes
  - 右边是 variable nodes
  - 若 $A_{ij} \neq 0$，则约束 $i$ 与变量 $j$ 连边
- 模型是一层 bipartite GCNN，但拆成两个 half-convolutions：
  - variable-to-constraint
  - constraint-to-variable
- 核心直觉：
  - branch 的对象是变量
  - 变量的重要性取决于它与哪些约束相连
  - 所以必须先让变量信息传到约束，再由约束传回变量
- 中间使用 2-layer MLP 作为 message/update functions
- 最终只在 variable side 上输出分数

### 必放定义/公式
图结构定义：

$$
C \in \mathbb{R}^{m \times c}, \qquad
V \in \mathbb{R}^{n \times d}, \qquad
E \in \mathbb{R}^{m \times n \times e}.
$$

连边规则：

$$
(i,j) \in E \iff A_{ij} \neq 0.
$$

消息传递的核心公式：

$$
c_i \leftarrow f_C\left(c_i,\ \sum_{(i,j)\in E} g_C(c_i, v_j, e_{i,j})\right),
$$

$$
v_j \leftarrow f_V\left(v_j,\ \sum_{(i,j)\in E} g_V(c_i, v_j, e_{i,j})\right). \tag{4}
$$

其中 $f_C, f_V, g_C, g_V$ 都是 2-layer perceptrons with ReLU。

### 关键设计点
- 使用 `sum aggregation` 而不是 `mean aggregation`
- 在 sum 之后加 `prenorm`
- 理由是保留计数能力并稳定训练

### 补充说明
- 可以在页底补一句：
  > 本文的创新不在“发明全新 GCNN 层”，而在“把图表示、消息传递方向和 branching 动作空间高度任务化地对齐”。

---

## Slide 6
### 标题
Network Output, Loss, and Inference Policy

### 核心内容
- 这篇论文最干净的一点，是 `output / loss / inference` 这条链在局部动作空间里高度对齐
- Output：
  - 最终输出的是当前 candidate branching variables 上的 masked softmax 分布
  - 不是 subtree size
  - 不是 continuous branching score
  - 不是最终解
- Loss：
  - 用 expert action 做 cross-entropy / negative log-likelihood
- Inference：
  - 仍在同一个 candidate action set 内按模型分数排序
  - 选择 top candidate 接回 SCIP 做 branching

### 必放定义/公式
训练损失：

$$
L(\theta) = -\frac{1}{N}\sum_{(s,a^\star)\in\mathcal{D}} \log \pi_\theta(a^\star \mid s). \tag{3}
$$

这表示：最大化 expert action 在当前状态下的概率。

输出分布可以按 masked softmax 形式展示为：

$$
\pi_\theta(a=j \mid s)
=
\frac{\exp(z_j)}{\sum_{k \in A(s)} \exp(z_k)},
\qquad j \in A(s),
$$

其中：
- $z_j$ 是 variable node $j$ 的最终分数
- $A(s)$ 是当前合法 candidate action set

### 建议上屏文字
> Output 是动作分布，Loss 是 expert-action cross-entropy，Inference 还是在同一个动作空间里选动作。  
> 所以本文在局部动作层面非常对齐。

### 需要点出的关键判断
- 本文是 **action classification**
- 不是 **score regression**
- 不是 **candidate ranking** 的主路线

### 补充说明
- 如果要加一句更深一点的评价：
  > 真正的 mismatch 不在 output/loss/inference 局部链条内部，而在更高层：训练学 imitation，最终评估看 solve time / nodes。

---

## Slide 7
### 标题
Novel Contribution and Main Experimental Results

### 核心内容
- Novel contribution 不能说成“第一次做 learned branching”
- 更准确地说，创新点有四层：
  - 用 bipartite MILP graph 做 state encoding
  - 用 GCNN 做 branching policy parametrization
  - 用 action-classification 式 imitation learning，而不是 ranking/regression
  - 在 essentially full-fledged SCIP solver 中做真实比较
- 主实验结果分三层证据：
  1. Table 1：GCNN 在四类问题上 imitation accuracy 全面优于 TREES、SVMRANK、LMART
  2. Table 2：accuracy 提升会反映到更少的 B&B nodes
  3. 最终在大多数配置下，GCNN 的 solve time 也优于 SCIP 默认 RPB

### 建议上屏文字
> 论文真正新的是：  
> **graph state + GCNN policy + action-classification imitation + full-solver evaluation**

### 必须讲清的实验逻辑
- `Nodes` 和 `Time` 必须一起看：
  - 节点更少通常说明决策质量更高
  - 但每步推理开销也会影响总时间
- 文中给出一个很好的例子：
  - SVMRANK 节点略优于 LMART
  - 但时间更差
  - 原因是 `running time / number of nodes trade-off` 更差

### 关键实验结论
- GCNN 整体表现最强
- 训练只在 easy 实例上做，但在 medium / hard 上仍然能保持较强表现
- 在 set covering 和 combinatorial auction 的 medium / hard 上，对默认 RPB 的 node 优势特别明显
- FSB 作为老师，树虽然很小，但总时间不竞争
- Maximum Independent Set 更难泛化，是重要边界案例

### 补充说明
- 这一页可以配 `Table 1 + Table 2` 的裁剪图。
- 若听众偏优化方向，可以强调“不是玩具环境，而是在完整 solver 中比较”的现实意义。

---

## Slide 8
### 标题
Strengths, Limitations, and Final Takeaway

### 核心内容
- Ablation（Table 3）支持两个关键架构判断：
  - `sum` 比 `mean` 更适合 branching
  - `prenorm` 对训练稳定和大规模泛化有帮助
- Strengths：
  - 减少特征工程
  - output-loss-inference 局部对齐好
  - 在完整 SCIP 中验证有效
- Limitations：
  - state 只是 solver state 的子集，严格说更像 POMDP
  - imitation objective 不是最终 solver objective
  - 泛化强但不是无限强，随着规模继续拉大性能会下降
  - 更深更大的模型不一定更好，因为 inference cost 可能压过节点收益

### 必放定义/公式
prenorm：

$$
x \leftarrow \frac{x - \beta}{\sigma}.
$$

它在 sum aggregation 之后用于稳定训练。

MDP 轨迹表达式可作为全文方法总结公式：

$$
p_\pi(\tau) = p(s_0)\prod_{t=0}^{T-1}\sum_{a \in A(s_t)} \pi(a \mid s_t)\, p(s_{t+1} \mid s_t, a).
$$

它说明 branching 是一个长期序列决策问题，但本文最终选的是 imitation learning，而不是直接 RL。

### 最终 Takeaway
建议最后一页直接写成一句结论：

> 这篇论文最重要的贡献，不是“把 GNN 用到了 MILP”这么简单，  
> 而是证明了 **结构化 solver state + GCNN policy + imitation learning**  
> 可以作为一种有效的 learned branching 路线，真正嵌回 exact MILP solver，并在现实评测中取得优势。

### 可作为结尾口播的版本
> 如果只记一句话，这篇论文告诉我们：  
> 学习方法不是要替代 exact solver，而是可以非常有效地改造 solver 中最关键、最昂贵的局部决策环节。

---

## 备选附录
### 可以放在备份页但不一定上主报告的内容
- Table 2 的 state / edge / variable feature 细节表
- `vanillafullstrong` 为什么必须重写
- `root-only cuts` 为什么让图结构在整棵树里保持稳定
- `SCIP` 是什么、`RPB` / `FSB` 各是什么角色
- 为什么本文不是 regression / ranking，而是 action classification
- 为什么 `Nodes` 更少不保证 `Time` 更快

### 适合做问答时补充的关键句
- `masked softmax` 的 mask 不是模型自己预测出来的，而是 solver 当前合法动作集合先定义好的。
- 图上的 variable nodes 可以继续参与 message passing，但最终只有合法 candidate variables 会进入动作分布。
- 本文的强点是局部动作空间对齐好，弱点是全局目标仍然只能通过 surrogate imitation 间接优化。
