# Chapter 02 - Problem Setup & Hypergraph Construction

## 1. 本章范围
- Status: active
- Source: Section 3.1 Preliminaries; Definition 1; Section 3.2.1 Hypergraph Construction
- Why this chapter: 这一章是全文的“符号地基”。如果这一层没读顺，后面卷积公式、attention 聚合、自监督目标都会像在空中飘着。我们先把数据对象、预测目标、超图定义，以及 motif 如何变成三个 hypergraph 讲清楚。

## 2. 阅读前你先抓住这三个问题
1. 论文在形式上到底处理什么推荐问题？$U$、$I$、$R$、$S$、$\hat r_{ui}$、$P^{(l)}$、$Q^{(l)}$ 分别是什么？
2. 这里的 hypergraph 和普通 graph 的关键区别是什么？为什么作者要引入 incidence matrix $H$、vertex degree matrix $D$、edge degree matrix $L$？
3. 作者到底是怎么把“带语义的三角关系”变成三个可计算的 motif-induced hypergraphs 的？

## 3. 英文原文分段
### Passage 1
Let $U=\{u_1,u_2,\ldots,u_m\}$ denote the user set and $I=\{i_1,i_2,\ldots,i_n\}$ denote the item set. $I(u)$ is the set of items consumed by user $u$. $R\in \mathbb{R}^{m\times n}$ is a binary matrix that stores user-item interactions. For each pair $(u,i)$, $r_{ui}=1$ indicates that user $u$ consumed item $i$, while $r_{ui}=0$ means that item $i$ is unexposed to user $u$, or user $u$ is not interested in item $i$.

### Passage 2
In this paper, we focus on top-$K$ recommendation, and $\hat r_{ui}$ denotes the probability of item $i$ to be recommended to user $u$. As for the social relations, we use $S\in \mathbb{R}^{m\times m}$ to denote the relation matrix which is asymmetric because we work on directed social networks. In our model, we have multiple convolutional layers, and we use $\{P^{(1)},P^{(2)},\ldots,P^{(l)}\}$ and $\{Q^{(1)},Q^{(2)},\ldots,Q^{(l)}\}$ to denote the user and item embeddings learned at each layer.

### Passage 3
Let $G=(V,E)$ denote a hypergraph, where $V$ is the vertex set and $E$ is the edge set containing hyperedges. Each hyperedge $\epsilon\in E$ can contain any number of vertices and is assigned a positive weight. The hypergraph can be represented by an incidence matrix $H$ where $H_{i\epsilon}=1$ if hyperedge $\epsilon$ contains vertex $v_i$, otherwise $0$. The vertex and edge degree matrices are denoted by $D$ and $L$, respectively.

### Passage 4
To formulate the high-order information among users, we first align the social network and user-item interaction graph in social recommender systems and then build hypergraphs over this heterogeneous network. Unlike prior models which construct hyperedges by unifying given types of entities, our model constructs hyperedges according to the graph structure.

### Passage 5
As the relations in social networks are often directed, the connectivity of social networks can be of various types. In this paper, we use a set of carefully designed motifs to depict the common types of triangular structures in social networks, which guide the hypergraph construction. We only focus on triangular motifs because of the ubiquitous triadic closure in social networks.

### Passage 6
Given motifs $M_1-M_{10}$, we categorize them into three groups according to the underlying semantics. $M_1-M_7$ summarize all the possible triangular relations in explicit social networks and describe the high-order social connectivity like “having a mutual friend”. We name this group “Social Motifs”. $M_8-M_9$ represent the compound relation, that is, “friends purchasing the same item”. We name them “Joint Motifs”. $M_{10}$ defines the implicit high-order social relation that users who are not socially connected but purchased the same item. We name $M_{10}$ “Purchase Motif”.

### Passage 7
Under the regulation of these three types of motifs, we can construct three hypergraphs that contain different high-order user relation patterns. We use the incidence matrices $H_s$, $H_j$ and $H_p$ to represent these three motif-induced hypergraphs, respectively, where each column of these matrices denotes a hyperedge.

### Passage 8
We use $A^{M_k}_{i,j}$ to represent that vertices $i$ and $j$ appear in one instance of motif $M_k$. As two vertices can appear in multiple instances of $M_k$, $A^{M_k}_{i,j}$ is computed by:
$$
A^{M_k}_{i,j} = \#(i,j \text{ occur in the same instance of } M_k).
$$

## 4. 重点概念与术语
- top-$K$ recommendation: 目标不是回归一个精确评分，而是给每个用户排出一个推荐列表。
- $R$: 用户-物品二值交互矩阵。这里的监督信号首先是“是否有过交互”。
- $S$: 用户-用户社交关系矩阵，而且是有向的，所以一般不对称。
- $P^{(l)}, Q^{(l)}$: 第 $l$ 层学到的 user / item embeddings。
- hypergraph: 一条 hyperedge 可以同时连接多个顶点，不再局限于二元边。
- incidence matrix $H$: 用来表示“哪个顶点属于哪条 hyperedge”。
- vertex degree matrix $D$: 每个顶点参加了多少条 hyperedge。
- edge degree matrix $L$: 每条 hyperedge 里包含多少个顶点。
- motif: 局部结构模板，用来定义“什么样的三角关系值得被当成一种高阶关系”。
- triadic closure: 社交网络中三角闭包倾向，是作者只先看三角 motif 的直觉依据。
- motif-induced hypergraph: 先按 motif 找到结构实例，再把这些实例变成 hypergraph 的超边。

## 5. 本章核心内容
### 5.1 用中文讲清楚
这一章真正做的事，不是开始训练模型，而是先把“论文要处理的对象”摆平。最外层的数据对象有四个：用户集合 $U$、物品集合 $I$、用户-物品交互矩阵 $R$、用户-用户社交矩阵 $S$。其中 $R$ 是二值的，表示是否发生过消费/交互；$\hat r_{ui}$ 则是模型最终要产出的推荐分数，用来做 top-$K$ 排序。也就是说，从任务角度看，本文仍然是一个标准的 social recommendation 排序问题。

紧接着作者把表示学习对象也定义出来了：$P^{(l)}$ 是第 $l$ 层的用户表示，$Q^{(l)}$ 是第 $l$ 层的物品表示。你可以把这一步理解成“先记号备案”，因为后面所有卷积和聚合，都会围绕这两组 embedding 展开。

然后作者插入了一个超图定义，这一步很关键。普通图的 edge 只能连两个点，但作者想表达的是“多个用户共同处在某种高阶关系模式里”，所以需要 hypergraph。这里的 incidence matrix $H$ 比邻接矩阵更适合，因为它不是只问“点和点之间有没有边”，而是直接记录“点是否属于某条超边”。如果某个用户落在很多 motif 实例里，那么她对应的顶点度就高；如果某条超边里包含很多用户，那么它的 edge degree 就高。

接下来才是这一章的真正核心：作者不是直接在原始 social graph 上做 hypergraph convolution，而是先把 social network 和 user-item graph 对齐，在这个异构结构上寻找“有语义的三角关系”。为什么只先看三角？因为作者认为 social network 里最常见、最稳定、又计算上可控的高阶局部结构就是 triadic closure。于是他们手工定义了 $M_1-M_{10}$ 十种三角 motif，再把这些 motif 归成三大类。

第一类是 `Social Motifs`，对应 $M_1-M_7$，它们只利用显式社交关系，想表达的是各种高阶社会连接。第二类是 `Joint Motifs`，对应 $M_8-M_9$，它们把社交关系和共同购买揉在一起，表达“社交连接 + 共购”这种更强的复合关系。第三类是 `Purchase Motif`，对应 $M_{10}$，它强调的是“虽然没有显式社交边，但买过同样物品”，这是隐式高阶关系。

最值得你记住的一步是：`motif` 只是规则，`hypergraph` 才是数据结构。作者先用 motif 规则在异构图里找到一个个三角实例，再把这些实例转成三组 hyperedges，于是得到三个 hypergraphs：$H_s$、$H_j$、$H_p$。从论文后续用 $H_s,H_j,H_p$ 去传播 user embeddings 这点可以推断，这三个 hypergraph 的顶点主体是用户；物品虽然参与了某些 motif 的判定，但更像“诱导用户高阶关系的证据”，而不是后续 hypergraph channel 中被直接卷积的顶点。这一点是我根据 Section 3.2.1 与后续公式关系做的推断。

最后，作者还定义了 $A^{M_k}_{i,j}$，表示用户 $i$ 和用户 $j$ 一起出现在 motif $M_k$ 的实例里多少次。你可以把它理解成“motif 诱导出来的用户-用户共现强度”。Table 1 的作用，就是把这些 motif 实例数高效地算出来，从而避免真的把所有三角形一个个枚举成超边再慢慢处理。

### 5.2 这段在全文中的作用
这一章是后面所有公式的词典和地图。

- Section 3.1 告诉你：模型的输入对象、预测目标和表示对象分别是什么。
- Definition 1 告诉你：作者为什么有理由把问题从普通图迁移到 hypergraph。
- Section 3.2.1 告诉你：三个 channel 的输入不是凭空出现的，而是由三类 motif-induced hypergraphs 构造出来的。

如果这一章没吃透，后面你看到 $H_c$、$A_c$、$P_c^{(l)}$ 时，会分不清“这是原始社交图里的边”“这是 motif 规则”“还是这是 hypergraph channel 里的传播结构”。

### 5.3 容易误解的点
- $R$ 是二值交互矩阵，不是显式评分矩阵；本文当前设定更接近隐反馈推荐。
- $\hat r_{ui}$ 虽然文中写成 probability，但在后面训练里更应理解成 ranking score。
- 作者不是把用户和物品一起直接塞进同一个 hypergraph channel 里卷积；更准确地说，是先用 user-item relation 辅助诱导用户之间的高阶关系，再把这些关系写成用户超图。这一点是基于后续传播公式的推断。
- $M_1-M_{10}$ 不是十个训练样本，也不是十个 channel，而是十种 motif 模板，被归并成三个语义通道。
- `problem formulation` 在这篇论文里并不是一个单独的 optimization problem 小节，它更像“符号定义 + 结构建模前提”的组合。

## 6. 证据指针
- Section 3.1 Preliminaries: $U$、$I$、$R$、$S$、$\hat r_{ui}$、$P^{(l)}$、$Q^{(l)}$ 的定义。
- Definition 1: hypergraph、incidence matrix $H$、degree matrices $D/L$ 的正式定义。
- Section 3.2.1 first paragraph: 为什么要对齐 social network 与 user-item graph，并按 graph structure 构造 hyperedges。
- Section 3.2.1 motif paragraph; Figure 2: 为什么只看 triangular motifs，以及三类 motif 的语义划分。
- Section 3.2.1 later paragraph: $H_s$、$H_j$、$H_p$ 三个 motif-induced hypergraphs 的定义。
- Section 3.2.1; Table 1: $A^{M_k}_{i,j}$ 作为 motif 共现计数，以及高效计算思路。

## 7. 一分钟回顾
- Chapter 2 先把问题对象摆平：用户、物品、交互矩阵、社交矩阵、预测分数和多层 embedding。
- 作者引入 hypergraph，是因为他们要表达“多个用户共同处在某种高阶关系模式里”，普通 pairwise edge 不够。
- 三个 channel 的输入本质上是三组由 motif 诱导出来的用户超图：social、joint、purchase。

## 8. 你的问答区
- 你可以直接问某个符号，比如 `$H$`、`$A^{M_k}$`、`$D/L$`、`$S$` 为什么不对称。
- 你也可以让我单独讲一遍“motif 是规则，hypergraph 是结果”这句话到底怎么落到 Figure 2 上。
- 如果你想自测，先试着回答：为什么作者不直接在原始 social graph 上做三层 GNN，而还要先构造 motif-induced hypergraph？
- 如果你觉得这一章已经顺了，直接回复“进入下一章”即可。

## 9. Codex 补充讲解
- Round 0: Chapter 2 已初始化。
- 先给你一个四层对象速记版：
  1. 原始数据层：$U, I, R, S$
  2. 预测目标层：$\hat r_{ui}$
  3. 结构建模层：$H_s, H_j, H_p$ 或对应的 motif-induced adjacency
  4. 表示学习层：$P^{(l)}, Q^{(l)}$
- 你读这一章时，最重要的是别把三件事混掉：
  1. `motif` 是局部结构模板。
  2. `hyperedge` 是某个模板在真实图里的一个实例。
  3. `channel` 是拿一整类这类实例去做编码的分支。
- 关于 hypergraph 为什么比普通 graph 更合适：
  如果你只用 pairwise edge，你最多能说“用户 A 和用户 B 连着”；但作者现在想表达的是“用户 A、B、C 一起构成了某种有语义的关系模式”。这种“共同属于一个结构”的信息，用 hyperedge 表达更自然。
- 关于 $A^{M_k}_{i,j}$ 的直觉：
  它不是最终要学的参数，而是统计量。它告诉你“用户 $i$ 和用户 $j$ 在 motif $M_k$ 里共同出现了多少次”。共同出现次数越多，说明这对用户在该类高阶关系下越接近。
- 如果你现在只想抓住 Chapter 2 的一句话：
  `problem formulation` 这一章真正建立的是一条管线：`交互与社交原始数据 -> motif 规则 -> 三类用户超图 -> 后续通道卷积的输入`。
