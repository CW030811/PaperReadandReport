# Chapter 04 - Self-Supervision, Objective & Complexity

## 1. 本章范围
- Status: mastered
- Source: Section 3.2.4 Model Optimization; Section 3.3 Enhancing MHCN with Self-Supervised Learning; Section 3.4 Complexity Analysis
- Why this chapter: 前三章已经把“结构”搭起来了，这一章终于回答“模型到底怎么训练”。你需要把三件事分开看：主任务怎么学推荐排序，自监督辅助任务怎么补结构信息，以及整个模型的参数量和计算量到底落在哪些部件上。

## 2. 阅读前你先抓住这三个问题
1. `vanilla MHCN` 的主训练目标是什么，为什么作者选 BPR 而不是评分回归？
2. `S^2-MHCN` 的自监督任务具体在最大化什么信息，它为什么不是普通的 node-graph MI 复刻？
3. 最终总目标是怎样把 recommendation task 和 self-supervised task 合起来的，模型复杂度主要花在哪里？

## 3. 英文原文分段
### Passage 1
To learn the parameters of MHCN, we employ the Bayesian Personalized Ranking (BPR) loss, which is a pairwise loss that promotes an observed entry to be ranked higher than its unobserved counterparts.

### Passage 2
The predicted score is $\hat r_{u,i}=p_u^\top q_i$. Each time a triplet including the current user $u$, the positive item $i$ purchased by $u$, and the randomly sampled negative item $j$ which is disliked by or unknown to $u$, is fed to MHCN.

### Passage 3
Owing to the exploitation of high-order relations, MHCN shows great performance. However, a shortcoming of MHCN is that the aggregation operations might lead to a loss of high-order information, as different channels would learn embeddings with varying distributions on different hypergraphs.

### Passage 4
In this paper, we follow the primary & auxiliary paradigm, and set up a self-supervised auxiliary task to enhance the recommendation task (primary task). We inherit the merits of DGI to consider mutual information and further extend the graph-node MI maximization to a fine-grained level by exploiting the hierarchical structure in hypergraphs.

### Passage 5
Each row in $A_c$ represents a subgraph of the corresponding hypergraph centering around the user denoted by the row index. Then we can induce a hierarchy: user node $\leftarrow$ user-centered sub-hypergraph $\leftarrow$ hypergraph and create self-supervision signals from this structure.

### Passage 6
Our intuition of the self-supervised task is that the comprehensive user representation should reflect the user node’s local and global high-order connectivity patterns in different hypergraphs. The goal can be achieved by hierarchically maximizing the mutual information between representations of the user, the user-centered sub-hypergraph, and the hypergraph in each channel.

### Passage 7
We define the objective function of the self-supervised task as $L_s$. The discriminator function $f_D(\cdot)$ takes two vectors as input and then scores their agreement between them. We simply implement the discriminator as the dot product between two representations. Negative examples are created by corrupting the sub-hypergraph representations with row-wise and column-wise shuffling.

### Passage 8
Finally, we unify the objectives of the recommendation task (primary) and the task of maximizing hierarchical mutual information (auxiliary) for joint learning. The overall objective is defined as:
$$
L = L_r + \beta L_s,
$$
where $\beta$ controls the effect of the auxiliary task.

### Passage 9
The trainable parameters of our model consist of three parts: user and item embeddings, gate parameters, and attention parameters. In total, the model size approximates $(m+n+8d)d$.

### Passage 10
The computational cost mainly derives from four parts: hypergraph/graph convolution, attention, self-gating, and mutual information maximization. Since we remove the learnable matrix for linear transformation and the nonlinear activation function in graph propagation, the time complexity of our model is much lower than that of previous GNN-based social recommendation models.

## 4. 重点概念与术语
- BPR loss: 成对排序损失，核心目标是“正样本比分到负样本更高”。
- triplet sampling: 每次训练看一个三元组 $(u,i,j)$，其中 $i$ 是正物品，$j$ 是负物品。
- primary & auxiliary paradigm: 主任务负责推荐，辅助任务负责结构正则化，二者联合训练。
- hierarchical mutual information: 不是只做 user-graph 一层 MI，而是围绕 `user <- sub-hypergraph <- hypergraph` 这个层级来做。
- discriminator $f_D$: 用来打分两种表示是否“相容”的函数，本文里直接用 dot product。
- corrupted negatives: 通过打乱表示构造负样本，让模型学会区分真实结构配对和伪配对。
- $L_r$: recommendation loss，也就是 BPR 主任务。
- $L_s$: self-supervised auxiliary loss，也就是分层 MI 最大化目标。
- $\beta$: 平衡主任务与辅助任务强度的超参数。
- model size / time complexity: 本节不是证明最优，而是交代模型的参数来源和主要计算瓶颈。

## 5. 本章核心内容
### 5.1 用中文讲清楚
Chapter 4 要回答的第一个问题其实很朴素：MHCN 最后拿什么信号来学推荐？作者给出的答案是 BPR。也就是说，模型不是去回归一个用户会给物品打几分，而是对每个训练样本采一个三元组 $(u,i,j)$，要求正样本 $i$ 的分数高于负样本 $j$。因此这篇论文从训练目标上看，始终是一个 ranking-based recommendation 模型。前面所有 hypergraph、channel、attention 的设计，最后都是为了让 $\hat r_{u,i}=p_u^\top q_i$ 这个打分在排序上更靠谱。

然后作者指出，光有主任务还不够。原因不是“标签太少”这么简单，而是多通道聚合会丢高阶结构信息。Chapter 3 我们已经看到，模型会先在 social / joint / purchase 三个 channel 内各自传播，再融合成 comprehensive user representation。问题在于，不同 channel 上的表示分布不同，结构语义也不同，一旦聚合，就可能把这些差异冲淡。所以作者把自监督任务当成一个结构正则器，专门盯住“这些高阶结构信息不要在聚合后丢掉”。

这一节最核心的创新点，是它不是简单照搬 DGI 那种 `node <-> whole graph` 的 MI 最大化，而是把层级改成：
`user node <- user-centered sub-hypergraph <- hypergraph`。
这里每个用户在每个 channel 中，都对应一个以自己为中心的 sub-hypergraph。作者希望最终用户表示既能反映局部结构，也能反映全局结构，所以他们把 MI 最大化拆成两段来做：一段是用户表示与局部子超图表示之间的相容性，另一段是子超图表示与整个 hypergraph 表示之间的相容性。这样一来，局部和全局两层结构都被拉进训练信号里。

[??? 什么是MI最大化。如何理解“以自己为中心的 sub-hypergraph”的构造。]

从实现上看，$L_s$ 不需要外部标签。它靠结构本身构造正负配对：真实的 user / sub-hypergraph / hypergraph 组合是正样本，打乱后的 sub-hypergraph 表示是负样本。判别器 $f_D$ 也不复杂，论文里直接用 dot product。这一点很重要，因为它说明本文的“自监督”不是另起一个很重的对比学习头，而是一个相对轻量的结构一致性约束。

等到主任务 $L_r$ 和辅助任务 $L_s$ 都定义好以后，作者在 Eq. (12) 用
$$
L=L_r+\beta L_s
$$
把它们合起来。这里的 $\beta$ 就是调节杆：太小，自监督起不到约束作用；太大，又可能反过来干扰推荐主任务。你后面在实验部分会看到，他们确实专门做了 $\beta$ 的敏感性分析，所以这个超参数不是装饰性的。

最后一节复杂度分析其实也很有信息量。论文明确说模型的 trainable 参数主要来自三部分：初始 user/item embeddings、gate parameters、attention parameters。注意这里没有把 hypergraph propagation 本身算成一大堆 learnable matrix，因为作者前面已经把传播设计成了“少参数甚至无额外传播参数”的形式。计算量主要花在四块：hypergraph/graph convolution、attention、self-gating、mutual information maximization。也就是说，这个模型虽然结构看起来复杂，但作者的策略是：把复杂性更多放在“结构构造与传播路径”上，而不是放在大矩阵变换上。

如果把这一章浓缩成一句话：
`Chapter 4 讲的是：MHCN 用 BPR 学推荐排序，用 hierarchical MI 学结构不丢失，再用一个加权总目标把二者绑在一起。`

### 5.2 这段在全文中的作用
这一章是方法部分的“训练说明书”和“代价说明书”。

- Section 3.2.4 把 Chapter 3 里得到的最终 embeddings 连接到一个明确的推荐优化目标上。
- Section 3.3 解释为什么还要有自监督，以及这个自监督到底在 regularize 什么。
- Section 3.4 说明模型并不是靠堆很重的 learnable transformation 才变强，而是更多利用结构设计。

如果这一章没看顺，后面你会容易把 `S^2-MHCN` 理解成“多加了一个损失项”而已，但作者更想表达的是：辅助任务在补偿 aggregation 所带来的结构信息损失。

### 5.3 容易误解的点
- BPR 不是在拟合显式评分，而是在优化相对排序。
- 自监督任务不是额外的数据预训练阶段，而是与推荐任务联合训练的辅助目标。
- $L_s$ 的重点不是“让表示彼此接近”这么泛，而是让用户表示保留局部与全局高阶结构信息。
- `hierarchical MI` 不是只做一层 user-graph 配对，而是通过 sub-hypergraph 做中介，把局部和全局一起约束。
- $\beta$ 不是模型参数，而是人工设定并在实验里调优的超参数。
- 复杂度分析里没有大额传播权重矩阵，恰恰反映了作者想让传播尽量轻量。

## 6. 证据指针
- Section 3.2.4; Eq. (8): BPR 主任务、三元组采样与推荐打分形式。
- Section 3.3 opening paragraphs: 为什么 aggregation 会导致高阶信息损失，以及为什么需要 auxiliary self-supervised task。
- Section 3.3 DGI comparison paragraph: 为什么作者认为普通 graph-node MI 不够细。
- Section 3.3 hierarchy paragraph: `user <- sub-hypergraph <- hypergraph` 的层级定义。
- Figure 4; Eq. (9)-(11): sub-hypergraph / graph 读出、自监督判别与 $L_s$。
- Section 3.3 final paragraph; Eq. (12): 总目标 $L=L_r+\beta L_s$。
- Section 3.4: 参数来源、模型规模与主要时间复杂度项。

## 7. 一分钟回顾
- 主任务 $L_r$ 用 BPR 学的是排序，不是评分回归。
- 辅助任务 $L_s$ 用 hierarchical mutual information 学的是“用户表示别把高阶结构信息丢掉”。
- 整体训练目标是 $L=L_r+\beta L_s$，而复杂度主要来自传播、attention、gating 和 MI 计算，不是大矩阵参数。

## 8. 你的问答区
- 你可以直接问：为什么作者觉得 DGI 那种 node-graph MI 不够？
- 你也可以问：$L_s$ 里的正负样本到底怎么构造？
- 如果你想自测，先回答：为什么说 $L_s$ 更像结构 regularizer，而不是替代主任务标签的第二份监督？
- 如果你觉得这一章已经顺了，直接回复“进入下一章”即可。
- Q1：我没看懂原文的“Recall that, for each channel of MHCN, we build the adjacency matrix 𝑨𝑐 to capture the high-order connectivity information. Each row in 𝑨𝑐 represents a subgraph of the corresponding hypergraph centering around the user denoted by the row index. Then we can induce a hierarchy: ‘user node ← user-centered sub-hypergraph ← hypergraph’ and create self-supervision signals from this structure.”是什么意思
- Q2：Self-Supervised Learning部分的数学公式我需要你具体展开解读每个Term的含义以及每个Equation的含义。里面提到的Ac，DGI，Equation9、10、11我都没懂。

Round2：
- Q3：什么是真实配对，什么是伪造配对？
- Q4：如何理解相容性？英文原文是什么？伪局部结构是怎么来的？

## 9. Codex 补充讲解
- Round 0: Chapter 4 已初始化。
- 先给你一个最短训练版：
  `BPR 负责把正物品排在负物品前面；hierarchical MI 负责让聚合后的用户表示别把高阶结构忘掉；二者用 beta 加权联合训练。`
- 关于为什么主任务选 BPR：
  因为本文做的是 top-$K$ recommendation，目标天然更接近排序而不是评分回归。BPR 正好直接优化“正样本比分高于负样本”的相对顺序。
- 关于 $L_s$ 最容易误读的点：
  它不是为了让模型“再学一份推荐信号”，而是为了约束表示空间。更准确地说，它在逼 comprehensive user representation 保留 hypergraph 里的局部和全局结构信息。
- 关于 DGI 和本文的差别：
  DGI 更像 `node <-> whole graph` 的 coarse MI；本文认为这还不够细，所以引入 `user <- sub-hypergraph <- hypergraph` 这一层级，把中间那层 sub-hypergraph 拉出来做桥梁。
- 关于总目标怎么理解：
  如果你只看 vanilla `MHCN`，训练主要是 Eq. (8) 的 $L_r$。
  如果你看完整的 `S^2-MHCN`，真正训练的是 Eq. (12) 的联合目标 $L=L_r+\beta L_s$。
- 关于复杂度分析最该记住的一点：
  作者的卖点之一就是：模型的“聪明”主要来自结构设计，而不是给每层传播都塞很重的 learnable transformation matrix。
- 如果你现在只想记一句：
  `Chapter 4 讲的是：模型为什么会学，以及它学这些东西要付出多大计算代价。`
- Round 1: 回答你这次补进来的 `MI / A_c / sub-hypergraph / Eq. (9)(10)(11)` 问题。
- 先回答你正文里的问号：什么是 MI 最大化？
  这里的 `MI` 是 mutual information，直观上可以先理解成“两个表示之间共享了多少有效信息”。如果两个表示真的来自同一个结构对象，它们应该更相容；如果是乱配的，它们应该不相容。所谓“最大化 MI”，在本文里不用死抠信息论定义，先抓训练直觉就够了：让真实配对的相似度高于伪造配对的相似度。论文里就是用 dot product 打分，再让正配对比分高于负配对。
- 再回答“以自己为中心的 sub-hypergraph”怎么理解：
  对于某个 channel $c$，作者已经有了一个 motif-induced adjacency $A_c$。你可以把第 $u$ 行 $\mathbf a_u^c$ 理解成“用户 $u$ 在这个 channel 里和哪些用户通过高阶 motif 发生了连接，以及连接强度各是多少”。所以：
  1. 行号 $u$ 决定中心用户是谁。
  2. 这一行里非零的位置，告诉你哪些其他用户属于围绕 $u$ 的那片局部高阶结构。
  3. 这些非零值的大小，告诉你这些连接在当前 hypergraph 里有多强。
  这样得到的就不是整个全局 hypergraph，而是“围绕用户 $u$ 的那块局部结构”，作者把它叫 `user-centered sub-hypergraph`。
- 你可以把原文那句长话压成一句短话：
  `A_c` 给的是整张高阶关系图；`A_c` 的第 $u$ 行给的是“用户 u 的局部结构切片”；于是就自然得到 `user -> local sub-hypergraph -> global hypergraph` 这三层。
- Q1：你贴的这段原文到底在说什么？
  它在说：作者已经在每个 channel 里构造好了 $A_c$，所以不需要再重新定义“局部结构”了。对某个用户 $u$ 来说，$A_c$ 的第 $u$ 行就是他在这个 channel 里的高阶邻域摘要。于是作者可以同时拿到三种粒度的对象：
  1. 用户自己的表示 `user node`
  2. 由第 $u$ 行诱导出来的局部结构表示 `user-centered sub-hypergraph`
  3. 由所有用户局部结构汇总得到的整体表示 `hypergraph`
  然后就能在这三层之间制造 self-supervision。
- 这里顺手补一个你很容易卡住的点：
  论文说“Each row in $A_c$ represents a subgraph”时，不是说“一行矩阵本身就完整等于一个图对象”，而是说“这一行足以指定以该用户为中心的局部高阶连接模式”，后面 readout 就基于这行去构造对应的局部表示。
- Q2：先讲 DGI 是什么。
  `DGI` 是 Deep Graph Infomax。你可以把它先记成一种典型的 graph self-supervised baseline：它主要做的是 `node representation` 和 `whole-graph summary` 之间的 mutual information 最大化。本文觉得这种约束太粗，因为它直接把“节点”和“整图”配对，中间缺了一层局部结构，所以才加进 `sub-hypergraph` 这一层，做成 hierarchical MI。
- 下面逐式拆公式。
- Eq. (9) 的作用：
  它把“以用户 $u$ 为中心的 sub-hypergraph”读成一个向量表示 $\mathbf z_u^c$。
- Eq. (9) 里每个符号怎么读：
  1. $\mathbf P_c$：当前 channel 下参与 self-supervised readout 的用户表示矩阵。原文还说它是经过 gate 处理过的，目的是减轻主任务和辅助任务的梯度冲突。
  2. $\mathbf a_u^c$：$A_c$ 的第 $u$ 行，也就是用户 $u$ 在 channel $c$ 下的局部高阶连接权重。
  3. $\mathbf z_u^c$：最终得到的“用户 $u$ 的局部 sub-hypergraph 表示”。
  4. $f_{\text{out}1}$：第一个 readout 函数，本质上是在做一个加权汇总。
- Eq. (9) 这条式子的直觉：
  它不是简单把局部邻居平均一下，而是按 $\mathbf a_u^c$ 提供的连接强度，对局部用户表示做加权 pooling，得到围绕用户 $u$ 的局部结构表示。所以如果某些邻居和 $u$ 在更多 motif 实例里共现，它们对 $\mathbf z_u^c$ 的贡献也更大。
- Eq. (10) 的作用：
  它把所有局部 sub-hypergraph 表示再汇总成整个 channel 的全局 hypergraph 表示 $\mathbf h^c$。
- Eq. (10) 里每个符号怎么读：
  1. $\mathbf Z_c$：把所有用户的局部表示 $\mathbf z_u^c$ 收集起来形成的集合/矩阵。
  2. $\mathbf h^c$：channel $c$ 的全局 hypergraph 表示。
  3. $f_{\text{out}2}$：第二个 readout 函数，论文里明确说它就是 `AveragePooling`。
- Eq. (10) 的直觉：
  Eq. (9) 先把“每个人周围的小环境”编码出来；Eq. (10) 再把所有人的小环境平均成“整个 channel 的全局结构摘要”。
- Eq. (11) 的作用：
  它定义 self-supervised loss $L_s$，也就是 hierarchical MI 的训练目标。
- Eq. (11) 的整体结构不要一上来硬读符号，先读成两段：
  1. 第一段：让用户表示和“自己的真实局部 sub-hypergraph 表示”更匹配，而不是和打乱后的伪局部表示匹配。
  2. 第二段：让真实局部 sub-hypergraph 表示和“整个 hypergraph 表示”更匹配，而不是让伪局部表示和整图匹配。
- Eq. (11) 里关键 Term 的含义：
  1. $\mathbf p_u^c$：用户 $u$ 在 channel $c$ 下的表示。
  2. $\mathbf z_u^c$：真实的局部 sub-hypergraph 表示。
  3. $\tilde{\mathbf z}_u^c$：打乱后的伪局部表示，也就是负样本。
  4. $\mathbf h^c$：全局 hypergraph 表示。
  5. $f_D(\cdot,\cdot)$：判别器，本文里直接就是两个向量的 dot product。
  6. $\sigma(\cdot)$：sigmoid，用来把“正样本比分大于负样本分”这种差异转成可优化目标。
- Eq. (11) 第一半项在做什么：
  $\log \sigma\big(f_D(\mathbf p_u^c,\mathbf z_u^c)-f_D(\mathbf p_u^c,\tilde{\mathbf z}_u^c)\big)$
  它在逼模型满足：用户表示 $\mathbf p_u^c$ 和真实局部结构 $\mathbf z_u^c$ 的相容性，要高于它和伪局部结构 $\tilde{\mathbf z}_u^c$ 的相容性。这是“局部结构”那层的 MI 约束。
- Eq. (11) 第二半项在做什么：
  $\log \sigma\big(f_D(\mathbf z_u^c,\mathbf h^c)-f_D(\tilde{\mathbf z}_u^c,\mathbf h^c)\big)$
  它在逼模型满足：真实局部结构 $\mathbf z_u^c$ 应该比伪局部结构 $\tilde{\mathbf z}_u^c$ 更符合整张 hypergraph 的全局摘要 $\mathbf h^c$。这是“全局结构”那层的 MI 约束。
- 为什么 Eq. (11) 前面有负号：
  因为训练时要最小化 loss，所以作者把“想要最大化的相容性目标”写成负号包住的形式，最后通过最小化 $L_s$ 来等价实现“最大化正配对优于负配对”。
- 为什么论文说这里是 hierarchical MI：
  因为它不是只做一层配对，而是把监督拆成了两级：
  `user <-> local sub-hypergraph`
  `local sub-hypergraph <-> global hypergraph`
  中间那层 sub-hypergraph 就是层级里的桥梁。
- 你如果想把 Eq. (9)(10)(11) 串起来记，可以用这条链：
  `A_c 的第 u 行 -> 构造用户 u 的局部 sub-hypergraph 表示 z_u^c -> 所有 z_u^c 平均成全局表示 h^c -> 用真实/打乱配对去训练 user, local, global 三层的一致性。`
- 再给你一个最短白话版：
  作者先从 $A_c$ 里截出“用户 $u$ 周围的高阶结构”，把它读成局部向量；再把所有局部向量汇总成整图向量；然后训练模型让“用户像自己的局部结构，局部结构也像所属的整图”，而不像被打乱的伪结构。
- Round 2: 回答你这轮补进来的 `真实配对 / 伪造配对 / 相容性 / 伪局部结构` 问题。
- Q3：什么是真实配对，什么是伪造配对？
  在这一节里，“真实配对”就是本来应该来自同一个结构层级的表示组合；“伪造配对”就是作者故意打乱后拼出来的错误组合。
  具体到 Eq. (11) 有两组真实配对、两组伪造配对：
  1. 真实局部配对：$(\mathbf p_u^c,\mathbf z_u^c)$
  2. 伪局部配对：$(\mathbf p_u^c,\tilde{\mathbf z}_u^c)$
  3. 真实全局配对：$(\mathbf z_u^c,\mathbf h^c)$
  4. 伪全局配对：$(\tilde{\mathbf z}_u^c,\mathbf h^c)$
  训练目标就是让前两组里的“真实组合”打分高于对应的“伪组合”。
- 为什么说这些是“真实”：
  因为 $\mathbf z_u^c$ 真的是从用户 $u$ 在 channel $c$ 下的局部结构读出来的；$\mathbf h^c$ 也真的是由该 channel 全体局部结构汇总出来的。所以它们天然有对应关系。
- 为什么说这些是“伪造”：
  因为 $\tilde{\mathbf z}_u^c$ 不是从“用户 $u$ 的真实局部结构”读出来的，而是把原本的局部表示矩阵 $\mathbf Z_c$ 做了破坏性打乱之后得到的负样本表示。它的作用不是表示一个真实图对象，而是专门拿来当“错配参照物”。
- Q4：如何理解“相容性”？英文原文是什么？
  论文这里最直接的英文原话有两个：
  1. `scores the agreement between them`
  2. `the user should have a stronger connection with the sub-hypergraph centered with her`
  所以我前面翻成“相容性”，你也可以理解成：
  `agreement / compatibility / consistency / stronger connection`
  它们在这里表达的是同一个训练直觉：如果两个表示本来就该互相对应，它们的 dot product 应该更高；如果是错配的，dot product 应该更低。
- 再更白话一点：
  “相容性高”不是说两个向量长得像，而是说“这两个表示应该属于同一块结构上下文”。比如用户表示 $\mathbf p_u^c$ 和她自己的局部结构 $\mathbf z_u^c$ 应该更搭；而和乱打乱来的 $\tilde{\mathbf z}_u^c$ 就不该那么搭。
- 伪局部结构到底是怎么来的？
  原文写的是：`We corrupt Z_c by both row-wise and column-wise shuffling to create negative examples \tilde Z_c.`
  也就是说：
  1. 先把所有真实局部表示收集成 $\mathbf Z_c$
  2. 再对这个矩阵做 `row-wise` 和 `column-wise` shuffling
  3. 打乱后得到 $\tilde{\mathbf Z}_c$
  4. 从中取出对应用户的那一项，就得到 $\tilde{\mathbf z}_u^c$
- 这里要特别注意：
  $\tilde{\mathbf z}_u^c$ 不一定还能对应某个“真实存在的局部图”。它更像一个被破坏后的伪表示，而不是一张合法局部超图的严格编码。这正是它能当负样本的原因。
- 为什么 row-wise 和 column-wise 都要打乱？
  直观上这是在同时破坏：
  1. 不同用户之间“谁对应谁”的关系
  2. 向量内部特征维度的组织关系
  这样得到的负样本更难和真实局部结构混淆，也更不容易被模型用简单位置记忆投机过去。
- 你可以把 Eq. (11) 再读成一句最短判别规则：
  `真实的 user-local 配对应该比伪造的 user-local 配对更搭；真实的 local-global 配对也应该比伪造的 local-global 配对更搭。`
- 如果你想把“真实配对 / 伪造配对 / 相容性”一起记住：
  `真实配对` 是同一层级链条里本来就该连在一起的表示，`伪造配对` 是打乱后故意接错的表示，`相容性` 就是判别器给这两者“到底搭不搭”的分数。
