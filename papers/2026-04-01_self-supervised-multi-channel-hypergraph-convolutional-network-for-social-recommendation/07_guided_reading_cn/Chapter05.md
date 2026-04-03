# Chapter 05 - Equation Walkthrough & Term Table

## 1. 本章范围
- Status: reference supplementary chapter (prepared for lookup)
- Source: Eq. (1) - Eq. (12) across Section 3.2.2, Section 3.2.3, Section 3.2.4, and Section 3.3
- Why this chapter: 前四章已经把方法主线讲顺了，但真正读论文时，很多卡点都会集中在公式。这个插入章专门把全文已经定义过的 Equation 串成一条链，解释每条式子“在算什么”“为什么要定义它”“每个 Term 指什么”。
- How to use this chapter: 你可以把它当成正文和实验章之间的“公式索引层”。以后读到某个符号忘了什么意思，可以先查本章的 Term Table，再回到对应的 Equation 说明。

## 2. 阅读前你先抓住这三个问题
1. Eq. (1) - Eq. (7) 主要是在“造表示”，而 Eq. (8) - Eq. (12) 主要是在“定义训练目标”；你要先分清这两段公式的角色。
2. 不是每条公式都属于同一层级：有的在定义模型结构，有的在做工程上的等价改写，有的在定义辅助任务，有的在把主任务和辅助任务合并。
3. 同一个符号在不同段落里可能对应“全体用户 embedding 矩阵”“某个用户的向量”“某个 channel 的局部结构摘要”；读公式时一定要先判断对象层级，再判断计算含义。

## 3. 英文原文分段
### Passage 1: Eq. (1) 的上下文
As different patterns may show different importances to the final recommendation performance, directly feeding the full base user embeddings $\mathbf P^{(0)}$ to all the channels is unwise. To control the information flow from the base user embeddings $\mathbf P^{(0)}$ to each channel, we design a pre-filter with self-gating units (SGUs), which is defined as:
$$
\mathbf P_c^{(0)} = f_{\text{gate}}^c(\mathbf P^{(0)}) = \mathbf P^{(0)} \odot \sigma(\mathbf P^{(0)}\mathbf W_g^c + \mathbf b_g^c). \tag{1}
$$

### Passage 2: Eq. (2) 的上下文
Referring to the spectral hypergraph convolution proposed in [10], we define our hypergraph convolution as:
$$
\mathbf P_c^{(l+1)} = \mathbf D_c^{-1}\mathbf H_c\mathbf L_c^{-1}\mathbf H_c^\top \mathbf P_c^{(l)}. \tag{2}
$$
The hypergraph convolution can be viewed as a two-stage refinement performing `node-hyperedge-node` feature transformation upon hypergraph structure.

### Passage 3: Eq. (3) 和 Eq. (4) 的上下文
We use $\mathbf A^{M_k}$ to represent the motif-induced adjacency matrix and $(\mathbf A^{M_k})_{i,j}=1$ means that vertex $i$ and vertex $j$ appear in one instance of $M_k$. As two vertices can appear in multiple instances of $M_k$, $(\mathbf A^{M_k})_{i,j}$ is computed by:
$$
(\mathbf A^{M_k})_{i,j} = \#(i,j \text{ occur in the same instance of } M_k). \tag{3}
$$
Finally, we use $\mathbf A_s=\sum_{k=1}^7 \mathbf A^{M_k}$, $\mathbf A_j=\mathbf A^{M_8}+\mathbf A^{M_9}$, and $\mathbf A_p=\mathbf A^{M_{10}}-\mathbf A_j$ to replace $\mathbf H_s\mathbf H_s^\top$, $\mathbf H_j\mathbf H_j^\top$, and $\mathbf H_p\mathbf H_p^\top$ in Eq. (2), respectively. Then we have a transformed hypergraph convolution, defined as:
$$
\mathbf P_c^{(l+1)} = \hat{\mathbf D}_c^{-1}\mathbf A_c \mathbf P_c^{(l)}. \tag{4}
$$

### Passage 4: Eq. (5) - Eq. (7) 的上下文
After propagating the user embeddings through $L$ layers, we average the embeddings obtained at each layer to form the final channel-specific user representation. Then we use the attention mechanism to selectively aggregate information from different channel-specific user embeddings to form the comprehensive user embeddings. The attention function is defined as:
$$
\alpha_c = f_{\text{att}}(\mathbf p_c^*) =
\frac{\exp(\mathbf a^\top \mathbf W_{\text{att}}\mathbf p_c^*)}
{\sum_{c' \in \{s,j,p\}}\exp(\mathbf a^\top \mathbf W_{\text{att}}\mathbf p_{c'}^*)}. \tag{5}
$$
Besides, in our setting, the hypergraph convolution cannot directly aggregate information from the items. To tackle this problem, we additionally perform simple graph convolution on the user-item interaction graph to encode the purchase information and complement the multi-channel hypergraph convolution. The simple graph convolution is defined as:
$$
\mathbf P_r^{(l+1)}=\mathbf D_u^{-1}\mathbf R\mathbf Q^{(l)},\quad
\mathbf P_r^{(0)}=f_{\text{gate}}^r(\mathbf P^{(0)}),
$$
$$
\mathbf Q^{(l+1)}=\mathbf D_i^{-1}\mathbf R^\top \mathbf P_m^{(l)},\quad
\mathbf P_m^{(l)}=\sum_{c \in \{s,j,p\}}\alpha_c \mathbf p_c^{(l)} + \frac{1}{2}\mathbf P_r^{(l)}. \tag{6}
$$
Finally, we obtain the final user and item embeddings $\mathbf P$ and $\mathbf Q$ defined as:
$$
\mathbf P = \mathbf P^* + \frac{1}{L+1}\sum_{l=0}^L \mathbf P_r^{(l)},\quad
\mathbf Q = \frac{1}{L+1}\sum_{l=0}^L \mathbf Q^{(l)}. \tag{7}
$$

### Passage 5: Eq. (8) 的上下文
To learn the parameters of MHCN, we employ the Bayesian Personalized Ranking (BPR) loss, which is a pairwise loss that promotes an observed entry to be ranked higher than its unobserved counterparts:
$$
L_r = \sum_{i \in I(u), j \notin I(u)} -\log \sigma\big(\hat r_{u,i}(\Phi)-\hat r_{u,j}(\Phi)\big) + \lambda \lVert \Phi \rVert_2^2. \tag{8}
$$
where $\Phi$ denotes the parameters of MHCN, $\hat r_{u,i}=\mathbf p_u^\top \mathbf q_i$ is the predicted score of $u$ on $i$, and each time a triplet $(u,i,j)$ is fed to MHCN.

### Passage 6: Eq. (9) 的上下文
Recall that, for each channel of MHCN, we build the adjacency matrix $\mathbf A_c$ to capture the high-order connectivity information. Each row in $\mathbf A_c$ represents a subgraph of the corresponding hypergraph centering around the user denoted by the row index. Then we can induce a hierarchy: `user node <- user-centered sub-hypergraph <- hypergraph`.

To get the sub-hypergraph representation, instead of averaging the embeddings of the users in the sub-hypergraph, we design a readout function $f_{\text{out}1}$, which is permutation-invariant and formulated as:
$$
\mathbf z_u^c = f_{\text{out}1}(\mathbf P_c,\mathbf a_u^c) = \frac{\mathbf P_c \mathbf a_u^c}{\operatorname{sum}(\mathbf a_u^c)}. \tag{9}
$$

### Passage 7: Eq. (10) 的上下文
Analogously, we define the other readout function $f_{\text{out}2}$, which is actually an average pooling to summarize the obtained sub-hypergraph embeddings into a graph-level representation:
$$
\mathbf h^c = f_{\text{out}2}(\mathbf Z_c) = \operatorname{AveragePooling}(\mathbf Z_c). \tag{10}
$$

### Passage 8: Eq. (11) 的上下文
We tried to use InfoNCE as our learning objective to maximize the hierarchical mutual information. But we find that the pairwise ranking loss is more compatible with the recommendation task. We then define the objective function of the self-supervised task as follows:
$$
L_s = - \sum_{c \in \{s,j,p\}} \left\{
\sum_{u \in U}\log \sigma \big(f_D(\mathbf p_u^c,\mathbf z_u^c)-f_D(\mathbf p_u^c,\tilde{\mathbf z}_u^c)\big)
+ \sum_{u \in U}\log \sigma \big(f_D(\mathbf z_u^c,\mathbf h^c)-f_D(\tilde{\mathbf z}_u^c,\mathbf h^c)\big)
\right\}. \tag{11}
$$

### Passage 9: Eq. (11) 后的解释原文
$f_D(\cdot)$ is the discriminator function that takes two vectors as the input and then scores the agreement between them. We simply implement the discriminator as the dot product between two representations. We corrupt $\mathbf Z_c$ by both row-wise and column-wise shuffling to create negative examples $\tilde{\mathbf Z}_c$. We consider that, the user should have a stronger connection with the sub-hypergraph centered with her (local structure), so we directly maximize the mutual information between their representations. By contrast, the user would not care all the other users too much (global structure), so we indirectly maximize the mutual information between the representations of the user and the complete hypergraph by regarding the sub-hypergraph as the mediator.

### Passage 10: Eq. (12) 的上下文
Finally, we unify the objectives of the recommendation task (primary) and the task of maximizing hierarchical mutual information (auxiliary) for joint learning. The overall objective is defined as:
$$
L = L_r + \beta L_s, \tag{12}
$$
where $\beta$ is a hyper-parameter used to control the effect of the auxiliary task and $L_s$ can be seen as a regularizer leveraging hierarchical structural information of the hypergraphs.

## 4. 重点概念与术语
### 4.1 Equation 总览表
| Equation | 它在做什么 | 在全文里的角色 |
|---|---|---|
| Eq. (1) | 把 base user embedding 变成 channel-specific 输入 | 传播前的入口门控 |
| Eq. (2) | 定义标准超图卷积 | 理论上的 hypergraph propagation 主式 |
| Eq. (3) | 用 motif 共现次数定义邻接强度 | 从 motif 实例到邻接矩阵的桥 |
| Eq. (4) | 用 $\mathbf A_c$ 改写 Eq. (2) | 高效实现版传播 |
| Eq. (5) | 计算三个 channel 的注意力权重 | 跨 channel 聚合 |
| Eq. (6) | 在 user-item graph 上做补充分支传播 | 把 item 信息接回模型 |
| Eq. (7) | 产出最终 user / item embedding | 供打分与训练使用 |
| Eq. (8) | 定义 recommendation 主损失 $L_r$ | vanilla MHCN 的训练目标 |
| Eq. (9) | 读出用户对应的局部 sub-hypergraph 表示 | self-supervision 的 local 层 |
| Eq. (10) | 读出 channel 级 global hypergraph 表示 | self-supervision 的 global 层 |
| Eq. (11) | 定义 hierarchical MI 辅助损失 $L_s$ | $S^2$-MHCN 的关键增强项 |
| Eq. (12) | 合并主任务和辅助任务 | 最终联合训练目标 |

### 4.2 Term Table：符号含义对照查找表
| Symbol / Term | 首次关键出现 | 含义 | 你读公式时要怎么记 |
|---|---|---|---|
| $m,n,d,L$ | Eq. (1) - Eq. (7) | 用户数、物品数、embedding 维度、传播层数 | 它们是所有矩阵形状和平均层操作的背景参数 |
| $c \in \{s,j,p\}$ | Eq. (1) | channel 索引，分别是 social / joint / purchase | 几乎所有带上标 $c$ 的量都表示“这个量在某个 channel 内” |
| $\mathbf P^{(0)}$ | Eq. (1) | 初始用户 embedding 矩阵 | 三个 hypergraph channel 的共同起点 |
| $\mathbf Q^{(0)}$ | Eq. (7) 前 | 初始物品 embedding 矩阵 | user-item graph 分支的起点 |
| $\mathbf P_c^{(0)}$ | Eq. (1) | 经过 gate 后、送入 channel $c$ 的用户 embedding | 不是复制粘贴，而是按 channel 过滤后的输入 |
| $\mathbf W_g^c,\mathbf b_g^c$ | Eq. (1) | SGU 的可学习参数 | 负责决定每个 feature 维度该放大还是抑制 |
| $\sigma(\cdot)$ | Eq. (1), Eq. (8), Eq. (11) | sigmoid 函数 | 在 Eq. (1) 里做 gate，在 Eq. (8)(11) 里把差分分数转成可优化目标 |
| $\odot$ | Eq. (1) | element-wise product | 表示 feature-wise 的门控重加权 |
| $\mathbf H_c$ | Eq. (2) | channel $c$ 的超图 incidence matrix | 把“节点属于哪些 hyperedge”编码出来 |
| $\mathbf D_c,\mathbf L_c$ | Eq. (2) | 节点度矩阵、超边度矩阵 | 只做重标定，不引入新语义对象 |
| $\mathbf P_c^{(l)}$ | Eq. (2), Eq. (4) | 第 $l$ 层时 channel $c$ 的用户 embedding | 这是 channel 内传播的主状态量 |
| $\mathbf A^{M_k}$ | Eq. (3) | motif $M_k$ 诱导出的邻接矩阵 | 记录“两个用户一起出现在多少个 motif 实例里” |
| $\mathbf A_s,\mathbf A_j,\mathbf A_p$ | Eq. (4) 前 | 三个 channel 的 motif-induced adjacency | Eq. (4) 真正使用的邻接矩阵 |
| $\hat{\mathbf D}_c$ | Eq. (4) | $\mathbf A_c$ 的度矩阵 | 对 $\mathbf A_c$ 做归一化重标定 |
| $\mathbf P_c^*,\mathbf p_c^*$ | Eq. (5) | 某个 channel 在层平均后的最终用户表示 | Eq. (5) 用它来比较不同 channel 的贡献 |
| $\alpha_c$ | Eq. (5) | channel attention 权重 | 它不是固定超参，而是注意力网络动态算出来的 |
| $\mathbf a,\mathbf W_{\text{att}}$ | Eq. (5) | attention 模块参数 | 用来给不同 channel embedding 打分 |
| $\mathbf p^*$ | Eq. (5) 后 | 融合三个 channel 后的综合用户表示 | 是 hypergraph 主干汇总出的 user 表示 |
| $\mathbf R$ | Eq. (6) | user-item interaction matrix | user-item graph 分支的传播基础 |
| $\mathbf P_r^{(l)}$ | Eq. (6) | simple graph convolution 分支的用户表示 | 专门拿来补 item-aware 信息 |
| $\mathbf Q^{(l)}$ | Eq. (6), Eq. (7) | 第 $l$ 层物品表示 | 从 user-item 图分支里更新 |
| $\mathbf P_m^{(l)}$ | Eq. (6) | 第 $l$ 层用于更新物品的组合用户表示 | 由三 channel 表示和 graph 分支一起组成 |
| $\mathbf D_u,\mathbf D_i$ | Eq. (6) | $\mathbf R$ 与 $\mathbf R^\top$ 的度矩阵 | user-item 图上的归一化因子 |
| $\mathbf P,\mathbf Q$ | Eq. (7) | 最终 user / item embedding | 后续打分 $\hat r_{u,i}$ 直接使用它们 |
| $\hat r_{u,i}$ | Eq. (8) | 用户 $u$ 对物品 $i$ 的预测分数 | 论文里直接用内积 $\mathbf p_u^\top \mathbf q_i$ |
| $\Phi$ | Eq. (8) | 模型参数总集合 | 包括 embedding、gate、attention 等可学习参数 |
| $\lambda$ | Eq. (8) | $L_2$ 正则系数 | 不是模型参数，是训练时设定的超参数 |
| $\mathbf a_u^c$ | Eq. (9) | $\mathbf A_c$ 的第 $u$ 行 | 它对应“以用户 $u$ 为中心的局部高阶结构切片” |
| $\mathbf z_u^c$ | Eq. (9), Eq. (11) | 用户 $u$ 在 channel $c$ 中的局部 sub-hypergraph 表示 | self-supervision 的 local 对象 |
| $\mathbf Z_c$ | Eq. (10) | 把所有 $\mathbf z_u^c$ 收集起来形成的矩阵 | 用来汇总 global hypergraph 表示 |
| $\mathbf h^c$ | Eq. (10), Eq. (11) | channel $c$ 的全局 hypergraph 表示 | self-supervision 的 global 对象 |
| $\tilde{\mathbf Z}_c,\tilde{\mathbf z}_u^c$ | Eq. (11) | 打乱后的负样本局部表示 | 用来构造伪配对 |
| $f_{\text{out}1},f_{\text{out}2}$ | Eq. (9), Eq. (10) | 两个 readout 函数 | 前者读局部，后者读全局 |
| $f_D$ | Eq. (11) | discriminator | 论文里直接实现为 dot product |
| $L_r$ | Eq. (8) | recommendation loss | 主任务目标 |
| $L_s$ | Eq. (11) | self-supervised loss | 结构 regularizer / 辅助任务目标 |
| $\beta$ | Eq. (12) | 平衡主任务和辅助任务的超参数 | 调大它会让结构约束更强 |
| $L$ | Eq. (12) | 最终联合目标 | 整篇方法真正优化的是它 |

## 5. 本章核心内容
### 5.1 先把全文公式分成四段
- 第一段是 Eq. (1) - Eq. (4)：回答“每个 channel 怎么拿到输入，又怎么在 hypergraph 上传播”。
- 第二段是 Eq. (5) - Eq. (7)：回答“多个 channel 怎么聚合，以及 item 信息怎么补回来”。
- 第三段是 Eq. (8)：回答“vanilla MHCN 主任务到底怎么训练”。
- 第四段是 Eq. (9) - Eq. (12)：回答“$S^2$-MHCN 怎么做自监督，以及怎么和主任务合并”。

### 5.2 Eq. (1)：Self-Gating Unit
$$
\mathbf P_c^{(0)} = f_{\text{gate}}^c(\mathbf P^{(0)}) = \mathbf P^{(0)} \odot \sigma(\mathbf P^{(0)}\mathbf W_g^c + \mathbf b_g^c).
$$
- 上下文原文在说什么: 作者先说，不同高阶关系 pattern 对最终推荐的重要性不同，所以不能把同一份 base user embedding 毫无区分地喂给三个 channel。
- 这条式子的含义: 它先用一个可学习 gate 给每个 feature 维度打一个 0 到 1 之间的“通过强度”，再把原始用户 embedding 做逐维重加权，得到 channel-specific 初始表示。
- 这条式子的目的: 在真正做超图传播之前，先把“哪个 channel 更应该看哪些特征”这件事显式学出来。
- Term 拆解: $\mathbf P^{(0)}$ 是全体用户的 base embedding；$\mathbf W_g^c,\mathbf b_g^c$ 是 channel $c$ 的 gate 参数；$\sigma$ 把 gate 压到 $(0,1)$；$\odot$ 表示逐维乘法；$\mathbf P_c^{(0)}$ 是送进 channel $c$ 的起始用户表示。
- 和后文的衔接: Eq. (1) 的输出正是 Eq. (2) / Eq. (4) 的输入，所以它决定了每个 channel 一开始“带着什么信息上路”。

### 5.3 Eq. (2)：标准 Hypergraph Convolution
$$
\mathbf P_c^{(l+1)} = \mathbf D_c^{-1}\mathbf H_c\mathbf L_c^{-1}\mathbf H_c^\top \mathbf P_c^{(l)}.
$$
- 上下文原文在说什么: 作者借用了 spectral hypergraph convolution 的写法，先定义一个理论上干净的超图传播主式。
- 这条式子的含义: 这是一个 `node -> hyperedge -> node` 的两步传播。先把节点信息汇到超边，再把超边信息返还给节点，得到下一层用户表示。
- 这条式子的目的: 显式在高阶关系结构上做信息传播，而不是只在普通 pairwise graph edge 上传播。
- Term 拆解: $\mathbf H_c$ 是 incidence matrix，描述“用户属于哪些 hyperedge”；$\mathbf H_c^\top \mathbf P_c^{(l)}$ 对应节点到超边的汇聚；再左乘 $\mathbf H_c$ 对应超边回到节点；$\mathbf D_c$ 和 $\mathbf L_c$ 只是做归一化重标定；$\mathbf P_c^{(l+1)}$ 是下一层 channel 表示。
- 和后文的衔接: Eq. (2) 是“概念正确但实现偏重”的版本，后面 Eq. (3)(4) 会把它改写成更高效的邻接传播形式。

### 5.4 Eq. (3)：Motif-Induced Adjacency 的定义
$$
(\mathbf A^{M_k})_{i,j} = \#(i,j \text{ occur in the same instance of } M_k).
$$
- 上下文原文在说什么: 因为论文只考虑三角 motif，所以作者不想真的把所有 hyperedge 单独建出来，而是改为统计用户对在 motif 实例里的共现次数。
- 这条式子的含义: 对某个 motif 类型 $M_k$，如果用户 $i$ 和用户 $j$ 一起出现在很多个该 motif 的实例里，那么 $(\mathbf A^{M_k})_{i,j}$ 就会更大。
- 这条式子的目的: 用“共现次数”把显式的超边集合压缩成一个可直接传播的邻接矩阵。
- Term 拆解: $\mathbf A^{M_k}$ 是 motif $M_k$ 对应的邻接矩阵；$i,j$ 是用户索引；$M_k$ 是第 $k$ 类 motif；$\#(\cdot)$ 是计数操作。
- 和后文的衔接: 有了每类 motif 的 $\mathbf A^{M_k}$ 之后，才能把它们合并成 social / joint / purchase 三个 channel 的 $\mathbf A_c$，也就能进入 Eq. (4)。

### 5.5 Eq. (4)：高效改写后的传播式
$$
\mathbf P_c^{(l+1)} = \hat{\mathbf D}_c^{-1}\mathbf A_c \mathbf P_c^{(l)}.
$$
- 上下文原文在说什么: 作者把 $\mathbf H_s\mathbf H_s^\top$、$\mathbf H_j\mathbf H_j^\top$、$\mathbf H_p\mathbf H_p^\top$ 用三个 $\mathbf A_c$ 代替，从而绕开昂贵的超边构造。
- 这条式子的含义: 这已经更像标准归一化邻接传播了，只不过这里的邻接不是普通 social edge，而是 motif-induced high-order adjacency。
- 这条式子的目的: 在不丢掉高阶 motif 语义的前提下，把 Eq. (2) 变成一个计算成本更低的等价传播。
- Term 拆解: $\mathbf A_c$ 是第 $c$ 个 channel 的 motif-induced adjacency；$\hat{\mathbf D}_c$ 是它的度矩阵；$\mathbf P_c^{(l)}$ 和 $\mathbf P_c^{(l+1)}$ 是传播前后用户表示。
- 和后文的衔接: 实际实现时你更应该把 Eq. (4) 记成“真正被执行的传播式”，而不是只记 Eq. (2)。

### 5.6 Eq. (5)：Channel Attention
$$
\alpha_c = f_{\text{att}}(\mathbf p_c^*) =
\frac{\exp(\mathbf a^\top \mathbf W_{\text{att}}\mathbf p_c^*)}
{\sum_{c' \in \{s,j,p\}}\exp(\mathbf a^\top \mathbf W_{\text{att}}\mathbf p_{c'}^*)}.
$$
- 上下文原文在说什么: 每个 channel 传播完以后，作者不会简单平均，而是为每个用户学一组 channel 权重。
- 这条式子的含义: 对用户 $u$ 来说，三个 channel-specific 表示会先各自打分，再经过 softmax 归一化成权重 $\alpha_s,\alpha_j,\alpha_p$。
- 这条式子的目的: 让 social / joint / purchase 三类高阶关系对不同用户有不同重要性，而不是被统一等权对待。
- Term 拆解: $\mathbf p_c^*$ 是某个用户在 channel $c$ 内经过层平均后的表示；$\mathbf a,\mathbf W_{\text{att}}$ 是 attention 参数；$\alpha_c$ 是这个 channel 的归一化贡献权重。
- 容易误解的点: $\alpha_c$ 不是单独存着的一组自由参数，而是由当前用户的 channel 表示动态算出来的。
- 和后文的衔接: Eq. (5) 算出的 $\alpha_c$ 会继续进入 Eq. (6)，参与 user-item 图分支里的组合表示。

### 5.7 Eq. (6)：User-Item Graph 分支与组合表示
$$
\mathbf P_r^{(l+1)}=\mathbf D_u^{-1}\mathbf R\mathbf Q^{(l)},\quad
\mathbf P_r^{(0)}=f_{\text{gate}}^r(\mathbf P^{(0)}),
$$
$$
\mathbf Q^{(l+1)}=\mathbf D_i^{-1}\mathbf R^\top \mathbf P_m^{(l)},\quad
\mathbf P_m^{(l)}=\sum_{c \in \{s,j,p\}}\alpha_c \mathbf p_c^{(l)} + \frac{1}{2}\mathbf P_r^{(l)}.
$$
- 上下文原文在说什么: 作者专门提醒，hypergraph channel 主要在用户关系结构上传播，不能直接把 item 信息编码进来，所以要额外补一条 user-item graph convolution 分支。
- 这条式子的含义: 一方面，用户通过交互矩阵 $\mathbf R$ 从物品侧吸收信息；另一方面，物品也通过 $\mathbf R^\top$ 从当前组合后的用户表示吸收信息。
- 这条式子的目的: 把“高阶用户关系”与“真实购买交互”这两种信息源接起来，避免模型只懂 social high-order pattern 却不够 item-aware。
- Term 拆解: $\mathbf R$ 是 user-item 交互矩阵；$\mathbf P_r^{(l)}$ 是简单图分支里的用户表示；$\mathbf Q^{(l)}$ 是物品表示；$\mathbf D_u,\mathbf D_i$ 是归一化用的度矩阵；$\mathbf P_m^{(l)}$ 是用于更新物品的组合用户表示。
- 容易误解的点: 论文在 Eq. (6) 附近会在“全体用户矩阵”和“某个用户向量”的记号之间切换，所以你不要被行列转置卡住，核心意思就是“channel 融合后的用户表示再加上一条 graph 分支表示，一起去更新 item”。
- 和后文的衔接: Eq. (6) 让 item embedding 真正进入模型主线，后面的 Eq. (7) 会把它们汇总成最终表示。

### 5.8 Eq. (7)：最终 User / Item Embedding
$$
\mathbf P = \mathbf P^* + \frac{1}{L+1}\sum_{l=0}^L \mathbf P_r^{(l)},\quad
\mathbf Q = \frac{1}{L+1}\sum_{l=0}^L \mathbf Q^{(l)}.
$$
- 上下文原文在说什么: 到这里方法部分终于从“中间层表示”收束成“最终可用于推荐打分的表示”。
- 这条式子的含义: 最终用户表示由两部分组成，一部分是 multi-channel hypergraph 主干聚合得到的 $\mathbf P^*$，另一部分是 simple graph 分支 across layers 的平均结果；最终物品表示则来自 user-item graph 分支的层平均。
- 这条式子的目的: 产出统一的 user / item embedding，供后面的打分与训练损失使用。
- Term 拆解: $\mathbf P^*$ 是经过 channel attention 融合的综合用户表示；$\mathbf P_r^{(l)}$ 是 graph 分支第 $l$ 层用户表示；$\mathbf Q^{(l)}$ 是第 $l$ 层物品表示；$L$ 是传播层数。
- 和后文的衔接: Eq. (7) 的输出直接进入 Eq. (8) 的打分函数 $\hat r_{u,i}=\mathbf p_u^\top \mathbf q_i$。

### 5.9 Eq. (8)：Recommendation 主损失
$$
L_r = \sum_{i \in I(u), j \notin I(u)} -\log \sigma\big(\hat r_{u,i}(\Phi)-\hat r_{u,j}(\Phi)\big) + \lambda \lVert \Phi \rVert_2^2.
$$
- 上下文原文在说什么: 作者在 Section 3.2.4 明确说，vanilla MHCN 的主训练目标用的是 BPR。
- 这条式子的含义: 它要求用户真实交互过的正物品 $i$，在模型打分上高于未交互或负物品 $j$。
- 这条式子的目的: 直接优化 top-$K$ recommendation 更关心的相对排序，而不是拟合一个显式评分回归值。
- Term 拆解: $I(u)$ 是用户 $u$ 的正样本物品集合；$j \notin I(u)$ 是负样本；$\hat r_{u,i}=\mathbf p_u^\top \mathbf q_i$ 是预测分数；$\Phi$ 是模型参数；$\lambda \lVert \Phi \rVert_2^2$ 是正则项。
- 容易误解的点: 这里的求和写法看起来像“把所有正负对全都枚举一遍”，但训练时实际是按采样 triplet $(u,i,j)$ 来喂模型。
- 和后文的衔接: 如果你只看 vanilla `MHCN`，Eq. (8) 就已经是主目标；如果你看完整的 `S^2-MHCN`，它还会被 Eq. (12) 和 $L_s$ 绑定起来。

### 5.10 Eq. (9)：Local Sub-Hypergraph Readout
$$
\mathbf z_u^c = f_{\text{out}1}(\mathbf P_c,\mathbf a_u^c) = \frac{\mathbf P_c \mathbf a_u^c}{\operatorname{sum}(\mathbf a_u^c)}.
$$
- 上下文原文在说什么: 作者先提出层级 `user node <- user-centered sub-hypergraph <- hypergraph`，然后需要真的把中间这层 `sub-hypergraph` 读成一个向量。
- 这条式子的含义: 对用户 $u$ 而言，$\mathbf A_c$ 的第 $u$ 行告诉我们她在 channel $c$ 里的局部高阶邻域；Eq. (9) 用这行权重对相关用户 embedding 做一次 weighted readout，得到局部结构表示 $\mathbf z_u^c$。
- 这条式子的目的: 给 self-supervision 准备一个“局部结构对象”，否则只能直接做粗粒度的 `node <-> graph` 配对。
- Term 拆解: $\mathbf P_c$ 是 auxiliary task 里参与 readout 的用户表示；$\mathbf a_u^c$ 是 $\mathbf A_c$ 的第 $u$ 行；$\operatorname{sum}(\mathbf a_u^c)$ 表示这片局部结构的总连接权重；$\mathbf z_u^c$ 是读出的局部 sub-hypergraph 表示。
- 容易误解的点: 如果你被矩阵维度卡住，可以先忽略转置约定，抓住本质就行: Eq. (9) 在做的就是“按 $\mathbf a_u^c$ 加权的 pooling”，不是另起一个新网络。
- 和后文的衔接: Eq. (9) 产生的所有 $\mathbf z_u^c$ 会组成 $\mathbf Z_c$，再进入 Eq. (10) 和 Eq. (11)。

### 5.11 Eq. (10)：Global Hypergraph Readout
$$
\mathbf h^c = f_{\text{out}2}(\mathbf Z_c) = \operatorname{AveragePooling}(\mathbf Z_c).
$$
- 上下文原文在说什么: 既然已经为每个用户得到了局部 sub-hypergraph 表示，就可以进一步把整张 channel hypergraph 读成一个全局摘要。
- 这条式子的含义: 它把该 channel 下所有用户对应的局部结构表示平均起来，得到一个 graph-level summary。
- 这条式子的目的: 为 self-supervision 提供 global 层对象，让“局部结构”和“整张 hypergraph”的关系也能被训练到。
- Term 拆解: $\mathbf Z_c$ 是所有 $\mathbf z_u^c$ 组成的矩阵；$\mathbf h^c$ 是 channel $c$ 的全局 hypergraph 表示；$\operatorname{AveragePooling}$ 表示直接做平均池化。
- 和后文的衔接: Eq. (10) 产出的 $\mathbf h^c$ 会进入 Eq. (11) 的第二组配对，即 `local <-> global` 那一半。

### 5.12 Eq. (11)：Hierarchical Mutual Information Self-Supervision
$$
L_s = - \sum_{c \in \{s,j,p\}} \left\{
\sum_{u \in U}\log \sigma \big(f_D(\mathbf p_u^c,\mathbf z_u^c)-f_D(\mathbf p_u^c,\tilde{\mathbf z}_u^c)\big)
+ \sum_{u \in U}\log \sigma \big(f_D(\mathbf z_u^c,\mathbf h^c)-f_D(\tilde{\mathbf z}_u^c,\mathbf h^c)\big)
\right\}.
$$
- 上下文原文在说什么: 作者说他们试过 InfoNCE，但发现 pairwise ranking 风格的目标更兼容推荐任务，所以最终把 hierarchical MI 写成这条式子。
- 这条式子的含义: 它由两半组成。第一半要求用户表示 $\mathbf p_u^c$ 跟真实局部结构 $\mathbf z_u^c$ 的匹配，强于它跟伪局部结构 $\tilde{\mathbf z}_u^c$ 的匹配；第二半要求真实局部结构 $\mathbf z_u^c$ 跟全局结构 $\mathbf h^c$ 的匹配，强于伪局部结构 $\tilde{\mathbf z}_u^c$ 跟全局结构 $\mathbf h^c$ 的匹配。
- 这条式子的目的: 分层保住 local 和 global 的高阶结构信息，不让它们在跨 channel 聚合后被冲淡。
- Term 拆解: $\mathbf p_u^c$ 是用户 $u$ 在 channel $c$ 中的表示；$\mathbf z_u^c$ 是真实局部结构表示；$\tilde{\mathbf z}_u^c$ 是通过打乱 $\mathbf Z_c$ 得到的负样本局部表示；$\mathbf h^c$ 是全局 hypergraph 表示；$f_D$ 是 discriminator，论文里直接用 dot product；$\sigma$ 是 sigmoid。
- 真实配对和伪造配对: 真实 local 配对是 $(\mathbf p_u^c,\mathbf z_u^c)$，伪 local 配对是 $(\mathbf p_u^c,\tilde{\mathbf z}_u^c)$；真实 global 配对是 $(\mathbf z_u^c,\mathbf h^c)$，伪 global 配对是 $(\tilde{\mathbf z}_u^c,\mathbf h^c)$。
- 为什么叫 hierarchical: 因为它不是只做一层 `node <-> graph`，而是拆成了 `user <-> local sub-hypergraph` 和 `local sub-hypergraph <-> global hypergraph` 两级。
- 和后文的衔接: Eq. (11) 定义的就是辅助损失 $L_s$，最后会被 Eq. (12) 加到总目标里。

### 5.13 Eq. (12)：最终联合目标
$$
L = L_r + \beta L_s.
$$
- 上下文原文在说什么: 方法定义部分最后一锤定音，说明完整的 $S^2$-MHCN 到底在优化什么。
- 这条式子的含义: 推荐主任务 $L_r$ 和结构辅助任务 $L_s$ 一起训练，$\beta$ 决定辅助项有多强。
- 这条式子的目的: 让模型既学会“把正确物品排前面”，又学会“别把高阶结构信息在聚合时丢掉”。
- Term 拆解: $L_r$ 是 BPR recommendation loss；$L_s$ 是 hierarchical MI self-supervised loss；$\beta$ 是平衡系数；$L$ 是最终最小化的目标。
- 为什么 $L_s$ 更像 regularizer: 因为它不提供新的推荐标签，而是约束用户表示保留 hypergraph 结构信息，所以它更像结构正则项。

### 5.14 把 12 个公式串成一条白话链
你可以把整套方法压成下面这条链来记：

$$
\mathbf P^{(0)}
\xrightarrow{\text{Eq. (1)}} \mathbf P_c^{(0)}
\xrightarrow{\text{Eq. (2)(4)}} \mathbf P_c^{(l)}
\xrightarrow{\text{Eq. (5)}} \mathbf P^*
\xrightarrow{\text{Eq. (6)(7)}} \mathbf P,\mathbf Q
\xrightarrow{\text{Eq. (8)}} L_r
$$

同时，另一条辅助链是：

$$
\mathbf P,\mathbf A_c
\xrightarrow{\text{Eq. (9)}} \mathbf z_u^c
\xrightarrow{\text{Eq. (10)}} \mathbf h^c
\xrightarrow{\text{Eq. (11)}} L_s
\xrightarrow{\text{Eq. (12)}} L
$$

如果你只记一句白话，那就是：
`Eq. (1)-(7) 负责把 user / item 表示造出来，Eq. (8) 负责让推荐排序变好，Eq. (9)-(11) 负责让这些表示别忘掉高阶结构，Eq. (12) 负责把两件事绑在一起训练。`

## 6. 证据指针
- Section 3.2.2; Eq. (1): SGU 的定义与动机。
- Section 3.2.2; Eq. (2): 标准 hypergraph convolution 的写法。
- Table 1; Eq. (3): motif-induced adjacency 的计数定义与矩阵构造。
- Section 3.2.2 ending; Eq. (4): 用 $\mathbf A_c$ 替换 $\mathbf H_c\mathbf H_c^\top$ 的高效改写。
- Section 3.2.3; Eq. (5): channel attention 权重。
- Section 3.2.3; Eq. (6): simple graph convolution 与组合表示。
- Section 3.2.3; Eq. (7): 最终 user / item embedding。
- Section 3.2.4; Eq. (8): BPR recommendation loss。
- Section 3.3 hierarchy paragraph; Eq. (9): user-centered sub-hypergraph readout。
- Section 3.3 readout paragraph; Eq. (10): graph-level average pooling。
- Figure 4; Section 3.3; Eq. (11): hierarchical MI 与正负配对。
- Section 3.3 final paragraph; Eq. (12): 联合目标。

## 7. 一分钟回顾
- Eq. (1) 先让三个 channel 不再共用完全相同的输入，而是各自 gated。
- Eq. (2)(3)(4) 把“motif 实例”变成“可传播的高阶邻接”，再完成 channel 内传播。
- Eq. (5)(6)(7) 先跨 channel 聚合，再补一条 user-item graph 分支，最后得到 user / item 表示。
- Eq. (8) 是主推荐损失，Eq. (9)(10)(11) 是辅助结构损失，Eq. (12) 把它们联合起来。
- 读不懂单条公式时，先查 Term Table，再问自己“这条式子是在造表示，还是在定义训练目标”。

## 8. 你的问答区
- 你可以继续问：Eq. (2) 和 Eq. (4) 为什么能看成等价，直觉上到底省掉了什么？
- 你也可以继续问：Eq. (6) 里为什么是“加上 $\frac{1}{2}\mathbf P_r^{(l)}$”，这半权重该怎么理解？
- 如果你想专项练公式，我可以继续把 Eq. (1) - Eq. (12) 改写成一版“中文伪代码流程图”。
- 如果你想自测，先回答：哪几条公式在“造 user/item 表示”，哪几条公式在“定义优化目标”？

## 9. Codex 补充讲解
- Round 0: 这是按你的要求插在原本 Chapter04 和实验章之间的一章，目的不是推进阅读进度，而是给你留一份可反复回查的公式总索引。
- 读公式的推荐顺序不是从 Eq. (1) 一路背到 Eq. (12)，而是先读 `Eq. (1) -> Eq. (4) -> Eq. (5) -> Eq. (7)`，先看清“表示怎么造出来”；再读 `Eq. (8)`；最后再读 `Eq. (9) -> Eq. (10) -> Eq. (11) -> Eq. (12)`。
- 最容易卡住的两个点是：
  1. 论文会在“矩阵级符号”和“单个用户向量级符号”之间来回切换，所以不要被维度表面写法带偏，先抓语义对象。
  2. `A_c` 的一行不是一张真正独立存储的小图文件，而是“足以定义用户 $u$ 局部高阶结构切片”的权重视图。
- 如果你后面继续往这章里补问题，我会继续把增量解释追加在这里，而不是打散到别的章节里。
