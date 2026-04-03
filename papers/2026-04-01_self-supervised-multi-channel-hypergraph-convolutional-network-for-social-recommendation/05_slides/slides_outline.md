# Slide Outline

## Slide 1
### Title
Problem, Claim, and Research Question
### Bullets
- Paper: `Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation`, WWW 2021.
- Target task: improve social recommendation under sparse user-item interactions, especially for cold-start users.
- Core limitation claim: existing social recommenders mainly model `pairwise` relations, so they miss high-order patterns such as `friends + shared purchase behavior`.
- Central question: how can we explicitly encode these high-order social relations and turn them into better top-$K$ ranking performance?
### Suggested Figure
- Figure 1, plus one small contrast line: `pairwise social graph -> insufficient`, `motif-induced hypergraph -> richer signal`.
### Speaker Notes
- 开场把 `motivation + limitation claim + research question` 一次讲清楚，不要先陷进公式。重点是：作者不是想做一般性的 user-item 增强，而是想补 social recommendation 里被忽略的高阶关系。

## Slide 2
### Title
Method Roadmap
### Bullets
- Start from interaction matrix $\mathbf R$ and social matrix $\mathbf S$, then detect 10 local `triangular motifs` from the heterogeneous user-item-social view.
- Group motif instances into three semantic channels: `social`, `joint`, and `purchase`, each inducing one motif-based user relation structure.
- Learn user representations through channel-specific hypergraph propagation, while a parallel user-item graph branch keeps item-side collaborative signal.
- Fuse channel outputs with attention, then train the model with `BPR + hierarchical self-supervised regularization`.
### Suggested Figure
- Figure 3 as the main overview figure.
### Speaker Notes
- 这一页给组员一个整图脑图：`motif extraction -> 3 channels -> propagation/fusion -> prediction -> self-supervised regularization`。后面几页再逐块拆开。

## Slide 3
### Title
Motif Construction and Channel Semantics
### Bullets
- The 10 motifs $M_1$-$M_{10}$ are `local triangle instances`, not a partition of the whole graph.
- The motifs are grouped into three channels: `social` ($M_1$-$M_7$), `joint` ($M_8$-$M_9$), and `purchase` ($M_{10}$).
- Each channel induces one user hypergraph, or equivalently one motif-induced adjacency $\mathbf A_c$, that captures one type of high-order user relation.
- This design separates different semantics instead of collapsing all high-order evidence into one noisy graph.
### Suggested Figure
- Figure 2 for motif examples and channel grouping.
### Speaker Notes
- 这里一定要讲准一件事：不是“整张图被切成 motif 类型”，而是“图里的局部三角实例被识别并归类”。这样后面讲 channel 才不会误解成图划分问题。

## Slide 4
### Title
Representation Learning in Vanilla MHCN
### Bullets
- `SGU` first gates the shared user embedding into channel-specific inputs: $\mathbf P_c^{(0)} = \mathbf P^{(0)} \odot \sigma(\mathbf P^{(0)}\mathbf W_g^c + \mathbf b_g^c)$.
- Inside each channel, Eq. (2) is the standard hypergraph convolution and Eq. (4) is its efficient rewrite: $\mathbf P_c^{(l+1)} = \hat{\mathbf D}_c^{-1}\mathbf A_c\mathbf P_c^{(l)}$.
- Layer-wise averaging yields $\mathbf P_s^*, \mathbf P_j^*, \mathbf P_p^*$, and attention learns user-specific weights $(\alpha_s,\alpha_j,\alpha_p)$ to fuse the three channels.
- A parallel LightGCN-style simple graph branch on the user-item graph supplies item-side signal; the final user/item embeddings $\mathbf P,\mathbf Q$ come from branch combination.
- Recommendation uses $\hat r_{u,i} = \mathbf p_u^\top \mathbf q_i$ and optimizes a BPR ranking objective instead of a rating-regression objective.
### Suggested Figure
- Figure 3, with a compact formula box for Eq. (1), Eq. (4), Eq. (6), Eq. (7), and Eq. (8).
### Speaker Notes
- 这一页把你在 Chapter 03 里梳理出的主干完整落地：`SGU -> channel propagation -> layer average -> attention fusion -> simple graph branch -> P/Q -> BPR`。强调传播算子本身基本不是靠大变换矩阵取胜。

## Slide 5
### Title
Why Self-Supervised Learning Is Added
### Bullets
- Motivation: after multi-channel aggregation, some high-order structural information may be diluted, which the paper calls `aggregating loss`.
- The auxiliary task builds a hierarchy `user <- user-centered sub-hypergraph <- hypergraph` and maximizes mutual information across this hierarchy.
- Positive pairs come from the real hierarchy, while negative pairs are built from shuffled sub-hypergraph representations; Eq. (9)-Eq. (11) define the discriminator and objective.
- The final training objective is $L = L_r + \beta L_s$, where $L_s$ acts as a structural regularizer rather than a second recommendation label.
### Suggested Figure
- Figure 4, plus a compact equation box for Eq. (9), Eq. (11), and Eq. (12).
### Speaker Notes
- 这页专门回答“为什么还要加一层自监督”。核心不是再造标签，而是用 hierarchical mutual information 去约束聚合后的表示别把高阶结构信息丢掉。

## Slide 6
### Title
Experimental Setup and Main Results
### Bullets
- Datasets: `LastFM`, `Douban`, and `Yelp`; Douban is converted from explicit ratings to implicit positive feedback for top-$K$ recommendation.
- Evaluation uses `5-fold cross-validation` and ranks over `all candidate items`, with both `complete test` and `cold-start test` settings.
- Strong baselines include `BPR`, `SBPR`, `GraphRec`, `DiffNet++`, `LightGCN`, and `DHCF`; the paper compares both `MHCN` and `S^2-MHCN`.
- Table 3: `S^2-MHCN` is best on all three datasets; for example on Yelp, `NDCG@10 = 0.06061`, higher than `LightGCN = 0.04998` and `MHCN = 0.05356`.
- Table 4 is the stronger evidence for this paper: on cold-start users, gains are larger, e.g. Douban `Recall@10` rises from `9.646%` to `10.632%`, and Yelp `NDCG@10` rises from `0.04354` to `0.05143`.
### Suggested Figure
- Table 3 and Table 4, with one visual callout highlighting `MHCN > baseline` and `S^2-MHCN > MHCN`.
### Speaker Notes
- 这一页不要只报“第一名”。要把两个层次讲清楚：`MHCN > baseline` 证明结构设计有效，`S^2-MHCN > MHCN` 证明自监督不是装饰项。冷启动表比完整测试表更贴这篇 paper 的任务价值。

## Slide 7
### Title
Ablation, Sensitivity, and Mechanism Validation
### Bullets
- Figure 5 shows that removing any channel hurts performance, but `purchase channel` contributes the most; without it, the model drops close to `LightGCN` level.
- Figure 6 shows attention weights consistent with ablation: `purchase` is usually the strongest channel, `joint` is in the middle, and `social` is often the weakest.
- Figure 7 validates the SSL design: `hierarchical MIM` performs best, better than `local-only`, `global-only`, or `DGI-style node-graph` mutual information.
- Figure 8 and Figure 9 reveal the model boundary: best performance appears at small $\beta$ (peak around `0.01`) and shallow depth (`L=2`), while deeper models suffer from over-smoothing.
### Suggested Figure
- Figure 7 as the main figure, plus a compact takeaway strip from Figure 5, Figure 8, and Figure 9.
### Speaker Notes
- 这页的作用是“证明为什么它有效，以及边界在哪里”。尤其要讲清 Figure 7 是在回头验收 Chapter 4 的 self-supervised 设计，而 Figure 8/9 则暴露了方法不能乱加权、不能乱堆深度。

## Slide 8
### Title
Contribution, Limitation, and Takeaway
### Bullets
- Contribution 1: the paper reframes social recommendation from `pairwise social graph` modeling to `multi-channel motif-induced hypergraph` modeling.
- Contribution 2: it gives a practical representation-learning pipeline that combines SGU, efficient hypergraph propagation, channel attention, and a simple graph branch.
- Contribution 3: it introduces `hierarchical mutual information maximization` that better matches `user <- sub-hypergraph <- hypergraph` than generic DGI-style SSL.
- Limitation: the strongest gains still rely heavily on `purchase-related` signal, and performance is sensitive to self-supervised weight and model depth.
- Main talk takeaway: this paper is best introduced as `high-order relation construction + controlled structural regularization`, not just “another GNN recommender with better numbers”.
### Suggested Figure
- One final synthesized takeaway table, or a compact summary panel built from Figure 5, Figure 7, and Table 4.
### Speaker Notes
- 收尾时给出平衡判断。它的亮点不是单一技巧，而是把高阶关系建模和辅助自监督较完整地接起来；但它也明确暴露了“更依赖 purchase signal、需要小 beta 和浅层结构”的边界。
