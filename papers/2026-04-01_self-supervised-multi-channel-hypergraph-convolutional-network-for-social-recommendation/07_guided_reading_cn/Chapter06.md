# Chapter 06 - Experimental Setup & Main Results

## 1. 本章范围
- Status: mastered
- Source: Section 4.1 Experimental Settings; Section 4.2 Recommendation Performance; Table 2; Table 3; Table 4
- Why this chapter: 方法章讲完以后，真正要问的就是“这个模型到底有没有被实验撑住”。这一章不先看 ablation，而是先把实验设计和主结果读扎实，判断作者的主结论是否成立。
- Reading focus: 你这章先不要急着记所有表格数字，而是先抓住四件事：用了什么数据、和谁比、怎么评、结果最强的证据是什么。

## 2. 阅读前你先抓住这三个问题
1. 作者为什么要同时汇报 `complete test set` 和 `cold-start test set`，这和 social recommendation 的任务目标有什么关系？
2. Table 3 和 Table 4 真正想证明的是两层结论：`MHCN` 本身有效，以及 `S^2-MHCN` 的 self-supervised 增强确实继续带来增益。你要分开读。
3. 实验里最需要警惕的不是“作者赢了没”，而是“赢的是谁”“赢在哪些数据上最明显”“这个胜利更像结构设计的功劳，还是 self-supervised 的功劳”。

## 3. 英文原文分段
### Passage 1
Datasets. Three real-world datasets: LastFM, Douban, and Yelp are used in our experiments. As our aim is to generate Top-K recommendation, for Douban which is based on explicit ratings, we leave out ratings less than 4 and assign 1 to the rest. We perform 5-fold cross-validation on the three datasets and report the average results.

### Passage 2
Baselines. We compare MHCN with a set of strong and commonly-used baselines including MF-based and GNN-based models: BPR, SBPR, LightGCN, GraphRec, DiffNet++, and DHCF. Two versions of the proposed multi-channel hypergraph convolutional network are investigated in the experiments. MHCN denotes the vanilla version and $S^2$-MHCN denotes the self-supervised version.

### Passage 3
Metrics. To evaluate the performance of all methods, two relevancy-based metrics Precision@10 and Recall@10 and one ranking-based metric NDCG@10 are used. We perform item ranking on all the candidate items instead of the sampled item sets to calculate the values of these three metrics, which guarantees that the evaluation process is unbiased.

### Passage 4
In this part, we validate if MHCN outperforms existing social recommendation baselines. Since the primary goal of social recommendation is to mitigate data sparsity issue and improve the recommendation performance for cold-start users. Therefore, we respectively conduct experiments on the complete test set and the cold-start test set in which only the cold-start users with less than 20 interactions are contained.

### Passage 5
The improvement is calculated by subtracting the best performance value of the baselines from that of $S^2$-MHCN and then using the difference to divide the former. Analogously, $S^2$-improvement is calculated by comparing the values of the performance of MHCN and $S^2$-MHCN.

### Passage 6
MHCN shows great performance in both the general and cold-start recommendation tasks. Even without self-supervised learning, it beats all the baselines by a fair margin. Meanwhile, self-supervised learning has great ability to further improve MHCN. Particularly, in the cold-start recommendation task, self-supervised learning brings significant gains. Besides, it seems that, the sparser the dataset, the more improvements self-supervised learning brings.

### Passage 7
GNN-based recommendation models significantly outperform the MF-based recommendation models. However, when compared with the counterparts based on the same building block, social recommendation models are still competitive and by and large outperform the corresponding general recommendation models except LightGCN.

### Passage 8
LightGCN is a very strong baseline. Without considering the two variants of MHCN, LightGCN shows the best or the second best performance in most cases.

### Passage 9
Although DHCF is also based on hypergraph convolution, it does not show any competence in all the cases. There are two possible causes which might lead to its failure. Firstly, it only exploits the user-item high-order relations. Secondly, the way to construct hyperedges is very impractical in this model, which leads to a very dense incidence matrix.

## 4. 重点概念与术语
- complete test set: 完整测试集，评估所有测试用户。
- cold-start test set: 冷启动测试集，只保留交互数少于 20 的用户，用来检查模型在稀疏用户上的能力。
- Precision@10 / Recall@10 / NDCG@10: 一个看前 10 个推荐里命中的精度，一个看覆盖率，一个看排序质量。
- unbiased evaluation: 这里指作者不是在 sampled candidates 上做评估，而是在全部候选物品上做排序。
- baseline family: 这里既有 MF-based baseline，也有 GNN-based baseline，还有 hypergraph-based baseline。
- vanilla `MHCN`: 没有 self-supervised 辅助任务的版本。
- `S^2-MHCN`: 加了 hierarchical mutual information 自监督后的版本。
- `Improv.`: `S^2-MHCN` 相对最佳 baseline 的提升比例。
- `S^2-Improv.`: `S^2-MHCN` 相对 vanilla `MHCN` 的提升比例。
- sparse dataset effect: 数据越稀疏，自监督带来的收益越可能更明显；这是论文在当前三组数据上观察到的趋势。
- strong baseline warning: 如果一个方法只比弱 baseline 强，结论价值有限；所以 LightGCN 在这篇论文里其实是非常关键的参照物。

## 5. 本章核心内容
### 5.1 先看实验设计，不然表格会读飘
这章的第一层不是“谁赢了”，而是“这场比较是否公平、是否对准任务”。作者用了三个真实数据集：LastFM、Douban、Yelp。Table 2 里最应该记住的不是所有绝对规模，而是它们都挺稀疏，而且 Yelp 最稀疏，density 只有 `0.11%`。这一点很重要，因为后面你会看到，self-supervised 增益在 Yelp 和 cold-start 场景里最明显。

Douban 原本是显式评分数据，所以作者把评分小于 4 的样本去掉，把剩下的评分转成 1，本质上把问题改写成隐式反馈下的 top-$K$ recommendation。评测时他们做 `5-fold cross-validation`，并且不是在 sampled items 上测，而是在全候选物品上做 item ranking，所以这套评估比很多“采样候选集”的设定更严格。

baseline 的选择也有层次。`BPR`、`SBPR` 是 MF 路线，`GraphRec`、`DiffNet++` 是 social recommendation 的 GNN 路线，`LightGCN` 是 general recommendation 的强 GNN baseline，`DHCF` 是 hypergraph recommendation baseline。与此同时，作者把自己方法拆成两个版本一起测：`MHCN` 和 `S^2-MHCN`。这意味着表格其实同时在回答两个问题：
- 多通道 hypergraph 设计本身行不行？
- 自监督增强到底有没有额外价值？

### 5.2 Table 3：完整测试集结果到底说明了什么
Table 3 是 general recommendation performance，也就是完整测试集上的结果。最直接的结论是：`S^2-MHCN` 在三组数据的 `P@10`、`R@10`、`NDCG@10` 上都排第一，而且 `MHCN` 本身通常也已经超过所有 baseline。

你可以抓三组最有代表性的数字：
- LastFM 上，`S^2-MHCN` 的 `NDCG@10 = 0.24395`，高于 `LightGCN = 0.23392`，也高于 `MHCN = 0.23834`。
- Douban 上，`S^2-MHCN` 的 `Recall@10 = 6.681%`，高于 `LightGCN = 6.247%` 和 `MHCN = 6.556%`。
- Yelp 上差距最明显，`S^2-MHCN` 的 `NDCG@10 = 0.06061`，而 `LightGCN = 0.04998`，`MHCN = 0.05356`。

所以 Table 3 传达的不是一句空泛的“作者赢了”，而是两层更细的判断：
第一层，`MHCN > baseline`，说明 motif-induced multi-channel hypergraph 结构本身就有效。
第二层，`S^2-MHCN > MHCN`，说明 self-supervised hierarchical MI 不是装饰项，而是在已有强结构上还能继续带来收益。

### 5.3 Table 4：为什么 cold-start 结果更关键
Table 4 是 cold-start recommendation performance，只看交互数少于 20 的用户。这张表比 Table 3 更重要，因为 social recommendation 的一个核心动机就是缓解 data sparsity，尤其是冷启动用户的问题。

这张表里自监督的价值更明显。你可以先看几组特别有代表性的数：
- LastFM 冷启动上，`NDCG@10` 从 `MHCN = 0.17218` 提到 `S^2-MHCN = 0.19138`，而 `DiffNet++` 是 `0.16031`。
- Douban 冷启动上，`Recall@10` 从 `MHCN = 9.646%` 提到 `S^2-MHCN = 10.632%`。
- Yelp 冷启动上，`NDCG@10` 从 `MHCN = 0.04354` 提到 `S^2-MHCN = 0.05143`，提升非常显眼。

这也是为什么作者特别强调：self-supervised learning 在 cold-start 任务里带来了显著收益。你可以把这里的直觉和 Chapter 4 连起来理解：当用户交互本来就少时，主任务监督更稀薄，辅助结构信号就更容易发挥价值。

### 5.4 `Improv.` 和 `S^2-Improv.` 要分开读
很多人读表格时会把这两列混掉，但它们回答的是两种完全不同的问题。

- `Improv.` 问的是：`S^2-MHCN` 相比“最佳 baseline”强多少。
- `S^2-Improv.` 问的是：在作者自己的模型内部，“加不加 self-supervised”差多少。

比如 Yelp 的 general recommendation 上，`NDCG@10` 的 `Improv. = 21.268%`，`S^2-Improv. = 13.162%`。这说明两件事同时成立：
- 它比最强 baseline 的优势很大。
- 它比自己去掉自监督的版本也明显更强。

所以这篇论文的实验不是只证明“我的方法整体最好”，还证明了“self-supervised 模块本身不是可有可无”。

### 5.5 作者自己怎么总结这些结果
作者在 Section 4.2 里一共强调了四个实验结论，这四条都值得记：

1. `MHCN` 和 `S^2-MHCN` 在 general 与 cold-start 两类测试上都很强，而且 `S^2-MHCN` 全面更优。
2. GNN-based recommendation 普遍强于 MF-based recommendation，这说明图传播框架确实比纯矩阵分解更适合这一任务。
3. `LightGCN` 是一个非常强的 baseline，说明“去掉多余非线性和变换矩阵”的简洁传播设计本身就有竞争力。
4. `DHCF` 虽然也是 hypergraph 方法，但表现并不好。作者认为主要是因为它只建模 user-item 高阶关系，而且 hyperedge 构造太稠密，容易带来 heavy computation 和 over-smoothing。

这里最值得你注意的是第 3 条，因为它让论文的胜利更有说服力。换句话说，这篇论文不是只赢了老旧 baseline，而是在一个很强的 `LightGCN` 面前仍然占优。

### 5.6 这一章在全文里的作用
这章的作用是把前面方法章的主张第一次拿到数据上检验。前面作者一直在说：
- pairwise social graph 不够；
- multi-channel motif-induced hypergraph 更好；
- hierarchical self-supervision 可以补聚合损失。

而 Section 4.1 + 4.2 负责先给出第一轮证据：这个方法不只是概念上漂亮，在主指标上也确实更强，尤其是在稀疏和 cold-start 场景里更强。

### 5.7 容易误解的点
- 这章只证明“主结果成立”，还没有拆清楚每个组件分别贡献多少；那是下一章 ablation 的任务。
- `S^2-MHCN` 在所有表里都最好，不等于每个子模块都一定必要；这个结论要等 Fig. 5 / Fig. 7 再看。
- Yelp 上提升最大，不代表“任何越稀疏的数据都一定越受益”；这是这篇论文在这三组数据上的经验观察。
- `LightGCN` 虽然不是 social recommendation model，但它是非常关键的强 baseline，因为它能帮我们判断“模型收益到底来自 social/high-order 设计，还是只是来自简洁传播结构”。

## 6. 证据指针
- Table 2; Section 4.1: 数据集规模、relation 数和 density。
- Section 4.1 Datasets paragraph: Douban 从显式评分转成 top-$K$ 隐式反馈。
- Section 4.1 Metrics paragraph: Precision@10、Recall@10、NDCG@10 与 unbiased evaluation。
- Section 4.1 Baselines paragraph: baseline 列表与 `MHCN` / `S^2-MHCN` 两版本设置。
- Section 4.2 opening paragraph: 为什么同时做 complete test 与 cold-start test。
- Table 3: general recommendation performance。
- Table 4: cold-start recommendation performance。
- Section 4.2 bullet conclusions: 作者对 LightGCN、DHCF、自监督增益和稀疏性趋势的解释。

## 7. 一分钟回顾
- 作者先用三组真实数据、全候选排序评估和一组强 baseline，搭了一个相对扎实的实验框架。
- Table 3 说明 `MHCN` 本身已经强，`S^2-MHCN` 又在其上进一步提升。
- Table 4 更关键，因为它表明 self-supervised 增益在 cold-start 用户上更明显。
- `LightGCN` 是强 baseline，`DHCF` 是失败对照，这两者一起帮助作者说明“胜利不是偶然”。
- 这章先回答“方法总体有没有用”，下一章才会回答“到底是哪一部分最有用”。

## 8. 你的问答区
- 你可以直接问：为什么 `LightGCN` 这么强，反而让这篇论文的结果更可信？
- 你也可以继续问：`Improv.` 和 `S^2-Improv.` 我还是会混，能不能帮我专门拿一行表格演算一遍？
- 如果你想自测，先回答：为什么 cold-start 表比 complete test 表更能体现这篇论文的价值？
- 如果你觉得这章已经顺了，直接回复“进入下一章”，我们就继续拆 ablation 和 parameter sensitivity。
- Q1：Precision，Recall，NDCG的定义是什么？
- Q2：“把剩下的评分转成 1，本质上把问题改写成隐式反馈下的 top-$K$ recommendation。评测时他们做 `5-fold cross-validation`，并且不是在 sampled items 上测，而是在全候选物品上做 item ranking，所以这套评估比很多“采样候选集”的设定更严格。”这一块我还不够理解。
- Q3：什么是MF路线
- Q4：如何理解Cold Start？

## 9. Codex 补充讲解
- Round 0: 实验章已经开始了。你这章最该抓住的一句话是：
  `作者先证明 multi-channel hypergraph 本身有效，再证明 self-supervised 在稀疏和 cold-start 场景里还能继续拉开差距。`
- 如果你现在只想先抓一个阅读主轴，就按这个顺序读：
  1. Table 2 看数据稀疏度；
  2. Table 3 看 general performance；
  3. Table 4 看 cold-start performance；
  4. 最后再回头读作者自己列出的四条结论。
- 这章有一个非常重要但容易忽略的判断：
  作者真正最想卖的，不只是“分数第一”，而是“在一个很强的 LightGCN baseline 前面，MHCN 和 S^2-MHCN 仍然能赢”。这会让后面的 ablation 更值得看。
- 先给你一个检查题，我们下一轮再继续拆：
  为什么说 `Table 4` 比 `Table 3` 更能体现这篇论文提出 social recommendation 方法的价值？
- Round 1: 回答你这轮补进来的 `Precision / Recall / NDCG / implicit feedback / 5-fold CV / sampled items / MF / cold start` 问题。
- Q1: `Precision@10`、`Recall@10`、`NDCG@10` 到底是什么意思？
  你可以先把它们都理解成“看前 10 个推荐结果”的三种角度。
  1. `Precision@10`
     意思是：推荐给用户的前 10 个物品里，有多少个是真的用户会喜欢的。
     直觉公式可以记成：
     `Precision@10 = 前10个里命中的个数 / 10`
     所以它强调的是“推荐列表有多准”。
  2. `Recall@10`
     意思是：用户真正喜欢的那些物品里，有多少被你的前 10 个推荐覆盖到了。
     直觉公式可以记成：
     `Recall@10 = 前10个里命中的个数 / 用户真实喜欢的物品总数`
     所以它强调的是“你漏掉了多少”。
  3. `NDCG@10`
     它不只看有没有命中，还看“命中的物品排得靠不靠前”。
     命中同样是 2 个，如果一个方法把它们排在第 1、2 名，另一个排在第 8、9 名，`NDCG@10` 会认为前者更好。
     所以它强调的是“排序质量”，不是只看命中数。
- 你可以把三者压成一句话记：
  `Precision 看准不准，Recall 看漏不漏，NDCG 看排得好不好。`
- Q2: “把剩下的评分转成 1、本质上改写成 implicit feedback top-K recommendation；5-fold cross-validation；不在 sampled items 上测而在全候选上测”这段怎么理解？
  这里其实是三件事。
  1. `Douban` 从显式评分改成隐式反馈
     原始 Douban 有 1-5 分这类 rating。
     但这篇论文做的是 `top-K recommendation`，它更关心“这个物品是不是用户偏好的正例”，而不是“用户到底打了几分”。
     所以作者把评分小于 4 的样本丢掉，把评分大于等于 4 的样本记成 1。
     这样问题就从“预测评分多少分”变成了“识别哪些物品是用户喜欢的正反馈”。
  2. `5-fold cross-validation`
     意思是把数据分成 5 份，轮流拿其中 1 份做测试、另外 4 份做训练，最后把 5 次结果取平均。
     它的作用是减少“一次随机划分刚好太幸运或太倒霉”的偶然性。
     所以你可以把它理解成：不是只测一次，而是测 5 次再平均，结果更稳。
  3. `sampled items` vs `all candidate items`
     很多论文为了省算力，不是在“全部候选物品”里排序，而是只从里面抽一小撮负样本物品，再和正样本一起排。
     这种做法更便宜，但也更容易把任务变简单。
     本文说自己是在 `all candidate items` 上做 item ranking，也就是用户可能被推荐到的全部物品都进入排序。
     这会更严格、更接近真实推荐场景，所以作者才说这样的 evaluation 更 `unbiased`。
- 再白话一点：
  `显式评分转 1` 是在改任务定义；
  `5-fold` 是在让结果更稳；
  `全候选排序` 是在让评估更严格。
- Q3: 什么是 `MF 路线`？
  `MF` 就是 `Matrix Factorization`。
  它的基本思想是：把用户和物品都表示成低维向量，然后用这两个向量的内积去预测偏好。
  例如本文里的打分形式 $\hat r_{u,i} = \mathbf p_u^\top \mathbf q_i$，本质上就有 MF 的味道。
  但当论文说 `MF-based baseline` 时，通常指的是：
  这些方法的核心能力主要来自“用户向量 + 物品向量 + 打分函数”，而不是来自图传播。
  在这篇论文里：
  - `BPR` 是 MF 路线；
  - `SBPR` 也是 MF 路线，只不过加了 social 信息；
  - `GraphRec`、`DiffNet++`、`LightGCN` 更偏 GNN / graph propagation 路线。
- 所以 `MF 路线` 你可以记成：
  `主要靠向量分解建模用户-物品关系，而不是靠多层图消息传播。`
- Q4: 如何理解 `cold start`？
  `cold start` 指的是：用户或物品刚进入系统，历史交互很少，模型没有足够行为数据可学。
  这篇论文这里讨论的是 `cold-start users`，也就是交互次数少于 20 的用户。
  为什么它重要？
  因为普通推荐模型很依赖历史交互；一旦交互太少，主监督信号就弱了，模型很容易学不好。
  social recommendation 的一个核心卖点就是：
  就算这个用户自己买过/点过的东西不多，系统也许还能借助她的 social relation 或高阶结构去补信息。
- 所以为什么我前面一直说 `Table 4` 更关键？
  因为 `Table 4` 测的正是这篇论文最想解决的痛点场景：数据稀疏、用户冷启动。
- 把 `cold start` 和本文连起来的一句白话是：
  `当用户自己的交互历史不够时，模型要靠 social + high-order structure 去替她“补背景”。`
- 如果你还想继续往下追，我下一轮可以专门做两件事里的任意一个：
  1. 直接拿 `Table 3` 或 `Table 4` 的某一行，手把手给你演算 `Improv.` 和 `S^2-Improv.`。
  2. 把 `Precision / Recall / NDCG` 画成一个“前10推荐列表示意图”式的中文直观讲解。
