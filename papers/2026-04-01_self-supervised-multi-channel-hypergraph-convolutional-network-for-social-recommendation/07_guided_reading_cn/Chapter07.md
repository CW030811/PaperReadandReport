# Chapter 07 - Ablation, Sensitivity & Final Takeaways

## 1. 本章范围
- Status: active
- Source: Section 4.3 Ablation Study; Section 4.4 Parameter Sensitivity Analysis; Section 5 Conclusion; Figure 5; Figure 6; Figure 7; Figure 8; Figure 9
- Why this chapter: 前一章只证明“整体方法有用”，这一章才真正回答“到底哪一部分最有用”“自监督是不是合理设计”“超参数和深度透露了什么模型边界”，最后再用结论段把整篇论文收束起来。
- Reading focus: 你这章要把“证明贡献”和“暴露限制”同时读出来。好的 ablation 不是只会夸模型，而是也会告诉你模型最依赖什么、最怕什么。

## 2. 阅读前你先抓住这三个问题
1. Figure 5 和 Figure 6 真正想回答的是：三个 channel 里谁最关键，attention 学出来的权重分布是否和 ablation 观察一致？
2. Figure 7 真正想回答的是：为什么作者坚持 hierarchical MIM，而不是只做 local、只做 global，或直接照搬 DGI？
3. Figure 8、Figure 9 和 Conclusion 连起来后，你应该能说清：这个模型最有效的配置是什么，它的主要风险边界又是什么？

## 3. 英文原文分段
### Passage 1
We first investigate the multi-channel setting by removing any of the three channels from $S^2$-MHCN and leaving the other two to observe the changes of performance. From Fig. 5, we can observe that removing any channel would cause performance degradation. But it is obvious that purchase channel contributes the most to the final performance.

### Passage 2
Without this channel, $S^2$-MHCN falls to the level of LightGCN shown in Table 3. By contrast, removing Social channel or Joint channel would not have such a large impact on the final performance. Comparing Social channel with Joint channel, we can observe that the former contributes slightly more on LastFM and Yelp, while the latter is more important on Douban.

### Passage 3
According to Fig. 6, we can observe that, for the large majority of users in LastFM, Social channel has limited influence on the comprehensive user representations. In line with the conclusions from Fig. 5, Purchase channel plays the most important role in shaping the comprehensive user representations. The importance of Joint channel falls between the other two.

### Passage 4
To investigate the effectiveness of the hierarchical mutual information maximization (MIM), we break this procedure into two parts: local MIM between the user and user-centered sub-hypergraph, and global MIM between the user-centered sub-hypergraph and hypergraph. We also compare hierarchical MIM with the node-graph MIM used in DGI to validate the rationality of our design.

### Passage 5
As can be seen, hierarchical MIM shows the best performance while local MIM achieves the second best performance. By contrast, global MIM contributes less but it still shows better performance on Douban and Yelp when compared with DGI. Actually, DGI almost rarely contributes on the latter two datasets and we can hardly find a proper parameter that can make it compatible with our task.

### Passage 6
As we adopt the primary & auxiliary paradigm, to avoid the negative interference from the auxiliary task in gradient propagating, we can only choose small values for $\beta$. With the increase of the value of $\beta$, the performance of $S^2$-MHCN on all the datasets rises. After reaching the peak when $\beta$ is 0.01 on all the datasets, it steadily declines.

### Passage 7
According to Fig. 8, we can draw a conclusion that even a very small $\beta$ can promote the recommendation task, while a larger $\beta$ would mislead it. The recommendation task is sensitive to the magnitude of self-supervised task.

### Passage 8
According to Fig. 9, the best performance of $S^2$-MHCN is achieved when the depth of $S^2$-MHCN is 2. With the continuing increase of the number of layer, the performance of $S^2$-MHCN declines on all the datasets. Obviously, a shallow structure fits $S^2$-MHCN more.

### Passage 9
A possible reason is that $S^2$-MHCN aggregates high-order information from distant neighbors. As a result, it is more prone to encounter the over-smoothing problem with the increase of depth.

### Passage 10
In this paper, we fuse hypergraph modeling and graph neural networks and then propose a multi-channel hypergraph convolutional network (MHCN) which works on multiple motif-induced hypergraphs to improve social recommendation. To compensate for the aggregating loss in MHCN, we innovatively integrate self-supervised learning into the training of MHCN.

### Passage 11
The self-supervised task serves as the auxiliary task to improve the recommendation task by maximizing hierarchical mutual information between the user, user-centered sub-hypergraph, and hypergraph representations. The extensive experiments conducted on three public datasets verify the effectiveness of each component of MHCN, and also demonstrate its state-of-the-art performance.

## 4. 重点概念与术语
- ablation study: 消融实验。通过去掉某个组件，看性能如何变化，从而判断该组件是否真正有贡献。
- multi-channel setting: 本文的 social / joint / purchase 三通道设计。
- purchase channel: 由购买相关 motif 诱导出的 channel，实验里被证明是最关键的一支。
- attention weight distribution: 不只是看平均权重，而是看不同用户上三个 channel 权重的分布情况。
- local MIM: `user <-> user-centered sub-hypergraph` 这一层 mutual information。
- global MIM: `user-centered sub-hypergraph <-> hypergraph` 这一层 mutual information。
- hierarchical MIM: 把 local 和 global 两层一起做的 self-supervised 设计，也就是本文最终采用的版本。
- DGI baseline: `node <-> graph` 的 coarse mutual information 设计，是本文拿来对照的自监督 baseline。
- Disabled: Figure 7 里表示不使用 self-supervised task，也就是 vanilla `MHCN`。
- parameter sensitivity: 看模型性能对超参数变化是否稳定。
- $\beta$ sensitivity: 看 self-supervised 辅助项权重多大最合适。
- depth sensitivity / $L$: 看超图卷积层数多深最合适。
- over-smoothing: 图/超图网络层数太深时，节点表示越来越相似，导致区分能力下降。
- final takeaway: 最后一章真正应该带走的结论，不只是“分数高”，而是“为什么高”和“边界在哪里”。

## 5. 本章核心内容
### 5.1 Figure 5：三个 channel 不是同等重要
这一小节最重要的不是“删了哪个都会掉点”，而是“掉得有多不一样”。Figure 5 里作者把 `S^2-MHCN` 的三个 channel 分别去掉，看性能怎么掉。结果很明确：去掉任何一个 channel，性能都会下降，说明三者都不是完全冗余的；但其中 `purchase channel` 掉得最狠。

作者甚至明确说，去掉 purchase channel 以后，模型性能会掉到接近 `LightGCN` 的水平。这句话其实信息量很大，因为它意味着：
- 多通道结构里最能撑住最终性能的是 purchase-related high-order signal。
- social / joint 两支虽然有用，但更像是在强主干上的补充增强，而不是决定胜负的唯一来源。

同时，不同数据集上 `social channel` 和 `joint channel` 的相对重要性并不完全一样。作者观察到：
- 在 LastFM 和 Yelp 上，`social channel` 略强一些。
- 在 Douban 上，`joint channel` 更重要一些。

这也说明一个很好的点：论文没有把三种高阶关系说成“固定排序”，而是承认它们会随数据集语义而变。

### 5.2 Figure 6：attention 权重分布和 ablation 结论是否一致
Figure 5 只是告诉你“删谁更疼”，Figure 6 则是在问：当三个 channel 同时存在时，attention 实际把谁看得更重？

作者把 learned attention scores 可视化后发现，结论基本和 Figure 5 一致：
- `purchase channel` 对综合用户表示最重要；
- `joint channel` 处在中间；
- `social channel` 在很多用户上影响有限，尤其是 LastFM。

这一点非常关键，因为它构成了“结构消融”和“内部注意力权重”之间的相互验证。也就是说，模型不是嘴上说 purchase channel 重要，而是：
- 去掉它，性能明显下降；
- 保留它时，attention 也真的给了它更高权重。

作者给出的一个解释也值得记：仅仅 socially connected 的用户未必真的偏好相近，因为显式 social relation 可能带噪声。这会让纯 social signal 没有我们直觉中那么强。

### 5.3 Figure 7：为什么 hierarchical MIM 比 local / global / DGI 更合理
这一组实验是整篇论文最关键的自监督证据。作者把 self-supervised 设计拆成几种版本来比：
- `Disabled`：完全不加 self-supervised，也就是 vanilla `MHCN`
- `Local`：只做 `user <-> sub-hypergraph`
- `Global`：只做 `sub-hypergraph <-> hypergraph`
- `DGI`：做 `node <-> graph` 式的 coarse MIM
- `Hierarchical`：本文完整设计

结果是：
- `Hierarchical` 最好；
- `Local` 第二；
- `Global` 贡献较弱；
- `DGI` 在后两个数据集上几乎没什么帮助，甚至有时还会拖后腿。

这个排序其实和 Chapter 4 的方法直觉完全对上了。因为用户最直接相关的是“自己周围那片局部高阶结构”，所以 `local MIM` 比 `global MIM` 更有效很合理；而完整的 `hierarchical MIM` 在 local 基础上再补了一层 global 结构约束，所以最好。

反过来，`DGI` 不太行，也正好支持了作者的论点：`node <-> whole graph` 这一层太粗了，不够贴合这个 social recommendation 场景。

### 5.4 Figure 8：为什么 $\beta$ 不能大
Figure 8 在看 self-supervised 辅助项强度，也就是 $\beta$。作者一开始就强调，采用 `primary & auxiliary` 范式时，为了避免辅助任务在梯度上传播时干扰主任务，$\beta$ 只能选小值。

实验上他们发现：
- 随着 $\beta$ 从很小的值开始增加，性能先上升；
- 当 $\beta = 0.01` 时，三个数据集都达到峰值；
- 再继续增大，性能稳定下降。

这说明两件事：
第一，self-supervised task 确实有用，哪怕很小的权重都能帮到主任务。
第二，它不能喧宾夺主。权重大了以后，辅助目标反而会“误导” recommendation task。

所以这里最该记住的不是某个具体最佳数值，而是这个结构性结论：
`辅助任务适合做轻量结构正则，不适合压过主任务。`

### 5.5 Figure 9：为什么 2 层最好，深了反而差
Figure 9 看的是深度 $L$。作者把 hypergraph convolution 从 1 层堆到 5 层，结果发现：
- 最好的是 2 层；
- 再往深堆，三个数据集上都在掉。

这说明 `S^2-MHCN` 更适合 shallow structure，而不是越深越强。作者给出的解释也很自然：超图卷积本来就在聚合高阶、远距离的结构信息，如果再不断叠层，很容易让不同节点表示越来越像，也就是 `over-smoothing`。

这一点其实也让整篇论文更可信，因为它没有把方法包装成“无限堆深都没问题”。相反，作者承认：
- hypergraph modeling 很强；
- 但也更容易遇到过平滑；
- 因此 shallow model 更适合当前方法。

### 5.6 Section 5 Conclusion：整篇论文最后到底留下了什么
最后的结论段其实很短，但它做了三件事。

第一，它重新强调问题缺口：现有 social recommendation 大多只建模 `pairwise interactions`，忽略了现实中的 `high-order user interactions`。

第二，它重新概括方法贡献：作者把 hypergraph modeling 和 graph neural networks 融合，提出 `MHCN`；又通过 `hierarchical mutual information` 的 self-supervised auxiliary task 去补偿 aggregation loss。

第三，它给整篇论文下了一个实验层面的结论：三组公开数据上的 extensive experiments 支持了两个说法：
- MHCN 的各个组件是有效的；
- 整体方法达到了 state-of-the-art performance。

### 5.7 这一章在全文里的作用
如果说 Chapter 6 负责证明“方法整体赢了”，那 Chapter 7 负责把这个“赢”拆开。

它回答了三类更高级的问题：
- 赢主要靠哪个 channel？
- 自监督设计到底是不是合理、是不是比 DGI 更贴场景？
- 这个模型的最佳工作区间和结构边界在哪里？

所以这章读完以后，你不应该只会说“这篇论文结果很好”，而应该能说出更完整的一句：
`这篇论文的主胜负手是 purchase-related high-order signal，加上的 hierarchical self-supervision 在冷启动场景尤其有用，但它需要小 beta 和浅层结构，否则会被 over-smoothing 和任务干扰反噬。`

### 5.8 容易误解的点
- `purchase channel` 最重要，不等于 social channel 没用；更准确地说，它是“最强主干”，而 social / joint 是增益来源。
- `local MIM` 比 `global MIM` 强，不等于 global 没价值；真正最强的是两者合起来的 `hierarchical MIM`。
- `beta = 0.01` 在本文实验里最好，不等于别的数据集永远也是这个值；更稳的理解是“beta 应该小”。
- 2 层最好，不等于作者方法弱；相反，这揭示了高阶超图模型本来就更容易过平滑，所以更需要克制深度。
- conclusion 里说 `state-of-the-art`，你读的时候要把它和本文实验范围绑定起来理解：是相对这组 baselines、这组三个数据集而言的结果。

## 6. 证据指针
- Section 4.3.1; Figure 5: 去掉任一 channel 都降性能，且 purchase channel 最关键。
- Section 4.3.1; Figure 6: attention weight 分布与 channel 贡献排序基本一致。
- Section 4.3.2; Figure 7: hierarchical / local / global / DGI / disabled 的自监督对照实验。
- Section 4.4; Figure 8: $\beta$ 从小到大先升后降，0.01 附近最好。
- Section 4.4; Figure 9: 深度 2 最优，继续加深性能下降。
- Section 4.4 closing paragraph: over-smoothing 是 hypergraph convolution based models 的潜在共性问题。
- Section 5 Conclusion: pairwise limitation、MHCN + self-supervised 贡献、以及整体实验结论。

## 7. 一分钟回顾
- Figure 5 / 6 说明三个 channel 都有用，但 `purchase channel` 是最关键的一支。
- Figure 7 说明本文的 `hierarchical MIM` 确实比只做 local、只做 global 或照搬 DGI 更合理。
- Figure 8 说明 self-supervised 要“小剂量”最好，$\beta$ 太大反而干扰主任务。
- Figure 9 说明 `S^2-MHCN` 更适合浅层结构，深了容易 over-smoothing。
- 最终结论不是一句“结果很好”，而是“方法有效、组件有证据、边界也被实验暴露出来了”。

## 8. 你的问答区
- 你可以直接问：为什么 `purchase channel` 会比 `social channel` 更重要，这会不会削弱这篇 social recommendation 论文的说服力？
- 你也可以继续问：Figure 7 里为什么 `local MIM` 会比 `global MIM` 更强？
- 如果你想自测，先回答：为什么作者说 `beta` 应该小，而且深度 2 比更深更合适？
- 如果你觉得这章已经顺了，直接回复“整篇论文总结一下”，我就带你做全篇总复盘。

## 9. Codex 补充讲解
- Round 0: Chapter 7 已经开始了。这章最值得你记住的一句话是：
  `作者不只证明模型整体有效，还证明了最重要的组件是谁、辅助任务为什么合理，以及模型在哪些地方会失效。`
- 读这一章的推荐顺序不是按论文顺序硬读到底，而是：
  1. 先看 Figure 5 / 6，搞清谁最重要；
  2. 再看 Figure 7，判断自监督设计是否真的站得住；
  3. 最后看 Figure 8 / 9，理解模型边界；
  4. 再回头读 Conclusion，会更容易判断作者有没有夸大。
- 这章里有一个很重要的“平衡感”：
  前面两组图在证明“模型为什么强”，后面两组图在提醒“模型不能乱调、也不能乱堆层”。这正是好论文的收尾方式。
- 先给你一个检查题，我们下一轮再继续拆：
  为什么说 Figure 7 其实是在给 Chapter 4 的 `hierarchical mutual information` 设计做“回头验收”？
