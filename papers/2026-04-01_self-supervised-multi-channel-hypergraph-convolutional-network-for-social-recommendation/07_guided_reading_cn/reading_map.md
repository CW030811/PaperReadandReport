# Guided Reading Map

## Paper
- Title: Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation
- Paper Folder: papers\2026-04-01_self-supervised-multi-channel-hypergraph-convolutional-network-for-social-recommendation
- Main Source: `01_source_text/paper_text.txt`
- Supporting Files:
  - `03_extraction/paper_brief.md`
  - `02_notes/deep_notes.md`
  - `05_slides/slides_outline.md`

## Reading Goal
- 通过逐章中文领读吃透论文主线，而不是只停留在摘要级理解。
- 优先搞清楚：问题定义、三类高阶关系、MHCN 的多通道结构、自监督目标、实验结论和真实创新点。
- 每一章都要回答“这一部分在全文里到底起什么作用”。

## Chapter Plan
| Chapter | Title | Source Range | Why This Split | Status |
|---|---|---|---|---|
| Chapter01 | Abstract & Introduction | Abstract + Section 1 | 先建立任务背景、问题缺口与方法总览 | mastered |
| Chapter02 | Problem Setup & Hypergraph Construction | Section 3.1 + Section 3.2.1 | 先把符号、任务和 motif-induced hypergraph 建起来 | active |
| Chapter03 | Multi-Channel Propagation & Aggregation | Section 3.2.2 + Section 3.2.3 | 分清三通道卷积、self-gating 和 attention 聚合 | planned |
| Chapter04 | Self-Supervision, Objective & Complexity | Section 3.2.4 + Section 3.3 + Section 3.4 | 单独消化 BPR、自监督目标和复杂度边界 | planned |
| Chapter05 | Experimental Setup & Main Results | Section 4.1 + Section 4.2 | 先看数据、基线、指标，再判断主结果是否站得住 | planned |
| Chapter06 | Ablation, Sensitivity & Final Takeaways | Section 4.3 + Section 4.4 + Section 5 | 收尾看组件贡献、参数敏感性和结论边界 | planned |

## Unlock Rule
- 任意时刻只允许一个 `active` 或 `reviewing` 章节。
- 只有当用户明确表示“这章吃透了”或“进入下一章”时，当前章节才可标记为 `mastered`。
- 如果用户继续追问当前章节内容，先补充当前章，不提前生成下一章正文。

## Notes for Codex
- 先读 `03_extraction/paper_brief.md`、`02_notes/deep_notes.md`、`05_slides/slides_outline.md`，再写领读内容。
- 章节切分优先服从“主题单一、逻辑闭环、单次阅读负担适中”，不机械照搬论文一级标题。
- 每次只维护当前激活章节；用户未确认掌握前，不生成后续章节正文。
