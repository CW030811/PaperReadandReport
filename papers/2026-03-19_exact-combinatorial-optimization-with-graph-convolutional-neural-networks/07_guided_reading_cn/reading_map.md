# Guided Reading Map

## Paper
- Title: Exact Combinatorial Optimization with Graph Convolutional Neural Networks
- Paper Folder: `papers/2026-03-19_exact-combinatorial-optimization-with-graph-convolutional-neural-networks`
- Main Source: `01_source_text/paper_text.txt`
- Supporting Files:
  - `03_extraction/paper_brief.md`
  - `02_notes/deep_notes.md`
  - `05_slides/slides_outline.md`

## Reading Goal
- 通过逐章中文领读吃透这篇论文，而不是停留在“GCNN 学 branching”这句口号上。
- 重点理解四条主线：问题定义、图表示与模型、output/loss/inference 对齐、实验到底证明了什么。
- 每一章都要能回答“这部分为什么在这里出现，以及它对全文贡献了什么”。

## Chapter Plan
| Chapter | Title | Source Range | Why This Split | Status |
|---|---|---|---|---|
| Chapter01 | Abstract & Introduction | Abstract; Section 1 Introduction | 先建立问题背景、任务对象、论文主张与全文路线图 | mastered |
| Chapter02 | Related Work | Section 2 Related work | 先分清这篇论文相对已有 imitation-learning branching 工作到底新在哪 | mastered |
| Chapter03 | MILP Problem Definition | Section 3.1 | 先把 MILP 与 B&B 的基本对象和符号讲明白 | mastered |
| Chapter04 | Branching Rules and MDP Framing | Sections 3.2-3.3; Figure 1 | 这一章承接背景并引出“为什么 branching 可视为 sequential decision making” | mastered |
| Chapter05 | Why Imitation Learning Instead of RL | Section 4 opening paragraphs | 单独拆出训练范式选择，帮助理解作者为何避开 RL | active |
| Chapter06 | State Representation | Section 4.2; Figure 2 left | 图表示是这篇论文最关键的表示层设计，值得单拆 | planned |
| Chapter07 | GCNN Architecture | Section 4.3; Figure 2 right | 模型结构本身术语密度高，需要单独读 | planned |
| Chapter08 | Output, Loss, and Inference Alignment | Section 4.1; Section 4.3; Equation (3) | 这篇论文最值得吃透的链条之一，单独成章最清楚 | planned |
| Chapter09 | Dataset Collection and Training Details | Section 5.1; Supplementary dataset/training details | 标签与数据采集是方法可信度的重要来源 | planned |
| Chapter10 | Main Experimental Results | Section 5.2; Tables 1-2 | 单独讨论结果与泛化，避免和实验设置混在一起 | planned |
| Chapter11 | Ablation, Limitations, and Takeaway | Section 5.3; Section 6; Section 7; Table 3 | 最后收束 architectural evidence、局限和真正 takeaway | planned |

## Unlock Rule
- 任意时刻只允许一个 `active` 或 `reviewing` 章节。
- 只有当用户明确表示“这章吃透了”或“进入下一章”时，当前章节才可标记为 `mastered`。
- 用户如果在当前章底部提问，Codex 必须先补当前章，再决定是否进入下一章。

## Notes for Codex
- 先读 `03_extraction/paper_brief.md`、`02_notes/deep_notes.md`、`05_slides/slides_outline.md` 再写领读，保证讲解与标准产物一致。
- 这一篇 paper 的教学重点不是“它用了 GNN”，而是“它把 learned branching 嵌回 exact solver 的哪个局部环节”。
- 要反复提醒用户：这篇论文学的是 brancher，不是端到端求解器。
- 遇到和后文高度相关但本章尚未展开的内容，只做轻提示，不要提前剧透太多技术细节。
