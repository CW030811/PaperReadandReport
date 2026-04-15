# Guided Reading Map

## Paper
- Title: Apollo-MILP: An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear Programming
- Paper Folder: papers/2026-04-13_apollo-milp-an-alternating-prediction-correction-neural-solving-framework-for-mixed-integer-linear-programming
- Main Source: `01_source_text/paper_text.txt`
- Supporting Files:
  - `03_extraction/paper_brief.md`
  - `02_notes/deep_notes.md`
  - `05_slides/slides_outline.md`

## Reading Goal
- 通过逐章中文领读吃透 Apollo-MILP 的主线，而不是把它压缩成“又一个更强的 MILP predictor”。
- 重点看清楚作者如何从 ND / PS 的 fixing-vs-search 矛盾，推到 prediction-correction 的新框架。
- 这一轮领读默认沿着论文原始章节顺序推进，尽量保留作者从动机、到方法、到理论、到实验的展开节奏。
- 每一章都要能回答“这部分在全文里起什么作用”，以及“它为后文埋了什么伏笔”。

## Chapter Plan
| Chapter | Title | Source Range | Why This Split | Status |
|---|---|---|---|---|
| Chapter01 | Abstract & Section 1 Introduction | Abstract + Section 1 Introduction | 先建立问题、基线局限、Apollo-MILP 总览与 contribution list | mastered |
| Chapter02 | Section 2 Related Works | Section 2 Related Works | 按原文顺序澄清本文位于哪条 ML4CO / MILP solution prediction 脉络里 | active |
| Chapter03 | Section 3 Preliminaries | Section 3.1 + Section 3.2 + Section 3.3 | 不跳步补齐 MILP 形式、bipartite graph 表示和 PS 基线 | planned |
| Chapter04 | Section 4.1 Prediction Step | Section 4.1 | 进入方法正文，先看 predictor、augmentation、target 与训练 loss | planned |
| Chapter05 | Section 4.2 Correction Step | Section 4.2 | 顺着原文看 trust-region corrector 与 UEBO 的来源和作用 | planned |
| Chapter06 | Section 4.3 Analysis of the Fixing Strategy | Section 4.3 + Algorithm 1 | 完成 consistency rule、理论保证和算法闭环 | planned |
| Chapter07 | Section 5.1 Experiment Settings | Section 5.1 | 先看 benchmark、baseline、metrics、iteration budget | planned |
| Chapter08 | Section 5.2 Main Evaluation | Section 5.2 | 检查主结果是否真正支撑作者的核心主张 | planned |
| Chapter09 | Section 5.3 Ablation Study | Section 5.3 | 看 fixing strategy、warm-start、迭代次数与 augmentation 是否必要 | planned |
| Chapter10 | Section 6 Conclusion & Future Works | Section 6 Conclusion and Future Works | 收束贡献、边界和后续值得继续追问的点 | planned |

## Unlock Rule
- 任意时刻只允许一个 `active` 或 `reviewing` 章节。
- 只有当用户明确表示“这章吃透了”或“进入下一章”时，当前章节才可标记为 `mastered`。
- 用户如果在当前章底部提问，Codex 必须先补当前章，再决定是否进入下一章。

## Notes for Codex
- 先读 `03_extraction/paper_brief.md`、`02_notes/deep_notes.md`、`05_slides/slides_outline.md` 再写领读。
- 优先遵循原文一级/二级章节顺序；除非章节过长，否则不要重排分章顺序。
- 不要把公式总览、实验截图或结论摘句提前抽成“插播章”；若需要补充，也只能作为对应原章里的小节。
- `## 3. 英文原文分段` 要覆盖当前 Source Range 的主要论证链条，避免过度压缩成零散摘句。
