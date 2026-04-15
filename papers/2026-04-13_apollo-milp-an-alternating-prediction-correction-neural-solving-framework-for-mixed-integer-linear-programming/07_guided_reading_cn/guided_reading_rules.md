# Guided Reading Rules

## Purpose
- 本目录用于交互式中文领读，不替代标准的 paper extraction、deep notes、slides 或 compare 输出。
- 目标是让用户逐章吃透论文，并把问题沉淀在对应章节文件里。
- 领读默认尊重论文原始章节铺陈顺序，尽量保留作者的论证展开，而不是重排成一份压缩讲义。

## Generation Policy
- 一次只生成一个新的 `ChapterXX.md` 文件。
- 开始领读时，先更新 `reading_map.md`，再生成当前激活章节。
- 默认优先复用：
  - `03_extraction/paper_brief.md`
  - `02_notes/deep_notes.md`
  - `05_slides/slides_outline.md`
  - `01_source_text/paper_text.txt`
- 默认按论文原始一级/二级章节顺序推进，不跨章节重排内容。
- 只有当单个章节明显过长、超出一次阅读负担时，才做最小必要拆分；拆分后仍必须保持原文先后顺序。
- 原文领读部分应覆盖当前章节的大部分关键段落，并按连续 meaning blocks 排列，而不是只摘几句结论。

## Revision Policy
- 用户在当前章的问答区留言后，Codex 必须先读取该章节并补充 `## 9. Codex 补充讲解`。
- 补充讲解应优先解释：
  - 具体句子在说什么
  - 术语在本论文中的确切含义
  - 前后逻辑是怎样衔接的
  - 容易误解的点是什么
- 需要纠错时，保留原章节结构，尽量用增量补充而不是整体重写。
- 如果发现早先分章与论文原始顺序偏差过大，应先修正 `reading_map.md`，再在不打乱已学内容的前提下渐进调整后续章节。

## Graduation Rule
- 只有当用户明确确认“这章吃透了”“进入下一章”或同等意思时，当前章才能从 `active/reviewing` 变为 `mastered`。
- 未确认掌握前，不生成下一章正文。

## Chapter Writing Contract
- 每个章节必须保留下列一级结构：
  - `## 1. 本章范围`
  - `## 2. 阅读前你先抓住这三个问题`
  - `## 3. 英文原文分段`
  - `## 4. 重点概念与术语`
  - `## 5. 本章核心内容`
  - `## 6. 证据指针`
  - `## 7. 一分钟回顾`
  - `## 8. 你的问答区`
  - `## 9. Codex 补充讲解`
- 英文原文要按原文顺序的连续语义块展示，长段必须主动拆分，但不能跳段重排。
- 解释应以中文为主，重点概念可保留英文原词。
- 所有关键判断尽量给出 section、figure、table 或 equation 指针；拿不准时标记为 `uncertain`。
