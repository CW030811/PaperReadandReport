---
name: paper-guided-read-cn
description: Create and maintain an interactive Chinese guided-reading track for one paper. Use when the user asks for 中文领读, 陪读, 逐章讲解, chapter-by-chapter reading notes, chapter Q&A inside markdown, or wants to unlock the next chapter only after confirming understanding.
---

# Paper Guided Read CN

Create and maintain the optional `07_guided_reading_cn/` teaching track for the current paper workspace.

## Workflow

1. Confirm the current paper workspace path.
2. If `07_guided_reading_cn/` is missing, initialize it with:
   - `python3 scripts/init_guided_reading.py <paper_dir>`
   - On Windows without Python, use `scripts\init_guided_reading.cmd <paper_dir>`
3. Read the current paper workspace, especially:
   - `03_extraction/paper_brief.md`
   - `02_notes/deep_notes.md`
   - `05_slides/slides_outline.md`
   - `01_source_text/paper_text.txt`
   - existing files under `07_guided_reading_cn/`
4. Update `07_guided_reading_cn/reading_map.md` before writing or advancing chapters.
5. On the first guided-reading request for a paper:
   - create or refine the chapter plan
   - set exactly one chapter to `active`
   - generate only that chapter file
6. On later turns:
   - if the user asks questions about the current chapter, update the same `ChapterXX.md` under `## 9. Codex 补充讲解`
   - if the user explicitly confirms mastery and asks to continue, mark the current chapter `mastered`, activate the next one, and generate only the next chapter

## Chapter Split Rules

- Prefer learning-sized units over raw paper section boundaries.
- Short sections can be merged, such as `Abstract + Introduction`.
- Long sections must be split proactively, especially:
  - formulation
  - methodology
  - model definition
  - training objective
  - experiment setup
  - main results
  - ablations and discussion
- Each chapter should be thematically narrow, logically self-contained, and light enough for one reading session.

## File Contract

Required scaffold files:
- `07_guided_reading_cn/reading_map.md`
- `07_guided_reading_cn/guided_reading_rules.md`
- `07_guided_reading_cn/progress_log.md`

Per-chapter file structure must preserve:
- `## 1. 本章范围`
- `## 2. 阅读前你先抓住这三个问题`
- `## 3. 英文原文分段`
- `## 4. 重点概念与术语`
- `## 5. 本章核心内容`
- `## 6. 证据指针`
- `## 7. 一分钟回顾`
- `## 8. 你的问答区`
- `## 9. Codex 补充讲解`

## Writing Rules

- Keep the English source text segmented by meaning blocks; do not dump an entire long section unstructured.
- Explain in Chinese, but retain important paper terminology in English.
- Reuse the repository's existing extraction artifacts so the teaching track stays aligned with the core reading pipeline.
- Treat uncertain claims as `uncertain`; do not overstate evidence.
- Do not generate the next chapter unless the user clearly says the current chapter is understood.

## Validation

- After initializing or editing the guided-reading track, run:
  - `python3 scripts/check_guided_reading.py <paper_dir>`
  - On Windows without Python, use `scripts\check_guided_reading.cmd <paper_dir>`
- Keep the main paper checker separate:
  - `python3 scripts/check_paper_outputs.py <paper_dir>`
  - On Windows without Python, use `scripts\check_paper_outputs.cmd <paper_dir>`
