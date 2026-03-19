---
name: paper-to-slides
description: Convert one paper's structured extraction into a presentation-ready reading-report outline. Use when the current paper already has extraction notes and the task is to generate or refine `05_slides/slides_outline.md` for a talk, report, or biweekly update.
---

# Paper To Slides

Turn the current paper workspace into the repository's default 8-slide outline.

## Workflow

1. Read the current paper workspace, especially:
   - `03_extraction/paper_brief.md`
   - `03_extraction/paper_schema.json`
   - `02_notes/deep_notes.md`
2. Write or refine `05_slides/slides_outline.md`.
3. Keep the default slide order:
   - paper info and background
   - research question
   - high-level idea
   - dataset and label
   - network structure
   - output, loss, and inference
   - novel contribution
   - strengths, weaknesses, and takeaways
4. For every slide include:
   - title
   - 3 to 5 bullets
   - suggested figure
   - short speaker notes
5. Keep one slide per `## Slide N` heading.
6. Run `python3 scripts/check_paper_outputs.py <paper_dir>` before finishing.
   On Windows without Python, use `scripts\check_paper_outputs.cmd <paper_dir>`.

## Output Rules

- Prefer concise bullets over paragraphs.
- Keep speaker notes short and presentation-ready.
- Reuse terminology from the paper extraction.
- Do not introduce claims that are missing from the evidence.
