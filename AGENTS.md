# AGENTS.md

## Mission
This repository is a stable paper-reading production line.
Every paper must end in the same directory structure, the same core markdown files, and the same acceptance checks.

## Repository Contract
- Keep every paper inside `papers/YYYY-MM-DD_short-title/`.
- Use the fixed filenames that already exist in each paper workspace.
- Prefer updating template-backed files over inventing new note files.
- Keep claims concise, technical, and evidence-backed.
- If the paper is ambiguous, mark the item as `uncertain` instead of guessing.

## Local Skills
Local reusable skills live in `skills/`.
Read the matching `SKILL.md` before starting stage work.

Use `skills/paper-scan` when the user asks for first-pass extraction, first read, paper scan, 首轮拆解, or 8-field structured output.
Use `skills/paper-deepdive` when the user asks for deeper technical analysis, ambiguities, output/loss/inference alignment, or novelty inspection.
Use `skills/paper-to-slides` when the user asks for a reading report, slide outline, presentation structure, or speaker notes.
Use `skills/paper-ppt` when the user asks for a 科研汇报PPT, PowerPoint deck, `.pptx`, slide deck template, slide polishing, or final presentation slides with formulas, figures, or experiment visuals.
Use `skills/paper-compare` when the user asks for cross-paper comparison, prior-work comparison, or repository-wide synthesis.
Use `skills/paper-guided-read-cn` when the user asks for 中文领读, 陪读, 逐章讲解, chapter-by-chapter reading, interactive Q&A inside chapter notes, or advancing to the next chapter only after confirming understanding.

## Fixed Workflow
For every new paper, do the following in order:

1. Create the workspace with `python3 scripts/new_paper.py --title "<paper title>" [--date YYYY-MM-DD]`
   On Windows without Python, use `scripts\new_paper.cmd --title "<paper title>" [--date YYYY-MM-DD]`
2. Put the PDF in `00_pdf/`
3. If needed, extract plain text or notes into `01_source_text/`
4. Produce or refine:
   - `02_notes/deep_notes.md`
   - `03_extraction/paper_brief.md`
   - `03_extraction/paper_schema.json`
   - `05_slides/slides_outline.md`
   - `06_compare/comparison_table.md`
5. Before finishing, run `python3 scripts/check_paper_outputs.py <paper_dir>`
   On Windows without Python, use `scripts\check_paper_outputs.cmd <paper_dir>`

## Evidence Discipline
- Every major claim must cite at least one section, figure, table, or equation pointer when possible.
- Record unsupported or unclear claims under `Unresolved Questions`.
- In `paper_schema.json`, set each field status to `done`, `uncertain`, or `pending`.
- Do not upgrade an uncertain claim to `done` without paper evidence.

## Required Paper Fields
Every paper extraction must cover these eight fields:

1. research question
2. high-level algorithmic idea
3. training label/dataset
4. network structure
5. network output
6. loss function
7. inference policy
8. novel contribution

## Output Discipline
- Keep headings stable.
- Preserve the schema keys in `paper_schema.json`.
- Keep one slide = one heading in `slides_outline.md`.
- Keep comparison output in a single markdown table unless the user asks for more.
- Prefer minimal rewrites when refining existing outputs.
- Optional editable PPT artifacts should stay under `05_slides/ppt/` and must not replace `05_slides/slides_outline.md`.
- In markdown files, write math in Preview-renderable LaTeX form: use `$...$` for inline math and `$$...$$` for display math instead of plain Unicode formula text.
- `07_guided_reading_cn/` is an optional teaching track for interactive Chinese guided reading. It must not replace or weaken the required standard outputs.

## Quality Gates
Before claiming completion:

1. Confirm the required files exist.
2. Confirm `paper_schema.json` is valid JSON.
3. Confirm all eight required fields are present.
4. Confirm unresolved points are explicit.
5. Confirm no unsupported claim is presented as certain.
6. Run the checker script and fix any reported issues.

## Done Criteria
- [ ] Standard paper workspace exists
- [ ] `03_extraction/paper_brief.md` is filled
- [ ] `03_extraction/paper_schema.json` is valid and complete
- [ ] `02_notes/deep_notes.md` records key technical uncertainties
- [ ] `05_slides/slides_outline.md` contains the 8-slide structure
- [ ] Every key conclusion has an evidence pointer or an `uncertain` marker
