---
name: paper-scan
description: Create the first-pass structured extraction for one paper. Use when the task is to read a new paper for the first time, fill the repository's standard 8 fields, or produce the fixed outputs `03_extraction/paper_brief.md` and `03_extraction/paper_schema.json`.
---

# Paper Scan

Create a concise, evidence-backed first-pass extraction for the current paper workspace.

## Workflow

1. Confirm the current paper workspace path.
2. If the workspace does not exist yet, create it with `python3 scripts/new_paper.py --title "<paper title>"`.
   On Windows without Python, use `scripts\new_paper.cmd --title "<paper title>"`.
3. Read the PDF or extracted text from the current paper folder.
4. Fill or refine:
   - `03_extraction/paper_brief.md`
   - `03_extraction/paper_schema.json`
5. Cover all 8 required fields:
   - research question
   - high-level algorithmic idea
   - training label/dataset
   - network structure
   - network output
   - loss function
   - inference policy
   - novel contribution
6. Add section, figure, table, or equation pointers whenever possible.
7. If a field is unclear, keep the summary explicit and set the schema status to `uncertain`.
8. Run `python3 scripts/check_paper_outputs.py <paper_dir>` before finishing.
   On Windows without Python, use `scripts\check_paper_outputs.cmd <paper_dir>`.

## Output Rules

- Keep the markdown headings unchanged.
- Preserve the JSON schema keys.
- Do not invent unsupported details.
- Keep the writing concise and presentation-friendly.

## Completion

- Both extraction files exist.
- All 8 fields are covered.
- JSON stays valid.
- Open questions are listed explicitly.
