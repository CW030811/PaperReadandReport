---
name: paper-deepdive
description: Deepen the technical reading of one paper after the first scan. Use when a paper already has structured extraction and the task is to inspect labels, outputs, loss, inference, heuristics, novelty, ambiguities, or underspecified technical details.
---

# Paper Deepdive

Write the second-pass technical notes for the current paper without loosening evidence standards.

## Workflow

1. Read the current paper workspace, especially:
   - `03_extraction/paper_brief.md`
   - `03_extraction/paper_schema.json`
   - paper PDF or extracted text if needed
2. Write or refine `02_notes/deep_notes.md`.
3. Focus on:
   - how labels are constructed
   - what the network predicts
   - whether the training target matches the inference target
   - whether the method depends on extra heuristics or search
   - what the real novelty is
   - what remains weak, ambiguous, or underspecified
4. Separate paper-stated facts from your interpretation.
5. Keep unresolved issues explicit.
6. Run `python3 scripts/check_paper_outputs.py <paper_dir>` before finishing.
   On Windows without Python, use `scripts\check_paper_outputs.cmd <paper_dir>`.

## Output Rules

- Preserve the template headings.
- Keep original paper terminology where possible.
- Do not smuggle interpretations in as paper facts.
- Prefer precise evidence pointers over generic summary.
