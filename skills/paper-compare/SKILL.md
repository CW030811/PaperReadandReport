---
name: paper-compare
description: Compare the current paper with prior papers already stored in this repository. Use when the task is cross-paper comparison, prior-work contrast, or filling `06_compare/comparison_table.md` from existing repository outputs.
---

# Paper Compare

Create a reusable comparison table without redoing work that already exists in the repository.

## Workflow

1. Read the current paper workspace.
2. Scan other folders under `papers/`.
3. Prefer existing extraction outputs over re-reading PDFs.
4. Write or refine `06_compare/comparison_table.md`.
5. Compare with these dimensions:
   - task
   - label type
   - dataset
   - encoder
   - decoder
   - network output
   - loss
   - inference policy
   - novelty
   - weakness
   - relevance to my research
6. Keep the result concise and tabular.
7. Run `python3 scripts/check_paper_outputs.py <paper_dir>` before finishing.
   On Windows without Python, use `scripts\check_paper_outputs.cmd <paper_dir>`.

## Output Rules

- Preserve the standard header row.
- Prefer repository facts over fresh speculation.
- If a comparison field is unclear, say `uncertain` rather than guessing.
