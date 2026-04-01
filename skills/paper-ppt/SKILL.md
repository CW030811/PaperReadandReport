---
name: paper-ppt
description: Create a presentation-ready research paper deck and editable `.pptx` workflow from one paper workspace. Use when the task is to build a 科研汇报PPT, PowerPoint deck, slide deck, PPT模板, `.pptx`, presentation polish, or final slides with equations, experiment figures, or editable charts based on `05_slides/slides_outline.md`.
---

# Paper PPT

Turn one paper workspace into an editable research presentation deck without weakening the repository's standard markdown outputs.

## Workflow

1. Confirm the current paper workspace path.
2. Keep `05_slides/slides_outline.md` as the content source of truth. If it is missing or weak, first use `skills/paper-to-slides`.
3. Keep the repository's fixed outputs unchanged. Store PPT source files and rendered outputs only under `05_slides/ppt/`.
4. Bootstrap the PPT workspace:
   - Use `scripts/bootstrap_ppt_workspace.cmd <paper_dir>` from this skill, or
   - Manually copy `assets/deck-starter/research_deck_template.js` to `05_slides/ppt/src/build_deck.js`.
5. Read `references/research-deck-style.md` before authoring. Follow its 8-slide mapping, typography, and figure rules.
6. For actual `.pptx` generation and layout validation, also follow the installed `slides` skill at `C:/Users/WIN/.codex/skills/slides/SKILL.md`.
7. Build the deck in JavaScript with PptxGenJS. Deliver both the editable `.js` source and the final `.pptx`.
8. Source visual material from:
   - `04_figures/`
   - paper figures extracted from the PDF
   - tables, equations, and notes already backed by the paper workspace
9. Keep claims evidence-backed. If a slide needs a claim or number that is not supported by the paper workspace, omit it or mark it as uncertain in speaker notes.
10. Before finishing:
   - confirm `05_slides/slides_outline.md` still exists
   - confirm PPT artifacts stay under `05_slides/ppt/`
   - render and inspect slide previews if the `slides` toolchain is available

## Output Rules

- Default to a simple 16:9 layout with a light background, dark text, and one accent color.
- Keep one slide = one claim.
- Prefer 3 to 5 bullets per slide.
- Use LaTeX-rendered equations when math is important.
- Use native charts for simple comparisons and raster images for paper figures.
- Keep dense details in speaker notes instead of the slide body.
- Preserve the repository's default 8-slide order unless the user explicitly asks for a different structure.

## Deliverables

- `05_slides/slides_outline.md`
- `05_slides/ppt/src/build_deck.js`
- `05_slides/ppt/output/<deck-name>.pptx`
- image assets required to rebuild the deck under `05_slides/ppt/assets/`

## Guardrails

- Do not replace or relocate the repository's required markdown outputs.
- Do not create extra top-level folders outside `05_slides/ppt/` for PPT work.
- If the deck cannot be rendered in the current environment, still leave a clean, editable JS source ready for later rendering.
