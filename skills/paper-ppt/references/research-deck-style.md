# Research Deck Style

Use this reference after the skill triggers and before writing the deck source.

## Visual Defaults

- Slide size: `LAYOUT_WIDE` (16:9)
- Background: off-white `#F8FAFC`
- Primary text: slate `#0F172A`
- Accent: teal `#0F766E`
- Secondary line or caption color: gray `#475569`
- Title size: 24 to 28 pt
- Body size: 16 to 18 pt
- Caption size: 10 to 12 pt
- Preferred tone: minimal, technical, high contrast, no decorative gradients

## Layout Rules

- Keep a strong title band at the top with a short takeaway title.
- Limit each slide to one visual center: either one large figure, one table, or one two-column comparison.
- Reserve the right third or bottom strip for figures when the slide includes qualitative results.
- Leave visible margins. Avoid edge-to-edge content unless the slide is figure-first.
- When showing equations, render the main equation once and keep the derivation in speaker notes.

## Standard 8-Slide Mapping

1. Slide 1: paper title, venue/year, authors, task setting, and one-sentence takeaway.
2. Slide 2: research question, practical pain point, and why existing methods are insufficient.
3. Slide 3: high-level idea with one pipeline diagram or method sketch.
4. Slide 4: dataset, supervision source, label construction, and any preprocessing pipeline.
5. Slide 5: network structure with one architecture figure and 3 to 4 component bullets.
6. Slide 6: network output, loss function, and inference policy on one slide. Use a left-text right-equation or left-equation right-results layout.
7. Slide 7: novel contribution and main experimental evidence. Use one results table or one key ablation.
8. Slide 8: strengths, weaknesses, failure modes, and your takeaway for future work.

## Figure Rules

- Prefer the paper's original figures when they are legible and central to the claim.
- Crop tightly around the relevant panel instead of shrinking an entire multi-panel figure.
- When using result tables, highlight only the row or column that supports the spoken claim.
- Put experiment setup screenshots or qualitative examples in `05_slides/ppt/assets/` after extraction so the deck can be rebuilt.

## Equation Rules

- Use LaTeX source when available instead of typing Unicode math.
- Show at most one main optimization target per slide.
- If the paper has multiple losses, summarize them as a decomposition and move exact weights to speaker notes.

## Speaker Notes Rules

- Keep notes factual and short.
- Include unresolved points as `uncertain` instead of turning them into confident claims.
- Include evidence pointers such as section, figure, table, or equation labels when possible.
