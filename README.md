# Paper Reading Workflow

This repository turns each paper into fixed, reusable outputs for extraction, deep reading, slide prep, and cross-paper comparison.

## Repository Layout

```text
.
├── AGENTS.md
├── README.md
├── .codex/config.toml
├── scripts/
│   ├── new_paper.py / new_paper.ps1 / new_paper.cmd
│   ├── check_paper_outputs.py / check_paper_outputs.ps1 / check_paper_outputs.cmd
│   ├── init_guided_reading.py / init_guided_reading.ps1 / init_guided_reading.cmd
│   └── check_guided_reading.py / check_guided_reading.ps1 / check_guided_reading.cmd
├── skills/
│   ├── paper-scan/
│   ├── paper-deepdive/
│   ├── paper-to-slides/
│   ├── paper-compare/
│   └── paper-guided-read-cn/
├── templates/
│   ├── paper_brief.md
│   ├── deep_notes.md
│   ├── slides_outline.md
│   ├── comparison_table.md
│   └── paper_schema.json
└── papers/
```

## Quick Start

1. Create a new paper workspace:

   ```bash
   python3 scripts/new_paper.py --title "Example Paper Title"
   ```

   On Windows without Python, use:

   ```powershell
   scripts\new_paper.cmd --title "Example Paper Title"
   ```

2. Put the PDF into the new folder's `00_pdf/`.
3. Open the repo in Codex CLI or Cursor.
4. Use one stage at a time with the local skills in `skills/`.
5. Run the checker before you finish:

   ```bash
   python3 scripts/check_paper_outputs.py papers/YYYY-MM-DD_short-title
   ```

   On Windows without Python, use:

   ```powershell
   scripts\check_paper_outputs.cmd papers\YYYY-MM-DD_short-title
   ```

## Multi-Device Sync

If you work on this repo from both a Mac and a Windows machine, use GitHub as the single shared remote and keep both devices on the same branch.

### One-time setup

1. Create an empty GitHub repository.
2. In this repo, add the remote and push the first branch:

   ```bash
   git remote add origin <your-github-repo-url>
   git add .
   git commit -m "chore: initial repo setup"
   git push -u origin main
   ```

3. On the second device, clone the same GitHub repository instead of copying files manually.

### Daily workflow

Before you start work on a device, pull the latest changes:

- Windows:

  ```powershell
  scripts\update_repo.cmd
  ```

- Mac:

  ```bash
  bash scripts/update_repo.sh
  ```

After you finish a work session, commit and push everything back to GitHub:

- Windows:

  ```powershell
  scripts\sync_repo.cmd -Message "notes: update current paper"
  ```

- Mac:

  ```bash
  bash scripts/sync_repo.sh origin "notes: update current paper"
  ```

These helper scripts stop if the remote is missing, if the repo is in detached HEAD state, or if a pull would overwrite uncommitted work.

### Switching devices checklist

When moving from Windows to Mac or from Mac to Windows:

1. On the device you are leaving, run the sync script and wait for push to finish.
2. On the device you are starting, run the update script before editing anything.
3. Keep both devices on `main` unless you intentionally start using branches.
4. Avoid editing the same file on both devices before syncing, or Git will require a manual conflict resolution.

## Fixed Trigger Prompts

Use these as the default prompts in Codex CLI or the IDE extension.

### 1. First-pass extraction

```text
Use the local skill in skills/paper-scan.

Read the current paper workspace and create or refine:
- 03_extraction/paper_brief.md
- 03_extraction/paper_schema.json

Cover all 8 required fields, include evidence pointers when possible, and mark uncertain items explicitly.
Run the checker before finishing.
```

### 2. Technical deep dive

```text
Use the local skill in skills/paper-deepdive.

Read the current paper outputs and write 02_notes/deep_notes.md.
Focus on labels, outputs, loss, inference, heuristics, novelty, and ambiguity.
Keep paper-stated facts separate from interpretation.
Run the checker before finishing.
```

### 3. Slide outline

```text
Use the local skill in skills/paper-to-slides.

Read the current paper outputs and generate 05_slides/slides_outline.md.
Keep the default 8-slide structure.
Each slide must include a title, 3 to 5 bullets, a suggested figure, and short speaker notes.
Run the checker before finishing.
```

### 4. Cross-paper comparison

```text
Use the local skill in skills/paper-compare.

Compare the current paper with prior papers in papers/.
Prefer existing extraction outputs over re-reading PDFs.
Write 06_compare/comparison_table.md and keep the standard comparison dimensions.
Run the checker before finishing.
```

## Acceptance Standard

- Standard paper directory exists
- The fixed markdown files exist in their expected locations
- `paper_schema.json` is valid and preserves the stable keys
- The eight required paper fields are covered
- Key claims carry evidence pointers or an `uncertain` marker
- `done` schema fields include evidence pointers
- unresolved points are explicit whenever uncertainty remains
- The checker script exits successfully
