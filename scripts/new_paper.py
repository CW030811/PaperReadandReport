#!/usr/bin/env python3
"""Create a standard workspace for a new paper."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from datetime import date
from pathlib import Path


SUBDIRS = [
    "00_pdf",
    "01_source_text",
    "02_notes",
    "03_extraction",
    "04_figures",
    "05_slides",
    "06_compare",
    "07_guided_reading_cn",
]

FILE_MAP = {
    "02_notes/deep_notes.md": "deep_notes.md",
    "03_extraction/paper_brief.md": "paper_brief.md",
    "03_extraction/paper_schema.json": "paper_schema.json",
    "05_slides/slides_outline.md": "slides_outline.md",
    "06_compare/comparison_table.md": "comparison_table.md",
    "07_guided_reading_cn/reading_map.md": "guided_reading_map.md",
    "07_guided_reading_cn/guided_reading_rules.md": "guided_reading_rules.md",
    "07_guided_reading_cn/progress_log.md": "guided_reading_progress_log.md",
}


def slugify(value: str) -> str:
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "untitled-paper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True, help="Paper title used for the folder slug.")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Workspace date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--root",
        default="papers",
        help="Root folder that stores paper workspaces. Defaults to papers.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing workspace without failing.",
    )
    return parser.parse_args()


def load_template(template_dir: Path, template_name: str) -> str:
    return template_dir.joinpath(template_name).read_text(encoding="utf-8")


def render_template(text: str, *, title: str, paper_folder: str, scan_date: str) -> str:
    replacements = {
        "{{PAPER_TITLE}}": title,
        "{{PAPER_FOLDER}}": paper_folder,
        "{{SCAN_DATE}}": scan_date,
    }
    rendered = text
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    template_dir = repo_root / "templates"
    workspace_name = f"{args.date}_{slugify(args.title)}"
    workspace_dir = repo_root / args.root / workspace_name

    if workspace_dir.exists() and not args.force:
        raise SystemExit(
            f"Workspace already exists: {workspace_dir}\n"
            "Use --force to reuse it."
        )

    workspace_dir.mkdir(parents=True, exist_ok=True)
    for subdir in SUBDIRS:
        current_dir = workspace_dir / subdir
        current_dir.mkdir(exist_ok=True)
        if subdir in {"00_pdf", "01_source_text", "04_figures"}:
            current_dir.joinpath(".gitkeep").touch()

    for relative_path, template_name in FILE_MAP.items():
        target = workspace_dir / relative_path
        if template_name == "paper_schema.json":
            template_data = json.loads(load_template(template_dir, template_name))
            schema = deepcopy(template_data)
            schema["paper_title"] = args.title
            schema["paper_folder"] = str(Path(args.root) / workspace_name)
            schema["scan_date"] = args.date
            target.write_text(
                json.dumps(schema, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            target.write_text(
                render_template(
                    load_template(template_dir, template_name),
                    title=args.title,
                    paper_folder=str(Path(args.root) / workspace_name),
                    scan_date=args.date,
                ),
                encoding="utf-8",
            )

    print(workspace_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
