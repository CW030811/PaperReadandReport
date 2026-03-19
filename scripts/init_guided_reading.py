#!/usr/bin/env python3
"""Initialize the optional Chinese guided-reading track for one paper workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FILE_MAP = {
    "reading_map.md": "guided_reading_map.md",
    "guided_reading_rules.md": "guided_reading_rules.md",
    "progress_log.md": "guided_reading_progress_log.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_dir", help="Path to a paper workspace under papers/.")
    parser.add_argument(
        "--title",
        help="Paper title override. Defaults to the title found in paper_schema.json when available.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing guided-reading files.",
    )
    parser.add_argument(
        "--chapter-template",
        action="store_true",
        help="Also create an empty Chapter01.md from the template.",
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


def infer_title(paper_dir: Path, override: str | None) -> str:
    if override:
        return override

    schema_path = paper_dir / "03_extraction/paper_schema.json"
    if schema_path.is_file():
        try:
            data = json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        title = data.get("paper_title")
        if isinstance(title, str) and title.strip():
            return title.strip()

    brief_path = paper_dir / "03_extraction/paper_brief.md"
    if brief_path.is_file():
        for line in brief_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("- Title:"):
                title = line.split(":", 1)[1].strip()
                if title:
                    return title

    return paper_dir.name


def write_if_needed(target: Path, content: str, *, force: bool) -> bool:
    if target.exists() and not force:
        return False
    target.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    template_dir = repo_root / "templates"
    paper_dir = Path(args.paper_dir).resolve()

    if not paper_dir.is_dir():
        raise SystemExit(f"Paper workspace not found: {paper_dir}")

    try:
        paper_folder = str(paper_dir.relative_to(repo_root))
    except ValueError:
        paper_folder = str(paper_dir)

    title = infer_title(paper_dir, args.title)
    scan_date = paper_dir.name.split("_", 1)[0] if "_" in paper_dir.name else ""
    guided_dir = paper_dir / "07_guided_reading_cn"
    guided_dir.mkdir(exist_ok=True)

    created: list[str] = []
    for relative_name, template_name in FILE_MAP.items():
        target = guided_dir / relative_name
        content = render_template(
            load_template(template_dir, template_name),
            title=title,
            paper_folder=paper_folder,
            scan_date=scan_date,
        )
        if write_if_needed(target, content, force=args.force):
            created.append(str(target.relative_to(paper_dir)))

    if args.chapter_template:
        chapter_target = guided_dir / "Chapter01.md"
        chapter_content = render_template(
            load_template(template_dir, "guided_reading_chapter.md"),
            title=title,
            paper_folder=paper_folder,
            scan_date=scan_date,
        )
        if write_if_needed(chapter_target, chapter_content, force=args.force):
            created.append(str(chapter_target.relative_to(paper_dir)))

    if created:
        for item in created:
            print(f"[CREATED] {item}")
    else:
        print("[SKIP] Guided-reading files already exist. Use --force to overwrite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
