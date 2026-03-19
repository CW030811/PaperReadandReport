#!/usr/bin/env python3
"""Validate the optional Chinese guided-reading track for one paper workspace."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_FILES = [
    "07_guided_reading_cn/reading_map.md",
    "07_guided_reading_cn/guided_reading_rules.md",
    "07_guided_reading_cn/progress_log.md",
]

READING_MAP_HEADINGS = [
    "## Paper",
    "## Reading Goal",
    "## Chapter Plan",
    "## Unlock Rule",
    "## Notes for Codex",
]

RULES_HEADINGS = [
    "## Purpose",
    "## Generation Policy",
    "## Revision Policy",
    "## Graduation Rule",
    "## Chapter Writing Contract",
]

PROGRESS_HEADINGS = [
    "## Status Snapshot",
    "## Chapter History",
    "## Recurring Questions",
    "## Mastery Signals",
]

CHAPTER_HEADINGS = [
    "## 1. 本章范围",
    "## 2. 阅读前你先抓住这三个问题",
    "## 3. 英文原文分段",
    "## 4. 重点概念与术语",
    "## 5. 本章核心内容",
    "## 6. 证据指针",
    "## 7. 一分钟回顾",
    "## 8. 你的问答区",
    "## 9. Codex 补充讲解",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_dir", help="Path to a paper workspace under papers/.")
    return parser.parse_args()


def require_file(base: Path, relative_path: str, errors: list[str]) -> Path:
    target = base / relative_path
    if not target.is_file():
        errors.append(f"Missing guided-reading file: {relative_path}")
    return target


def require_headings(path: Path, headings: list[str], errors: list[str]) -> None:
    if not path.is_file():
        return
    content = path.read_text(encoding="utf-8")
    for heading in headings:
        if heading not in content:
            errors.append(f"Missing heading in {path.name}: {heading}")


def check_active_status(reading_map: Path, errors: list[str]) -> None:
    if not reading_map.is_file():
        return

    active_like = 0
    for line in reading_map.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| Chapter"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        status = cells[-1]
        if status in {"active", "reviewing"}:
            active_like += 1

    if active_like > 1:
        errors.append("reading_map.md has more than one active/reviewing chapter")


def check_chapters(guided_dir: Path, errors: list[str]) -> None:
    for chapter_path in sorted(guided_dir.glob("Chapter*.md")):
        require_headings(chapter_path, CHAPTER_HEADINGS, errors)


def main() -> int:
    args = parse_args()
    paper_dir = Path(args.paper_dir).resolve()
    errors: list[str] = []

    required = {
        relative: require_file(paper_dir, relative, errors)
        for relative in REQUIRED_FILES
    }

    require_headings(required["07_guided_reading_cn/reading_map.md"], READING_MAP_HEADINGS, errors)
    require_headings(required["07_guided_reading_cn/guided_reading_rules.md"], RULES_HEADINGS, errors)
    require_headings(required["07_guided_reading_cn/progress_log.md"], PROGRESS_HEADINGS, errors)
    check_active_status(required["07_guided_reading_cn/reading_map.md"], errors)
    check_chapters(paper_dir / "07_guided_reading_cn", errors)

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    print("[OK] Guided-reading workspace passes the checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
