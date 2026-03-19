#!/usr/bin/env python3
"""Validate the standard outputs for one paper workspace."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_FILES = [
    "02_notes/deep_notes.md",
    "03_extraction/paper_brief.md",
    "03_extraction/paper_schema.json",
    "05_slides/slides_outline.md",
    "06_compare/comparison_table.md",
]

REQUIRED_SCHEMA_FIELDS = [
    "research_question",
    "high_level_algorithmic_idea",
    "training_label_dataset",
    "network_structure",
    "network_output",
    "loss_function",
    "inference_policy",
    "novel_contribution",
]

BRIEF_HEADINGS = [
    "## 1. Research Question",
    "## 2. High-Level Algorithmic Idea",
    "## 3. Training Label / Dataset",
    "## 4. Network Structure",
    "## 5. Network Output",
    "## 6. Loss Function",
    "## 7. Inference Policy",
    "## 8. Novel Contribution",
    "## Evidence Pointers",
    "## Unresolved Questions",
]

DEEP_HEADINGS = [
    "## Label Construction",
    "## Output Semantics",
    "## Loss and Optimization Target",
    "## Inference and Search",
    "## Real Novelty",
    "## Ablation Support",
    "## Weaknesses / Ambiguities",
    "## My Interpretation",
]

FIELD_HEADINGS = {
    "## 1. Research Question": "research_question",
    "## 2. High-Level Algorithmic Idea": "high_level_algorithmic_idea",
    "## 3. Training Label / Dataset": "training_label_dataset",
    "## 4. Network Structure": "network_structure",
    "## 5. Network Output": "network_output",
    "## 6. Loss Function": "loss_function",
    "## 7. Inference Policy": "inference_policy",
    "## 8. Novel Contribution": "novel_contribution",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_dir", help="Path to a paper workspace under papers/.")
    return parser.parse_args()


def require_file(base: Path, relative_path: str, errors: list[str]) -> Path:
    target = base / relative_path
    if not target.is_file():
        errors.append(f"Missing required file: {relative_path}")
    return target


def require_headings(path: Path, headings: list[str], errors: list[str]) -> None:
    if not path.is_file():
        return
    content = path.read_text(encoding="utf-8")
    for heading in headings:
        if heading not in content:
            errors.append(f"Missing heading in {path.name}: {heading}")


def extract_section(content: str, heading: str) -> str | None:
    pattern = rf"(?ms)^{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)"
    match = re.search(pattern, content)
    if not match:
        return None
    return match.group(1)


def extract_label_value(section: str, label: str) -> str | None:
    match = re.search(rf"(?m)^-\s*{re.escape(label)}\s*:\s*(.*)$", section)
    if not match:
        return None
    return match.group(1).strip()


def has_meaningful_bullet(section: str, *, disallowed: set[str]) -> bool:
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        if stripped in disallowed:
            continue
        value = stripped[2:].strip()
        if value:
            return True
    return False


def check_brief_content(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        return

    content = path.read_text(encoding="utf-8")
    metadata = extract_section(content, "## Paper Metadata")
    if metadata is None:
        errors.append("Missing section in paper_brief.md: ## Paper Metadata")
        return

    for label in ["Title", "Authors", "Venue / Year", "Paper Folder"]:
        value = extract_label_value(metadata, label)
        if not value:
            errors.append(f"paper_brief.md is missing filled metadata: {label}")

    for heading in FIELD_HEADINGS:
        section = extract_section(content, heading)
        if section is None:
            continue
        summary = extract_label_value(section, "Summary")
        evidence = extract_label_value(section, "Evidence")
        if not summary:
            errors.append(f"paper_brief.md has empty summary under {heading}")
        if not evidence:
            errors.append(f"paper_brief.md has empty evidence under {heading}")

    evidence_section = extract_section(content, "## Evidence Pointers")
    if evidence_section is None:
        errors.append("Missing section in paper_brief.md: ## Evidence Pointers")
    elif not has_meaningful_bullet(
        evidence_section,
        disallowed={"- Section:", "- Figure:", "- Table:", "- Equation:"},
    ):
        errors.append("paper_brief.md must contain at least one filled evidence pointer")

    unresolved_section = extract_section(content, "## Unresolved Questions")
    if unresolved_section is None:
        errors.append("Missing section in paper_brief.md: ## Unresolved Questions")
    elif not has_meaningful_bullet(unresolved_section, disallowed={"- None yet."}):
        errors.append(
            "paper_brief.md must list an unresolved question or explicitly justify why none remain"
        )


def check_deep_notes_content(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        return

    content = path.read_text(encoding="utf-8")
    technical_sections = DEEP_HEADINGS[:-1]

    for heading in technical_sections:
        section = extract_section(content, heading)
        if section is None:
            continue
        for label in ["Paper-stated facts", "Evidence", "Open questions"]:
            value = extract_label_value(section, label)
            if not value:
                errors.append(f"deep_notes.md has empty '{label}' under {heading}")

    interpretation = extract_section(content, "## My Interpretation")
    if interpretation is None:
        return
    for label in ["Inference", "Confidence", "What to verify next"]:
        value = extract_label_value(interpretation, label)
        if not value:
            errors.append(f"deep_notes.md has empty '{label}' under ## My Interpretation")


def check_schema(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in {path.name}: {exc}")
        return

    for top_key in ["paper_title", "paper_folder", "scan_date", "global_evidence_pointers", "unresolved_questions"]:
        if top_key not in data:
            errors.append(f"Missing schema key: {top_key}")

    done_fields = 0
    incomplete_fields = 0
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in data:
            errors.append(f"Missing schema field: {field}")
            continue

        payload = data[field]
        if not isinstance(payload, dict):
            errors.append(f"Schema field must be an object: {field}")
            continue

        for nested_key in ["summary", "evidence", "status"]:
            if nested_key not in payload:
                errors.append(f"Missing nested key for {field}: {nested_key}")

        status = payload.get("status")
        if isinstance(status, str) and status not in {"done", "uncertain", "pending"}:
            errors.append(f"Invalid status for {field}: {payload['status']}")

        summary = payload.get("summary")
        if isinstance(status, str) and status in {"done", "uncertain"}:
            if not isinstance(summary, str) or not summary.strip():
                errors.append(f"Schema field must have a non-empty summary when status is {status}: {field}")

        evidence = payload.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{field}.evidence must be a list")
            evidence = []
        elif isinstance(status, str) and status == "done":
            if not evidence or not all(isinstance(item, str) and item.strip() for item in evidence):
                errors.append(f"Schema field marked done must include evidence pointers: {field}")

        if status == "done":
            done_fields += 1
        elif status in {"uncertain", "pending"}:
            incomplete_fields += 1

    if not isinstance(data.get("global_evidence_pointers", []), list):
        errors.append("global_evidence_pointers must be a list")
    elif done_fields and not data.get("global_evidence_pointers"):
        errors.append("global_evidence_pointers must not be empty when done fields are present")
    if not isinstance(data.get("unresolved_questions", []), list):
        errors.append("unresolved_questions must be a list")
    elif incomplete_fields and not data.get("unresolved_questions"):
        errors.append("unresolved_questions must not be empty when uncertain/pending fields remain")


def check_slide_count(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        return
    content = path.read_text(encoding="utf-8")
    slide_count = sum(1 for line in content.splitlines() if line.startswith("## Slide "))
    if slide_count < 8:
        errors.append(f"slides_outline.md must contain at least 8 slides, found {slide_count}")


def check_comparison_table(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        return
    content = path.read_text(encoding="utf-8")
    if "| Paper | Task | Label Type | Dataset | Encoder | Decoder | Output | Loss | Inference | Novelty | Weakness | Relevance |" not in content:
        errors.append("comparison_table.md is missing the standard header row")
    data_rows = [
        line for line in content.splitlines()
        if line.startswith("|")
        and not line.startswith("|---")
        and "| Paper | Task | Label Type | Dataset | Encoder | Decoder | Output | Loss | Inference | Novelty | Weakness | Relevance |" not in line
    ]
    if not data_rows:
        errors.append("comparison_table.md must contain at least one paper row")


def main() -> int:
    args = parse_args()
    paper_dir = Path(args.paper_dir).resolve()
    errors: list[str] = []

    required_paths = {
        relative_path: require_file(paper_dir, relative_path, errors)
        for relative_path in REQUIRED_FILES
    }

    require_headings(required_paths["03_extraction/paper_brief.md"], BRIEF_HEADINGS, errors)
    require_headings(required_paths["02_notes/deep_notes.md"], DEEP_HEADINGS, errors)
    check_brief_content(required_paths["03_extraction/paper_brief.md"], errors)
    check_deep_notes_content(required_paths["02_notes/deep_notes.md"], errors)
    check_schema(required_paths["03_extraction/paper_schema.json"], errors)
    check_slide_count(required_paths["05_slides/slides_outline.md"], errors)
    check_comparison_table(required_paths["06_compare/comparison_table.md"], errors)

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    print("[OK] Paper workspace passes the standard checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
