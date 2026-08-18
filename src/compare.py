#!/usr/bin/env python3
"""Print the frozen-test comparison table from saved eval JSON files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUTPUTS  # noqa: E402
from evaluate import SCORE_KEYS  # noqa: E402

# Display order. Missing files are skipped, not fatal.
COLUMNS: list[tuple[str, str]] = [
    ("base_eval.json", "stock thin"),
    ("schema_eval.json", "stock+schema"),
    ("icl_eval.json", "8-shot ICL"),
    ("curve/lora-2.json", "LoRA-2"),
    ("curve/lora-4.json", "LoRA-4"),
    ("adapter_eval.json", "LoRA-8"),
    ("curve/lora-16.json", "LoRA-16"),
]

HEADLINE = ("has_ticket_tags", "format_pass", "task_pass", "note_facts_pass")


def load_reports() -> list[tuple[str, dict]]:
    found: list[tuple[str, dict]] = []
    for name, label in COLUMNS:
        path = OUTPUTS / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        found.append((label, data))
    return found


def fmt_pct(value: float | None, *, n: int | None = None) -> str:
    if value is None:
        return "—"
    if n is not None:
        hits = int(round(value * n))
        return f"{value:5.0%} ({hits}/{n})"
    return f"{value:5.0%}"


def markdown_table(reports: list[tuple[str, dict]], keys: tuple[str, ...] | list[str]) -> str:
    labels = [label for label, _ in reports]
    header = "| check | " + " | ".join(labels) + " |"
    sep = "|-------|" + "|".join(["------:"] * len(labels)) + "|"
    lines = [header, sep]
    for key in keys:
        cells = []
        for _, data in reports:
            n = int(data.get("n") or 0) or None
            val = (data.get("summary") or {}).get(key)
            cells.append(fmt_pct(val, n=n if key in HEADLINE else None))
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def text_table(reports: list[tuple[str, dict]], keys: tuple[str, ...] | list[str]) -> str:
    labels = [label for label, _ in reports]
    col_w = max(8, max(len(l) for l in labels))
    head = f"{'check':<18}" + "".join(f"{l:>{col_w + 1}}" for l in labels)
    lines = [head, "-" * len(head)]
    for key in keys:
        row = f"{key:<18}"
        for _, data in reports:
            n = int(data.get("n") or 0) or None
            val = (data.get("summary") or {}).get(key)
            row += f"{fmt_pct(val, n=n if key in HEADLINE else None):>{col_w + 1}}"
        lines.append(row)
    return "\n".join(lines)


def per_ticket_markdown(reports: list[tuple[str, dict]]) -> str:
    tickets: list[str] = []
    for _, data in reports:
        for row in data.get("per_prompt") or []:
            tid = row.get("trace_id")
            if tid and tid not in tickets:
                tickets.append(tid)
    if not tickets:
        return ""
    labels = [label for label, _ in reports]
    header = "| ticket | " + " | ".join(labels) + " |"
    sep = "|--------|" + "|".join(["------"] * len(labels)) + "|"
    lines = [header, sep]
    by_label = []
    for _, data in reports:
        by_id = {r.get("trace_id"): r for r in data.get("per_prompt") or []}
        by_label.append(by_id)
    for tid in tickets:
        cells = []
        for lookup in by_label:
            row = lookup.get(tid) or {}
            if row.get("task_pass"):
                mark = "PASS"
            elif row.get("has_ticket_tags"):
                mark = "wrap"
            elif row.get("generation"):
                mark = "FAIL"
            else:
                mark = "—"
            cells.append(mark)
        lines.append(f"| `{tid}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results_md(reports: list[tuple[str, dict]], dest: Path) -> None:
    ns = {data.get("n") for _, data in reports}
    n = next(iter(ns)) if len(ns) == 1 else "mixed"
    body = [
        "# Frozen-test results",
        "",
        f"Same **{n}** tickets. Greedy decode. Same scorer.",
        "",
        "- **stock thin** — base 0.5B, original one-line system prompt",
        "- **stock+schema** — base 0.5B, card layout spelled out in the prompt (no SFT)",
        "- **8-shot ICL** — base 0.5B, the 8 train cards in context (no SFT)",
        "- **LoRA-k** — completion-only LoRA trained on *k* cards, thin prompt. LoRA-8 is the published adapter.",
        "",
        markdown_table(reports, HEADLINE),
        "",
        "<details><summary>All checks</summary>",
        "",
        markdown_table(reports, SCORE_KEYS),
        "",
        "</details>",
        "",
        "## Per ticket (`task_pass` / wrapper only / fail)",
        "",
        per_ticket_markdown(reports),
        "",
        "`wrap` means `<<TICKET>>` tags landed but the decision or enum was wrong.",
        "",
        "Regenerate: `./run.sh compare`",
        "",
    ]
    dest.write_text("\n".join(body), encoding="utf-8")


def main() -> int:
    reports = load_reports()
    if not reports:
        print("No eval JSON found under outputs/. Run ./run.sh eval-base (and friends).", file=sys.stderr)
        return 1
    print(text_table(reports, SCORE_KEYS))
    print()
    dest = OUTPUTS / "RESULTS.md"
    write_results_md(reports, dest)
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
