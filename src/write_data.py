#!/usr/bin/env python3
"""Materialize the 8 / 4 / 6 JSONL splits from dataset.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA, write_jsonl  # noqa: E402
from dataset import (  # noqa: E402
    CURVE_IDS,
    HOLDOUT,
    TEST,
    TRAIN,
    TRAIN_EXTRA,
    curve_rows,
    rows_for,
    to_sft_row,
)


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in ("train", "holdout", "test"):
        rows = rows_for(name)
        path = DATA / f"{name}.jsonl"
        write_jsonl(path, rows)
        counts[name] = len(rows)
        print(f"{name:8s}  {len(rows):2d} cards  → {path}")

    extra_path = DATA / "train_extra.jsonl"
    write_jsonl(extra_path, [to_sft_row(ex) for ex in TRAIN_EXTRA])
    print(f"{'extra':8s}  {len(TRAIN_EXTRA):2d} cards  → {extra_path}")

    curve_dir = DATA / "curve"
    curve_dir.mkdir(parents=True, exist_ok=True)
    curve_counts = {}
    for n in sorted(CURVE_IDS):
        path = curve_dir / f"train_{n}.jsonl"
        write_jsonl(path, curve_rows(n))
        curve_counts[n] = n
        print(f"curve/{n:<3}  {n:2d} cards  → {path}")

    manifest = {
        "model_target": "Qwen/Qwen2.5-0.5B-Instruct",
        "thesis": "least-data SFT: teach a new ticket-card schema",
        "counts": counts,
        "extra_ids": [ex["id"] for ex in TRAIN_EXTRA],
        "curve": {str(n): list(ids) for n, ids in CURVE_IDS.items()},
        "train_ids": [ex["id"] for ex in TRAIN],
        "holdout_ids": [ex["id"] for ex in HOLDOUT],
        "test_ids": [ex["id"] for ex in TEST],
        "rule": "split by ticket id; test is frozen; format lives in completions only",
        "stack_pin": "see requirements.txt",
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest → {DATA / 'manifest.json'}")
    print("train.jsonl is the 8-card lab. curve/ is the 2/4/8/16 experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
