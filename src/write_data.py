#!/usr/bin/env python3
"""Materialize the 8 / 4 / 6 JSONL splits from dataset.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA, write_jsonl  # noqa: E402
from dataset import HOLDOUT, TEST, TRAIN, rows_for  # noqa: E402


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in ("train", "holdout", "test"):
        rows = rows_for(name)
        path = DATA / f"{name}.jsonl"
        write_jsonl(path, rows)
        counts[name] = len(rows)
        print(f"{name:8s}  {len(rows):2d} cards  → {path}")

    manifest = {
        "model_target": "Qwen/Qwen2.5-0.5B-Instruct",
        "thesis": "least-data SFT: teach a new ticket-card schema",
        "counts": counts,
        "train_ids": [ex["id"] for ex in TRAIN],
        "holdout_ids": [ex["id"] for ex in HOLDOUT],
        "test_ids": [ex["id"] for ex in TEST],
        "rule": "split by ticket id; test is frozen; format lives in completions only",
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest → {DATA / 'manifest.json'}")
    print("train examples are the only rows that update weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
