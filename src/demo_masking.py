#!/usr/bin/env python3
"""
Show completion-only masking on a real DeskCard example.

The model reads the whole chat. Cross-entropy only hits assistant tokens.
TRL spelling: SFTConfig(completion_only_loss=True) → prompt labels = -100.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import MODEL_ID  # noqa: E402
from dataset import TRAIN, card_text  # noqa: E402
from dataset import SYSTEM  # noqa: E402


def show_masking(model_id: str) -> None:
    from transformers import AutoTokenizer

    example = TRAIN[0]
    prompt = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": example["user"]},
    ]
    completion = [{"role": "assistant", "content": card_text(example)}]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    full_text = tokenizer.apply_chat_template(
        prompt + completion, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    n_prompt = len(prompt_ids)
    prefix_ok = full_ids[:n_prompt] == prompt_ids
    if not prefix_ok:
        print("NOTE: prompt tokens are not an exact prefix; demo is illustrative.")
        n_prompt = min(n_prompt, len(full_ids) // 2)
    labels = [-100] * n_prompt + full_ids[n_prompt:]

    print("=" * 62)
    print("Completion-only masking — Qwen2.5 chat template")
    print("=" * 62)
    print(f"model              : {model_id}")
    print(f"prompt prefix match: {prefix_ok}")
    print(f"total tokens       : {len(full_ids)}")
    print(f"masked (prompt)    : {n_prompt}")
    print(f"supervised (card)  : {len(full_ids) - n_prompt}")
    print()
    print("--- prompt (MASK, labels = -100) ---")
    print(prompt_text)
    print("--- completion (LOSS) ---")
    print(completion[0]["content"])
    print()
    print(f"{'i':>5}  {'tag':<4}  token")
    print("-" * 50)
    start = max(0, n_prompt - 4)
    end = min(len(full_ids), n_prompt + 16)
    if start > 0:
        print("  ...  (earlier prompt tokens omitted)")
    for i in range(start, end):
        piece = tokenizer.decode([full_ids[i]]).replace("\n", "\\n")
        tag = "MASK" if labels[i] == -100 else "LOSS"
        print(f"{i:5d}  {tag:<4}  {piece!r}")
    if end < len(full_ids):
        print("  ...  (rest of card is LOSS)")
    print()
    print("The model still *reads* the ticket. It is only *graded* on the card.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()
    show_masking(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
