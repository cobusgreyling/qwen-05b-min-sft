#!/usr/bin/env python3
"""
LoRA SFT on the 8-card train split. Hold-out is eval-only. Test is never loaded.

Qwen2.5-0.5B-Instruct fits a free Colab T4 in fp16 without 4-bit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA, MODEL_ID, OUTPUTS, load_jsonl  # noqa: E402
from dataset import CURVE_IDS, curve_rows  # noqa: E402

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Least-data LoRA SFT on DeskCard.")
    p.add_argument("--train", type=Path, default=DATA / "train.jsonl")
    p.add_argument("--holdout", type=Path, default=DATA / "holdout.jsonl")
    p.add_argument("--model", default=os.environ.get("SFT_MODEL", MODEL_ID))
    p.add_argument("--output-dir", type=Path, default=OUTPUTS / "lora-sft")
    p.add_argument(
        "--curve",
        type=int,
        choices=sorted(CURVE_IDS),
        default=None,
        help="Train the n-card curve subset instead of --train.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override step count. Default is --epochs × n / (batch × accum).",
    )
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_sft_jsonl(path: Path):
    from datasets import Dataset

    rows = [{"prompt": r["prompt"], "completion": r["completion"]} for r in load_jsonl(path)]
    if not rows:
        raise ValueError(f"no examples in {path}")
    return Dataset.from_list(rows)


def main() -> int:
    args = parse_args()
    if args.curve is not None:
        from datasets import Dataset

        rows = [{"prompt": r["prompt"], "completion": r["completion"]} for r in curve_rows(args.curve)]
        train_ds = Dataset.from_list(rows)
        if args.output_dir == OUTPUTS / "lora-sft":
            args.output_dir = OUTPUTS / f"lora-sft-{args.curve}"
    else:
        if not args.train.exists():
            print(f"ERROR: {args.train} missing. Run: python src/write_data.py", file=sys.stderr)
            return 1
        train_ds = load_sft_jsonl(args.train)

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    eval_ds = load_sft_jsonl(args.holdout) if args.holdout.exists() else None
    effective_batch = max(args.batch_size * args.grad_accum, 1)
    if args.max_steps is None:
        args.max_steps = max(1, math.ceil(args.epochs * len(train_ds) / effective_batch))

    use_cuda = torch.cuda.is_available()
    use_mps = (
        (not use_cuda)
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    device = "cuda" if use_cuda else "mps" if use_mps else "cpu"

    print("=" * 60)
    print("Least-data LoRA SFT")
    print("=" * 60)
    print(f"model       : {args.model}")
    print(f"train cards : {len(train_ds)}")
    print(f"holdout     : {len(eval_ds) if eval_ds is not None else 0}")
    print("test        : 0  (frozen — not loaded)")
    print(f"max_steps   : {args.max_steps}")
    print(f"lora_r      : {args.lora_r}")
    print(f"device      : {device}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict = {"trust_remote_code": True}
    if use_cuda:
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    elif use_mps:
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if use_mps:
        model = model.to("mps")
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )

    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, args.max_steps // 10),
        logging_steps=1,
        save_steps=max(args.max_steps, 1),
        save_total_limit=1,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=max(args.max_steps // 2, 1) if eval_ds is not None else None,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=bool(use_cuda),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training ...")
    result = trainer.train()
    metrics = result.metrics
    print("Train metrics:", json.dumps(metrics, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    card = {
        "base_model": args.model,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "curve": args.curve,
        "max_length": args.max_length,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "completion_only_loss": True,
        "train_samples": len(train_ds),
        "holdout_samples": len(eval_ds) if eval_ds is not None else 0,
        "test_samples": 0,
        "stack": {
            "transformers": __import__("transformers").__version__,
            "trl": __import__("trl").__version__,
            "peft": __import__("peft").__version__,
        },
        "metrics": metrics,
    }
    (args.output_dir / "run_card.json").write_text(json.dumps(card, indent=2))
    print(f"Saved LoRA adapter → {args.output_dir}")
    print("Next: python src/evaluate.py --adapter", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
