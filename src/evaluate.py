#!/usr/bin/env python3
"""
Score generations against the ticket-card contract.

format_pass  — looks like a DeskCard (tags, parseable fields, allowed enums)
task_pass    — format plus id / verdict / amount / reason match gold

This is the "test afterwards" step. Hold-out loss is not this file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DATA, MODEL_ID, OUTPUTS, load_jsonl  # noqa: E402
from dataset import REASONS, VERDICTS  # noqa: E402

ROLE_LEAK_RE = re.compile(
    r"(<\|im_start\|>|<\|im_end\|>|^\s*user\s*:|^\s*system\s*:)",
    re.IGNORECASE | re.MULTILINE,
)

CARD_RE = re.compile(
    r"<<TICKET>>\s*"
    r"id:\s*(?P<id>\S+)\s*"
    r"verdict:\s*(?P<verdict>\S+)\s*"
    r"amount_cents:\s*(?P<amount>-?\d+)\s*"
    r"reason_code:\s*(?P<reason>\S+)\s*"
    r"note:\s*(?P<note>.+?)\s*"
    r"<</TICKET>>",
    re.DOTALL | re.IGNORECASE,
)


def parse_card(text: str) -> dict[str, Any] | None:
    m = CARD_RE.search(text or "")
    if not m:
        return None
    return {
        "id": m.group("id").strip(),
        "verdict": m.group("verdict").strip().upper(),
        "amount_cents": int(m.group("amount")),
        "reason_code": m.group("reason").strip().upper(),
        "note": m.group("note").strip(),
    }


def score_generation(text: str, gold: dict[str, Any]) -> dict[str, Any]:
    text = (text or "").strip()
    parsed = parse_card(text)
    checks: dict[str, Any] = {
        "non_empty": len(text) > 0,
        "no_role_leak": not bool(ROLE_LEAK_RE.search(text)),
        "has_ticket_tags": "<<TICKET>>" in text and "<</TICKET>>" in text,
        "parsed_ok": parsed is not None,
        "id_match": False,
        "verdict_match": False,
        "amount_match": False,
        "reason_match": False,
        "verdict_allowed": False,
        "reason_allowed": False,
        "predicted": parsed,
    }
    if parsed is not None:
        checks["verdict_allowed"] = parsed["verdict"] in VERDICTS
        checks["reason_allowed"] = parsed["reason_code"] in REASONS
        checks["id_match"] = parsed["id"] == gold.get("id")
        checks["verdict_match"] = parsed["verdict"] == gold.get("verdict")
        checks["amount_match"] = parsed["amount_cents"] == gold.get("amount_cents")
        checks["reason_match"] = parsed["reason_code"] == gold.get("reason_code")

    checks["format_pass"] = (
        checks["non_empty"]
        and checks["no_role_leak"]
        and checks["has_ticket_tags"]
        and checks["parsed_ok"]
        and checks["verdict_allowed"]
        and checks["reason_allowed"]
    )
    checks["task_pass"] = (
        checks["format_pass"]
        and checks["id_match"]
        and checks["verdict_match"]
        and checks["amount_match"]
        and checks["reason_match"]
    )
    preview = text.replace("\n", " | ")
    checks["preview"] = preview[:180]
    return checks


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "non_empty",
        "no_role_leak",
        "has_ticket_tags",
        "parsed_ok",
        "id_match",
        "verdict_match",
        "amount_match",
        "reason_match",
        "format_pass",
        "task_pass",
    ]
    n = max(len(rows), 1)
    return {k: sum(1 for r in rows if r.get(k)) / n for k in keys}


def generate_local(
    examples: list[dict[str, Any]],
    *,
    base_model: str,
    adapter: Path | None,
    max_new_tokens: int,
    temperature: float,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_src = (
        str(adapter)
        if adapter and (adapter / "tokenizer_config.json").exists()
        else base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    use_cuda = torch.cuda.is_available()
    use_mps = (
        (not use_cuda)
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if use_cuda:
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    elif use_mps:
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if use_mps:
        model = model.to("mps")
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        if use_mps:
            model = model.to("mps")
    model.eval()

    out: list[dict[str, Any]] = []
    for ex in examples:
        prompt = tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        with torch.inference_mode():
            ids = model.generate(**inputs, **gen_kwargs)
        gen = ids[0, inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        scores = score_generation(text, ex.get("gold") or {})
        out.append(
            {
                "trace_id": ex.get("trace_id"),
                "family": ex.get("family"),
                "gold": ex.get("gold"),
                "generation": text,
                **scores,
            }
        )
        status = "PASS" if scores["task_pass"] else "FAIL"
        print(f"  [{status}] {ex.get('trace_id')}  {scores['preview'][:90]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=DATA / "test.jsonl")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--base-model", default=MODEL_ID)
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if not args.split.exists():
        print(f"ERROR: {args.split} missing. Run: python src/write_data.py", file=sys.stderr)
        return 1

    adapter = None if args.base_only else args.adapter
    if adapter is None and not args.base_only:
        default = OUTPUTS / "lora-sft"
        adapter = default if default.exists() else None
    if adapter is None and not args.base_only:
        print("No adapter. Pass --adapter or --base-only.", file=sys.stderr)
        return 1

    examples = load_jsonl(args.split)
    label = "base" if args.base_only else "adapter"
    print(f"Scoring {len(examples)} {label} generations on {args.split.name}")
    scored = generate_local(
        examples,
        base_model=args.base_model,
        adapter=adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    summary = aggregate(scored)
    print(f"\n=== {label} on frozen test (fraction) ===")
    for k, v in summary.items():
        print(f"  {k:18s} {v:5.1%}")

    out = args.output or (OUTPUTS / f"{label}_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "source": label,
                "base_model": args.base_model,
                "adapter": str(adapter) if adapter else None,
                "n": len(scored),
                "summary": summary,
                "per_prompt": scored,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
