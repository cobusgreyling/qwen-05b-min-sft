#!/usr/bin/env python3
"""
Score generations against the ticket-card contract.

format_pass       — looks like a DeskCard (tags, parseable fields, allowed enums)
task_pass         — format plus id / verdict / amount / reason match gold
note_facts_pass   — parsed note does not invent numbers absent from the ticket

Modes
-----
thin     stock or adapter, original thin system prompt (the SFT setup)
schema   stock model + schema spelled out in the system prompt (no SFT)
icl      stock model + the 8 train cards as few-shot turns (no SFT)

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
from dataset import (  # noqa: E402
    REASONS,
    SCHEMA_SYSTEM,
    SYSTEM,
    TEST,
    TRAIN,
    VERDICTS,
    card_text,
    example_by_id,
)

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

NOTE_NUM_RE = re.compile(r"-?\d+")

SCORE_KEYS = [
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
    "note_facts_pass",
]


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


def allowed_note_numbers(gold: dict[str, Any], facts: dict[str, Any] | None) -> set[int]:
    allowed: set[int] = set()
    if gold.get("amount_cents") is not None:
        allowed.add(int(gold["amount_cents"]))
    if facts:
        for n in facts.get("amounts_cents") or []:
            allowed.add(int(n))
    ticket_id = str(gold.get("id") or "")
    m = re.search(r"(\d+)$", ticket_id)
    if m:
        allowed.add(int(m.group(1)))
    return allowed


def note_facts_ok(
    parsed: dict[str, Any] | None,
    gold: dict[str, Any],
    facts: dict[str, Any] | None,
) -> bool:
    if parsed is None:
        return False
    nums = {int(n) for n in NOTE_NUM_RE.findall(parsed.get("note") or "")}
    if not nums:
        return True
    return nums <= allowed_note_numbers(gold, facts)


def score_generation(
    text: str,
    gold: dict[str, Any],
    facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        "note_facts_pass": False,
        "predicted": parsed,
    }
    if parsed is not None:
        checks["verdict_allowed"] = parsed["verdict"] in VERDICTS
        checks["reason_allowed"] = parsed["reason_code"] in REASONS
        checks["id_match"] = parsed["id"] == gold.get("id")
        checks["verdict_match"] = parsed["verdict"] == gold.get("verdict")
        checks["amount_match"] = parsed["amount_cents"] == gold.get("amount_cents")
        checks["reason_match"] = parsed["reason_code"] == gold.get("reason_code")
        checks["note_facts_pass"] = note_facts_ok(parsed, gold, facts)

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
    n = max(len(rows), 1)
    return {k: sum(1 for r in rows if r.get(k)) / n for k in SCORE_KEYS}


def as_eval_example(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize a dataset example or a JSONL row."""
    if "user" in obj and "gold" in obj and ("id" in obj or "trace_id" in obj):
        tid = obj.get("id") or obj.get("trace_id")
        facts = obj.get("facts")
        if facts is None:
            try:
                facts = example_by_id(tid).get("facts")
            except KeyError:
                facts = {"amounts_cents": []}
        return {
            "id": tid,
            "trace_id": tid,
            "family": obj.get("family"),
            "user": obj["user"],
            "gold": obj["gold"],
            "facts": facts or {"amounts_cents": []},
        }
    tid = obj.get("trace_id")
    user = None
    for msg in obj.get("prompt") or []:
        if msg.get("role") == "user":
            user = msg.get("content")
            break
    facts = obj.get("facts")
    if facts is None and tid:
        try:
            facts = example_by_id(tid).get("facts")
        except KeyError:
            facts = {"amounts_cents": []}
    return {
        "id": tid,
        "trace_id": tid,
        "family": obj.get("family"),
        "user": user or "",
        "gold": obj.get("gold") or {},
        "facts": facts or {"amounts_cents": []},
    }


def messages_for(
    example: dict[str, Any],
    *,
    mode: str = "thin",
    shots: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    if mode == "schema":
        system = SCHEMA_SYSTEM
    elif mode in {"thin", "icl"}:
        system = SYSTEM
    else:
        raise ValueError(f"unknown mode: {mode}")

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if mode == "icl":
        for shot in shots if shots is not None else TRAIN:
            messages.append({"role": "user", "content": shot["user"]})
            messages.append({"role": "assistant", "content": card_text(shot)})
    messages.append({"role": "user", "content": example["user"]})
    return messages


def load_causal(base_model: str, adapter: Path | None = None):
    """Load the 0.5B (and optional LoRA) once. Reuse across thin / schema / ICL."""
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
    return model, tokenizer


def generate_one(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def score_examples(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    *,
    mode: str = "thin",
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    label: str = "",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tag = label or mode
    print(f"=== {tag} (mode={mode}) ===")
    for raw in examples:
        ex = as_eval_example(raw)
        text = generate_one(
            model,
            tokenizer,
            messages_for(ex, mode=mode),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        scores = score_generation(text, ex.get("gold") or {}, facts=ex.get("facts"))
        out.append(
            {
                "trace_id": ex.get("trace_id"),
                "family": ex.get("family"),
                "gold": ex.get("gold"),
                "facts": ex.get("facts"),
                "mode": mode,
                "generation": text,
                **scores,
            }
        )
        status = "PASS" if scores["task_pass"] else "FAIL"
        facts_flag = "facts+" if scores["note_facts_pass"] else "facts-"
        print(f"  [{status} {facts_flag}] {ex.get('trace_id')}  {scores['preview'][:80]}")
    summary = aggregate(out)
    print()
    for k, v in summary.items():
        print(f"  {k:18s} {v:5.1%}")
    return out


def generate_local(
    examples: list[dict[str, Any]],
    *,
    base_model: str,
    adapter: Path | None,
    max_new_tokens: int,
    temperature: float,
    mode: str = "thin",
) -> list[dict[str, Any]]:
    model, tokenizer = load_causal(base_model, adapter)
    return score_examples(
        model,
        tokenizer,
        examples,
        mode=mode,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def rescore_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("per_prompt") or []
    rescored = []
    for row in rows:
        tid = row.get("trace_id")
        gold = row.get("gold") or {}
        facts = row.get("facts")
        if facts is None and tid:
            try:
                facts = example_by_id(tid).get("facts")
            except KeyError:
                facts = {"amounts_cents": []}
        scores = score_generation(row.get("generation") or "", gold, facts=facts)
        updated = {
            **row,
            **scores,
            "facts": facts,
        }
        rescored.append(updated)
    data["per_prompt"] = rescored
    data["summary"] = aggregate(rescored)
    data["n"] = len(rescored)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def write_report(
    path: Path,
    *,
    source: str,
    base_model: str,
    adapter: Path | None,
    mode: str,
    scored: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source": source,
                "base_model": base_model,
                "adapter": str(adapter) if adapter else None,
                "mode": mode,
                "n": len(scored),
                "summary": aggregate(scored),
                "per_prompt": scored,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=DATA / "test.jsonl")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--base-model", default=MODEL_ID)
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument(
        "--mode",
        choices=("thin", "schema", "icl"),
        default="thin",
        help="thin = SFT prompt; schema = spell out the card; icl = 8-shot train cards",
    )
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--rescore",
        type=Path,
        default=None,
        help="Re-run the scorer on an existing eval JSON (no model load).",
    )
    args = ap.parse_args()

    if args.rescore is not None:
        if not args.rescore.exists():
            print(f"ERROR: {args.rescore} missing", file=sys.stderr)
            return 1
        data = rescore_report(args.rescore)
        print(f"Rescored {args.rescore}  n={data['n']}")
        for k, v in data["summary"].items():
            print(f"  {k:18s} {v:5.1%}")
        return 0

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

    raw_rows = load_jsonl(args.split)
    examples = [as_eval_example(r) for r in raw_rows]
    if args.base_only:
        label = {"thin": "base", "schema": "schema", "icl": "icl"}[args.mode]
    else:
        label = "adapter"

    print(f"Scoring {len(examples)} {label} generations on {args.split.name}  mode={args.mode}")
    scored = generate_local(
        examples,
        base_model=args.base_model,
        adapter=adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        mode=args.mode,
    )
    summary = aggregate(scored)
    print(f"\n=== {label} on frozen test (fraction) ===")
    for k, v in summary.items():
        print(f"  {k:18s} {v:5.1%}")

    out = args.output or (OUTPUTS / f"{label}_eval.json")
    write_report(
        out,
        source=label,
        base_model=args.base_model,
        adapter=adapter,
        mode=args.mode,
        scored=scored,
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
