#!/usr/bin/env python3
"""Emit colab_qwen_05b_min_sft.ipynb — self-contained Colab lab."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "colab_qwen_05b_min_sft.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": _lines(source),
    }


def _lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    lines = text.splitlines(keepends=True)
    return lines


cells = [
    md(
        """<p align="center">
  <img src="https://raw.githubusercontent.com/cobusgreyling/qwen-05b-min-sft/main/assets/header.png" alt="Qwen2.5-0.5B-Instruct Open Weights" width="100%" />
</p>

# Least-data SFT — Qwen2.5-0.5B-Instruct (Colab)

Fine-tune **open-weight** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) on **8 labeled cards**. Then score the same 6 frozen tickets **before and after** so you can see the model actually change.

Companion repo: [cobusgreyling/qwen-05b-min-sft](https://github.com/cobusgreyling/qwen-05b-min-sft) · write-up: [`BLOG.md`](https://github.com/cobusgreyling/qwen-05b-min-sft/blob/main/BLOG.md)

**Runtime → Change runtime type → GPU (T4).** CPU works but is slower. A T4 run is about 5–10 minutes including downloads.

| Split | Cards | Touches weights? |
|-------|------:|------------------|
| train | **8** | yes |
| hold-out | 4 | eval loss only |
| test | 6 | **no** — scored before and after |

What you are teaching is a brand-new output contract (`<<TICKET>>…<</TICKET>>`) that stock 0.5B will not emit. The system prompt only names the clerk. The schema lives in the eight completions.
"""
    ),
    md("## 0 — Runtime check"),
    code(
        r"""
from pathlib import Path
import os, sys

if Path("/content").exists():
    PROJECT = Path("/content/qwen-05b-min-sft")
else:
    PROJECT = Path.cwd() if Path.cwd().name == "qwen-05b-min-sft" else Path("qwen-05b-min-sft")

PROJECT.mkdir(parents=True, exist_ok=True)
(PROJECT / "data").mkdir(exist_ok=True)
(PROJECT / "outputs").mkdir(exist_ok=True)
os.chdir(PROJECT)
print("cwd:", Path.cwd())

import torch
print("torch:", torch.__version__)
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"gpu  : {torch.cuda.get_device_name(0)}")
    print(f"vram : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("gpu  : Apple MPS")
else:
    print("gpu  : CPU (train will be slow)")
"""
    ),
    md("## 1 — Install"),
    code(
        r"""
%pip install -q -U "transformers>=4.46" datasets "trl>=0.14" peft accelerate sentencepiece

import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no")
"""
    ),
    md(
        """## 2 — The data (8 train / 4 hold-out / 6 test)

Least data that still covers the contract: **two cards per family**.

| Family | What it teaches | Train ids |
|--------|-----------------|-----------|
| duplicate | refund the extra charge | T-1042, T-2108 |
| fraud | escalate, amount 0 | T-3302, T-4410 |
| policy | refuse oversized / late refunds | T-8001, T-5519 |
| other | taste → no refund; outage → escalate | T-6622, T-7730 |

Hold-out and test reuse the families with **new ids and wording**. The trainer never opens `test`.
"""
    ),
    code(
        r'''
import json
from pathlib import Path

SYSTEM = (
    "You are DeskCard, a first-line support clerk. "
    "Reply with one ticket card and nothing else."
)
VERDICTS = ("REFUND", "NO_REFUND", "ESCALATE")
REASONS = ("DUPLICATE", "FRAUD", "POLICY", "OTHER")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN = [
    {"id":"T-1042","split":"train","family":"duplicate",
     "user":"Ticket T-1042. Customer says they were charged twice for the same Pro seat. Invoice shows two $49.99 lines. Refund the extra charge.",
     "gold":{"id":"T-1042","verdict":"REFUND","amount_cents":4999,"reason_code":"DUPLICATE"},
     "note":"Duplicate Pro seat line; refund the extra 4999 cents."},
    {"id":"T-2108","split":"train","family":"duplicate",
     "user":"Ticket T-2108. Billing listed the annual plan twice. Customer wants the second 7500-cent charge reversed.",
     "gold":{"id":"T-2108","verdict":"REFUND","amount_cents":7500,"reason_code":"DUPLICATE"},
     "note":"Annual plan billed twice; refund the second 7500 cents."},
    {"id":"T-3302","split":"train","family":"fraud",
     "user":"Ticket T-3302. Customer was in Ohio all week. Card just paid for a laptop in Jakarta. They did not authorize it.",
     "gold":{"id":"T-3302","verdict":"ESCALATE","amount_cents":0,"reason_code":"FRAUD"},
     "note":"Unauthorized foreign charge; escalate to fraud review."},
    {"id":"T-4410","split":"train","family":"fraud",
     "user":"Ticket T-4410. Forty failed logins overnight, then a password reset from a new device. Customer says that was not them.",
     "gold":{"id":"T-4410","verdict":"ESCALATE","amount_cents":0,"reason_code":"FRAUD"},
     "note":"Credential-stuffing pattern; escalate, do not refund here."},
    {"id":"T-8001","split":"train","family":"policy",
     "user":"Ticket T-8001. Angry customer wants a 9000-cent refund right now. The invoice total is only 2000 cents.",
     "gold":{"id":"T-8001","verdict":"NO_REFUND","amount_cents":0,"reason_code":"POLICY"},
     "note":"Asked amount exceeds invoice total; refuse under policy."},
    {"id":"T-5519","split":"train","family":"policy",
     "user":"Ticket T-5519. Purchase was 11 months ago. Customer wants a full refund. Our window is 30 days.",
     "gold":{"id":"T-5519","verdict":"NO_REFUND","amount_cents":0,"reason_code":"POLICY"},
     "note":"Outside the 30-day refund window; no refund."},
    {"id":"T-6622","split":"train","family":"other",
     "user":"Ticket T-6622. Customer wants a refund because they do not like the new UI color. No billing error, no outage.",
     "gold":{"id":"T-6622","verdict":"NO_REFUND","amount_cents":0,"reason_code":"OTHER"},
     "note":"Taste complaint is not a billing defect; no refund."},
    {"id":"T-7730","split":"train","family":"other",
     "user":"Ticket T-7730. VIP cannot download their invoice PDF. The billing page returns HTTP 500. They need a human.",
     "gold":{"id":"T-7730","verdict":"ESCALATE","amount_cents":0,"reason_code":"OTHER"},
     "note":"Broken invoice download for a VIP; escalate to engineering."},
]
HOLDOUT = [
    {"id":"T-8844","split":"holdout","family":"duplicate",
     "user":"Ticket T-8844. Two identical 3200-cent charges landed on the same invoice. Please reverse the extra one.",
     "gold":{"id":"T-8844","verdict":"REFUND","amount_cents":3200,"reason_code":"DUPLICATE"},
     "note":"Identical 3200-cent lines; refund the extra charge."},
    {"id":"T-9901","split":"holdout","family":"fraud",
     "user":"Ticket T-9901. MFA was bypassed and the account signed in from two countries in ten minutes. Customer is locked out.",
     "gold":{"id":"T-9901","verdict":"ESCALATE","amount_cents":0,"reason_code":"FRAUD"},
     "note":"Impossible travel plus MFA bypass; escalate to security."},
    {"id":"T-1015","split":"holdout","family":"policy",
     "user":"Ticket T-1015. Customer demands a 99999-cent refund. Invoice total is 1250 cents.",
     "gold":{"id":"T-1015","verdict":"NO_REFUND","amount_cents":0,"reason_code":"POLICY"},
     "note":"Refund request larger than the invoice; refuse."},
    {"id":"T-1120","split":"holdout","family":"other",
     "user":"Ticket T-1120. Package arrived one day late. Customer wants the full 16600 cents back. Delivery itself succeeded.",
     "gold":{"id":"T-1120","verdict":"NO_REFUND","amount_cents":0,"reason_code":"OTHER"},
     "note":"One-day late delivery is not a billing defect; no refund."},
]
TEST = [
    {"id":"T-2001","split":"test","family":"duplicate",
     "user":"Ticket T-2001. I was billed twice for invoice INV-2001 — forty-five dollars twice. Please fix the extra charge.",
     "gold":{"id":"T-2001","verdict":"REFUND","amount_cents":4500,"reason_code":"DUPLICATE"},
     "note":"Double 4500-cent charge; refund the extra line."},
    {"id":"T-2002","split":"test","family":"fraud",
     "user":"Ticket T-2002. Someone used my card in Jakarta this morning. I have been in Ohio the whole time and I did not buy anything.",
     "gold":{"id":"T-2002","verdict":"ESCALATE","amount_cents":0,"reason_code":"FRAUD"},
     "note":"Unauthorized overseas charge; escalate to fraud."},
    {"id":"T-2003","split":"test","family":"policy",
     "user":"Ticket T-2003. Refund 21000 cents immediately. The invoice is only 3300 cents but the customer is yelling.",
     "gold":{"id":"T-2003","verdict":"NO_REFUND","amount_cents":0,"reason_code":"POLICY"},
     "note":"Requested refund exceeds invoice total; refuse."},
    {"id":"T-2004","split":"test","family":"other",
     "user":"Ticket T-2004. The new icon is ugly. Cancel my plan and refund 6700 cents. Nothing is broken.",
     "gold":{"id":"T-2004","verdict":"NO_REFUND","amount_cents":0,"reason_code":"OTHER"},
     "note":"Aesthetic preference is not a defect; no refund."},
    {"id":"T-2005","split":"test","family":"other",
     "user":"Ticket T-2005. Partner webhooks are retrying in a loop and their quota is exhausted. The status page is blank. Need a human.",
     "gold":{"id":"T-2005","verdict":"ESCALATE","amount_cents":0,"reason_code":"OTHER"},
     "note":"Webhook storm plus blank status page; escalate."},
    {"id":"T-2006","split":"test","family":"duplicate",
     "user":"Ticket T-2006. We were charged for a duplicate seat license — an extra 12000 cents on this cycle. Reverse that seat only.",
     "gold":{"id":"T-2006","verdict":"REFUND","amount_cents":12000,"reason_code":"DUPLICATE"},
     "note":"Duplicate seat license; refund the extra 12000 cents."},
]

def card_text(ex):
    g = ex["gold"]
    return (
        "<<TICKET>>\n"
        f"id: {g['id']}\n"
        f"verdict: {g['verdict']}\n"
        f"amount_cents: {g['amount_cents']}\n"
        f"reason_code: {g['reason_code']}\n"
        f"note: {ex['note']}\n"
        "<</TICKET>>"
    )

def to_row(ex):
    return {
        "trace_id": ex["id"],
        "family": ex["family"],
        "split": ex["split"],
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["user"]},
        ],
        "completion": [{"role": "assistant", "content": card_text(ex)}],
        "gold": ex["gold"],
    }

def write_split(name, examples):
    path = Path("data") / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(to_row(ex), ensure_ascii=False) + "\n")
    return path

write_split("train", TRAIN)
write_split("holdout", HOLDOUT)
write_split("test", TEST)

print(f"{'id':<8} {'split':<8} {'family':<10} verdict        reason")
print("-" * 58)
for ex in TRAIN + HOLDOUT + TEST:
    g = ex["gold"]
    print(f"{ex['id']:<8} {ex['split']:<8} {ex['family']:<10} {g['verdict']:<14} {g['reason_code']}")

print("\n--- one training card (T-1042) ---")
print("USER:\n", TRAIN[0]["user"])
print("\nCOMPLETION (this is the label):\n", card_text(TRAIN[0]))
'''
    ),
    md(
        """## 3 — How the data is masked

The model **reads** the system + user text. It is **graded** only on the ticket card.

TRL does this when the dataset is prompt/completion and `completion_only_loss=True`:

```
INPUT:   [system][user][assistant card]
LABELS:  -100    -100   card token ids
```

`-100` is ignored by cross-entropy. You are not hiding the ticket. You are not scoring the clerk for reciting the customer.
"""
    ),
    code(
        r"""
from transformers import AutoTokenizer

example = TRAIN[0]
prompt = to_row(example)["prompt"]
completion = to_row(example)["completion"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
full_text = tokenizer.apply_chat_template(prompt + completion, tokenize=False, add_generation_prompt=False)
prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
n_prompt = len(prompt_ids)
assert full_ids[:n_prompt] == prompt_ids, "chat template prefix mismatch"
labels = [-100] * n_prompt + full_ids[n_prompt:]

print(f"total={len(full_ids)}  masked={n_prompt}  supervised={len(full_ids)-n_prompt}")
print()
print("--- prompt (MASK) ---")
print(prompt_text)
print("--- completion (LOSS) ---")
print(completion[0]["content"])
print()
print(f"{'i':>5}  {'tag':<4}  token")
print("-" * 48)
start = max(0, n_prompt - 3)
end = min(len(full_ids), n_prompt + 14)
if start:
    print("  ...")
for i in range(start, end):
    piece = tokenizer.decode([full_ids[i]]).replace("\n", "\\n")
    tag = "MASK" if labels[i] == -100 else "LOSS"
    print(f"{i:5d}  {tag:<4}  {piece!r}")
print("  ...  (rest of the card is LOSS)")
"""
    ),
    md(
        """## 4 — Scorer (how we prove improvement)

Same six test tickets, greedy decoding, two checkpoints: **stock 0.5B** then **0.5B + LoRA**.

| Check | Means |
|-------|--------|
| `format_pass` | `<<TICKET>>` card with allowed verdict / reason |
| `task_pass` | format **and** id / verdict / amount / reason match gold |

`note` is not scored for wording. Role-marker leaks fail format.
"""
    ),
    code(
        r"""
import re
from typing import Any

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

def parse_card(text: str):
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
    checks = {
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
        checks["non_empty"] and checks["no_role_leak"] and checks["has_ticket_tags"]
        and checks["parsed_ok"] and checks["verdict_allowed"] and checks["reason_allowed"]
    )
    checks["task_pass"] = (
        checks["format_pass"] and checks["id_match"] and checks["verdict_match"]
        and checks["amount_match"] and checks["reason_match"]
    )
    checks["preview"] = text.replace("\n", " | ")[:160]
    return checks

def aggregate(rows):
    keys = ["has_ticket_tags","format_pass","id_match","verdict_match","amount_match","reason_match","task_pass"]
    n = max(len(rows), 1)
    return {k: sum(1 for r in rows if r.get(k)) / n for k in keys}

# Gold cards must all pass — otherwise the test is broken.
for ex in TEST:
    s = score_generation(card_text(ex), ex["gold"])
    assert s["task_pass"], ex["id"]
print("scorer: all 6 gold test cards pass")
"""
    ),
    md(
        """## 5 — BEFORE: stock Qwen2.5-0.5B-Instruct

Same six test tickets. Greedy decode. No adapter. This is the baseline you compare against.
"""
    ),
    code(
        r"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def pick_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32

def load_base():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"trust_remote_code": True, "torch_dtype": pick_dtype()}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    if (not torch.cuda.is_available()) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    return model, tok

def generate_one(model, tok, prompt_messages, max_new_tokens=96):
    text = tok.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen = ids[0, inputs["input_ids"].shape[-1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()

def score_split(model, tok, examples, label):
    rows = []
    print(f"=== {label} ===")
    for ex in examples:
        row = to_row(ex)
        text = generate_one(model, tok, row["prompt"])
        scores = score_generation(text, ex["gold"])
        rec = {"trace_id": ex["id"], "family": ex["family"], "gold": ex["gold"], "generation": text, **scores}
        rows.append(rec)
        flag = "PASS" if scores["task_pass"] else "FAIL"
        print(f"  [{flag}] {ex['id']}")
        print(f"         gold: {ex['gold']}")
        print(f"         got : {text.replace(chr(10), ' | ')[:140]}")
    summary = aggregate(rows)
    print()
    for k, v in summary.items():
        print(f"  {k:18s} {v:5.1%}")
    return {"source": label, "n": len(rows), "summary": summary, "per_prompt": rows}

base_model, base_tok = load_base()
BASE_REPORT = score_split(base_model, base_tok, TEST, "base")
Path("outputs/base_eval.json").write_text(json.dumps(BASE_REPORT, indent=2, ensure_ascii=False))
print("\nwrote outputs/base_eval.json")

# Free the base copy so the trainer can load cleanly.
del base_model, base_tok
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""
    ),
    md(
        """## 6 — Train LoRA on the 8 cards

| Knob | Value |
|------|--------|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | LoRA rank 8, α = 16 |
| Loss | completion only |
| Steps | 40 (~10 epochs on 8 cards, batch 1 × accum 2) |
| LR | 2e-4 cosine |
| Max length | 384 |

Hold-out is eval loss only. `test.jsonl` is not loaded.
"""
    ),
    code(
        r"""
import json
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

def load_sft(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append({"prompt": rec["prompt"], "completion": rec["completion"]})
    return Dataset.from_list(rows)

train_ds = load_sft("data/train.jsonl")
hold_ds = load_sft("data/holdout.jsonl")
print(f"train={len(train_ds)}  holdout={len(hold_ds)}  test=0 (frozen)")

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

use_cuda = torch.cuda.is_available()
use_mps = (not use_cuda) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
kwargs = {"trust_remote_code": True, "torch_dtype": pick_dtype()}
if use_cuda:
    kwargs["device_map"] = "auto"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
if use_mps:
    model = model.to("mps")
model.config.use_cache = False

peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

sft_args = SFTConfig(
    output_dir="outputs/lora-sft",
    max_steps=40,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=4,
    logging_steps=1,
    save_steps=40,
    save_total_limit=1,
    eval_strategy="steps",
    eval_steps=20,
    max_length=384,
    completion_only_loss=True,   # mask the prompt
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    fp16=False,
    bf16=False,
    max_grad_norm=1.0,
    optim="adamw_torch",
    report_to="none",
    seed=42,
    remove_unused_columns=False,
    dataloader_pin_memory=bool(use_cuda),
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_ds,
    eval_dataset=hold_ds,
    processing_class=tok,
    peft_config=peft_config,
)

result = trainer.train()
print("train metrics:", json.dumps(result.metrics, indent=2))
trainer.save_model("outputs/lora-sft")
tok.save_pretrained("outputs/lora-sft")
print("saved adapter → outputs/lora-sft")
"""
    ),
    md(
        """## 7 — AFTER: same 6 tickets, now with the adapter

This is the improvement test. **Same prompts. Same scorer. Same greedy decode.** Only the weights changed (a small LoRA on top of the frozen 0.5B).
"""
    ),
    code(
        r"""
ADAPTER_REPORT = score_split(trainer.model, tok, TEST, "adapter")
Path("outputs/adapter_eval.json").write_text(json.dumps(ADAPTER_REPORT, indent=2, ensure_ascii=False))

print("\n=== before vs after (frozen test) ===")
print(f"{'check':<18} {'base':>8} {'adapter':>8}")
print("-" * 36)
for k in BASE_REPORT["summary"]:
    b = BASE_REPORT["summary"][k]
    a = ADAPTER_REPORT["summary"][k]
    print(f"{k:<18} {b:8.1%} {a:8.1%}")

print("\n=== side-by-side generations ===")
by_base = {r["trace_id"]: r for r in BASE_REPORT["per_prompt"]}
by_adp = {r["trace_id"]: r for r in ADAPTER_REPORT["per_prompt"]}
for ex in TEST:
    tid = ex["id"]
    print("\n" + "=" * 64)
    print(f"{tid}  gold={ex['gold']}")
    print("BEFORE:", by_base[tid]["generation"].replace("\n", " | ")[:220])
    print("AFTER :", by_adp[tid]["generation"].replace("\n", " | ")[:220])
    print(f"       format {by_base[tid]['format_pass']}→{by_adp[tid]['format_pass']}   "
          f"task {by_base[tid]['task_pass']}→{by_adp[tid]['task_pass']}")
"""
    ),
    md(
        """## 8 — Try a ticket the trainer never saw

Type a new ticket (keep an id like `T-9xxx`). If SFT worked, you should get a card, not an essay.
"""
    ),
    code(
        r"""
def deskcard(user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
    ]
    return generate_one(trainer.model, tok, messages)

probe = "Ticket T-9001. Charged twice for the same add-on, an extra 2800 cents. Refund the duplicate only."
print(deskcard(probe))
"""
    ),
    md(
        """## How to read the result

Improvement is **not** “loss went down.” Loss going down only means the 8 cards got easier to copy.

Look at the table in cell 7:

1. **`format_pass` up** — stock 0.5B started writing prose; the adapter emits `<<TICKET>>` cards.
2. **`task_pass` up** — on *unseen* ids, verdict / amount / reason match gold.
3. **Side-by-side text** — same user ticket, different assistant.

If format moves and task does not, you taught the wrapper but not the policy. Add a couple more cards in that family and train again. Eight examples is the point of the lab, not a production desk.

Adapter folder: `outputs/lora-sft/` (a few MB). The 0.5B base stays on Hugging Face; you only own the LoRA.
"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
