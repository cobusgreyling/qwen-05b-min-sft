#!/usr/bin/env python3
"""Emit colab_qwen_05b_min_sft.ipynb — clones the repo, then imports src/.

The dataset, scorer, and prompts live in src/dataset.py and src/evaluate.py.
This file must not copy them.
"""

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
    return text.splitlines(keepends=True)


cells = [
    md(
        """<p align="center">
  <img src="https://raw.githubusercontent.com/cobusgreyling/qwen-05b-min-sft/main/assets/header.png" alt="Qwen2.5-0.5B-Instruct Open Weights" width="100%" />
</p>

# Least-data SFT — Qwen2.5-0.5B-Instruct (Colab)

Fine-tune **open-weight** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) on **8 labeled cards**. Then score the same 6 frozen tickets **before and after**, and against two prompt-only baselines, so you can see what SFT is actually doing.

Companion repo: [cobusgreyling/qwen-05b-min-sft](https://github.com/cobusgreyling/qwen-05b-min-sft) · write-up: [`BLOG.md`](https://github.com/cobusgreyling/qwen-05b-min-sft/blob/main/BLOG.md)

**Runtime → Change runtime type → GPU (T4).** A T4 run is about 10 minutes including downloads.

This notebook **clones the repo** and imports `src/`. The 8 / 4 / 6 cards live in one place (`src/dataset.py`). Edit a ticket there and the notebook picks it up — there is no second copy of the data in these cells.
"""
    ),
    md("## 0 — Bootstrap (clone + install + import)"),
    code(
        r"""
from pathlib import Path
import os, sys, subprocess

REPO = "https://github.com/cobusgreyling/qwen-05b-min-sft.git"

if Path("/content").exists():
    os.chdir("/content")
    if not Path("/content/qwen-05b-min-sft/src/dataset.py").exists():
        subprocess.check_call(["git", "clone", "--depth", "1", REPO])
    os.chdir("/content/qwen-05b-min-sft")
else:
    if not (Path.cwd() / "src" / "dataset.py").exists():
        raise SystemExit("Run this notebook from the repo root, or open it in Colab.")

sys.path.insert(0, str(Path.cwd() / "src"))
print("cwd:", Path.cwd())

%pip install -q "transformers==5.15.0" "datasets==5.0.1" "trl==1.10.0" "peft==0.20.0" "accelerate==1.14.0" sentencepiece

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

from common import MODEL_ID, OUTPUTS
from dataset import HOLDOUT, SCHEMA_SYSTEM, SYSTEM, TEST, TRAIN, TRAIN_EXTRA, card_text, to_sft_row
from evaluate import (
    aggregate,
    generate_one,
    load_causal,
    messages_for,
    score_examples,
    score_generation,
    write_report,
)
from write_data import main as write_data

write_data()
print("system (SFT):", SYSTEM)
print("schema prompt mentions <<TICKET>>:", "<<TICKET>>" in SCHEMA_SYSTEM)
print(f"train={len(TRAIN)} extra={len(TRAIN_EXTRA)} holdout={len(HOLDOUT)} test={len(TEST)}")
"""
    ),
    md(
        """## 1 — The data (imported, not pasted)

Least data that still covers the contract: **two cards per family**. `TRAIN_EXTRA` is only for the 2/4/8/16 curve (`./run.sh curve`); this notebook trains on the eight.
"""
    ),
    code(
        r"""
print(f"{'id':<8} {'split':<12} {'family':<10} verdict        reason")
print("-" * 62)
for ex in TRAIN + HOLDOUT + TEST:
    g = ex["gold"]
    print(f"{ex['id']:<8} {ex['split']:<12} {ex['family']:<10} {g['verdict']:<14} {g['reason_code']}")

print("\n--- one training card (T-1042) ---")
print("USER:\n", TRAIN[0]["user"])
print("\nCOMPLETION (this is the label):\n", card_text(TRAIN[0]))
"""
    ),
    md(
        """## 2 — How the data is masked

The model **reads** the system + user text. It is **graded** only on the ticket card.

```
INPUT:   [system][user][assistant card]
LABELS:  -100    -100   card token ids
```

TRL does this when the dataset is prompt/completion and `completion_only_loss=True`.
"""
    ),
    code(
        r"""
from transformers import AutoTokenizer

example = TRAIN[0]
row = to_sft_row(example)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
full_text = tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=False, add_generation_prompt=False)
prompt_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
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
print(row["completion"][0]["content"])
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
        """## 3 — Scorer (imported from `src/evaluate.py`)

| Check | Means |
|-------|--------|
| `has_ticket_tags` | both `<<TICKET>>` wrappers |
| `format_pass` | parseable card, allowed verdict / reason, no role leak |
| `task_pass` | format **and** id / verdict / amount / reason match gold |
| `note_facts_pass` | the note does not invent numbers absent from the ticket |

`note` is not scored for wording. T-2003 can `task_pass` and still fail facts if it writes `4900`.
"""
    ),
    code(
        r"""
for ex in TEST:
    s = score_generation(card_text(ex), ex["gold"], facts=ex["facts"])
    assert s["task_pass"] and s["note_facts_pass"], ex["id"]
print("scorer: all 6 gold test cards pass task + facts")

t2003 = next(ex for ex in TEST if ex["id"] == "T-2003")
lie = card_text(t2003).replace("Requested refund exceeds invoice total; refuse.",
                               "Asked amount (3300) exceeds invoice (4900).")
s = score_generation(lie, t2003["gold"], facts=t2003["facts"])
print(f"T-2003 invented 4900 → task_pass={s['task_pass']}  note_facts_pass={s['note_facts_pass']}")
"""
    ),
    md(
        """## 4 — Three stock pictures (one model load)

Same six tickets, greedy decode, **no adapter**:

1. **thin** — the SFT system prompt (one line, no schema)
2. **schema** — the card layout spelled out in the prompt
3. **icl** — the 8 train cards as few-shot turns

If schema or ICL already solve the task, SFT is optional. If they do not, the adapter has something to do.
"""
    ),
    code(
        r"""
import json

base_model, base_tok = load_causal(MODEL_ID)

def dump(rows, source, mode, path):
    write_report(path, source=source, base_model=MODEL_ID, adapter=None, mode=mode, scored=rows)
    return {"source": source, "mode": mode, "n": len(rows), "summary": aggregate(rows), "per_prompt": rows}

BASE = dump(score_examples(base_model, base_tok, TEST, mode="thin", label="stock thin"),
            "base", "thin", OUTPUTS / "base_eval.json")
SCHEMA = dump(score_examples(base_model, base_tok, TEST, mode="schema", label="stock+schema"),
              "schema", "schema", OUTPUTS / "schema_eval.json")
ICL = dump(score_examples(base_model, base_tok, TEST, mode="icl", label="8-shot ICL"),
           "icl", "icl", OUTPUTS / "icl_eval.json")

print("\n=== stock baselines ===")
print(f"{'check':<18} {'thin':>8} {'schema':>8} {'icl':>8}")
print("-" * 44)
for k in ("has_ticket_tags", "format_pass", "task_pass", "note_facts_pass"):
    print(f"{k:<18} {BASE['summary'][k]:8.1%} {SCHEMA['summary'][k]:8.1%} {ICL['summary'][k]:8.1%}")

del base_model, base_tok
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""
    ),
    md(
        """## 5 — Train LoRA on the 8 cards

| Knob | Value |
|------|--------|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | LoRA rank 8, α = 16 |
| Loss | completion only |
| Steps | 40 (~10 epochs on 8 cards, batch 1 × accum 2) |
| LR | 2e-4 cosine |

Hold-out is eval loss only. `test.jsonl` is not loaded. The 2/4/8/16 curve is `./run.sh curve` — too long for this notebook.
"""
    ),
    code(
        r"""
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
dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (
    torch.float16 if (use_cuda or use_mps) else torch.float32
)
kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
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
    completion_only_loss=True,
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
        """## 6 — AFTER: same 6 tickets, now with the adapter

Same prompts (thin system). Same scorer. Same greedy decode. Only the weights changed.
"""
    ),
    code(
        r"""
ADAPTER_ROWS = score_examples(trainer.model, tok, TEST, mode="thin", label="adapter")
ADAPTER = dump(ADAPTER_ROWS, "adapter", "thin", OUTPUTS / "adapter_eval.json")

print("\n=== before vs after vs prompt baselines ===")
print(f"{'check':<18} {'thin':>8} {'schema':>8} {'icl':>8} {'lora-8':>8}")
print("-" * 54)
for k in ("has_ticket_tags", "format_pass", "task_pass", "note_facts_pass"):
    print(f"{k:<18} {BASE['summary'][k]:8.1%} {SCHEMA['summary'][k]:8.1%} {ICL['summary'][k]:8.1%} {ADAPTER['summary'][k]:8.1%}")

print("\n=== side-by-side (thin stock vs LoRA-8) ===")
by_base = {r["trace_id"]: r for r in BASE["per_prompt"]}
by_adp = {r["trace_id"]: r for r in ADAPTER["per_prompt"]}
for ex in TEST:
    tid = ex["id"]
    print("\n" + "=" * 64)
    print(f"{tid}  gold={ex['gold']}")
    print("BEFORE:", by_base[tid]["generation"].replace("\n", " | ")[:220])
    print("AFTER :", by_adp[tid]["generation"].replace("\n", " | ")[:220])
    print(f"       tags {by_base[tid]['has_ticket_tags']}→{by_adp[tid]['has_ticket_tags']}   "
          f"task {by_base[tid]['task_pass']}→{by_adp[tid]['task_pass']}   "
          f"facts {by_base[tid]['note_facts_pass']}→{by_adp[tid]['note_facts_pass']}")
"""
    ),
    md(
        """## 7 — Try a ticket the trainer never saw

Type a new ticket (keep an id like `T-9xxx`). If SFT worked, you should get a card, not an essay.
"""
    ),
    code(
        r"""
def deskcard(user_text: str) -> str:
    return generate_one(
        trainer.model,
        tok,
        messages_for({"user": user_text, "gold": {}, "id": "probe"}, mode="thin"),
    )

probe = "Ticket T-9001. Charged twice for the same add-on, an extra 2800 cents. Refund the duplicate only."
print(deskcard(probe))
"""
    ),
    md(
        """## How to read the result

Improvement is **not** “loss went down.” Look at the table in cell 6:

1. **`has_ticket_tags`** — did the wrapper transfer?
2. **`task_pass`** — did unseen ids get the right verdict / amount / reason?
3. **`note_facts_pass`** — did the note invent amounts?
4. **schema / ICL vs LoRA** — if the prompt baselines already score well, you did not need SFT for the wrapper. If they do not, the eight cards did something context cannot.

The 2 / 4 / 8 / 16 data-size curve is a local command, not this notebook:

```bash
./run.sh curve
./run.sh compare
```

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

if __name__ == "__main__":
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")
