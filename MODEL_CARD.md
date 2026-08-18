---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
license: mit
pipeline_tag: text-generation
tags:
  - lora
  - sft
  - qwen2.5
  - peft
  - trl
  - ticket-classification
---

# DeskCard LoRA — Qwen2.5-0.5B-Instruct

Least-data SFT adapter that teaches stock [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) a rigid five-field support ticket card.

This is a **teaching adapter**, not a refund policy.

- Lab: [cobusgreyling/qwen-05b-min-sft](https://github.com/cobusgreyling/qwen-05b-min-sft)
- Write-up: [BLOG.md](https://github.com/cobusgreyling/qwen-05b-min-sft/blob/main/BLOG.md)
- Base weights: Apache 2.0 (Qwen / Alibaba)
- Adapter: MIT

## What it emits

```
<<TICKET>>
id: T-NNNN
verdict: REFUND | NO_REFUND | ESCALATE
amount_cents: <integer>
reason_code: DUPLICATE | FRAUD | POLICY | OTHER
note: one short sentence
<</TICKET>>
```

The system prompt used at train and eval time is deliberately thin:

```
You are DeskCard, a first-line support clerk.
Reply with one ticket card and nothing else.
```

The five fields are **not** in the prompt. They were learned from eight labeled completions.

## Training

| Knob | Value |
|------|--------|
| Base | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | LoRA r=8, α=16, dropout 0 |
| Targets | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| Loss | `completion_only_loss=True` |
| Data | 8 synthetic DeskCard tickets (2 per family) |
| Hold-out | 4 cards, eval loss only |
| Test | 6 cards, never loaded by the trainer |
| Steps | 40 (~10 epochs, batch 1 × accum 2) |
| LR | 2e-4 cosine |
| Stack | transformers 5.15.0, trl 1.10.0, peft 0.20.0 |

## Frozen-test snapshot

Same 6 tickets, greedy decode. Full table (schema prompt, 8-shot ICL, 2/4/8/16 curve): [`outputs/RESULTS.md`](https://github.com/cobusgreyling/qwen-05b-min-sft/blob/main/outputs/RESULTS.md).

| check | stock thin | stock+schema | 8-shot ICL | this adapter (8) | LoRA-16 |
|-------|-----------:|-------------:|-----------:|-----------------:|--------:|
| `<<TICKET>>` tags | 0 / 6 | 0 / 6 | 6 / 6 | **6 / 6** | 6 / 6 |
| `task_pass` | 0 / 6 | 0 / 6 | 2 / 6 | **4 / 6** | 6 / 6 |
| `note_facts_pass` | 0 / 6 | 0 / 6 | 6 / 6 | **5 / 6** | 6 / 6 |

The two 8-card misses are honest least-data: `CANCEL` leaked into `verdict` and `QUOTA` leaked into `reason_code` when those words were in the user text. Sixteen cards on the curve target that tail.

T-2003 `task_pass`es and fails `note_facts_pass` — the policy is right, the note invented `4900`. `note` is not part of `task_pass`.

## Quick start

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-0.5B-Instruct"
adapter = "cobusgreyling/qwen-05b-deskcard-lora"  # or a local outputs/lora-sft

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter)

messages = [
    {"role": "system", "content": "You are DeskCard, a first-line support clerk. Reply with one ticket card and nothing else."},
    {"role": "user", "content": "Ticket T-9001. Charged twice for the same add-on, an extra 2800 cents."},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

If the Hugging Face repo is empty, grab the adapter from the [GitHub release](https://github.com/cobusgreyling/qwen-05b-min-sft/releases) and point `PeftModel.from_pretrained` at the extracted folder.

## Intended use / limitations

- Demo of least-data format SFT. Not a production clerk.
- Synthetic tickets only. Do not use the verdicts as a real refund policy.
- 0.5B + 8 cards will copy user nouns into the enum (`CANCEL`, `QUOTA`).
- `note` is free text and is scored only for invented numbers (`note_facts_pass`).

## Citation

```
@software{qwen05b_min_sft,
  title  = {Least-data SFT of Qwen2.5-0.5B-Instruct},
  author = {Greyling, Cobus},
  year   = {2026},
  url    = {https://github.com/cobusgreyling/qwen-05b-min-sft}
}
```
