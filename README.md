<p align="center">
  <img src="assets/header.png" alt="Qwen2.5-0.5B-Instruct Open Weights" width="100%" />
</p>

# Least-Data SFT of Qwen2.5-0.5B-Instruct

<p align="center">
  <strong>Qwen2.5-0.5B-Instruct</strong> · open weights · Apache 2.0 · LoRA SFT · Colab T4<br/>
  Teach a 0.5B clerk a new ticket-card format with <strong>8 training examples</strong>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/cobusgreyling/qwen-05b-min-sft/blob/main/colab_qwen_05b_min_sft.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" /></a>
  <a href="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"><img src="https://img.shields.io/badge/base-Qwen2.5--0.5B--Instruct-ffcc00" alt="Hugging Face base model" /></a>
  <a href="https://github.com/cobusgreyling/qwen-05b-min-sft/releases"><img src="https://img.shields.io/badge/adapter-LoRA%20release-green" alt="LoRA adapter release" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license" /></a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/cobusgreyling/qwen-05b-min-sft/blob/main/colab_qwen_05b_min_sft.ipynb">Open notebook</a> ·
  <a href="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct">Base model card</a> ·
  <a href="https://github.com/cobusgreyling/qwen-05b-min-sft/releases">Adapter</a> ·
  <a href="BLOG.md">Full write-up</a> ·
  <a href="outputs/RESULTS.md">Result table</a>
</p>

---

Stock [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) is a polite chat model. Ask it to handle a duplicate charge and you get an apology and a request for more details.

This lab teaches it a **rigid five-field ticket card** a downstream system can parse — using **eight labeled examples**, completion-only LoRA, and a frozen before/after test.

Same 6 frozen tickets, greedy decode (n=6 is a lab, not a benchmark):

| check | stock thin | stock + schema | 8-shot ICL | LoRA-8 | LoRA-16 |
|-------|-----------:|---------------:|-----------:|-------:|--------:|
| `<<TICKET>>` tags | 0 / 6 | 0 / 6 | **6 / 6** | **6 / 6** | **6 / 6** |
| `task_pass` | 0 / 6 | 0 / 6 | 2 / 6 | **4 / 6** | **6 / 6** |
| `note_facts_pass` | 0 / 6 | 0 / 6 | **6 / 6** | 5 / 6 | **6 / 6** |

Spelling the schema out in the prompt is not enough — 0.5B writes field-ish prose and still skips the wrapper. Eight-shot ICL gets the wrapper. Eight-card LoRA gets the **decisions**. Sixteen cards close the `CANCEL` / `QUOTA` tail. Full table: [`outputs/RESULTS.md`](outputs/RESULTS.md).

**Keywords:** Qwen2.5-0.5B-Instruct, open weights, least-data SFT, LoRA, TRL, PEFT, Colab T4, completion-only loss, supervised fine-tuning

---

## What this notebook is

[`colab_qwen_05b_min_sft.ipynb`](colab_qwen_05b_min_sft.ipynb) is a self-contained Colab lab. Cell 0 clones this repo and **imports `src/`** — the 8 / 4 / 6 cards live only in [`src/dataset.py`](src/dataset.py). Run all cells on a **free T4** (~10 minutes including the model download).

It answers one question: **how little labeled data does it take to change a small open-weight instruct model?**

Answer we actually ran: **8 training cards** beat a schema prompt and beat 8-shot ICL on `task_pass`. Frozen test the trainer never saw: tags **0 / 6 → 6 / 6**, task **0 / 6 → 4 / 6**. Sixteen cards, same recipe: **6 / 6**.

The intended path is the notebook. The `src/` scripts are the same recipe, runnable locally.

---

## 60-second start (Colab — recommended)

1. **[Open `colab_qwen_05b_min_sft.ipynb` in Colab](https://colab.research.google.com/github/cobusgreyling/qwen-05b-min-sft/blob/main/colab_qwen_05b_min_sft.ipynb)**
2. **Runtime → Change runtime type → GPU → T4**
3. **Runtime → Run all**

If the GitHub badge 404s in a fork, upload the `.ipynb` in Colab (**File → Upload notebook**) and pick a T4.

---

## What the notebook does

```
clone repo, import src/
     → write 8 / 4 / 6 cards
     → show the mask on the real Qwen tokenizer
     → score stock 0.5B three ways: thin / schema / 8-shot ICL
     → LoRA SFT for 40 steps (completion_only_loss=True)
     → score the adapter on the same 6 tickets
     → print the table + side-by-side
     → optional: type a new ticket
```

The 2 / 4 / 8 / 16 curve is `./run.sh curve` (four trains). Too long for the notebook.

---

## The idea in one page

### 1. A new output contract

The model is not being taught “be nicer.” It is being taught a format stock 0.5B will not emit:

```
<<TICKET>>
id: T-NNNN
verdict: REFUND | NO_REFUND | ESCALATE
amount_cents: <integer>
reason_code: DUPLICATE | FRAUD | POLICY | OTHER
note: one short sentence
<</TICKET>>
```

The system prompt used for SFT is deliberately thin:

```
You are DeskCard, a first-line support clerk.
Reply with one ticket card and nothing else.
```

It names the clerk. It does **not** spell out the five fields. If the LoRA learns `<<TICKET>>`, it learned it from the eight completions.

### 2. Least data that still covers the contract

| Split | Cards | Touches weights? |
|-------|------:|------------------|
| train | **8** | yes — two cards per family |
| hold-out | 4 | eval loss only |
| test | 6 | **no** — scored before and after |
| train extra | 8 | only the 16-point on the curve |

Families: `duplicate` · `fraud` · `policy` · `other`. Hold-out and test reuse the families with **new ids and wording**. Splits are by ticket id, so a test ticket cannot leak a train ticket.

### 3. Prompting is not a silent baseline

Same 6 tickets, no adapter:

| Condition | What we did | Tags | `task_pass` |
|-----------|-------------|-----:|------------:|
| stock thin | original one-line system prompt | 0 / 6 | 0 / 6 |
| stock + schema | card layout pasted into the system prompt | 0 / 6 | 0 / 6 |
| 8-shot ICL | the 8 train cards as few-shot turns | 6 / 6 | 2 / 6 |

0.5B *almost* follows a schema prompt (it writes `id:` / `verdict:` lines) and still skips `<<TICKET>>`. ICL copies the wrapper and then invents `verdict: REVERSAL` / refunds the asked amount. Gradient is doing something context is not.

### 4. Mask the prompt

The model **reads** the whole chat. It is **graded** only on the card.

```
INPUT:   [system] [user] [assistant card]
LABELS:  -100     -100   card token ids
```

TRL spelling: `SFTConfig(completion_only_loss=True)`.

On T-1042 with the real Qwen2.5 tokenizer: **131 tokens, 70 masked, 61 supervised.**

### 5. LoRA, not a full retrain

The 0.5B stays frozen. You train a few megabytes of adapters (`r=8`, `α=16`) on `q/k/v/o` and the MLP projections. You save `outputs/lora-sft/`, not a new 0.5B. Download the published adapter from [Releases](https://github.com/cobusgreyling/qwen-05b-min-sft/releases).

### 6. Improvement is a different assistant, not a lower loss

| Check | Means |
|-------|--------|
| `has_ticket_tags` | Both `<<TICKET>>` and `<</TICKET>>` are present |
| `format_pass` | Parseable card, allowed verdict / reason, no role leak |
| `task_pass` | Format **and** `id` / `verdict` / `amount_cents` / `reason_code` match gold |
| `note_facts_pass` | Integers in `note` all appear on the ticket |

`note` is not scored for wording. T-2003 on the 8-card adapter is the worked example: policy is right (`NO_REFUND` / `POLICY` / `0`) so `task_pass` holds, but the note writes `4900` (not on the ticket) so facts fail. Sixteen cards write `Asked 21000 exceeds invoice 3300`.

**User:** *Ticket T-2001. I was billed twice for invoice INV-2001 — forty-five dollars twice.*

**Before (stock):**

```
I'm sorry to hear that you're experiencing billing issues.
Could you please provide me with more details about the invoices…
```

**After (8-card LoRA):**

```
<<TICKET>>
id: T-2001
verdict: REFUND
amount_cents: 4500
reason_code: DUPLICATE
note: Duplicate invoice ID; refund the second charge.
<</TICKET>>
```

The two 8-card misses (`CANCEL`, `QUOTA` leaked into the enums) are the honest least-data signature: the wrapper transfers, the long tail does not. The extra eight cards in `TRAIN_EXTRA` target exactly those words. After 16-card LoRA both tickets pass.

Full story, token table, and per-ticket scores: [`BLOG.md`](BLOG.md).

---

## Local (optional)

Pinned stack (the one that produced the published numbers): `transformers==5.15.0`, `trl==1.10.0`, `peft==0.20.0`. See [`requirements.txt`](requirements.txt).

```bash
git clone https://github.com/cobusgreyling/qwen-05b-min-sft.git
cd qwen-05b-min-sft

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
chmod +x run.sh

./run.sh data          # write JSONL + unit tests
./run.sh mask          # tokenizer mask demo
./run.sh eval-base     # stock, thin prompt
./run.sh eval-schema   # stock, schema in the prompt
./run.sh eval-icl      # stock, 8-shot train cards
./run.sh train         # LoRA on the 8 cards
./run.sh eval          # adapter on frozen test
./run.sh compare       # print the table → outputs/RESULTS.md

# optional
./run.sh curve         # train+eval 2 / 4 / 8 / 16
./run.sh all           # data + three stock baselines + train + eval + compare
```

`outputs/*_eval.json` are the pictures. `./run.sh compare` is the table.

A Colab T4 run is about 10 minutes including download. The write-up’s local MPS run was **24.6 s** of 8-card training once the weights were cached.

---

## Layout

```
qwen-05b-min-sft/
├── assets/header.png                 # README / notebook hero
├── colab_qwen_05b_min_sft.ipynb      # ★ run this (imports src/)
├── BLOG.md                           # long-form explanation
├── MODEL_CARD.md                     # adapter card (also the HF README)
├── data/{train,holdout,test}.jsonl
├── data/curve/train_{2,4,8,16}.jsonl
├── src/{dataset,evaluate,compare,
│        train_sft,write_data,...}.py
└── outputs/{base,schema,icl,adapter}_eval.json
    └── RESULTS.md                    # last comparison table
```

---

## Why open weights matter here

You cannot attach LoRA to a closed chat API. [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) is **open weight** (Apache 2.0): you download the parameters, freeze them, and train a small adapter on top. That is the whole lab.

This repo is a teaching companion, not an official Qwen or Alibaba project.

---

## Related reading

- [`BLOG.md`](BLOG.md) — eight cards, the mask, the run, how to test improvement
- [`outputs/RESULTS.md`](outputs/RESULTS.md) — frozen-test table (prompt baselines + curve)
- [`MODEL_CARD.md`](MODEL_CARD.md) — adapter card
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — **base** weights, not this adapter
- [Qwen2.5 announcement](https://qwenlm.github.io/blog/qwen2.5/)
- [TRL SFTTrainer — completion-only loss](https://huggingface.co/docs/trl/en/sft_trainer#train-on-completion-only)

---

## License

MIT — see [`LICENSE`](LICENSE).

Synthetic DeskCard tickets are free to reuse for teaching. They are **not** a production refund policy.

Qwen2.5-0.5B-Instruct weights are Apache 2.0 (Qwen / Alibaba). Check the [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) before you ship a derivative.
