# Eight cards is enough to change Qwen 0.5B

**The data. How it was masked. A real LoRA run. How to test that the model actually improved.**

**Tags:** `SFT` · `LoRA` · `Qwen2.5-0.5B` · `Colab` · `least-data`

---

Stock `Qwen/Qwen2.5-0.5B-Instruct` is a polite clerk. Ask it to handle a duplicate charge and you get:

> I'm sorry to hear that you're experiencing billing issues. Could you please provide me with more details…

That is not a bug. It was trained to chat. We wanted a **ticket card** — a rigid five-field object a downstream system can parse.

The question for this lab: **how little labeled data does that take?**

Answer we actually ran: **8 training cards.** 40 LoRA steps. About 25 seconds on Apple MPS (a Colab T4 is in the same ballpark once the weights are cached). Frozen test, never seen by the trainer: **tags 0 / 6 → 6 / 6, task 0 / 6 → 4 / 6.** A schema prompt still scores 0 / 6. Eight-shot ICL gets the wrapper (6 / 6) and only 2 / 6 task. Sixteen cards, same recipe: **6 / 6**.

Companion: [`colab_qwen_05b_min_sft.ipynb`](colab_qwen_05b_min_sft.ipynb). Open it in Colab, **Runtime → GPU (T4)**, run all.

---

## 1. The training data

The whole corpus is 18 synthetic support tickets. Only **eight** of them update weights.

| Split | Cards | Job |
|-------|------:|-----|
| **train** | 8 | LoRA updates |
| **hold-out** | 4 | eval loss only |
| **test** | 6 | frozen — scored before and after, never loaded by the trainer |

We did not scrape production. We did not generate a teacher corpus. We wrote a **new output contract** the base model will not emit on its own, then wrote two examples per family so the schema is not a one-off.

### The contract

```
<<TICKET>>
id: T-NNNN
verdict: REFUND | NO_REFUND | ESCALATE
amount_cents: <integer>
reason_code: DUPLICATE | FRAUD | POLICY | OTHER
note: one short sentence
<</TICKET>>
```

The system prompt is deliberately thin:

```
You are DeskCard, a first-line support clerk.
Reply with one ticket card and nothing else.
```

It names the clerk. It does **not** spell out the five fields. If the model learns `<<TICKET>>`, it learned it from the eight completions, not from a page of instructions.

### The eight training cards

Two per family. New ids. Short, varied wording.

| Id | Family | Verdict | Amount | Reason |
|----|--------|---------|-------:|--------|
| T-1042 | duplicate | REFUND | 4999 | DUPLICATE |
| T-2108 | duplicate | REFUND | 7500 | DUPLICATE |
| T-3302 | fraud | ESCALATE | 0 | FRAUD |
| T-4410 | fraud | ESCALATE | 0 | FRAUD |
| T-8001 | policy | NO_REFUND | 0 | POLICY |
| T-5519 | policy | NO_REFUND | 0 | POLICY |
| T-6622 | other | NO_REFUND | 0 | OTHER |
| T-7730 | other | ESCALATE | 0 | OTHER |

One raw row (`T-1042`) looks like this:

```
system     You are DeskCard, a first-line support clerk. Reply with one
           ticket card and nothing else.
user       Ticket T-1042. Customer says they were charged twice for the
           same Pro seat. Invoice shows two $49.99 lines. Refund the
           extra charge.
assistant  <<TICKET>>
           id: T-1042
           verdict: REFUND
           amount_cents: 4999
           reason_code: DUPLICATE
           note: Duplicate Pro seat line; refund the extra 4999 cents.
           <</TICKET>>
```

Hold-out and test reuse the four families with **new ids and paraphrases** (`T-2001`…`T-2006` on test). Splitting by ticket id means a test ticket cannot leak a train ticket.

This set is a teaching baseline. It is not a production refund policy.

---

## 2. How it was masked

The model **reads** the whole chat. It is **scored** only on the card.

```
INPUT:   [system] [user] [assistant card]
LABELS:  -100     -100   card token ids
```

`-100` is ignored by cross-entropy. TRL spelling:

```python
SFTConfig(completion_only_loss=True)
```

That is the whole mask. Prompt-completion rows in, prompt labels out.

| Tokens | Mask? | Why |
|--------|-------|-----|
| System | yes | Do not recite “You are DeskCard…” |
| User | yes | Do not imitate the customer |
| Assistant card | **no** | This is the label |

On the T-1042 demo, with the real Qwen2.5 tokenizer: **131 tokens, 70 masked, 61 supervised.** The prefix of the full sequence matches the prompt exactly (`prompt prefix match: True`), so the cut is clean.

The boundary looks like this:

```
  i    tag   token
  67   MASK  <|im_start|>
  68   MASK  assistant
  69   MASK  \n
  70   LOSS  <<
  71   LOSS  T
  72   LOSS  ICK
  73   LOSS  ET
  74   LOSS  >>\n
  75   LOSS  id
  76   LOSS  :
  …
```

You are not hiding the ticket. You are not grading the model for predicting “Customer says they were charged twice.”

Notebook cell 3 prints this table on whatever tokenizer Colab downloads. Locally: `python src/demo_masking.py`.

---

## 3. The fine-tune

**Student:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Method:** LoRA, not full fine-tune. The 0.5B stays frozen; we train a few megabytes of adapters.

| Knob | Value |
|------|--------|
| Rank / α | 8 / 16 |
| Targets | `q_proj` `k_proj` `v_proj` `o_proj` `gate_proj` `up_proj` `down_proj` |
| Loss | completion only |
| Steps | 40 (~10 epochs on 8 cards) |
| Batch | 1 × grad accum 2 |
| LR | 2e-4, cosine |
| Max length | 384 |
| Device (this write-up) | Apple MPS, fp16, no 4-bit |
| Wall time | **24.6 s** |

Train loss started at **3.29** and collapsed. Mean train loss **0.614**. Token accuracy on the 8 cards hit 1.0 by the last few steps.

Hold-out eval loss went **0.72 → 0.88** between step 20 and 40. That is the honest small-data signature: the 8 cards got easy to copy; the 4 hold-out cards did not keep getting easier. We still publish the **frozen test generations**, not the loss curve.

The trainer never opened `test.jsonl`. That is deliberate.

---

## 4. How to test that the model improved

Do not look at train loss and declare victory. Loss going down means the eight cards got cheaper to imitate.

The test is:

1. Freeze six tickets the trainer has never seen.
2. Generate with **stock** 0.5B. Greedy decode. Temperature 0.
3. Generate with **0.5B + LoRA**. Same prompts. Same decoder.
4. Score both with the same rules.

### The scorer

| Check | Means |
|-------|--------|
| `has_ticket_tags` | Both `<<TICKET>>` and `<</TICKET>>` are present |
| `format_pass` | Parseable card, allowed verdict, allowed reason, no role leak |
| `task_pass` | Format **and** `id` / `verdict` / `amount_cents` / `reason_code` match gold |
| `note_facts_pass` | Every integer in `note` appears on the ticket |

`note` is not scored for wording. We want the wrapper and the decision, not a copied sentence. Facts catch the other failure: a correct policy card that invents `4900`.

Gold cards all pass this scorer before anyone trains. If they did not, the test would be broken.

### Before: stock 0.5B

| Check | Base |
|-------|-----:|
| `has_ticket_tags` | **0 / 6** |
| `format_pass` | **0%** |
| `task_pass` | **0%** |

Every reply is an apology or a request for more details. T-2006 even invents `T-2007` and `T-2008`. The thin system prompt is not enough. The model does not know the card.

### After: 0.5B + 8-card LoRA

| Check | Adapter |
|-------|--------:|
| `has_ticket_tags` | **6 / 6** |
| `id_match` | **6 / 6** |
| `verdict_match` | 5 / 6 |
| `reason_match` | 4 / 6 |
| `format_pass` | **67%** (4 / 6) |
| `task_pass` | **67%** (4 / 6) |
| `note_facts_pass` | **83%** (5 / 6) — T-2003 invented `4900` |

Same six user tickets:

| Ticket | Gold | After |
|--------|------|-------|
| T-2001 | REFUND / 4500 / DUPLICATE | **PASS** — refunded the extra $45 |
| T-2002 | ESCALATE / 0 / FRAUD | **PASS** — Jakarta vs Ohio |
| T-2003 | NO_REFUND / 0 / POLICY | **PASS** — 21000 on a 3300 invoice |
| T-2004 | NO_REFUND / 0 / OTHER | FAIL — invented `verdict: CANCEL`, refunded 6700 |
| T-2005 | ESCALATE / 0 / OTHER | FAIL — right verdict, invented `reason_code: QUOTA` |
| T-2006 | REFUND / 12000 / DUPLICATE | **PASS** — duplicate seat |

The two misses are useful. Eight examples taught the wrapper and the two “easy” families (duplicate, fraud, oversized refund). They did not lock the four-way `reason_code` enum when the user said “cancel” or “quota.” That is what least-data looks like: the shape transfers, the long tail does not.

### Side-by-side (T-2001)

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

That is the improvement. Not a lower loss. A different assistant.

---

## How to run the test yourself

Colab is the path. Upload `colab_qwen_05b_min_sft.ipynb`, set a T4, run all.

The notebook does the comparison in this order on purpose:

```
clone repo, import src/
     → write 8/4/6 cards
     → show the mask
     → score stock 0.5B three ways     # thin / schema / 8-shot ICL
     → LoRA on the 8 train cards
     → score adapter on the same 6     # AFTER
     → print the table + side-by-side
     → optional: type a new ticket
```

If you clone the repo locally:

```bash
git clone https://github.com/cobusgreyling/qwen-05b-min-sft.git
cd qwen-05b-min-sft
./run.sh data
./run.sh eval-base
./run.sh eval-schema
./run.sh eval-icl
./run.sh train
./run.sh eval
./run.sh compare
```

`outputs/RESULTS.md` is the table. The JSON files under `outputs/` are the pictures. Diff `summary`, then read `per_prompt[].generation`.

A ticket the trainer never saw is the last check. In the notebook:

```python
deskcard("Ticket T-9001. Charged twice for the same add-on, an extra 2800 cents.")
```

If SFT worked you get a card, not an essay.

---

## The run in one table

| Item | Value |
|------|--------|
| Train / hold-out / test | **8 / 4 / 6** cards |
| Mask | prompt 100%; completion 0% (T-1042: 70 / 61 tokens) |
| Student | Qwen2.5-0.5B-Instruct + LoRA r=8 |
| Steps / wall | 40 / 24.6 s (MPS) |
| Train loss | 3.29 → ~0.01 (mean 0.61) |
| Stock `task_pass` | **0 / 6** |
| Schema-prompt `task_pass` | **0 / 6** |
| 8-shot ICL `task_pass` | **2 / 6** |
| Adapter (8) `task_pass` | **4 / 6** |
| Adapter (16) `task_pass` | **6 / 6** |
| Adapter (8) `<<TICKET>>` tags | **6 / 6** |
| Adapter (8) `note_facts` | **5 / 6** |

---

## 5. Prompting first

The obvious objection: *you just needed a better prompt.*

Same 6 tickets. Same greedy decode. No adapter.

| Condition | Tags | `format_pass` | `task_pass` |
|-----------|-----:|--------------:|------------:|
| stock thin | 0 / 6 | 0 / 6 | 0 / 6 |
| stock + schema in the system prompt | 0 / 6 | 0 / 6 | 0 / 6 |
| 8-shot ICL (the train cards in context) | **6 / 6** | 5 / 6 | 2 / 6 |
| 8-card LoRA | **6 / 6** | 4 / 6 | **4 / 6** |

The schema prompt is not a silent win. 0.5B writes field-ish lines (`id: T-2001`, `verdict: REFUND`) and still never emits `<<TICKET>>`. Amounts come out wrong (`450` instead of `4500`, `-12000`, `10000` invented). Putting the contract in the prompt is not the same as teaching it.

Eight-shot ICL *does* copy the wrapper. Then it invents `verdict: REVERSAL` on T-2006, escalates a duplicate charge (T-2001), and refunds the asked `21000` / `6700` instead of writing `0`. The shape transfers. The policy does not.

That is why the 8-card LoRA is the interesting row: same thin prompt as stock, **4 / 6** task against ICL's **2 / 6**.

---

## 6. The data-size curve

Same recipe, more (or fewer) labeled cards. Ten epochs. Frozen test never loaded.

| n | Tags | `format_pass` | `task_pass` | `note_facts` |
|--:|-----:|--------------:|------------:|-------------:|
| 2 | 4 / 6 | 0 / 6 | 0 / 6 | 0 / 6 |
| 4 | 6 / 6 | 1 / 6 | 0 / 6 | 2 / 6 |
| 8 | 6 / 6 | 4 / 6 | 4 / 6 | 5 / 6 |
| 16 | 6 / 6 | 6 / 6 | 6 / 6 | 6 / 6 |

Two cards (one refund, one escalate) start the wrapper and do not lock the schema. Four cards — one per family — get `<<TICKET>>` on every test ticket and still leak `DUPLICATE_INVOICE`, `CANCEL`, `QUOTA_EXHAUSTED`. Eight cards are the first time `task_pass` moves. Sixteen cards add the long-tail paraphrases (`cancel my plan`, `quota is blown`, asked-vs-invoice numbers) and the two 8-card misses go away. T-2003's note becomes `Asked 21000 exceeds invoice 3300`.

`./run.sh curve` reproduces this. The notebook stays on the 8-card lab.

---

## Takeaways

1. **Least data** here means a *narrow contract*, not a smaller slice of the internet. Eight cards covering four families is enough to move a 0.5B model from prose to a parseable card — and enough to beat a schema prompt and 8-shot ICL on the decision.
2. **Mask the prompt.** Otherwise you spend gradient on “You are DeskCard” and the customer text.
3. **Test with the same prompts, before and after.** Loss is a training heartbeat. Improvement is “did the assistant change on tickets the optimizer never saw.”
4. **Score the note for invented numbers.** T-2003 `task_pass`es with a swapped 3300 / 4900. That is a scorer hole unless you name it (`note_facts_pass`).
5. Eight examples will not teach a full policy. `CANCEL` and `QUOTA` leaked in because those words were in the user text and not in the enum. Add two cards in that family if that is the next failure you care about — that is the 16-card point on the curve.

This is SFT, not a general clerk. Replace the synthetic cards before you claim a desk.
