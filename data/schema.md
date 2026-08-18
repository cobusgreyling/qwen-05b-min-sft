# DeskCard schema

One assistant completion is one ticket card. Nothing else.

```
<<TICKET>>
id: T-NNNN
verdict: REFUND | NO_REFUND | ESCALATE
amount_cents: <integer>
reason_code: DUPLICATE | FRAUD | POLICY | OTHER
note: <one short sentence>
<</TICKET>>
```

| Field | Rule |
|-------|------|
| `id` | Copy the ticket id from the user text |
| `verdict` | One of the three labels |
| `amount_cents` | Extra charge to refund, else `0` |
| `reason_code` | One of the four labels |
| `note` | One sentence; not scored for wording |

The thin system prompt only names the clerk. The card layout lives in the eight training completions.

`note` is not scored for wording. It **is** scored for invented numbers (`note_facts_pass`): every integer in the note must appear on the ticket or in `facts.amounts_cents`. T-2003 is the worked example — a correct `NO_REFUND` / `POLICY` card that writes `4900` fails facts.

`data/train_extra.jsonl` and `data/curve/train_{2,4,8,16}.jsonl` are the data-size curve. They do not replace the 8-card lab.
