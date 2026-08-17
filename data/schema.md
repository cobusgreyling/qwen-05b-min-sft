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

The system prompt only names the clerk. The card layout lives in the eight training completions.
