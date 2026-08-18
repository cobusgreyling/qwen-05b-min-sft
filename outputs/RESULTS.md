# Frozen-test results

Same **6** tickets. Greedy decode. Same scorer.

- **stock thin** — base 0.5B, original one-line system prompt
- **stock+schema** — base 0.5B, card layout spelled out in the prompt (no SFT)
- **8-shot ICL** — base 0.5B, the 8 train cards in context (no SFT)
- **LoRA-k** — completion-only LoRA trained on *k* cards, thin prompt. LoRA-8 is the published adapter.

| check | stock thin | stock+schema | 8-shot ICL | LoRA-2 | LoRA-4 | LoRA-8 | LoRA-16 |
|-------|------:|------:|------:|------:|------:|------:|------:|
| `has_ticket_tags` |    0% (0/6) |    0% (0/6) |  100% (6/6) |   67% (4/6) |  100% (6/6) |  100% (6/6) |  100% (6/6) |
| `format_pass` |    0% (0/6) |    0% (0/6) |   83% (5/6) |    0% (0/6) |   17% (1/6) |   67% (4/6) |  100% (6/6) |
| `task_pass` |    0% (0/6) |    0% (0/6) |   33% (2/6) |    0% (0/6) |    0% (0/6) |   67% (4/6) |  100% (6/6) |
| `note_facts_pass` |    0% (0/6) |    0% (0/6) |  100% (6/6) |    0% (0/6) |   33% (2/6) |   83% (5/6) |  100% (6/6) |

<details><summary>All checks</summary>

| check | stock thin | stock+schema | 8-shot ICL | LoRA-2 | LoRA-4 | LoRA-8 | LoRA-16 |
|-------|------:|------:|------:|------:|------:|------:|------:|
| `non_empty` |  100% |  100% |  100% |  100% |  100% |  100% |  100% |
| `no_role_leak` |  100% |  100% |  100% |  100% |  100% |  100% |  100% |
| `has_ticket_tags` |    0% (0/6) |    0% (0/6) |  100% (6/6) |   67% (4/6) |  100% (6/6) |  100% (6/6) |  100% (6/6) |
| `parsed_ok` |    0% |    0% |  100% |    0% |   50% |  100% |  100% |
| `id_match` |    0% |    0% |  100% |    0% |   50% |  100% |  100% |
| `verdict_match` |    0% |    0% |   67% |    0% |   33% |   83% |  100% |
| `amount_match` |    0% |    0% |   67% |    0% |   50% |   83% |  100% |
| `reason_match` |    0% |    0% |  100% |    0% |    0% |   67% |  100% |
| `format_pass` |    0% (0/6) |    0% (0/6) |   83% (5/6) |    0% (0/6) |   17% (1/6) |   67% (4/6) |  100% (6/6) |
| `task_pass` |    0% (0/6) |    0% (0/6) |   33% (2/6) |    0% (0/6) |    0% (0/6) |   67% (4/6) |  100% (6/6) |
| `note_facts_pass` |    0% (0/6) |    0% (0/6) |  100% (6/6) |    0% (0/6) |   33% (2/6) |   83% (5/6) |  100% (6/6) |

</details>

## Per ticket (`task_pass` / wrapper only / fail)

| ticket | stock thin | stock+schema | 8-shot ICL | LoRA-2 | LoRA-4 | LoRA-8 | LoRA-16 |
|--------|------|------|------|------|------|------|------|
| `T-2001` | FAIL | FAIL | wrap | wrap | wrap | PASS | PASS |
| `T-2002` | FAIL | FAIL | PASS | wrap | wrap | PASS | PASS |
| `T-2003` | FAIL | FAIL | wrap | wrap | wrap | PASS | PASS |
| `T-2004` | FAIL | FAIL | wrap | wrap | wrap | wrap | PASS |
| `T-2005` | FAIL | FAIL | PASS | FAIL | wrap | wrap | PASS |
| `T-2006` | FAIL | FAIL | wrap | FAIL | wrap | PASS | PASS |

`wrap` means `<<TICKET>>` tags landed but the decision or enum was wrong.

Regenerate: `./run.sh compare`
