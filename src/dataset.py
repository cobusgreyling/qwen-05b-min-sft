"""
The entire corpus: 8 train + 4 hold-out + 6 frozen test.

Least-data thesis
-----------------
Teach a *new output contract* the base model will not emit on its own.
The system prompt names the clerk. The card schema lives only in the
completions. Eight labeled cards is the training set.
"""

from __future__ import annotations

from typing import Any

SYSTEM = (
    "You are DeskCard, a first-line support clerk. "
    "Reply with one ticket card and nothing else."
)

VERDICTS = ("REFUND", "NO_REFUND", "ESCALATE")
REASONS = ("DUPLICATE", "FRAUD", "POLICY", "OTHER")

# Eight training cards. Two per family so the schema is not a one-off.
# IDs, amounts, and wording do not overlap the frozen test set.
TRAIN: list[dict[str, Any]] = [
    {
        "id": "T-1042",
        "split": "train",
        "family": "duplicate",
        "user": (
            "Ticket T-1042. Customer says they were charged twice for the same "
            "Pro seat. Invoice shows two $49.99 lines. Refund the extra charge."
        ),
        "gold": {
            "id": "T-1042",
            "verdict": "REFUND",
            "amount_cents": 4999,
            "reason_code": "DUPLICATE",
        },
        "note": "Duplicate Pro seat line; refund the extra 4999 cents.",
    },
    {
        "id": "T-2108",
        "split": "train",
        "family": "duplicate",
        "user": (
            "Ticket T-2108. Billing listed the annual plan twice. Customer "
            "wants the second 7500-cent charge reversed."
        ),
        "gold": {
            "id": "T-2108",
            "verdict": "REFUND",
            "amount_cents": 7500,
            "reason_code": "DUPLICATE",
        },
        "note": "Annual plan billed twice; refund the second 7500 cents.",
    },
    {
        "id": "T-3302",
        "split": "train",
        "family": "fraud",
        "user": (
            "Ticket T-3302. Customer was in Ohio all week. Card just paid "
            "for a laptop in Jakarta. They did not authorize it."
        ),
        "gold": {
            "id": "T-3302",
            "verdict": "ESCALATE",
            "amount_cents": 0,
            "reason_code": "FRAUD",
        },
        "note": "Unauthorized foreign charge; escalate to fraud review.",
    },
    {
        "id": "T-4410",
        "split": "train",
        "family": "fraud",
        "user": (
            "Ticket T-4410. Forty failed logins overnight, then a password "
            "reset from a new device. Customer says that was not them."
        ),
        "gold": {
            "id": "T-4410",
            "verdict": "ESCALATE",
            "amount_cents": 0,
            "reason_code": "FRAUD",
        },
        "note": "Credential-stuffing pattern; escalate, do not refund here.",
    },
    {
        "id": "T-8001",
        "split": "train",
        "family": "policy",
        "user": (
            "Ticket T-8001. Angry customer wants a 9000-cent refund right now. "
            "The invoice total is only 2000 cents."
        ),
        "gold": {
            "id": "T-8001",
            "verdict": "NO_REFUND",
            "amount_cents": 0,
            "reason_code": "POLICY",
        },
        "note": "Asked amount exceeds invoice total; refuse under policy.",
    },
    {
        "id": "T-5519",
        "split": "train",
        "family": "policy",
        "user": (
            "Ticket T-5519. Purchase was 11 months ago. Customer wants a "
            "full refund. Our window is 30 days."
        ),
        "gold": {
            "id": "T-5519",
            "verdict": "NO_REFUND",
            "amount_cents": 0,
            "reason_code": "POLICY",
        },
        "note": "Outside the 30-day refund window; no refund.",
    },
    {
        "id": "T-6622",
        "split": "train",
        "family": "other",
        "user": (
            "Ticket T-6622. Customer wants a refund because they do not like "
            "the new UI color. No billing error, no outage."
        ),
        "gold": {
            "id": "T-6622",
            "verdict": "NO_REFUND",
            "amount_cents": 0,
            "reason_code": "OTHER",
        },
        "note": "Taste complaint is not a billing defect; no refund.",
    },
    {
        "id": "T-7730",
        "split": "train",
        "family": "other",
        "user": (
            "Ticket T-7730. VIP cannot download their invoice PDF. The billing "
            "page returns HTTP 500. They need a human."
        ),
        "gold": {
            "id": "T-7730",
            "verdict": "ESCALATE",
            "amount_cents": 0,
            "reason_code": "OTHER",
        },
        "note": "Broken invoice download for a VIP; escalate to engineering.",
    },
]

# Hold-out: same families, different IDs. Used for eval loss only.
HOLDOUT: list[dict[str, Any]] = [
    {
        "id": "T-8844",
        "split": "holdout",
        "family": "duplicate",
        "user": (
            "Ticket T-8844. Two identical 3200-cent charges landed on the "
            "same invoice. Please reverse the extra one."
        ),
        "gold": {
            "id": "T-8844",
            "verdict": "REFUND",
            "amount_cents": 3200,
            "reason_code": "DUPLICATE",
        },
        "note": "Identical 3200-cent lines; refund the extra charge.",
    },
    {
        "id": "T-9901",
        "split": "holdout",
        "family": "fraud",
        "user": (
            "Ticket T-9901. MFA was bypassed and the account signed in from "
            "two countries in ten minutes. Customer is locked out."
        ),
        "gold": {
            "id": "T-9901",
            "verdict": "ESCALATE",
            "amount_cents": 0,
            "reason_code": "FRAUD",
        },
        "note": "Impossible travel plus MFA bypass; escalate to security.",
    },
    {
        "id": "T-1015",
        "split": "holdout",
        "family": "policy",
        "user": (
            "Ticket T-1015. Customer demands a 99999-cent refund. Invoice "
            "total is 1250 cents."
        ),
        "gold": {
            "id": "T-1015",
            "verdict": "NO_REFUND",
            "amount_cents": 0,
            "reason_code": "POLICY",
        },
        "note": "Refund request larger than the invoice; refuse.",
    },
    {
        "id": "T-1120",
        "split": "holdout",
        "family": "other",
        "user": (
            "Ticket T-1120. Package arrived one day late. Customer wants "
            "the full 16600 cents back. Delivery itself succeeded."
        ),
        "gold": {
            "id": "T-1120",
            "verdict": "NO_REFUND",
            "amount_cents": 0,
            "reason_code": "OTHER",
        },
        "note": "One-day late delivery is not a billing defect; no refund.",
    },
]

# Frozen test. Trainer never opens these. Same schema, new IDs and wording.
TEST: list[dict[str, Any]] = [
    {
        "id": "T-2001",
        "split": "test",
        "family": "duplicate",
        "user": (
            "Ticket T-2001. I was billed twice for invoice INV-2001 — "
            "forty-five dollars twice. Please fix the extra charge."
        ),
        "gold": {
            "id": "T-2001",
            "verdict": "REFUND",
            "amount_cents": 4500,
            "reason_code": "DUPLICATE",
        },
        "note": "Double 4500-cent charge; refund the extra line.",
    },
    {
        "id": "T-2002",
        "split": "test",
        "family": "fraud",
        "user": (
            "Ticket T-2002. Someone used my card in Jakarta this morning. "
            "I have been in Ohio the whole time and I did not buy anything."
        ),
        "gold": {
            "id": "T-2002",
            "verdict": "ESCALATE",
            "amount_cents": 0,
            "reason_code": "FRAUD",
        },
        "note": "Unauthorized overseas charge; escalate to fraud.",
    },
    {
        "id": "T-2003",
        "split": "test",
        "family": "policy",
        "user": (
            "Ticket T-2003. Refund 21000 cents immediately. The invoice "
            "is only 3300 cents but the customer is yelling."
        ),
        "gold": {
            "id": "T-2003",
            "verdict": "NO_REFUND",
            "amount_cents": 0,
            "reason_code": "POLICY",
        },
        "note": "Requested refund exceeds invoice total; refuse.",
    },
    {
        "id": "T-2004",
        "split": "test",
        "family": "other",
        "user": (
            "Ticket T-2004. The new icon is ugly. Cancel my plan and "
            "refund 6700 cents. Nothing is broken."
        ),
        "gold": {
            "id": "T-2004",
            "verdict": "NO_REFUND",
            "amount_cents": 0,
            "reason_code": "OTHER",
        },
        "note": "Aesthetic preference is not a defect; no refund.",
    },
    {
        "id": "T-2005",
        "split": "test",
        "family": "other",
        "user": (
            "Ticket T-2005. Partner webhooks are retrying in a loop and "
            "their quota is exhausted. The status page is blank. Need a human."
        ),
        "gold": {
            "id": "T-2005",
            "verdict": "ESCALATE",
            "amount_cents": 0,
            "reason_code": "OTHER",
        },
        "note": "Webhook storm plus blank status page; escalate.",
    },
    {
        "id": "T-2006",
        "split": "test",
        "family": "duplicate",
        "user": (
            "Ticket T-2006. We were charged for a duplicate seat license — "
            "an extra 12000 cents on this cycle. Reverse that seat only."
        ),
        "gold": {
            "id": "T-2006",
            "verdict": "REFUND",
            "amount_cents": 12000,
            "reason_code": "DUPLICATE",
        },
        "note": "Duplicate seat license; refund the extra 12000 cents.",
    },
]


def card_text(example: dict[str, Any]) -> str:
    g = example["gold"]
    return (
        "<<TICKET>>\n"
        f"id: {g['id']}\n"
        f"verdict: {g['verdict']}\n"
        f"amount_cents: {g['amount_cents']}\n"
        f"reason_code: {g['reason_code']}\n"
        f"note: {example['note']}\n"
        "<</TICKET>>"
    )


def to_sft_row(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": example["id"],
        "family": example["family"],
        "split": example["split"],
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": example["user"]},
        ],
        "completion": [{"role": "assistant", "content": card_text(example)}],
        "gold": example["gold"],
    }


def all_examples() -> list[dict[str, Any]]:
    return [*TRAIN, *HOLDOUT, *TEST]


def rows_for(split: str) -> list[dict[str, Any]]:
    return [to_sft_row(ex) for ex in all_examples() if ex["split"] == split]
