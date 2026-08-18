"""
The entire corpus: 8 train + 8 extra (curve) + 4 hold-out + 6 frozen test.

Least-data thesis
-----------------
Teach a *new output contract* the base model will not emit on its own.
The thin system prompt names the clerk. The card schema lives only in
the eight training completions. Extra cards are for the 2/4/8/16 curve
only — they never replace the 8-card lab.
"""

from __future__ import annotations

from typing import Any

SYSTEM = (
    "You are DeskCard, a first-line support clerk. "
    "Reply with one ticket card and nothing else."
)

# Used only for the prompt-baseline. SFT still trains on SYSTEM.
SCHEMA_SYSTEM = (
    "You are DeskCard, a first-line support clerk. "
    "Reply with one ticket card and nothing else.\n"
    "\n"
    "The card must be exactly this shape:\n"
    "<<TICKET>>\n"
    "id: T-NNNN\n"
    "verdict: REFUND | NO_REFUND | ESCALATE\n"
    "amount_cents: <integer>\n"
    "reason_code: DUPLICATE | FRAUD | POLICY | OTHER\n"
    "note: one short sentence\n"
    "<</TICKET>>\n"
    "\n"
    "Use only those verdict and reason_code values. "
    "Copy the ticket id from the user. "
    "amount_cents is the extra charge to refund, or 0. "
    "Words like cancel or quota in the ticket are not labels."
)

VERDICTS = ("REFUND", "NO_REFUND", "ESCALATE")
REASONS = ("DUPLICATE", "FRAUD", "POLICY", "OTHER")

# Eight training cards. Two per family so the schema is not a one-off.
# IDs, amounts, and wording do not overlap the frozen test set.
# facts.amounts_cents = numbers that may honestly appear in the note.
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
        "facts": {"amounts_cents": [4999]},
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
        "facts": {"amounts_cents": [7500]},
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
        "facts": {"amounts_cents": []},
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
        "facts": {"amounts_cents": [40]},
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
        "facts": {"amounts_cents": [9000, 2000]},
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
        "facts": {"amounts_cents": [11, 30]},
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
        "facts": {"amounts_cents": []},
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
        "facts": {"amounts_cents": [500]},
    },
]

# Extra eight cards for the 16-point on the data-size curve.
# They teach the long tail the 8-card run missed: "cancel" is not a
# verdict, "quota" is not a reason_code, and notes must not invent amounts.
TRAIN_EXTRA: list[dict[str, Any]] = [
    {
        "id": "T-3101",
        "split": "train_extra",
        "family": "other",
        "user": (
            "Ticket T-3101. The new logo is ugly. Cancel my plan and "
            "refund 4100 cents. Nothing is broken."
        ),
        "gold": {
            "id": "T-3101",
            "verdict": "NO_REFUND",
            "amount_cents": 0,
            "reason_code": "OTHER",
        },
        "note": "Taste complaint is not a defect; no refund.",
        "facts": {"amounts_cents": [4100]},
    },
    {
        "id": "T-3102",
        "split": "train_extra",
        "family": "other",
        "user": (
            "Ticket T-3102. Please cancel the subscription because they hate "
            "the new font. Refund 8800 cents. The product works."
        ),
        "gold": {
            "id": "T-3102",
            "verdict": "NO_REFUND",
            "amount_cents": 0,
            "reason_code": "OTHER",
        },
        "note": "Branding preference is not a billing defect; no refund.",
        "facts": {"amounts_cents": [8800]},
    },
    {
        "id": "T-3103",
        "split": "train_extra",
        "family": "other",
        "user": (
            "Ticket T-3103. Integration webhooks keep retrying and the partner "
            "quota is blown. Status dashboard is empty. Need a human."
        ),
        "gold": {
            "id": "T-3103",
            "verdict": "ESCALATE",
            "amount_cents": 0,
            "reason_code": "OTHER",
        },
        "note": "Webhook storm plus empty status page; escalate.",
        "facts": {"amounts_cents": []},
    },
    {
        "id": "T-3104",
        "split": "train_extra",
        "family": "other",
        "user": (
            "Ticket T-3104. API rate limit is stuck at zero after a retry "
            "storm. Customer cannot ship. Need engineering."
        ),
        "gold": {
            "id": "T-3104",
            "verdict": "ESCALATE",
            "amount_cents": 0,
            "reason_code": "OTHER",
        },
        "note": "Retry storm exhausted the API; escalate to engineering.",
        "facts": {"amounts_cents": [0]},
    },
    {
        "id": "T-3105",
        "split": "train_extra",
        "family": "policy",
        "user": (
            "Ticket T-3105. Refund 15000 cents immediately. The invoice "
            "is only 4000 cents but the customer is yelling."
        ),
        "gold": {
            "id": "T-3105",
            "verdict": "NO_REFUND",
            "amount_cents": 0,
            "reason_code": "POLICY",
        },
        "note": "Asked 15000 exceeds invoice 4000; refuse under policy.",
        "facts": {"amounts_cents": [15000, 4000]},
    },
    {
        "id": "T-3106",
        "split": "train_extra",
        "family": "duplicate",
        "user": (
            "Ticket T-3106. Billed twice for the same add-on — thirty "
            "dollars twice. Reverse the extra charge."
        ),
        "gold": {
            "id": "T-3106",
            "verdict": "REFUND",
            "amount_cents": 3000,
            "reason_code": "DUPLICATE",
        },
        "note": "Duplicate add-on; refund the extra 3000 cents.",
        "facts": {"amounts_cents": [3000, 30]},
    },
    {
        "id": "T-3107",
        "split": "train_extra",
        "family": "fraud",
        "user": (
            "Ticket T-3107. Card just paid for a phone in Lagos. Customer "
            "has been in Denver all month and did not buy it."
        ),
        "gold": {
            "id": "T-3107",
            "verdict": "ESCALATE",
            "amount_cents": 0,
            "reason_code": "FRAUD",
        },
        "note": "Unauthorized overseas charge; escalate to fraud.",
        "facts": {"amounts_cents": []},
    },
    {
        "id": "T-3108",
        "split": "train_extra",
        "family": "other",
        "user": (
            "Ticket T-3108. Customer typed CANCEL in all caps. They want "
            "out because onboarding emails are too frequent. Service works. "
            "Refund 2200 cents."
        ),
        "gold": {
            "id": "T-3108",
            "verdict": "NO_REFUND",
            "amount_cents": 0,
            "reason_code": "OTHER",
        },
        "note": "Email volume is not a billing defect; no refund.",
        "facts": {"amounts_cents": [2200]},
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
        "facts": {"amounts_cents": [3200]},
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
        "facts": {"amounts_cents": [2, 10]},
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
        "facts": {"amounts_cents": [99999, 1250]},
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
        "facts": {"amounts_cents": [1, 16600]},
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
        "facts": {"amounts_cents": [4500, 2001]},
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
        "facts": {"amounts_cents": []},
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
        "facts": {"amounts_cents": [21000, 3300]},
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
        "facts": {"amounts_cents": [6700]},
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
        "facts": {"amounts_cents": []},
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
        "facts": {"amounts_cents": [12000]},
    },
]

# Data-size curve: same recipe, more (or fewer) labeled cards.
# n=2 / n=4 keep family coverage as wide as the budget allows.
CURVE_IDS: dict[int, tuple[str, ...]] = {
    2: ("T-1042", "T-3302"),
    4: ("T-1042", "T-3302", "T-8001", "T-6622"),
    8: tuple(ex["id"] for ex in TRAIN),
    16: tuple(ex["id"] for ex in TRAIN + TRAIN_EXTRA),
}


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
        "user": example["user"],
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": example["user"]},
        ],
        "completion": [{"role": "assistant", "content": card_text(example)}],
        "gold": example["gold"],
        "facts": example.get("facts") or {"amounts_cents": []},
    }


def all_examples() -> list[dict[str, Any]]:
    return [*TRAIN, *TRAIN_EXTRA, *HOLDOUT, *TEST]


def example_by_id(ticket_id: str) -> dict[str, Any]:
    for ex in all_examples():
        if ex["id"] == ticket_id:
            return ex
    raise KeyError(ticket_id)


def rows_for(split: str) -> list[dict[str, Any]]:
    return [to_sft_row(ex) for ex in all_examples() if ex["split"] == split]


def curve_examples(n: int) -> list[dict[str, Any]]:
    if n not in CURVE_IDS:
        raise ValueError(f"curve n must be one of {sorted(CURVE_IDS)}, got {n}")
    wanted = set(CURVE_IDS[n])
    found = [ex for ex in TRAIN + TRAIN_EXTRA if ex["id"] in wanted]
    if len(found) != n:
        raise RuntimeError(f"curve {n} resolved {len(found)} cards")
    order = {tid: i for i, tid in enumerate(CURVE_IDS[n])}
    return sorted(found, key=lambda ex: order[ex["id"]])


def curve_rows(n: int) -> list[dict[str, Any]]:
    return [to_sft_row(ex) for ex in curve_examples(n)]
