#!/usr/bin/env python3
"""CPU tests for data, mask, and scorer. No GPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import HOLDOUT, SYSTEM, TEST, TRAIN, card_text, rows_for, to_sft_row  # noqa: E402
from evaluate import parse_card, score_generation  # noqa: E402


class TestCounts(unittest.TestCase):
    def test_least_data_budget(self) -> None:
        self.assertEqual(len(TRAIN), 8)
        self.assertEqual(len(HOLDOUT), 4)
        self.assertEqual(len(TEST), 6)

    def test_no_id_leakage(self) -> None:
        train_ids = {ex["id"] for ex in TRAIN}
        hold_ids = {ex["id"] for ex in HOLDOUT}
        test_ids = {ex["id"] for ex in TEST}
        self.assertFalse(train_ids & hold_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(hold_ids & test_ids)

    def test_families_in_train(self) -> None:
        families = {ex["family"] for ex in TRAIN}
        self.assertEqual(families, {"duplicate", "fraud", "policy", "other"})


class TestCard(unittest.TestCase):
    def test_gold_cards_parse_and_pass(self) -> None:
        for ex in TRAIN + HOLDOUT + TEST:
            text = card_text(ex)
            parsed = parse_card(text)
            self.assertIsNotNone(parsed, ex["id"])
            s = score_generation(text, ex["gold"])
            self.assertTrue(s["task_pass"], ex["id"])

    def test_prose_fails_format(self) -> None:
        gold = TRAIN[0]["gold"]
        s = score_generation("Sure, I can refund that duplicate charge for you.", gold)
        self.assertFalse(s["format_pass"])
        self.assertFalse(s["task_pass"])

    def test_wrong_verdict_fails_task(self) -> None:
        gold = {"id": "T-1042", "verdict": "REFUND", "amount_cents": 4999, "reason_code": "DUPLICATE"}
        text = (
            "<<TICKET>>\n"
            "id: T-1042\n"
            "verdict: NO_REFUND\n"
            "amount_cents: 4999\n"
            "reason_code: DUPLICATE\n"
            "note: nope.\n"
            "<</TICKET>>"
        )
        s = score_generation(text, gold)
        self.assertTrue(s["format_pass"])
        self.assertFalse(s["task_pass"])


class TestRows(unittest.TestCase):
    def test_sft_shape(self) -> None:
        row = to_sft_row(TRAIN[0])
        self.assertEqual(row["prompt"][0]["content"], SYSTEM)
        self.assertEqual(row["completion"][0]["role"], "assistant")
        self.assertIn("<<TICKET>>", row["completion"][0]["content"])
        self.assertEqual(len(rows_for("train")), 8)
        self.assertEqual(len(rows_for("test")), 6)


if __name__ == "__main__":
    unittest.main()
