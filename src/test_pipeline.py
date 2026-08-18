#!/usr/bin/env python3
"""CPU tests for data, mask, scorer, baselines, and the curve. No GPU."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (  # noqa: E402
    CURVE_IDS,
    HOLDOUT,
    SCHEMA_SYSTEM,
    SYSTEM,
    TEST,
    TRAIN,
    TRAIN_EXTRA,
    all_examples,
    card_text,
    curve_examples,
    example_by_id,
    rows_for,
    to_sft_row,
)
from evaluate import as_eval_example, messages_for, parse_card, rescore_report, score_generation  # noqa: E402


class TestCounts(unittest.TestCase):
    def test_least_data_budget(self) -> None:
        self.assertEqual(len(TRAIN), 8)
        self.assertEqual(len(TRAIN_EXTRA), 8)
        self.assertEqual(len(HOLDOUT), 4)
        self.assertEqual(len(TEST), 6)

    def test_no_id_leakage(self) -> None:
        groups = [
            {ex["id"] for ex in TRAIN},
            {ex["id"] for ex in TRAIN_EXTRA},
            {ex["id"] for ex in HOLDOUT},
            {ex["id"] for ex in TEST},
        ]
        for i, a in enumerate(groups):
            for b in groups[i + 1 :]:
                self.assertFalse(a & b, f"id leak {a & b}")

    def test_families_in_train(self) -> None:
        families = {ex["family"] for ex in TRAIN}
        self.assertEqual(families, {"duplicate", "fraud", "policy", "other"})

    def test_every_example_has_facts(self) -> None:
        for ex in all_examples():
            self.assertIn("facts", ex, ex["id"])
            self.assertIn("amounts_cents", ex["facts"], ex["id"])


class TestCard(unittest.TestCase):
    def test_gold_cards_parse_and_pass(self) -> None:
        for ex in all_examples():
            text = card_text(ex)
            parsed = parse_card(text)
            self.assertIsNotNone(parsed, ex["id"])
            s = score_generation(text, ex["gold"], facts=ex.get("facts"))
            self.assertTrue(s["task_pass"], ex["id"])
            self.assertTrue(s["note_facts_pass"], ex["id"])

    def test_prose_fails_format(self) -> None:
        gold = TRAIN[0]["gold"]
        s = score_generation("Sure, I can refund that duplicate charge for you.", gold)
        self.assertFalse(s["format_pass"])
        self.assertFalse(s["task_pass"])
        self.assertFalse(s["note_facts_pass"])

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

    def test_invented_amount_fails_note_facts(self) -> None:
        ex = example_by_id("T-2003")
        text = (
            "<<TICKET>>\n"
            "id: T-2003\n"
            "verdict: NO_REFUND\n"
            "amount_cents: 0\n"
            "reason_code: POLICY\n"
            "note: Asked amount (3300) exceeds invoice (4900), refuse under policy.\n"
            "<</TICKET>>"
        )
        s = score_generation(text, ex["gold"], facts=ex["facts"])
        self.assertTrue(s["task_pass"], "policy decision is still correct")
        self.assertFalse(s["note_facts_pass"], "4900 is not on the ticket")

    def test_honest_numbers_pass_note_facts(self) -> None:
        ex = example_by_id("T-2003")
        text = (
            "<<TICKET>>\n"
            "id: T-2003\n"
            "verdict: NO_REFUND\n"
            "amount_cents: 0\n"
            "reason_code: POLICY\n"
            "note: Asked 21000 exceeds invoice 3300; refuse.\n"
            "<</TICKET>>"
        )
        s = score_generation(text, ex["gold"], facts=ex["facts"])
        self.assertTrue(s["task_pass"])
        self.assertTrue(s["note_facts_pass"])


class TestRows(unittest.TestCase):
    def test_sft_shape(self) -> None:
        row = to_sft_row(TRAIN[0])
        self.assertEqual(row["prompt"][0]["content"], SYSTEM)
        self.assertEqual(row["completion"][0]["role"], "assistant")
        self.assertIn("<<TICKET>>", row["completion"][0]["content"])
        self.assertEqual(row["user"], TRAIN[0]["user"])
        self.assertIn("facts", row)
        self.assertEqual(len(rows_for("train")), 8)
        self.assertEqual(len(rows_for("test")), 6)


class TestPrompts(unittest.TestCase):
    def test_schema_prompt_is_not_the_sft_prompt(self) -> None:
        self.assertNotEqual(SYSTEM, SCHEMA_SYSTEM)
        self.assertIn("<<TICKET>>", SCHEMA_SYSTEM)
        self.assertNotIn("<<TICKET>>", SYSTEM)

    def test_icl_contains_train_cards_not_test(self) -> None:
        ex = as_eval_example(TEST[0])
        messages = messages_for(ex, mode="icl")
        self.assertEqual(messages[0]["content"], SYSTEM)
        assistant_turns = [m for m in messages if m["role"] == "assistant"]
        self.assertEqual(len(assistant_turns), 8)
        blob = "\n".join(m["content"] for m in messages[:-1])
        for train in TRAIN:
            self.assertIn(train["id"], blob)
        self.assertNotIn(TEST[0]["id"], blob)
        self.assertEqual(messages[-1]["content"], TEST[0]["user"])

    def test_schema_mode_uses_schema_system(self) -> None:
        ex = as_eval_example(TEST[0])
        messages = messages_for(ex, mode="schema")
        self.assertEqual(messages[0]["content"], SCHEMA_SYSTEM)
        self.assertEqual(len(messages), 2)


class TestCurve(unittest.TestCase):
    def test_curve_sizes_and_coverage(self) -> None:
        self.assertEqual(set(CURVE_IDS), {2, 4, 8, 16})
        two = curve_examples(2)
        four = curve_examples(4)
        eight = curve_examples(8)
        sixteen = curve_examples(16)
        self.assertEqual(len(two), 2)
        self.assertEqual(len(four), 4)
        self.assertEqual(len(eight), 8)
        self.assertEqual(len(sixteen), 16)
        self.assertTrue({ex["id"] for ex in two} <= {ex["id"] for ex in four})
        self.assertTrue({ex["id"] for ex in four} <= {ex["id"] for ex in eight})
        self.assertTrue({ex["id"] for ex in eight} <= {ex["id"] for ex in sixteen})
        self.assertEqual({ex["family"] for ex in four}, {"duplicate", "fraud", "policy", "other"})
        extra_ids = {ex["id"] for ex in TRAIN_EXTRA}
        self.assertTrue(extra_ids <= {ex["id"] for ex in sixteen})
        self.assertFalse(extra_ids & {ex["id"] for ex in eight})


class TestNotebookDoesNotVendorData(unittest.TestCase):
    def test_builder_imports_instead_of_copying_tickets(self) -> None:
        builder = Path(__file__).resolve().parent.parent / "_build_notebook.py"
        text = builder.read_text(encoding="utf-8")
        self.assertIn("from dataset import", text)
        self.assertNotIn("Customer says they were charged twice for the same", text)
        self.assertNotIn('TRAIN = [', text)


class TestRescore(unittest.TestCase):
    def test_rescore_adds_note_facts(self) -> None:
        ex = example_by_id("T-2003")
        payload = {
            "source": "adapter",
            "n": 1,
            "summary": {},
            "per_prompt": [
                {
                    "trace_id": "T-2003",
                    "gold": ex["gold"],
                    "generation": (
                        "<<TICKET>>\n"
                        "id: T-2003\n"
                        "verdict: NO_REFUND\n"
                        "amount_cents: 0\n"
                        "reason_code: POLICY\n"
                        "note: Asked amount (3300) exceeds invoice (4900).\n"
                        "<</TICKET>>"
                    ),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            data = rescore_report(path)
            row = data["per_prompt"][0]
            self.assertTrue(row["task_pass"])
            self.assertFalse(row["note_facts_pass"])
            self.assertIn("note_facts_pass", data["summary"])


if __name__ == "__main__":
    unittest.main()
