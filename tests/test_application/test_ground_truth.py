from __future__ import annotations

import unittest

from src.application.ground_truth import (
    GroundTruthError,
    clear_ground_truth,
    score_result,
)


def _result() -> dict:
    return {
        "metrics": None,
        "pages": [
            {
                "page_number": 1,
                "data_row_count": 2,
                "metrics": None,
                "columns": [
                    {
                        "key": "current_temperature",
                        "name": "Current Temperature",
                        "value_type": "temperature",
                    },
                    {
                        "key": "comments",
                        "name": "Comments",
                        "value_type": "english_text",
                    },
                ],
                "cells": [
                    {
                        "row": 1,
                        "column_key": "current_temperature",
                        "ocr_text": "67.8",
                        "validation_error": None,
                        "state": "unscored",
                    },
                    {
                        "row": 1,
                        "column_key": "comments",
                        "ocr_text": "All good",
                        "validation_error": None,
                        "state": "unscored",
                    },
                    {
                        "row": 2,
                        "column_key": "current_temperature",
                        "ocr_text": "",
                        "validation_error": "expected temperature pattern",
                        "state": "validation_warning",
                    },
                    {
                        "row": 2,
                        "column_key": "comments",
                        "ocr_text": "",
                        "validation_error": None,
                        "state": "unscored",
                    },
                ],
            }
        ],
    }


class TestGroundTruthScoring(unittest.TestCase):
    def test_scores_exact_matches_blanks_and_validation_mismatches(self) -> None:
        scored = score_result(
            _result(),
            "current temperature,COMMENTS\n67.8,All good\n68.0,\n",
        )

        self.assertEqual(scored["metrics"]["correct_cells"], 3)
        self.assertEqual(scored["metrics"]["incorrect_cells"], 1)
        self.assertEqual(scored["metrics"]["exact_accuracy"], 0.75)
        cells = scored["pages"][0]["cells"]
        self.assertEqual(cells[0]["state"], "correct")
        self.assertEqual(cells[2]["state"], "mismatch_and_warning")
        self.assertEqual(cells[3]["state"], "correct")

    def test_uses_field_aware_normalized_matches_for_review_state(self) -> None:
        result = {
            "metrics": None,
            "pages": [
                {
                    "page_number": 1,
                    "data_row_count": 1,
                    "metrics": None,
                    "columns": [
                        {
                            "key": "date",
                            "name": "Date",
                            "value_type": "date_dd_mon",
                        },
                        {
                            "key": "comments",
                            "name": "Comments",
                            "value_type": "english_text",
                        },
                    ],
                    "cells": [
                        {
                            "row": 1,
                            "column_key": "date",
                            "ocr_text": "02 May",
                            "validation_error": None,
                            "state": "unscored",
                        },
                        {
                            "row": 1,
                            "column_key": "comments",
                            "ocr_text": "Allgood.",
                            "validation_error": None,
                            "state": "unscored",
                        },
                    ],
                }
            ],
        }

        scored = score_result(result, "Date,Comments\n02-May,ALL GOOD\n")

        self.assertEqual(scored["metrics"]["correct_cells"], 2)
        self.assertEqual(scored["metrics"]["exact_accuracy"], 0.0)
        self.assertEqual(scored["metrics"]["normalized_accuracy"], 1.0)
        cells = scored["pages"][0]["cells"]
        self.assertEqual(cells[0]["state"], "correct")
        self.assertEqual(cells[1]["state"], "correct")

    def test_rejects_unknown_missing_and_wrong_row_count(self) -> None:
        with self.assertRaisesRegex(GroundTruthError, "Unknown"):
            score_result(
                _result(),
                "Current Temperature,Comments,Other\n67.8,All good,x\n68,,x\n",
            )
        with self.assertRaisesRegex(GroundTruthError, "Missing"):
            score_result(_result(), "Current Temperature\n67.8\n68\n")
        with self.assertRaisesRegex(GroundTruthError, "1 data rows"):
            score_result(
                _result(),
                "Current Temperature,Comments\n67.8,All good\n",
            )

    def test_clear_ground_truth_preserves_validation_state(self) -> None:
        scored = score_result(
            _result(),
            "Current Temperature,Comments\n67.8,All good\n68.0,\n",
        )

        cleared = clear_ground_truth(scored)

        self.assertIsNone(cleared["metrics"])
        cells = cleared["pages"][0]["cells"]
        self.assertEqual(cells[0]["state"], "unscored")
        self.assertEqual(cells[2]["state"], "validation_warning")
        self.assertIsNone(cells[2]["ground_truth_text"])


if __name__ == "__main__":
    unittest.main()
