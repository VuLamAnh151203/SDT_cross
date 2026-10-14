import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute_ablation_classification_metrics import (  # noqa: E402
    calculate_metrics,
    calculate_test_metrics,
)


class ClassificationMetricTests(unittest.TestCase):
    def test_standard_metrics_deduplicate_targets(self):
        fieldnames = [
            "conversation_id", "conversation_length", "target_idx",
            "target_emotion", "pred_full",
        ]
        input_rows = [
            ("dialogue", 3, 0, 0, 0),
            ("dialogue", 3, 0, 0, 0),
            ("dialogue", 3, 1, 1, 0),
            ("dialogue", 3, 1, 1, 0),
            ("dialogue", 3, 2, 1, 1),
            ("dialogue", 3, 2, 1, 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "impact.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dict(zip(fieldnames, row)) for row in input_rows)

            result = calculate_test_metrics([path])

        self.assertEqual(result["conversation_count"], 1)
        self.assertEqual(result["utterance_count"], 3)
        self.assertAlmostEqual(result["accuracy"], 100.0 * 2 / 3)
        self.assertAlmostEqual(result["weighted_f1"], 100.0 * 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 100.0 * 2 / 3)

    def test_pair_weighted_classification_metrics(self):
        fieldnames = [
            "replacement_strategy", "modality", "target_emotion",
            "pred_full", "pred_mask",
        ]
        input_rows = [
            ("train_mean", "text", 0, 0, 0),
            ("train_mean", "text", 1, 0, 1),
            ("train_mean", "text", 1, 1, 1),
            ("train_mean", "text", 2, 2, 2),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "impact.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dict(zip(fieldnames, row)) for row in input_rows)

            rows = calculate_metrics([path])

        self.assertEqual(len(rows), 1)
        result = rows[0]
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["baseline_correct"], 3)
        self.assertEqual(result["ablation_correct"], 4)
        self.assertAlmostEqual(result["baseline_pair_weighted_accuracy"], 75.0)
        self.assertAlmostEqual(result["ablation_pair_weighted_accuracy"], 100.0)
        self.assertAlmostEqual(result["delta_accuracy"], 25.0)
        self.assertAlmostEqual(result["baseline_pair_weighted_wf1"], 75.0)
        self.assertAlmostEqual(result["ablation_pair_weighted_wf1"], 100.0)
        self.assertEqual(result["changed_prediction_count"], 1)
        self.assertAlmostEqual(result["changed_prediction_fraction"], 0.25)


if __name__ == "__main__":
    unittest.main()
