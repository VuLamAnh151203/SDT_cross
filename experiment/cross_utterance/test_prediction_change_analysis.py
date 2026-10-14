import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_prediction_changes import analyze  # noqa: E402


class PredictionChangeAnalysisTests(unittest.TestCase):
    def test_analysis_joins_baseline_and_compares_strategies(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            baseline_dir = run_dir / "baseline"
            result_dir = run_dir / "results"
            baseline_dir.mkdir()
            result_dir.mkdir()

            baseline_rows = [
                {
                    "conversation_id": "dialogue", "utterance_index": 0,
                    "label": 0, "prediction": 0, "correct": 1,
                    "prob_true": 0.51, "prob_0": 0.51, "prob_1": 0.49,
                },
                {
                    "conversation_id": "dialogue", "utterance_index": 1,
                    "label": 1, "prediction": 1, "correct": 1,
                    "prob_true": 0.90, "prob_0": 0.10, "prob_1": 0.90,
                },
            ]
            with (baseline_dir / "baseline_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(baseline_rows[0]))
                writer.writeheader()
                writer.writerows(baseline_rows)

            impact_fields = [
                "conversation_id", "source_idx", "target_idx", "source_speaker",
                "target_speaker", "same_speaker", "absolute_distance",
                "source_before_target", "modality", "replacement_strategy",
                "source_emotion", "target_emotion", "signed_impact",
                "prob_full_true", "prob_mask_true", "delta_true_class_prob",
                "pred_full", "pred_mask", "correct_full", "correct_mask",
                "changed_prediction", "js_divergence", "source_feature_norm",
                "replacement_delta_l2",
            ]
            for strategy in ("zero", "train_mean"):
                impact_rows = []
                for modality in ("text", "audio", "visual"):
                    changed = modality == "text"
                    impact_rows.extend([
                        {
                            "conversation_id": "dialogue", "source_idx": 1,
                            "target_idx": 0, "source_speaker": 1,
                            "target_speaker": 0, "same_speaker": 0,
                            "absolute_distance": 1, "source_before_target": 0,
                            "modality": modality, "replacement_strategy": strategy,
                            "source_emotion": 1, "target_emotion": 0,
                            "signed_impact": 0.2 if changed else 0.0,
                            "prob_full_true": 0.51,
                            "prob_mask_true": 0.40 if changed else 0.51,
                            "delta_true_class_prob": 0.11 if changed else 0.0,
                            "pred_full": 0, "pred_mask": 1 if changed else 0,
                            "correct_full": 1,
                            "correct_mask": 0 if changed else 1,
                            "changed_prediction": int(changed),
                            "js_divergence": 0.03 if changed else 0.0,
                            "source_feature_norm": 2.0,
                            "replacement_delta_l2": 2.0,
                        },
                        {
                            "conversation_id": "dialogue", "source_idx": 0,
                            "target_idx": 1, "source_speaker": 0,
                            "target_speaker": 1, "same_speaker": 0,
                            "absolute_distance": 1, "source_before_target": 1,
                            "modality": modality, "replacement_strategy": strategy,
                            "source_emotion": 0, "target_emotion": 1,
                            "signed_impact": 0.0, "prob_full_true": 0.90,
                            "prob_mask_true": 0.90, "delta_true_class_prob": 0.0,
                            "pred_full": 1, "pred_mask": 1, "correct_full": 1,
                            "correct_mask": 1, "changed_prediction": 0,
                            "js_divergence": 0.0, "source_feature_norm": 2.0,
                            "replacement_delta_l2": 2.0,
                        },
                    ])
                path = result_dir / f"impact_pairs_{strategy}.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=impact_fields)
                    writer.writeheader()
                    writer.writerows(impact_rows)

            analyze(run_dir, top_k=10)

            with (result_dir / "changed_prediction_summary.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                summary = list(csv.DictReader(stream))
            zero_text = next(
                row for row in summary
                if row["replacement_strategy"] == "zero" and row["modality"] == "text"
            )
            self.assertEqual(int(zero_text["changed_prediction_count"]), 1)
            self.assertAlmostEqual(float(zero_text["changed_prediction_fraction"]), 0.5)
            self.assertAlmostEqual(float(zero_text["median_baseline_margin_changed"]), 0.02)

            with (result_dir / "flip_robustness_zero_vs_mean.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                robustness = list(csv.DictReader(stream))
            shared_text = next(
                row for row in robustness
                if row["modality"] == "text"
                and row["category"] == "both_same_new_prediction"
            )
            self.assertEqual(int(shared_text["count"]), 1)

            with (result_dir / "top_changed_cases.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                top_cases = list(csv.DictReader(stream))
            self.assertEqual(len(top_cases), 2)


if __name__ == "__main__":
    unittest.main()
