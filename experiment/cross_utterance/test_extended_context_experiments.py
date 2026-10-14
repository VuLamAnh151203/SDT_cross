from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from analyze_contextual_oracle import analyze as analyze_oracle
from analyze_contextual_relationships import analyze as analyze_relationships
from train_harm_predictor import train_and_evaluate


FIELDS = [
    "split", "conversation_id", "conversation_length", "source_idx", "target_idx",
    "source_speaker", "target_speaker", "same_speaker", "signed_offset",
    "absolute_distance", "source_before_target", "modality", "replacement_strategy",
    "source_emotion", "target_emotion", "loss_full", "loss_mask", "signed_impact",
    "pred_full", "pred_mask", "correct_full", "correct_mask",
    "source_feature_norm", "replacement_delta_l2",
]


class ExtendedContextExperimentTests(unittest.TestCase):
    def _make_run(self, root: Path) -> Path:
        run_dir = root / "run"
        result_dir = run_dir / "results"
        result_dir.mkdir(parents=True)
        for strategy in ("zero", "train_mean"):
            rows = []
            for target_idx, label, pred_full, loss_full in (
                (0, 1, 0, 1.0),
                (1, 0, 0, 0.2),
            ):
                for modality in ("text", "audio", "visual"):
                    # Only zero/text corrects the initially wrong target 0.
                    corrects = strategy == "zero" and modality == "text" and target_idx == 0
                    loss_mask = 0.5 if corrects else loss_full + 0.1
                    pred_mask = label if corrects else pred_full
                    source_idx = 1 - target_idx
                    rows.append({
                        "split": "test",
                        "conversation_id": "C1",
                        "conversation_length": 2,
                        "source_idx": source_idx,
                        "target_idx": target_idx,
                        "source_speaker": "A",
                        "target_speaker": "A",
                        "same_speaker": 1,
                        "signed_offset": source_idx - target_idx,
                        "absolute_distance": 1,
                        "source_before_target": int(source_idx < target_idx),
                        "modality": modality,
                        "replacement_strategy": strategy,
                        "source_emotion": 1 - label,
                        "target_emotion": label,
                        "loss_full": loss_full,
                        "loss_mask": loss_mask,
                        "signed_impact": loss_mask - loss_full,
                        "pred_full": pred_full,
                        "pred_mask": pred_mask,
                        "correct_full": int(pred_full == label),
                        "correct_mask": int(pred_mask == label),
                        "source_feature_norm": 1.0,
                        "replacement_delta_l2": 1.0,
                    })
            path = result_dir / f"impact_pairs_{strategy}.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        return run_dir

    def _make_predictor_run(
        self, root: Path, name: str, split: str, conversation_id: str
    ) -> Path:
        run_dir = root / name
        result_dir = run_dir / "results"
        baseline_dir = run_dir / "baseline"
        result_dir.mkdir(parents=True)
        baseline_dir.mkdir(parents=True)
        with (baseline_dir / "baseline_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "conversation_id", "utterance_index", "prob_0", "prob_1"
                ],
            )
            writer.writeheader()
            writer.writerow({
                "conversation_id": conversation_id,
                "utterance_index": 0,
                "prob_0": 0.55,
                "prob_1": 0.45,
            })

        predictor_fields = FIELDS
        for strategy in ("zero", "train_mean"):
            rows = []
            for source_idx in (1, 2):
                for modality in ("text", "audio", "visual"):
                    harmful = source_idx == 1 or modality == "text"
                    impact = -0.2 if harmful else 0.2
                    rows.append({
                        "split": split,
                        "conversation_id": conversation_id,
                        "conversation_length": 3,
                        "source_idx": source_idx,
                        "target_idx": 0,
                        "source_speaker": "A" if source_idx == 1 else "B",
                        "target_speaker": "A",
                        "same_speaker": int(source_idx == 1),
                        "signed_offset": source_idx,
                        "absolute_distance": source_idx,
                        "source_before_target": 0,
                        "modality": modality,
                        "replacement_strategy": strategy,
                        "source_emotion": 0,
                        "target_emotion": 0,
                        "loss_full": 1.0,
                        "loss_mask": 1.0 + impact,
                        "signed_impact": impact,
                        "pred_full": 0,
                        "pred_mask": 0,
                        "correct_full": 1,
                        "correct_mask": 1,
                        "source_feature_norm": 2.0 + source_idx,
                        "replacement_delta_l2": 1.0 + source_idx,
                    })
            with (result_dir / f"impact_pairs_{strategy}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=predictor_fields)
                writer.writeheader()
                writer.writerows(rows)
        return run_dir

    def _make_structural_run(self, root: Path) -> Path:
        run_dir = root / "structural"
        result_dir = run_dir / "results"
        result_dir.mkdir(parents=True)
        speakers = ("A", "B", "A", "A")
        emotions = (3, 0, 3, 2)
        for strategy in ("zero", "train_mean"):
            rows = []
            for source_idx in range(4):
                for target_idx in range(4):
                    if source_idx == target_idx:
                        continue
                    impact = (
                        -0.5 if source_idx == 2 and target_idx == 3 else 0.2
                    )
                    rows.append({
                        "split": "test",
                        "conversation_id": "Dialogue",
                        "conversation_length": 4,
                        "source_idx": source_idx,
                        "target_idx": target_idx,
                        "source_speaker": speakers[source_idx],
                        "target_speaker": speakers[target_idx],
                        "same_speaker": int(
                            speakers[source_idx] == speakers[target_idx]
                        ),
                        "signed_offset": source_idx - target_idx,
                        "absolute_distance": abs(source_idx - target_idx),
                        "source_before_target": int(source_idx < target_idx),
                        "modality": "text",
                        "replacement_strategy": strategy,
                        "source_emotion": emotions[source_idx],
                        "target_emotion": emotions[target_idx],
                        "loss_full": 1.0,
                        "loss_mask": 1.0 + impact,
                        "signed_impact": impact,
                        "pred_full": emotions[target_idx],
                        "pred_mask": emotions[target_idx],
                        "correct_full": 1,
                        "correct_mask": 1,
                        "source_feature_norm": 1.0,
                        "replacement_delta_l2": 1.0,
                    })
            with (result_dir / f"impact_pairs_{strategy}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        return run_dir

    def test_relationship_groups_and_harmful_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._make_run(Path(directory))
            analyze_relationships(
                run_dir, bootstrap_reps=50, seed=7, generate_figures=False
            )
            path = run_dir / "results" / "experiment2_relationship_impact.csv"
            with path.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            row = next(
                item for item in rows
                if item["replacement_strategy"] == "zero"
                and item["modality"] == "text"
                and item["emotion_relation"] == "different_emotion"
            )
            self.assertEqual(row["speaker_relation"], "same_speaker")
            self.assertEqual(row["distance_bin"], "01-05")
            self.assertEqual(int(row["count"]), 2)
            self.assertAlmostEqual(float(row["harmful_impact_rate"]), 0.5)
            self.assertAlmostEqual(
                float(row["conditional_harmful_fraction"]), 0.5
            )
            self.assertAlmostEqual(float(row["mean_harmful_magnitude"]), 0.5)

    def test_unique_corrections_and_oracle_are_target_weighted(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._make_run(Path(directory))
            analyze_oracle(run_dir)
            summary_path = (
                run_dir / "results" / "experiment3_unique_wrong_targets_corrected.csv"
            )
            with summary_path.open("r", newline="", encoding="utf-8") as stream:
                summaries = list(csv.DictReader(stream))
            row = next(
                item for item in summaries
                if item["replacement_strategy_scope"] == "zero"
                and item["modality_scope"] == "text"
            )
            self.assertEqual(int(row["baseline_wrong_target_count"]), 1)
            self.assertEqual(int(row["unique_wrong_targets_corrected"]), 1)

            metric_path = run_dir / "results" / "experiment3_oracle_metrics.csv"
            with metric_path.open("r", newline="", encoding="utf-8") as stream:
                metrics = list(csv.DictReader(stream))
            metric = next(
                item for item in metrics
                if item["replacement_strategy_scope"] == "zero"
                and item["modality_scope"] == "text"
            )
            self.assertEqual(int(metric["target_count"]), 2)
            self.assertAlmostEqual(float(metric["baseline_accuracy"]), 50.0)
            self.assertAlmostEqual(float(metric["oracle_accuracy"]), 100.0)

    def test_previous_same_speaker_and_within_target_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._make_structural_run(Path(directory))
            analyze_relationships(
                run_dir, bootstrap_reps=30, seed=11, generate_figures=False
            )
            previous_path = (
                run_dir / "results" /
                "experiment2_previous_same_speaker_transition.csv"
            )
            with previous_path.open("r", newline="", encoding="utf-8") as stream:
                previous = list(csv.DictReader(stream))
            row = next(
                item for item in previous
                if item["impact_eps"] == "0.0001"
                and item["replacement_strategy"] == "zero"
                and item["modality"] == "text"
                and item["emotion_relation"] == "different_emotion"
            )
            # Target 3's previous same-speaker source is index 2, not index 0.
            self.assertEqual(int(row["count"]), 1)
            self.assertAlmostEqual(float(row["harmful_impact_rate"]), 1.0)

            detail_path = (
                run_dir / "results" /
                "experiment2_within_target_matched_details.csv"
            )
            with detail_path.open("r", newline="", encoding="utf-8") as stream:
                details = list(csv.DictReader(stream))
            matched = next(
                item for item in details
                if item["replacement_strategy"] == "zero"
                and item["target_idx"] == "3"
                and item["distance_bin"] == "01-05"
            )
            self.assertEqual(int(matched["same_speaker_count"]), 2)
            self.assertEqual(int(matched["cross_speaker_count"]), 1)

    def test_harm_predictor_uses_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_run = self._make_predictor_run(root, "train", "train", "TrainC")
            eval_run = self._make_predictor_run(root, "eval", "test", "TestC")
            output = train_and_evaluate(
                train_run, eval_run, max_iter=100, output_dir=root / "predictor"
            )
            with (output / "harm_predictor_metrics.csv").open(
                "r", newline="", encoding="utf-8"
            ) as stream:
                metrics = list(csv.DictReader(stream))
            self.assertEqual(len(metrics), 6)
            self.assertIn("full_lightweight", {row["model"] for row in metrics})
            metadata = json.loads(
                (output / "harm_predictor_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["train_splits"], ["train"])
            self.assertEqual(metadata["eval_splits"], ["test"])
            self.assertIn("target_emotion", metadata["forbidden_predictor_inputs"])


if __name__ == "__main__":
    unittest.main()
