from __future__ import annotations

import sys
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
for path in (HERE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import classification_metrics, distance_bin, margin_bin
from d1_harm_learnability import precision_at_k, spearman
from d1_prepare_edge_dataset import predictor_splits
from e1_mixed_contribution import aggregate_pairs
from e3_three_class_abstention import GateConfiguration, selective_edge_gate
from e3_target_coverage_abstention import (
    target_scores, threshold_selection, validation_coverage_selection,
)
from e3_train_compare import ImpactTeacher
from generate_train_edge_teacher import select_teacher_action


class ResearchGapValidationTests(unittest.TestCase):
    def test_target_modality_mixed_is_not_pair_weighted(self):
        baseline = {("c", 2): {
            "baseline_margin": 0.3, "baseline_correct": 1, "target_emotion": 2,
        }}
        rows = [
            {"conversation_id": "c", "target_idx": "2", "source_idx": "0",
             "modality": "text", "replacement_strategy": "train_mean",
             "signed_impact": "0.02", "conversation_length": "3"},
            {"conversation_id": "c", "target_idx": "2", "source_idx": "1",
             "modality": "text", "replacement_strategy": "train_mean",
             "signed_impact": "-0.03", "conversation_length": "3"},
        ]
        result = aggregate_pairs(rows, baseline, 1e-4)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["mixed"], 1)
        self.assertEqual(result[0]["n_beneficial"], 1)
        self.assertEqual(result[0]["n_harmful"], 1)
        self.assertAlmostEqual(result[0]["mixed_span"], 0.05)

    def test_standard_metrics_count_each_target_once(self):
        metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(metrics["accuracy"], 75.0)

    def test_pre_registered_bins_are_deterministic(self):
        self.assertEqual(distance_bin(5), "01-05")
        self.assertEqual(distance_bin(6), "06-10")
        self.assertEqual(margin_bin(0.049), "low_<0.05")
        self.assertEqual(margin_bin(0.2), "high_>=0.20")

    def test_legacy_impact_teacher_uses_soft_magnitude_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    "conversation_id", "target_idx", "source_idx", "modality",
                    "signed_impact",
                ))
                writer.writeheader()
                writer.writerows((
                    {"conversation_id": "c", "target_idx": 1, "source_idx": 0,
                     "modality": "text", "signed_impact": -0.02},
                    {"conversation_id": "c", "target_idx": 0, "source_idx": 1,
                     "modality": "text", "signed_impact": 0.02},
                ))
            teacher = ImpactTeacher(path, {"c"}, 1e-4, 0.75, 0.01)
            self.assertEqual(teacher.values[("c", 1, 0, 0)], (0.75, 1.0))
            self.assertEqual(teacher.values[("c", 0, 1, 0)], (1.0, 1.0))

    def test_edge_teacher_preserves_continuous_gate_and_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    "conversation_id", "target_idx", "source_idx", "modality",
                    "teacher_gate", "teacher_weight",
                ))
                writer.writeheader()
                writer.writerow({
                    "conversation_id": "c", "target_idx": 1, "source_idx": 0,
                    "modality": "text", "teacher_gate": 0.5,
                    "teacher_weight": 0.4,
                })
            teacher = ImpactTeacher(path, {"c"}, 1e-4, 0.75, 0.01)
            target, weight = teacher.values[("c", 1, 0, 0)]
            self.assertAlmostEqual(target, 0.5)
            self.assertAlmostEqual(weight, 0.4)

    def test_edge_teacher_strong_impact_filter_uses_decision_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    "conversation_id", "target_idx", "source_idx", "modality",
                    "teacher_gate", "teacher_weight", "teacher_decision_margin",
                ))
                writer.writeheader()
                writer.writerows((
                    {
                        "conversation_id": "c", "target_idx": 1, "source_idx": 0,
                        "modality": "text", "teacher_gate": 0.5,
                        "teacher_weight": 0.1, "teacher_decision_margin": 0.001,
                    },
                    {
                        "conversation_id": "c", "target_idx": 0, "source_idx": 1,
                        "modality": "text", "teacher_gate": 0.75,
                        "teacher_weight": 1.0, "teacher_decision_margin": 0.02,
                    },
                ))
            teacher = ImpactTeacher(
                path, {"c"}, 1e-4, 0.75, 0.01, min_margin=0.01
            )
            self.assertNotIn(("c", 1, 0, 0), teacher.values)
            self.assertIn(("c", 0, 1, 0), teacher.values)
            summary = teacher.summary()
            self.assertEqual(summary["teacher_raw_pair_count"], 2)
            self.assertEqual(summary["teacher_filtered_weak_count"], 1)
            self.assertEqual(summary["teacher_pair_count"], 1)

    def test_edge_teacher_selects_aligned_suppression_action(self):
        gate, weight, decision, margin = select_teacher_action(
            {0.0: 0.75, 0.5: 0.70, 0.75: 0.72, 1.0: 0.80},
            weight_scale=0.2, margin_epsilon=0.0,
        )
        self.assertEqual(gate, 0.5)
        self.assertEqual(decision, "suppress")
        self.assertAlmostEqual(margin, 0.1)
        self.assertAlmostEqual(weight, 0.5)

    def test_edge_teacher_prefers_identity_on_exact_tie(self):
        gate, weight, decision, margin = select_teacher_action(
            {0.0: 0.8, 0.5: 0.8, 1.0: 0.8},
            weight_scale=0.1, margin_epsilon=1e-4,
        )
        self.assertEqual(gate, 1.0)
        self.assertEqual(decision, "keep")
        self.assertEqual(margin, 0.0)
        self.assertEqual(weight, 0.0)

    def test_d1_holds_out_a_complete_session_without_impact_labels(self):
        splits, metadata = predictor_splits({
            "train": ["Ses01A", "Ses02A", "Ses02B"],
            "valid": [],
            "test": ["Ses03A"],
        }, validation_sessions=("Ses02",))
        self.assertEqual(splits["train"], ["Ses01A"])
        self.assertEqual(splits["valid"], ["Ses02A", "Ses02B"])
        self.assertEqual(splits["test"], ["Ses03A"])
        self.assertEqual(metadata["validation_source"], "held_out_train_session")

    def test_d1_spearman_and_precision_at_k_follow_harm_ranking(self):
        impact = np.asarray([-0.4, 0.2, -0.1, 0.3], dtype=np.float64)
        predicted_signed = np.asarray([-0.3, 0.1, -0.05, 0.4], dtype=np.float64)
        self.assertAlmostEqual(spearman(predicted_signed, impact), 1.0)
        rows = precision_at_k(
            impact=impact,
            harm_score=-predicted_signed,
            conversations=np.asarray(["c", "c", "c", "c"]),
            targets=np.asarray([0, 0, 1, 1]),
            modalities=np.asarray(["text", "text", "text", "text"]),
            epsilon=1e-4,
            k_values=(1,),
            scope="all",
        )
        self.assertEqual(rows[0]["target_groups"], 2)
        self.assertAlmostEqual(rows[0]["macro_random_precision"], 0.5)
        self.assertAlmostEqual(rows[0]["macro_precision_at_k"], 1.0)
        self.assertAlmostEqual(rows[0]["macro_precision_lift"], 2.0)

    def test_three_class_abstention_keeps_identity_below_threshold(self):
        probabilities = np.full((3, 3, 3), np.nan, dtype=np.float32)
        probabilities[0, 1] = [0.70, 0.20, 0.10]
        probabilities[0, 2] = [0.60, 0.20, 0.20]
        probabilities[1, 0] = [0.40, 0.50, 0.10]
        probabilities[1, 2] = [0.30, 0.40, 0.30]
        probabilities[2, 0] = [0.45, 0.40, 0.15]
        probabilities[2, 1] = [0.35, 0.40, 0.25]
        gate, decisions = selective_edge_gate(
            probabilities, GateConfiguration(0.8, 0.2, 1.0), 0,
        )
        np.testing.assert_array_equal(gate, np.ones_like(gate))
        self.assertEqual(sum(int(row["intervened"]) for row in decisions), 0)

    def test_three_class_abstention_selects_one_edge_and_uses_soft_gate(self):
        probabilities = np.full((3, 3, 3), np.nan, dtype=np.float32)
        probabilities[0, 1] = [0.70, 0.20, 0.10]  # margin 0.60: selected
        probabilities[0, 2] = [0.75, 0.20, 0.05]  # margin 0.70: selected instead
        probabilities[1, 0] = [0.40, 0.50, 0.10]
        probabilities[1, 2] = [0.30, 0.40, 0.30]
        probabilities[2, 0] = [0.45, 0.40, 0.15]
        probabilities[2, 1] = [0.35, 0.40, 0.25]
        gate, decisions = selective_edge_gate(
            probabilities, GateConfiguration(0.7, 0.5, 0.5), 0,
        )
        self.assertEqual(decisions[0]["selected_source_idx"], 2)
        self.assertAlmostEqual(float(gate[0, 2, 0]), 0.625)
        self.assertEqual(int(np.sum(gate[:, :, 0] < 1.0)), 1)
        np.testing.assert_array_equal(gate[:, :, 1:], np.ones_like(gate[:, :, 1:]))

    def test_coverage_abstention_selects_exact_validation_fraction(self):
        matrix = np.full((3, 3, 3), np.nan, dtype=np.float32)
        matrix[0, 1], matrix[0, 2] = [0.8, 0.1, 0.1], [0.6, 0.2, 0.2]
        matrix[1, 0], matrix[1, 2] = [0.5, 0.3, 0.2], [0.4, 0.3, 0.3]
        matrix[2, 0], matrix[2, 1] = [0.3, 0.4, 0.3], [0.2, 0.4, 0.4]
        ranked = target_scores({"c": matrix})
        selected, cutoff = validation_coverage_selection(ranked, 1 / 3)
        self.assertEqual(selected, {("c", 0)})
        self.assertAlmostEqual(cutoff, 0.7)
        self.assertEqual(threshold_selection(ranked, cutoff), selected)

    def test_coverage_allowlist_prevents_unselected_intervention(self):
        probabilities = np.full((2, 2, 3), np.nan, dtype=np.float32)
        probabilities[0, 1] = [0.9, 0.05, 0.05]
        probabilities[1, 0] = [0.8, 0.1, 0.1]
        gate, decisions = selective_edge_gate(
            probabilities, GateConfiguration(0.0, -1.0, 1.0), 0, {1},
        )
        self.assertEqual(decisions[0]["intervened"], 0)
        self.assertEqual(decisions[1]["intervened"], 1)
        self.assertEqual(float(gate[0, 1, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
