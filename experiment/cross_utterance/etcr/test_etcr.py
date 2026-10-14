from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from etcr_common import classification_metrics, previous_same_speaker
from phase_b_analyze_errors import _error_row
from phase_c_train_policy import _policy_masks, _select_threshold


def make_row(index, *, eligible=True, transition=False, source_label=0,
             target_label=0, pred_full=0, pred_mask=0, full=0.5, mask=0.5,
             margin=0.1):
    return {
        "conversation_id": f"c{index // 3}", "target_idx": str(index),
        "eligible": str(int(eligible)), "transition": str(int(transition)),
        "continuity": str(int(eligible and not transition)),
        "source_label": str(source_label), "target_label": str(target_label),
        "pred_full": str(pred_full), "pred_mask": str(pred_mask),
        "correct_full": str(int(pred_full == target_label)),
        "correct_mask": str(int(pred_mask == target_label)),
        "nll_full": str(full), "nll_mask": str(mask),
        "signed_impact": str(mask - full), "baseline_margin": str(margin),
    }


class ETCRTests(unittest.TestCase):
    def test_previous_same_speaker_is_nearest_and_strictly_previous(self):
        self.assertEqual(previous_same_speaker([0, 1, 0, 0, 1]), [-1, -1, 0, 2, 1])

    def test_classification_metrics_counts_targets_once(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(result["accuracy"], 75.0)

    def test_persistence_errors_are_unique_target_events(self):
        rows = [
            make_row(0, transition=True, source_label=1, target_label=2,
                     pred_full=1, pred_mask=2),
            make_row(1, transition=True, source_label=1, target_label=2,
                     pred_full=1, pred_mask=1),
            make_row(2, transition=True, source_label=1, target_label=2,
                     pred_full=2, pred_mask=1),
        ]
        result = _error_row("train_mean", rows)
        self.assertEqual(result["persistence_error_count"], 2)
        self.assertEqual(result["corrected_count"], 1)
        self.assertEqual(result["remains_count"], 1)
        self.assertEqual(result["damaged_count"], 1)

    def test_validation_threshold_minimizes_nll(self):
        rows = [
            make_row(0, full=1.0, mask=0.1),
            make_row(1, full=1.0, mask=0.2),
            make_row(2, full=1.0, mask=1.5),
        ]
        threshold = _select_threshold(rows, np.asarray([0, 1, 2]),
                                      np.asarray([0.9, 0.8, 0.1]))
        self.assertEqual(threshold, 0.8)

    def test_policy_baselines_use_learned_budget_and_keep_ineligible(self):
        rows = [make_row(index, eligible=index != 3, margin=0.1 * index)
                for index in range(4)]
        policies = _policy_masks(
            rows, np.asarray([0, 1, 2]),
            learned_scores=np.asarray([0.9, 0.8, 0.1]),
            uncertainty_scores=np.asarray([0.2, 0.7, 0.4]),
            structure_scores=np.asarray([0.1, 0.2, 0.9]),
            threshold=0.8, seed=7,
        )
        for name in ("random_same_budget", "low_margin_same_budget",
                     "uncertainty_same_budget", "structure_same_budget"):
            self.assertEqual(int(policies[name].sum()), 2)
            self.assertFalse(bool(policies[name][3]))


if __name__ == "__main__":
    unittest.main()
