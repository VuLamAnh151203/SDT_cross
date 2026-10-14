import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from impact_core import (
    ConversationBatch,
    apply_replacement,
    build_impact_rows,
    compute_training_means,
    expected_pair_count,
    js_divergence_from_log_probs,
    signed_impact,
)
class CrossUtteranceImpactTests(unittest.TestCase):
    def test_replacement_changes_only_one_source_row(self):
        original = torch.arange(24, dtype=torch.float32).reshape(4, 1, 6)
        snapshot = original.clone()
        replacement = torch.full((6,), -3.0)
        modified = apply_replacement(original, 2, replacement)
        torch.testing.assert_close(original, snapshot, rtol=0, atol=0)
        torch.testing.assert_close(modified[2, 0], replacement, rtol=0, atol=0)
        torch.testing.assert_close(modified[:2], original[:2], rtol=0, atol=0)
        torch.testing.assert_close(modified[3:], original[3:], rtol=0, atol=0)

    def test_signed_impact_convention(self):
        full = torch.tensor([-0.2])
        masked = torch.tensor([-0.8])
        impact = signed_impact(full, masked)
        self.assertGreater(impact.item(), 0)
        self.assertAlmostEqual(impact.item(), 0.6, places=6)
        self.assertAlmostEqual(
            impact.item(), (-masked - (-full)).item(), places=6
        )

    def test_js_divergence_is_zero_for_identical_distributions(self):
        logits = torch.tensor([[2.0, -1.0, 0.5]])
        log_prob = logits.log_softmax(dim=-1)
        divergence = js_divergence_from_log_probs(log_prob, log_prob)
        torch.testing.assert_close(divergence, torch.zeros_like(divergence))

    def test_expected_pair_count_excludes_self_pairs(self):
        self.assertEqual(expected_pair_count(0), 0)
        self.assertEqual(expected_pair_count(1), 0)
        self.assertEqual(expected_pair_count(4), 36)

    def test_rows_have_correct_sign_offsets_and_no_self_pair(self):
        labels = torch.tensor([[0, 1, 0]])
        speakers = torch.tensor([[[1, 0], [0, 1], [1, 0]]], dtype=torch.float32)
        batch = ConversationBatch(
            conversation_id="dialogue",
            text=torch.randn(3, 1, 4),
            visual=torch.randn(3, 1, 2),
            audio=torch.randn(3, 1, 5),
            speakers=speakers,
            mask=torch.ones(1, 3),
            labels=labels,
        )
        full_logits = torch.tensor([[[3.0, 0.0], [0.0, 2.0], [2.0, 0.0]]])
        masked_logits = full_logits.clone()
        masked_logits[0, 0, 0] -= 1.0
        masked_logits[0, 2, 0] += 0.5

        def output(logits):
            return {
                "logits": logits,
                "log_prob": logits.log_softmax(dim=-1),
                "prob": logits.softmax(dim=-1),
            }

        replacement = torch.zeros(4)
        rows = build_impact_rows(
            dataset_name="IEMOCAP",
            split="test",
            checkpoint_id="abc",
            seed=1,
            batch=batch,
            source_idx=1,
            modality="text",
            strategy="zero",
            original_feature=batch.text,
            replacement=replacement,
            full_output=output(full_logits),
            masked_output=output(masked_logits),
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["target_idx"] for row in rows}, {0, 2})
        by_target = {row["target_idx"]: row for row in rows}
        self.assertEqual(by_target[0]["signed_offset"], 1)
        self.assertEqual(by_target[0]["source_before_target"], 0)
        self.assertEqual(by_target[2]["signed_offset"], -1)
        self.assertEqual(by_target[2]["source_before_target"], 1)
        for row in rows:
            self.assertAlmostEqual(
                row["signed_impact"], row["loss_mask"] - row["loss_full"], places=6
            )

    def test_training_mean_uses_only_requested_train_ids(self):
        class FakeDataset:
            keys = ["train", "test"]
            feature_dims = {"D_text": 2, "D_visual": 1, "D_audio": 3}

            def __getitem__(self, index):
                value = 2.0 if self.keys[index] == "train" else 100.0
                return (
                    torch.full((2, 2), value),
                    torch.full((2, 1), value + 1),
                    torch.full((2, 3), value + 2),
                    torch.tensor([[1, 0], [0, 1]], dtype=torch.float32),
                    torch.ones(2),
                    torch.zeros(2, dtype=torch.long),
                    self.keys[index],
                )

        means, metadata = compute_training_means(FakeDataset(), ["train"])
        np.testing.assert_allclose(means["text"], np.full(2, 2.0))
        np.testing.assert_allclose(means["visual"], np.full(1, 3.0))
        np.testing.assert_allclose(means["audio"], np.full(3, 4.0))
        self.assertEqual(metadata["train_conversations"], 1)
        self.assertEqual(metadata["train_utterances"], 2)


if __name__ == "__main__":
    unittest.main()
