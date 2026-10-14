from __future__ import annotations

import sys
import unittest
from pathlib import Path


try:
    import torch
except ImportError:  # Local analysis-only environments may not include PyTorch.
    torch = None


HERE = Path(__file__).resolve().parent
SDT_ROOT = HERE.parents[1]
for path in (HERE, SDT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class ReliabilityModelTests(unittest.TestCase):
    def _base(self):
        from model import Transformer_Based_Model
        return Transformer_Based_Model(
            dataset="IEMOCAP", temp=1.0, D_text=2, D_visual=2, D_audio=2,
            n_head=1, n_classes=3, hidden_dim=4, n_speakers=2, dropout=0.0,
            fusion_variant="sdt",
        ).eval()

    def _inputs(self):
        torch.manual_seed(3)
        text = torch.randn(3, 1, 2)
        visual = torch.randn(3, 1, 2)
        audio = torch.randn(3, 1, 2)
        mask = torch.ones(1, 3)
        speakers = torch.tensor([[[1., 0.], [0., 1.], [1., 0.]]])
        return text, visual, audio, mask, speakers, [3]

    def test_identity_edge_gate_is_exact(self):
        from reliability_model import ReliabilitySDT
        base = self._base()
        wrapper = ReliabilitySDT(base, "B0").eval()
        inputs = self._inputs()
        with torch.inference_mode():
            expected = base(*inputs)
            actual = wrapper(*inputs, fixed_edge_gate=torch.ones(1, 3, 3, 3))
        torch.testing.assert_close(expected["logits"], actual["logits"], atol=0, rtol=0)

    def test_source_target_axis_and_self_identity(self):
        from reliability_model import ReliabilitySDT
        wrapper = ReliabilitySDT(self._base(), "B0").eval()
        edge = torch.ones(1, 3, 3, 3)
        edge[0, 1, 0, 0] = 0.0
        edge[0, 2, 2, 0] = 0.0  # Must be overridden to preserve self.
        with torch.inference_mode():
            output = wrapper(*self._inputs(), fixed_edge_gate=edge,
                             capture_attention=True)
        for branch in ("t_t", "t_a", "t_v"):
            before, after = output["attention_audit"][branch]
            self.assertTrue(torch.all(after[:, :, 1, 0] == 0))
            torch.testing.assert_close(before[:, :, 2, 2], after[:, :, 2, 2])


if __name__ == "__main__":
    unittest.main()
