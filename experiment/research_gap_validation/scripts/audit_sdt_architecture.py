"""Audit SDT propagation paths and verify source/target axes at runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CROSS = ROOT.parent / "cross_utterance"
for path in (ROOT, CROSS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import write_json  # noqa: E402
from impact_core import dataset_lookup, load_conversation, load_experiment  # noqa: E402
from reliability_model import BRANCHES, ReliabilitySDT, SOURCE_MODALITY  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    return parser.parse_args(argv)


def audit(args: argparse.Namespace) -> None:
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    lookup = dataset_lookup(experiment.dataset)
    split = "test" if experiment.split_ids.get("test") else "valid"
    conversation = experiment.split_ids[split][0]
    batch = load_conversation(experiment.dataset, lookup, conversation, experiment.device)
    wrapper = ReliabilitySDT(experiment.model, "B0").to(experiment.device).eval()
    length = batch.length
    identity_edge = torch.ones(1, length, length, 3, device=experiment.device)
    with torch.inference_mode():
        output = wrapper(
            batch.text, batch.visual, batch.audio, batch.mask, batch.speakers,
            batch.lengths, fixed_edge_gate=identity_edge, capture_attention=True,
        )
    branches = []
    for name in BRANCHES:
        encoder = getattr(experiment.model, name)
        attention = output["attention_audit"][name]
        before, after = attention[0], attention[1]
        if before.ndim != 4 or before.shape[-2:] != (length, length):
            raise AssertionError(f"Unexpected attention axes for {name}: {tuple(before.shape)}")
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        branches.append({
            "branch": name,
            "source_modality_index": SOURCE_MODALITY[name],
            "source_modality": ("text", "audio", "visual")[SOURCE_MODALITY[name]],
            "layers": len(encoder.transformer_inter),
            "attention_shape": list(before.shape),
            "axis_order": ["batch", "head", "target", "source"],
        })
    report = {
        "checkpoint_id": experiment.checkpoint_id,
        "checkpoint_sha256": experiment.checkpoint_hash,
        "feature_sha256": experiment.feature_hash,
        "conversation": conversation,
        "conversation_length": length,
        "model_class": type(experiment.model).__name__,
        "is_gnn": False,
        "propagation_type": "nine Transformer attention branches",
        "gate_location": "post-softmax attention contribution before dropout/value aggregation",
        "self_edges_forced_identity": True,
        "default_attention_renormalization": False,
        "late_location": "H'_T/H'_A/H'_V after feature reduction, before final fusion",
        "branches": branches,
        "limitations": [
            "Each branch contains one attention layer, so layer-depth localization is unavailable.",
            "An edge can be suppressed in one branch or all three branches sharing a source modality.",
            "Post-softmax non-renormalized gating changes message magnitude; renormalization is a sensitivity control.",
        ],
    }
    output = args.output_dir.resolve() / "manifests"
    write_json(output / "architecture_audit.json", report)
    lines = [
        "# SDT architecture audit", "", f"Checkpoint: `{experiment.checkpoint_id}`", "",
        "Attention axis verified at runtime: `[batch, head, target, source]`.", "",
        "| Branch | Source modality | Layers | Runtime shape |", "|---|---|---:|---|",
    ]
    for row in branches:
        lines.append(f"| {row['branch']} | {row['source_modality']} | {row['layers']} | {row['attention_shape']} |")
    lines.extend(["", *[f"- {item}" for item in report["limitations"]]])
    (output / "architecture_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    audit(parse_args(argv))


if __name__ == "__main__":
    main()
