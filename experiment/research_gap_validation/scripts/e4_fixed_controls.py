"""Evaluate random/distance/speaker/margin edge controls at B4's action rate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CROSS = ROOT.parent / "cross_utterance"
SDT_ROOT = ROOT.parents[1]
for path in (ROOT, CROSS, SDT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import classification_metrics, write_csv  # noqa: E402
from dataloader import make_loaders  # noqa: E402
from impact_core import load_experiment, seed_everything  # noqa: E402
from model import Transformer_Based_Model  # noqa: E402
from reliability_model import ReliabilitySDT  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--trained-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    return parser.parse_args(argv)


def _wrapper(experiment, path):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact["model_id"] != "B4":
        return None, None
    config = artifact["configuration"]
    base = Transformer_Based_Model(**experiment.checkpoint["model_config"]).to(experiment.device)
    base.load_state_dict(experiment.checkpoint["model_state_dict"], strict=True)
    if hasattr(base, "set_tical_epoch"):
        base.set_tical_epoch(int(experiment.checkpoint.get("epoch", 0)))
    model = ReliabilitySDT(
        base, "B4", int(config["gate_hidden"]), float(config["initial_gate"]),
        int(config["capacity_hidden"]), bool(config["renormalize_attention"]),
    ).to(experiment.device)
    model.freeze_backbone(True)
    model.load_trainable_state_dict(artifact["trainable_state_dict"])
    model.eval()
    return model, artifact


def _top_k_gate(scores, valid_pairs, count):
    gate = torch.ones(*valid_pairs.shape, 3, device=scores.device)
    expanded = valid_pairs.unsqueeze(-1).expand_as(gate)
    flat_scores = scores.masked_fill(~expanded, -torch.inf).reshape(-1)
    if count:
        chosen = torch.topk(flat_scores, min(count, int(expanded.sum()))).indices
        gate.reshape(-1)[chosen] = 0.0
    return gate


def run(args):
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    rows = []
    for path in args.trained_checkpoints:
        model, artifact = _wrapper(experiment, path)
        if model is None:
            continue
        seed = int(artifact["seed"])
        seed_everything(seed)
        loader = make_loaders(experiment.dataset, experiment.split_ids,
                              args.batch_size, seed)["test"]
        storage = {name: {"labels": [], "predictions": [], "nll": [], "actions": 0,
                          "possible": 0}
                   for name in ("learned_B4", "random", "distance_only",
                                "same_speaker_only", "margin_only")}
        rng = torch.Generator(device=experiment.device).manual_seed(seed)
        for batch in loader:
            text, visual, audio, speakers_lbs, mask, labels = [
                item.to(experiment.device) for item in batch[:6]
            ]
            speakers = speakers_lbs.transpose(0, 1)
            valid = mask.bool()
            lengths = valid.sum(1).tolist()
            length = mask.size(1)
            diagonal = torch.eye(length, dtype=torch.bool, device=mask.device)
            valid_pairs = valid[:, :, None] & valid[:, None, :] & ~diagonal[None]
            with torch.inference_mode():
                learned = model(text, visual, audio, mask, speakers, lengths)
            learned_gate = learned["reliability_pair_gate"]
            action_count = int(((learned_gate < 0.5)
                                & valid_pairs.unsqueeze(-1)).sum().item())
            position = torch.arange(length, device=mask.device)
            distance = (position[:, None] - position[None, :]).abs().float()
            distance_scores = distance[None, :, :, None].expand_as(learned_gate)
            speaker_id = speakers.argmax(-1)
            same = speaker_id[:, :, None].eq(speaker_id[:, None, :]).float()
            same_scores = same.unsqueeze(-1).expand_as(learned_gate)
            probability = learned["prob"].detach()
            top2 = torch.topk(probability, 2, dim=-1).values
            uncertainty = 1.0 - (top2[..., 0] - top2[..., 1])
            margin_scores = uncertainty[:, :, None, None].expand_as(learned_gate)
            random_scores = torch.rand(learned_gate.shape, generator=rng,
                                       device=experiment.device)
            gates = {
                "random": _top_k_gate(random_scores, valid_pairs, action_count),
                "distance_only": _top_k_gate(distance_scores, valid_pairs, action_count),
                "same_speaker_only": _top_k_gate(same_scores, valid_pairs, action_count),
                "margin_only": _top_k_gate(margin_scores, valid_pairs, action_count),
            }
            outputs = {"learned_B4": learned}
            with torch.inference_mode():
                for name, gate in gates.items():
                    outputs[name] = model(text, visual, audio, mask, speakers, lengths,
                                          fixed_edge_gate=gate)
            for name, output in outputs.items():
                predictions = output["logits"].argmax(-1)
                valid_log = output["log_prob"][valid]
                target = labels[valid]
                nll = -valid_log.gather(1, target.unsqueeze(-1)).squeeze(-1)
                storage[name]["labels"].extend(target.cpu().tolist())
                storage[name]["predictions"].extend(predictions[valid].cpu().tolist())
                storage[name]["nll"].extend(nll.cpu().tolist())
                storage[name]["actions"] += action_count
                storage[name]["possible"] += int(valid_pairs.sum().item()) * 3
        for name, values in storage.items():
            metrics = classification_metrics(values["labels"], values["predictions"])
            rows.append({
                "model": "B4", "seed": seed, "control": name, **metrics,
                "nll": float(np.mean(values["nll"])),
                "action_count": values["actions"],
                "action_rate": values["actions"] / values["possible"],
                "same_action_budget_as_learned": 1,
            })
    write_csv(args.output_dir.resolve() / "E4" / "fixed_control_metrics.csv", rows)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
