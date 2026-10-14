"""Generate in-distribution train-donor source replacements for E1 robustness."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CROSS = ROOT.parent / "cross_utterance"
for path in (ROOT, CROSS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import MODALITIES, write_csv, write_json  # noqa: E402
from impact_core import (apply_replacement, dataset_lookup, forward_conversation,
                         load_conversation, load_experiment, seed_everything)  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES,
                        default=("text",))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    return parser.parse_args(argv)


def _donor_pools(experiment, modalities):
    lookup = dataset_lookup(experiment.dataset)
    pools = {name: [] for name in modalities}
    origins = {name: [] for name in modalities}
    for conversation in experiment.split_ids["train"]:
        text, visual, audio, _, mask, _, _ = experiment.dataset[lookup[conversation]]
        values = {"text": text, "audio": audio, "visual": visual}
        length = int(mask.sum().item())
        for modality in modalities:
            pools[modality].append(values[modality][:length].numpy())
            origins[modality].extend((conversation, index) for index in range(length))
    return ({name: np.concatenate(parts, axis=0).astype(np.float32)
             for name, parts in pools.items()}, origins)


def _donor_index(conversation: str, source: int, modality: str, seed: int,
                 pool_size: int) -> int:
    token = f"{seed}|{conversation}|{source}|{modality}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % pool_size


def generate(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    pools, origins = _donor_pools(experiment, args.modalities)
    lookup = dataset_lookup(experiment.dataset)
    ids = list(experiment.split_ids[args.split])
    if args.max_conversations:
        ids = ids[:args.max_conversations]
    rows = []
    for number, conversation in enumerate(ids, start=1):
        batch = load_conversation(experiment.dataset, lookup, conversation,
                                  experiment.device)
        length = batch.length
        labels = batch.labels[0, :length]
        speakers = batch.speakers[0, :length].argmax(dim=-1)
        full = forward_conversation(experiment.model, batch)
        for modality in args.modalities:
            original = batch.features[modality]
            for source in range(length):
                donor_index = _donor_index(
                    conversation, source, modality, args.seed, len(pools[modality])
                )
                donor = torch.from_numpy(pools[modality][donor_index]).to(experiment.device)
                modified = apply_replacement(original, source, donor)
                masked = forward_conversation(experiment.model, batch, {modality: modified})
                for target in range(length):
                    if source == target:
                        continue
                    label = int(labels[target].item())
                    full_nll = float(-full["log_prob"][0, target, label].cpu())
                    mask_nll = float(-masked["log_prob"][0, target, label].cpu())
                    full_pred = int(full["logits"][0, target].argmax().item())
                    mask_pred = int(masked["logits"][0, target].argmax().item())
                    donor_origin = origins[modality][donor_index]
                    rows.append({
                        "dataset": experiment.dataset.dataset, "split": args.split,
                        "checkpoint_id": experiment.checkpoint_id, "seed": args.seed,
                        "conversation_id": conversation, "conversation_length": length,
                        "source_idx": source, "target_idx": target,
                        "source_speaker": int(speakers[source].item()),
                        "target_speaker": int(speakers[target].item()),
                        "same_speaker": int((speakers[source] == speakers[target]).item()),
                        "signed_offset": source - target,
                        "absolute_distance": abs(source - target),
                        "source_before_target": int(source < target),
                        "modality": modality, "replacement_strategy": "train_donor",
                        "source_emotion": int(labels[source].item()),
                        "target_emotion": label, "loss_full": full_nll,
                        "loss_mask": mask_nll, "signed_impact": mask_nll - full_nll,
                        "prob_full_true": float(full["prob"][0, target, label].cpu()),
                        "prob_mask_true": float(masked["prob"][0, target, label].cpu()),
                        "pred_full": full_pred, "pred_mask": mask_pred,
                        "correct_full": int(full_pred == label),
                        "correct_mask": int(mask_pred == label),
                        "changed_prediction": int(full_pred != mask_pred),
                        "source_feature_norm": float(original[source, 0].norm().cpu()),
                        "replacement_vector_norm": float(donor.norm().cpu()),
                        "replacement_delta_l2": float(
                            (original[source, 0] - donor).norm().cpu()
                        ),
                        "donor_conversation_id": donor_origin[0],
                        "donor_utterance_idx": donor_origin[1],
                        "donor_from_training_split": 1,
                    })
        print(f"donor {number}/{len(ids)} {conversation}", flush=True)
    output = args.output_dir.resolve() / "E1"
    path = write_csv(output / "impact_pairs_train_donor.csv", rows)
    write_json(output / "donor_manifest.json", {
        "checkpoint_id": experiment.checkpoint_id,
        "checkpoint_sha256": experiment.checkpoint_hash,
        "feature_sha256": experiment.feature_hash,
        "split": args.split, "modalities": args.modalities, "seed": args.seed,
        "donor_split": "train", "sampling": "deterministic SHA256 uniform donor",
        "rows": len(rows), "output": str(path),
    })


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    generate(args)


if __name__ == "__main__":
    main()
