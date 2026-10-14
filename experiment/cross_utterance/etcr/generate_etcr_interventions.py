"""Generate frozen-SDT previous-same-speaker intervention caches for ETCR."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from impact_core import (  # noqa: E402
    apply_replacement,
    compute_training_means,
    dataset_lookup,
    forward_conversation,
    load_conversation,
    load_experiment,
    seed_everything,
    write_csv_atomic,
    write_json_atomic,
)
from etcr_common import previous_same_speaker  # noqa: E402


PREFIX_FIELDS = [
    "dataset", "split", "checkpoint_id", "seed", "replacement_strategy",
    "conversation_id", "conversation_length", "target_idx", "target_position",
    "target_speaker", "target_label", "eligible", "source_idx", "source_position",
    "source_speaker", "source_label", "source_distance", "continuity", "transition",
    "pred_full", "pred_mask", "correct_full", "correct_mask", "changed_prediction",
    "nll_full", "nll_mask", "signed_impact", "baseline_margin", "full_entropy",
    "old_emotion_probability_bias_B", "old_vs_current_logit_shift_R",
    "current_emotion_recovery_G", "text_cosine_similarity", "text_l2_distance",
    "source_text_norm", "target_text_norm",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=("train", "valid", "test"))
    parser.add_argument("--strategies", nargs="+", choices=("train_mean", "zero"),
                        default=("train_mean", "zero"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--prob-atol", type=float, default=2e-4)
    parser.add_argument("--prob-rtol", type=float, default=2e-4)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    parser.add_argument(
        "--allow-missing-baseline", action="store_true",
        help="Permit test generation without Experiment 1 baseline equivalence check",
    )
    return parser.parse_args(argv)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _class_fields(n_classes: int) -> list[str]:
    fields = []
    for prefix in ("logit_full", "logit_mask", "prob_full", "prob_mask"):
        fields.extend(f"{prefix}_{index}" for index in range(n_classes))
    return fields


def _row(
    *, experiment, split: str, strategy: str, batch, target_idx: int, source_idx: int,
    full: Mapping[str, torch.Tensor], masked: Mapping[str, torch.Tensor], seed: int,
) -> dict[str, object]:
    length = batch.length
    labels = batch.labels[0, :length]
    speakers = batch.speakers[0, :length].argmax(dim=-1)
    full_logits = full["logits"][0, target_idx]
    mask_logits = masked["logits"][0, target_idx]
    full_prob = full["prob"][0, target_idx]
    mask_prob = masked["prob"][0, target_idx]
    full_log_prob = full["log_prob"][0, target_idx]
    mask_log_prob = masked["log_prob"][0, target_idx]
    for name, tensor in (("full logits", full_logits), ("masked logits", mask_logits),
                         ("full probabilities", full_prob),
                         ("masked probabilities", mask_prob)):
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Non-finite {name} at {batch.conversation_id}:{target_idx}")
    if not math.isclose(float(full_prob.sum().cpu()), 1.0, abs_tol=2e-5):
        raise ValueError("Full class probabilities do not sum to one")
    if not math.isclose(float(mask_prob.sum().cpu()), 1.0, abs_tol=2e-5):
        raise ValueError("Masked class probabilities do not sum to one")
    target_label = int(labels[target_idx].item())
    pred_full = int(full_logits.argmax().item())
    pred_mask = int(mask_logits.argmax().item())
    eligible = source_idx >= 0
    ranked = torch.topk(full_prob, k=2).values
    margin = float((ranked[0] - ranked[1]).cpu())
    entropy = float((-(full_prob * full_log_prob).sum()).cpu())
    nll_full = float((-full_log_prob[target_label]).cpu())
    nll_mask = float((-mask_log_prob[target_label]).cpu())
    target_feature = batch.text[target_idx, 0]
    if eligible:
        source_label = int(labels[source_idx].item())
        source_speaker = int(speakers[source_idx].item())
        source_feature = batch.text[source_idx, 0]
        source_norm = float(source_feature.norm().cpu())
        target_norm = float(target_feature.norm().cpu())
        denominator = max(source_norm * target_norm, 1e-12)
        cosine = float(torch.dot(source_feature, target_feature).cpu()) / denominator
        l2 = float((source_feature - target_feature).norm().cpu())
        bias_b = float((full_prob[source_label] - mask_prob[source_label]).cpu())
        shift_r = float(((full_logits[source_label] - full_logits[target_label]) -
                         (mask_logits[source_label] - mask_logits[target_label])).cpu())
        recovery_g = float((mask_prob[target_label] - full_prob[target_label]).cpu())
        if source_speaker != int(speakers[target_idx].item()):
            raise AssertionError("Selected ETCR source is not from the target speaker")
    else:
        source_label = source_speaker = -1
        source_norm = cosine = l2 = 0.0
        target_norm = float(target_feature.norm().cpu())
        bias_b = shift_r = recovery_g = 0.0
    row: dict[str, object] = {
        "dataset": experiment.dataset.dataset,
        "split": split,
        "checkpoint_id": experiment.checkpoint_id,
        "seed": seed,
        "replacement_strategy": strategy,
        "conversation_id": batch.conversation_id,
        "conversation_length": length,
        "target_idx": target_idx,
        "target_position": target_idx / max(1, length - 1),
        "target_speaker": int(speakers[target_idx].item()),
        "target_label": target_label,
        "eligible": int(eligible),
        "source_idx": source_idx,
        "source_position": source_idx / max(1, length - 1) if eligible else -1.0,
        "source_speaker": source_speaker,
        "source_label": source_label,
        "source_distance": target_idx - source_idx if eligible else 0,
        "continuity": int(eligible and source_label == target_label),
        "transition": int(eligible and source_label != target_label),
        "pred_full": pred_full,
        "pred_mask": pred_mask,
        "correct_full": int(pred_full == target_label),
        "correct_mask": int(pred_mask == target_label),
        "changed_prediction": int(pred_full != pred_mask),
        "nll_full": nll_full,
        "nll_mask": nll_mask,
        "signed_impact": nll_mask - nll_full,
        "baseline_margin": margin,
        "full_entropy": entropy,
        "old_emotion_probability_bias_B": bias_b,
        "old_vs_current_logit_shift_R": shift_r,
        "current_emotion_recovery_G": recovery_g,
        "text_cosine_similarity": cosine,
        "text_l2_distance": l2,
        "source_text_norm": source_norm,
        "target_text_norm": target_norm,
    }
    for prefix, tensor in (("logit_full", full_logits), ("logit_mask", mask_logits),
                           ("prob_full", full_prob), ("prob_mask", mask_prob)):
        row.update({f"{prefix}_{index}": float(value.cpu())
                    for index, value in enumerate(tensor)})
    return row


def _load_baseline(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        return {(row["conversation_id"], int(row["utterance_index"])): row
                for row in csv.DictReader(stream)}


def _validate_baseline(rows, baseline, atol: float, rtol: float) -> None:
    if not baseline:
        return
    for row in rows:
        reference = baseline.get((str(row["conversation_id"]), int(row["target_idx"])))
        if reference is None:
            raise AssertionError("ETCR target is missing from saved baseline")
        if int(reference["prediction"]) != int(row["pred_full"]):
            raise AssertionError("ETCR full prediction differs from saved baseline")
        if reference.get("checkpoint_id") not in (None, "", row["checkpoint_id"]):
            raise AssertionError("ETCR cache and baseline use different checkpoints")
        class_columns = sorted((name for name in row if name.startswith("prob_full_")),
                               key=lambda name: int(name.rsplit("_", 1)[1]))
        for index, name in enumerate(class_columns):
            if not np.isclose(float(row[name]), float(reference[f"prob_{index}"]),
                              atol=atol, rtol=rtol):
                raise AssertionError(
                    f"ETCR baseline probability mismatch at {row['conversation_id']}:"
                    f"{row['target_idx']}, class {index}"
                )


def generate(args: argparse.Namespace) -> None:
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    seed_everything(args.seed)
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    if experiment.model.training or any(p.requires_grad for p in experiment.model.parameters()):
        raise AssertionError("SDT must remain frozen and in eval mode")
    means, mean_metadata = compute_training_means(
        experiment.dataset, experiment.split_ids["train"]
    )
    replacements = {
        "train_mean": torch.from_numpy(means["text"]).to(experiment.device),
        "zero": torch.zeros_like(torch.from_numpy(means["text"])).to(experiment.device),
    }
    output_root = args.run_dir.expanduser().resolve() / "etcr" / "raw"
    lookup = dataset_lookup(experiment.dataset)
    manifest_path = args.run_dir.expanduser().resolve() / "etcr" / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("checkpoint_id") != experiment.checkpoint_id:
            raise ValueError(
                "Existing ETCR cache belongs to another checkpoint; use another RUN_DIR"
            )
        if manifest.get("feature_sha256") != experiment.feature_hash:
            raise ValueError(
                "Existing ETCR cache belongs to another feature file; use another RUN_DIR"
            )
    else:
        manifest = {
            "checkpoint_id": experiment.checkpoint_id,
            "checkpoint_sha256": experiment.checkpoint_hash,
            "feature_sha256": experiment.feature_hash,
            "seed": args.seed,
            "modality": "text",
            "training_mean": mean_metadata,
            "splits": {},
        }
    for split in args.splits:
        if split not in experiment.split_ids:
            raise ValueError(f"Checkpoint has no split {split!r}")
        ids = list(experiment.split_ids[split])
        if args.max_conversations:
            ids = ids[:args.max_conversations]
        strategy_rows = {strategy: [] for strategy in args.strategies}
        source_embeddings, target_embeddings, eligible_flags = [], [], []
        row_keys = []
        for conversation_number, conversation_id in enumerate(ids, start=1):
            batch = load_conversation(
                experiment.dataset, lookup, conversation_id, experiment.device
            )
            length = batch.length
            speakers = [int(value) for value in
                        batch.speakers[0, :length].argmax(dim=-1).cpu().tolist()]
            sources = previous_same_speaker(speakers)
            for target_idx, source_idx in enumerate(sources):
                if source_idx >= 0:
                    if speakers[source_idx] != speakers[target_idx]:
                        raise AssertionError("Previous source has the wrong speaker")
                    if any(speakers[index] == speakers[target_idx]
                           for index in range(source_idx + 1, target_idx)):
                        raise AssertionError(
                            "Selected source is not the nearest previous speaker turn"
                        )
            full = forward_conversation(experiment.model, batch)
            if conversation_number == 1:
                repeated = forward_conversation(experiment.model, batch)
                torch.testing.assert_close(full["logits"], repeated["logits"],
                                           atol=args.prob_atol, rtol=args.prob_rtol)
            targets_cpu = batch.text[:length, 0].detach().cpu().numpy().astype(np.float32)
            for target_idx, source_idx in enumerate(sources):
                target_embeddings.append(targets_cpu[target_idx])
                source_embeddings.append(
                    targets_cpu[source_idx] if source_idx >= 0
                    else np.zeros_like(targets_cpu[target_idx])
                )
                eligible_flags.append(source_idx >= 0)
                row_keys.append(f"{conversation_id}\t{target_idx}")
            for strategy in args.strategies:
                for target_idx, source_idx in enumerate(sources):
                    if source_idx < 0:
                        masked = full
                    else:
                        # Every utterance can be the immediately previous item for
                        # its speaker only once, so retaining masked model outputs
                        # would consume GPU memory without providing cache hits.
                        modified = apply_replacement(
                            batch.text, source_idx, replacements[strategy]
                        )
                        masked = forward_conversation(
                            experiment.model, batch, {"text": modified}
                        )
                    strategy_rows[strategy].append(_row(
                        experiment=experiment, split=split, strategy=strategy,
                        batch=batch, target_idx=target_idx, source_idx=source_idx,
                        full=full, masked=masked, seed=args.seed,
                    ))
            print(f"[{split}] {conversation_number}/{len(ids)} {conversation_id}", flush=True)
        embedding_path = output_root / f"{split}_text_embeddings.npz"
        _atomic_npz(
            embedding_path,
            source=np.stack(source_embeddings), target=np.stack(target_embeddings),
            eligible=np.asarray(eligible_flags, dtype=np.bool_),
            row_key=np.asarray(row_keys),
        )
        split_meta = {
            "conversations": len(ids), "targets": len(row_keys),
            "eligible_targets": int(sum(eligible_flags)),
            "ineligible_targets": int(len(row_keys) - sum(eligible_flags)),
            "embedding_path": str(embedding_path), "strategies": {},
        }
        baseline = _load_baseline(args.run_dir / "baseline" / "baseline_predictions.csv")
        if split == "test" and not baseline and not args.allow_missing_baseline:
            raise FileNotFoundError(
                f"Missing {args.run_dir / 'baseline' / 'baseline_predictions.csv'}; "
                "run Experiment 1 Task A first, or explicitly pass "
                "--allow-missing-baseline"
            )
        for strategy, rows in strategy_rows.items():
            if len({(row["conversation_id"], row["target_idx"]) for row in rows}) != len(rows):
                raise AssertionError("One-target-one-row invariant failed")
            if split == "test":
                _validate_baseline(rows, baseline, args.prob_atol, args.prob_rtol)
            path = output_root / f"{split}_{strategy}.csv"
            n_classes = len([name for name in rows[0] if name.startswith("prob_full_")])
            write_csv_atomic(path, rows, PREFIX_FIELDS + _class_fields(n_classes))
            split_meta["strategies"][strategy] = str(path)
        manifest["splits"][split] = split_meta
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest["splits"], indent=2), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.seed < 0 or args.max_conversations < 0:
        raise ValueError("seed and max-conversations must be nonnegative")
    generate(args)


if __name__ == "__main__":
    main()
