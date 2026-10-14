"""Prepare leakage-safe direct-edge data for D1 harmful-influence prediction.

The frozen SDT is evaluated with one propagation edge suppressed at a time.
Predictor inputs are observable at inference; true emotions are retained only
as analysis metadata and are never included in ``FEATURE_NAMES``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CROSS = ROOT.parent / "cross_utterance"
for path in (ROOT, CROSS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import session_id, sha256, write_csv, write_json  # noqa: E402
from impact_core import (dataset_lookup, load_conversation, load_experiment,
                         seed_everything)  # noqa: E402
from reliability_model import ReliabilitySDT, source_branches  # noqa: E402


MODALITY_INDEX = {"text": 0, "audio": 1, "visual": 2}
FEATURE_NAMES = (
    "target_margin",
    "target_top1_probability",
    "target_entropy",
    "source_margin",
    "source_top1_probability",
    "source_entropy",
    "prediction_agreement",
    "probability_cosine",
    "log1p_absolute_distance",
    "normalized_signed_offset",
    "same_speaker",
    "source_before_target",
    "normalized_source_position",
    "normalized_target_position",
    "log1p_conversation_length",
    "log1p_source_feature_norm",
    "log1p_target_feature_norm",
    "feature_cosine",
    "log1p_feature_l2",
    "attention_mean",
    "attention_max",
    "modality_audio",
    "modality_visual",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modalities", nargs="+", choices=tuple(MODALITY_INDEX),
                        default=("text",))
    parser.add_argument("--validation-sessions", nargs="*", default=())
    parser.add_argument("--intervention-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-conversations-per-split", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    parser.add_argument("--renormalize-attention", action="store_true")
    return parser.parse_args(argv)


def predictor_splits(
    split_ids: Mapping[str, Sequence[str]],
    validation_sessions: Sequence[str] = (),
) -> tuple[dict[str, list[str]], dict[str, object]]:
    """Return disjoint train/valid/test conversations without using impacts."""
    train = list(split_ids.get("train", ()))
    valid = list(split_ids.get("valid", ()))
    test = list(split_ids.get("test", ()))
    if not train or not test:
        raise ValueError("D1 requires non-empty checkpoint train and test splits")

    split_source = "checkpoint_valid"
    held_out_sessions: list[str] = []
    if not valid:
        sessions = sorted({session_id(item) for item in train})
        requested = list(dict.fromkeys(validation_sessions))
        held_out_sessions = requested or sessions[-1:]
        unknown = sorted(set(held_out_sessions).difference(sessions))
        if unknown:
            raise ValueError(f"Unknown validation sessions: {unknown}; available={sessions}")
        valid = [item for item in train if session_id(item) in held_out_sessions]
        train = [item for item in train if session_id(item) not in held_out_sessions]
        split_source = "held_out_train_session"

    result = {"train": train, "valid": valid, "test": test}
    sets = {name: set(values) for name, values in result.items()}
    if not all(sets.values()):
        raise ValueError("D1 predictor train/valid/test splits must all be non-empty")
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(f"D1 conversation leakage between {left}/{right}: {sorted(overlap)[:5]}")
    metadata = {
        "validation_source": split_source,
        "held_out_validation_sessions": held_out_sessions,
        "sessions": {
            name: sorted({session_id(item) for item in values})
            for name, values in result.items()
        },
    }
    return result, metadata


def _repeat_batch(batch, count: int):
    return (
        batch.text.repeat(1, count, 1),
        batch.visual.repeat(1, count, 1),
        batch.audio.repeat(1, count, 1),
        batch.mask.repeat(count, 1),
        batch.speakers.repeat(count, 1, 1),
        [batch.length] * count,
    )


def _distribution_features(probabilities: np.ndarray) -> tuple[np.ndarray, ...]:
    ranked = np.sort(probabilities, axis=-1)
    margin = ranked[:, -1] - ranked[:, -2]
    top1 = ranked[:, -1]
    entropy = -np.sum(
        np.where(probabilities > 0.0, probabilities * np.log(
            np.clip(probabilities, 1e-12, 1.0)
        ), 0.0),
        axis=-1,
    )
    prediction = np.argmax(probabilities, axis=-1)
    return margin, top1, entropy, prediction


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def _projected_features(wrapper: ReliabilitySDT, batch) -> tuple[np.ndarray, ...]:
    with torch.inference_mode():
        projected = wrapper._project_inputs(batch.text, batch.visual, batch.audio)
    return tuple(value[0, :batch.length].detach().cpu().numpy() for value in projected)


def _pair_features(
    batch,
    probabilities: np.ndarray,
    distribution: tuple[np.ndarray, ...],
    projected: Sequence[np.ndarray],
    attention: Mapping[str, np.ndarray],
    modality_name: str,
    target: int,
    source: int,
) -> dict[str, float | int]:
    length = batch.length
    modality = MODALITY_INDEX[modality_name]
    margins, top1, entropy, predictions = distribution
    source_feature = projected[modality][source]
    target_feature = projected[modality][target]
    source_norm = float(np.linalg.norm(source_feature))
    target_norm = float(np.linalg.norm(target_feature))
    probability_cosine = _cosine(probabilities[source], probabilities[target])
    branches = source_branches(modality)
    masses = np.asarray([
        attention[name][:, target, source] for name in branches
    ], dtype=np.float64).reshape(-1)
    source_speaker = int(batch.speakers[0, source].argmax().item())
    target_speaker = int(batch.speakers[0, target].argmax().item())
    scale = max(length - 1, 1)
    distance = abs(source - target)
    return {
        "target_margin": float(margins[target]),
        "target_top1_probability": float(top1[target]),
        "target_entropy": float(entropy[target]),
        "source_margin": float(margins[source]),
        "source_top1_probability": float(top1[source]),
        "source_entropy": float(entropy[source]),
        "prediction_agreement": int(predictions[source] == predictions[target]),
        "probability_cosine": probability_cosine,
        "log1p_absolute_distance": math.log1p(distance),
        "normalized_signed_offset": (source - target) / scale,
        "same_speaker": int(source_speaker == target_speaker),
        "source_before_target": int(source < target),
        "normalized_source_position": source / scale,
        "normalized_target_position": target / scale,
        "log1p_conversation_length": math.log1p(length),
        "log1p_source_feature_norm": math.log1p(source_norm),
        "log1p_target_feature_norm": math.log1p(target_norm),
        "feature_cosine": _cosine(source_feature, target_feature),
        "log1p_feature_l2": math.log1p(float(np.linalg.norm(source_feature - target_feature))),
        "attention_mean": float(masses.mean()),
        "attention_max": float(masses.max()),
        "modality_audio": int(modality_name == "audio"),
        "modality_visual": int(modality_name == "visual"),
    }


def _conversation_rows(wrapper, experiment, batch, predictor_split: str,
                       checkpoint_split: str, modalities: Sequence[str],
                       intervention_batch_size: int) -> list[dict[str, object]]:
    length = batch.length
    identity = torch.ones(1, length, length, 3, device=experiment.device)
    with torch.inference_mode():
        baseline = wrapper(
            batch.text, batch.visual, batch.audio, batch.mask, batch.speakers,
            batch.lengths, fixed_edge_gate=identity, capture_attention=True,
        )
    probabilities = baseline["prob"][0, :length].detach().cpu().numpy()
    logits = baseline["logits"][0, :length].detach().cpu().numpy()
    label_tensor = batch.labels[0, :length]
    labels = label_tensor.detach().cpu().numpy().astype(np.int64)
    predictions = logits.argmax(axis=-1)
    positions = torch.arange(length, device=experiment.device)
    baseline_nll = -baseline["log_prob"][0, positions, label_tensor].detach().cpu().numpy()
    distribution = _distribution_features(probabilities)
    projected = _projected_features(wrapper, batch)
    attention = {
        name: values[0][0].detach().cpu().numpy()
        for name, values in baseline["attention_audit"].items()
    }
    rows: list[dict[str, object]] = []

    for modality_name in modalities:
        modality = MODALITY_INDEX[modality_name]
        actions = [
            (target, source)
            for target in range(length)
            for source in range(length)
            if source != target
        ]
        for start in range(0, len(actions), intervention_batch_size):
            chunk = actions[start:start + intervention_batch_size]
            count = len(chunk)
            edge = torch.ones(count, length, length, 3, device=experiment.device)
            for index, (target, source) in enumerate(chunk):
                edge[index, target, source, modality] = 0.0
            with torch.inference_mode():
                output = wrapper(
                    *_repeat_batch(batch, count), fixed_edge_gate=edge,
                    active_branches=source_branches(modality),
                )
            for index, (target, source) in enumerate(chunk):
                label = int(labels[target])
                suppressed_nll = float(-output["log_prob"][index, target, label].cpu())
                suppressed_prediction = int(output["logits"][index, target].argmax().cpu())
                source_speaker = int(batch.speakers[0, source].argmax().item())
                target_speaker = int(batch.speakers[0, target].argmax().item())
                row: dict[str, object] = {
                    "dataset": experiment.dataset.dataset,
                    "checkpoint_id": experiment.checkpoint_id,
                    "checkpoint_split": checkpoint_split,
                    "predictor_split": predictor_split,
                    "conversation_id": batch.conversation_id,
                    "conversation_length": length,
                    "source_idx": source,
                    "target_idx": target,
                    "modality": modality_name,
                    "intervention": "propagation_edge_gate_0",
                    "path_scope": "all_source_branches",
                    "renormalize_attention": int(wrapper.renormalize_attention),
                    "source_label_analysis_only": int(labels[source]),
                    "target_label_analysis_only": label,
                    "source_speaker": source_speaker,
                    "target_speaker": target_speaker,
                    "baseline_prediction": int(predictions[target]),
                    "baseline_correct_analysis_only": int(predictions[target] == label),
                    "suppressed_prediction": suppressed_prediction,
                    "suppressed_correct_analysis_only": int(suppressed_prediction == label),
                    "baseline_nll": float(baseline_nll[target]),
                    "suppressed_nll": suppressed_nll,
                    "signed_impact": suppressed_nll - float(baseline_nll[target]),
                    "changed_prediction": int(suppressed_prediction != predictions[target]),
                }
                row.update(_pair_features(
                    batch, probabilities, distribution, projected, attention, modality_name,
                    target, source,
                ))
                rows.append(row)
    return rows


def _valid_cached_shard(path: Path, conversation: str, expected_rows: int,
                        checkpoint_id: str, predictor_split: str,
                        modalities: Sequence[str],
                        renormalize_attention: bool) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        return (
            len(rows) == expected_rows
            and all(row["conversation_id"] == conversation for row in rows)
            and all(row["checkpoint_id"] == checkpoint_id for row in rows)
            and all(row["predictor_split"] == predictor_split for row in rows)
            and {row["modality"] for row in rows} == set(modalities)
            and all(int(row["renormalize_attention"]) == int(renormalize_attention)
                    for row in rows)
        )
    except (KeyError, OSError, ValueError, csv.Error):
        return False


def _merge_shards(shards: Sequence[Path], output: Path) -> tuple[int, list[str]]:
    if not shards:
        raise ValueError("No D1 shards to merge")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    fieldnames: list[str] | None = None
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = None
        for shard in shards:
            with shard.open("r", newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or ())
                    writer = csv.DictWriter(target, fieldnames=fieldnames)
                    writer.writeheader()
                elif list(reader.fieldnames or ()) != fieldnames:
                    raise ValueError(f"D1 shard schema mismatch: {shard}")
                assert writer is not None
                for row in reader:
                    writer.writerow(row)
                    count += 1
    if not fieldnames or count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("Refusing to create an empty D1 dataset")
    os.replace(temporary, output)
    return count, fieldnames


def prepare(args: argparse.Namespace) -> Path:
    if args.intervention_batch_size < 1 or args.max_conversations_per_split < 0:
        raise ValueError("Invalid D1 batch/conversation limit")
    seed_everything(args.seed)
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    split_map, split_metadata = predictor_splits(
        experiment.split_ids, args.validation_sessions,
    )
    if args.max_conversations_per_split:
        split_map = {
            name: values[:args.max_conversations_per_split]
            for name, values in split_map.items()
        }
        split_metadata["sessions"] = {
            name: sorted({session_id(item) for item in values})
            for name, values in split_map.items()
        }
    checkpoint_membership = {
        conversation: name
        for name, values in experiment.split_ids.items()
        for conversation in values
    }
    wrapper = ReliabilitySDT(
        experiment.model, "B0", renormalize_attention=args.renormalize_attention,
    ).to(experiment.device).eval()
    lookup = dataset_lookup(experiment.dataset)
    root = args.output_dir.resolve() / "D1"
    shards: list[Path] = []
    conversation_lengths: dict[str, int] = {}

    total = sum(len(values) for values in split_map.values())
    completed = 0
    for predictor_split in ("train", "valid", "test"):
        for conversation in split_map[predictor_split]:
            batch = load_conversation(
                experiment.dataset, lookup, conversation, experiment.device,
            )
            expected = len(args.modalities) * batch.length * (batch.length - 1)
            conversation_lengths[conversation] = batch.length
            shard = root / "cache" / predictor_split / f"{conversation}.csv"
            if not args.overwrite and _valid_cached_shard(
                shard, conversation, expected, experiment.checkpoint_id,
                predictor_split, args.modalities, args.renormalize_attention,
            ):
                completed += 1
                shards.append(shard)
                print(f"D1 cached {completed}/{total} {predictor_split} {conversation}",
                      flush=True)
                continue
            rows = _conversation_rows(
                wrapper, experiment, batch, predictor_split,
                checkpoint_membership.get(conversation, "unknown"),
                args.modalities, args.intervention_batch_size,
            )
            if len(rows) != expected:
                raise AssertionError(
                    f"D1 row mismatch for {conversation}: {len(rows)} != {expected}"
                )
            write_csv(shard, rows)
            shards.append(shard)
            completed += 1
            print(f"D1 generated {completed}/{total} {predictor_split} {conversation}",
                  flush=True)

    dataset_path = root / "edge_impact_pairs.csv"
    row_count, columns = _merge_shards(shards, dataset_path)
    split_counts = {
        name: sum(
            len(args.modalities) * conversation_lengths[conversation]
            * (conversation_lengths[conversation] - 1)
            for conversation in values
        )
        for name, values in split_map.items()
    }
    # The row-count identity is a strict no-self-pair/completeness check.
    if row_count != sum(split_counts.values()):
        raise AssertionError(f"Merged D1 rows {row_count} != expected {sum(split_counts.values())}")
    write_json(root / "dataset_manifest.json", {
        "checkpoint_id": experiment.checkpoint_id,
        "checkpoint_sha256": experiment.checkpoint_hash,
        "feature_sha256": experiment.feature_hash,
        "dataset_sha256": sha256(dataset_path),
        "modalities": list(args.modalities),
        "intervention": "propagation_edge_gate_0",
        "path_scope": "all_source_branches",
        "renormalize_attention": bool(args.renormalize_attention),
        "feature_names": list(FEATURE_NAMES),
        "forbidden_predictor_inputs": [
            "source_label_analysis_only", "target_label_analysis_only",
            "baseline_correct_analysis_only", "suppressed_correct_analysis_only",
            "signed_impact", "suppressed_nll", "suppressed_prediction",
        ],
        "split_ids": split_map,
        "split_metadata": split_metadata,
        "row_counts": split_counts,
        "row_count": row_count,
        "columns": columns,
        "smoke_limited": bool(args.max_conversations_per_split),
        "output": str(dataset_path),
    })
    print(f"D1 dataset complete: rows={row_count}; output={dataset_path}", flush=True)
    return dataset_path


def main(argv: Sequence[str] | None = None) -> Path:
    return prepare(parse_args(argv))


if __name__ == "__main__":
    main()
