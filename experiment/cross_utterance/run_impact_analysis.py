"""Run one frozen cross-utterance experiment task at a time.

Task A creates and validates the baseline manifest.  Task B performs zero
replacement.  Task C computes training-split feature means and performs mean
replacement.  The shell entry points intentionally invoke these tasks
separately.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from impact_core import (
    IMPACT_FIELDS,
    MODALITIES,
    STRATEGIES,
    apply_replacement,
    build_impact_rows,
    compute_training_means,
    conversation_shard_name,
    dataset_lookup,
    expected_pair_count,
    forward_conversation,
    load_conversation,
    load_experiment,
    merge_csv_files,
    read_json,
    save_means_atomic,
    seed_everything,
    validate_impact_csv,
    write_csv_atomic,
    write_json_atomic,
)


BASELINE_PREFIX_FIELDS = [
    "dataset", "split", "checkpoint_id", "conversation_id",
    "conversation_length", "utterance_index", "speaker", "label",
    "prediction", "correct", "log_prob_true", "prob_true",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("baseline", "zero", "train_mean"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=("test", "valid"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--allow-smoke-checkpoint", action="store_true")
    # Batched padded evaluation and B=1 dialogue evaluation can select
    # different CUDA kernels. Their float32 probabilities may consequently
    # differ by a few e-6 even when logits, predictions, and metrics are
    # semantically equivalent.
    parser.add_argument(
        "--prob-atol", type=float, default=2e-4,
        help="absolute tolerance for original-batch versus B=1 CUDA probabilities",
    )
    parser.add_argument(
        "--prob-rtol", type=float, default=2e-4,
        help="relative tolerance for original-batch versus B=1 CUDA probabilities",
    )
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _selected_ids(experiment, split: str, maximum: int) -> List[str]:
    if split not in experiment.split_ids or not experiment.split_ids[split]:
        raise ValueError(f"Checkpoint split {split!r} is empty or unavailable")
    ids = list(experiment.split_ids[split])
    if maximum:
        if maximum < 1:
            raise ValueError("--max-conversations must be nonnegative")
        ids = ids[:maximum]
    return ids


def _baseline_fieldnames(n_classes: int) -> List[str]:
    return BASELINE_PREFIX_FIELDS + [f"prob_{index}" for index in range(n_classes)]


def _baseline_rows_for_conversation(experiment, split: str, batch, output):
    length = batch.length
    labels = batch.labels[0, :length]
    speakers = batch.speakers[0, :length].argmax(dim=-1)
    probabilities = output["prob"][0, :length]
    log_probabilities = output["log_prob"][0, :length]
    predictions = output["logits"][0, :length].argmax(dim=-1)
    rows = []
    for index in range(length):
        label = int(labels[index].item())
        prediction = int(predictions[index].item())
        row = {
            "dataset": experiment.dataset.dataset,
            "split": split,
            "checkpoint_id": experiment.checkpoint_id,
            "conversation_id": batch.conversation_id,
            "conversation_length": length,
            "utterance_index": index,
            "speaker": int(speakers[index].item()),
            "label": label,
            "prediction": prediction,
            "correct": int(label == prediction),
            "log_prob_true": float(log_probabilities[index, label].cpu()),
            "prob_true": float(probabilities[index, label].cpu()),
        }
        row.update(
            {f"prob_{class_index}": float(value.cpu())
             for class_index, value in enumerate(probabilities[index])}
        )
        rows.append(row)
    return rows


def _canonical_repository_baseline(
    experiment, selected_ids, split, batch_size_override=None
):
    """Run the repository evaluator so Task A has an independent reference."""
    from dataloader import make_loaders
    from train import build_parser, make_criterion, run_epoch

    defaults = build_parser().parse_args([])
    for key, value in experiment.checkpoint.get("args", {}).items():
        setattr(defaults, key, value)
    defaults.dataset = experiment.dataset.dataset
    defaults.feature_path = str(experiment.feature_path)
    batch_size = (
        int(batch_size_override)
        if batch_size_override is not None
        else int(getattr(defaults, "batch_size", 16))
    )
    split_ids = {name: list(ids) for name, ids in experiment.split_ids.items()}
    split_ids[split] = list(selected_ids)
    loaders = make_loaders(
        experiment.dataset, split_ids, batch_size, int(getattr(defaults, "seed", 2024)),
        num_workers=0, pin_memory=experiment.device.type == "cuda",
    )
    criterion = make_criterion(defaults, experiment.device)
    return run_epoch(
        experiment.model, criterion, loaders[split], experiment.device,
        collect_predictions=True, epoch=int(experiment.checkpoint.get("epoch", 0)),
    )


def _compare_baselines(
    canonical_rows: Sequence[Mapping[str, object]],
    dialogue_rows: Sequence[Mapping[str, object]],
    n_classes: int,
    atol: float,
    rtol: float,
) -> Dict[str, object]:
    def row_key(row):
        conversation_id = row.get("dialogue_id", row.get("conversation_id"))
        if conversation_id is None:
            raise KeyError("Baseline row has no dialogue/conversation ID")
        return str(conversation_id), int(row["utterance_index"])

    canonical = {row_key(row): row for row in canonical_rows}
    independent = {row_key(row): row for row in dialogue_rows}
    if canonical.keys() != independent.keys():
        missing = sorted(canonical.keys() - independent.keys())[:5]
        extra = sorted(independent.keys() - canonical.keys())[:5]
        raise AssertionError(f"Baseline key mismatch; missing={missing}, extra={extra}")
    maximum_difference = 0.0
    for key, reference in canonical.items():
        actual = independent[key]
        if int(reference["label"]) != int(actual["label"]):
            raise AssertionError(f"Baseline label mismatch at {key}")
        if int(reference["prediction"]) != int(actual["prediction"]):
            raise AssertionError(f"Baseline prediction mismatch at {key}")
        for class_index in range(n_classes):
            ref_prob = float(reference[f"prob_{class_index}"])
            actual_prob = float(actual[f"prob_{class_index}"])
            difference = abs(ref_prob - actual_prob)
            maximum_difference = max(maximum_difference, difference)
            if not np.isclose(ref_prob, actual_prob, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"Baseline probability mismatch at {key}, class {class_index}: "
                    f"{ref_prob} versus {actual_prob}"
                )
    return {
        "rows_compared": len(canonical),
        "maximum_probability_difference": maximum_difference,
        "probability_atol": atol,
        "probability_rtol": rtol,
        "predictions_identical": True,
    }


def _classification_metrics(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    labels = [int(row["label"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    return {
        "accuracy": 100.0 * accuracy_score(labels, predictions),
        "weighted_f1": 100.0 * f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "macro_f1": 100.0 * f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "utterances": len(labels),
    }


def _environment_text(experiment) -> str:
    lines = [
        f"created_utc={_utc_now()}",
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"numpy={np.__version__}",
        f"device={experiment.device}",
        f"cuda_available={torch.cuda.is_available()}",
        f"checkpoint={experiment.checkpoint_path}",
        f"checkpoint_sha256={experiment.checkpoint_hash}",
        f"feature_path={experiment.feature_path}",
        f"feature_sha256={experiment.feature_hash}",
    ]
    if experiment.device.type == "cuda":
        lines.append(f"cuda_device_name={torch.cuda.get_device_name(experiment.device)}")
    return "\n".join(lines) + "\n"


def run_baseline(args, experiment, run_dir: Path) -> None:
    selected_ids = _selected_ids(experiment, args.split, args.max_conversations)
    lookup = dataset_lookup(experiment.dataset)
    first_batch = load_conversation(
        experiment.dataset, lookup, selected_ids[0], experiment.device
    )
    first = forward_conversation(experiment.model, first_batch)
    second = forward_conversation(experiment.model, first_batch)
    maximum_deterministic_difference = float(
        (first["logits"] - second["logits"]).abs().max().cpu()
    )
    torch.testing.assert_close(
        first["logits"], second["logits"], atol=args.prob_atol, rtol=args.prob_rtol
    )

    print("Running repository baseline evaluator with checkpoint batch size...", flush=True)
    canonical_metrics, canonical_rows, _, _ = _canonical_repository_baseline(
        experiment, selected_ids, args.split
    )
    print("Running repository baseline evaluator with batch size 1...", flush=True)
    single_metrics, single_rows, _, _ = _canonical_repository_baseline(
        experiment, selected_ids, args.split, batch_size_override=1
    )

    baseline_rows = []
    conversation_entries = []
    estimated_seconds_per_strategy = 0.0
    for index, conversation_id in enumerate(selected_ids):
        batch = load_conversation(
            experiment.dataset, lookup, conversation_id, experiment.device
        )
        _synchronize(experiment.device)
        started = time.perf_counter()
        output = forward_conversation(experiment.model, batch)
        _synchronize(experiment.device)
        elapsed = time.perf_counter() - started
        estimated_seconds_per_strategy += elapsed * 3 * batch.length
        baseline_rows.extend(
            _baseline_rows_for_conversation(experiment, args.split, batch, output)
        )
        conversation_entries.append({
            "index": index,
            "conversation_id": conversation_id,
            "length": batch.length,
            "shard": conversation_shard_name(index, conversation_id),
        })
        print(
            f"Baseline {index + 1}/{len(selected_ids)}: {conversation_id} "
            f"(T={batch.length}, {elapsed:.4f}s)",
            flush=True,
        )

    batch_shape_equivalence = _compare_baselines(
        canonical_rows, single_rows, experiment.dataset.n_classes,
        args.prob_atol, args.prob_rtol,
    )
    dialogue_pipeline_equivalence = _compare_baselines(
        single_rows, baseline_rows, experiment.dataset.n_classes,
        2e-7, 2e-6,
    )
    independent_metrics = _classification_metrics(baseline_rows)
    for name in ("accuracy", "weighted_f1", "macro_f1"):
        if not np.isclose(
            float(canonical_metrics[name]), independent_metrics[name], atol=1e-10, rtol=0
        ):
            raise AssertionError(f"Baseline metric mismatch for {name}")
        if not np.isclose(
            float(single_metrics[name]), independent_metrics[name], atol=1e-10, rtol=0
        ):
            raise AssertionError(f"Batch-size-1 baseline metric mismatch for {name}")

    equivalence = {
        "original_batch_vs_batch_size_1": batch_shape_equivalence,
        "batch_size_1_vs_impact_pipeline": dialogue_pipeline_equivalence,
        "classification_metrics_identical": True,
    }

    expected_rows = sum(
        expected_pair_count(int(entry["length"])) for entry in conversation_entries
    )
    manifest = {
        "format_version": 1,
        "created_utc": _utc_now(),
        "task_a_complete": True,
        "dataset": experiment.dataset.dataset,
        "split": args.split,
        "seed": args.seed,
        "checkpoint_path": str(experiment.checkpoint_path),
        "checkpoint_sha256": experiment.checkpoint_hash,
        "checkpoint_id": experiment.checkpoint_id,
        "checkpoint_epoch": int(experiment.checkpoint.get("epoch", 0)),
        "checkpoint_is_smoke": bool(
            experiment.checkpoint.get("args", {}).get("max_batches", 0)
        ),
        "feature_path": str(experiment.feature_path),
        "feature_sha256": experiment.feature_hash,
        "model_config": experiment.checkpoint["model_config"],
        "intervention_scope": "all_valid_source_target_pairs",
        "self_impacts_included": False,
        "conversations": conversation_entries,
        "conversation_count": len(conversation_entries),
        "utterance_count": sum(int(item["length"]) for item in conversation_entries),
        "expected_rows_per_strategy": expected_rows,
        "expected_intervention_forwards_per_strategy": 3 * sum(
            int(item["length"]) for item in conversation_entries
        ),
        "baseline_equivalence": equivalence,
        "determinism": {
            "conversation_id": selected_ids[0],
            "maximum_logit_difference": maximum_deterministic_difference,
            "passed": True,
        },
        "runtime_estimate": {
            "seconds_per_strategy_from_baseline_timings": estimated_seconds_per_strategy,
            "minutes_per_strategy_from_baseline_timings": estimated_seconds_per_strategy / 60,
            "note": "Estimate excludes CSV writing and validation overhead.",
        },
    }

    baseline_dir = run_dir / "baseline"
    write_csv_atomic(
        baseline_dir / "baseline_predictions.csv", baseline_rows,
        _baseline_fieldnames(experiment.dataset.n_classes),
    )
    write_json_atomic(baseline_dir / "baseline_metrics.json", {
        "repository_evaluator_original_batch": canonical_metrics,
        "repository_evaluator_batch_size_1": single_metrics,
        "impact_pipeline_per_conversation": independent_metrics,
    })
    write_json_atomic(baseline_dir / "baseline_equivalence.json", equivalence)
    write_json_atomic(run_dir / "experiment_manifest.json", manifest)
    write_json_atomic(run_dir / "config_used.json", {
        "task": "baseline",
        "arguments": vars(args),
        "model_config": experiment.checkpoint["model_config"],
    })
    (run_dir / "environment_info.txt").write_text(
        _environment_text(experiment), encoding="utf-8"
    )
    print(
        f"Task A complete: {len(conversation_entries)} conversations, "
        f"{len(baseline_rows)} utterances; manifest={run_dir / 'experiment_manifest.json'}",
        flush=True,
    )


def _load_and_validate_manifest(args, experiment, run_dir: Path):
    path = run_dir / "experiment_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Task A manifest is missing; run Task A first")
    manifest = read_json(path)
    checks = {
        "checkpoint_sha256": experiment.checkpoint_hash,
        "feature_sha256": experiment.feature_hash,
        "dataset": experiment.dataset.dataset,
        "split": args.split,
        "seed": args.seed,
    }
    for key, actual in checks.items():
        if manifest.get(key) != actual:
            raise ValueError(
                f"Task configuration differs from Task A for {key}: "
                f"manifest={manifest.get(key)!r}, current={actual!r}"
            )
    if not manifest.get("task_a_complete"):
        raise ValueError("Task A did not complete successfully")
    return manifest


def _load_baseline_index(run_dir: Path) -> Dict[tuple[str, int], Mapping[str, str]]:
    path = run_dir / "baseline" / "baseline_predictions.csv"
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    index = {
        (row["conversation_id"], int(row["utterance_index"])): row for row in rows
    }
    if len(index) != len(rows):
        raise ValueError("Duplicate baseline prediction keys")
    return index


def _validate_recomputed_baseline(
    baseline_index, conversation_id: str, output, labels, length: int, atol: float, rtol: float
) -> None:
    probabilities = output["prob"][0, :length]
    predictions = output["logits"][0, :length].argmax(dim=-1)
    for index in range(length):
        saved = baseline_index[(conversation_id, index)]
        if int(predictions[index].item()) != int(saved["prediction"]):
            raise AssertionError(f"Recomputed baseline prediction mismatch at {conversation_id}:{index}")
        label = int(labels[0, index].item())
        actual = float(probabilities[index, label].cpu())
        if not np.isclose(actual, float(saved["prob_true"]), atol=atol, rtol=rtol):
            raise AssertionError(f"Recomputed baseline probability mismatch at {conversation_id}:{index}")


def _load_or_compute_means(experiment, run_dir: Path):
    means, metadata = compute_training_means(
        experiment.dataset, experiment.split_ids["train"]
    )
    metadata.update({
        "created_utc": _utc_now(),
        "dataset": experiment.dataset.dataset,
        "checkpoint_id": experiment.checkpoint_id,
        "checkpoint_sha256": experiment.checkpoint_hash,
        "feature_sha256": experiment.feature_hash,
        "split": "train",
        "train_conversation_ids": list(experiment.split_ids["train"]),
    })
    feature_dir = run_dir / "feature_means"
    save_means_atomic(feature_dir / "train_feature_means.npz", means)
    write_json_atomic(feature_dir / "train_feature_mean_metadata.json", metadata)
    return means


def run_interventions(args, experiment, run_dir: Path, strategy: str) -> None:
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    manifest = _load_and_validate_manifest(args, experiment, run_dir)
    baseline_index = _load_baseline_index(run_dir)
    replacements = None
    if strategy == "train_mean":
        replacements = _load_or_compute_means(experiment, run_dir)

    lookup = dataset_lookup(experiment.dataset)
    shard_dir = run_dir / "shards" / strategy
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    started_task = time.perf_counter()
    for ordinal, entry in enumerate(manifest["conversations"]):
        conversation_id = entry["conversation_id"]
        length = int(entry["length"])
        shard_path = shard_dir / entry["shard"]
        expected_rows = expected_pair_count(length)
        if shard_path.is_file():
            try:
                validate_impact_csv(
                    shard_path, strategy, expected_rows, conversation_id
                )
                shard_paths.append(shard_path)
                print(
                    f"{strategy} {ordinal + 1}/{len(manifest['conversations'])}: "
                    f"resume valid shard {conversation_id}",
                    flush=True,
                )
                continue
            except (ValueError, OSError):
                print(f"Recomputing invalid shard: {shard_path}", flush=True)

        batch = load_conversation(
            experiment.dataset, lookup, conversation_id, experiment.device
        )
        if batch.length != length:
            raise ValueError(f"Conversation length changed for {conversation_id}")
        metadata_snapshot = (
            batch.speakers.clone(), batch.mask.clone(), batch.labels.clone()
        )
        full_output = forward_conversation(experiment.model, batch)
        _validate_recomputed_baseline(
            baseline_index, conversation_id, full_output, batch.labels, length,
            args.prob_atol, args.prob_rtol,
        )
        rows = []
        conversation_started = time.perf_counter()
        for modality in MODALITIES:
            original_feature = batch.features[modality]
            original_snapshot = original_feature.clone()
            if strategy == "zero":
                replacement = torch.zeros(
                    original_feature.size(-1), device=experiment.device,
                    dtype=original_feature.dtype,
                )
            else:
                replacement = torch.from_numpy(replacements[modality]).to(
                    device=experiment.device, dtype=original_feature.dtype
                )
            for source_idx in range(length):
                modified = apply_replacement(
                    original_feature, source_idx, replacement
                )
                masked_output = forward_conversation(
                    experiment.model, batch, {modality: modified}
                )
                rows.extend(build_impact_rows(
                    dataset_name=experiment.dataset.dataset,
                    split=args.split,
                    checkpoint_id=experiment.checkpoint_id,
                    seed=args.seed,
                    batch=batch,
                    source_idx=source_idx,
                    modality=modality,
                    strategy=strategy,
                    original_feature=original_feature,
                    replacement=replacement,
                    full_output=full_output,
                    masked_output=masked_output,
                ))
            torch.testing.assert_close(
                original_feature, original_snapshot, rtol=0, atol=0
            )

        for actual, expected in zip(
            (batch.speakers, batch.mask, batch.labels), metadata_snapshot
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if len(rows) != expected_rows:
            raise AssertionError(
                f"{conversation_id}: produced {len(rows)} rows, expected {expected_rows}"
            )
        write_csv_atomic(shard_path, rows, IMPACT_FIELDS)
        validate_impact_csv(shard_path, strategy, expected_rows, conversation_id)
        shard_paths.append(shard_path)
        elapsed = time.perf_counter() - conversation_started
        print(
            f"{strategy} {ordinal + 1}/{len(manifest['conversations'])}: "
            f"{conversation_id} T={length}, rows={len(rows)}, {elapsed:.1f}s",
            flush=True,
        )

    result_path = run_dir / "results" / f"impact_pairs_{strategy}.csv"
    merged_rows = merge_csv_files(shard_paths, result_path)
    expected_total = int(manifest["expected_rows_per_strategy"])
    if merged_rows != expected_total:
        raise AssertionError(f"Merged {merged_rows} rows, expected {expected_total}")
    validation = validate_impact_csv(result_path, strategy, expected_total)
    validation.update({
        "completed_utc": _utc_now(),
        "elapsed_seconds": time.perf_counter() - started_task,
        "checkpoint_id": experiment.checkpoint_id,
        "conversation_count": len(manifest["conversations"]),
    })
    write_json_atomic(
        run_dir / "results" / f"{strategy}_validation.json", validation
    )
    print(f"Task {strategy} complete: {result_path} ({merged_rows} rows)", flush=True)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.num_threads:
        if args.num_threads < 1:
            raise ValueError("--num-threads must be positive")
        torch.set_num_threads(args.num_threads)
    if args.max_conversations < 0:
        raise ValueError("--max-conversations must be nonnegative")
    if min(args.prob_atol, args.prob_rtol) < 0:
        raise ValueError("Probability tolerances must be nonnegative")
    seed_everything(args.seed)
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    experiment = load_experiment(
        args.checkpoint, args.feature_path, args.device, args.gpu_id,
        args.allow_smoke_checkpoint,
    )
    print(
        f"Task={args.task}; dataset={experiment.dataset.dataset}; "
        f"device={experiment.device}; checkpoint={experiment.checkpoint_id}",
        flush=True,
    )
    if args.task == "baseline":
        run_baseline(args, experiment, run_dir)
    elif args.task == "zero":
        run_interventions(args, experiment, run_dir, "zero")
    else:
        run_interventions(args, experiment, run_dir, "train_mean")
    return run_dir


if __name__ == "__main__":
    main()
