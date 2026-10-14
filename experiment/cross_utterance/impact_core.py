"""Shared primitives for frozen cross-utterance impact analysis.

The intervention point is the raw utterance-level modality tensor passed to
``Transformer_Based_Model.forward``.  This module deliberately contains no
training or optimizer code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SDT_ROOT = HERE.parents[1]
if str(SDT_ROOT) not in sys.path:
    sys.path.insert(0, str(SDT_ROOT))

from dataloader import DialogueDataset  # noqa: E402
from model import Transformer_Based_Model  # noqa: E402


MODALITIES = ("text", "audio", "visual")
STRATEGIES = ("zero", "train_mean")
FEATURE_DIM_KEYS = {
    "text": "D_text",
    "audio": "D_audio",
    "visual": "D_visual",
}

IMPACT_FIELDS = [
    "dataset", "split", "checkpoint_id", "seed",
    "conversation_id", "conversation_length",
    "source_idx", "target_idx", "source_speaker", "target_speaker",
    "same_speaker", "signed_offset", "absolute_distance",
    "source_before_target", "modality", "replacement_strategy",
    "source_emotion", "target_emotion",
    "loss_full", "loss_mask", "signed_impact",
    "prob_full_true", "prob_mask_true", "delta_true_class_prob",
    "pred_full", "pred_mask", "correct_full", "correct_mask",
    "changed_prediction", "js_divergence",
    "source_feature_norm", "replacement_vector_norm",
    "replacement_delta_l2",
]


@dataclass(frozen=True)
class LoadedExperiment:
    checkpoint_path: Path
    checkpoint: Mapping[str, object]
    checkpoint_hash: str
    checkpoint_id: str
    feature_path: Path
    feature_hash: str
    dataset: DialogueDataset
    model: Transformer_Based_Model
    device: torch.device
    split_ids: Mapping[str, List[str]]


@dataclass
class ConversationBatch:
    conversation_id: str
    text: torch.Tensor
    visual: torch.Tensor
    audio: torch.Tensor
    speakers: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.mask.sum().item())

    @property
    def lengths(self) -> List[int]:
        return [self.length]

    @property
    def features(self) -> Dict[str, torch.Tensor]:
        return {"text": self.text, "audio": self.audio, "visual": self.visual}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Some Windows CPU builds import an unavailable POSIX ``pwd`` module while
    # enabling this global switch.  Eval mode plus the repeated-forward gate in
    # Task A remains authoritative when that optional switch is unavailable.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (ImportError, OSError):
        pass


def resolve_device(name: str, gpu_id: int) -> torch.device:
    if name == "cpu" or (name == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    return torch.device("cuda", gpu_id)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_torch_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility.
        return torch.load(path, map_location="cpu")


def _resolve_feature_path(
    checkpoint: Mapping[str, object], feature_path: str | None
) -> Path:
    if feature_path:
        resolved = Path(feature_path).expanduser().resolve()
    else:
        saved = checkpoint.get("feature_path")
        if saved and Path(str(saved)).is_file():
            resolved = Path(str(saved)).resolve()
        else:
            dataset_name = str(checkpoint["model_config"]["dataset"]).lower()
            candidates = [
                SDT_ROOT / "data" / f"{dataset_name}_multimodal_features.pkl",
                SDT_ROOT.parent / "SDT" / "data" / f"{dataset_name}_multimodal_features.pkl",
            ]
            resolved = next((item.resolve() for item in candidates if item.is_file()), None)
            if resolved is None:
                raise FileNotFoundError(
                    "Feature pickle is unavailable; pass --feature-path explicitly"
                )
    if not resolved.is_file():
        raise FileNotFoundError(f"Feature pickle not found: {resolved}")
    return resolved


def load_experiment(
    checkpoint_path: str,
    feature_path: str | None,
    device_name: str,
    gpu_id: int,
    allow_smoke_checkpoint: bool,
) -> LoadedExperiment:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    checkpoint = load_torch_checkpoint(checkpoint_file)
    required = {"model_state_dict", "model_config", "args", "epoch", "split_ids"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")
    checkpoint_args = checkpoint.get("args", {})
    is_smoke = bool(checkpoint_args.get("max_batches", 0))
    if is_smoke and not allow_smoke_checkpoint:
        raise ValueError(
            "This checkpoint was created with --max-batches and is a smoke checkpoint. "
            "Pass --allow-smoke-checkpoint only for implementation testing."
        )

    resolved_feature = _resolve_feature_path(checkpoint, feature_path)
    model_config = checkpoint["model_config"]
    dataset = DialogueDataset(model_config["dataset"], str(resolved_feature))
    for dimension_key, actual in dataset.feature_dims.items():
        if int(model_config[dimension_key]) != int(actual):
            raise ValueError(
                f"Checkpoint {dimension_key}={model_config[dimension_key]} but dataset has {actual}"
            )

    split_ids = {name: list(values) for name, values in checkpoint["split_ids"].items()}
    dataset_ids = set(dataset.keys)
    for split_name, ids in split_ids.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate conversation IDs in checkpoint split {split_name}")
        unknown = sorted(set(ids).difference(dataset_ids))
        if unknown:
            raise ValueError(
                f"Checkpoint split {split_name} contains unknown IDs: {unknown[:5]}"
            )

    device = resolve_device(device_name, gpu_id)
    model = Transformer_Based_Model(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if hasattr(model, "set_tical_epoch"):
        model.set_tical_epoch(int(checkpoint.get("epoch", 0)))
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Not all model parameters were frozen")

    checkpoint_hash = sha256_file(checkpoint_file)
    feature_hash = sha256_file(resolved_feature)
    return LoadedExperiment(
        checkpoint_path=checkpoint_file,
        checkpoint=checkpoint,
        checkpoint_hash=checkpoint_hash,
        checkpoint_id=checkpoint_hash[:16],
        feature_path=resolved_feature,
        feature_hash=feature_hash,
        dataset=dataset,
        model=model,
        device=device,
        split_ids=split_ids,
    )


def dataset_lookup(dataset: DialogueDataset) -> Dict[str, int]:
    return {conversation_id: index for index, conversation_id in enumerate(dataset.keys)}


def load_conversation(
    dataset: DialogueDataset,
    lookup: Mapping[str, int],
    conversation_id: str,
    device: torch.device,
) -> ConversationBatch:
    text, visual, audio, speakers, mask, labels, loaded_id = dataset[lookup[conversation_id]]
    if loaded_id != conversation_id:
        raise AssertionError("Dataset lookup returned the wrong conversation")
    return ConversationBatch(
        conversation_id=conversation_id,
        text=text.unsqueeze(1).to(device),
        visual=visual.unsqueeze(1).to(device),
        audio=audio.unsqueeze(1).to(device),
        speakers=speakers.unsqueeze(0).to(device),
        mask=mask.unsqueeze(0).to(device),
        labels=labels.unsqueeze(0).to(device),
    )


def forward_conversation(
    model: Transformer_Based_Model,
    batch: ConversationBatch,
    feature_overrides: Mapping[str, torch.Tensor] | None = None,
) -> Mapping[str, torch.Tensor]:
    features = batch.features
    if feature_overrides:
        features.update(feature_overrides)
    with torch.inference_mode():
        return model(
            features["text"], features["visual"], features["audio"],
            batch.mask, batch.speakers, batch.lengths,
        )


def apply_replacement(
    original: torch.Tensor, source_idx: int, replacement: torch.Tensor
) -> torch.Tensor:
    """Return a clone with exactly one utterance row replaced."""
    if original.ndim != 3 or original.size(1) != 1:
        raise ValueError("Expected a single-conversation feature tensor [T,1,D]")
    if not 0 <= source_idx < original.size(0):
        raise IndexError("source_idx is outside the valid conversation")
    replacement = replacement.to(device=original.device, dtype=original.dtype)
    if replacement.shape != original[source_idx, 0].shape:
        raise ValueError(
            f"Replacement shape {tuple(replacement.shape)} does not match "
            f"feature row {tuple(original[source_idx, 0].shape)}"
        )
    modified = original.clone()
    modified[source_idx, 0] = replacement
    if source_idx:
        torch.testing.assert_close(modified[:source_idx], original[:source_idx], rtol=0, atol=0)
    if source_idx + 1 < original.size(0):
        torch.testing.assert_close(
            modified[source_idx + 1 :], original[source_idx + 1 :], rtol=0, atol=0
        )
    torch.testing.assert_close(modified[source_idx, 0], replacement, rtol=0, atol=0)
    return modified


def js_divergence_from_log_probs(
    first_log_prob: torch.Tensor, second_log_prob: torch.Tensor
) -> torch.Tensor:
    if first_log_prob.shape != second_log_prob.shape:
        raise ValueError("JS divergence inputs must have identical shapes")
    log_mixture = torch.logaddexp(first_log_prob, second_log_prob) - math.log(2.0)
    first_prob = first_log_prob.exp()
    second_prob = second_log_prob.exp()
    return 0.5 * (
        (first_prob * (first_log_prob - log_mixture)).sum(dim=-1)
        + (second_prob * (second_log_prob - log_mixture)).sum(dim=-1)
    )


def signed_impact(
    full_log_prob_true: torch.Tensor, masked_log_prob_true: torch.Tensor
) -> torch.Tensor:
    return full_log_prob_true - masked_log_prob_true


def expected_pair_count(length: int, modalities: int = 3) -> int:
    if length < 0:
        raise ValueError("Conversation length cannot be negative")
    return modalities * length * max(0, length - 1)


def _as_number(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def build_impact_rows(
    *,
    dataset_name: str,
    split: str,
    checkpoint_id: str,
    seed: int,
    batch: ConversationBatch,
    source_idx: int,
    modality: str,
    strategy: str,
    original_feature: torch.Tensor,
    replacement: torch.Tensor,
    full_output: Mapping[str, torch.Tensor],
    masked_output: Mapping[str, torch.Tensor],
) -> List[Dict[str, object]]:
    if modality not in MODALITIES or strategy not in STRATEGIES:
        raise ValueError("Unknown modality or replacement strategy")
    length = batch.length
    labels = batch.labels[0, :length]
    speakers = batch.speakers[0, :length].argmax(dim=-1)
    full_log_prob = full_output["log_prob"][0, :length]
    mask_log_prob = masked_output["log_prob"][0, :length]
    full_prob = full_output["prob"][0, :length]
    mask_prob = masked_output["prob"][0, :length]
    full_pred = full_output["logits"][0, :length].argmax(dim=-1)
    mask_pred = masked_output["logits"][0, :length].argmax(dim=-1)
    target_indices = torch.arange(length, device=labels.device)
    full_true_log = full_log_prob[target_indices, labels]
    mask_true_log = mask_log_prob[target_indices, labels]
    full_true_prob = full_prob[target_indices, labels]
    mask_true_prob = mask_prob[target_indices, labels]
    impacts = signed_impact(full_true_log, mask_true_log)
    divergences = js_divergence_from_log_probs(full_log_prob, mask_log_prob)

    source_feature = original_feature[source_idx, 0]
    replacement = replacement.to(source_feature)
    source_norm = _as_number(source_feature.norm())
    replacement_norm = _as_number(replacement.norm())
    replacement_delta = _as_number((source_feature - replacement).norm())
    source_speaker = int(speakers[source_idx].item())
    source_emotion = int(labels[source_idx].item())

    rows: List[Dict[str, object]] = []
    for target_idx in range(length):
        if target_idx == source_idx:
            continue
        target_label = int(labels[target_idx].item())
        pred_full = int(full_pred[target_idx].item())
        pred_mask = int(mask_pred[target_idx].item())
        target_speaker = int(speakers[target_idx].item())
        offset = source_idx - target_idx
        row = {
            "dataset": dataset_name,
            "split": split,
            "checkpoint_id": checkpoint_id,
            "seed": seed,
            "conversation_id": batch.conversation_id,
            "conversation_length": length,
            "source_idx": source_idx,
            "target_idx": target_idx,
            "source_speaker": source_speaker,
            "target_speaker": target_speaker,
            "same_speaker": int(source_speaker == target_speaker),
            "signed_offset": offset,
            "absolute_distance": abs(offset),
            "source_before_target": int(source_idx < target_idx),
            "modality": modality,
            "replacement_strategy": strategy,
            "source_emotion": source_emotion,
            "target_emotion": target_label,
            "loss_full": _as_number(-full_true_log[target_idx]),
            "loss_mask": _as_number(-mask_true_log[target_idx]),
            "signed_impact": _as_number(impacts[target_idx]),
            "prob_full_true": _as_number(full_true_prob[target_idx]),
            "prob_mask_true": _as_number(mask_true_prob[target_idx]),
            "delta_true_class_prob": _as_number(
                full_true_prob[target_idx] - mask_true_prob[target_idx]
            ),
            "pred_full": pred_full,
            "pred_mask": pred_mask,
            "correct_full": int(pred_full == target_label),
            "correct_mask": int(pred_mask == target_label),
            "changed_prediction": int(pred_full != pred_mask),
            "js_divergence": _as_number(divergences[target_idx]),
            "source_feature_norm": source_norm,
            "replacement_vector_norm": replacement_norm,
            "replacement_delta_l2": replacement_delta,
        }
        rows.append(row)
    return rows


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv_atomic(
    path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def conversation_shard_name(index: int, conversation_id: str) -> str:
    suffix = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:10]
    return f"{index:04d}_{suffix}.csv"


def validate_impact_csv(
    path: Path,
    strategy: str,
    expected_rows: int | None = None,
    conversation_id: str | None = None,
) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    count = 0
    keys = set()
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != IMPACT_FIELDS:
            raise ValueError(f"Unexpected columns in {path}")
        for row in reader:
            count += 1
            if row["replacement_strategy"] != strategy:
                raise ValueError(f"Wrong strategy in {path}: {row['replacement_strategy']}")
            if conversation_id is not None and row["conversation_id"] != conversation_id:
                raise ValueError(f"Wrong conversation ID in {path}")
            source_idx, target_idx = int(row["source_idx"]), int(row["target_idx"])
            if source_idx == target_idx:
                raise ValueError(f"Self-impact row found in {path}")
            key = (row["conversation_id"], source_idx, target_idx, row["modality"])
            if key in keys:
                raise ValueError(f"Duplicate impact key in {path}: {key}")
            keys.add(key)
            for field in (
                "loss_full", "loss_mask", "signed_impact", "prob_full_true",
                "prob_mask_true", "delta_true_class_prob", "js_divergence",
            ):
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"Non-finite {field} in {path}")
            difference = float(row["loss_mask"]) - float(row["loss_full"])
            if not math.isclose(
                float(row["signed_impact"]), difference, rel_tol=2e-6, abs_tol=2e-6
            ):
                raise ValueError(f"Impact sign/equation mismatch in {path}")
    if expected_rows is not None and count != expected_rows:
        raise ValueError(f"{path} has {count} rows; expected {expected_rows}")
    return {"path": str(path), "rows": count, "unique_keys": len(keys)}


def merge_csv_files(paths: Sequence[Path], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=IMPACT_FIELDS)
        writer.writeheader()
        for path in paths:
            with path.open("r", newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != IMPACT_FIELDS:
                    raise ValueError(f"Unexpected columns in shard {path}")
                for row in reader:
                    writer.writerow(row)
                    count += 1
    os.replace(temporary, destination)
    return count


def compute_training_means(
    dataset: DialogueDataset,
    train_ids: Sequence[str],
) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    lookup = dataset_lookup(dataset)
    sums = {
        name: np.zeros(dataset.feature_dims[FEATURE_DIM_KEYS[name]], dtype=np.float64)
        for name in MODALITIES
    }
    squared_sums = {name: 0.0 for name in MODALITIES}
    utterance_count = 0
    for conversation_id in train_ids:
        text, visual, audio, _, mask, _, _ = dataset[lookup[conversation_id]]
        values = {"text": text, "audio": audio, "visual": visual}
        valid_count = int(mask.sum().item())
        utterance_count += valid_count
        for name, tensor in values.items():
            array = tensor[:valid_count].numpy().astype(np.float64, copy=False)
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite {name} feature in {conversation_id}")
            sums[name] += array.sum(axis=0)
            squared_sums[name] += float(np.square(array).sum())
    if not utterance_count:
        raise ValueError("Training split has no valid utterances")

    means = {name: (value / utterance_count).astype(np.float32) for name, value in sums.items()}
    modality_metadata = {}
    for name, mean in means.items():
        feature_rms = math.sqrt(
            squared_sums[name] / (utterance_count * int(mean.size))
        )
        mean_rms = float(np.sqrt(np.mean(np.square(mean.astype(np.float64)))))
        modality_metadata[name] = {
            "dimension": int(mean.size),
            "mean_norm": float(np.linalg.norm(mean)),
            "mean_rms": mean_rms,
            "feature_rms": feature_rms,
            "mean_to_feature_rms": mean_rms / feature_rms if feature_rms else 0.0,
            "zero_fraction": float(np.mean(mean == 0)),
            "near_zero_fraction_1e-8": float(np.mean(np.abs(mean) <= 1e-8)),
        }
    metadata = {
        "train_conversations": len(train_ids),
        "train_utterances": utterance_count,
        "dtype": "float32",
        "modalities": modality_metadata,
    }
    return means, metadata


def save_means_atomic(path: Path, means: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(temporary, **means)
    os.replace(temporary, path)


def load_means(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as archive:
        means = {name: archive[name].copy() for name in MODALITIES}
    return means
