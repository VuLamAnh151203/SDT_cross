"""Torch-only variational mixed-effects logistic analysis of contextual harm.

This is an offline explanatory model.  It uses mean-field variational Bayes
with Gaussian fixed effects and Gaussian random intercepts for conversation
and target.  No statsmodels, pandas, or ground-truth-free inference claim is
required here; emotion labels are legitimate offline control variables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from experiment_analysis_common import STRATEGIES, resolve_impact_paths, write_rows


BASE_FEATURE_NAMES = (
    "intercept",
    "same_speaker",
    "log_distance_z",
    "emotion_difference",
    "baseline_margin_z",
    "modality_audio",
    "modality_visual",
    "source_before_target",
    "correct_full",
)


@dataclass
class ModelData:
    features: np.ndarray
    signed_impacts: np.ndarray
    conversation_index: np.ndarray
    target_index: np.ndarray
    feature_names: Tuple[str, ...]
    conversation_count: int
    target_count: int
    row_count_before_subsampling: int


class VariationalMixedLogistic(torch.nn.Module):
    def __init__(
        self,
        feature_count: int,
        conversation_count: int,
        target_count: int,
        fixed_prior_sd: float,
        random_scale_prior_sd: float,
    ) -> None:
        super().__init__()
        self.fixed_mean = torch.nn.Parameter(torch.zeros(feature_count))
        self.fixed_log_sd = torch.nn.Parameter(torch.full((feature_count,), -2.0))
        self.conversation_mean = torch.nn.Parameter(torch.zeros(conversation_count))
        self.conversation_log_sd = torch.nn.Parameter(
            torch.full((conversation_count,), -2.0)
        )
        self.target_mean = torch.nn.Parameter(torch.zeros(target_count))
        self.target_log_sd = torch.nn.Parameter(torch.full((target_count,), -2.0))
        self.conversation_raw_scale = torch.nn.Parameter(torch.tensor(-0.5))
        self.target_raw_scale = torch.nn.Parameter(torch.tensor(-0.5))
        self.fixed_prior_sd = fixed_prior_sd
        self.random_scale_prior_sd = random_scale_prior_sd

    @staticmethod
    def _sd(log_sd: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.clamp(log_sd, min=-8.0, max=3.0))

    @staticmethod
    def _scale(raw_scale: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw_scale) + 1e-3

    def random_scales(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._scale(self.conversation_raw_scale),
            self._scale(self.target_raw_scale),
        )

    def sampled_logits(
        self,
        features: torch.Tensor,
        conversation_index: torch.Tensor,
        target_index: torch.Tensor,
    ) -> torch.Tensor:
        fixed = self.fixed_mean + self._sd(self.fixed_log_sd) * torch.randn_like(
            self.fixed_mean
        )
        conversation = self.conversation_mean + self._sd(
            self.conversation_log_sd
        ) * torch.randn_like(self.conversation_mean)
        target = self.target_mean + self._sd(
            self.target_log_sd
        ) * torch.randn_like(self.target_mean)
        return (
            features @ fixed
            + conversation[conversation_index]
            + target[target_index]
        )

    def mean_logits(
        self,
        features: torch.Tensor,
        conversation_index: torch.Tensor,
        target_index: torch.Tensor,
    ) -> torch.Tensor:
        return (
            features @ self.fixed_mean
            + self.conversation_mean[conversation_index]
            + self.target_mean[target_index]
        )

    @staticmethod
    def _normal_kl(
        mean: torch.Tensor, posterior_sd: torch.Tensor, prior_sd: torch.Tensor
    ) -> torch.Tensor:
        return torch.sum(
            torch.log(prior_sd / posterior_sd)
            + (posterior_sd.square() + mean.square()) / (2.0 * prior_sd.square())
            - 0.5
        )

    def kl_divergence(self) -> torch.Tensor:
        device = self.fixed_mean.device
        fixed_prior = torch.tensor(self.fixed_prior_sd, device=device)
        conversation_scale, target_scale = self.random_scales()
        kl = self._normal_kl(
            self.fixed_mean, self._sd(self.fixed_log_sd), fixed_prior
        )
        kl = kl + self._normal_kl(
            self.conversation_mean,
            self._sd(self.conversation_log_sd),
            conversation_scale,
        )
        kl = kl + self._normal_kl(
            self.target_mean,
            self._sd(self.target_log_sd),
            target_scale,
        )
        # Half-normal priors on the learned random-effect standard deviations.
        prior = torch.tensor(self.random_scale_prior_sd, device=device)
        kl = kl + 0.5 * (
            conversation_scale.square() + target_scale.square()
        ) / prior.square()
        return kl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--impact-eps-values", type=float, nargs="+",
        default=(1e-6, 1e-4, 1e-3),
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--fixed-prior-sd", type=float, default=2.0)
    parser.add_argument("--random-scale-prior-sd", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-rows", type=int, default=0,
        help="Deterministic diagnostic subsample; 0 uses every row",
    )
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args(argv)


def _baseline_margins(run_dir: Path) -> Tuple[Dict[tuple[str, int], float], int]:
    path = Path(run_dir) / "baseline" / "baseline_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    margins = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        probability_columns = sorted(
            (name for name in fieldnames if name.startswith("prob_") and name != "prob_true"),
            key=lambda name: int(name.split("_")[1]),
        )
        required = {"conversation_id", "utterance_index"}
        missing = sorted(required - set(fieldnames))
        if missing or len(probability_columns) < 2:
            raise ValueError(
                f"Invalid baseline file: missing={missing}, probabilities={probability_columns}"
            )
        for line_number, row in enumerate(reader, start=2):
            probabilities = sorted(float(row[name]) for name in probability_columns)
            key = (row["conversation_id"], int(row["utterance_index"]))
            if key in margins:
                raise ValueError(f"{path}:{line_number}: duplicate baseline target {key}")
            margins[key] = probabilities[-1] - probabilities[-2]
    if not margins:
        raise ValueError(f"No baseline rows in {path}")
    return margins, len(probability_columns)


def _count_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "conversation_id", "target_idx", "same_speaker", "absolute_distance",
            "source_before_target", "modality", "source_emotion", "target_emotion",
            "signed_impact", "correct_full",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        return sum(1 for _ in reader)


def _load_strategy_data(
    path: Path,
    baseline_margins: Dict[tuple[str, int], float],
    class_count: int,
    max_rows: int,
    seed: int,
) -> ModelData:
    row_count = _count_rows(path)
    target_dummy_names = tuple(
        f"target_emotion_{label}" for label in range(1, class_count)
    )
    feature_names = BASE_FEATURE_NAMES + target_dummy_names + (
        "same_speaker_x_emotion_difference",
        "modality_audio_x_log_distance_z",
        "modality_visual_x_log_distance_z",
    )
    features = np.zeros((row_count, len(feature_names)), dtype=np.float32)
    impacts = np.empty(row_count, dtype=np.float32)
    conversation_indices = np.empty(row_count, dtype=np.int32)
    target_indices = np.empty(row_count, dtype=np.int32)
    conversation_lookup: Dict[str, int] = {}
    target_lookup: Dict[tuple[str, int], int] = {}

    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row_index, row in enumerate(reader):
            conversation_id = row["conversation_id"]
            target_idx = int(row["target_idx"])
            target_key = (conversation_id, target_idx)
            if target_key not in baseline_margins:
                raise ValueError(f"Missing baseline margin for {target_key}")
            conversation_index = conversation_lookup.setdefault(
                conversation_id, len(conversation_lookup)
            )
            target_index = target_lookup.setdefault(target_key, len(target_lookup))
            same_speaker = int(row["same_speaker"])
            source_emotion = int(row["source_emotion"])
            target_emotion = int(row["target_emotion"])
            if target_emotion < 0 or target_emotion >= class_count:
                raise ValueError(f"Invalid target emotion {target_emotion}")
            emotion_difference = int(source_emotion != target_emotion)
            modality = row["modality"]
            if modality not in ("text", "audio", "visual"):
                raise ValueError(f"Invalid modality {modality}")
            features[row_index, 0] = 1.0
            features[row_index, 1] = same_speaker
            features[row_index, 2] = math.log1p(int(row["absolute_distance"]))
            features[row_index, 3] = emotion_difference
            features[row_index, 4] = baseline_margins[target_key]
            features[row_index, 5] = int(modality == "audio")
            features[row_index, 6] = int(modality == "visual")
            features[row_index, 7] = int(row["source_before_target"])
            features[row_index, 8] = int(row["correct_full"])
            if target_emotion > 0:
                features[row_index, 8 + target_emotion] = 1.0
            impacts[row_index] = float(row["signed_impact"])
            conversation_indices[row_index] = conversation_index
            target_indices[row_index] = target_index

    for column in (2, 4):
        standard_deviation = float(features[:, column].std())
        if standard_deviation == 0:
            raise ValueError(f"Cannot standardize constant feature {feature_names[column]}")
        features[:, column] = (
            features[:, column] - float(features[:, column].mean())
        ) / standard_deviation
    features[:, -3] = features[:, 1] * features[:, 3]
    features[:, -2] = features[:, 5] * features[:, 2]
    features[:, -1] = features[:, 6] * features[:, 2]

    if max_rows < 0:
        raise ValueError("max_rows must be nonnegative")
    if max_rows and row_count > max_rows:
        indices = np.random.default_rng(seed).choice(
            row_count, size=max_rows, replace=False
        )
        indices.sort()
        features = features[indices]
        impacts = impacts[indices]
        conversation_indices = conversation_indices[indices]
        target_indices = target_indices[indices]

    return ModelData(
        features=features,
        signed_impacts=impacts,
        conversation_index=conversation_indices,
        target_index=target_indices,
        feature_names=feature_names,
        conversation_count=len(conversation_lookup),
        target_count=len(target_lookup),
        row_count_before_subsampling=row_count,
    )


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def _fit_one(
    data: ModelData,
    epsilon: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    fixed_prior_sd: float,
    random_scale_prior_sd: float,
    patience: int,
    relative_tolerance: float,
    device: torch.device,
    seed: int,
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    x = torch.from_numpy(data.features).to(device)
    impacts = torch.from_numpy(data.signed_impacts).to(device)
    labels = (impacts < -epsilon).float()
    conversation_index = torch.from_numpy(data.conversation_index).long().to(device)
    target_index = torch.from_numpy(data.target_index).long().to(device)
    if labels.min() == labels.max():
        raise ValueError(f"Harm labels contain one class at epsilon={epsilon}")
    model = VariationalMixedLogistic(
        x.shape[1], data.conversation_count, data.target_count,
        fixed_prior_sd, random_scale_prior_sd,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    row_count = len(x)
    effective_batch_size = min(batch_size, row_count)
    history = []
    best_loss = math.inf
    stale_epochs = 0

    for epoch in range(epochs):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + epoch)
        order = torch.randperm(row_count, generator=generator, device=device)
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, row_count, effective_batch_size):
            index = order[start:start + effective_batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model.sampled_logits(
                x[index], conversation_index[index], target_index[index]
            )
            negative_log_likelihood = F.binary_cross_entropy_with_logits(
                logits, labels[index], reduction="sum"
            ) * (row_count / len(index))
            loss = negative_log_likelihood + model.kl_divergence()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batch_count += 1
        average_loss = epoch_loss / batch_count
        history.append(average_loss)
        relative_improvement = (
            (best_loss - average_loss) / max(abs(best_loss), 1.0)
            if math.isfinite(best_loss) else math.inf
        )
        if average_loss < best_loss:
            best_loss = average_loss
        if relative_improvement > relative_tolerance:
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"  epoch={epoch + 1}/{epochs}, negative_elbo={average_loss:.4f}",
                flush=True,
            )
        if stale_epochs >= patience:
            break

    with torch.no_grad():
        fixed_mean = model.fixed_mean.detach().cpu().numpy().astype(np.float64)
        fixed_sd = model._sd(model.fixed_log_sd).detach().cpu().numpy().astype(np.float64)
        conversation_scale, target_scale = model.random_scales()
        random_summary = {
            "conversation": {
                "level_count": data.conversation_count,
                "learned_prior_standard_deviation": float(conversation_scale.cpu()),
                "mean_posterior_standard_deviation": float(
                    model._sd(model.conversation_log_sd).mean().cpu()
                ),
                "posterior_mean_empirical_standard_deviation": float(
                    model.conversation_mean.std(unbiased=False).cpu()
                ),
            },
            "target": {
                "level_count": data.target_count,
                "learned_prior_standard_deviation": float(target_scale.cpu()),
                "mean_posterior_standard_deviation": float(
                    model._sd(model.target_log_sd).mean().cpu()
                ),
                "posterior_mean_empirical_standard_deviation": float(
                    model.target_mean.std(unbiased=False).cpu()
                ),
            },
        }
        prevalence = float(labels.mean().cpu())
    return fixed_mean, fixed_sd, random_summary, history, prevalence


def _probability_positive(mean: float, standard_deviation: float) -> float:
    if standard_deviation <= 0:
        return float(mean > 0)
    return 0.5 * (1.0 + math.erf(mean / standard_deviation / math.sqrt(2.0)))


def fit_models(
    run_dir: Path,
    impact_eps_values: Sequence[float] = (1e-6, 1e-4, 1e-3),
    strategies: Sequence[str] = STRATEGIES,
    epochs: int = 60,
    batch_size: int = 8192,
    learning_rate: float = 0.02,
    fixed_prior_sd: float = 2.0,
    random_scale_prior_sd: float = 1.0,
    patience: int = 8,
    relative_tolerance: float = 1e-4,
    device_name: str = "auto",
    max_rows: int = 0,
    seed: int = 2024,
) -> Path:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if fixed_prior_sd <= 0 or random_scale_prior_sd <= 0:
        raise ValueError("Prior standard deviations must be positive")
    if patience < 1 or relative_tolerance < 0:
        raise ValueError("Invalid early-stopping configuration")
    epsilons = tuple(sorted(set(float(value) for value in impact_eps_values)))
    if any(value < 0 for value in epsilons):
        raise ValueError("Impact thresholds must be nonnegative")

    run_dir = Path(run_dir).expanduser().resolve()
    result_dir = run_dir / "results"
    paths = {
        path.stem.removeprefix("impact_pairs_"): path
        for path in resolve_impact_paths(run_dir)
    }
    baseline_margins, class_count = _baseline_margins(run_dir)
    device = _resolve_device(device_name)
    fixed_rows = []
    random_rows = []
    metadata: Dict[str, object] = {
        "backend": "torch_mean_field_variational_bayes",
        "device": str(device),
        "impact_eps_values": list(epsilons),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "fixed_prior_sd": fixed_prior_sd,
        "random_scale_prior_sd": random_scale_prior_sd,
        "max_rows": max_rows,
        "seed": seed,
        "random_effects": ["conversation", "target_nested_in_conversation"],
        "fits": [],
        "limitations": (
            "Mean-field variational posterior intervals are approximate. "
            "Conversation-bootstrap intervals from the descriptive analysis should be "
            "reported alongside this adjusted model."
        ),
    }

    for strategy in strategies:
        if strategy not in paths:
            raise FileNotFoundError(f"Missing impact file for strategy={strategy}")
        data = _load_strategy_data(
            paths[strategy], baseline_margins, class_count, max_rows, seed
        )
        for epsilon_index, epsilon in enumerate(epsilons):
            print(
                f"Fitting Torch variational mixed logistic: strategy={strategy}, "
                f"epsilon={epsilon:g}, rows={len(data.features)}, device={device}",
                flush=True,
            )
            fixed_mean, fixed_sd, random_summary, history, prevalence = _fit_one(
                data, epsilon, epochs, batch_size, learning_rate,
                fixed_prior_sd, random_scale_prior_sd, patience,
                relative_tolerance, device, seed + epsilon_index,
            )
            for name, mean, standard_deviation in zip(
                data.feature_names, fixed_mean, fixed_sd
            ):
                low = float(mean - 1.96 * standard_deviation)
                high = float(mean + 1.96 * standard_deviation)
                fixed_rows.append({
                    "replacement_strategy": strategy,
                    "impact_eps": epsilon,
                    "row_count": len(data.features),
                    "harmful_prevalence": prevalence,
                    "term": name,
                    "log_odds_posterior_mean": float(mean),
                    "log_odds_posterior_sd": float(standard_deviation),
                    "log_odds_interval95_low": low,
                    "log_odds_interval95_high": high,
                    "odds_ratio": math.exp(float(np.clip(mean, -700, 700))),
                    "odds_ratio_interval95_low": math.exp(float(np.clip(low, -700, 700))),
                    "odds_ratio_interval95_high": math.exp(float(np.clip(high, -700, 700))),
                    "posterior_probability_positive": _probability_positive(
                        float(mean), float(standard_deviation)
                    ),
                })
            for effect_name, values in random_summary.items():
                random_rows.append({
                    "replacement_strategy": strategy,
                    "impact_eps": epsilon,
                    "random_effect": effect_name,
                    **values,
                })
            metadata["fits"].append({
                "replacement_strategy": strategy,
                "impact_eps": epsilon,
                "row_count": len(data.features),
                "row_count_before_subsampling": data.row_count_before_subsampling,
                "harmful_prevalence": prevalence,
                "epochs_completed": len(history),
                "negative_elbo_history": history,
            })

    robustness_rows = []
    for term in sorted({row["term"] for row in fixed_rows}):
        items = [row for row in fixed_rows if row["term"] == term]
        means = [float(row["log_odds_posterior_mean"]) for row in items]
        robust_positive = all(
            float(row["log_odds_interval95_low"]) > 0 for row in items
        )
        robust_negative = all(
            float(row["log_odds_interval95_high"]) < 0 for row in items
        )
        robustness_rows.append({
            "term": term,
            "fit_count": len(items),
            "positive_coefficient_count": sum(value > 0 for value in means),
            "negative_coefficient_count": sum(value < 0 for value in means),
            "consistent_sign_across_strategy_and_epsilon": int(
                all(np.sign(value) == np.sign(means[0]) for value in means)
            ),
            "interval_excludes_zero_same_direction_in_every_fit": int(
                robust_positive or robust_negative
            ),
            "robust_direction": (
                "positive" if robust_positive else "negative" if robust_negative
                else "not_robust"
            ),
            "minimum_odds_ratio": min(float(row["odds_ratio"]) for row in items),
            "maximum_odds_ratio": max(float(row["odds_ratio"]) for row in items),
        })

    write_rows(result_dir / "structural_harm_mixed_effects_fixed.csv", fixed_rows)
    write_rows(result_dir / "structural_harm_mixed_effects_random.csv", random_rows)
    write_rows(
        result_dir / "structural_harm_mixed_effects_robustness.csv",
        robustness_rows,
    )
    metadata_path = result_dir / "structural_harm_mixed_effects_metadata.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, metadata_path)
    print(f"Torch mixed-effects analysis complete: {result_dir}", flush=True)
    return result_dir


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    return fit_models(
        args.run_dir,
        args.impact_eps_values,
        args.strategies,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.fixed_prior_sd,
        args.random_scale_prior_sd,
        args.early_stop_patience,
        args.early_stop_relative_tolerance,
        args.device,
        args.max_rows,
        args.seed,
    )


if __name__ == "__main__":
    main()
