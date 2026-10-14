"""Train and evaluate the original SDT baseline.

Two checkpoint-selection protocols are supported:

* ``validation`` (default): hold out training conversations and select the
  epoch on validation weighted F1; evaluate test only after selection.
* ``test``: reproduce the original repository protocol, which selects the
  epoch directly on test weighted F1.

The saved checkpoint schema is shared with the cross-utterance experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from dataloader import BASE_DIR, DialogueDataset, make_loaders, split_dialogues
from losses import SDTLoss
from model import MODALITIES, Transformer_Based_Model


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--Dataset", "--dataset", dest="dataset",
        choices=("IEMOCAP", "MELD"), default="IEMOCAP",
    )
    parser.add_argument("--feature-path")
    parser.add_argument(
        "--fusion-variant", choices=("sdt",), default="sdt",
        help="Retained for command compatibility; SDT_cross contains only SDT.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--hidden-dim", "--hidden_dim", dest="hidden_dim", type=int,
        default=1024,
    )
    parser.add_argument(
        "--n-head", "--n_head", dest="n_head", type=int, default=8
    )
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--l2", "--weight-decay", dest="l2", type=float, default=1e-5
    )
    parser.add_argument(
        "--temp", "--temperature", dest="temp", type=float, default=1.0
    )
    parser.add_argument("--gamma-1", type=float, default=1.0)
    parser.add_argument("--gamma-2", type=float, default=1.0)
    parser.add_argument("--gamma-3", type=float, default=1.0)
    # Accepted for compatibility with prior baseline commands. They are not
    # part of the original SDT objective and therefore are intentionally unused.
    parser.add_argument("--lambda-co", type=float, default=0.0)
    parser.add_argument("--lambda-reg", type=float, default=0.0)
    class_weight = parser.add_mutually_exclusive_group()
    class_weight.add_argument(
        "--class-weight", dest="no_class_weight", action="store_false"
    )
    class_weight.add_argument("--no-class-weight", action="store_true")
    parser.set_defaults(no_class_weight=False)
    parser.add_argument(
        "--selection-protocol", choices=("validation", "test"),
        default="validation",
        help="validation is leakage-safe; test reproduces original SDT selection.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--max-batches", type=int, default=0,
        help="Smoke-test limit per split; zero processes the full split.",
    )
    parser.add_argument("--output-dir", default=str(BASE_DIR / "results"))
    parser.add_argument("--eval-checkpoint")
    return parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(args):
    if args.no_cuda or args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if torch.cuda.is_available():
        return torch.device("cuda", args.gpu_id)
    return torch.device("cpu")


def make_model_config(args, dataset):
    return {
        "dataset": dataset.dataset,
        "temp": args.temp,
        **dataset.feature_dims,
        "n_head": args.n_head,
        "n_classes": dataset.n_classes,
        "hidden_dim": args.hidden_dim,
        "n_speakers": dataset.n_speakers,
        "dropout": args.dropout,
        "fusion_variant": "sdt",
        "fusion_feature_source": "enhanced",
    }


def make_criterion(args, device):
    weights = None
    if args.dataset == "IEMOCAP" and not args.no_class_weight:
        frequencies = (0.086747, 0.144406, 0.227883,
                       0.160585, 0.127711, 0.252668)
        weights = torch.tensor([1.0 / value for value in frequencies], device=device)
    return SDTLoss(
        weights,
        gamma_1=args.gamma_1,
        gamma_2=args.gamma_2,
        gamma_3=args.gamma_3,
    ).to(device)


def prediction_rows(dialogue_ids, lengths, labels, prediction, output, errors):
    del errors
    labels = labels.detach().cpu()
    prediction = prediction.detach().cpu()
    probabilities = output["prob"].detach().cpu()
    rows = []
    for batch_index, (dialogue_id, length) in enumerate(zip(dialogue_ids, lengths)):
        for utterance_index in range(int(length)):
            row = {
                "dialogue_id": dialogue_id,
                "utterance_index": utterance_index,
                "label": int(labels[batch_index, utterance_index]),
                "prediction": int(prediction[batch_index, utterance_index]),
            }
            row.update({
                "prob_{}".format(class_index): float(value)
                for class_index, value in enumerate(
                    probabilities[batch_index, utterance_index]
                )
            })
            rows.append(row)
    return rows


def run_epoch(
    model,
    criterion,
    loader,
    device,
    optimizer=None,
    max_batches=0,
    grad_clip=0.0,
    collect_predictions=False,
    reliability_table=None,
    epoch=0,
):
    if loader is None:
        return {}, [], [], []
    if reliability_table is not None:
        raise ValueError("Original SDT does not use reliability supervision")
    training = optimizer is not None
    model.train(training)
    if hasattr(model, "set_tical_epoch"):
        model.set_tical_epoch(epoch)
    totals, true, predicted, rows = {}, [], [], []
    count = 0
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            text, visual, audio, speakers, mask, labels = [
                item.to(device) for item in batch[:6]
            ]
            valid = mask.bool()
            lengths = valid.sum(dim=1).tolist()
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(
                text, visual, audio, mask, speakers.transpose(0, 1), lengths
            )
            loss, parts, errors = criterion(output, labels, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss in batch {}".format(batch_index)
                )
            if training:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            n_valid = int(valid.sum().item())
            count += n_valid
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item() * n_valid
            batch_prediction = output["logits"].argmax(dim=-1)
            true.extend(labels[valid].detach().cpu().tolist())
            predicted.extend(batch_prediction[valid].detach().cpu().tolist())
            if collect_predictions:
                rows.extend(
                    prediction_rows(
                        batch[6], lengths, labels, batch_prediction, output, errors
                    )
                )
    if not count:
        raise ValueError("No valid utterances were processed")
    metrics = {name: value / count for name, value in totals.items()}
    metrics.update({
        "accuracy": 100.0 * accuracy_score(true, predicted),
        "weighted_f1": 100.0 * f1_score(
            true, predicted, average="weighted", zero_division=0
        ),
        "macro_f1": 100.0 * f1_score(
            true, predicted, average="macro", zero_division=0
        ),
        "utterances": count,
    })
    return metrics, rows, true, predicted


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_test_outputs(run_dir, metrics, rows, true, predicted, n_classes):
    _write_json(run_dir / "test_metrics.json", metrics)
    _write_csv(run_dir / "test_predictions.csv", rows)
    matrix = confusion_matrix(true, predicted, labels=list(range(n_classes)))
    np.savetxt(run_dir / "test_confusion_matrix.csv", matrix, fmt="%d", delimiter=",")


def _validate_args(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.hidden_dim < 2 or args.hidden_dim % 2:
        raise ValueError("hidden dimension must be a positive even number")
    if args.n_head < 1 or args.hidden_dim % args.n_head:
        raise ValueError("hidden dimension must be divisible by n_head")
    if args.lr <= 0 or args.l2 < 0 or not 0 <= args.dropout < 1:
        raise ValueError("invalid optimizer/dropout setting")
    if args.num_workers < 0 or args.num_threads < 0 or args.max_batches < 0:
        raise ValueError("worker/thread/batch limits must be nonnegative")


def main(argv=None):
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    seed_everything(args.seed)
    device = resolve_device(args)

    loaded = None
    if args.eval_checkpoint:
        loaded = _load_checkpoint(args.eval_checkpoint, "cpu")
        args.dataset = loaded["model_config"]["dataset"]
        saved_args = loaded.get("args", {})
        for name in (
            "selection_protocol", "valid_ratio", "seed", "batch_size",
            "gamma_1", "gamma_2", "gamma_3", "no_class_weight",
        ):
            if name in saved_args:
                setattr(args, name, saved_args[name])
        if args.feature_path is None:
            saved_feature = Path(str(loaded.get("feature_path", "")))
            if saved_feature.is_file():
                args.feature_path = str(saved_feature)

    dataset = DialogueDataset(args.dataset, args.feature_path)
    split_ids = split_dialogues(
        dataset, args.selection_protocol, args.valid_ratio
    )
    if loaded is not None:
        saved_splits = {
            name: list(values) for name, values in loaded["split_ids"].items()
        }
        if split_ids != saved_splits:
            raise ValueError("Checkpoint dialogue split differs from current data")
        model_config = dict(loaded["model_config"])
    else:
        model_config = make_model_config(args, dataset)

    model = Transformer_Based_Model(**model_config).to(device)
    criterion = make_criterion(args, device)
    loaders = make_loaders(
        dataset,
        split_ids,
        args.batch_size,
        args.seed,
        args.num_workers,
        device.type == "cuda",
    )

    run_name = "{}_sdt_seed{}_{}".format(
        args.dataset.lower(), args.seed,
        datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "config.json",
        {
            "args": vars(args),
            "model_config": model_config,
            "device": str(device),
            "feature_path": str(dataset.feature_path),
            "smoke_test": bool(args.max_batches),
        },
    )
    _write_json(run_dir / "split_ids.json", split_ids)
    print(
        "Device: {}; model: original SDT; selection: {}; output: {}".format(
            device, args.selection_protocol, run_dir
        ),
        flush=True,
    )
    print(
        "Dialogues: {}; parameters: {:,}".format(
            {name: len(ids) for name, ids in split_ids.items()},
            sum(parameter.numel() for parameter in model.parameters()),
        ),
        flush=True,
    )

    if loaded is not None:
        model.load_state_dict(loaded["model_state_dict"], strict=True)
        metrics, rows, true, predicted = run_epoch(
            model,
            criterion,
            loaders["test"],
            device,
            max_batches=args.max_batches,
            collect_predictions=True,
            epoch=int(loaded.get("epoch", 0)),
        )
        save_test_outputs(
            run_dir, metrics, rows, true, predicted, dataset.n_classes
        )
        print(json.dumps(metrics, indent=2), flush=True)
        return run_dir

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2
    )
    selection_split = (
        "valid" if args.selection_protocol == "validation" else "test"
    )
    best_score = float("-inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics, _, _, _ = run_epoch(
            model,
            criterion,
            loaders["train"],
            device,
            optimizer=optimizer,
            max_batches=args.max_batches,
            grad_clip=args.grad_clip,
            epoch=epoch,
        )
        selected_metrics, _, _, _ = run_epoch(
            model,
            criterion,
            loaders[selection_split],
            device,
            max_batches=args.max_batches,
            epoch=epoch,
        )
        row = {"epoch": epoch, "seconds": time.time() - started}
        row.update({"train_" + key: value for key, value in train_metrics.items()})
        row.update({
            selection_split + "_" + key: value
            for key, value in selected_metrics.items()
        })
        history.append(row)
        _write_csv(run_dir / "epoch_metrics.csv", history)
        score = float(selected_metrics["weighted_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "args": vars(args),
                    "epoch": epoch,
                    "selection_metrics": selected_metrics,
                    "split_ids": split_ids,
                    "feature_path": str(dataset.feature_path),
                },
                run_dir / "best_checkpoint.pt",
            )
        print(
            "Epoch {:03d}: train loss={:.4f}, ACC={:.2f}, WF1={:.2f}; "
            "{} ACC={:.2f}, WF1={:.2f} ({:.1f}s)".format(
                epoch,
                train_metrics["total"],
                train_metrics["accuracy"],
                train_metrics["weighted_f1"],
                selection_split,
                selected_metrics["accuracy"],
                selected_metrics["weighted_f1"],
                row["seconds"],
            ),
            flush=True,
        )

    best = _load_checkpoint(run_dir / "best_checkpoint.pt", device)
    model.load_state_dict(best["model_state_dict"], strict=True)
    metrics, rows, true, predicted = run_epoch(
        model,
        criterion,
        loaders["test"],
        device,
        max_batches=args.max_batches,
        collect_predictions=True,
        epoch=best_epoch,
    )
    metrics.update({
        "best_epoch": best_epoch,
        "selection_split": selection_split,
        "best_selection_weighted_f1": best_score,
    })
    save_test_outputs(run_dir, metrics, rows, true, predicted, dataset.n_classes)
    print("Best epoch: {}; test metrics:".format(best_epoch), flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return run_dir


if __name__ == "__main__":
    main()
