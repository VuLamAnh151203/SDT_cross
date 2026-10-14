"""Aggregate test metrics from one SDT multi-seed experiment directory."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


PRIMARY_METRICS = (
    ("accuracy", "ACC (%)"),
    ("weighted_f1", "Weighted F1 (%)"),
    ("macro_f1", "Macro F1 (%)"),
    ("task", "Fused NLL"),
    ("total", "Total SDT loss"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_runs(run_dir):
    rows = []
    for metrics_path in sorted(run_dir.glob("*/test_metrics.json")):
        child = metrics_path.parent
        config_path = child / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError("Missing config.json in {}".format(child))
        config = _read_json(config_path)
        metrics = _read_json(metrics_path)
        args = config.get("args", {})
        row = {
            "seed": int(args["seed"]),
            "run_directory": child.name,
            "selection_protocol": str(args.get("selection_protocol", "")),
            "best_epoch": int(metrics["best_epoch"]),
        }
        for metric, _label in PRIMARY_METRICS:
            if metric not in metrics:
                raise KeyError("{} is missing metric {!r}".format(metrics_path, metric))
            row[metric] = float(metrics[metric])
        rows.append(row)
    if not rows:
        raise FileNotFoundError(
            "No immediate child directory contains test_metrics.json in {}".format(
                run_dir
            )
        )
    seeds = [row["seed"] for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Each seed must occur exactly once: {}".format(seeds))
    protocols = {row["selection_protocol"] for row in rows}
    if len(protocols) != 1:
        raise ValueError("Runs use different selection protocols: {}".format(protocols))
    return sorted(rows, key=lambda row: row["seed"])


def summarize(rows):
    summary = []
    for metric, label in PRIMARY_METRICS:
        values = [row[metric] for row in rows]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append({
            "metric": metric,
            "label": label,
            "runs": len(values),
            "mean": mean,
            "std": std,
            "mean_plus_minus_std": "{:.4f} +/- {:.4f}".format(mean, std),
            "minimum": min(values),
            "maximum": max(values),
        })
    return summary


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    rows = collect_runs(run_dir)
    summary = summarize(rows)
    per_seed_fields = [
        "seed", "run_directory", "selection_protocol", "best_epoch",
        *(metric for metric, _label in PRIMARY_METRICS),
    ]
    _write_csv(run_dir / "per_seed_metrics.csv", rows, per_seed_fields)
    _write_csv(
        run_dir / "multiseed_summary.csv",
        summary,
        ("metric", "label", "runs", "mean", "std", "mean_plus_minus_std",
         "minimum", "maximum"),
    )
    payload = {
        "run_directory": str(run_dir),
        "seeds": [row["seed"] for row in rows],
        "selection_protocol": rows[0]["selection_protocol"],
        "standard_deviation": "sample (ddof=1); zero for a single run",
        "per_seed": rows,
        "summary": summary,
    }
    (run_dir / "multiseed_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "SDT multi-seed summary",
        "Seeds: {}".format(" ".join(str(seed) for seed in payload["seeds"])),
        "Checkpoint selection: {}".format(payload["selection_protocol"]),
        "",
    ]
    lines.extend(
        "{}: {:.4f} +/- {:.4f}".format(item["label"], item["mean"], item["std"])
        for item in summary
    )
    text = "\n".join(lines) + "\n"
    (run_dir / "multiseed_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
