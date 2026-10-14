"""Recover standard utterance-level test metrics from impact-pair results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from compute_ablation_classification_metrics import (
    calculate_test_metrics,
    write_test_metrics,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more completed impact_pairs_*.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: test_classification_metrics.csv beside input)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    output = args.output or args.input[0].parent / "test_classification_metrics.csv"
    metrics = calculate_test_metrics(args.input)
    write_test_metrics(output, metrics)
    print(
        f"ACC={metrics['accuracy']:.6f}, "
        f"weighted-F1={metrics['weighted_f1']:.6f}, "
        f"macro-F1={metrics['macro_f1']:.6f}, "
        f"utterances={metrics['utterance_count']}",
        flush=True,
    )
    print(f"Wrote standard test metrics to {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
