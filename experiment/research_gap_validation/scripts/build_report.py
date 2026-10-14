"""Build the five-question research-gap report without forcing positive claims."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common import read_csv  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _mean(rows, name):
    return float(np.mean([float(row[name]) for row in rows]))


def build(args: argparse.Namespace) -> Path:
    root = args.output_dir.resolve()
    e1 = read_csv(root / "E1" / "mixed_summary.csv")
    primary = [row for row in e1 if row["modality"] == "text"
               and row["replacement_strategy"] == "train_mean"
               and row["margin_group"] == "all"
               and row["minimum_source_count"] == "2"
               and row["impact_eps"] == "0.0001"
               and row["effect_threshold"] == "0.0001"]
    e2 = read_csv(root / "E2" / "oracle_stage_comparison.csv")
    mixed_rate = float(primary[0]["mixed_rate"]) if primary else float("nan")
    early_advantage = _mean(e2, "early_advantage_nll")
    early_ci = (float(e2[0]["aggregate_ci95_low"]),
                float(e2[0]["aggregate_ci95_high"]))
    e3_path = root / "E3" / "model_comparison.csv"
    if e3_path.is_file():
        e3 = read_csv(e3_path)
        means = defaultdict_list(e3)
        def delta(first, second, metric):
            return float(np.mean(means[first][metric]) - np.mean(means[second][metric]))
        placement = delta("B2", "B1", "weighted_f1") if "B2" in means and "B1" in means else None
        granularity = delta("B3", "B2", "weighted_f1") if "B3" in means and "B2" in means else None
        supervision = delta("B4", "B3", "weighted_f1") if "B4" in means and "B3" in means else None
        relevance = delta("B4", "B6", "weighted_f1") if "B4" in means and "B6" in means else None
        e3_text = f"""3. **Placement/granularity:** mean ΔWF1 B2−B1 = `{placement}`; B3−B2 = `{granularity}`. Dùng `paired_deltas.csv` để kết luận theo CI, không chỉ theo mean.
4. **Impact supervision:** mean ΔWF1 B4−B3 = `{supervision}`; B4−B6 = `{relevance}`. Nếu CI đi qua 0 thì H3 chưa được hỗ trợ.
5. **Generalization/mechanism:** xem `transition_subset_metrics.csv`, `posttraining_influence_audit.csv` và E4. Kết luận unseen-session chỉ hợp lệ nếu checkpoint/fold manifest thực sự giữ session đó ngoài training."""
    else:
        e3_text = """3. **Placement/granularity:** E3 chưa chạy; chưa có bằng chứng B2>B1 hay B3>B2.
4. **Impact supervision:** E3 chưa chạy; chưa có bằng chứng B4>B3/B6.
5. **Generalization/mechanism:** chưa được đánh giá. Pipeline cố ý dừng trước E3 cho đến khi review E1/E2."""
    h1 = ("được hỗ trợ mô tả" if mixed_rate > 0 else "không được hỗ trợ")
    h2 = ("early oracle có lợi thế" if early_ci[0] > 0 else
          "không chắc chắn; CI không chứng minh early tốt hơn late")
    report = f"""# Research-gap validation report

## Trả lời năm câu hỏi

1. **Mixed source contributions:** text/train-mean mixed rate = `{mixed_rate:.6f}` ở `eps=1e-4`, minimum 2 sources. H1 {h1}; cần xem thêm effect-size thresholds, margin strata, donor và edge controls.
2. **Frozen stage localization:** mean `(late oracle NLL − early oracle NLL)` = `{early_advantage:.6f}`, 95% conversation-bootstrap CI `[{early_ci[0]:.6f}, {early_ci[1]:.6f}]`. Diễn giải: {h2}. Hai action spaces khác granularity nên đây không phải hiệu ứng vị trí thuần túy.
{e3_text}

## Giới hạn bắt buộc

- Signed impact là predictive, intervention-dependent; không phải causal ground truth.
- Attention edge suppression có thể chạy post-softmax có/không renormalization; phải báo cáo cấu hình.
- Mỗi SDT branch hiện chỉ có một layer; path test là one-branch vs all source-modality branches.
- Oracle dùng true target label và không deploy được.
- Test IEMOCAP đã được khám phá ở thí nghiệm trước; strong claim cần fold/session hoặc dataset độc lập với checkpoint phù hợp.
"""
    path = root / "report.md"
    path.write_text(report, encoding="utf-8")
    print(path)
    return path


def defaultdict_list(rows):
    output = {}
    for row in rows:
        model = row["model"]
        output.setdefault(model, {})
        for metric in ("accuracy", "weighted_f1", "macro_f1", "nll"):
            output[model].setdefault(metric, []).append(float(row[metric]))
    return output


def main(argv: Sequence[str] | None = None) -> None:
    build(parse_args(argv))


if __name__ == "__main__":
    main()

