"""Build a concise Vietnamese ETCR report from completed A/B/C outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from etcr_common import read_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args(argv)


def _find(rows, **conditions):
    for row in rows:
        if all(row.get(name) == str(value) for name, value in conditions.items()):
            return row
    return None


def _pct(value) -> str:
    return f"{100 * float(value):.2f}%"


def _number(value, digits=4) -> str:
    return f"{float(value):.{digits}f}"


def build(args: argparse.Namespace) -> Path:
    root = args.run_dir.expanduser().resolve() / "etcr"
    a = read_rows(root / "phase_a" / "emotion_persistence_summary.csv")
    errors = read_rows(root / "phase_b" / "persistence_error_transitions.csv")
    adjusted = read_rows(root / "phase_b" / "adjusted_transition_effects.csv")
    predictor = read_rows(root / "phase_c" / "predictor_metrics.csv")
    mitigation = read_rows(root / "phase_c" / "mitigation_metrics.csv")
    relation_r = _find(a, group_type="relation", group_value="transition", metric="R")
    relation_b = _find(a, group_type="relation", group_value="transition",
                       metric="fraction_R_positive")
    primary_error = _find(errors, replacement_strategy="train_mean")
    risk = _find(adjusted, effect="transition_average_marginal_risk_difference")
    learned = _find(predictor, model="learned_full", split="test")
    original = _find(mitigation, policy="original_sdt", subset="all_test")
    learned_policy = _find(mitigation, policy="learned_policy", subset="all_test")
    oracle = _find(mitigation, policy="oracle_ground_truth", subset="all_test")
    required = [relation_r, relation_b, primary_error, risk, learned, original,
                learned_policy, oracle]
    if any(item is None for item in required):
        raise ValueError("One or more required report rows are missing")
    if float(relation_r["ci95_low"]) > 0:
        phase_a_verdict = "CI của R nằm hoàn toàn trên 0: dữ liệu hỗ trợ old-emotion bias."
    elif float(relation_r["ci95_high"]) < 0:
        phase_a_verdict = (
            "R nghiêng âm: dữ liệu không hỗ trợ giả thuyết old-emotion bias theo hướng dự kiến."
        )
    else:
        phase_a_verdict = (
            "CI của R đi qua 0: chưa có bằng chứng chắc chắn cho old-emotion bias trung bình."
        )
    if float(risk["ci95_low"]) > 0 or float(risk["ci95_high"]) < 0:
        phase_b_verdict = "Adjusted transition effect không đi qua 0."
    else:
        phase_b_verdict = (
            "Adjusted transition effect đi qua 0; association sau kiểm soát chưa chắc chắn."
        )
    if float(learned_policy["delta_nll_ci95_high"]) < 0:
        phase_c_verdict = "Learned policy giảm NLL với CI hoàn toàn dưới 0."
    elif float(learned_policy["delta_nll"]) < 0:
        phase_c_verdict = (
            "Learned policy giảm NLL điểm ước lượng, nhưng CI chưa loại trừ không cải thiện."
        )
    else:
        phase_c_verdict = (
            "Learned policy không giảm NLL; đây là negative finding về khả năng khai thác hiện tại."
        )
    report = f"""# Experiment 3 — Emotion-Transition-Aware Contextual Reliability (ETCR)

## Thiết lập

- Intervention chính: thay text của utterance gần nhất đứng trước target và cùng speaker bằng training mean.
- `signed_impact = NLL_mask - NLL_full`; giá trị âm nghĩa là masking có lợi.
- Predictor Phase C chỉ dùng feature quan sát được lúc inference. Nhãn/impact test chỉ được dùng để chấm điểm.
- Confidence interval được bootstrap theo conversation; mỗi target chỉ được tính một lần.

## A — Previous-emotion bias

Trên transition targets, mean `R` = **{_number(relation_r['mean'], 6)}** (95% CI [{_number(relation_r['ci95_low'], 6)}, {_number(relation_r['ci95_high'], 6)}]); tỷ lệ `R > 0` = **{_pct(relation_b['mean'])}**. `R > 0` nghĩa là khi giữ previous-same-speaker context, mô hình nghiêng về emotion cũ hơn so với khi masking.

Kết quả này chỉ hỗ trợ old-emotion bias nếu CI và các phân tích theo nhóm nhất quán; association không chứng minh cơ chế nội tại của Transformer.

**Kết luận Phase A:** {phase_a_verdict}

## B — Error severity và upper bound

- Transition targets: **{primary_error['transition_targets']}**.
- Persistence errors: **{primary_error['persistence_error_count']}** ({_pct(primary_error['persistence_error_rate'])}).
- Được sửa bởi masking: **{primary_error['corrected_count']}** ({_pct(primary_error['correction_rate_among_persistence_errors'])} trong persistence errors).
- Bị làm hỏng từ đúng thành sai: **{primary_error['damaged_count']}** ({_pct(primary_error['damage_rate_among_baseline_correct'])} trong baseline-correct transition targets).
- Adjusted transition risk difference: **{_number(risk['estimate'], 6)}** (95% CI [{_number(risk['ci95_low'], 6)}, {_number(risk['ci95_high'], 6)}]).

Oracle dùng nhãn thật đạt ΔACC = **{_number(oracle['delta_accuracy'])}** điểm, ΔWF1 = **{_number(oracle['delta_weighted_f1'])}** điểm và ΔNLL = **{_number(oracle['delta_nll'], 6)}** trên toàn test. Đây chỉ là diagnostic upper bound, không phải policy hợp lệ khi inference.

**Kết luận Phase B:** {phase_b_verdict}

## C — Predictability và selective mitigation

Learned predictor trên test: AUROC = **{_number(learned['auroc'])}**, AUPRC = **{_number(learned['auprc'])}**, action precision = **{learned['action_precision']}**.

Policy learned mask **{learned_policy['mask_count']}** targets; so với SDT gốc:

- ΔACC = **{_number(learned_policy['delta_accuracy'])}** điểm.
- ΔWF1 = **{_number(learned_policy['delta_weighted_f1'])}** điểm.
- ΔNLL = **{_number(learned_policy['delta_nll'], 6)}**.
- Corrected / damaged = **{learned_policy['corrected_count']} / {learned_policy['damaged_count']}**.

Baseline SDT: ACC **{_number(original['policy_accuracy'])}**, WF1 **{_number(original['policy_weighted_f1'])}**, NLL **{_number(original['policy_nll'], 6)}**.

**Kết luận Phase C:** {phase_c_verdict}

## Cách diễn giải

1. Phase A trả lời context có kéo mô hình về emotion trước hay không.
2. Phase B kiểm tra bias đó có tạo lỗi quyết định thực sự và còn tồn tại sau adjustment hay không.
3. Phase C là bằng chứng quyết định: chỉ khi learned policy vượt các baseline cùng budget trên held-out test, với CI hợp lý, mới có cơ sở cho reliability gating.
4. Nếu oracle tốt nhưng learned policy không tốt, tín hiệu harm tồn tại nhưng feature hiện tại chưa dự đoán được. Nếu oracle cũng gần như không cải thiện, singleton removal có ít tiềm năng và nên chuyển sang groupwise interventions.

Các bảng đầy đủ nằm trong `phase_a/`, `phase_b/`, `phase_c/`; figure nằm trong `figures/`.
"""
    path = root / "report.md"
    path.write_text(report, encoding="utf-8")
    print(path)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    build(parse_args(argv))


if __name__ == "__main__":
    main()

