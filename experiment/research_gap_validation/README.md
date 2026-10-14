# Conversation-Aware Reliability at Propagation Time

Thư mục này triển khai chuỗi kiểm chứng E1 → E4 cho research gap về reliability
ở mức `source → target → modality`. Đây là pipeline mới, tách biệt với
`cross_utterance` và ETCR.

## Kết luận architecture audit trước khi chạy

SDT trong repository là Transformer, không phải GNN. Backbone có chín branch
attention: `t_t, a_t, v_t, a_a, t_a, v_a, v_v, t_v, a_v`. Tên branch có dạng
`<source modality>_<target representation>`. Attention sau softmax có shape
`[batch, head, target, source]`.

Do đó:

- Early edge gate `g[j→i,m]` được nhân vào attention weight tại
  `[:, :, target=i, source=j]` trong toàn bộ ba branch có source modality `m`.
- Early modality gate `g[i,m]` được broadcast trên mọi source `j != i` của ba
  branch tương ứng.
- Late modality gate được áp dụng lên `H'_T/H'_A/H'_V` sau contextual encoders
  và feature reduction, ngay trước final fusion.
- Self edges luôn được ép bằng 1 để bảo toàn self-information.
- Mặc định gate đặt sau softmax và không renormalize, nên nó thực sự giảm
  message magnitude. `--renormalize-attention` là sensitivity control.

Mỗi Transformer branch hiện chỉ có một layer. Vì vậy “one layer vs all layers”
không có ý nghĩa trong checkpoint này; E2 thay bằng `one branch vs all three
source-modality branches` và ghi limitation rõ trong manifest.

## Cấu trúc

```text
research_gap_validation/
  configs/
  scripts/
  outputs/                 # được tạo khi chạy
  reliability_model.py
  common.py
  report.md                # được tạo sau khi có kết quả
  README.md
```

Các entry point nằm trong `scripts/` đúng theo guideline:

- `audit_sdt_architecture.py`
- `e1_mixed_contribution.py`
- `generate_donor_impacts.py`
- `e2_stage_interventions.py`
- `e3_train_compare.py`
- `e3_posttraining_audit.py`
- `e4_robustness.py`
- `build_report.py`
- `run_all.sh`
- `d1_prepare_edge_dataset.py`
- `d1_harm_learnability.py`
- `run_d1.sh`

## Protocol và điều kiện dừng

`run_all.sh` chạy theo thứ tự E1 → E2 → E3 → E4. Sau E1 và E2, script tạo
decision files. Full E3 chỉ chạy khi đặt `RUN_FULL_E3=1`; điều này ngăn việc tự
động tiêu tốn nhiều GPU khi H1 yếu hoặc late oracle ngang early oracle.

E3 không giả vờ tạo LOSO từ một checkpoint đã nhìn thấy session test. Muốn
đánh giá LOSO đúng, chạy `e3_train_compare.py` cho từng checkpoint fold-specific
với `--fold-name` và `--require-session-disjoint`. Script kiểm tra
train/validation/test conversation disjoint, có thể bắt buộc session-disjoint,
và luôn ghi membership vào manifest. File
`configs/independent_folds.example.json` mô tả contract; một checkpoint đã train
trên session test không được tái sử dụng cho fold đó.

## Chạy nhanh

Sửa `configs/experiment_config.sh`, sau đó:

```bash
cd /path/to/SDT_cross/experiment/research_gap_validation
bash scripts/run_all.sh
```

Chỉ audit/E1/E2:

```bash
export RUN_FULL_E3=0
bash scripts/run_all.sh
```

Full learned comparison:

```bash
export RUN_FULL_E3=1
bash scripts/run_all.sh
```

### D1 — Harmful influence có học được không?

D1 tách bài toán harmful-edge prediction khỏi reliability gate. Frozen SDT tạo
direct propagation-edge impact bằng đúng action `gate=0`; sau đó ba predictor có
cùng capacity so sánh binary, three-class và magnitude-aware supervision.
Threshold `eps_strong` chỉ được chọn trên validation conversations. Nếu checkpoint
không có validation split, script giữ riêng một session từ checkpoint-train làm
predictor-validation. Test impact không được dùng để train, early stopping hoặc
chọn threshold.

```bash
cd /path/to/SDT_cross/experiment/research_gap_validation
bash scripts/run_d1.sh
```

Chạy nhanh một seed, text-only:

```bash
D1_SEEDS="2024" \
D1_MODALITIES="text" \
D1_STRONG_EPS="0.0001 0.001 0.01" \
bash scripts/run_d1.sh
```

GPU phase sinh impact được cache theo conversation. Các lần đổi supervision,
threshold hoặc predictor sẽ reuse `D1/edge_impact_pairs.csv`. Đặt
`D1_REGENERATE_DATA=1` khi đổi checkpoint, modality, intervention hoặc split.

Outputs:

```text
outputs/full_run/D1/
  edge_impact_pairs.csv
  dataset_manifest.json
  threshold_selection.csv
  model_metrics.csv
  precision_at_k.csv
  summary.md
  test_predictions/
  checkpoints/
  experiment_manifest.json
```

`model_metrics.csv` báo cáo AUROC, AUPRC và Spearman cho tất cả pairs
(`scope=all`) và strong-impact pairs sau khi loại neutral (`scope=strong`).
`precision_at_k.csv` đánh giá khả năng xếp harmful source trong từng
conversation–target–modality, kèm prevalence/random baseline và precision lift.
AUPRC phải được so với `harmful_prevalence`, không so mặc định với 0.5. True
emotion và signed impact chỉ dùng để tạo supervision/scoring, không nằm trong
predictor features.

Kết quả chính thức nên dùng `SEEDS="2024 2025 2026 2027 2028"` và
`BOOTSTRAP_REPS=2000`. Smoke test có thể giảm seeds, epochs, conversations và
bootstrap repetitions; manifest luôn ghi rõ các giới hạn này.

## Input bắt buộc

E1 tái sử dụng hai raw files của Experiment 1:

```text
SOURCE_RUN_DIR/baseline/baseline_predictions.csv
SOURCE_RUN_DIR/results/impact_pairs_train_mean.csv
SOURCE_RUN_DIR/results/impact_pairs_zero.csv
```

Training-distribution donor được sinh bởi `generate_donor_impacts.py`. Teacher
chính cho B4 được sinh bằng `generate_train_edge_teacher.py`: script quét gate
trên đúng propagation edge và đúng attention hooks mà B4 sử dụng. Teacher chỉ
dùng train conversations, lưu continuous `teacher_gate` cùng confidence
`teacher_weight`, và từ chối cross-split leakage. File input-replacement cũ từ
`generate_train_impact_teacher.py` vẫn được hỗ trợ theo chế độ legacy, nhưng
được đổi thành soft magnitude-aware targets thay vì hard BCE 0/1.

Chạy lại B3/B4 với teacher đã sửa:

```bash
cd /path/to/SDT_cross/experiment/research_gap_validation
bash scripts/run_e3_b4_fixed.sh
```

Pipeline chẩn đoán B3/B4 gồm 5 bước (multi-seed, thống kê từng modality,
teacher-fit trên validation, sweep hyperparameter và text-only gate):

```bash
bash scripts/run_e3_five_diagnostics.sh
```

Mặc định pipeline dùng 5 seed `2024..2028`, chọn checkpoint bằng validation
weighted-F1 và tái sử dụng các teacher trong `outputs/full_run/manifests/`.
Nếu bị ngắt giữa chừng, chạy lại cùng lệnh sẽ bỏ qua các run đã có manifest hoàn
chỉnh (`FIVE_REUSE_COMPLETED_RUNS=1`, là mặc định). Đặt biến này bằng `0` để ép
chạy lại toàn bộ.
Smoke test có thể chạy bằng:

```bash
OUTPUT_DIR="$PWD/outputs/smoke_five" FIVE_SEEDS="2024" FIVE_EPOCHS=2 \
  MAX_CONVERSATIONS=2 MAX_BATCHES=1 \
  bash scripts/run_e3_five_diagnostics.sh
```

Teacher edge khởi tạo có thể mất nhiều thời gian như E2 vì phải quét từng
source-target edge. Các lần sau script tự reuse
`outputs/full_run/manifests/train_edge_impact_teacher.csv`. Đặt
`B4_REGENERATE_TEACHER=1` nếu cần sinh lại.

## Outputs

```text
outputs/full_run/
  E1/
    target_modality_mixed.csv
    mixed_summary.csv
    source_impact_pairs.csv
    intervention_robustness.csv
    heterogeneity_by_structure.csv
    negative_controls.csv
  E2/
    stage_intervention_pairs.csv
    oracle_stage_comparison.csv
    oracle_stage_summary.csv
    layer_path_sensitivity.csv
  E3/
    model_comparison.csv
    epoch_history.csv
    paired_deltas.csv
    transition_subset_metrics.csv
    classwise_f1.csv
    posttraining_influence_audit.csv
    ablation_results.csv
    checkpoints/
    audits/
  E4/
    robustness_matrix.csv
    ranking_metrics.csv
    fixed_control_metrics.csv
    negative_controls.csv
  figures/
  manifests/
  report.md
```

E2 direct edge scan lớn hơn input-source scan vì mỗi action chỉ thay đúng một
`j→i` edge. `INTERVENTION_BATCH_SIZE` batch nhiều counterfactual copies của cùng
conversation để giảm số forward calls; giảm biến này nếu GPU thiếu bộ nhớ.

## Fairness của B0–B6

- B1/B2 dùng cùng `ModalityGateNetwork`; chỉ vị trí late/propagation khác nhau.
- B3/B4/B6 dùng cùng `PairGateNetwork`; chỉ auxiliary supervision khác nhau.
- B5 tự chọn residual width gần nhất với số parameter của B3 và vẫn báo cáo
  actual parameter count/FLOPs/runtime/memory.
- Tất cả dùng cùng base checkpoint, frozen contextual backbone và trainable
  final classifier. Validation NLL chọn checkpoint; test không chọn threshold.
- Gate khởi tạo gần identity, self edge luôn bằng 1, có identity regularization.
- E4 chạy random/distance/speaker/margin controls với đúng số edge suppression
  của learned B4 trong từng batch.

