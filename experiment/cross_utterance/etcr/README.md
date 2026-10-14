# Experiment 3 — Emotion-Transition-Aware Contextual Reliability (ETCR)

Pipeline này chỉ can thiệp text của utterance gần nhất đứng trước target và cùng
speaker. SDT checkpoint luôn ở `eval()` và toàn bộ parameter được freeze; không
train lại SDT.

## Chạy

Sửa `CHECKPOINT`, `FEATURE_PATH`, `RUN_DIR`, `DEVICE` và `GPU_ID` trong
`../experiment_config.sh`, sau đó từ Linux server chạy:

```bash
cd /path/to/SDT_cross/experiment/cross_utterance/etcr
bash run_experiment3.sh
```

Có thể chạy riêng từng phase:

```bash
bash run_phase_a.sh
bash run_phase_b.sh
bash run_phase_c.sh
```

`run_phase_a.sh` và `run_phase_b.sh` tự sinh test cache nếu chưa có.
`run_phase_c.sh` tự sinh train/valid/test `train_mean` cache nếu thiếu. Để smoke
test nhanh có thể export `BOOTSTRAP_REPS=50`; kết quả chính thức nên giữ mặc
định 2,000.

Mặc định test cache bắt buộc đối chiếu với
`RUN_DIR/baseline/baseline_predictions.csv` của Experiment 1 Task A. Chỉ khi
chủ động chạy độc lập mới dùng `export ALLOW_MISSING_BASELINE=1`; khi đó gate
equivalence này bị bỏ qua và manifest vẫn ghi provenance checkpoint/feature.

## Dữ liệu trung gian

- `RUN_DIR/etcr/raw/{split}_{strategy}.csv`: đúng một dòng trên mỗi target,
  gồm full/masked logits và probabilities của mọi class.
- `RUN_DIR/etcr/raw/{split}_text_embeddings.npz`: source/target embeddings căn
  hàng tuyệt đối với CSV.
- Target không có previous same-speaker source có `eligible=0`; masked output
  được đặt bằng full output và mọi policy bắt buộc giữ nguyên.

Không tái sử dụng trực tiếp `impact_pairs_*.csv` của Experiment 1 vì các file đó
không có masked logits/probabilities của toàn bộ class, nên không đủ để tính `R`.

## Output

### Phase A

- `phase_a/emotion_persistence_pairs.csv`
- `phase_a/emotion_persistence_summary.csv`
- `figures/phase_a_bias.png`

### Phase B

- `phase_b/persistence_error_transitions.csv`
- `phase_b/target_level_metrics.csv`
- `phase_b/adjusted_transition_effects.csv`
- `phase_b/strategy_robustness.csv`
- `phase_b/oracle_metrics.csv`

Adjusted model dùng logistic regression của scikit-learn và conversation-cluster
bootstrap, không dùng statsmodels.

### Phase C

- `phase_c/train_val_labels.csv`
- `phase_c/predictor_metrics.csv`
- `phase_c/policy_decisions.csv`
- `phase_c/mitigation_metrics.csv`
- `phase_c/reliability_predictor.pkl`

PCA embedding chỉ fit trên train; predictor chỉ fit trên train; threshold chỉ
chọn trên validation. Test label/impact không tham gia feature, fit hoặc chọn
threshold. Oracle là ngoại lệ được đánh dấu rõ và chỉ dùng làm diagnostic upper
bound.

Sau khi chạy đủ A–B–C, `report.md` được sinh tự động bằng kết quả thực tế.

## Quy ước dấu

`signed_impact = NLL_mask - NLL_full`:

- `< -eps`: masking giúp giảm loss, context gốc harmful;
- `> eps`: giữ context giúp giảm loss, context gốc beneficial;
- còn lại: near-zero.

