# SDT_cross

Thư mục độc lập cho SDT gốc và các thí nghiệm cross-utterance. Không chứa
appraisal, COLD, TiCAL, emotion wheel hoặc dữ liệu pickle.

## Cấu trúc

- `model.py`, `sdt_backbone.py`: kiến trúc SDT gốc; `model.py` chỉ bổ sung
  dictionary output để các experiment đọc logits/probabilities.
- `losses.py`: đúng objective SDT gồm fused CE, ba unimodal CE và ba KL
  self-distillation terms.
- `train.py`: train/evaluate và lưu checkpoint có manifest split.
- `experiment/cross_utterance`: toàn bộ experiment cross-utterance/ETCR.
- `experiment/research_gap_validation`: E1-E4, D1 và các audit.

Data không được copy. Mặc định loader tìm lần lượt:

1. `SDT_cross/data/<dataset>_multimodal_features.pkl`;
2. thư mục data của project gốc cạnh nó: `../SDT/data/...`.

Cũng có thể truyền `--feature-path /path/to/features.pkl`.

## Train

Mặc định dùng tập validation tách từ training conversations để chọn epoch:

```bash
bash exec_iemocap.sh --gpu-id 0
```

Để tái hiện cách chọn checkpoint trên test của repository SDT gốc:

```bash
bash exec_iemocap.sh --selection-protocol test --gpu-id 0
```

Hai protocol chỉ khác split dùng để chọn epoch. Kiến trúc, loss, optimizer và
hyperparameters SDT không đổi. `validation` là mặc định vì không dùng test label
để model selection.

## Cross-utterance experiments

Sửa `CHECKPOINT` một lần trong:

- `experiment/cross_utterance/experiment_config.sh`;
- `experiment/research_gap_validation/configs/experiment_config.sh`.

Hai config mặc định dùng data ở `../SDT/data` và ghi output bên trong
`SDT_cross/experiment/.../outputs`; có thể override bằng environment variables.
Chi tiết từng task nằm trong README của mỗi experiment.
