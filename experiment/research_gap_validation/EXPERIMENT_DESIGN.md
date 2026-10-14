# Thiết kế triển khai và validity contract

## 1. Mapping từ giả thuyết sang phép kiểm tra

| Hypothesis | Estimand chính | Contrast quyết định |
|---|---|---|
| H1: heterogeneous contribution | Tỷ lệ target–modality có ít nhất một `I>eps` và một `I<-eps` | Trong cùng target/modality; sensitivity theo source count, effect size, margin và perturbation |
| H2: location matters | Khả năng giảm target NLL dưới cùng checkpoint và budget đã khai báo | E2 early-edge oracle vs late-modality oracle; E3 B2−B1 |
| H3: learnable benefit | Held-out target-level ACC/WF1/macro-F1/NLL | B3−B2, B4−B3, B4−B6 và B3/B4−B5 |

H2 không được suy ra chỉ từ E2 vì early và late có action-space granularity khác
nhau. Kết luận learned placement chỉ dựa vào B2−B1, nơi gate network, inputs,
capacity, seed và optimization budget giống nhau.

## 2. Vị trí can thiệp đã xác minh

`MultiHeadedAttention.softmax` trả tensor `[B,H,target,source]`. Gate được nhân
sau softmax và trước dropout/value aggregation. Với source modality `text`, một
edge được áp dụng đồng thời lên `t_t`, `t_a`, `t_v`; tương tự cho audio/visual.

Late gate được hook vào output của `features_reduce_t/a/v`, tức enhanced
representations sau chín contextual branches nhưng trước `last_gate` và
classifier. Đây là late contextualized modality contribution, không phải raw
input modality.

## 3. Action-space contract của E2

- Early: tối đa một `(source,target,modality)` edge, gate grid cố định.
- Late: tối đa một `(target,modality)` representation, cùng gate grid.
- Identity (`gate=1`) được lưu rõ; self edges không bao giờ bị đóng.
- `normalized_cost` early là attention mass bị loại; late là representation norm
  bị loại. Pareto plot phải được xem cùng raw budget vì hai cost không đồng đơn vị.
- Input-source train-mean được chạy lại để lưu đủ logits/probabilities; raw E1
  chỉ dùng làm equivalence/retention reference.

## 4. Training contract của E3

- Primary: frozen contextual backbone, trainable final classifier và module mới.
- Secondary: `--unfreeze-backbone`, áp dụng giống nhau cho toàn bộ models.
- B1/B2 dùng cùng kiến trúc `ModalityGateNetwork`.
- B3/B4/B6 dùng cùng kiến trúc `PairGateNetwork`.
- B4 teacher chỉ lấy propagation-edge interventions của train conversations,
  dùng đúng attention hook/action space của learned gate. Teacher lưu continuous
  best-gate target và confidence theo NLL decision margin; B4 tối ưu weighted
  soft-MSE thay vì hard BCE 0/1.
- B6 relevance target chỉ dùng distance và speaker identity, không dùng emotion,
  correctness, impact hoặc target label.
- Validation NLL chọn epoch; test chỉ được đọc sau khi state tốt nhất đã khóa.
- B5 residual width được chọn gần số parameter của B3; actual params/FLOPs/time
  vẫn được báo cáo thay vì tuyên bố capacity bằng tuyệt đối.

## 5. Những claim bị chặn bằng thiết kế

- Không gọi signed impact là causal reliability.
- Không dùng pair-weighted ACC/WF1.
- Không gọi one-branch suppression là xóa toàn bộ source effect.
- Không gọi checkpoint-split exploratory test là unseen-session confirmation nếu
  checkpoint đã được lựa chọn sau khi xem test.
- Không kết luận propagation placement thắng nếu B2−B1 không nhất quán theo
  seed/fold hoặc nếu late oracle ngang early oracle.
- Không kết luận impact supervision hữu ích nếu B4 không hơn cả B3 và B6.

## 6. D1 harmful-edge learnability contract

- Label D1 được tạo bằng frozen SDT và đúng direct propagation-edge action:
  một `source→target→modality` gate bằng 0, mọi edge còn lại bằng 1.
- Nếu checkpoint không có validation split, một session hoàn chỉnh trong train
  được giữ riêng cho predictor validation. Conversation không xuất hiện ở nhiều
  predictor split.
- `eps_strong`, epoch và supervision variant chỉ được chọn trên validation.
  Test impact chỉ dùng để tính metric sau khi model đã khóa.
- Binary loại neutral và học harmful so với beneficial. Three-class giữ
  beneficial/neutral/harmful. Magnitude-aware hồi quy signed impact và tăng
  trọng số cho effect lớn.
- Input chỉ gồm baseline uncertainty/probability summaries, projected-feature
  similarity, attention mass, modality, speaker, direction, distance và position.
  True emotion, correctness, signed impact và suppressed outputs bị cấm tại
  inference.
- Báo cáo đồng thời `scope=all` và `scope=strong`. AUROC gần 0.5 trên strong
  pairs là negative finding về predictability dưới representation/predictor đã
  thử, không phải bằng chứng rằng harm về nguyên tắc không thể học được.

