# Phân tích ảnh hưởng cross-utterance của Frozen SDT

Thư mục này hiện thực Experiment 1: đo xem đặc trưng của một utterance nguồn ảnh hưởng như thế nào đến dự đoán cảm xúc của một utterance đích khác trong cùng hội thoại. Mô hình SDT và checkpoint được giữ nguyên hoàn toàn; thí nghiệm chỉ thay thế đầu vào rồi chạy suy luận lại.

README này đồng thời là tài liệu mô tả thí nghiệm, báo cáo tổng hợp kết quả của lần chạy đầy đủ hiện tại, hướng dẫn diễn giải từng file và kế hoạch phân tích tiếp theo. Kế hoạch thiết kế ban đầu nằm tại `SDT_Experiment1_Cross_Utterance_Impact_Plan.md`.

## 1. Câu hỏi nghiên cứu

Với một hội thoại có utterance nguồn `j`, utterance đích `i` (`i != j`) và modality `m` thuộc `{text, audio, visual}`:

> Nếu thay riêng đặc trưng modality `m` của utterance nguồn `j`, dự đoán cảm xúc tại utterance đích `i` thay đổi bao nhiêu?

Hai cách thay thế được sử dụng:

- `zero`: thay vector đặc trưng nguồn bằng vector 0;
- `train_mean`: thay bằng vector trung bình được tính chỉ từ training split.

Mỗi lần can thiệp chỉ thay **một modality của một utterance nguồn**. Vị trí utterance, speaker embedding, attention mask, các modality khác và toàn bộ tham số mô hình được giữ nguyên. Mô hình vẫn nhận toàn bộ hội thoại, vì vậy đây là **full-conversation contextual impact**, không phải suy luận chỉ dựa trên lịch sử.

## 2. Thiết lập của lần chạy được báo cáo

| Thuộc tính | Giá trị |
|---|---:|
| Dataset / split | IEMOCAP / test |
| Checkpoint ID | `5150fd0cf373d005` |
| Số hội thoại test | 31 |
| Số utterance hợp lệ | 1.623 |
| Số cặp nguồn–đích trên mỗi modality và strategy | 92.724 |
| Tổng số dòng trên mỗi strategy | 278.172 |
| Ngưỡng near-zero chính `impact_eps` | `1e-4` |

Nhãn cảm xúc trong các bảng số:

| ID | Cảm xúc |
|---:|---|
| 0 | happy |
| 1 | sad |
| 2 | neutral |
| 3 | angry |
| 4 | excited |
| 5 | frustrated |

## 3. Các đại lượng cần phân biệt

### 3.1 Signed predictive impact

Đại lượng chính là:

```text
signed_impact = loss_mask(target) - loss_full(target)
              = log P_full(y_target) - log P_mask(y_target)
```

- `impact > 0`: thay đặc trưng nguồn làm loss của nhãn thật tăng; thông tin nguồn đã hỗ trợ dự đoán đích dưới phép can thiệp này.
- `impact < 0`: thay đặc trưng nguồn làm loss của nhãn thật giảm; thông tin nguồn đã gây bất lợi dưới phép can thiệp này.
- `|impact| <= impact_eps`: thay đổi quá nhỏ theo ngưỡng đang xét.

Impact đo thay đổi loss/xác suất của nhãn thật. Nó có thể khác 0 dù nhãn dự đoán cuối cùng không đổi.

### 3.2 Prediction flip

```text
changed_prediction = (pred_full != pred_mask)
```

Flip chỉ xảy ra khi lớp top-1 đổi. Đây là tiêu chuẩn mạnh hơn impact: mô hình có thể thay đổi xác suất đáng kể nhưng vẫn giữ nguyên argmax.

### 3.3 Baseline margin

```text
baseline_margin = P_full(top-1) - P_full(top-2)
```

Margin nhỏ nghĩa là quyết định gốc nằm gần ranh giới phân lớp. Kết quả cho thấy đây là biến giải thích flip mạnh nhất.

### 3.4 Hai loại ACC/F1

- **Test ACC/WF1 chuẩn**: mỗi utterance test xuất hiện đúng một lần. Được lấy từ baseline hoặc `test_classification_metrics.csv` sau khi loại trùng theo `(conversation_id, target_idx)`.
- **Pair-weighted ablation ACC/WF1**: mỗi utterance đích lặp lại một lần cho mỗi nguồn được can thiệp. Đây là số trong `ablation_classification_metrics.csv`; utterance thuộc hội thoại dài có trọng số lớn hơn.

Không được trình bày pair-weighted ACC/WF1 như test ACC/WF1 chuẩn sau ablation. Thiết kế hiện tại can thiệp từng nguồn riêng lẻ nên không tạo ra duy nhất một dự đoán ablated cho mỗi utterance test.

## 4. Kết quả signed impact tổng thể

| Strategy | Modality | Mean | Median | Positive | Negative | Near zero |
|---|---|---:|---:|---:|---:|---:|
| zero | text | -0,00000288 | 0,0001145 | 50,51% | 38,11% | 11,39% |
| zero | audio | -0,0001916 | 0,0000317 | 45,20% | 37,50% | 17,29% |
| zero | visual | -0,0000400 | -0,0000027 | 20,58% | 24,84% | 54,59% |
| train_mean | text | -0,0002536 | 0,0001341 | 51,18% | 37,41% | 11,41% |
| train_mean | audio | -0,0001916 | 0,0000317 | 45,20% | 37,50% | 17,30% |
| train_mean | visual | 0,0000004 | 0,0000005 | 14,79% | 12,98% | 72,23% |

Ý nghĩa:

- **Text tạo ra thay đổi mạnh và thường xuyên nhất.** Khoảng một nửa số cặp có impact dương vượt `1e-4`, và chỉ khoảng 11% là near-zero.
- **Audio đứng thứ hai.** Phân phối hẹp hơn text và có khoảng 17% near-zero.
- **Visual rất yếu**, đặc biệt với `train_mean`, nơi 72,23% cặp là near-zero.
- Mean gần 0 không có nghĩa là “không có impact”. Impact dương và âm có thể triệt tiêu nhau khi lấy trung bình. Median, độ phân tán và tỷ lệ dấu cho thấy text/audio vẫn tạo thay đổi ở nhiều cặp.
- Mean âm rất nhỏ không đủ để kết luận context nhìn chung có hại. Đây là trung bình gộp trên nhiều dòng phụ thuộc lẫn nhau và bị chi phối bởi cả độ lớn lẫn dấu.

### Độ nhạy theo ngưỡng near-zero

Khi tăng `impact_eps` từ `1e-6` lên `1e-3`, tỷ lệ near-zero tăng mạnh:

| Strategy / modality | Near-zero tại `1e-6` | tại `1e-4` | tại `1e-3` |
|---|---:|---:|---:|
| zero / text | 0,54% | 11,39% | 35,15% |
| zero / audio | 0,76% | 17,29% | 47,80% |
| zero / visual | 7,35% | 54,59% | 94,06% |
| train_mean / text | 0,47% | 11,41% | 35,03% |
| train_mean / audio | 0,76% | 17,30% | 47,81% |
| train_mean / visual | 9,20% | 72,23% | 98,79% |

Do đó “có ảnh hưởng” phải luôn đi kèm thang đo và ngưỡng. Kết luận tương đối `text > audio >> visual` vẫn nhất quán qua các ngưỡng, nhưng tỷ lệ tuyệt đối không bất biến.

## 5. Độ bền giữa zero và training mean

| Modality | Pearson | Spearman | Đồng thuận dấu trên các cặp đủ lớn |
|---|---:|---:|---:|
| text | 0,9529 | 0,8883 | 0,9206 |
| audio | 1,0000 | 1,0000 | 1,0000 |
| visual | 0,3244 | 0,2532 | 0,6673 |

Ý nghĩa:

- **Text có robustness cao**: độ lớn, thứ hạng và dấu impact khá ổn định giữa hai cách thay thế.
- **Audio giống hệt không phải là bằng chứng robustness độc lập.** Vector train mean của audio gần như vector 0 (`mean_to_feature_rms ≈ 1,88e-7`), nên hai can thiệp về thực tế là cùng một phép thử.
- **Visual không robust giữa hai strategy.** Zero khác xa phân phối đặc trưng trung bình, trong khi train mean của visual gần với thang đặc trưng thông thường (`mean_to_feature_rms ≈ 0,937`). Vì vậy zero có khả năng là đầu vào ngoài phân phối đối với visual.

Tóm lại, bằng chứng đáng tin nhất về ảnh hưởng cross-utterance hiện nằm ở text. Audio cần một baseline thay thế khác zero; visual hiện chưa có bằng chứng ổn định.

## 6. Độ ổn định ở mức quyết định phân lớp

| Strategy | Modality | Số flip | Flip rate |
|---|---|---:|---:|
| zero | text | 608 / 92.724 | 0,6557% |
| train_mean | text | 644 / 92.724 | 0,6945% |
| zero / train_mean | audio | 260 / 92.724 | 0,2804% |
| zero | visual | 0 / 92.724 | 0% |
| train_mean | visual | 7 / 92.724 | 0,00755% |

Điều này hỗ trợ kết luận:

> Mặc dù nhiều can thiệp làm thay đổi xác suất và loss, can thiệp một source-modality hiếm khi đổi nhãn cảm xúc top-1; quyết định cuối của mô hình tương đối ổn định.

“Ổn định” ở đây không có nghĩa source không ảnh hưởng. Phần lớn thay đổi chưa đủ lớn để vượt qua khoảng cách giữa lớp top-1 và top-2.

### Pair-weighted ACC/F1

Baseline pair-weighted chung là ACC `73,9787`, WF1 `74,1252`, macro-F1 `73,5771`. Thay đổi lớn nhất vẫn rất nhỏ:

| Strategy / modality | ΔACC (điểm %) | ΔWF1 (điểm %) | Tỷ lệ đổi dự đoán |
|---|---:|---:|---:|
| zero / text | -0,0388 | -0,0413 | 0,6557% |
| train_mean / text | -0,0431 | -0,0425 | 0,6945% |
| zero hoặc mean / audio | -0,0421 | -0,0358 | 0,2804% |
| zero / visual | 0 | 0 | 0% |
| train_mean / visual | -0,00755 | -0,00676 | 0,00755% |

Các delta nhỏ vì flip hiếm và vì flip gồm cả `correct→wrong` lẫn `wrong→correct`, hai chiều bù trừ nhau.

### Hướng chuyển đúng/sai

| Strategy / modality | Correct→wrong | Wrong→correct | Wrong→wrong nhưng đổi lớp |
|---|---:|---:|---:|
| zero / text | 276 | 240 | 92 |
| train_mean / text | 306 | 266 | 72 |
| zero hoặc mean / audio | 145 | 106 | 9 |
| zero / visual | 0 | 0 | 0 |
| train_mean / visual | 7 | 0 | 0 |

Text và audio đều có nhiều `correct→wrong` hơn `wrong→correct`, nên ACC/WF1 giảm nhẹ. Tuy nhiên, số `wrong→correct` đáng kể cho thấy context không phải lúc nào cũng hỗ trợ; có source cụ thể đang đẩy mô hình theo hướng sai.

## 7. Tại sao prediction bị đổi?

### 7.1 Baseline margin là yếu tố mạnh nhất

| Margin | Mean text | Zero text | Audio | Mean visual |
|---|---:|---:|---:|---:|
| `[0,00; 0,01)` | 31,99% | 28,46% | 20,65% | 1,76% |
| `[0,01; 0,02)` | 17,31% | 16,94% | 8,78% | 0% |
| `[0,02; 0,05)` | 11,02% | 10,76% | 2,38% | 0% |
| `[0,05; 0,10)` | 4,47% | 4,07% | 1,18% | 0% |
| `[0,10; 0,20)` | 1,11% | 1,13% | 0,39% | 0% |
| `[0,20; 0,40)` | 0,274% | 0,231% | 0,128% | 0% |
| `[0,40; 1,00]` | 0,0085% | 0,0071% | 0,0085% | 0% |

Chỉ 5,61% cặp có margin dưới 0,1 nhưng nhóm này chứa khoảng 84% số flip của text và audio. Flip vì vậy chủ yếu là **can thiệp làm vượt ranh giới ở những target vốn đã không chắc chắn**, không phải thay đổi mạnh hàng loạt các dự đoán tự tin.

Tương quan Spearman giữa margin của target và tỷ lệ source làm target đó flip cũng âm mạnh: khoảng `-0,705` cho zero/text, `-0,739` cho mean/text và `-0,677` cho audio.

### 7.2 Khoảng cách nguồn–đích

| Distance | Mean text | Zero text | Audio |
|---|---:|---:|---:|
| 1–5 | 1,712% | 1,575% | 0,301% |
| 6–10 | 0,887% | 0,953% | 0,305% |
| 11–20 | 0,547% | 0,490% | 0,298% |
| 21–40 | 0,398% | 0,369% | 0,283% |
| 41–60 | 0,207% | 0,187% | 0,236% |
| 61+ | 0,108% | 0,072% | 0,036% |

- Text có locality rất rõ: utterance gần target dễ làm prediction đổi hơn.
- Các cặp cách 1–10 vị trí chỉ chiếm khoảng 31% số cặp nhưng tạo gần 60% flip text.
- Audio gần như phẳng trong khoảng 1–40 rồi mới giảm, nên phạm vi ảnh hưởng audio có vẻ rộng hơn text hoặc ít phụ thuộc khoảng cách hơn.
- Visual có quá ít flip để suy luận xu hướng khoảng cách.

Đây mới là quan hệ mô tả. Distance có thể liên hệ với speaker relation, cảm xúc và margin, nên cần mô hình đa biến trước khi nói distance có hiệu ứng độc lập.

### 7.3 Source ở trước hay sau target

| Modality | Source sau target | Source trước target |
|---|---:|---:|
| train_mean / text | 0,7247% | 0,6643% |
| zero / text | 0,6988% | 0,6126% |
| audio | 0,3128% | 0,2480% |

Source phía sau target có flip rate cao hơn một chút. Điều này phù hợp với attention hai chiều của SDT: prediction hiện tại dùng cả quá khứ lẫn tương lai. Kết quả này **không thể** được diễn giải như mô hình causal/history-only.

### 7.4 Quan hệ speaker

| Modality | Same speaker | Cross speaker | Tỷ số xấp xỉ |
|---|---:|---:|---:|
| train_mean / text | 1,2344% | 0,1495% | 8,3× |
| zero / text | 1,1486% | 0,1582% | 7,3× |
| audio | 0,4358% | 0,1235% | 3,5× |

Khoảng 88–89% flip text và 78% flip audio đến từ source cùng speaker, trong khi số cặp same/cross gần cân bằng. Đây là một trong những phát hiện mạnh nhất: SDT đặc biệt nhạy với chuỗi phát ngôn của cùng người nói.

Tuy nhiên chưa nên gọi đây là “speaker effect” độc lập, vì cùng speaker có thể đồng thời gần hơn về khoảng cách hoặc gắn với target margin thấp. Bước tiếp theo cần kiểm soát các biến này đồng thời.

### 7.5 Cảm xúc của source

Flip rate nổi bật:

- Với text, source `frustrated` và `excited` thường gây flip nhất; `neutral` đứng tiếp theo.
- Với audio, source `excited` cao nhất (`0,510%`), còn `sad` thấp nhất (`0,080%`).
- Kết quả zero và train mean của text có khác nhau nhẹ theo từng lớp nhưng giữ xu hướng tổng quát.
- Visual không đủ flip để kết luận theo source emotion.

Đây là tỷ lệ đã chuẩn hóa theo số cặp của từng source emotion, tốt hơn so sánh raw count. Dù vậy emotion, speaker và distance vẫn có thể gây nhiễu lẫn nhau.

### 7.6 Cảm xúc của target

| Target | Mean text | Zero text | Audio |
|---|---:|---:|---:|
| happy | 0,257% | 0,150% | 0,150% |
| sad | 0,541% | 0,559% | 0,230% |
| neutral | 0,844% | 0,768% | 0,287% |
| angry | 0,357% | 0,441% | 0,052% |
| excited | 0,768% | 0,617% | 0,494% |
| frustrated | 0,893% | 0,936% | 0,280% |

Target `frustrated` nhạy nhất với text, tiếp theo là `neutral`/`excited`. Audio tác động mạnh nhất lên target `excited` và rất ít lên `angry`. Đây có thể phản ánh độ khó/phần chồng lấn lớp, đặc biệt giữa `neutral`, `excited` và `frustrated`, chứ chưa chứng minh một cơ chế cảm xúc riêng biệt.

### 7.7 Các chuyển lớp cụ thể

Các flow đáng chú ý của text tập trung ở `neutral ↔ frustrated` và `neutral ↔ excited`. Audio nổi bật ở `excited ↔ happy`, `neutral ↔ frustrated` và `angry ↔ frustrated`.

Không nên kết luận chỉ từ raw transition count, vì tần suất lớp gốc khác nhau. Cần chuẩn hóa từng hàng của ma trận chuyển lớp hoặc báo cáo xác suất có điều kiện theo `pred_full`/`target_emotion`.

## 8. Flip có robust giữa zero và train mean không?

| Modality | Không flip | Chỉ zero | Chỉ mean | Cả hai, cùng lớp mới | Cả hai, khác lớp mới |
|---|---:|---:|---:|---:|---:|
| text | 91.988 | 92 | 128 | 511 | 5 |
| audio | 92.464 | 0 | 0 | 260 | 0 |
| visual | 92.717 | 0 | 7 | 0 | 0 |

Với text, hai strategy cùng phát hiện 516 cặp flip; 511/516 cặp đi tới cùng lớp mới. Tập flip giao nhau chiếm 84,9% flip zero và 80,1% flip mean; Jaccard khoảng 70,1%. Như vậy phần lớn flip text là có thể tái hiện giữa hai phép thay thế, nhưng vẫn có một vùng biên phụ thuộc strategy.

Audio trùng khớp hoàn toàn vì zero và train mean gần như giống nhau ở feature space. Visual chỉ flip với train mean và không cho thấy robustness.

## 9. Flip tập trung ở những target nào?

| Strategy / modality | Target có ít nhất một flip | Tỷ lệ trên 1.623 target | Số hội thoại chứa target nhạy |
|---|---:|---:|---:|
| zero / text | 149 | 9,18% | 30 / 31 |
| train_mean / text | 153 | 9,43% | 30 / 31 |
| audio | 86 | 5,30% | 22 / 31 |
| zero / visual | 0 | 0% | 0 |
| train_mean / visual | 1 | 0,062% | 1 |

Median số source làm flip trên một target nhạy là 2 cho text và audio. Vì vậy flip không chỉ đến từ một vài target duy nhất, dù mức độ tập trung vẫn đáng kể:

- top 10 target chiếm khoảng 27,0% flip zero/text;
- top 10 chiếm 30,4% flip mean/text;
- top 10 chiếm 47,3% flip audio.

Một số target nhạy nhất:

- `Ses05M_impro05:18`, target neutral, dự đoán gốc đúng, margin `0,002554`: 18/41 source làm flip với zero/text, 26/41 với mean/text và 12/41 với audio.
- `Ses05F_impro05:40`, target neutral, dự đoán gốc đúng, margin `0,007559`: 27/90 zero/text, 37/90 mean/text và 10/90 audio.
- `Ses05F_script03_1:36`, target neutral, dự đoán gốc đúng, margin `0,000114`: 23/36 audio và 7/36 mean/visual; đây là target visual duy nhất.
- `Ses05F_script03_1:12`, target excited nhưng baseline dự đoán happy, margin `0,007765`: 22/36 mean/text và 12/36 zero/text.

Các ví dụ này củng cố kết luận rằng susceptibility chủ yếu đến từ target có margin cực thấp.

Đặc biệt, **toàn bộ 7 visual flip cùng thuộc một target rất mơ hồ** (`Ses05F_script03_1:36`), target này cũng nhạy với text/audio. Do đó không có bằng chứng cho một hiệu ứng visual cross-utterance tổng quát.

## 10. Kết luận nghiên cứu hiện tại

1. SDT có sử dụng thông tin cross-utterance: thay một source feature có thể làm thay đổi loss, xác suất và đôi khi cả lớp dự đoán của target khác.
2. Mức ảnh hưởng theo modality là **text mạnh nhất, audio thứ hai, visual rất yếu**.
3. Text impact tương đối robust giữa zero và train mean. Audio chưa có kiểm tra robustness độc lập; visual không robust.
4. Ở mức quyết định top-1, mô hình khá ổn định: flip rate dưới 0,7% cho mọi modality/strategy.
5. Flip tập trung mạnh ở target có baseline margin thấp, source gần target và source cùng speaker.
6. Text có tính cục bộ theo khoảng cách rõ hơn audio. Cả source trước và sau target đều tác động, phù hợp với kiến trúc hai chiều.
7. Một số nhóm cảm xúc nhạy hơn, đặc biệt target `frustrated`, `neutral`, `excited`, nhưng cần kiểm soát margin và các biến liên quan trước khi diễn giải sâu.
8. Kết quả visual không đủ để khẳng định hiệu ứng tổng quát; bảy flip chỉ là một target ngoại lệ.

Phát biểu an toàn cho báo cáo:

> Các can thiệp đơn nguồn tạo ra thay đổi có hệ thống ở xác suất và loss, đặc biệt qua modality văn bản. Tuy nhiên, chúng hiếm khi thay đổi nhãn dự đoán cuối cùng. Những thay đổi quyết định chủ yếu tập trung ở các target vốn có độ phân cách top-1/top-2 thấp, ở source gần target và cùng người nói. Kết quả mô tả ảnh hưởng dự đoán phụ thuộc mô hình và phép can thiệp, không phải quan hệ nhân quả trong thế giới thực.

## 11. Giới hạn diễn giải

- Đây là **predictive intervention**, không phải causal effect thực tế.
- Thay feature không xóa hoàn toàn utterance: position, speaker embedding và cấu trúc sequence vẫn còn.
- SDT dùng full-conversation bidirectional context; source phía sau target được phép ảnh hưởng target.
- Các dòng cặp trong cùng hội thoại phụ thuộc mạnh vào nhau. Không được coi 92.724 dòng là các quan sát IID để chạy kiểm định thông thường.
- Zero replacement có thể ngoài phân phối, đặc biệt với visual.
- Audio zero và train mean gần như cùng một intervention, nên correlation 1,0 không phải xác nhận từ hai perturbation độc lập.
- Pair-weighted metrics ưu tiên hội thoại dài và không phải test metrics chuẩn.
- Các bảng một biến chưa tách được ảnh hưởng riêng của margin, distance, speaker và emotion.
- Kết quả mới dựa trên một checkpoint và một test split IEMOCAP.

## 12. Nên làm gì tiếp theo?

Các ưu tiên 1–3 bên dưới hiện đã được hiện thực trong structural-harm workflow (`analyze_contextual_relationships.py` và `fit_structural_harm_mixed_effects.py`). Chúng vẫn được giữ ở đây như rationale phân tích và checklist diễn giải kết quả sau khi chạy.

### Ưu tiên 1 — Phân tích đa biến cho xác suất flip

Fit mô hình logistic có biến phụ thuộc `changed_prediction`, với các biến giải thích:

```text
baseline_margin + log(distance) + same_speaker + source_direction
+ source_emotion + target_emotion + modality + replacement_strategy
+ các interaction quan trọng
```

Nên dùng cluster-robust standard error theo conversation hoặc bootstrap theo conversation. Phân tích này trả lời liệu same-speaker và distance còn quan trọng sau khi đã kiểm soát margin hay không. Do flip hiếm, cân nhắc Firth logistic, penalized logistic hoặc bootstrap thay vì logistic MLE đơn giản.

### Ưu tiên 2 — Mô hình hóa signed impact thay vì chỉ flip

Flip làm mất phần lớn tín hiệu liên tục. Hãy phân tích `signed_impact` hoặc `abs(signed_impact)` bằng mô hình phân cấp/mixed-effects, với conversation và target là nhóm. Báo cáo effect size và conversation-level confidence interval, không chỉ p-value.

### Ưu tiên 3 — Tách ba yếu tố margin, speaker và distance

Sinh các bảng/heatmap tương tác:

- same-speaker × distance;
- margin bin × distance;
- margin bin × same-speaker;
- modality × same-speaker × distance.

Điều này kiểm tra xem tỷ lệ flip cao của same-speaker có chỉ vì utterance cùng speaker thường ở gần target hơn hay không.

### Ưu tiên 4 — Kiểm tra các target/case ngoại lệ

Đọc transcript và phân phối xác suất của các target nhạy nhất, nhất là:

- `Ses05M_impro05:18`;
- `Ses05F_impro05:40`;
- `Ses05F_script03_1:36`;
- `Ses05F_script03_1:12`.

Với mỗi case, hiển thị source text, speaker, distance, emotion, xác suất trước/sau và lớp chuyển. Mục tiêu là xác định mô hình đang dùng continuity, contrast, speaker history hay chỉ phản ứng với nhiễu gần decision boundary.

### Ưu tiên 5 — Bổ sung perturbation thực tế hơn

Để robustness có ý nghĩa hơn:

- dùng sampled in-distribution replacement theo modality;
- permutation trong cùng speaker hoặc cùng emotion nhưng không dùng nhãn ở inference;
- learned mask token hoặc mask-aware fine-tuning trong một experiment riêng;
- kiểm tra noise có scale khớp phân phối feature.

Đặc biệt cần baseline khác cho audio vì mean gần bằng zero, và cần tránh dùng zero làm bằng chứng chính cho visual.

### Ưu tiên 6 — Nếu cần ACC/WF1 sau ablation chuẩn

Thiết kế một experiment mới tạo đúng một prediction cho mỗi target, ví dụ:

- ablate toàn bộ context ngoài target cho một modality;
- chỉ giữ cửa sổ context cố định;
- ablate toàn bộ source trước target hoặc toàn bộ source cùng speaker;
- tổng hợp một tập source được xác định trước rồi chạy một forward duy nhất cho mỗi target/conversation.

Khi đó mới tính được utterance-level test ACC/WF1 sau ablation và so sánh trực tiếp với baseline chuẩn. Không nên suy ra con số này từ các singleton intervention hiện tại bằng cách chọn tùy ý một dòng.

### Ưu tiên 7 — Kiểm tra khả năng khái quát

- lặp lại trên nhiều checkpoint/seed;
- bootstrap theo conversation;
- nếu pipeline hỗ trợ, xác nhận trên MELD hoặc split khác;
- so sánh full-context với history-only để đánh giá vai trò future context.

## 13. Ý nghĩa của từng file kết quả

| File | Nội dung và cách dùng |
|---|---|
| `baseline/baseline_predictions.csv` | Một dòng trên mỗi utterance; dự đoán gốc, label và xác suất baseline. |
| `baseline/baseline_metrics.json` | Test metrics chuẩn của baseline. |
| `feature_means/train_feature_means.npz` | Vector mean của từng modality, chỉ tính từ training split. |
| `results/impact_pairs_zero.csv` | Toàn bộ cặp nguồn–đích với zero replacement. |
| `results/impact_pairs_train_mean.csv` | Toàn bộ cặp nguồn–đích với train-mean replacement. |
| `results/impact_pairs.csv` | Hai strategy ghép lại, nếu bước merge được chạy. |
| `results/summary_by_modality.csv` | Mean/median/quantile, tỷ lệ dấu và CI theo modality. |
| `results/impact_eps_sensitivity.csv` | Độ nhạy positive/negative/near-zero theo `impact_eps`. |
| `results/summary_by_distance.csv` | Signed impact theo khoảng cách và hướng nguồn–đích. |
| `results/summary_by_speaker_relation.csv` | Signed impact theo cùng/khác speaker. |
| `results/masking_robustness.csv` | Pearson, Spearman và sign agreement giữa zero/mean. |
| `results/ablation_classification_metrics.csv` | ACC/WF1/macro-F1 pair-weighted; không phải test metrics chuẩn. |
| `results/test_classification_metrics.csv` | Baseline test metrics sau khi deduplicate target. |
| `results/changed_prediction_summary.csv` | Tổng số flip, confidence/margin/entropy của nhóm flip và không flip. |
| `results/flip_rate_by_baseline_margin.csv` | Flip rate theo độ chắc chắn ban đầu. |
| `results/flip_rate_by_distance.csv` | Flip rate theo khoảng cách nguồn–đích. |
| `results/flip_rate_by_source_direction.csv` | So sánh source trước và sau target. |
| `results/flip_rate_by_speaker_relation.csv` | So sánh cùng và khác speaker. |
| `results/flip_rate_by_source_emotion.csv` | Flip rate chuẩn hóa theo emotion của source. |
| `results/flip_rate_by_target_emotion.csv` | Flip rate chuẩn hóa theo emotion của target. |
| `results/flip_class_transitions.csv` | Ma trận các lớp trước/sau can thiệp và hướng correct/wrong. |
| `results/flip_target_susceptibility.csv` | Target nào nhạy, bao nhiêu source làm target đó flip. |
| `results/flip_robustness_zero_vs_mean.csv` | Giao/khác nhau của flip giữa zero và mean trên đúng cùng cặp. |
| `results/top_changed_cases.csv` | Các flip có JS divergence lớn nhất để kiểm tra định tính. |
| `results/experiment2_relationship_impact.csv` | Impact theo speaker × distance × emotion relation × modality, kèm conversation-bootstrap CI. |
| `results/experiment2_speaker_distance_harm.csv` | Harm theo speaker × bốn distance bins; gồm unconditional/conditional rate và harmful magnitude. |
| `results/experiment2_harm_sensitivity.csv` | Toàn bộ phân tích quan hệ tại các ngưỡng `1e-6`, `1e-4`, `1e-3`. |
| `results/experiment2_emotion_shift_contrasts.csv` | Contrast emotion-shift trừ same-emotion ở ba scope same-speaker. |
| `results/experiment2_previous_same_speaker_transition.csv` | Phân tích utterance gần nhất trước target trong chuỗi riêng của cùng speaker. |
| `results/experiment2_previous_same_speaker_emotion_pairs.csv` | Impact theo từng cặp source-emotion → target-emotion trực tiếp của cùng speaker. |
| `results/experiment2_within_target_matched_summary.csv` | Same-vs-cross comparison trên cùng target và cùng distance bin. |
| `results/experiment2_strategy_robustness.csv` | Contrast train-mean trừ zero theo đúng nhóm quan hệ. |
| `results/structural_harm_mixed_effects_fixed.csv` | Fixed effects, odds ratio và posterior interval của mixed-effects logistic. |
| `results/structural_harm_mixed_effects_random.csv` | Độ lệch chuẩn random intercept theo conversation và target. |
| `results/structural_harm_mixed_effects_robustness.csv` | Độ ổn định coefficient qua strategy và epsilon. |
| `results/experiment3_target_influence_profiles.csv` | Hồ sơ helpful/harmful/near-zero của từng target, tổng thể và theo modality. |
| `results/experiment3_unique_wrong_targets_corrected.csv` | Số baseline-wrong target duy nhất được ít nhất một intervention sửa đúng. |
| `results/experiment3_oracle_metrics.csv` | Utterance-level ACC/WF1 của singleton loss oracle, mỗi target đúng một lần. |
| `figures/*.png` | Hình signed-impact dùng cho báo cáo/paper. |
| `report.md` | Báo cáo tự động ngắn từ pipeline. |

## 14. Cách chạy

### Cấu hình

Điền đường dẫn trong `experiment_config.sh`, hoặc export biến môi trường trước khi chạy:

```bash
export CHECKPOINT="/absolute/path/to/best_checkpoint.pt"
export FEATURE_PATH="/absolute/path/to/iemocap_multimodal_features.pkl"
export RUN_DIR="$PWD/outputs/full_run"
export DEVICE="cuda"
export GPU_ID="0"
export NUM_THREADS="0"
export PROB_ATOL="2e-4"
export PROB_RTOL="2e-4"
```

`PROB_ATOL` và `PROB_RTOL` chỉ cho phép sai số số học nhỏ giữa inference theo batch gốc và batch size 1 trên CUDA. Không tăng tolerance để che một sai lệch dự đoán thực sự.

### Chạy từng task độc lập

```bash
bash run_task_a_baseline.sh
bash run_task_b_zero_replacement.sh
bash run_task_c_train_mean_replacement.sh
bash run_analysis.sh
```

- Task A: baseline frozen inference và kiểm tra equivalence.
- Task B: zero replacement cho mọi utterance nguồn/modalities.
- Task C: tính train mean và chạy mean replacement.
- Analysis: tạo bảng impact, biểu đồ, flip analysis và classification metrics.

Task B/C yêu cầu manifest của Task A và sẽ từ chối chạy nếu checkpoint, feature file, split hoặc seed không khớp. Mỗi hội thoại được lưu thành shard riêng nên có thể resume.

### Chỉ chạy phân tích prediction flip

```bash
bash run_prediction_change_analysis.sh
```

Hoặc:

```bash
python analyze_prediction_changes.py \
  --run-dir "$PWD/outputs/full_run"
```

### Tính lại classification metrics từ file có sẵn

Pair-weighted ablation metrics:

```bash
python compute_ablation_classification_metrics.py \
  --input /path/to/results/impact_pairs_train_mean.csv
```

Baseline test metrics chuẩn, mỗi target đúng một lần:

```bash
python compute_test_classification_metrics.py \
  --input /path/to/results/impact_pairs_train_mean.csv
```

### Structural Harm Experiment — Speaker, distance và emotion transition

Đây là implementation đầy đủ cho giả thuyết H2 “harm phụ thuộc cấu trúc hội thoại”. Cài toàn bộ dependency trước khi fit mixed-effects model:

```bash
pip install -r ../../requirements.txt
```

Chỉ chạy phần descriptive, bootstrap, within-target matching và heatmap:

```bash
bash run_experiment_2.sh
```

Mặc định:

- distance bins: `1–5`, `6–10`, `11–20`, `21+`;
- harmful: `signed_impact < -1e-4`;
- influential: `|signed_impact| > 1e-4`;
- sensitivity thresholds: `1e-6`, `1e-4`, `1e-3`;
- 2.000 lần cluster bootstrap theo conversation.

Có thể thay đổi cấu hình mà không sửa code:

```bash
bash run_experiment_2.sh \
  --impact-eps 1e-4 \
  --eps-values 1e-6 1e-4 1e-3 \
  --bootstrap-reps 5000
```

Mỗi nhóm báo cáo đồng thời:

```text
mean signed impact
mean absolute impact
harmful rate = N(I < -epsilon) / N(all pairs)
conditional harmful fraction = N(I < -epsilon) / N(|I| > epsilon)
influential rate = N(|I| > epsilon) / N(all pairs)
mean harmful magnitude = mean(|I| | I < -epsilon)
beneficial rate và conditional beneficial fraction
```

Các CI được tạo bằng cách resample toàn bộ conversation, không coi từng pair là IID. Heatmap nằm tại:

```text
figures/structural_harm/structural_harm_heatmap_<strategy>_<modality>.png
```

`experiment2_emotion_shift_contrasts.csv` so sánh `different emotion − same emotion` tại ba scope:

1. tất cả same-speaker pairs;
2. các same-speaker source đứng trước target;
3. utterance gần nhất trước target trong **chuỗi riêng của cùng speaker**.

Scope thứ ba không đồng nhất với `j=i-1`: lượt nói của speaker khác có thể nằm giữa source và target. Contrast harmful dương với CI 95% không chứa 0 là bằng chứng mô tả rằng emotion transition làm harm thường xuyên hơn.

`experiment2_within_target_matched_summary.csv` so sánh same-speaker với cross-speaker trên cùng target và cùng distance bin. Vì target được giữ cố định, margin, true emotion và baseline correctness cũng được giữ cố định.

Chạy toàn bộ workflow, bao gồm Bayesian mixed-effects logistic regression:

```bash
bash run_structural_harm_experiment.sh
```

Mixed-effects backend được hiện thực trực tiếp bằng PyTorch với mean-field variational Bayes; không cần `statsmodels` hoặc `pandas`. Fixed effects, random intercept theo conversation/target và standard deviation của hai random-effect distributions đều được học đồng thời. Posterior interval là xấp xỉ variational, vì vậy phải báo cáo cùng conversation-bootstrap CI từ phần descriptive thay vì xem nó như exact frequentist confidence interval.

Hoặc fit riêng adjusted model:

```bash
python fit_structural_harm_mixed_effects.py \
  --run-dir "$RUN_DIR" \
  --impact-eps-values 1e-6 1e-4 1e-3
```

Adjusted model gồm:

```text
same speaker
standardized log(1 + distance)
emotion difference
standardized target margin
modality
source direction
baseline target correctness
target emotion class
same speaker × emotion difference
modality × log distance
random intercept: conversation
random intercept: target nested by conversation
```

Model được fit riêng cho zero/train-mean và từng epsilon. File robustness chỉ đánh dấu một coefficient là robust khi giữ cùng hướng và interval loại 0 qua tất cả các fit. `--max-rows` chỉ dành cho smoke/debug; không sử dụng subsampling khi báo cáo kết quả paper.

Các tham số runtime có thể điều chỉnh nếu GPU/RAM hạn chế:

```bash
python fit_structural_harm_mixed_effects.py \
  --run-dir "$RUN_DIR" \
  --device cuda \
  --epochs 60 \
  --batch-size 8192 \
  --learning-rate 0.02
```

### Experiment 3 — Contextual profile và diagnostic oracle

```bash
bash run_experiment_3.sh
```

`harmful_rate` của target được tính đúng theo:

```text
count(signed_impact < -epsilon) / count(all allowed source-modality pairs)
```

Oracle so sánh baseline với mọi singleton source-modality intervention và chọn prediction có true-class loss nhỏ nhất. ACC/WF1 sau đó được tính **mỗi target đúng một lần**. Oracle sử dụng ground-truth loss nên chỉ là diagnostic upper bound trong tập lựa chọn hiện tại, không phải phương pháp có thể triển khai.

### Experiment 4 — Dự đoán harmful context

Experiment này cần hai run intervention tách biệt:

1. một run trên training split để tạo nhãn huấn luyện;
2. run test hiện tại để đánh giá trên các conversation chưa xuất hiện khi train.

Tạo training intervention run:

```bash
export SPLIT="train"
export RUN_DIR="$PWD/outputs/train_interventions"

bash run_task_a_baseline.sh
bash run_task_b_zero_replacement.sh
bash run_task_c_train_mean_replacement.sh
```

Training split lớn hơn test split nên bước sinh toàn bộ singleton interventions sẽ tốn đáng kể thời gian và dung lượng.

Huấn luyện và đánh giá predictor:

```bash
export TRAIN_RUN_DIR="$PWD/outputs/train_interventions"
export EVAL_RUN_DIR="$PWD/outputs/full_run"

bash run_experiment_4.sh --impact-eps 1e-4
```

Script so sánh:

- constant-prevalence baseline;
- margin-only;
- distance-only;
- speaker-only;
- relationship baseline gồm margin, distance, speaker và direction;
- full lightweight logistic predictor.

Ngưỡng phân loại được chọn trên training data rồi khóa trước khi đánh giá. Chỉ dùng các feature có thể biết tại inference như baseline confidence/margin/entropy, distance, speaker relation, direction, modality, strategy và norm của feature/replacement. `source_emotion`, `target_emotion`, test `signed_impact`, `loss_mask`, `pred_mask` và `correct_mask` tuyệt đối không đi vào input.

Kết quả nằm trong:

```text
results/experiment4_harm_predictor/harm_predictor_metrics.csv
results/experiment4_harm_predictor/harm_predictor_coefficients.csv
results/experiment4_harm_predictor/harm_predictor_metadata.json
```

Chỉ nên phát triển reliability gating nếu `full_lightweight` cải thiện rõ average precision/ROC-AUC so với margin-only, distance-only và speaker-only trên held-out conversations. Nếu oracle của Experiment 3 gần như không cải thiện ACC/WF1, nên ưu tiên groupwise intervention thay vì tiếp tục tối ưu singleton gating.

### Smoke test

```bash
export ALLOW_SMOKE_CHECKPOINT=1
export MAX_CONVERSATIONS=1
```

Không dùng kết quả từ smoke checkpoint làm kết quả nghiên cứu.

## 15. Kiểm thử code

```bash
python -m unittest \
  test_cross_utterance_impact.py \
  test_classification_metrics.py \
  test_prediction_change_analysis.py \
  test_extended_context_experiments.py \
  -v
```

Các kiểm tra quan trọng gồm: đúng dấu impact, không có self-pair, đúng số dòng, không sửa tensor gốc, zero/mean join theo đúng key, deduplicate target đúng cách và tính flip rate bằng count đã chuẩn hóa.

> Pipeline **Experiment 3 ETCR** (previous-same-speaker emotion transition,
> Phase A–B–C) nằm tại [`etcr/README.md`](etcr/README.md).
