# Model card: Parkinson Voice Research Classifier

## Intended use

Mô hình minh họa quy trình phân loại giọng nói có kiểm soát rò rỉ theo bệnh nhân. Đối tượng sử dụng là người học và nhà tuyển dụng muốn xem một pipeline ML có thể tái lập.

Mô hình không được thiết kế để chẩn đoán, sàng lọc, điều trị hoặc hỗ trợ quyết định y tế.

## Dữ liệu

- Nguồn: UCI Parkinsons, 195 bản ghi từ 32 bệnh nhân.
- Nhãn: 24 bệnh nhân lớp 1 và 8 bệnh nhân lớp 0.
- Mọi split đều được thực hiện theo `subject_id`.
- Không có external cohort trong repository.

## Mô hình và xác suất

Champion deployable được chọn bằng F1-macro CV trên train trong các mô hình hỗ trợ xác suất. Mô hình được calibration bằng sigmoid; cả outer OOF evaluation và inner calibration folds đều không trùng bệnh nhân. Cách gộp xác suất và threshold cũng được chọn hoàn toàn từ OOF train, không dùng holdout.

Metric tái tạo được nằm trong `artifacts/metrics.json`; confidence interval nằm trong `artifacts/holdout_bootstrap_ci.csv`.

Artifact hiện tại dùng xác suất lớn nhất (`max`) trong các bản ghi của bệnh nhân và threshold `0,835`. Trên holdout, Recall là `1,00`, Specificity `0,50` và Balanced Accuracy `0,75`. Các con số này chỉ dựa trên 8 bệnh nhân test; không nên tiếp tục điều chỉnh quy tắc dựa trên holdout rồi vẫn xem nó là kiểm tra độc lập.

## Giới hạn và rủi ro

- Cỡ mẫu rất nhỏ, đặc biệt chỉ có 8 bệnh nhân lớp 0.
- Holdout chỉ có 8 bệnh nhân; confidence interval rộng và nhạy với cách chia.
- Không có validation trên bệnh viện, thiết bị ghi âm, ngôn ngữ hoặc dân số khác.
- Không có metadata nhân khẩu học để đánh giá fairness theo tuổi, giới hoặc dân tộc.
- Calibration trên tập nhỏ không bảo đảm xác suất phản ánh nguy cơ lâm sàng.
- Feature importance chỉ mang tính mô tả, không phải giải thích nhân quả.

## Quality gates

Schema, nhãn, group leakage, fold đủ lớp, preprocessing pipeline, artifact reload, xác suất, calibration và bootstrap đều có test tự động. CI chạy pytest trên Python 3.11.
