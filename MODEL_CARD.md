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

Champion deployable được chọn bằng F1-macro CV trên train trong các mô hình hỗ trợ xác suất. Mô hình được calibration bằng sigmoid; cả outer OOF evaluation và inner calibration folds đều không trùng bệnh nhân. Xác suất bệnh nhân là trung bình xác suất các bản ghi, với ngưỡng 0,5 cố định trước khi xem test.

Metric tái tạo được nằm trong `artifacts/metrics.json`; confidence interval nằm trong `artifacts/holdout_bootstrap_ci.csv`.

Ở artifact hiện tại, threshold 0,5 cho Recall 1,00 nhưng Specificity 0,00 trên holdout. ROC-AUC 1,00 chỉ phản ánh thứ hạng trên 8 bệnh nhân test và không bù được lỗi phân loại lớp âm tại ngưỡng triển khai. Không nên điều chỉnh threshold trên chính holdout này rồi tiếp tục xem nó là test độc lập.

## Giới hạn và rủi ro

- Cỡ mẫu rất nhỏ, đặc biệt chỉ có 8 bệnh nhân lớp 0.
- Holdout chỉ có 8 bệnh nhân; confidence interval rộng và nhạy với cách chia.
- Không có validation trên bệnh viện, thiết bị ghi âm, ngôn ngữ hoặc dân số khác.
- Không có metadata nhân khẩu học để đánh giá fairness theo tuổi, giới hoặc dân tộc.
- Calibration trên tập nhỏ không bảo đảm xác suất phản ánh nguy cơ lâm sàng.
- Feature importance chỉ mang tính mô tả, không phải giải thích nhân quả.

## Quality gates

Schema, nhãn, group leakage, fold đủ lớp, preprocessing pipeline, artifact reload, xác suất, calibration và bootstrap đều có test tự động. CI chạy pytest trên Python 3.11.
