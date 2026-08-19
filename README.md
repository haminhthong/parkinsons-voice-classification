# Leakage-Aware Parkinson’s Voice Classification

Dự án phân loại Parkinson từ đặc trưng giọng nói, tập trung vào một câu hỏi quan trọng hơn Accuracy: **phép đánh giá có thực sự đo khả năng dự đoán cho bệnh nhân chưa từng gặp hay không?**

> Chỉ phục vụ nghiên cứu và học tập. Mô hình không phải thiết bị y tế, không dùng để chẩn đoán hoặc thay thế đánh giá của chuyên gia.

## Câu chuyện của dự án

Phép chia ngẫu nhiên theo từng bản ghi ban đầu đạt Accuracy **97,44%**, nhưng audit cho thấy **27/27 bệnh nhân trong test cũng xuất hiện trong train** thông qua những bản ghi giọng nói khác. Vì mỗi người có nhiều lần ghi âm, mô hình có thể học dấu hiệu riêng của người đã gặp thay vì khái quát sang bệnh nhân mới.

Repository này xây dựng lại quy trình theo đúng đơn vị độc lập:

1. Tạo bảng một dòng cho mỗi `subject_id`.
2. Chia holdout và cross-validation trên bảng bệnh nhân, có stratify theo nhãn.
3. Ánh xạ bệnh nhân train/validation trở lại toàn bộ bản ghi tương ứng.
4. Đặt `StandardScaler` và `SelectKBest` trong scikit-learn `Pipeline`.
5. Chọn mô hình bằng CV trên train; chỉ mở holdout test sau khi lựa chọn xong.
6. Báo cáo cả mức bản ghi và mức bệnh nhân, kèm độ ổn định và giới hạn dữ liệu.

Kết quả benchmark hiện tại cho thấy điểm F1-macro CV dao động đáng kể giữa fold. Đây không phải điểm yếu cần che giấu: bộ dữ liệu chỉ có 32 bệnh nhân, trong đó 8 người thuộc lớp 0. Độ bất định này đáng tin cậy hơn một Accuracy rất cao sinh ra từ phép chia rò rỉ.

## Những lỗi đã sửa

| Vấn đề | Cách xử lý |
|---|---|
| Chia ngẫu nhiên 195 bản ghi | Chia trên 32 `subject_id`, sau đó ánh xạ về bản ghi |
| Validation fold có riêng lớp 1 | `StratifiedKFold` trên bảng subject và `assert y_valid.nunique() == 2` |
| ROC–AUC benchmark thành NaN | Mọi fold bắt buộc có hai lớp; benchmark dừng nếu không thể tạo fold hợp lệ |
| Repeated CV có fold mất lớp | Dùng cùng cơ chế subject-level split cho từng repeat |
| Scaler học trước khi chia | Scaler là một bước trong pipeline và chỉ fit trên train fold |
| `SVC(probability=True)` calibration theo dòng | Dùng `probability=False` và `decision_function` cho ROC–AUC benchmark |
| Chọn mô hình theo test | Champion được chọn bằng mean CV F1-macro trên train |

## Dữ liệu và schema

Bộ [UCI Parkinsons](https://archive.ics.uci.edu/dataset/174/parkinsons) có 195 bản ghi, 32 bệnh nhân, 22 đặc trưng số và nhãn `status` (`0/1`). `subject_id` được suy ra bằng cách bỏ hậu tố lần ghi khỏi `name`, ví dụ `phon_R01_S01_1 → phon_R01_S01`.

Inference yêu cầu CSV có cột `name` và đủ 22 đặc trưng gốc. Người dùng không phải nhập thủ công từng giá trị. Hai đặc trưng phụ thuộc tuyến tính (`Jitter:DDP`, `Shimmer:DDA`) được kiểm tra ở schema nhưng loại khỏi tập đặc trưng mô hình.

## Demo

Luồng xử lý:

```text
Upload CSV → kiểm tra schema → pipeline tiền xử lý → dự đoán từng bản ghi
           → trung bình xác suất theo bệnh nhân → bảng + biểu đồ + tải CSV
```

Giao diện Streamlit hiển thị:

- số bản ghi của từng bệnh nhân;
- xác suất `status = 1`;
- kết quả từng bản ghi và kết quả tổng hợp;
- biểu đồ xác suất và biểu đồ đặc trưng được chọn;
- nút tải kết quả dự đoán;
- cảnh báo giới hạn sử dụng nghiên cứu.

## Cài đặt và chạy

Yêu cầu Python 3.11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --data data/parkinsons.csv --artifacts artifacts
streamlit run app/streamlit_app.py
```

API tùy chọn:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
curl -X POST -F "file=@data/parkinsons.csv" http://localhost:8000/predict
```

Docker:

```bash
docker build -t parkinsons-voice-demo .
docker run --rm -p 8501:8501 parkinsons-voice-demo
```

## Kiểm thử

```bash
pytest -q
```

Test bảo vệ các điều kiện quan trọng:

- đúng 22 đặc trưng gốc và nhãn chỉ gồm `0/1`;
- một bệnh nhân không xuất hiện đồng thời trong train/test hoặc fit/validation;
- mỗi validation fold có cả hai lớp;
- scaler chưa được fit bên ngoài pipeline;
- CSV inference thiếu cột bị từ chối;
- artifact load lại cho cùng dự đoán;
- mọi xác suất nằm trong `[0, 1]`.

## Cấu trúc repository

```text
.
├── app/                 # Streamlit và FastAPI dùng chung pipeline dự đoán
├── artifacts/           # Pipeline đã huấn luyện và bảng benchmark
├── data/                # CSV cùng mô tả nguồn/schema
├── notebooks/           # Notebook thí nghiệm đã làm sạch
├── reports/figures/     # Hình xuất từ phân tích
├── src/                 # Dữ liệu, đặc trưng, train, evaluate, predict
├── tests/               # Test schema, leakage và artifact
├── Dockerfile
├── requirements.txt
└── README.md
```

## Diễn giải kết quả đúng cách

Accuracy không đủ cho dữ liệu lệch lớp. Benchmark báo cáo F1-macro, Balanced Accuracy và ROC–AUC; phân tích cuối cần xem thêm Recall/Sensitivity và Specificity. Xác suất bệnh nhân là trung bình xác suất của các bản ghi thuộc cùng `subject_id`, với ngưỡng cố định 0,5.

Bootstrap phải resample theo bệnh nhân, không theo bản ghi. Khoảng tin cậy có thể rộng vì holdout chỉ có hai bệnh nhân lớp 0. Kết quả không chứng minh quan hệ nhân quả giữa đặc trưng giọng nói và bệnh, cũng không thay thế validation trên cohort độc lập.

## Tái lập và hướng phát triển

Seed mặc định là `42`; phiên bản thư viện được khóa trong `requirements.txt`; artifact lưu cả pipeline lẫn metadata đặc trưng/ngưỡng. Hướng phát triển phù hợp là kiểm định trên cohort độc lập, calibration xác suất theo group-aware folds, model card và giám sát drift. Không nên tối ưu thêm trên holdout hiện tại rồi tiếp tục gọi đó là test độc lập.

