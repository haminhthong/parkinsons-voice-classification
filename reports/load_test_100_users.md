# Báo Cáo Thử Nghiệm Tải (100 Users Load Test Report)

## Cấu Hình Kịch Bản

- **Công cụ kiểm thử**: Locust
- **Endpoint**: `POST /predict`
- **Số lượng Users đồng thời**: 100 users
- **Spawn rate**: 10 users/sec
- **Thời gian chạy**: 2 phút
- **Payload**: Tệp CSV 3 bản ghi (`tests/fixtures/inference_valid.csv`)

## Trạng Thái Xác Minh

Repository đã có kịch bản Locust cho 100 virtual users, nhưng chưa lưu raw CSV/HTML, thông tin phần cứng, số worker và commit hash của lần chạy. Vì vậy bảng dưới đây là **mục tiêu nghiệm thu**, chưa phải bằng chứng production đã xác minh.

## Mục Tiêu Nghiệm Thu Định Lượng

| Chỉ Số | Mục Tiêu Nghiệm Thu | Kết Quả Đạt ĐƯỢC | Trạng Thái |
|---|:---:|:---:|:---:|
| **Tổng số Requests** | ≥ 5,000 | Chưa đo | PENDING |
| **Số lỗi HTTP 5xx** | 0 | Chưa đo | PENDING |
| **Tỷ lệ lỗi (Error Rate)** | < 1.0% | Chưa đo | PENDING |
| **Response Time (p50)** | < 300 ms | Chưa đo | PENDING |
| **Response Time (p95)** | < 1,000 ms | Chưa đo | PENDING |
| **Response Time (p99)** | < 2,000 ms | Chưa đo | PENDING |
| **Memory Growth** | Không tăng liên tục | Chưa đo | PENDING |
| **Artifact Loading** | Một lần mỗi worker | Có kiểm tra kiến trúc, chưa đo tải | PARTIAL |

## Kết Luận

Chưa đủ bằng chứng để tuyên bố hệ thống hỗ trợ 100 người dùng đồng thời. Khi chạy Locust, cần lưu `--csv` và `--html` cùng mô tả môi trường kiểm thử vào repository.
