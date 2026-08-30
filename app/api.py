"""Ứng dụng REST API FastAPI cho dịch vụ phân loại giọng nói Parkinson.

Cung cấp 2 endpoint chính:
- GET `/health`: Kiểm tra trạng thái hoạt động của dịch vụ API.
- POST `/predict`: Tiếp nhận tệp dữ liệu CSV giọng nói, thực hiện suy luận
  qua mô hình Calibrated Pipeline và trả về kết quả dự đoán chi tiết.
"""


from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.predict import load_bundle, predict_records

# Khởi tạo ứng dụng FastAPI với thông tin tiêu đề và mô tả
app = FastAPI(
    title="Parkinson Voice Research API",
    description="API phân loại giọng nói Parkinson chống rò rỉ dữ liệu (Research Only)",
    version="1.0.0",
)

ARTIFACT_PATH = Path("artifacts/parkinsons_calibrated_pipeline.joblib")


@app.get("/health")
def health() -> dict[str, str]:
    """Endpoint kiểm tra sức khỏe hệ thống API (Health Check).

    Returns:
        dict[str, str]: Trạng thái hoạt động dịch vụ và phạm vi sử dụng.
    """
    return {"status": "ok", "scope": "research-only"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Endpoint nhận tệp CSV dữ liệu giọng nói và trả về dự đoán phân loại.

    Args:
        file: Tệp tải lên dạng CSV chứa cột `name` và các đặc trưng giọng nói.

    Returns:
        dict: Kết quả dự đoán ở mức bản ghi và gộp theo bệnh nhân.

    Raises:
        HTTPException: 400 nếu tệp không phải CSV, 422 nếu lỗi schema hoặc xử lý.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận tệp định dạng CSV.")
        
    try:
        content = await file.read()
        frame = pd.read_csv(io.BytesIO(content))
        bundle = load_bundle(ARTIFACT_PATH)
        records, subjects = predict_records(frame, bundle)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Lỗi khi xử lý dữ liệu: {exc}") from exc
        
    return {
        "warning": "Chỉ phục vụ nghiên cứu, không dùng để chẩn đoán hoặc thay thế tư vấn y khoa.",
        "model": bundle["champion_name"],
        "probability_aggregation": bundle.get("probability_aggregation", "mean"),
        "decision_threshold": float(bundle["decision_threshold"]),
        "records": records.to_dict(orient="records"),
        "subjects": subjects.to_dict(orient="records"),
    }


