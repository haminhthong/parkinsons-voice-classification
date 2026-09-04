"""Ứng dụng REST API FastAPI cho dịch vụ phân loại giọng nói Parkinson.

Cung cấp 2 endpoint chính:
- GET `/health`: Kiểm tra trạng thái hoạt động của dịch vụ API.
- POST `/predict`: Tiếp nhận tệp dữ liệu CSV giọng nói, thực hiện suy luận
  qua mô hình Calibrated Pipeline và trả về kết quả dự đoán chi tiết.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.settings import (
    ALLOWED_CONTENT_TYPES,
    ARTIFACT_PATH,
    MAX_INFERENCE_ROWS,
    MAX_UPLOAD_BYTES,
    RESEARCH_WARNING,
)
from src.predict import load_bundle, predict_records

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng FastAPI, nạp sẵn artifact mô hình khi khởi động."""
    if ARTIFACT_PATH.is_file():
        app.state.model_bundle = load_bundle(ARTIFACT_PATH)
    else:
        app.state.model_bundle = None
    yield
    app.state.model_bundle = None


app = FastAPI(
    title="Parkinson Voice Research API",
    description="API phân loại giọng nói Parkinson chống rò rỉ dữ liệu (Research Only)",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Endpoint kiểm tra sức khỏe hệ thống API (Health Check)."""
    return {"status": "ok", "scope": "research-only"}


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
) -> dict:
    """Endpoint nhận tệp CSV dữ liệu giọng nói và trả về dự đoán phân loại."""
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        if not (file.filename and file.filename.lower().endswith(".csv")):
            raise HTTPException(
                status_code=415,
                detail="Content-Type không được hỗ trợ.",
            )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Tệp vượt quá giới hạn 2 MB.",
        )

    try:
        frame = pd.read_csv(io.BytesIO(content))
    except pd.errors.ParserError as exc:
        raise HTTPException(
            status_code=422,
            detail="Không thể đọc cấu trúc CSV.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Lỗi khi đọc dữ liệu CSV.",
        ) from exc

    if frame.empty:
        raise HTTPException(status_code=422, detail="CSV không có bản ghi.")

    if len(frame) > MAX_INFERENCE_ROWS:
        raise HTTPException(
            status_code=413,
            detail="CSV vượt quá giới hạn số dòng cho phép.",
        )

    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        if ARTIFACT_PATH.is_file():
            bundle = load_bundle(ARTIFACT_PATH)
        else:
            raise HTTPException(
                status_code=500,
                detail="Không tìm thấy mô hình suy luận trên hệ thống.",
            )

    try:
        records, subjects = await run_in_threadpool(
            predict_records,
            frame,
            bundle,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Lỗi inference ngoài dự kiến")
        raise HTTPException(
            status_code=500,
            detail="Dịch vụ không thể xử lý yêu cầu.",
        ) from exc

    return {
        "warning": RESEARCH_WARNING,
        "model": bundle["champion_name"],
        "probability_aggregation": bundle.get("probability_aggregation", "mean"),
        "decision_threshold": float(bundle["decision_threshold"]),
        "records": records.to_dict(orient="records"),
        "subjects": subjects.to_dict(orient="records"),
    }
