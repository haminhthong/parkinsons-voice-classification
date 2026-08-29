"""API tối giản cho cùng pipeline mà Streamlit sử dụng."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.predict import load_bundle, predict_records

app = FastAPI(title="Parkinson Voice Research API", version="1.0.0")
ARTIFACT_PATH = Path("artifacts/parkinsons_calibrated_pipeline.joblib")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "scope": "research-only"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận tệp CSV.")
    try:
        frame = pd.read_csv(io.BytesIO(await file.read()))
        bundle = load_bundle(ARTIFACT_PATH)
        records, subjects = predict_records(frame, bundle)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "warning": "Chỉ phục vụ nghiên cứu, không dùng để chẩn đoán.",
        "model": bundle["champion_name"],
        "probability_aggregation": bundle.get("probability_aggregation", "mean"),
        "decision_threshold": float(bundle["decision_threshold"]),
        "records": records.to_dict(orient="records"),
        "subjects": subjects.to_dict(orient="records"),
    }
