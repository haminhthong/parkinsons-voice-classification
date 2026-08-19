"""Nạp artifact và dự đoán CSV ở mức bản ghi lẫn bệnh nhân."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import ID_COLUMN, ORIGINAL_FEATURES, SUBJECT_COLUMN, validate_dataframe


def load_bundle(path: str | Path) -> dict:
    bundle = joblib.load(path)
    required = {"model", "feature_columns", "decision_threshold", "champion_name"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Artifact thiếu metadata: {sorted(missing)}")
    return bundle


def _probability_status_1(model, features: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("Mô hình triển khai phải hỗ trợ predict_proba.")
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(features)[:, positive_index]
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Mô hình trả về xác suất ngoài [0, 1].")
    return probabilities


def predict_records(frame: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dự đoán từng bản ghi và gộp trung bình xác suất theo bệnh nhân."""
    validated = validate_dataframe(frame, require_target=False, require_name=True)
    # Inference vẫn yêu cầu đủ 22 đặc trưng gốc để kiểm soát đúng schema nguồn.
    missing = sorted(set(ORIGINAL_FEATURES).difference(validated.columns))
    if missing:
        raise ValueError(f"Thiếu cột đặc trưng: {missing}")
    features = validated[bundle["feature_columns"]]
    probabilities = _probability_status_1(bundle["model"], features)
    threshold = float(bundle["decision_threshold"])
    records = validated[[ID_COLUMN, SUBJECT_COLUMN]].copy()
    records["probability_status_1"] = probabilities
    records["predicted_status"] = (probabilities >= threshold).astype(int)
    subjects = records.groupby(SUBJECT_COLUMN, as_index=False).agg(
        n_recordings=(ID_COLUMN, "size"),
        probability_status_1=("probability_status_1", "mean"),
        positive_record_predictions=("predicted_status", "sum"),
    )
    subjects["predicted_status"] = (
        subjects["probability_status_1"] >= threshold
    ).astype(int)
    return records, subjects

