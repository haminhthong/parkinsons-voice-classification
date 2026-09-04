"""Module suy luận dự đoán từ tệp dữ liệu CSV đầu vào.

Nạp gói artifact mô hình đã hiệu chỉnh (`parkinsons_calibrated_pipeline.joblib`), kiểm tra schema
tệp CSV đầu vào, tính toán xác suất dự đoán cho từng bản ghi âm và tự động gộp kết quả
ở cấp độ bệnh nhân theo quy tắc đã được khóa từ OOF Train.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import ID_COLUMN, ORIGINAL_FEATURES, SUBJECT_COLUMN, validate_dataframe
from src.utils import normalize_aggregation, positive_class_probability

DANGEROUS_CSV_PREFIXES = ("=", "+", "-", "@")


def sanitize_csv_value(value: str) -> str:
    """Chống tấn công CSV Formula Injection cho chuỗi xuất ra."""
    val_str = str(value)
    if val_str.startswith(DANGEROUS_CSV_PREFIXES):
        return "'" + val_str
    return val_str


def load_bundle(path: str | Path) -> dict:
    """Nạp gói artifact mô hình và kiểm tra các trường siêu dữ liệu bắt buộc."""
    filepath = Path(path)
    if not filepath.is_file():
        raise RuntimeError("Không tìm thấy model artifact.")

    bundle = joblib.load(filepath)
    required = {"model", "feature_columns", "decision_threshold", "champion_name"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Artifact thiếu siêu dữ liệu bắt buộc: {sorted(missing)}")
    return bundle


def predict_records(frame: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thực hiện dự đoán bản ghi và gộp kết quả theo từng bệnh nhân."""
    validated = validate_dataframe(frame, require_target=False, require_name=True)

    # Kiểm tra đảm bảo dữ liệu đầu vào chứa đủ 22 đặc trưng số của bộ UCI Parkinson's
    missing = sorted(set(ORIGINAL_FEATURES).difference(validated.columns))
    if missing:
        raise ValueError(f"Thiếu các cột đặc trưng bắt buộc: {missing}")

    features = validated[bundle["feature_columns"]]
    probabilities = positive_class_probability(bundle["model"], features)
    threshold = float(bundle["decision_threshold"])

    # Xây dựng kết quả mức bản ghi
    records = validated[[ID_COLUMN, SUBJECT_COLUMN]].copy()
    records[ID_COLUMN] = records[ID_COLUMN].map(sanitize_csv_value)
    records[SUBJECT_COLUMN] = records[SUBJECT_COLUMN].map(sanitize_csv_value)
    records["probability_status_1"] = probabilities
    records["predicted_status"] = (probabilities >= threshold).astype(int)

    # Đọc quy tắc gộp xác suất từ artifact ('max', 'mean', 'median')
    aggregation = normalize_aggregation(bundle.get("probability_aggregation", "mean"))

    # Gom nhóm theo từng bệnh nhân
    subjects = records.groupby(SUBJECT_COLUMN, as_index=False).agg(
        n_recordings=(ID_COLUMN, "size"),
        probability_status_1=(
            "probability_status_1",
            aggregation,
        ),
        positive_record_predictions=("predicted_status", "sum"),
    )
    subjects["predicted_status"] = (subjects["probability_status_1"] >= threshold).astype(int)

    return records, subjects
