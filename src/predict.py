"""Module suy luận dự đoán từ tệp dữ liệu CSV đầu vào.

Nạp gói artifact mô hình đã hiệu chỉnh (`parkinsons_calibrated_pipeline.joblib`), kiểm tra schema
tệp CSV đầu vào, tính toán xác suất dự đoán cho từng bản ghi âm và tự động gộp kết quả
ở cấp độ bệnh nhân theo quy tắc đã được khóa từ OOF Train.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import ID_COLUMN, ORIGINAL_FEATURES, SUBJECT_COLUMN, validate_dataframe


def load_bundle(path: str | Path) -> dict:
    """Nạp gói artifact mô hình và kiểm tra các trường siêu dữ liệu bắt buộc.

    Args:
        path: Đường dẫn tệp `.joblib` chứa artifact.

    Returns:
        dict: Từ điển chứa mô hình `model`, siêu dữ liệu và cấu hình huấn luyện.

    Raises:
        ValueError: Nếu tệp artifact bị thiếu bất kỳ siêu dữ liệu bắt buộc nào.
    """

    bundle = joblib.load(path)
    required = {"model", "feature_columns", "decision_threshold", "champion_name"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Artifact thiếu siêu dữ liệu bắt buộc: {sorted(missing)}")
    return bundle


def _probability_status_1(model, features: pd.DataFrame) -> np.ndarray:
    """Trích xuất xác suất dự đoán lớp dương (status=1) từ mô hình triển khai.

    Args:
        model: Pipeline mô hình đã hiệu chỉnh xác suất (`CalibratedClassifierCV`).
        features: DataFrame chứa các đặc trưng số đã qua kiểm tra.

    Returns:
        np.ndarray: Mảng xác suất dự đoán nằm trong khoảng [0, 1].

    Raises:
        TypeError: Nếu mô hình không hỗ trợ `predict_proba`.
        ValueError: Nếu xác suất nằm ngoài khoảng hợp lệ [0, 1].
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError("Mô hình triển khai bắt buộc phải hỗ trợ phương thức predict_proba.")
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(features)[:, positive_index]
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Mô hình trả về xác suất ngoài phạm vi hợp lệ [0, 1].")
    return probabilities


def predict_records(frame: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thực hiện dự đoán bản ghi và gộp kết quả theo từng bệnh nhân.

    Args:
        frame: DataFrame đầu vào chứa cột `name` và các đặc trưng giọng nói.
        bundle: Từ điển artifact nạp từ `load_bundle`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - `records`: Kết quả chi tiết từng bản ghi âm.
            - `subjects`: Kết quả tổng hợp cấp độ bệnh nhân.

    Raises:
        ValueError: Nếu DataFrame đầu vào thiếu đặc trưng hoặc quy tắc gộp sai.
    """

    validated = validate_dataframe(frame, require_target=False, require_name=True)
    
    # Kiểm tra đảm bảo dữ liệu đầu vào chứa đủ 22 đặc trưng số của bộ UCI Parkinson's
    missing = sorted(set(ORIGINAL_FEATURES).difference(validated.columns))
    if missing:
        raise ValueError(f"Thiếu các cột đặc trưng bắt buộc: {missing}")

    features = validated[bundle["feature_columns"]]
    probabilities = _probability_status_1(bundle["model"], features)
    threshold = float(bundle["decision_threshold"])

    # Xây dựng kết quả mức bản ghi
    records = validated[[ID_COLUMN, SUBJECT_COLUMN]].copy()
    records["probability_status_1"] = probabilities
    records["predicted_status"] = (probabilities >= threshold).astype(int)

    # Đọc quy tắc gộp xác suất từ artifact ('max', 'mean', 'median')
    aggregation = bundle.get("probability_aggregation", "mean")
    if aggregation == "mean_by_subject":
        aggregation = "mean"
        
    aggregation_functions = {"mean": "mean", "median": "median", "max": "max"}
    if aggregation not in aggregation_functions:
        raise ValueError(f"Artifact chứa cách gộp xác suất không hợp lệ: {aggregation!r}")

    # Gom nhóm theo từng bệnh nhân
    subjects = records.groupby(SUBJECT_COLUMN, as_index=False).agg(
        n_recordings=(ID_COLUMN, "size"),
        probability_status_1=(
            "probability_status_1",
            aggregation_functions[aggregation],
        ),
        positive_record_predictions=("predicted_status", "sum"),
    )
    subjects["predicted_status"] = (subjects["probability_status_1"] >= threshold).astype(int)

    return records, subjects

