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


def check_recording_ood(
    row: pd.Series,
    feature_ranges: dict[str, tuple[float, float]] | None,
) -> list[str]:
    """Kiểm tra xem các giá trị đặc trưng trong bản ghi có nằm ngoài dải P1-P99 tập train không."""
    if not feature_ranges:
        return []

    warnings = []
    for col, (p_low, p_high) in feature_ranges.items():
        if col in row:
            val = float(row[col])
            if val < p_low or val > p_high:
                warnings.append(
                    f"FEATURE_OUTSIDE_TRAINING_RANGE: {col}={val:.4f} "
                    f"nằm ngoài dải P1-P99 [{p_low:.4f}, {p_high:.4f}]"
                )
    return warnings


def predict_records(frame: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thực hiện dự đoán bản ghi và gộp kết quả theo từng bệnh nhân kèm kiểm tra độ tin cậy."""
    # Đảm bảo cột nhãn status (nếu có) bị loại bỏ khỏi luồng suy luận
    inference_frame = frame.copy()
    if "status" in inference_frame.columns:
        inference_frame = inference_frame.drop(columns=["status"])

    validated = validate_dataframe(inference_frame, require_target=False, require_name=True)

    # Kiểm tra đảm bảo dữ liệu đầu vào chứa đủ 22 đặc trưng số của bộ UCI Parkinson's
    missing = sorted(set(ORIGINAL_FEATURES).difference(validated.columns))
    if missing:
        raise ValueError(f"Thiếu các cột đặc trưng bắt buộc: {missing}")

    feature_columns = bundle["feature_columns"]
    feature_ranges = bundle.get("feature_p1_p99")
    features = validated[feature_columns]
    probabilities = positive_class_probability(bundle["model"], features)
    threshold = float(bundle["decision_threshold"])

    # Xây dựng kết quả mức bản ghi
    records = validated[[ID_COLUMN, SUBJECT_COLUMN]].copy()
    records[ID_COLUMN] = records[ID_COLUMN].map(sanitize_csv_value)
    records[SUBJECT_COLUMN] = records[SUBJECT_COLUMN].map(sanitize_csv_value)
    records["probability_status_1"] = probabilities
    records["predicted_status"] = (probabilities >= threshold).astype(int)

    # Kiểm tra OOD cho từng bản ghi
    record_warnings: list[list[str]] = []
    for _, row in validated[feature_columns].iterrows():
        record_warnings.append(check_recording_ood(row, feature_ranges))
    records["warnings"] = record_warnings

    # Đọc quy tắc gộp xác suất từ artifact ('max', 'mean', 'median')
    aggregation = normalize_aggregation(bundle.get("probability_aggregation", "mean"))

    # Gom nhóm theo từng bệnh nhân
    subject_rows = []
    for subject_id, group in records.groupby(SUBJECT_COLUMN):
        n_rec = len(group)
        prob = float(group["probability_status_1"].agg(aggregation))
        flag = bool(prob >= threshold)
        pred_status = int(flag)
        pos_recs = int(group["predicted_status"].sum())

        sub_warnings = []
        # Chính sách số lượng bản ghi tối thiểu
        if n_rec < 3:
            sub_warnings.append(
                f"ONLY_ONE_RECORDING: Đối tượng chỉ có {n_rec} bản ghi âm; "
                "độ tin cậy gộp xác suất bị hạn chế (khuyến nghị >= 3 bản ghi)"
            )

        # Tổng hợp cảnh báo OOD từ các bản ghi con
        all_rec_warnings = [w for w_list in group["warnings"] for w in w_list]
        sub_warnings.extend(sorted(set(all_rec_warnings)))

        reliability = "limited" if len(sub_warnings) > 0 else "standard"

        subject_rows.append(
            {
                SUBJECT_COLUMN: subject_id,
                "n_recordings": n_rec,
                "probability_status_1": prob,
                "screening_score": prob,
                "predicted_status": pred_status,
                "screening_flag": flag,
                "positive_record_predictions": pos_recs,
                "reliability": reliability,
                "warnings": "; ".join(sub_warnings) if sub_warnings else "none",
            }
        )

    subjects = pd.DataFrame(subject_rows)
    return records, subjects


def predict_subject_records(
    subject_id: str,
    recordings: list[dict[str, float]],
    bundle: dict,
) -> dict:
    """Suy luận sàng lọc mức bệnh nhân từ danh sách các bản ghi đặc trưng dạng cấu trúc JSON.

    Args:
        subject_id: Định danh bệnh nhân / đối tượng.
        recordings: Danh sách các bản ghi âm (mỗi dict chứa 20 hoặc 22 đặc trưng âm học).
        bundle: Gói artifact mô hình đã tải.

    Returns:
        dict: Báo cáo sàng lọc đối tượng gồm điểm nguy cơ, cờ sàng lọc, cảnh báo và độ tin cậy.
    """
    if not recordings:
        raise ValueError("Danh sách recordings không được để trống.")

    # Xây dựng DataFrame tạm từ danh sách bản ghi
    frame = pd.DataFrame(recordings)
    # Loại bỏ nhãn nếu có truyền nhầm
    if "status" in frame.columns:
        frame = frame.drop(columns=["status"])

    frame[SUBJECT_COLUMN] = subject_id
    frame[ID_COLUMN] = [f"{subject_id}_{i+1}" for i in range(len(recordings))]

    # Bổ sung các đặc trưng dẫn xuất nếu thiếu
    for feat in ORIGINAL_FEATURES:
        if feat not in frame.columns:
            if feat == "Jitter:DDP" and "MDVP:RAP" in frame.columns:
                frame["Jitter:DDP"] = frame["MDVP:RAP"] * 3.0
            elif feat == "Shimmer:DDA" and "Shimmer:APQ3" in frame.columns:
                frame["Shimmer:DDA"] = frame["Shimmer:APQ3"] * 3.0
            else:
                raise ValueError(f"Thiếu đặc trưng bắt buộc: '{feat}'")

    records, subjects = predict_records(frame, bundle)
    sub = subjects.iloc[0]

    warnings_list = [w.strip() for w in str(sub["warnings"]).split("; ") if w and w != "none"]

    return {
        "subject_id": subject_id,
        "screening_score": float(sub["screening_score"]),
        "screening_flag": bool(sub["screening_flag"]),
        "reliability": str(sub["reliability"]),
        "warnings": warnings_list,
        "aggregation": bundle.get("probability_aggregation", "mean"),
        "decision_threshold": float(bundle["decision_threshold"]),
        "n_recordings": int(sub["n_recordings"]),
        "record_probabilities": records["probability_status_1"].tolist(),
    }
