"""Nạp dữ liệu, kiểm tra schema và chia dữ liệu theo bệnh nhân."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ID_COLUMN = "name"
TARGET_COLUMN = "status"
SUBJECT_COLUMN = "subject_id"

ORIGINAL_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE",
]


def subject_id_from_name(name: str) -> str:
    """Lấy mã bệnh nhân bằng cách bỏ số thứ tự bản ghi ở cuối tên."""
    value = str(name).strip()
    if not re.match(r"^.+_[^_]+$", value):
        raise ValueError(f"Tên bản ghi không đúng định dạng: {name!r}")
    return value.rsplit("_", 1)[0]


def validate_dataframe(
    frame: pd.DataFrame,
    *,
    require_target: bool = True,
    require_name: bool = True,
) -> pd.DataFrame:
    """Kiểm tra và trả về bản sao có thứ tự cột ổn định."""
    required = set(ORIGINAL_FEATURES)
    if require_target:
        required.add(TARGET_COLUMN)
    if require_name:
        required.add(ID_COLUMN)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    result = frame.copy()
    numeric_columns = ORIGINAL_FEATURES + ([TARGET_COLUMN] if require_target else [])
    for column in numeric_columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cột {column!r} phải chứa dữ liệu số.") from exc
    if result[numeric_columns].isna().any().any():
        raise ValueError("Dữ liệu có giá trị thiếu.")
    if not np.isfinite(result[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Dữ liệu có giá trị không hữu hạn (NaN hoặc ±inf).")
    if require_target and not set(result[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("status chỉ được gồm hai nhãn 0 và 1.")
    if require_name:
        if result[ID_COLUMN].isna().any():
            raise ValueError("Cột name không được để trống.")
        result[SUBJECT_COLUMN] = result[ID_COLUMN].map(subject_id_from_name)
        if require_target:
            label_counts = result.groupby(SUBJECT_COLUMN)[TARGET_COLUMN].nunique()
            if not label_counts.eq(1).all():
                invalid = label_counts[label_counts.ne(1)].index.tolist()
                raise ValueError(f"Nhãn không nhất quán trong bệnh nhân: {invalid}")
    return result


def load_data(path: str | Path) -> pd.DataFrame:
    """Đọc và kiểm tra tập huấn luyện từ CSV."""
    return validate_dataframe(pd.read_csv(path), require_target=True, require_name=True)


def build_subject_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng một dòng cho mỗi bệnh nhân để dùng khi chia fold."""
    return frame.groupby(SUBJECT_COLUMN, as_index=False).agg(
        status=(TARGET_COLUMN, "first"), recordings=(ID_COLUMN, "size")
    )


def subject_holdout_split(
    frame: pd.DataFrame, *, test_size: float = 0.25, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia holdout trên bảng bệnh nhân rồi ánh xạ về các bản ghi."""
    subjects = build_subject_table(frame)
    train_subjects, test_subjects = train_test_split(
        subjects, test_size=test_size, random_state=random_state,
        stratify=subjects[TARGET_COLUMN],
    )
    train_ids = set(train_subjects[SUBJECT_COLUMN])
    test_ids = set(test_subjects[SUBJECT_COLUMN])
    if not train_ids.isdisjoint(test_ids):
        raise AssertionError("Phát hiện bệnh nhân xuất hiện ở cả train và test.")
    return (
        frame[frame[SUBJECT_COLUMN].isin(train_ids)].copy(),
        frame[frame[SUBJECT_COLUMN].isin(test_ids)].copy(),
    )

