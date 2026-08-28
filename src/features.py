"""Khai báo đặc trưng và pipeline tiền xử lý."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import ORIGINAL_FEATURES

REDUNDANT_FEATURES = ["Jitter:DDP", "Shimmer:DDA"]
MODEL_FEATURES = [column for column in ORIGINAL_FEATURES if column not in REDUNDANT_FEATURES]


def make_pipeline(model: BaseEstimator, *, scale: bool = True) -> Pipeline:
    """Đặt chuẩn hóa và chọn đặc trưng trong pipeline của từng fold huấn luyện."""
    return Pipeline(
        [
            ("scale", StandardScaler() if scale else "passthrough"),
            ("select", SelectKBest(score_func=f_classif)),
            ("model", model),
        ]
    )
