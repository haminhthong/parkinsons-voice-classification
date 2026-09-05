"""Module quản lý đặc trưng và tạo Pipeline học máy.

Khai báo các đặc trưng dư thừa (Redundant Features - có tương quan hoàn hảo với các chỉ số khác),
lọc danh sách đặc trưng đầu vào cho mô hình (`MODEL_FEATURES`) và đóng gói các bước tiền xử lý
(`StandardScaler`, `SelectKBest`) cùng mô hình dự đoán vào `sklearn.pipeline.Pipeline`.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import ORIGINAL_FEATURES

# Đặc trưng dẫn xuất dư thừa toán học (Jitter:DDP = 3*RAP, Shimmer:DDA = 3*APQ3)
# Được loại bỏ để tránh thổi phồng trọng số thông tin trùng lặp; đây là quan hệ đại số, không phải data leakage.
REDUNDANT_FEATURES = ["Jitter:DDP", "Shimmer:DDA"]


# Danh sách 20 đặc trưng số độc lập đưa vào huấn luyện mô hình
MODEL_FEATURES = [column for column in ORIGINAL_FEATURES if column not in REDUNDANT_FEATURES]


def compute_feature_percentiles(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    lower: float = 0.01,
    upper: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """Tính toán dải phân vị P1-P99 của các đặc trưng trên tập huấn luyện để kiểm tra OOD.

    Args:
        frame: DataFrame chứa các đặc trưng đầu vào.
        features: Danh sách tên cột đặc trưng cần tính (mặc định MODEL_FEATURES).
        lower: Ngưỡng phân vị dưới (mặc định 0.01 cho P1).
        upper: Ngưỡng phân vị trên (mặc định 0.99 cho P99).

    Returns:
        dict[str, tuple[float, float]]: Ánh xạ tên đặc trưng sang cặp (giá_trị_P_lower, giá_trị_P_upper).
    """
    if features is None:
        features = MODEL_FEATURES

    ranges: dict[str, tuple[float, float]] = {}
    for col in features:
        p_low = float(frame[col].quantile(lower))
        p_high = float(frame[col].quantile(upper))
        ranges[col] = (p_low, p_high)
    return ranges


def make_pipeline(model: BaseEstimator, *, scale: bool = True) -> Pipeline:
    """Tạo Pipeline tiền xử lý và mô hình học máy đóng gói hoàn chỉnh.

    Đặt `StandardScaler` và `SelectKBest` bên trong Pipeline để đảm bảo việc tính mean/std
    và chọn đặc trưng chỉ diễn ra trên tập train của từng fold, ngăn ngừa triệt để rò rỉ dữ liệu.

    Args:
        model: Mô hình phân loại scikit-learn (LogisticRegression, KNN, SVM, RF,...).
        scale: Nếu True, thực hiện chuẩn hóa Z-score (`StandardScaler`). Đặt False cho mô hình cây.

    Returns:
        Pipeline: Đối tượng sklearn Pipeline đã sẵn sàng để fit.
    """
    return Pipeline(
        [
            ("scale", StandardScaler() if scale else "passthrough"),
            ("select", SelectKBest(score_func=f_classif)),
            ("model", model),
        ]
    )
