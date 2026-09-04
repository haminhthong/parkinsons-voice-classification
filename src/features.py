"""Module quản lý đặc trưng và tạo Pipeline học máy.

Khai báo các đặc trưng dư thừa (Redundant Features - có tương quan hoàn hảo với các chỉ số khác),
lọc danh sách đặc trưng đầu vào cho mô hình (`MODEL_FEATURES`) và đóng gói các bước tiền xử lý
(`StandardScaler`, `SelectKBest`) cùng mô hình dự đoán vào `sklearn.pipeline.Pipeline`.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import ORIGINAL_FEATURES

# Đặc trưng dư thừa toán học (Jitter:DDP = 3*RAP, Shimmer:DDA = 3*APQ3) được loại bỏ ban đầu
REDUNDANT_FEATURES = ["Jitter:DDP", "Shimmer:DDA"]


# Danh sách 20 đặc trưng số độc lập đưa vào huấn luyện mô hình
MODEL_FEATURES = [column for column in ORIGINAL_FEATURES if column not in REDUNDANT_FEATURES]


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
