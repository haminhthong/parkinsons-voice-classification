"""Module đánh giá mô hình phân loại và tính toán chỉ số y tế.

Cung cấp các hàm tạo phân chia K-Fold theo bệnh nhân, tính toán chỉ số hiệu năng
(Sensitivity/Recall, Specificity, Balanced Accuracy, F1-Macro, ROC-AUC, Brier score, ECE),
gộp dự đoán theo bệnh nhân, tìm ngưỡng tối ưu trên Out-Of-Fold (OOF) và tính khoảng tin cậy
95% CI bằng phương pháp Patient Cluster Bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.data import SUBJECT_COLUMN, TARGET_COLUMN, build_subject_table
from src.utils import normalize_aggregation, positive_class_probability


def make_subject_folds(
    frame: pd.DataFrame, *, n_splits: int = 5, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tạo danh sách các fold Cross-Validation chia phân tầng ở cấp độ bệnh nhân.

    Phân chia danh sách bệnh nhân duy nhất thành `n_splits` fold, sau đó ánh xạ chỉ số
    trở lại toàn bộ các dòng bản ghi tương ứng của bệnh nhân đó. Đảm bảo mọi validation fold
    đều chứa đủ cả 2 lớp (nhãn 0 và 1) và không bị rò rỉ bệnh nhân giữa fit/validation.

    Args:
        frame: DataFrame chứa toàn bộ bản ghi dữ liệu.
        n_splits: Số lượng fold Cross-Validation (mặc định 5).
        random_state: Seed ngẫu nhiên để tái lập cách chia fold.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: Danh sách các cặp (fit_indices, valid_indices).

    Raises:
        ValueError: Nếu số bệnh nhân ở lớp ít nhất không đủ để tạo `n_splits` fold.
        AssertionError: Nếu phát hiện rò rỉ bệnh nhân hoặc validation fold thiếu 1 lớp.
    """
    subject_table = build_subject_table(frame).reset_index(drop=True)
    class_counts = subject_table[TARGET_COLUMN].value_counts()

    if class_counts.min() < n_splits:
        raise ValueError(
            f"Không thể tạo {n_splits} fold có đủ hai lớp; lớp ít nhất chỉ có "
            f"{int(class_counts.min())} bệnh nhân."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for subject_fit, subject_valid in splitter.split(
        subject_table,
        subject_table[TARGET_COLUMN],
    ):
        fit_ids = set(subject_table.iloc[subject_fit][SUBJECT_COLUMN])
        valid_ids = set(subject_table.iloc[subject_valid][SUBJECT_COLUMN])

        # Kiểm tra bảo vệ không rò rỉ nhóm bệnh nhân
        if not fit_ids.isdisjoint(valid_ids):
            raise AssertionError("Phát hiện rò rỉ nhóm bệnh nhân trong cross-validation.")

        fit_index = np.flatnonzero(frame[SUBJECT_COLUMN].isin(fit_ids).to_numpy())
        valid_index = np.flatnonzero(frame[SUBJECT_COLUMN].isin(valid_ids).to_numpy())

        # Đảm bảo fold đánh giá luôn có cả 2 lớp 0 và 1
        if frame.iloc[valid_index][TARGET_COLUMN].nunique() != 2:
            raise AssertionError("Fold validation bắt buộc phải có cả lớp 0 và lớp 1.")

        folds.append((fit_index, valid_index))

    return folds


def positive_score(estimator, features: pd.DataFrame) -> np.ndarray:
    """Trích xuất xác suất lớp dương (status=1) hoặc điểm quyết định (decision_function).

    Args:
        estimator: Mô hình đã được huấn luyện (scikit-learn Pipeline hoặc Classifier).
        features: Các đặc trưng đầu vào dưới dạng DataFrame.

    Returns:
        np.ndarray: Mảng 1 chiều chứa xác suất hoặc điểm quyết định cho lớp dương.

    Raises:
        TypeError: Nếu mô hình không hỗ trợ `predict_proba` lẫn `decision_function`.
    """

    if hasattr(estimator, "predict_proba"):
        return positive_class_probability(estimator, features)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(features), dtype=float)
    raise TypeError("Mô hình không cung cấp phương thức predict_proba hoặc decision_function.")


def calculate_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Tính toán bộ chỉ số đánh giá hiệu năng thống nhất cho bài toán phân loại y tế.

    Bao gồm các chỉ số: Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity (Độ nhạy),
    Specificity (Độ đặc hiệu), F1-macro, ROC-AUC và Brier Score.

    Args:
        y_true: Nhãn thực tế (0 hoặc 1).
        y_pred: Nhãn dự đoán nhị phân (0 hoặc 1).
        y_score: Xác suất dự đoán liên tục hoặc điểm decision score.

    Returns:
        dict[str, float]: Từ điển chứa tên các chỉ số và giá trị đo lường tương ứng.

    Raises:
        ValueError: Nếu y_true không chứa đủ 2 lớp nhãn {0, 1}.
    """
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    if np.unique(y_true).size != 2:
        raise ValueError("Không thể đánh giá chỉ số: y_true bắt buộc phải có đủ hai lớp 0 và 1.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall/Sensitivity": recall,
        "Specificity": specificity,
        "NPV": npv,
        "F1-macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score),
    }

    if np.all((y_score >= 0.0) & (y_score <= 1.0)):
        metrics["Brier score"] = brier_score_loss(y_true, y_score)
    else:
        metrics["Brier score"] = np.nan

    return metrics


def calculate_clinical_likelihood_ratios(
    sensitivity: float,
    specificity: float,
) -> dict[str, float | None]:
    """Tính toán tỷ số khả dĩ dương và âm (Likelihood Ratios: LR+, LR-) cho phân tích khám phá.

    LR+ = Sensitivity / (1 - Specificity)
    LR- = (1 - Sensitivity) / Specificity

    Lưu ý: Chỉ dùng cho mục đích báo cáo khám phá trong nghiên cứu sàng lọc,
    không dùng để suy diễn chẩn đoán lâm sàng độc lập.

    Args:
        sensitivity: Độ nhạy (Recall/Sensitivity) trong khoảng [0, 1].
        specificity: Độ đặc hiệu (Specificity) trong khoảng [0, 1].

    Returns:
        dict[str, float | None]: Tỷ số khả dĩ LR+ và LR- (hoặc None nếu chia cho 0).
    """
    lr_pos = (sensitivity / (1.0 - specificity)) if specificity < 1.0 else None
    lr_neg = ((1.0 - sensitivity) / specificity) if specificity > 0.0 else None
    return {"LR+": lr_pos, "LR-": lr_neg}



def aggregate_subject_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Gộp xác suất các bản ghi của cùng một bệnh nhân thành dự đoán mức bệnh nhân.

    Hỗ trợ 3 quy tắc gộp xác suất: 'mean' (trung bình), 'median' (trung vị), hoặc 'max' (lớn nhất).


    Args:
        frame: DataFrame chứa cột `subject_id` và `status`.
        probabilities: Mảng xác suất dự đoán của từng bản ghi âm.
        threshold: Ngưỡng quyết định nhị phân (mặc định 0.5).
        aggregation: Quy tắc gộp xác suất ('mean', 'median', 'max').

    Returns:
        pd.DataFrame: Bảng kết quả tổng hợp theo bệnh nhân.

    Raises:
        ValueError: Nếu tên quy tắc gộp không nằm trong danh sách hỗ trợ.
    """
    aggregation = normalize_aggregation(aggregation)

    records = pd.DataFrame(
        {
            SUBJECT_COLUMN: frame[SUBJECT_COLUMN].to_numpy(),
            TARGET_COLUMN: frame[TARGET_COLUMN].to_numpy(),
            "probability": np.asarray(probabilities, dtype=float),
        }
    )

    subjects = records.groupby(SUBJECT_COLUMN, as_index=False).agg(
        status=(TARGET_COLUMN, "first"),
        probability=("probability", aggregation),
        recordings=("probability", "size"),
    )
    subjects["prediction"] = (subjects["probability"] >= threshold).astype(int)
    return subjects


def evaluate_subject_fold(
    estimator,
    validation_frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    aggregation: str = "mean",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Đánh giá một validation fold ở cấp độ bệnh nhân."""
    subjects = aggregate_subject_predictions(
        validation_frame,
        probabilities,
        aggregation=aggregation,
        threshold=threshold,
    )

    return calculate_metrics(
        subjects["status"],
        subjects["prediction"],
        subjects["probability"],
    )


def select_decision_threshold(
    subjects: pd.DataFrame,
    *,
    minimum_specificity: float = 0.5,
) -> tuple[float, pd.DataFrame]:
    """Tìm ngưỡng quyết định tối ưu trên tập OOF của tập huấn luyện.

    Tối ưu Balanced Accuracy với ràng buộc Specificity >= minimum_specificity,
    giúp giảm thiểu rủi ro báo động giả (False Alarm) trong chẩn đoán.

    Args:
        subjects: DataFrame đã gộp dự đoán ở cấp độ bệnh nhân.
        minimum_specificity: Mức độ đặc hiệu tối thiểu bắt buộc đạt được.

    Returns:
        tuple[float, pd.DataFrame]: Ngưỡng tối ưu và bảng tìm kiếm ứng viên.
    """

    probabilities = subjects["probability"].to_numpy(dtype=float)
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 181),
                probabilities,
            ]
        )
    )
    rows = []
    for threshold in candidates:
        prediction = (probabilities >= threshold).astype(int)
        metrics = calculate_metrics(subjects["status"], prediction, probabilities)
        rows.append({"Threshold": float(threshold), **metrics})

    table = pd.DataFrame(rows)
    # Lọc danh sách ứng viên thỏa mãn chỉ tiêu Specificity tối thiểu
    eligible = table[table["Specificity"] >= minimum_specificity]
    if eligible.empty:
        eligible = table

    # Xếp hạng ứng viên theo thứ tự ưu tiên: Balanced Accuracy -> F1-macro -> Specificity
    ranked = eligible.assign(distance_from_default=(eligible["Threshold"] - 0.5).abs()).sort_values(
        ["Balanced Accuracy", "F1-macro", "Specificity", "distance_from_default"],
        ascending=[False, False, False, True],
    )
    return float(ranked.iloc[0]["Threshold"]), table


def expected_calibration_error(y_true, probabilities, *, n_bins: int = 5) -> float:
    """Tính sai số hiệu chỉnh kỳ vọng (Expected Calibration Error - ECE).

    ECE đo lường khoảng cách giữa xác suất dự đoán của mô hình và tần suất thực tế của lớp dương.
    Giá trị ECE càng gần 0 thể hiện xác suất xuất ra càng có độ tin cậy thực tế cao.

    Lưu ý quan trọng về cỡ mẫu:
    - Trên tập kiểm thử nhỏ (ví dụ 8 bệnh nhân chia 5 bin), ECE chỉ mang tính mô tả (descriptive only)
      vì mỗi bin có quá ít hoặc không có quan sát.
    - Đánh giá chất lượng calibration thực chất nên dựa chủ yếu vào OOF train hoặc nested CV folds.

    Args:
        y_true: Nhãn thực tế (0 hoặc 1).
        probabilities: Xác suất dự đoán từ mô hình [0, 1].
        n_bins: Số lượng khoảng bin chia xác suất (mặc định 5 bin).

    Returns:
        float: Giá trị chỉ số ECE.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(probabilities, edges[1:-1]), n_bins - 1)
    error = 0.0
    for bin_index in range(n_bins):
        mask = bins == bin_index
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(error)


def bootstrap_subject_confidence_intervals(
    subjects: pd.DataFrame, *, n_bootstrap: int = 2000, random_state: int = 42
) -> pd.DataFrame:
    """Ước lượng Khoảng Tin Cậy 95% (95% CI) bằng Patient Cluster Bootstrap.

    Thực hiện lấy mẫu có hoàn lại 2,000 lần ở cấp độ bệnh nhân. loại các mẫu
    bootstrap không hợp lệ (mẫu chỉ chứa duy nhất 1 lớp nhãn).

    Args:
        subjects: DataFrame kết quả dự đoán của từng bệnh nhân.
        n_bootstrap: Số lần lấy mẫu ngẫu nhiên (mặc định 2,000 lần).
        random_state: Seed ngẫu nhiên.

    Returns:
        pd.DataFrame: Bảng điểm ước lượng điểm (Point estimate) và khoảng tin cậy 95% CI.
    """
    point = calculate_metrics(
        subjects["status"],
        subjects["prediction"],
        subjects["probability"],
    )
    rng = np.random.default_rng(random_state)
    samples: list[dict[str, float]] = []

    for _ in range(n_bootstrap):
        # Lấy mẫu có hoàn lại theo dòng bệnh nhân
        sampled = subjects.iloc[rng.integers(0, len(subjects), size=len(subjects))]
        if sampled["status"].nunique() != 2:
            continue
        samples.append(
            calculate_metrics(
                sampled["status"],
                sampled["prediction"],
                sampled["probability"],
            )
        )

    distribution = pd.DataFrame(samples)
    return pd.DataFrame(
        [
            {
                "Metric": metric,
                "Point estimate": value,
                "CI 2.5%": distribution[metric].quantile(0.025),
                "CI 97.5%": distribution[metric].quantile(0.975),
                "Valid bootstrap samples": len(distribution),
            }
            for metric, value in point.items()
        ]
    )
