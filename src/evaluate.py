"""Tạo cross-validation và tính chỉ số ở mức bản ghi lẫn bệnh nhân."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    brier_score_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.data import SUBJECT_COLUMN, TARGET_COLUMN, build_subject_table


def make_subject_folds(
    frame: pd.DataFrame, *, n_splits: int = 5, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chia trên bệnh nhân rồi ánh xạ các nhóm về từng dòng dữ liệu."""
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
        if not fit_ids.isdisjoint(valid_ids):
            raise AssertionError("Phát hiện rò rỉ nhóm trong cross-validation.")
        fit_index = np.flatnonzero(frame[SUBJECT_COLUMN].isin(fit_ids).to_numpy())
        valid_index = np.flatnonzero(frame[SUBJECT_COLUMN].isin(valid_ids).to_numpy())
        if frame.iloc[valid_index][TARGET_COLUMN].nunique() != 2:
            raise AssertionError("Fold validation phải có cả lớp 0 và lớp 1.")
        folds.append((fit_index, valid_index))
    return folds


def positive_score(estimator, features: pd.DataFrame) -> np.ndarray:
    """Trả về xác suất lớp 1 hoặc điểm quyết định của SVM khi benchmark."""
    if hasattr(estimator, "predict_proba"):
        class_index = int(np.flatnonzero(estimator.classes_ == 1)[0])
        return estimator.predict_proba(features)[:, class_index]
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(features), dtype=float)
    raise TypeError(
        "Mô hình không cung cấp phương thức predict_proba hoặc decision_function."
    )


def calculate_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Tính bộ chỉ số thống nhất; ROC-AUC yêu cầu cả hai lớp."""
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    if np.unique(y_true).size != 2:
        raise ValueError("Không thể đánh giá: y_true phải có cả lớp 0 và lớp 1.")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall/Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1-macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "Brier score": brier_score_loss(y_true, y_score),
    }


def aggregate_subject_predictions(
    frame: pd.DataFrame, probabilities: np.ndarray, *, threshold: float = 0.5
) -> pd.DataFrame:
    """Gộp xác suất bản ghi bằng trung bình trong từng bệnh nhân."""
    records = pd.DataFrame(
        {
            SUBJECT_COLUMN: frame[SUBJECT_COLUMN].to_numpy(),
            TARGET_COLUMN: frame[TARGET_COLUMN].to_numpy(),
            "probability": np.asarray(probabilities, dtype=float),
        }
    )
    subjects = records.groupby(SUBJECT_COLUMN, as_index=False).agg(
        status=(TARGET_COLUMN, "first"),
        probability=("probability", "mean"),
        recordings=("probability", "size"),
    )
    subjects["prediction"] = (subjects["probability"] >= threshold).astype(int)
    return subjects


def expected_calibration_error(y_true, probabilities, *, n_bins: int = 5) -> float:
    """Tính ECE với các khoảng xác suất có độ rộng bằng nhau trên [0, 1]."""
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
    """Bootstrap cluster ở mức bệnh nhân và loại mẫu chỉ có một lớp."""
    point = calculate_metrics(
        subjects["status"],
        subjects["prediction"],
        subjects["probability"],
    )
    rng = np.random.default_rng(random_state)
    samples: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
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
