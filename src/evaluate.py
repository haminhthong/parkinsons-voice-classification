"""Chia cross-validation và tính chỉ số ở mức bản ghi/bệnh nhân."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.data import SUBJECT_COLUMN, TARGET_COLUMN, build_subject_table


def make_subject_folds(
    frame: pd.DataFrame, *, n_splits: int = 5, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chia trên subject rồi ánh xạ subject train/validation về dòng dữ liệu."""
    subject_table = build_subject_table(frame).reset_index(drop=True)
    class_counts = subject_table[TARGET_COLUMN].value_counts()
    if class_counts.min() < n_splits:
        raise ValueError(
            f"Không thể tạo {n_splits} fold có đủ hai lớp; lớp ít nhất chỉ có "
            f"{int(class_counts.min())} bệnh nhân."
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for subject_fit, subject_valid in splitter.split(subject_table, subject_table[TARGET_COLUMN]):
        fit_ids = set(subject_table.iloc[subject_fit][SUBJECT_COLUMN])
        valid_ids = set(subject_table.iloc[subject_valid][SUBJECT_COLUMN])
        if not fit_ids.isdisjoint(valid_ids):
            raise AssertionError("Phát hiện group leakage trong cross-validation.")
        fit_index = np.flatnonzero(frame[SUBJECT_COLUMN].isin(fit_ids).to_numpy())
        valid_index = np.flatnonzero(frame[SUBJECT_COLUMN].isin(valid_ids).to_numpy())
        if frame.iloc[valid_index][TARGET_COLUMN].nunique() != 2:
            raise AssertionError("Validation fold phải có cả lớp 0 và lớp 1.")
        folds.append((fit_index, valid_index))
    return folds


def positive_score(estimator, features: pd.DataFrame) -> np.ndarray:
    """Trả về xác suất lớp 1; hỗ trợ decision score cho SVM khi benchmark."""
    if hasattr(estimator, "predict_proba"):
        class_index = int(np.flatnonzero(estimator.classes_ == 1)[0])
        return estimator.predict_proba(features)[:, class_index]
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(features), dtype=float)
    raise TypeError("Mô hình không cung cấp predict_proba hoặc decision_function.")


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
    }

