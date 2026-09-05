"""Module benchmark và lựa chọn mô hình ở cấp độ bệnh nhân (Subject-Level Model Selection & Nested CV).

Bao gồm các hàm:
- `evaluate_parameter_set`: Đánh giá 1 tập tham số qua các fold CV cấp bệnh nhân.
- `search_subject_level`: Quét lưới tham số và xếp hạng theo Subject F1-macro mean/std, Subject Balanced Accuracy mean, Subject ROC-AUC mean.
- `select_champion`: Tự động tìm mô hình Champion tốt nhất trong số danh sách ứng viên.
- `fit_complete_pipeline`: Huấn luyện mô hình Calibrated hoàn chỉnh kèm quy tắc gộp và ngưỡng quyết định OOF.
- `nested_subject_evaluation`: Đánh giá lồng (Nested Cross-Validation) toàn bộ pipeline lựa chọn mô hình.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import ParameterGrid

from src.data import TARGET_COLUMN, build_subject_table
from src.evaluate import (
    aggregate_subject_predictions,
    calculate_metrics,
    make_subject_folds,
    positive_score,
    select_decision_threshold,
)
from src.features import MODEL_FEATURES


@dataclass
class ChampionResult:
    """Kết quả lựa chọn mô hình Champion."""

    name: str
    estimator: Any
    parameters: dict[str, Any]
    metrics: dict[str, float]


def evaluate_parameter_set(
    estimator,
    parameters: dict[str, Any],
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_columns: list[str],
    *,
    aggregation: str = "mean",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Đánh giá một cấu hình tham số qua các fold CV ở cấp độ bệnh nhân."""
    fold_rows = []

    for fold_number, (fit_index, valid_index) in enumerate(folds, start=1):
        fit_frame = frame.iloc[fit_index]
        valid_frame = frame.iloc[valid_index]

        model = clone(estimator).set_params(**parameters)
        model.fit(
            fit_frame[feature_columns],
            fit_frame[TARGET_COLUMN],
        )

        probabilities = positive_score(
            model,
            valid_frame[feature_columns],
        )

        subjects = aggregate_subject_predictions(
            valid_frame,
            probabilities,
            aggregation=aggregation,
            threshold=threshold,
        )

        metrics = calculate_metrics(
            subjects["status"],
            subjects["prediction"],
            subjects["probability"],
        )

        fold_rows.append(
            {
                "Fold": fold_number,
                **metrics,
            }
        )

    return pd.DataFrame(fold_rows)


def search_subject_level(
    estimator,
    parameter_grid: dict[str, list[Any]],
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_columns: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    """Tìm kiếm siêu tham số tối ưu dựa trên metric gộp ở cấp bệnh nhân."""
    candidates = []

    grid = list(ParameterGrid(parameter_grid)) if parameter_grid else [{}]

    for parameters in grid:
        fold_table = evaluate_parameter_set(
            estimator,
            parameters,
            frame,
            folds,
            feature_columns,
        )

        candidates.append(
            {
                "Parameters": parameters,
                "Subject F1-macro mean": fold_table["F1-macro"].mean(),
                "Subject F1-macro std": fold_table["F1-macro"].std(ddof=0),
                "Subject Balanced Accuracy mean": (fold_table["Balanced Accuracy"].mean()),
                "Subject ROC-AUC mean": fold_table["ROC-AUC"].mean(),
            }
        )

    result = pd.DataFrame(candidates).sort_values(
        [
            "Subject F1-macro mean",
            "Subject Balanced Accuracy mean",
            "Subject ROC-AUC mean",
        ],
        ascending=False,
    )

    return result.iloc[0], result


def _model_complexity_rank(name: str) -> int:
    """Xếp hạng độ phức tạp tương đối của kiến trúc mô hình (giá trị nhỏ hơn là đơn giản hơn)."""
    complexity = {
        "Dummy": 0,
        "Logistic Regression": 1,
        "KNN": 2,
        "Random Forest": 3,
        "HistGradientBoosting": 4,
    }
    return complexity.get(name, 10)


def select_champion(
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_specs: dict[str, tuple[Any, dict[str, list[Any]]]],
    *,
    feature_columns: list[str] | None = None,
    f1_tolerance: float = 0.005,
) -> ChampionResult:
    """Lựa chọn mô hình Champion với cơ chế bảo vệ độ ổn định (Stability Guardrail).

    Quy tắc xếp hạng:
    1. Làm tròn Subject F1-macro mean theo ngưỡng sai khác nhỏ (tolerance) để nhận diện các mô hình hòa điểm.
    2. Khi các mô hình có F1-macro xấp xỉ nhau, ưu tiên phương sai thấp nhất (-F1 std) nhằm đảm bảo tính ổn định.
    3. Ưu tiên Balanced Accuracy mean.
    4. Ưu tiên ROC-AUC mean.
    5. Ưu tiên kiến trúc đơn giản hơn (simpler model) để tránh overfitting trên tập dữ liệu nhỏ (32 bệnh nhân).
    """
    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    champion_candidates = []

    for name, (estimator, grid) in model_specs.items():
        # Bỏ qua các mô hình không hỗ trợ xác suất dự đoán hoặc ghi rõ không triển khai
        if name == "SVM (RBF, decision score)":
            continue

        best_candidate, _ = search_subject_level(
            estimator,
            grid,
            frame,
            folds,
            feature_columns,
        )

        best_params = best_candidate["Parameters"]
        best_model = clone(estimator).set_params(**best_params)

        champion_candidates.append(
            ChampionResult(
                name=name,
                estimator=best_model,
                parameters=best_params,
                metrics={
                    "Subject F1-macro mean": float(best_candidate["Subject F1-macro mean"]),
                    "Subject F1-macro std": float(best_candidate["Subject F1-macro std"]),
                    "Subject Balanced Accuracy mean": float(
                        best_candidate["Subject Balanced Accuracy mean"]
                    ),
                    "Subject ROC-AUC mean": float(best_candidate["Subject ROC-AUC mean"]),
                },
            )
        )

    def _sort_key(c: ChampionResult) -> tuple:
        # Nhóm F1 theo bước tolerance để phát hiện tie
        f1_bucket = round(c.metrics["Subject F1-macro mean"] / f1_tolerance) * f1_tolerance
        stability = -round(c.metrics["Subject F1-macro std"], 4)
        bal_acc = round(c.metrics["Subject Balanced Accuracy mean"], 4)
        roc_auc = round(c.metrics["Subject ROC-AUC mean"], 4)
        simplicity = -_model_complexity_rank(c.name)
        return (f1_bucket, stability, bal_acc, roc_auc, simplicity)

    champion_candidates.sort(key=_sort_key, reverse=True)

    return champion_candidates[0]


def fit_selection_rule(
    train_frame: pd.DataFrame,
    oof_probabilities: np.ndarray,
    *,
    aggregation_candidates: tuple[str, ...] = ("mean", "median", "max"),
    minimum_specificity: float = 0.5,
) -> tuple[str, float]:
    """Tìm quy tắc gộp xác suất và ngưỡng quyết định tối ưu dựa trên OOF train."""
    candidates = []

    for aggregation in aggregation_candidates:
        subjects = aggregate_subject_predictions(
            train_frame,
            oof_probabilities,
            aggregation=aggregation,
        )
        threshold, _ = select_decision_threshold(
            subjects,
            minimum_specificity=minimum_specificity,
        )
        subjects["prediction"] = (subjects["probability"] >= threshold).astype(int)
        metrics = calculate_metrics(
            subjects["status"],
            subjects["prediction"],
            subjects["probability"],
        )
        candidates.append(
            {
                "Aggregation": aggregation,
                "Selected threshold": threshold,
                **metrics,
            }
        )

    comparison = pd.DataFrame(candidates).sort_values(
        ["Balanced Accuracy", "F1-macro", "Specificity"],
        ascending=[False, False, False],
    )
    best = comparison.iloc[0]
    return str(best["Aggregation"]), float(best["Selected threshold"])


def fit_complete_pipeline(
    champion: ChampionResult,
    train_frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[CalibratedClassifierCV, str, float]:
    """Huấn luyện mô hình Calibrated Sigmoid lồng nhóm bệnh nhân và khóa quy tắc quyết định."""
    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    subject_table = build_subject_table(train_frame)
    min_class_count = int(subject_table[TARGET_COLUMN].value_counts().min())
    outer_n_splits = max(2, min(n_splits, min_class_count))

    folds = make_subject_folds(
        train_frame,
        n_splits=outer_n_splits,
        random_state=random_state,
    )

    oof_probabilities = np.full(len(train_frame), np.nan)

    for fit_index, valid_index in folds:
        fit_frame = train_frame.iloc[fit_index].reset_index(drop=True)
        valid_frame = train_frame.iloc[valid_index]

        fit_subjects = build_subject_table(fit_frame)
        fit_min_class = int(fit_subjects[TARGET_COLUMN].value_counts().min())
        inner_n_splits = max(2, min(3, fit_min_class))

        inner_folds = make_subject_folds(
            fit_frame,
            n_splits=inner_n_splits,
            random_state=random_state + 17,
        )

        calibrated = CalibratedClassifierCV(
            estimator=clone(champion.estimator),
            method="sigmoid",
            cv=inner_folds,
        )
        calibrated.fit(fit_frame[feature_columns], fit_frame[TARGET_COLUMN])
        oof_probabilities[valid_index] = positive_score(
            calibrated,
            valid_frame[feature_columns],
        )

    aggregation, threshold = fit_selection_rule(
        train_frame,
        oof_probabilities,
    )

    final_folds = make_subject_folds(
        train_frame,
        n_splits=outer_n_splits,
        random_state=random_state + 42,
    )
    final_calibrated = CalibratedClassifierCV(
        estimator=clone(champion.estimator),
        method="sigmoid",
        cv=final_folds,
    )
    final_calibrated.fit(train_frame[feature_columns], train_frame[TARGET_COLUMN])

    return final_calibrated, aggregation, threshold


def nested_subject_evaluation(
    frame: pd.DataFrame,
    model_specs: dict[str, tuple[Any, dict[str, list[Any]]]],
    *,
    feature_columns: list[str] | None = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Đánh giá lồng (Nested Cross-Validation) toàn bộ quy trình lựa chọn và hiệu chỉnh mô hình."""
    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    subject_table = build_subject_table(frame)
    min_class_count = int(subject_table[TARGET_COLUMN].value_counts().min())
    effective_outer_splits = max(2, min(outer_splits, min_class_count))

    outer_folds = make_subject_folds(
        frame,
        n_splits=effective_outer_splits,
        random_state=random_state,
    )

    rows = []

    for outer_fold, (outer_fit, outer_valid) in enumerate(
        outer_folds,
        start=1,
    ):
        outer_train = frame.iloc[outer_fit].reset_index(drop=True)
        outer_validation = frame.iloc[outer_valid].reset_index(drop=True)

        outer_train_subjects = build_subject_table(outer_train)
        outer_train_min_class = int(outer_train_subjects[TARGET_COLUMN].value_counts().min())
        effective_inner_splits = max(2, min(inner_splits, outer_train_min_class))

        inner_folds = make_subject_folds(
            outer_train,
            n_splits=effective_inner_splits,
            random_state=random_state + outer_fold,
        )


        champion = select_champion(
            outer_train,
            inner_folds,
            model_specs,
            feature_columns=feature_columns,
        )

        calibrated_model, aggregation, threshold = fit_complete_pipeline(
            champion,
            outer_train,
            feature_columns=feature_columns,
            n_splits=inner_splits,
            random_state=random_state + outer_fold,
        )

        probabilities = positive_score(
            calibrated_model,
            outer_validation[feature_columns],
        )

        subjects = aggregate_subject_predictions(
            outer_validation,
            probabilities,
            aggregation=aggregation,
            threshold=threshold,
        )

        metrics = calculate_metrics(
            subjects["status"],
            subjects["prediction"],
            subjects["probability"],
        )

        rows.append(
            {
                "Outer fold": outer_fold,
                "Champion": champion.name,
                "Aggregation": aggregation,
                "Threshold": threshold,
                **metrics,
            }
        )

    return pd.DataFrame(rows)
