"""Benchmark mô hình không rò rỉ và lưu pipeline dùng cho triển khai."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data import ORIGINAL_FEATURES, TARGET_COLUMN, load_data, subject_holdout_split
from src.evaluate import (
    aggregate_subject_predictions, bootstrap_subject_confidence_intervals,
    calculate_metrics, expected_calibration_error, make_subject_folds, positive_score,
)
from src.features import MODEL_FEATURES, REDUNDANT_FEATURES, make_pipeline
from src.utils import sha256_file

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.json"
DEFAULT_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RANDOM_STATE = int(DEFAULT_CONFIG["random_state"])
SCORING = {
    "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro", "roc_auc": "roc_auc",
}


def _calibrated_estimator(estimator, frame: pd.DataFrame) -> CalibratedClassifierCV:
    """Hiệu chỉnh sigmoid bằng các fold không trùng bệnh nhân."""
    folds = make_subject_folds(
        frame, n_splits=int(DEFAULT_CONFIG["calibration_splits"]),
        random_state=RANDOM_STATE + 17,
    )
    return CalibratedClassifierCV(
        estimator=clone(estimator),
        method="sigmoid",
        cv=folds,
    )


def _group_oof_calibration(
    estimator,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Tạo xác suất OOF với các fold ngoài và trong đều theo bệnh nhân."""
    outer_folds = make_subject_folds(
        frame, n_splits=int(DEFAULT_CONFIG["calibration_splits"]),
        random_state=RANDOM_STATE + 29,
    )
    oof = np.full(len(frame), np.nan)
    for fit_index, valid_index in outer_folds:
        fit_frame = frame.iloc[fit_index].reset_index(drop=True)
        model = _calibrated_estimator(estimator, fit_frame)
        model.fit(fit_frame[MODEL_FEATURES], fit_frame[TARGET_COLUMN])
        validation_features = frame.iloc[valid_index][MODEL_FEATURES]
        oof[valid_index] = positive_score(model, validation_features)
    if np.isnan(oof).any():
        raise AssertionError("Hiệu chỉnh OOF chưa tạo xác suất cho mọi bản ghi.")
    threshold = float(DEFAULT_CONFIG["decision_threshold"])
    subjects = aggregate_subject_predictions(frame, oof, threshold=threshold)
    metrics = calculate_metrics(
        subjects["status"],
        subjects["prediction"],
        subjects["probability"],
    )
    metrics["ECE (5 bins)"] = expected_calibration_error(
        subjects["status"],
        subjects["probability"],
    )
    return subjects, metrics


def model_specs() -> dict:
    return {
        "Dummy": (Pipeline([("model", DummyClassifier(strategy="prior"))]), {}),
        "Logistic Regression": (
            make_pipeline(
                LogisticRegression(
                    max_iter=3000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                )
            ),
            {
                "select__k": [10, 15, "all"],
                "model__C": [0.1, 1, 10],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "KNN": (
            make_pipeline(KNeighborsClassifier()),
            {
                "select__k": [10, 15, "all"],
                "model__n_neighbors": [3, 5, 7, 9],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        # Không bật probability=True vì SVC sẽ hiệu chỉnh nội bộ mà không biết nhóm.
        "SVM (RBF, decision score)": (
            make_pipeline(SVC(probability=False, random_state=RANDOM_STATE)),
            {
                "select__k": [10, 15, "all"],
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", 0.1],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                scale=False,
            ),
            {
                "select__k": [15, "all"],
                "model__max_depth": [None, 5],
                "model__max_features": ["sqrt", 0.7],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "HistGradientBoosting": (
            make_pipeline(
                HistGradientBoostingClassifier(
                    max_iter=150,
                    random_state=RANDOM_STATE,
                ),
                scale=False,
            ),
            {
                "select__k": [10, 15, "all"],
                "model__learning_rate": [0.03, 0.1],
                "model__max_leaf_nodes": [7, 15],
                "model__l2_regularization": [0, 1],
            },
        ),
    }


def train(data_path: str | Path, artifact_dir: str | Path = "artifacts") -> pd.DataFrame:
    """Huấn luyện benchmark, hiệu chỉnh champion và ghi các artifact kết quả."""
    frame = load_data(data_path)
    train_frame, test_frame = subject_holdout_split(
        frame, test_size=float(DEFAULT_CONFIG["test_size"]), random_state=RANDOM_STATE
    )
    folds = make_subject_folds(
        train_frame, n_splits=int(DEFAULT_CONFIG["benchmark_splits"]),
        random_state=RANDOM_STATE,
    )
    rows, searches = [], {}
    X_train, y_train = train_frame[MODEL_FEATURES], train_frame[TARGET_COLUMN]
    for name, (estimator, grid) in model_specs().items():
        search = GridSearchCV(
            estimator,
            grid,
            scoring=SCORING,
            refit="f1_macro",
            cv=folds,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        result, index = search.cv_results_, search.best_index_
        rows.append({
            "Model": name,
            "CV F1-macro mean": result["mean_test_f1_macro"][index],
            "CV F1-macro std": result["std_test_f1_macro"][index],
            "CV Balanced Accuracy": result["mean_test_balanced_accuracy"][index],
            "CV ROC-AUC": result["mean_test_roc_auc"][index],
            "Best parameters": json.dumps(search.best_params_, ensure_ascii=False),
        })
        searches[name] = search
    benchmark = pd.DataFrame(rows).sort_values(
        ["CV F1-macro mean", "CV ROC-AUC"],
        ascending=False,
    )

    # Chỉ chọn champion triển khai trong các mô hình cung cấp xác suất.
    deployable = benchmark[benchmark["Model"] != "SVM (RBF, decision score)"]
    champion_name = str(deployable.iloc[0]["Model"])
    champion = searches[champion_name].best_estimator_
    oof_subjects, calibration_metrics = _group_oof_calibration(
        champion,
        train_frame,
    )
    calibrated_model = _calibrated_estimator(champion, train_frame)
    calibrated_model.fit(X_train, y_train)
    test_probability = positive_score(calibrated_model, test_frame[MODEL_FEATURES])
    threshold = float(DEFAULT_CONFIG["decision_threshold"])
    test_subjects = aggregate_subject_predictions(
        test_frame, test_probability, threshold=threshold
    )
    test_metrics = calculate_metrics(
        test_subjects["status"],
        test_subjects["prediction"],
        test_subjects["probability"],
    )
    test_metrics["ECE (5 bins)"] = expected_calibration_error(
        test_subjects["status"],
        test_subjects["probability"],
    )
    confidence_intervals = bootstrap_subject_confidence_intervals(test_subjects)

    bundle = {
        "model": calibrated_model,
        "champion_name": f"{champion_name} + sigmoid calibration",
        "feature_columns": MODEL_FEATURES,
        "original_feature_columns": ORIGINAL_FEATURES,
        "dropped_redundant_features": REDUNDANT_FEATURES,
        "decision_threshold": threshold,
        "random_state": RANDOM_STATE,
        "holdout_subjects": sorted(test_frame["subject_id"].unique()),
        "probability_aggregation": "mean_by_subject",
        "calibration": "sigmoid with subject-level folds",
        "oof_calibration_metrics": calibration_metrics,
        "holdout_subject_metrics": test_metrics,
        "training_config": DEFAULT_CONFIG,
        "data_sha256": sha256_file(data_path),
    }
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "parkinsons_calibrated_pipeline.joblib")
    benchmark.to_csv(output / "model_benchmark.csv", index=False)
    oof_subjects.to_csv(output / "oof_subject_predictions.csv", index=False)
    test_subjects.to_csv(output / "holdout_subject_predictions.csv", index=False)
    confidence_intervals.to_csv(output / "holdout_bootstrap_ci.csv", index=False)
    metrics = {
        "champion": bundle["champion_name"],
        "oof_calibration": calibration_metrics,
        "holdout_subject": test_metrics,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/parkinsons.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    print(train(args.data, args.artifacts).to_string(index=False))
