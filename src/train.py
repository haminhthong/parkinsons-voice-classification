"""Module huấn luyện mô hình, hiệu chỉnh xác suất OOF và đóng gói artifact.

Thực hiện toàn bộ quy trình:
1. Chia Holdout theo bệnh nhân (`subject_holdout_split`).
2. Benchmark các mô hình ứng viên ở cấp bệnh nhân (`search_subject_level`).
3. Lựa chọn mô hình Champion dựa trên Subject F1-macro CV trung bình.
4. Đánh giá lồng Nested Cross-Validation (`nested_subject_evaluation`).
5. Hiệu chỉnh xác suất Sigmoid lồng nhóm bệnh nhân.
6. Chọn quy tắc gộp xác suất và ngưỡng quyết định tối ưu từ OOF Train.
7. Đóng gói mô hình calibrated pipeline cùng siêu dữ liệu môi trường.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data import ORIGINAL_FEATURES, TARGET_COLUMN, load_data, subject_holdout_split
from src.evaluate import (
    aggregate_subject_predictions,
    bootstrap_subject_confidence_intervals,
    calculate_metrics,
    expected_calibration_error,
    make_subject_folds,
    positive_score,
    select_decision_threshold,
)
from src.features import (
    MODEL_FEATURES,
    REDUNDANT_FEATURES,
    compute_feature_percentiles,
    make_pipeline,
)
from src.model_selection import (
    nested_subject_evaluation,
    search_subject_level,
)
from src.utils import sha256_file

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.json"
DEFAULT_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RANDOM_STATE = int(DEFAULT_CONFIG["random_state"])


def _calibrated_estimator(estimator, frame: pd.DataFrame) -> CalibratedClassifierCV:
    """Tạo bộ hiệu chỉnh Sigmoid Calibration đóng gói các fold phân chia theo bệnh nhân."""
    folds = make_subject_folds(
        frame,
        n_splits=int(DEFAULT_CONFIG["calibration_splits"]),
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
) -> np.ndarray:
    """Sinh mảng xác suất Out-Of-Fold (OOF) được hiệu chỉnh Sigmoid theo nhóm bệnh nhân."""
    outer_folds = make_subject_folds(
        frame,
        n_splits=int(DEFAULT_CONFIG["calibration_splits"]),
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
        raise AssertionError("Hiệu chỉnh OOF chưa tạo xác suất cho mọi bản ghi train.")

    return oof


def _select_patient_rule(
    frame: pd.DataFrame,
    oof_probabilities: np.ndarray,
) -> tuple[str, float, pd.DataFrame, pd.DataFrame]:
    """Tìm quy tắc gộp xác suất và ngưỡng quyết định tối ưu từ OOF Train."""
    candidates = []
    threshold_tables = []
    selected_subjects: dict[str, pd.DataFrame] = {}

    for aggregation in DEFAULT_CONFIG["aggregation_candidates"]:
        subjects = aggregate_subject_predictions(
            frame,
            oof_probabilities,
            aggregation=aggregation,
        )
        threshold, threshold_table = select_decision_threshold(
            subjects,
            minimum_specificity=float(DEFAULT_CONFIG["minimum_specificity"]),
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
        threshold_table.insert(0, "Aggregation", aggregation)
        threshold_tables.append(threshold_table)
        selected_subjects[aggregation] = subjects

    comparison = pd.DataFrame(candidates).sort_values(
        ["Balanced Accuracy", "F1-macro", "Specificity", "Brier score"],
        ascending=[False, False, False, True],
    )
    best = comparison.iloc[0]
    aggregation = str(best["Aggregation"])
    threshold = float(best["Selected threshold"])

    return (
        aggregation,
        threshold,
        selected_subjects[aggregation],
        pd.concat(threshold_tables, ignore_index=True),
        comparison,
    )


def _feature_selection_stability(
    estimator,
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Thống kê tần suất lựa chọn của từng đặc trưng qua các fold Cross-Validation."""
    counts = pd.Series(0, index=MODEL_FEATURES, dtype=int)
    for fit_index, _ in folds:
        fold_model = clone(estimator)
        fit_frame = frame.iloc[fit_index]
        fold_model.fit(fit_frame[MODEL_FEATURES], fit_frame[TARGET_COLUMN])
        selector = fold_model.named_steps.get("select")
        selected = np.asarray(MODEL_FEATURES)[selector.get_support()]
        counts.loc[selected] += 1

    return pd.DataFrame(
        {
            "Feature": counts.index,
            "Selected folds": counts.values,
            "Selection frequency": counts.values / len(folds),
        }
    ).sort_values(["Selected folds", "Feature"], ascending=[False, True])


def model_specs() -> dict:
    """Định nghĩa cấu hình danh sách các mô hình ứng viên và lưới tham số."""
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
    """Thực hiện quy trình huấn luyện toàn diện, benchmark, hiệu chỉnh và xuất artifact kết quả."""
    frame = load_data(data_path)

    # 1. Chia Holdout độc lập theo từng bệnh nhân
    train_frame, test_frame = subject_holdout_split(
        frame,
        test_size=float(DEFAULT_CONFIG["test_size"]),
        random_state=RANDOM_STATE,
    )

    # 2. Tạo danh sách các fold Cross-Validation theo bệnh nhân cho tập Train
    folds = make_subject_folds(
        train_frame,
        n_splits=int(DEFAULT_CONFIG["benchmark_splits"]),
        random_state=RANDOM_STATE,
    )

    rows, searches = [], {}

    # 3. Benchmark tất cả mô hình ở cấp độ bệnh nhân
    specs = model_specs()
    for name, (estimator, grid) in specs.items():
        best_candidate, _ = search_subject_level(
            estimator,
            grid,
            train_frame,
            folds,
            MODEL_FEATURES,
        )

        rows.append(
            {
                "Model": name,
                "Subject F1-macro mean": float(best_candidate["Subject F1-macro mean"]),
                "Subject F1-macro std": float(best_candidate["Subject F1-macro std"]),
                "Subject Balanced Accuracy mean": float(
                    best_candidate["Subject Balanced Accuracy mean"]
                ),
                "Subject ROC-AUC mean": float(best_candidate["Subject ROC-AUC mean"]),
                "Best parameters": json.dumps(
                    best_candidate["Parameters"],
                    ensure_ascii=False,
                ),
            }
        )

        best_model = clone(estimator).set_params(**best_candidate["Parameters"])
        searches[name] = best_model

    benchmark = pd.DataFrame(rows).sort_values(
        ["Subject F1-macro mean", "Subject Balanced Accuracy mean", "Subject ROC-AUC mean"],
        ascending=False,
    )

    # 4. Lựa chọn Champion triển khai (chỉ chọn mô hình cung cấp xác suất)
    deployable = benchmark[benchmark["Model"] != "SVM (RBF, decision score)"]
    champion_name = str(deployable.iloc[0]["Model"])
    champion = searches[champion_name]

    # 5. Đánh giá Nested Cross-Validation cấp bệnh nhân
    nested_results = nested_subject_evaluation(
        train_frame,
        specs,
        feature_columns=MODEL_FEATURES,
        outer_splits=5,
        inner_splits=3,
        random_state=RANDOM_STATE,
    )

    # 6. Hiệu chỉnh Sigmoid lồng nhóm bệnh nhân và tìm quy tắc gộp dự đoán OOF
    oof_probabilities = _group_oof_calibration(
        champion,
        train_frame,
    )
    (
        aggregation,
        threshold,
        oof_subjects,
        threshold_table,
        aggregation_comparison,
    ) = _select_patient_rule(
        train_frame,
        oof_probabilities,
    )

    calibration_metrics = calculate_metrics(
        oof_subjects["status"],
        oof_subjects["prediction"],
        oof_subjects["probability"],
    )
    calibration_metrics["ECE (5 bins)"] = expected_calibration_error(
        oof_subjects["status"],
        oof_subjects["probability"],
    )

    # 7. Fit mô hình Calibrated cuối cùng trên toàn bộ tập Train
    calibrated_model = _calibrated_estimator(champion, train_frame)
    calibrated_model.fit(train_frame[MODEL_FEATURES], train_frame[TARGET_COLUMN])

    # 8. Đánh giá duy nhất 1 lần trên tập Holdout Test độc lập
    test_probability = positive_score(calibrated_model, test_frame[MODEL_FEATURES])
    test_subjects = aggregate_subject_predictions(
        test_frame,
        test_probability,
        threshold=threshold,
        aggregation=aggregation,
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

    # 9. Ước lượng khoảng tin cậy 95% CI bằng Patient Cluster Bootstrap
    confidence_intervals = bootstrap_subject_confidence_intervals(test_subjects)

    # 10. Đóng gói Artifact và xuất kết quả
    data_hash = sha256_file(data_path)
    feature_p1_p99 = compute_feature_percentiles(train_frame, MODEL_FEATURES)
    holdout_subjects_list = sorted(test_frame["subject_id"].unique())
    holdout_split_hash = hashlib.sha256(",".join(holdout_subjects_list).encode()).hexdigest()

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_sha = "unknown"

    bundle = {
        "model": calibrated_model,
        "champion_name": f"{champion_name} + sigmoid calibration",
        "feature_columns": MODEL_FEATURES,
        "original_feature_columns": ORIGINAL_FEATURES,
        "dropped_redundant_features": REDUNDANT_FEATURES,
        "decision_threshold": threshold,
        "probability_aggregation": aggregation,
        "random_state": RANDOM_STATE,
        "holdout_subjects": holdout_subjects_list,
        "holdout_split_hash": holdout_split_hash,
        "training_subject_count": int(train_frame["subject_id"].nunique()),
        "holdout_subject_count": int(test_frame["subject_id"].nunique()),
        "class_distribution": {
            "train_subjects": {
                str(k): int(v)
                for k, v in train_frame.groupby("subject_id")["status"]
                .first()
                .value_counts()
                .items()
            },
            "holdout_subjects": {
                str(k): int(v)
                for k, v in test_frame.groupby("subject_id")["status"]
                .first()
                .value_counts()
                .items()
            },
        },
        "feature_p1_p99": feature_p1_p99,
        "calibration": "sigmoid with subject-level folds",
        "oof_calibration_metrics": calibration_metrics,
        "holdout_subject_metrics": test_metrics,
        "training_config": DEFAULT_CONFIG,
        "data_sha256": data_hash,
        "artifact_version": "1.2.0",
        "schema_version": "1.0.0",
        "feature_contract_version": "1.0.0",
        "git_commit_sha": git_sha,
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
    }

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, output / "parkinsons_calibrated_pipeline.joblib")
    benchmark.to_csv(output / "model_benchmark.csv", index=False)
    nested_results.to_csv(output / "nested_cv_results.csv", index=False)
    oof_subjects.to_csv(output / "oof_subject_predictions.csv", index=False)
    test_subjects.to_csv(output / "holdout_subject_predictions.csv", index=False)
    confidence_intervals.to_csv(output / "holdout_bootstrap_ci.csv", index=False)
    threshold_table.to_csv(output / "oof_threshold_search.csv", index=False)
    aggregation_comparison.to_csv(output / "oof_aggregation_comparison.csv", index=False)

    feature_stability = _feature_selection_stability(champion, train_frame, folds)
    feature_stability.to_csv(output / "feature_selection_stability.csv", index=False)

    nested_metrics = {
        "F1-macro mean": float(nested_results["F1-macro"].mean()),
        "F1-macro std": float(nested_results["F1-macro"].std(ddof=0)),
        "Balanced Accuracy mean": float(nested_results["Balanced Accuracy"].mean()),
        "ROC-AUC mean": float(nested_results["ROC-AUC"].mean()),
    }

    metrics = {
        "dataset": {
            "records": len(frame),
            "subjects": int(frame["subject_id"].nunique()),
            "features_original": len(ORIGINAL_FEATURES),
            "features_model": len(MODEL_FEATURES),
        },
        "selection": {
            "unit": "subject",
            "primary_metric": "F1-macro",
            "champion": bundle["champion_name"],
            "tie_breaking_guardrail": "lower_std_then_bal_acc_then_simpler_model",
        },
        "nested_cv_subject": nested_metrics,
        "holdout_subject": test_metrics,
        "holdout_sample_size": int(test_frame["subject_id"].nunique()),
        "holdout_sample_caveat": (
            "Based on only 8 unseen subjects (6 PD, 2 control); high sampling variance. "
            "Nested CV is the primary indication of expected generalization."
        ),
        "calibration": {
            "method": "sigmoid",
            "aggregation": aggregation,
            "threshold": threshold,
        },
        "reproducibility": {
            "random_state": RANDOM_STATE,
            "data_sha256": data_hash,
            "git_commit_sha": git_sha,
            "sklearn_version": sklearn.__version__,
        },
    }

    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện và benchmark mô hình Parkinson.")
    parser.add_argument("--data", default="data/parkinsons.csv", help="Đường dẫn file dữ liệu CSV.")
    parser.add_argument("--artifacts", default="artifacts", help="Thư mục lưu trữ artifact.")
    args = parser.parse_args()
    print(train(args.data, args.artifacts).to_string(index=False))
