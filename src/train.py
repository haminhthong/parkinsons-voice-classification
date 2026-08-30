"""Module huấn luyện mô hình, hiệu chỉnh xác suất OOF và đóng gói artifact.

Thực hiện toàn bộ quy trình:
1. Chia Holdout theo bệnh nhân (`subject_holdout_split`).
2. Benchmark các mô hình ứng viên.
3. Lựa chọn mô hình Champion dựa trên F1-macro CV trung bình.
4. Hiệu chỉnh xác suất Sigmoid lồng nhóm bệnh nhân.
5. Chọn quy tắc gộp xác suất và ngưỡng quyết định tối ưu từ OOF Train.
6. Đóng gói mô hình calibrated pipeline cùng siêu dữ liệu.
"""


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
    aggregate_subject_predictions,
    bootstrap_subject_confidence_intervals,
    calculate_metrics,
    expected_calibration_error,
    make_subject_folds,
    positive_score,
    select_decision_threshold,
)
from src.features import MODEL_FEATURES, REDUNDANT_FEATURES, make_pipeline
from src.utils import sha256_file

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.json"
DEFAULT_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RANDOM_STATE = int(DEFAULT_CONFIG["random_state"])

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
}


def _calibrated_estimator(estimator, frame: pd.DataFrame) -> CalibratedClassifierCV:
    """Tạo bộ hiệu chỉnh Sigmoid Calibration đóng gói các fold phân chia theo bệnh nhân.

    Sử dụng `CalibratedClassifierCV` với `method='sigmoid'`, đảm bảo các fold hiệu chỉnh xác suất
    nội bộ tuân thủ nghiêm ngặt quy tắc độc lập bệnh nhân.

    Args:
        estimator: Mô hình gốc chưa hiệu chỉnh (đã được nhân bản bằng `clone`).
        frame: DataFrame tập huấn luyện.

    Returns:
        CalibratedClassifierCV: Đối tượng đã cấu hình các fold chia theo bệnh nhân.
    """
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
    """Sinh mảng xác suất Out-Of-Fold (OOF) được hiệu chỉnh Sigmoid theo nhóm bệnh nhân.

    Chia tập train thành các outer folds độc lập theo bệnh nhân. Trên mỗi outer fold fit,
    huấn luyện mô hình đã calibrated (bằng inner folds bệnh nhân), sau đó dự đoán xác suất
    cho outer fold validation.

    Args:
        estimator: Mô hình Champion chưa hiệu chỉnh.
        frame: DataFrame dữ liệu tập huấn luyện.

    Returns:
        np.ndarray: Mảng xác suất OOF đầy đủ cho mọi bản ghi trong tập train.

    Raises:
        AssertionError: Nếu có bản ghi nào bị khuyết xác suất OOF (chứa NaN).
    """
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
    """Tìm quy tắc gộp xác suất và ngưỡng quyết định tối ưu từ OOF Train.


    Tự động thử nghiệm các chiến lược gộp xác suất bản ghi theo bệnh nhân và quét ngưỡng quyết định
    để tối đa hóa Balanced Accuracy trên OOF Train mà không hề chạm đến tập Holdout Test.

    Args:
        frame: DataFrame tập huấn luyện.
        oof_probabilities: Mảng xác suất OOF của tập train.

    Returns:
        tuple[str, float, pd.DataFrame, pd.DataFrame]:
            (tên cách gộp tối ưu, ngưỡng chọn tối ưu, bảng dự đoán OOF bệnh nhân, bảng quét ngưỡng).
    """
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
        
    # Xếp hạng ứng viên gộp xác suất
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
    )


def _feature_selection_stability(
    estimator,
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Thống kê tần suất lựa chọn của từng đặc trưng qua các fold Cross-Validation.

    Đánh giá độ ổn định của bước `SelectKBest` nhằm hiểu rõ các đặc trưng nào thường xuyên
    được chọn nhất trong quá trình huấn luyện.

    Args:
        estimator: Pipeline huấn luyện mô hình.
        frame: DataFrame tập huấn luyện.
        folds: Danh sách các fold CV.

    Returns:
        pd.DataFrame: Bảng thống kê số lần và tỷ lệ chọn của từng đặc trưng.
    """
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
    """Định nghĩa cấu hình danh sách các mô hình ứng viên và lưới tham số (hyperparameter grid).

    Returns:
        dict: Từ điển ánh xạ tên mô hình -> (Pipeline chưa fit, GridSearchCV param_grid).
    """
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
        # Không bật probability=True cho SVC để tránh hiệu chỉnh ngầm rò rỉ nhóm
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
    """Thực hiện quy trình huấn luyện toàn diện, benchmark, hiệu chỉnh và xuất artifact kết quả.

    Args:
        data_path: Đường dẫn tới tệp CSV dữ liệu giọng nói Parkinson.
        artifact_dir: Thư mục lưu trữ các tệp mô hình joblib và báo cáo CSV/JSON.

    Returns:
        pd.DataFrame: Bảng kết quả Benchmark của tất cả mô hình ứng viên.
    """
    frame = load_data(data_path)
    
    # 1. Chia tập Holdout độc lập theo từng bệnh nhân
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
    X_train, y_train = train_frame[MODEL_FEATURES], train_frame[TARGET_COLUMN]
    
    # 3. Benchmark tất cả mô hình qua GridSearchCV với CV theo bệnh nhân
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
        
        rows.append(
            {
                "Model": name,
                "CV F1-macro mean": result["mean_test_f1_macro"][index],
                "CV F1-macro std": result["std_test_f1_macro"][index],
                "CV Balanced Accuracy": result["mean_test_balanced_accuracy"][index],
                "CV ROC-AUC": result["mean_test_roc_auc"][index],
                "Best parameters": json.dumps(
                    search.best_params_,
                    ensure_ascii=False,
                ),
            }
        )
        searches[name] = search

    benchmark = pd.DataFrame(rows).sort_values(
        ["CV F1-macro mean", "CV ROC-AUC"],
        ascending=False,
    )

    # 4. Lựa chọn Champion triển khai (chỉ chọn mô hình cung cấp xác suất)
    deployable = benchmark[benchmark["Model"] != "SVM (RBF, decision score)"]
    champion_name = str(deployable.iloc[0]["Model"])
    champion = searches[champion_name].best_estimator_

    # 5. Hiệu chỉnh Sigmoid lồng nhóm bệnh nhân và tìm quy tắc gộp dự đoán OOF
    oof_probabilities = _group_oof_calibration(
        champion,
        train_frame,
    )
    aggregation, threshold, oof_subjects, threshold_table = _select_patient_rule(
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

    # 6. Fit mô hình Calibrated cuối cùng trên toàn bộ tập Train
    calibrated_model = _calibrated_estimator(champion, train_frame)
    calibrated_model.fit(X_train, y_train)

    # 7. Đánh giá duy nhất 1 lần trên tập Holdout Test độc lập
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
    
    # 8. Ước lượng khoảng tin cậy 95% CI bằng Patient Cluster Bootstrap
    confidence_intervals = bootstrap_subject_confidence_intervals(test_subjects)

    # 9. Đóng gói Artifact và xuất kết quả
    bundle = {
        "model": calibrated_model,
        "champion_name": f"{champion_name} + sigmoid calibration",
        "feature_columns": MODEL_FEATURES,
        "original_feature_columns": ORIGINAL_FEATURES,
        "dropped_redundant_features": REDUNDANT_FEATURES,
        "decision_threshold": threshold,
        "random_state": RANDOM_STATE,
        "holdout_subjects": sorted(test_frame["subject_id"].unique()),
        "probability_aggregation": aggregation,
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
    threshold_table.to_csv(output / "oof_threshold_search.csv", index=False)
    
    feature_stability = _feature_selection_stability(champion, train_frame, folds)
    feature_stability.to_csv(output / "feature_selection_stability.csv", index=False)

    metrics = {
        "champion": bundle["champion_name"],
        "probability_aggregation": aggregation,
        "decision_threshold": threshold,
        "oof_calibration": calibration_metrics,
        "holdout_subject": test_metrics,
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

