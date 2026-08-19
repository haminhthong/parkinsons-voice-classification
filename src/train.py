"""Huấn luyện benchmark leakage-aware và lưu mô hình triển khai."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data import ORIGINAL_FEATURES, TARGET_COLUMN, load_data, subject_holdout_split
from src.evaluate import make_subject_folds
from src.features import MODEL_FEATURES, REDUNDANT_FEATURES, make_pipeline

RANDOM_STATE = 42
SCORING = {
    "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro", "roc_auc": "roc_auc",
}


def model_specs() -> dict:
    return {
        "Dummy": (Pipeline([("model", DummyClassifier(strategy="prior"))]), {}),
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=3000, solver="liblinear", random_state=RANDOM_STATE)),
            {"select__k": [10, 15, "all"], "model__C": [0.1, 1, 10], "model__class_weight": [None, "balanced"]},
        ),
        "KNN": (
            make_pipeline(KNeighborsClassifier()),
            {"select__k": [10, 15, "all"], "model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"], "model__p": [1, 2]},
        ),
        # Không bật probability=True vì calibration nội bộ của SVC không biết subject_id.
        "SVM (RBF, decision score)": (
            make_pipeline(SVC(probability=False, random_state=RANDOM_STATE)),
            {"select__k": [10, 15, "all"], "model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.1], "model__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), scale=False),
            {"select__k": [15, "all"], "model__max_depth": [None, 5], "model__max_features": ["sqrt", 0.7], "model__class_weight": [None, "balanced"]},
        ),
        "HistGradientBoosting": (
            make_pipeline(HistGradientBoostingClassifier(max_iter=150, random_state=RANDOM_STATE), scale=False),
            {"select__k": [10, 15, "all"], "model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [7, 15], "model__l2_regularization": [0, 1]},
        ),
    }


def train(data_path: str | Path, artifact_dir: str | Path = "artifacts") -> pd.DataFrame:
    frame = load_data(data_path)
    train_frame, test_frame = subject_holdout_split(frame, random_state=RANDOM_STATE)
    folds = make_subject_folds(train_frame, n_splits=5, random_state=RANDOM_STATE)
    rows, searches = [], {}
    X_train, y_train = train_frame[MODEL_FEATURES], train_frame[TARGET_COLUMN]
    for name, (estimator, grid) in model_specs().items():
        search = GridSearchCV(estimator, grid, scoring=SCORING, refit="f1_macro", cv=folds, n_jobs=-1, error_score="raise")
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
    benchmark = pd.DataFrame(rows).sort_values(["CV F1-macro mean", "CV ROC-AUC"], ascending=False)

    # Mô hình triển khai phải có xác suất; chọn tốt nhất trong các ứng viên hỗ trợ predict_proba.
    deployable = benchmark[benchmark["Model"] != "SVM (RBF, decision score)"]
    champion_name = str(deployable.iloc[0]["Model"])
    bundle = {
        "model": searches[champion_name].best_estimator_, "champion_name": champion_name,
        "feature_columns": MODEL_FEATURES, "original_feature_columns": ORIGINAL_FEATURES,
        "dropped_redundant_features": REDUNDANT_FEATURES, "decision_threshold": 0.5,
        "random_state": RANDOM_STATE, "holdout_subjects": sorted(test_frame["subject_id"].unique()),
    }
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "parkinsons_champion_pipeline.joblib")
    benchmark.to_csv(output / "model_benchmark.csv", index=False)
    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/parkinsons.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    print(train(args.data, args.artifacts).to_string(index=False))
