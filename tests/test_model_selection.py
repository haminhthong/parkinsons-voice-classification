"""Test suite kiểm thử module src/model_selection.py."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import build_subject_table, subject_holdout_split
from src.evaluate import make_subject_folds
from src.features import MODEL_FEATURES, make_pipeline
from src.model_selection import (
    search_subject_level,
    select_champion,
)


def test_subject_level_search_uses_one_row_per_subject(frame):
    train_frame, _ = subject_holdout_split(frame)

    folds = make_subject_folds(
        train_frame,
        n_splits=3,
        random_state=42,
    )

    for _, valid_index in folds:
        valid_frame = train_frame.iloc[valid_index]
        assert build_subject_table(valid_frame).shape[0] == valid_frame["subject_id"].nunique()


def test_search_subject_level_returns_valid_table(frame):
    train_frame, _ = subject_holdout_split(frame)
    folds = make_subject_folds(train_frame, n_splits=3, random_state=42)

    estimator = make_pipeline(LogisticRegression(max_iter=500, random_state=42))
    param_grid = {"model__C": [0.1, 1.0]}

    best_candidate, full_table = search_subject_level(
        estimator,
        param_grid,
        train_frame,
        folds,
        MODEL_FEATURES,
    )

    assert "Subject F1-macro mean" in best_candidate
    assert len(full_table) == 2
    assert "Subject F1-macro std" in full_table.columns
    assert "Subject Balanced Accuracy mean" in full_table.columns
    assert "Subject ROC-AUC mean" in full_table.columns


def test_select_champion_returns_top_deployable_model(frame):
    train_frame, _ = subject_holdout_split(frame)
    folds = make_subject_folds(train_frame, n_splits=3, random_state=42)

    specs = {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=500, random_state=42)),
            {"model__C": [1.0]},
        )
    }

    champion = select_champion(train_frame, folds, specs)
    assert champion.name == "Logistic Regression"
    assert isinstance(champion.estimator, Pipeline)
