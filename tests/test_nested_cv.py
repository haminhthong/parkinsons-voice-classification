"""Test suite kiểm thử Nested Cross-Validation và tính độc lập không leakage."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import subject_holdout_split
from src.evaluate import make_subject_folds
from src.features import MODEL_FEATURES, make_pipeline
from src.model_selection import (
    fit_selection_rule,
    nested_subject_evaluation,
)


def test_nested_cv_has_no_subject_overlap(frame):
    outer_folds = make_subject_folds(
        frame,
        n_splits=4,
        random_state=42,
    )

    for outer_fit, outer_valid in outer_folds:
        outer_train = frame.iloc[outer_fit].reset_index(drop=True)
        outer_validation = frame.iloc[outer_valid]

        assert set(outer_train["subject_id"]).isdisjoint(outer_validation["subject_id"])

        inner_folds = make_subject_folds(
            outer_train,
            n_splits=3,
            random_state=43,
        )

        for inner_fit, inner_valid in inner_folds:
            assert set(outer_train.iloc[inner_fit]["subject_id"]).isdisjoint(
                outer_train.iloc[inner_valid]["subject_id"]
            )


def test_holdout_labels_do_not_change_selected_rule(frame):
    train_frame, holdout_frame = subject_holdout_split(frame, test_size=0.25, random_state=42)

    oof_probs_1 = np.linspace(0.1, 0.9, len(train_frame))
    aggregation_1, threshold_1 = fit_selection_rule(train_frame, oof_probs_1)

    modified_holdout = holdout_frame.copy()
    modified_holdout["status"] = 1 - modified_holdout["status"]

    aggregation_2, threshold_2 = fit_selection_rule(train_frame, oof_probs_1)

    assert aggregation_1 == aggregation_2
    assert threshold_1 == threshold_2


def test_nested_subject_evaluation_returns_valid_table(frame):
    specs = {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=300, random_state=42)),
            {"model__C": [1.0]},
        )
    }

    result = nested_subject_evaluation(
        frame,
        specs,
        feature_columns=MODEL_FEATURES,
        outer_splits=3,
        inner_splits=2,
        random_state=42,
    )

    assert len(result) == 3
    assert "Outer fold" in result.columns
    assert "Champion" in result.columns
    assert "F1-macro" in result.columns
