import numpy as np
import pandas as pd

from src.data import subject_holdout_split
from src.evaluate import bootstrap_subject_confidence_intervals, expected_calibration_error
from src.evaluate import aggregate_subject_predictions, select_decision_threshold


def test_expected_calibration_error_is_bounded():
    value = expected_calibration_error([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0 <= value <= 1


def test_bootstrap_confidence_intervals_are_valid(frame):
    _, test = subject_holdout_split(frame)
    subjects = test.groupby("subject_id", as_index=False).agg(status=("status", "first"))
    subjects["probability"] = np.where(subjects["status"].eq(1), 0.8, 0.2)
    subjects["prediction"] = (subjects["probability"] >= 0.5).astype(int)
    intervals = bootstrap_subject_confidence_intervals(subjects, n_bootstrap=100)
    assert intervals["Valid bootstrap samples"].min() > 0
    assert (intervals["CI 2.5%"] <= intervals["CI 97.5%"] ).all()


def test_patient_probability_aggregation_supports_expected_methods(frame):
    probabilities = np.linspace(0.05, 0.95, len(frame))
    for method in ["mean", "median", "max"]:
        subjects = aggregate_subject_predictions(
            frame,
            probabilities,
            aggregation=method,
        )
        assert len(subjects) == frame["subject_id"].nunique()
        assert subjects["probability"].between(0, 1).all()


def test_threshold_selection_respects_specificity_constraint():
    subjects = pd.DataFrame(
        {
            "status": [0, 0, 1, 1],
            "probability": [0.1, 0.4, 0.6, 0.9],
        }
    )
    threshold, table = select_decision_threshold(
        subjects,
        minimum_specificity=0.5,
    )
    selected = table.loc[np.isclose(table["Threshold"], threshold)].iloc[0]
    assert 0 <= threshold <= 1
    assert selected["Specificity"] >= 0.5
