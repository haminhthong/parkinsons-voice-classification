import numpy as np

from src.data import subject_holdout_split
from src.evaluate import bootstrap_subject_confidence_intervals, expected_calibration_error


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

