import joblib
import numpy as np

from src.data import TARGET_COLUMN
from src.predict import load_bundle, predict_records


def test_saved_model_can_reload_and_reproduce_predictions(frame, artifact_path):
    inference = frame.drop(columns=[TARGET_COLUMN, "subject_id"]).head(12)
    first_bundle = load_bundle(artifact_path)
    first_records, first_subjects = predict_records(inference, first_bundle)
    second_bundle = joblib.load(artifact_path)
    second_records, second_subjects = predict_records(inference, second_bundle)
    np.testing.assert_allclose(
        first_records["probability_status_1"], second_records["probability_status_1"]
    )
    assert first_subjects.equals(second_subjects)


def test_probabilities_are_between_zero_and_one(frame, artifact_path):
    inference = frame.drop(columns=[TARGET_COLUMN, "subject_id"]).head(20)
    records, subjects = predict_records(inference, load_bundle(artifact_path))
    assert records["probability_status_1"].between(0, 1).all()
    assert subjects["probability_status_1"].between(0, 1).all()

