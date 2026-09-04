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


def test_artifact_records_group_aware_calibration(artifact_path):
    bundle = load_bundle(artifact_path)
    assert bundle["calibration"] == "sigmoid with subject-level folds"
    assert 0 <= bundle["oof_calibration_metrics"]["Brier score"] <= 1
    assert 0 <= bundle["oof_calibration_metrics"]["ECE (5 bins)"] <= 1


def test_artifact_records_oof_selected_patient_rule(artifact_path):
    bundle = load_bundle(artifact_path)
    assert bundle["probability_aggregation"] in {"mean", "median", "max"}
    assert 0 < bundle["decision_threshold"] < 1


def test_artifact_contains_environment_metadata(artifact_path):
    bundle = joblib.load(artifact_path)

    required = {
        "artifact_version",
        "schema_version",
        "python_version",
        "sklearn_version",
        "data_sha256",
        "feature_columns",
        "decision_threshold",
        "probability_aggregation",
    }

    assert required.issubset(bundle)
