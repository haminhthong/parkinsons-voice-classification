import pytest

from src.data import ORIGINAL_FEATURES, TARGET_COLUMN, validate_dataframe


def test_dataset_has_exactly_22_original_features(frame):
    assert len(ORIGINAL_FEATURES) == 22
    assert set(ORIGINAL_FEATURES).issubset(frame.columns)


def test_status_contains_only_zero_and_one(frame):
    assert set(frame[TARGET_COLUMN].unique()) == {0, 1}


def test_inference_missing_column_is_rejected(frame):
    invalid = frame.drop(columns=[ORIGINAL_FEATURES[0], TARGET_COLUMN])
    with pytest.raises(ValueError, match="Thiếu cột"):
        validate_dataframe(invalid, require_target=False)


def test_non_binary_status_is_rejected(frame):
    invalid = frame.copy()
    invalid.loc[invalid.index[0], TARGET_COLUMN] = 2
    with pytest.raises(ValueError, match="0 và 1"):
        validate_dataframe(invalid)
