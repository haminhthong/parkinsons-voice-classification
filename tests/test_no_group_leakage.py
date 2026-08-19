from sklearn.preprocessing import StandardScaler

from src.data import SUBJECT_COLUMN, subject_holdout_split
from src.evaluate import make_subject_folds
from src.features import make_pipeline
from sklearn.linear_model import LogisticRegression


def test_subject_never_appears_in_both_train_and_test(frame):
    train, test = subject_holdout_split(frame)
    assert set(train[SUBJECT_COLUMN]).isdisjoint(test[SUBJECT_COLUMN])


def test_every_validation_fold_has_both_classes_and_no_overlap(frame):
    train, _ = subject_holdout_split(frame)
    for fit_index, valid_index in make_subject_folds(train, n_splits=5):
        fit, valid = train.iloc[fit_index], train.iloc[valid_index]
        assert set(fit[SUBJECT_COLUMN]).isdisjoint(valid[SUBJECT_COLUMN])
        assert valid["status"].nunique() == 2


def test_scaler_is_only_a_pipeline_step():
    pipeline = make_pipeline(LogisticRegression())
    assert isinstance(pipeline.named_steps["scale"], StandardScaler)
    assert not hasattr(pipeline.named_steps["scale"], "mean_")

