import numpy as np
import pytest

from sleep_predictor import data as d
from sleep_predictor.metrics import regression_report


def test_grouped_split_has_no_user_overlap(dataset):
    train, val, test = d.grouped_split(dataset)
    users = [set(p[d.GROUP]) for p in (train, val, test)]
    assert not users[0] & users[1]
    assert not users[0] & users[2]
    assert not users[1] & users[2]
    assert len(train) + len(val) + len(test) == len(dataset)


def test_grouped_split_is_reproducible(dataset):
    first = d.grouped_split(dataset)
    second = d.grouped_split(dataset)
    for a, b in zip(first, second):
        assert set(a[d.GROUP]) == set(b[d.GROUP])


def test_lag_features_ignore_same_day(raw):
    base = d.add_lag_features(raw)
    spiked = raw.copy()
    row = spiked.index[(spiked.user_id == "U0001") & (spiked.date == spiked.date.min() + np.timedelta64(10, "D"))][0]
    spiked.loc[row, ["steps", "stress_score", "sleep_efficiency"]] = [99999, 999, 0.1]
    after = d.add_lag_features(spiked)

    # Day t's own lags are untouched; the next day's lags see the spike.
    for col in d.LAG_FEATURES:
        assert after.loc[row, col] == base.loc[row, col]
        assert after.loc[row + 1, col] != base.loc[row + 1, col]


def test_lag_features_do_not_cross_users(raw):
    lagged = d.add_lag_features(raw)
    first_days = lagged.groupby(d.GROUP).head(1)
    assert first_days[d.LAG_FEATURES].isna().all().all()


def test_no_leakage_columns_in_feature_sets():
    for features in (d.ORIGINAL_FEATURES, d.HABIT_FEATURES, d.LAG_FEATURES, d.RETEST_FEATURES):
        assert not set(features) & set(d.LEAKAGE_COLS)


def test_workout_one_hot_covers_every_type():
    for w in d.WORKOUT_TYPES:
        row = d.workout_one_hot(w)
        assert sum(row.values()) == 1 and row[f"workout_type_{w}"] == 1
    with pytest.raises(ValueError):
        d.workout_one_hot("swimming")


def test_metrics_in_percentage_points():
    report = regression_report([0.80, 0.90], [0.85, 0.90])
    assert report["mae_pp"] == pytest.approx(2.5)
    assert report["rmse_pp"] == pytest.approx(np.sqrt((0.05**2) / 2) * 100)
