import numpy as np
import pandas as pd
import pytest

from sleep_predictor import data as d
from sleep_predictor import model as m

STAGE2 = d.HABIT_FEATURES + d.LAG_FEATURES


@pytest.fixture
def fitted(dataset):
    model = m.TwoStageRegressor(
        m.spline_pipeline(), m.ridge_pipeline(), d.ACTIVITY_FEATURES, STAGE2, n_splits=3
    )
    return model.fit(dataset, dataset[d.TARGET], groups=dataset[d.GROUP])


def test_contributions_sum_to_unclipped_prediction(fitted, dataset):
    parts = fitted.predict_contributions(dataset)
    raw = parts["activity"] + parts["habits"]
    inside = raw.between(*fitted.target_range_)
    assert inside.any()
    np.testing.assert_allclose(fitted.predict(dataset)[inside], raw[inside], atol=1e-6)


def test_prediction_is_clipped_to_training_range(fitted, dataset):
    extreme = dataset.head(3).copy()
    extreme["steps"] = [0, 1e6, 5e5]
    extreme["alcohol_units"] = [1e3, 0, 0]
    pred = fitted.predict(extreme)
    low, high = fitted.target_range_
    assert (pred >= low).all() and (pred <= high).all()


def test_stage2_is_trained_on_out_of_fold_residuals(fitted, dataset):
    y = dataset[d.TARGET].to_numpy()
    in_sample = fitted.stage1_.predict(dataset[d.ACTIVITY_FEATURES])
    np.testing.assert_allclose(fitted.train_residuals_, y - fitted.oof_activity_)
    # Out-of-fold predictions differ from the in-sample refit; equal would mean leakage.
    assert not np.allclose(fitted.oof_activity_, in_sample)


def test_spline_does_not_curve_up_past_the_data(dataset):
    pipe = m.spline_pipeline().fit(dataset[["steps"]], dataset[d.TARGET])
    edge = dataset.steps.max()
    far = pipe.predict(pd.DataFrame({"steps": [edge, edge * 3]}))
    assert far[1] == pytest.approx(far[0])
