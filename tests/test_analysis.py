import numpy as np
import pandas as pd

import analysis
from sleep_predictor import data as d


def lagged_raw(n_users=10, n_days=60, seed=0):
    """Efficiency depends on the previous day's steps only, so day t-1 must win the timing check."""
    rng = np.random.default_rng(seed)
    frames = []
    for u in range(n_users):
        steps = rng.uniform(1000, 20000, n_days)
        eff = np.r_[np.nan, 0.6 + 0.3 * steps[:-1] / 20000] + rng.normal(0, 0.005, n_days)
        frames.append(
            pd.DataFrame(
                {
                    "user_id": f"U{u:04d}",
                    "date": pd.date_range("2025-01-01", periods=n_days),
                    "steps": steps,
                    "stress_score": rng.integers(20, 100, n_days),
                    "alcohol_units": rng.uniform(0, 3, n_days),
                    "caffeine_mg": rng.uniform(0, 300, n_days),
                    "screen_time_min": rng.uniform(30, 600, n_days),
                    d.TARGET: eff,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_within_user_removes_each_users_mean(raw):
    centred = analysis.within_user(raw["steps"], raw[d.GROUP])
    assert np.allclose(centred.groupby(raw[d.GROUP]).mean(), 0)


def test_timing_check_finds_the_day_that_drives_efficiency():
    timing = analysis.timing_check(lagged_raw())
    steps = timing[timing.feature == "steps"].set_index("alignment")["within_user_corr"]
    assert steps.idxmax() == "day t-1"
    assert steps["day t-1"] > 0.9
    assert abs(steps["day t"]) < 0.2 and abs(steps["day t+1"]) < 0.2


def test_collinearity_skips_within_user_corr_for_per_user_constants(dataset):
    table = analysis.collinearity_with_steps(dataset).set_index("feature")
    assert pd.isna(table.loc["age", "within_user_corr"])
    assert pd.isna(table.loc["bmi", "within_user_corr"])
    assert pd.notna(table.loc["stress_score", "within_user_corr"])
    assert table["pooled_corr"].between(-1, 1).all()


def test_retest_does_not_keep_pure_noise_features(dataset):
    # In the synthetic fixture efficiency depends on steps only, so every retested feature is noise.
    retest = analysis.retest_dropped_features(dataset, n_estimators=20)
    assert list(retest.feature) == d.RETEST_FEATURES
    assert not retest["keep"].any()
    expected = (retest.delta_r2 > 0.002) & (retest.folds_improved >= 4)
    assert (retest["keep"] == expected).all()
