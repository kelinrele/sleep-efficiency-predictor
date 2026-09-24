import numpy as np
import pandas as pd
import pytest

from sleep_predictor import data as d


def synthetic_raw(n_users=12, n_days=30, seed=0):
    """Small made-up frame with the same columns the real CSV uses."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for day in range(n_days):
            steps = rng.uniform(1000, 20000)
            rows.append(
                {
                    "user_id": f"U{u:04d}",
                    "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day),
                    "age": 30 + u,
                    "gender": ["female", "male", "other"][u % 3],
                    "bmi": 22.0 + u / 10,
                    "resting_hr_bpm": rng.normal(65, 3),
                    "hrv_rmssd_ms": rng.normal(50, 5),
                    "steps": steps,
                    "workout_type": d.WORKOUT_TYPES[(u + day) % len(d.WORKOUT_TYPES)],
                    "workout_minutes": rng.integers(0, 60),
                    "caffeine_mg": rng.uniform(0, 300),
                    "alcohol_units": rng.uniform(0, 3),
                    "screen_time_min": rng.uniform(30, 600),
                    "stress_score": rng.integers(20, 100),
                    "mindfulness_minutes": rng.integers(0, 30),
                    "sleep_efficiency": 0.6 + 0.3 * steps / 20000 + rng.normal(0, 0.02),
                    "sleep_duration_hours": rng.normal(7, 1),
                    "mood": "neutral",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def raw():
    return synthetic_raw()


@pytest.fixture
def dataset(raw):
    return d.encode(d.add_lag_features(raw).dropna().copy()).reset_index(drop=True)
