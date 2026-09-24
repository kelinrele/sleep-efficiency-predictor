"""Loading, cleaning, feature construction and user-grouped splitting.

Everything the app needs to turn raw inputs into model features lives here, so training and
serving build features the same way.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "wearables_health_6mo_daily.csv"

TARGET = "sleep_efficiency"
GROUP = "user_id"
SEED = 42

# Other measurements of the same night's sleep, plus same-day mood (likely reported after the
# night). Using any of them as an input would leak the answer into the question.
LEAKAGE_COLS = [
    "sleep_duration_hours",
    "sleep_latency_min",
    "wake_after_sleep_onset_min",
    "sleep_stage_rem_pct",
    "sleep_stage_deep_pct",
    "sleep_stage_light_pct",
    "mood",
]

WORKOUT_TYPES = ["none", "walk", "run", "cycling", "strength", "yoga", "mixed"]
WORKOUT_COLS = [f"workout_type_{w}" for w in WORKOUT_TYPES]

# The seven inputs of the original notebook model, kept to benchmark it honestly.
ORIGINAL_FEATURES = [
    "steps",
    "alcohol_units",
    "stress_score",
    "caffeine_mg",
    "workout_type_none",
    "screen_time_min",
    "workout_type_walk",
]

ACTIVITY_FEATURES = ["steps"]
HABIT_FEATURES = ["alcohol_units", "stress_score", "caffeine_mg", "screen_time_min"] + WORKOUT_COLS
LAG_FEATURES = ["prev_sleep_eff", "steps_7d", "stress_7d"]

# Features the original Lasso dropped, retested under grouped CV.
RETEST_FEATURES = [
    "hrv_rmssd_ms",
    "resting_hr_bpm",
    "workout_minutes",
    "mindfulness_minutes",
    "age",
    "bmi",
]


def load_raw(path=DATA_PATH):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values([GROUP, "date"]).reset_index(drop=True)


def add_lag_features(df, window=7, min_periods=3):
    """Per-user history features that only look at earlier days.

    Computed on the full calendar before rows with missing values are dropped, so a missing
    night leaves a gap instead of silently pulling in an older night.
    """
    df = df.sort_values([GROUP, "date"]).copy()
    by_user = df.groupby(GROUP, sort=False)
    df["prev_sleep_eff"] = by_user[TARGET].shift(1)
    for col, name in [("steps", "steps_7d"), ("stress_score", "stress_7d")]:
        df[name] = by_user[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()
        )
    return df


def encode(df):
    """One-hot workout type over every category (so the app can offer all of them) and gender."""
    df = df.copy()
    for w in WORKOUT_TYPES:
        df[f"workout_type_{w}"] = (df["workout_type"] == w).astype(int)
    df["gender_male"] = (df["gender"] == "male").astype(int)
    df["gender_other"] = (df["gender"] == "other").astype(int)
    return df


def build_dataset(path=DATA_PATH):
    """Raw CSV -> model-ready frame: lags, then drop incomplete rows, then encode."""
    df = add_lag_features(load_raw(path))
    df = df.dropna().copy()
    return encode(df).reset_index(drop=True)


def grouped_split(df, test_size=0.15, val_size=0.15, seed=SEED):
    """Split by user into train/val/test. No user appears in more than one part."""
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, groups=df[GROUP]))
    trainval = df.iloc[trainval_idx]

    inner = GroupShuffleSplit(
        n_splits=1, test_size=val_size / (1 - test_size), random_state=seed
    )
    train_idx, val_idx = next(inner.split(trainval, groups=trainval[GROUP]))
    train, val, test = (
        trainval.iloc[train_idx].copy(),
        trainval.iloc[val_idx].copy(),
        df.iloc[test_idx].copy(),
    )

    users = [set(part[GROUP]) for part in (train, val, test)]
    assert not (users[0] & users[1] or users[0] & users[2] or users[1] & users[2])
    return train, val, test


def group_kfold(n_splits=5):
    return GroupKFold(n_splits=n_splits)


def feature_ranges(df, columns):
    return {
        c: {
            "min": float(df[c].min()),
            "max": float(df[c].max()),
            "median": float(df[c].median()),
        }
        for c in columns
    }


def save_feature_ranges(ranges, path):
    Path(path).write_text(json.dumps(ranges, indent=2))


def workout_one_hot(workout_type):
    if workout_type not in WORKOUT_TYPES:
        raise ValueError(f"unknown workout type: {workout_type}")
    return {f"workout_type_{w}": int(w == workout_type) for w in WORKOUT_TYPES}


def make_row(values):
    """Single-row frame from a dict of raw inputs, with workout_type expanded to one-hot."""
    values = dict(values)
    values.update(workout_one_hot(values.pop("workout_type")))
    return pd.DataFrame([values]).astype(np.float64)
