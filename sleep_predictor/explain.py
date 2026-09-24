"""Attributions for the habits component, and advice text built from them.

Every attribution is in efficiency units (0-1) relative to a typical training day, so
attributions * 100 read as percentage points.
"""

import numpy as np
import pandas as pd

from . import data as d

# Inputs a person can change. Lag features describe the past and are not advice targets.
MODIFIABLE = ["alcohol_units", "stress_score", "caffeine_mg", "screen_time_min", "workout_type"]

LABELS = {
    "alcohol_units": "Alcohol",
    "stress_score": "Stress",
    "caffeine_mg": "Caffeine",
    "screen_time_min": "Screen time",
    "workout_type": "Workout type",
    "prev_sleep_eff": "Last night's efficiency",
    "steps_7d": "7-day average steps",
    "stress_7d": "7-day average stress",
}

# Direction in which common sleep guidance expects each habit to relate to sleep quality.
# A model association running the other way is reported as a likely data artefact.
EXPECTED_SIGN = {"alcohol_units": -1, "stress_score": -1, "caffeine_mg": -1, "screen_time_min": -1}

MIN_ADVICE_PP = 0.25  # attributions smaller than this are not worth mentioning


def combine_workout(attr):
    """Sum the one-hot workout columns into a single 'workout_type' attribution."""
    attr = attr.copy()
    workout_cols = [c for c in attr.columns if c.startswith("workout_type_")]
    attr["workout_type"] = attr[workout_cols].sum(axis=1)
    return attr.drop(columns=workout_cols)


def ridge_attributions(pipeline, X):
    """Exact linear attributions: coefficient * (x - training mean), in original units."""
    scaler, ridge = pipeline.named_steps["scale"], pipeline.named_steps["ridge"]
    values = (X.to_numpy() - scaler.mean_) / scaler.scale_ * ridge.coef_
    return pd.DataFrame(values, columns=X.columns, index=X.index)


def ridge_coefficients(pipeline, features):
    """Change in efficiency (pp) per unit of each raw feature."""
    scaler, ridge = pipeline.named_steps["scale"], pipeline.named_steps["ridge"]
    return pd.Series(ridge.coef_ / scaler.scale_ * 100, index=features)


def tree_attributions(pipeline, X):
    import shap

    explainer = shap.TreeExplainer(pipeline.named_steps["xgb"])
    return pd.DataFrame(explainer.shap_values(X), columns=X.columns, index=X.index)


def two_stage_attributions(model, frame):
    X = frame[list(model.habit_features)]
    stage2 = model.stage2_
    if "ridge" in stage2.named_steps:
        return ridge_attributions(stage2, X)
    return tree_attributions(stage2, X)


def sleepnet_habits_function(ens):
    """Habits-branch output (efficiency units) as a function of the raw habit matrix."""
    import torch

    def f(values):
        x = torch.tensor(ens.hab_scaler.transform(values))
        with torch.no_grad():
            out = np.mean([m.habits(x).squeeze(-1).numpy() for m in ens.models], axis=0)
        return out / 100.0

    return f


def sleepnet_attributions(ens, frame, background, nsamples=200, seed=d.SEED):
    """Kernel SHAP on the SleepNet habits branch, against a small training background."""
    import shap

    features = ens.habit_features
    explainer = shap.KernelExplainer(sleepnet_habits_function(ens), background[features].to_numpy())
    np.random.seed(seed)  # KernelExplainer samples coalitions from numpy's global generator
    values = explainer.shap_values(frame[features].to_numpy(), nsamples=nsamples, silent=True)
    return pd.DataFrame(values, columns=features, index=frame.index)


def counterintuitive_features(direction):
    """Features whose model direction (sign of the association) contradicts EXPECTED_SIGN."""
    return sorted(
        f for f, expected in EXPECTED_SIGN.items()
        if f in direction and np.sign(direction[f]) not in (0, expected)
    )


def advice(attr_row, inputs, counterintuitive=(), top=2):
    """Plain-language notes on the habits most associated with a lower prediction."""
    modifiable = attr_row.reindex(MODIFIABLE).dropna()
    negative = modifiable[modifiable * 100 <= -MIN_ADVICE_PP].sort_values()
    if negative.empty:
        return ["None of your habit inputs is associated with a notably lower prediction today."]

    notes = []
    for feature, value in negative.head(top).items():
        shown = inputs.get(feature, "")
        label = LABELS[feature]
        text = (
            f"**{label}** ({shown}) is associated with a {abs(value) * 100:.1f}-point lower "
            f"predicted efficiency than a typical day in the training data."
        )
        if feature in counterintuitive:
            text += (
                f" This model's association for {label.lower()} runs opposite to common sleep "
                "guidance and is likely an artefact of the synthetic training data, so it is "
                "not a recommendation."
            )
        notes.append(text)
    return notes
