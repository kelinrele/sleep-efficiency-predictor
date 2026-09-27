import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sleep_predictor import data as d
from sleep_predictor import explain

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DATASET_URL = (
    "https://www.kaggle.com/datasets/mftnakrsu/health-wearables-stresssleep-tracking-syntc"
)


@st.cache_resource
def load_model():
    """Load the model chosen by train.py once per process."""
    choice = json.loads((MODELS / "model_choice.json").read_text())
    ranges = json.loads((MODELS / "feature_ranges.json").read_text())
    if choice["model"] == "two_stage":
        model = joblib.load(MODELS / "two_stage.joblib")
    else:
        from sleep_predictor.sleepnet import SleepNetEnsemble

        model = SleepNetEnsemble.load(MODELS / "sleepnet.pt")
    return choice, ranges, model


def attributions(choice, model, ranges, row):
    if choice["model"] == "two_stage":
        return explain.two_stage_attributions(model, row)
    # One typical training day (medians) as the Kernel SHAP reference point.
    typical = pd.DataFrame([{f: ranges[f]["median"] for f in model.habit_features}])
    return explain.sleepnet_attributions(model, row, typical, nsamples=100)


@st.cache_resource
def load_model_card(model_name, sleepnet_variant):
    """Test-set scores and variance split for the deployed model, as written by train.py."""
    table = pd.read_csv(RESULTS / "results_table.csv", keep_default_na=False)
    if model_name == "two_stage":
        row = table[table["notes"] == "chosen Stage 2"].iloc[0]
    else:
        row = table[table["model"] == sleepnet_variant].iloc[0]
    original = table[table["model"].str.startswith("Original XGBoost")].iloc[0]
    results = json.loads((RESULTS / "variance_decomposition.json").read_text())
    alcohol = (results["habit_coefficients_pp_per_unit"]["residual_stage2"] or {}).get(
        "alcohol_units"
    )
    return row, original, results[model_name], alcohol


choice, ranges, model = load_model()
card, original, split, alcohol_pp = load_model_card(choice["model"], choice["sleepnet_variant"])


def bounds(feature):
    r = ranges[feature]
    return r["min"], r["max"], r["median"]


st.title("Sleep Efficiency Predictor")
st.caption(
    f"Trained on a synthetic wearables dataset (300 people, 6 months of daily records) and "
    f"tested on 45 people it never saw: typical error ±{card['test_mae_pp']:.1f} points."
)

with st.expander("About the model and training data"):
    if choice["model"] == "two_stage":
        habits_part = (
            "a linear model" if choice["two_stage_stage2"] == "ridge" else "gradient-boosted trees"
        )
        model_text = (
            f"**{card['model']}**. A curve on daily steps gives the *Activity* part, and "
            f"{habits_part} of habits and recent history gives the *Habits* shift."
        )
    else:
        model_text = (
            "**SleepNet**, a neural network with separate branches for steps (*Activity*) "
            "and for habits and recent history (*Habits*)."
        )
    alcohol_line = (
        f" Alcohol is the only habit with a sizeable association: about "
        f"{alcohol_pp:+.1f} points per unit."
        if alcohol_pp is not None
        else ""
    )
    st.markdown(
        f"""
**Training data**
- [Health + Wearables + Stress/Sleep Tracking]({DATASET_URL}) on Kaggle: daily wearable records
  for 300 people over 6 months. That comes to about 52,000 usable days after dropping incomplete
  days and each person's first days, which have no history.
- The dataset is **synthetic** (generated, not recorded from real people), so predictions
  describe patterns in that generated data.
- Sleep efficiency in the data runs from 60% to 99%, and about 18% of days sit exactly at the
  99% cap, so predictions never go above 99%.

**How it was trained and tested**
- People were split into three groups: 210 to train on, 45 to choose between models, and 45
  kept aside for the final test. Nobody appears in more than one group, so the test shows how
  the model does for someone new.
- Model: {model_text} It scored best on the 45 people used to choose between models;
  gradient boosting and a neural network scored about the same.

**Accuracy on the 45 test people**
- R² **{card["test_r2"]:.3f}** (95% range {card["test_r2_ci_low"]:.2f} to
  {card["test_r2_ci_high"]:.2f}). Typical error ±{card["test_mae_pp"]:.1f} points;
  root-mean-square error {card["test_rmse_pp"]:.1f} points.
- The project's original XGBoost model scores R² {original["test_r2"]:.3f} when tested the
  same way. Its earlier 0.757 came from testing on people it had already seen.

**What drives the predictions**
- Steps alone explain R² {split["activity_only_test_r2"]:.2f}. Habits and recent history explain
  {split["habits_r2_on_test_residuals"]:.0%} of what is left.{alcohol_line}

**Limits**
- Inputs are limited to the ranges seen in the training data.
- The contributions shown are associations in the data, not causes.
"""
    )

st.subheader("Today")
lo, hi, mid = bounds("steps")
steps = st.number_input(
    "Daily steps", min_value=int(lo), max_value=int(hi), value=int(mid), step=500
)

lo, hi, mid = bounds("stress_score")
stress = st.slider(
    f"Stress score (the dataset's own {int(lo)}–{int(hi)} scale; higher means more stressed)",
    min_value=int(lo),
    max_value=int(hi),
    value=int(mid),
)

lo, hi, mid = bounds("alcohol_units")
alcohol = st.slider("Alcohol (units)", min_value=0.0, max_value=round(hi, 1), value=0.0, step=0.1)

lo, hi, mid = bounds("caffeine_mg")
caffeine = st.slider("Caffeine (mg)", min_value=0, max_value=int(hi), value=int(mid), step=10)

lo, hi, mid = bounds("screen_time_min")
screen = st.slider(
    "Screen time (minutes)", min_value=int(lo), max_value=int(hi), value=int(mid), step=10
)

workout = st.selectbox(
    "Workout type",
    d.WORKOUT_TYPES,
    format_func=lambda w: "No workout" if w == "none" else w.capitalize(),
)

with st.expander("Recent history (defaults are typical values from the training data)"):
    lo, hi, mid = bounds("prev_sleep_eff")
    prev_eff = st.slider(
        "Last night's sleep efficiency (%)",
        min_value=int(round(lo * 100)),
        max_value=int(round(hi * 100)),
        value=int(round(mid * 100)),
    )
    lo, hi, mid = bounds("steps_7d")
    steps_7d = st.number_input(
        "Average daily steps over the past 7 days",
        min_value=int(lo),
        max_value=int(hi),
        value=int(mid),
        step=500,
    )
    lo, hi, mid = bounds("stress_7d")
    stress_7d = st.slider(
        "Average stress score over the past 7 days",
        min_value=int(lo),
        max_value=int(hi),
        value=int(mid),
    )

inputs = {
    "steps": steps,
    "alcohol_units": alcohol,
    "stress_score": stress,
    "caffeine_mg": caffeine,
    "screen_time_min": screen,
    "prev_sleep_eff": prev_eff / 100,
    "steps_7d": steps_7d,
    "stress_7d": stress_7d,
    "workout_type": workout,
}

if st.button("Predict sleep efficiency", type="primary"):
    row = d.make_row(inputs)
    prediction = float(model.predict(row)[0])
    parts = model.predict_contributions(row).iloc[0]
    activity = parts["activity"]
    habits = parts.drop("activity").sum()

    st.metric("Predicted sleep efficiency", f"{prediction * 100:.0f}%")
    left, right = st.columns(2)
    left.metric(
        "Activity (steps)",
        f"{activity * 100:.1f}%",
        help="What the model predicts from today's steps alone.",
    )
    right.metric(
        "Habits and history",
        f"{habits * 100:+.1f} pts",
        help="How your other inputs shift the prediction away from the steps-only value.",
    )
    if abs((activity + habits) - prediction) > 5e-4:
        low, high = model.target_range_
        st.caption(
            f"Activity plus habits comes to {(activity + habits) * 100:.1f}%, but predictions are "
            f"kept within {low * 100:.0f}–{high * 100:.0f}%, the range recorded in the training "
            "data (the dataset caps sleep efficiency at 99%)."
        )
    attr = explain.combine_workout(attributions(choice, model, ranges, row)).iloc[0]
    shown = {
        "alcohol_units": f"{alcohol:.1f} units",
        "stress_score": f"{stress}",
        "caffeine_mg": f"{caffeine} mg",
        "screen_time_min": f"{screen} min",
        "workout_type": "no workout" if workout == "none" else workout,
    }
    st.subheader("What the model associates with this prediction")
    for note in explain.advice(attr, shown, counterintuitive=choice["counterintuitive"]):
        st.markdown(f"- {note}")

    chart = (attr * 100).rename(lambda f: explain.LABELS.get(f, f)).sort_values()
    st.bar_chart(
        chart, horizontal=True, x_label="Contribution (percentage points vs a typical day)"
    )
    st.caption(
        "Contributions are associations learned from synthetic data, measured against a typical "
        "training day. They are not causal effects."
    )
