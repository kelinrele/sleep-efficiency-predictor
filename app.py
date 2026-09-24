import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from sleep_predictor import data as d
from sleep_predictor import explain

MODELS = Path(__file__).resolve().parent / "models"


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


choice, ranges, model = load_model()


def bounds(feature):
    r = ranges[feature]
    return r["min"], r["max"], r["median"]


st.title("Sleep Efficiency Predictor")
st.info(
    "**Not medical advice.** This is a modeling study. The model was trained on a synthetic "
    "wearables dataset, so its outputs describe patterns in that data, not in real people. "
    "Talk to a clinician about sleep concerns."
)

st.subheader("Today")
lo, hi, mid = bounds("steps")
steps = st.number_input("Daily steps", min_value=int(lo), max_value=int(hi), value=int(mid), step=500)

lo, hi, mid = bounds("stress_score")
stress = st.slider(
    f"Stress score (the dataset's own {int(lo)}–{int(hi)} scale; higher means more stressed)",
    min_value=int(lo), max_value=int(hi), value=int(mid),
)

lo, hi, mid = bounds("alcohol_units")
alcohol = st.slider("Alcohol (units)", min_value=0.0, max_value=round(hi, 1), value=0.0, step=0.1)

lo, hi, mid = bounds("caffeine_mg")
caffeine = st.slider("Caffeine (mg)", min_value=0, max_value=int(hi), value=int(mid), step=10)

lo, hi, mid = bounds("screen_time_min")
screen = st.slider("Screen time (minutes)", min_value=int(lo), max_value=int(hi), value=int(mid), step=10)

workout = st.selectbox(
    "Workout type",
    d.WORKOUT_TYPES,
    format_func=lambda w: "No workout" if w == "none" else w.capitalize(),
)

with st.expander("Recent history (defaults are typical values from the training data)"):
    lo, hi, mid = bounds("prev_sleep_eff")
    prev_eff = st.slider("Last night's sleep efficiency (%)", min_value=int(round(lo * 100)),
                         max_value=int(round(hi * 100)), value=int(round(mid * 100)))
    lo, hi, mid = bounds("steps_7d")
    steps_7d = st.number_input("Average daily steps over the past 7 days", min_value=int(lo),
                               max_value=int(hi), value=int(mid), step=500)
    lo, hi, mid = bounds("stress_7d")
    stress_7d = st.slider("Average stress score over the past 7 days", min_value=int(lo),
                          max_value=int(hi), value=int(mid))

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
    left.metric("Activity (steps)", f"{activity * 100:.1f}%",
                help="What the model predicts from today's steps alone.")
    right.metric("Habits and history", f"{habits * 100:+.1f} pts",
                 help="How your other inputs shift the prediction away from the steps-only value.")
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
    st.bar_chart(chart, horizontal=True, x_label="Contribution (percentage points vs a typical day)")
    st.caption(
        "Contributions are associations learned from synthetic data, measured against a typical "
        "training day. They are not causal effects."
    )
