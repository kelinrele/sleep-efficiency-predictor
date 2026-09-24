"""End-to-end checks of the Streamlit app against the committed model files."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    return at


def widget(at, kind, prefix):
    return next(w for w in getattr(at, kind) if w.label.startswith(prefix))


def predict(at):
    at.button[0].click().run()
    assert not at.exception
    return {m.label: m.value for m in at.metric}, " ".join(m.value for m in at.markdown)


def test_disclaimer_visible_on_load(app):
    assert "Not medical advice" in app.info[0].value
    assert "synthetic" in app.info[0].value


def test_prediction_shows_activity_and_habits_split(app):
    metrics, _ = predict(app)
    assert set(metrics) == {"Predicted sleep efficiency", "Activity (steps)", "Habits and history"}


def test_advice_follows_the_largest_negative_attribution(app):
    alcohol = widget(app, "slider", "Alcohol")
    alcohol.set_value(alcohol.max)
    _, advice = predict(app)
    assert "**Alcohol**" in advice


def test_no_unsupported_claims(app):
    for high in (False, True):
        alcohol = widget(app, "slider", "Alcohol")
        alcohol.set_value(alcohol.max if high else 0.0)
        _, advice = predict(app)
        text = (advice + " ".join(e.value for e in app.info)).lower()
        assert "deep sleep" not in text and "late-night" not in text


def test_inputs_are_bounded_by_training_ranges(app):
    import json

    ranges = json.loads((APP.parent / "models" / "feature_ranges.json").read_text())
    steps = widget(app, "number_input", "Daily steps")
    assert steps.max == int(ranges["steps"]["max"])
    stress = widget(app, "slider", "Stress score")
    assert (stress.min, stress.max) == (int(ranges["stress_score"]["min"]), int(ranges["stress_score"]["max"]))


def test_every_workout_type_predicts(app):
    for i in range(len(app.selectbox[0].options)):
        app.selectbox[0].select_index(i)
        metrics, _ = predict(app)
        assert metrics["Predicted sleep efficiency"].endswith("%")
