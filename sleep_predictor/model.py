"""Model definitions: baselines and the original benchmark."""

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .data import SEED


def mean_baseline():
    return DummyRegressor(strategy="mean")


def ridge_pipeline(alpha=1.0):
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def original_xgb():
    """The notebook's tuned configuration. Trees ignore feature scale, so no scaler."""
    return XGBRegressor(
        n_estimators=100, learning_rate=0.06, max_depth=3, random_state=SEED, n_jobs=-1
    )
