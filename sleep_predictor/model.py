"""Model definitions: baselines, the original benchmark and the two-stage model."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
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


def spline_pipeline(n_knots=8, degree=3, alpha=1e-3):
    """Uniform knots and a flat tail: quantile knots with linear extrapolation curved up past
    the sparse high-step region and predicted efficiencies above 100%."""
    return Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=n_knots, degree=degree, knots="uniform", extrapolation="constant"
                ),
            ),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def linear_pipeline():
    return Pipeline([("linear", LinearRegression())])


def shallow_xgb(**params):
    defaults = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=2,
        min_child_weight=20,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
    )
    defaults.update(params)
    return Pipeline([("xgb", XGBRegressor(**defaults))])


class TwoStageRegressor(RegressorMixin, BaseEstimator):
    """Activity term from steps alone, plus a habits term fit on what steps leave unexplained.

    Stage 2 is trained on out-of-fold Stage 1 residuals (grouped by user), so it never learns
    from Stage 1's in-sample fit. The raw prediction is exactly activity + habits; predict()
    clips it to the target range seen in training (the data is capped at 0.99).
    """

    def __init__(self, stage1, stage2, activity_features, habit_features, n_splits=5):
        self.stage1 = stage1
        self.stage2 = stage2
        self.activity_features = activity_features
        self.habit_features = habit_features
        self.n_splits = n_splits

    def fit(self, X, y, groups):
        X_act = X[list(self.activity_features)]
        y = np.asarray(y, dtype=float)
        self.target_range_ = (float(y.min()), float(y.max()))
        self.oof_activity_ = cross_val_predict(
            clone(self.stage1), X_act, y, groups=groups, cv=GroupKFold(self.n_splits)
        )
        self.train_residuals_ = y - self.oof_activity_
        self.stage2_ = clone(self.stage2).fit(X[list(self.habit_features)], self.train_residuals_)
        self.stage1_ = clone(self.stage1).fit(X_act, y)
        return self

    def predict_contributions(self, X):
        return pd.DataFrame(
            {
                "activity": self.stage1_.predict(X[list(self.activity_features)]),
                "habits": self.stage2_.predict(X[list(self.habit_features)]),
            },
            index=X.index,
        )

    def predict(self, X):
        raw = self.predict_contributions(X).sum(axis=1).to_numpy()
        return np.clip(raw, *self.target_range_)


def grouped_cv_r2(model, X, y, groups, n_splits=5):
    pred = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits))
    return float(r2_score(y, pred))


def tune_xgb_early_stopping(X, y, groups, grid, n_splits=5, max_rounds=2000, patience=50):
    """Grouped CV with early stopping inside every fold.

    Each fold holds out 15% of its own training users as the early-stopping set, so the
    stopping point is never chosen on the fold being scored. Returns one row per config
    with its mean held-out R2 and the median best iteration.
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    rows = []
    for params in grid:
        scores, best_iters = [], []
        for fit_idx, score_idx in GroupKFold(n_splits).split(X, y, groups):
            inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
            tr, es = next(inner.split(fit_idx, groups=groups[fit_idx]))
            tr, es = fit_idx[tr], fit_idx[es]
            model = XGBRegressor(
                n_estimators=max_rounds,
                early_stopping_rounds=patience,
                eval_metric="rmse",
                random_state=SEED,
                n_jobs=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                **params,
            )
            model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[es], y[es])], verbose=False)
            best_iters.append(model.best_iteration + 1)
            scores.append(r2_score(y[score_idx], model.predict(X.iloc[score_idx])))
        rows.append(
            {
                **params,
                "cv_r2": float(np.mean(scores)),
                "cv_r2_std": float(np.std(scores, ddof=1)),
                "n_estimators": int(np.median(best_iters)),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_r2", ascending=False).reset_index(drop=True)
