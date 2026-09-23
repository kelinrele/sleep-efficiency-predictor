"""Train and evaluate every model on one user-grouped split.

Usage: python train.py

Writes results/ (tables and diagnostics), models/ (deployable artifacts) and figures/.
All models are fit on training users, selected on validation users, and scored once on
test users.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sleep_predictor import data as d
from sleep_predictor import model as m
from sleep_predictor import plotting as p
from sleep_predictor.metrics import regression_report

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"

# Score of the original notebook model on a random row split (users in both train and test).
RANDOM_SPLIT_R2 = 0.7568


def lasso_candidates(df):
    """Every numeric column that is not an ID, the target, or a leakage column."""
    excluded = set(d.LEAKAGE_COLS) | {d.TARGET, d.GROUP, "date", "mood_encoded"}
    return [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def grouped_lasso(train):
    """Lasso feature selection with grouped CV, fit on training users only."""
    features = lasso_candidates(train)
    folds = list(d.group_kfold().split(train, groups=train[d.GROUP]))
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("lasso", LassoCV(cv=folds, alphas=50, random_state=d.SEED, max_iter=20000)),
        ]
    )
    pipe.fit(train[features], train[d.TARGET])
    coefs = pd.DataFrame(
        {"feature": features, "coefficient": pipe.named_steps["lasso"].coef_}
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coefs["kept"] = coefs["coefficient"] != 0
    return coefs, float(pipe.named_steps["lasso"].alpha_)


def user_bootstrap_r2(frame, pred, n_boot=1000, seed=d.SEED):
    """95% interval for test R2, resampling whole users (rows within a user are correlated)."""
    rng = np.random.default_rng(seed)
    y = frame[d.TARGET].to_numpy()
    pred = np.asarray(pred)
    user_rows = list(frame.reset_index(drop=True).groupby(d.GROUP).indices.values())
    scores = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(user_rows), len(user_rows))
        idx = np.concatenate([user_rows[i] for i in picks])
        scores.append(r2_score(y[idx], pred[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


class Evaluator:
    """Fits each model on train, and records validation and test scores in one table."""

    def __init__(self, train, val, test):
        self.train, self.val, self.test = train, val, test
        self.rows = []
        self.test_predictions = {}

    def add(self, name, model, features, notes=""):
        model.fit(self.train[features], self.train[d.TARGET])
        return self.add_fitted(name, model, features, notes)

    def add_fitted(self, name, model, features, notes="", predict=None):
        predict = predict or (lambda frame: model.predict(frame[features]))
        val_pred = predict(self.val)
        test_pred = predict(self.test)
        val = regression_report(self.val[d.TARGET], val_pred)
        test = regression_report(self.test[d.TARGET], test_pred)
        low, high = user_bootstrap_r2(self.test, test_pred)
        self.test_predictions[name] = np.asarray(test_pred)
        self.rows.append(
            {
                "model": name,
                "n_features": len(features),
                "val_r2": val["r2"],
                "test_r2": test["r2"],
                "test_r2_ci_low": low,
                "test_r2_ci_high": high,
                "test_rmse_pp": test["rmse_pp"],
                "test_mae_pp": test["mae_pp"],
                "notes": notes,
            }
        )
        print(f"  {name:<40} val R2 {val['r2']:.4f}  test R2 {test['r2']:.4f}  "
              f"RMSE {test['rmse_pp']:.2f} pp  MAE {test['mae_pp']:.2f} pp")
        return model

    def table(self):
        return pd.DataFrame(self.rows)


STAGE2_FEATURES = d.HABIT_FEATURES + d.LAG_FEATURES
MODEL_FEATURES = d.ACTIVITY_FEATURES + STAGE2_FEATURES

# Small grid; the number of trees comes from grouped early stopping, not the grid.
XGB_GRID = [
    dict(max_depth=depth, learning_rate=lr, min_child_weight=mcw, reg_lambda=5.0)
    for depth in (2, 3)
    for lr in (0.03, 0.1)
    for mcw in (20, 100)
]
SPLINE_MIN_GAIN = 0.002  # spline must beat a straight line by this much in grouped CV R2
XGB_MIN_GAIN = 0.01  # Stage 2 XGBoost must beat Ridge by this much on residual R2


def choose_stage1(train):
    """Spline on steps unless it is no better than a straight line."""
    X, y, g = train[d.ACTIVITY_FEATURES], train[d.TARGET], train[d.GROUP]
    spline_r2 = m.grouped_cv_r2(m.spline_pipeline(), X, y, g)
    linear_r2 = m.grouped_cv_r2(m.linear_pipeline(), X, y, g)
    use_spline = spline_r2 - linear_r2 >= SPLINE_MIN_GAIN
    info = {
        "spline_cv_r2": spline_r2,
        "linear_cv_r2": linear_r2,
        "min_gain": SPLINE_MIN_GAIN,
        "chosen": "spline" if use_spline else "linear",
    }
    return (m.spline_pipeline() if use_spline else m.linear_pipeline()), info


def plot_stage1(train, info):
    spline = m.spline_pipeline().fit(train[d.ACTIVITY_FEATURES], train[d.TARGET])
    linear = m.linear_pipeline().fit(train[d.ACTIVITY_FEATURES], train[d.TARGET])
    sample = train.sample(n=min(6000, len(train)), random_state=d.SEED)
    grid = pd.DataFrame({"steps": np.linspace(train.steps.min(), train.steps.max(), 300)})

    fig, ax = p.plt.subplots(figsize=(7, 4.2))
    ax.scatter(sample.steps, sample[d.TARGET] * 100, s=5, alpha=0.18, color=p.NEUTRAL,
               linewidths=0, label="training days (sample)")
    ax.plot(grid.steps, linear.predict(grid) * 100, color=p.ORANGE, linewidth=2,
            label=f"straight line (CV R² {info['linear_cv_r2']:.3f})")
    ax.plot(grid.steps, spline.predict(grid) * 100, color=p.BLUE, linewidth=2,
            label=f"spline (CV R² {info['spline_cv_r2']:.3f})")
    cap = train[d.TARGET].max() * 100
    ax.axhline(cap, color=p.TEXT_MUTED, linewidth=0.8, linestyle="--")
    ax.text(train.steps.min(), cap + 0.8, f"data capped at {cap:.0f}%", fontsize=8,
            color=p.TEXT_MUTED)
    ax.set_ylim(55, 106)
    ax.set_xlabel("Daily steps")
    ax.set_ylabel("Sleep efficiency (%)")
    ax.set_title("Stage 1: sleep efficiency against steps alone")
    ax.legend(loc="lower right")
    p.save(fig, FIGURES / "stage1_steps_curve.png")


def choose_stage2(train, stage1):
    """Tune XGBoost on out-of-fold Stage 1 residuals and compare it with Ridge."""
    y, g = train[d.TARGET].to_numpy(), train[d.GROUP]
    oof = cross_val_predict(
        stage1, train[d.ACTIVITY_FEATURES], y, groups=g, cv=d.group_kfold()
    )
    resid = y - oof
    X = train[STAGE2_FEATURES]

    tuning = m.tune_xgb_early_stopping(X, resid, g, XGB_GRID)
    tuning.to_csv(RESULTS / "stage2_xgb_tuning.csv", index=False)
    best = tuning.iloc[0]
    xgb_params = {
        "max_depth": int(best.max_depth),
        "learning_rate": float(best.learning_rate),
        "min_child_weight": int(best.min_child_weight),
        "reg_lambda": float(best.reg_lambda),
        "n_estimators": int(best.n_estimators),
    }

    xgb_r2 = m.grouped_cv_r2(m.shallow_xgb(**xgb_params), X, resid, g)
    ridge_r2 = m.grouped_cv_r2(m.ridge_pipeline(), X, resid, g)
    info = {
        "residual_cv_r2_xgb": xgb_r2,
        "residual_cv_r2_ridge": ridge_r2,
        "min_gain": XGB_MIN_GAIN,
        "chosen": "xgb" if xgb_r2 - ridge_r2 >= XGB_MIN_GAIN else "ridge",
        "xgb_params": xgb_params,
    }
    return xgb_params, info


def markdown_table(table):
    def fmt(col, v):
        if isinstance(v, float):
            return f"{v:.4f}" if "r2" in col else f"{v:.2f}"
        return str(v)

    lines = [
        "| " + " | ".join(table.columns) + " |",
        "|" + "|".join("---" for _ in table.columns) + "|",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(fmt(c, row[c]) for c in table.columns) + " |")
    return "\n".join(lines) + "\n"


def write_table(table, stem):
    table.to_csv(RESULTS / f"{stem}.csv", index=False)
    (RESULTS / f"{stem}.md").write_text(markdown_table(table))


def main():
    p.apply_style()
    for folder in (RESULTS, MODELS, FIGURES):
        folder.mkdir(exist_ok=True)

    df = d.build_dataset()
    train, val, test = d.grouped_split(df)
    print(f"rows: train {len(train)}, val {len(val)}, test {len(test)} | users: "
          f"{train[d.GROUP].nunique()}/{val[d.GROUP].nunique()}/{test[d.GROUP].nunique()}")

    print("Lasso feature selection (grouped CV, training users only)")
    coefs, alpha = grouped_lasso(train)
    coefs.to_csv(RESULTS / "lasso_features.csv", index=False)
    print(f"  alpha {alpha:.5f}; kept {int(coefs.kept.sum())} of {len(coefs)}: "
          f"{', '.join(coefs.loc[coefs.kept, 'feature'].head(12))}")

    d.save_feature_ranges(d.feature_ranges(train, MODEL_FEATURES), MODELS / "feature_ranges.json")

    print("Stage 1: steps curve")
    stage1, stage1_info = choose_stage1(train)
    (RESULTS / "stage1_choice.json").write_text(json.dumps(stage1_info, indent=2))
    plot_stage1(train, stage1_info)
    print(f"  spline CV R2 {stage1_info['spline_cv_r2']:.4f} vs straight line "
          f"{stage1_info['linear_cv_r2']:.4f} -> {stage1_info['chosen']}")

    print("Stage 2: habits on out-of-fold residuals")
    xgb_params, stage2_info = choose_stage2(train, stage1)
    (RESULTS / "stage2_choice.json").write_text(json.dumps(stage2_info, indent=2))
    print(f"  residual CV R2: XGBoost {stage2_info['residual_cv_r2_xgb']:.4f} "
          f"(params {xgb_params}), Ridge {stage2_info['residual_cv_r2_ridge']:.4f} "
          f"-> {stage2_info['chosen']}")

    print("Models (fit on train users, scored on held-out users)")
    ev = Evaluator(train, val, test)
    ev.add("Mean baseline", m.mean_baseline(), MODEL_FEATURES)
    ev.add("Ridge (steps + habits + lags)", m.ridge_pipeline(), MODEL_FEATURES)
    ev.add(
        "Original XGBoost (7 features)",
        m.original_xgb(),
        d.ORIGINAL_FEATURES,
        notes=f"was R2 {RANDOM_SPLIT_R2} on a random row split",
    )

    two_stage = {}
    for name, stage2 in [("xgb", m.shallow_xgb(**xgb_params)), ("ridge", m.ridge_pipeline())]:
        model = m.TwoStageRegressor(stage1, stage2, d.ACTIVITY_FEATURES, STAGE2_FEATURES)
        model.fit(train, train[d.TARGET], groups=train[d.GROUP])
        label = "XGBoost" if name == "xgb" else "Ridge"
        ev.add_fitted(
            f"Two-stage ({stage1_info['chosen']} + {label})",
            model,
            MODEL_FEATURES,
            notes="chosen Stage 2" if name == stage2_info["chosen"] else "",
        )
        two_stage[name] = model
    joblib.dump(two_stage[stage2_info["chosen"]], MODELS / "two_stage.joblib")

    write_table(ev.table(), "results_table")


if __name__ == "__main__":
    main()
