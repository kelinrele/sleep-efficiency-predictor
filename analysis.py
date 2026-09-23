"""Feature diagnostics run on training users only.

Usage: python analysis.py

1. Timing check: does a row's efficiency line up best with that day's activity, or a
   neighbouring day's?
2. Collinearity: how strongly each habit and lag feature moves with steps.
3. Retest: grouped-CV change in R2 from adding each feature the original Lasso dropped.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from sleep_predictor import data as d
from sleep_predictor import plotting as p

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TIMING_COLS = ["steps", "stress_score", "alcohol_units", "caffeine_mg", "screen_time_min"]


def within_user(series, groups):
    """Remove each user's own mean, so correlations reflect day-to-day variation."""
    return series - series.groupby(groups).transform("mean")


def timing_check(raw_train):
    """Within-user correlation of efficiency on day t with each input on days t-1, t, t+1."""
    df = raw_train.dropna(subset=[d.TARGET] + TIMING_COLS).copy()
    by_user = df.groupby(d.GROUP)
    eff = within_user(df[d.TARGET], df[d.GROUP])
    rows = []
    for col in TIMING_COLS:
        for offset, label in [(-1, "day t-1"), (0, "day t"), (1, "day t+1")]:
            shifted = by_user[col].shift(-offset)
            valid = shifted.notna()
            x = within_user(shifted[valid], df.loc[valid, d.GROUP])
            rows.append(
                {"feature": col, "alignment": label, "within_user_corr": float(np.corrcoef(eff[valid], x)[0, 1])}
            )
    return pd.DataFrame(rows)


def collinearity_with_steps(train):
    cols = [c for c in d.HABIT_FEATURES + d.LAG_FEATURES + d.RETEST_FEATURES if c != "steps"]
    rows = []
    for c in cols:
        if train[c].std() == 0:
            continue
        x_within = within_user(train[c], train[d.GROUP])
        # Per-user constants such as age have no day-to-day variation to correlate.
        within = (
            float(x_within.corr(within_user(train["steps"], train[d.GROUP])))
            if x_within.std() > 1e-9
            else None
        )
        rows.append(
            {
                "feature": c,
                "pooled_corr": float(train[c].corr(train["steps"])),
                "within_user_corr": within,
            }
        )
    return pd.DataFrame(rows).sort_values("pooled_corr", key=np.abs, ascending=False)


def grouped_cv_r2(train, features):
    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
        random_state=d.SEED, n_jobs=-1,
    )
    return cross_val_score(
        model, train[features], train[d.TARGET], groups=train[d.GROUP],
        cv=d.group_kfold(), scoring="r2",
    )


def retest_dropped_features(train):
    base = d.ACTIVITY_FEATURES + d.HABIT_FEATURES + d.LAG_FEATURES
    base_scores = grouped_cv_r2(train, base)
    rows = []
    for feature in d.RETEST_FEATURES:
        scores = grouped_cv_r2(train, base + [feature])
        delta = scores - base_scores
        rows.append(
            {
                "feature": feature,
                "base_r2": float(base_scores.mean()),
                "with_feature_r2": float(scores.mean()),
                "delta_r2": float(delta.mean()),
                "delta_std": float(delta.std(ddof=1)),
                "folds_improved": int((delta > 0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    # Keep a feature only if it helps on average and in at least 4 of 5 folds.
    out["keep"] = (out["delta_r2"] > 0.002) & (out["folds_improved"] >= 4)
    return out


def plot_timing(timing):
    pivot = timing.pivot(index="feature", columns="alignment", values="within_user_corr")
    pivot = pivot.loc[TIMING_COLS, ["day t-1", "day t", "day t+1"]]
    fig, ax = p.plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(pivot))
    height = 0.26
    colors = {"day t-1": p.NEUTRAL, "day t": p.BLUE, "day t+1": p.ORANGE}
    for i, col in enumerate(pivot.columns):
        ax.barh(y + (i - 1) * height, pivot[col], height=height - 0.03, color=colors[col], label=col)
    ax.set_yticks(y, pivot.index)
    ax.invert_yaxis()
    ax.axvline(0, color=p.TEXT_MUTED, linewidth=0.8)
    ax.set_xlabel("Within-user correlation with sleep efficiency on day t")
    ax.set_title("Efficiency lines up with the same row's activity")
    ax.legend(loc="lower right", title="input measured on")
    ax.grid(axis="y", visible=False)
    p.save(fig, FIGURES / "timing_check.png")


def plot_collinearity(col):
    col = col.sort_values("pooled_corr")
    fig, ax = p.plt.subplots(figsize=(7, 4.2))
    colors = [p.BLUE if v >= 0 else p.RED for v in col["pooled_corr"]]
    ax.barh(col["feature"], col["pooled_corr"], color=colors, height=0.6)
    ax.axvline(0, color=p.TEXT_MUTED, linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Correlation with steps (training users)")
    ax.set_title("How much each feature moves with steps")
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(col["pooled_corr"]):
        ax.text(v + (0.02 if v >= 0 else -0.02), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=p.TEXT_MUTED)
    p.save(fig, FIGURES / "collinearity_with_steps.png")


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    p.apply_style()

    df = d.build_dataset()
    train, _, _ = d.grouped_split(df)
    train_users = set(train[d.GROUP])
    raw = d.load_raw()
    raw_train = raw[raw[d.GROUP].isin(train_users)]

    timing = timing_check(raw_train)
    timing.to_csv(RESULTS / "timing_check.csv", index=False)
    plot_timing(timing)
    print("Timing check (within-user corr with efficiency on day t):")
    print(timing.pivot(index="feature", columns="alignment", values="within_user_corr").round(3))

    col = collinearity_with_steps(train)
    col.to_csv(RESULTS / "collinearity_with_steps.csv", index=False)
    plot_collinearity(col)
    print("\nCorrelation with steps:")
    print(col.round(3).to_string(index=False))

    stress = raw["stress_score"]
    scale = {"min": int(stress.min()), "max": int(stress.max()), "median": float(stress.median()),
             "integer_valued": bool((stress == stress.round()).all())}
    (RESULTS / "stress_scale.json").write_text(json.dumps(scale, indent=2))
    print(f"\nstress_score scale: {scale}")

    retest = retest_dropped_features(train)
    retest.to_csv(RESULTS / "feature_retest.csv", index=False)
    print("\nDropped-feature retest (grouped 5-fold CV, XGBoost):")
    print(retest.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
