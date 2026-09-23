"""Train and evaluate every model on one user-grouped split.

Usage: python train.py

Writes results/ (tables and diagnostics), models/ (deployable artifacts) and figures/.
All models are fit on training users, selected on validation users, and scored once on
test users.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sleep_predictor import data as d
from sleep_predictor import model as m
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
        self.test_predictions[name] = np.asarray(test_pred)
        self.rows.append(
            {
                "model": name,
                "n_features": len(features),
                "val_r2": val["r2"],
                "test_r2": test["r2"],
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

    model_features = d.ACTIVITY_FEATURES + d.HABIT_FEATURES + d.LAG_FEATURES

    print("Models (fit on train users, scored on held-out users)")
    ev = Evaluator(train, val, test)
    ev.add("Mean baseline", m.mean_baseline(), model_features)
    ev.add("Ridge (steps + habits + lags)", m.ridge_pipeline(), model_features)
    ev.add(
        "Original XGBoost (7 features)",
        m.original_xgb(),
        d.ORIGINAL_FEATURES,
        notes=f"was R2 {RANDOM_SPLIT_R2} on a random row split",
    )

    write_table(ev.table(), "results_table")


if __name__ == "__main__":
    main()
