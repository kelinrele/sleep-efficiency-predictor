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
from sleep_predictor import explain
from sleep_predictor import model as m
from sleep_predictor import plotting as p
from sleep_predictor.metrics import regression_report
from sleep_predictor.sleepnet import SleepNetEnsemble

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


SEEDS = (0, 1, 2, 3, 4)
SLEEPNET_VARIANTS = [
    ("SleepNet", dict(use_history=False, penalty_weight=0.0)),
    ("SleepNet + decorrelation penalty", dict(use_history=False, penalty_weight=5.0)),
    ("SleepNet + GRU history", dict(use_history=True, penalty_weight=0.0)),
]
KEEP_GAIN = 0.002  # an extra branch or penalty is kept only if validation R2 improves this much


def train_sleepnets(train, val, test, raw, ev):
    """Fit each SleepNet variant over several seeds; report the seed spread and the ensemble."""
    fitted = {}
    for name, kwargs in SLEEPNET_VARIANTS:
        ens = SleepNetEnsemble(d.ACTIVITY_FEATURES, STAGE2_FEATURES, **kwargs)
        ens.fit(train, val, raw, seeds=SEEDS)
        seed_r2 = [r2_score(test[d.TARGET], pr) for pr in ens.predict_each_seed(test, raw)]
        ev.add_fitted(
            name,
            ens,
            MODEL_FEATURES,
            notes=f"single-seed test R2 {np.mean(seed_r2):.4f} ± {np.std(seed_r2, ddof=1):.4f} "
                  f"over {len(SEEDS)} seeds; row is the seed ensemble",
            predict=lambda frame, ens=ens: ens.predict(frame, raw),
        )
        fitted[name] = (ens, r2_score(val[d.TARGET], ens.predict(val, raw)))

    base_name = SLEEPNET_VARIANTS[0][0]
    base_val = fitted[base_name][1]
    best_name = base_name
    for name, (_, val_r2) in fitted.items():
        if name != base_name and val_r2 - base_val >= KEEP_GAIN and val_r2 > fitted[best_name][1]:
            best_name = name
    info = {
        "val_r2": {name: v for name, (_, v) in fitted.items()},
        "keep_gain": KEEP_GAIN,
        "chosen": best_name,
    }
    return fitted, info


def plot_learning_curves(ens):
    fig, ax = p.plt.subplots(figsize=(7, 3.8))
    for i, curve in enumerate(ens.curves):
        ax.plot(np.arange(1, len(curve) + 1), np.sqrt(curve), color=p.BLUE,
                alpha=0.35 + 0.13 * i, linewidth=1.5, label="seeds" if i == 0 else None)
        best = int(np.argmin(curve))
        ax.scatter(best + 1, np.sqrt(curve[best]), s=36, color=p.ORANGE, zorder=3,
                   edgecolors=p.SURFACE, linewidths=1.5,
                   label="restored weights (best epoch)" if i == 0 else None)
    # The first few epochs start far above the plateau; zoom on the region early stopping acts in.
    best = min(np.sqrt(min(c)) for c in ens.curves)
    ax.set_ylim(best - 0.1, best + 0.6)
    ax.text(0.99, 0.02, "early epochs above this range are off-scale", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=p.TEXT_MUTED)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (percentage points)")
    ax.set_title("SleepNet training: early stopping on held-out users")
    ax.legend(loc="upper right")
    p.save(fig, FIGURES / "sleepnet_learning_curves.png")


def variance_decomposition(two_stage, net, train, test, raw):
    """How much steps explain alone, and how much habits explain of what steps leave over."""
    y = test[d.TARGET].to_numpy()
    parts = two_stage.predict_contributions(test)
    resid = y - parts["activity"].to_numpy()
    net_parts = net.predict_contributions(test, raw)
    net_resid = y - net_parts["activity"].to_numpy()

    # Fitting habits jointly with the steps spline (instead of on residuals) shows how much
    # of each habit's association the steps stage absorbs when the two are correlated.
    spline = two_stage.stage1_.named_steps["spline"]
    basis_cols = [f"spline_{i}" for i in range(spline.n_features_out_)]

    def with_basis(frame):
        basis = pd.DataFrame(spline.transform(frame[d.ACTIVITY_FEATURES]), columns=basis_cols,
                             index=frame.index)
        return pd.concat([basis, frame[STAGE2_FEATURES]], axis=1)

    joint = m.ridge_pipeline().fit(with_basis(train), train[d.TARGET])
    joint_coefs = explain.ridge_coefficients(joint, basis_cols + STAGE2_FEATURES)[STAGE2_FEATURES]

    # The same habits with steps entered as a straight line, as in the original notebook.
    linear = m.ridge_pipeline().fit(train[MODEL_FEATURES], train[d.TARGET])
    linear_coefs = explain.ridge_coefficients(linear, MODEL_FEATURES)[STAGE2_FEATURES]

    return {
        "two_stage": {
            "activity_only_test_r2": float(r2_score(y, parts["activity"])),
            "habits_r2_on_test_residuals": float(r2_score(resid, parts["habits"])),
            "full_test_r2": float(r2_score(y, two_stage.predict(test))),
        },
        "sleepnet": {
            "activity_only_test_r2": float(r2_score(y, net_parts["activity"])),
            "habits_r2_on_test_residuals": float(
                r2_score(net_resid, net_parts.drop(columns="activity").sum(axis=1))
            ),
            "full_test_r2": float(r2_score(y, net.predict(test, raw))),
        },
        "joint_test_r2": float(r2_score(y, joint.predict(with_basis(test)))),
        "habit_coefficients_pp_per_unit": {
            "residual_stage2": explain.ridge_coefficients(two_stage.stage2_, STAGE2_FEATURES).to_dict()
            if "ridge" in two_stage.stage2_.named_steps else None,
            "joint_with_steps_spline": joint_coefs.to_dict(),
            "with_linear_steps": linear_coefs.to_dict(),
        },
    }


def attribution_direction(attr, frame):
    """Sign of the association between each feature's value and its attribution."""
    out = {}
    for c in attr.columns:
        if frame[c].std() > 0 and attr[c].std() > 0:
            out[c] = float(np.corrcoef(frame[c], attr[c])[0, 1])
    return out


def plot_attribution_summary(attr, frame, path, title):
    """One row per feature: each dot is a test day's attribution, shaded by the input value."""
    attr = explain.combine_workout(attr) * 100
    order = attr.abs().mean().sort_values().index
    fig, ax = p.plt.subplots(figsize=(7.4, 0.42 * len(order) + 1.4))
    rng = np.random.default_rng(d.SEED)
    cmap = p.matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#86b6ef", "#0d366b"])
    for y, feature in enumerate(order):
        values = attr[feature].to_numpy()
        if feature == "workout_type":
            shade = np.full(len(values), 0.6)
        else:
            raw_values = frame[feature].to_numpy()
            span = np.ptp(raw_values) or 1.0
            shade = (raw_values - raw_values.min()) / span
        ax.scatter(values, y + rng.uniform(-0.28, 0.28, len(values)), c=shade, cmap=cmap,
                   vmin=0, vmax=1, s=6, alpha=0.6, linewidths=0)
    ax.set_yticks(range(len(order)), [explain.LABELS.get(f, f) for f in order])
    ax.axvline(0, color=p.TEXT_MUTED, linewidth=0.8)
    ax.set_xlabel("Contribution to predicted efficiency (percentage points, vs a typical day)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    sm = p.plt.cm.ScalarMappable(cmap=cmap, norm=p.matplotlib.colors.Normalize(0, 1))
    bar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.03)
    bar.set_ticks([0, 1], labels=["low", "high"])
    bar.set_label("input value", color=p.TEXT_MUTED)
    bar.outline.set_visible(False)
    p.save(fig, path)


def plot_importance(two_stage_attr, net_attr, path):
    a = (explain.combine_workout(two_stage_attr) * 100).abs().mean()
    b = (explain.combine_workout(net_attr) * 100).abs().mean()
    order = (a + b).sort_values().index
    y = np.arange(len(order))
    fig, ax = p.plt.subplots(figsize=(7, 0.45 * len(order) + 1.4))
    ax.barh(y + 0.2, a[order], height=0.36, color=p.BLUE, label="Two-stage (Stage 2)")
    ax.barh(y - 0.2, b[order], height=0.36, color=p.ORANGE, label="SleepNet (habits branch)")
    ax.set_yticks(y, [explain.LABELS.get(f, f) for f in order])
    ax.set_xlabel("Mean |contribution| on test days (percentage points)")
    ax.set_title("Habits and history: size of each input's contribution")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    p.save(fig, path)


def plot_pred_vs_actual(test, pred, name, path):
    fig, ax = p.plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(test[d.TARGET] * 100, np.asarray(pred) * 100, s=5, alpha=0.2, color=p.BLUE,
               linewidths=0)
    lims = [58, 101]
    ax.plot(lims, lims, color=p.TEXT_MUTED, linewidth=1, linestyle="--", label="perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    cap = test[d.TARGET].max() * 100
    ax.annotate(f"days recorded at the {cap:.0f}% cap", xy=(cap, 86), xytext=(84, 64),
                fontsize=8, color=p.TEXT_MUTED,
                arrowprops=dict(arrowstyle="-", color=p.TEXT_MUTED, linewidth=0.8))
    ax.set_xlabel("Actual sleep efficiency (%)")
    ax.set_ylabel("Predicted sleep efficiency (%)")
    ax.set_title(f"{name}: held-out users")
    ax.legend(loc="upper left")
    p.save(fig, path)


def markdown_table(table):
    def fmt(col, v):
        if isinstance(v, float):
            digits = 4 if "r2" in col else 2
            return f"{round(v, digits) + 0.0:.{digits}f}"  # + 0.0 turns -0.0 into 0.0
        return str(v)

    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
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

    print("SleepNet (5 seeds per variant)")
    raw = d.load_raw()
    sleepnets, sleepnet_info = train_sleepnets(train, val, test, raw, ev)
    (RESULTS / "sleepnet_choice.json").write_text(json.dumps(sleepnet_info, indent=2))
    plot_learning_curves(sleepnets[SLEEPNET_VARIANTS[0][0]][0])
    best_net, _ = sleepnets[sleepnet_info["chosen"]]
    best_net.save(MODELS / "sleepnet.pt")
    print(f"  kept variant: {sleepnet_info['chosen']}")

    write_table(ev.table(), "results_table")

    print("Interpretation")
    chosen = two_stage[stage2_info["chosen"]]
    decomposition = variance_decomposition(chosen, best_net, train, test, raw)
    (RESULTS / "variance_decomposition.json").write_text(json.dumps(decomposition, indent=2))
    for key in ("two_stage", "sleepnet"):
        part = decomposition[key]
        print(f"  {key}: steps alone R2 {part['activity_only_test_r2']:.4f}; habits explain "
              f"{part['habits_r2_on_test_residuals'] * 100:.1f}% of the remaining variance")

    sample = test.sample(n=200, random_state=d.SEED)
    background = train.sample(n=100, random_state=d.SEED)
    ts_attr = explain.two_stage_attributions(chosen, test)
    net_attr = explain.sleepnet_attributions(best_net, sample, background)
    plot_attribution_summary(ts_attr, test, FIGURES / "shap_summary_two_stage.png",
                             "Two-stage model: habit and history contributions")
    plot_attribution_summary(net_attr, sample, FIGURES / "shap_summary_sleepnet.png",
                             "SleepNet habits branch: Kernel SHAP (200 test days)")
    plot_importance(ts_attr.loc[sample.index], net_attr, FIGURES / "shap_importance.png")

    directions = {
        "two_stage": attribution_direction(ts_attr, test),
        "sleepnet": attribution_direction(net_attr, sample),
    }
    (RESULTS / "attribution_directions.json").write_text(json.dumps(directions, indent=2))

    # The app model is chosen on validation users; the test set plays no part in the choice.
    val_r2 = {
        "two_stage": float(r2_score(val[d.TARGET], chosen.predict(val))),
        "sleepnet": float(r2_score(val[d.TARGET], best_net.predict(val, raw))),
    }
    app_model = "sleepnet" if val_r2["sleepnet"] - val_r2["two_stage"] >= KEEP_GAIN else "two_stage"
    choice = {
        "model": app_model,
        "val_r2": val_r2,
        "keep_gain": KEEP_GAIN,
        "two_stage_stage2": stage2_info["chosen"],
        "sleepnet_variant": sleepnet_info["chosen"],
        "counterintuitive": explain.counterintuitive_features(directions[app_model]),
    }
    (MODELS / "model_choice.json").write_text(json.dumps(choice, indent=2))
    print(f"  app model: {app_model} (val R2 {val_r2}); counterintuitive: {choice['counterintuitive']}")

    app_pred = chosen.predict(test) if app_model == "two_stage" else best_net.predict(test, raw)
    name = "Two-stage model" if app_model == "two_stage" else "SleepNet"
    plot_pred_vs_actual(test, app_pred, name, FIGURES / "pred_vs_actual.png")


if __name__ == "__main__":
    main()
