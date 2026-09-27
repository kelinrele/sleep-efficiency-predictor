# Findings

All numbers come from `python train.py` and `python analysis.py`. Models are fit on 210 training
users; validation (45 users) is used only for early stopping and model choice; test (45 users)
is used only for reporting and never influences a choice. The dataset is synthetic, so everything below describes patterns in generated
data, and every relationship is an association, not an effect.

## 1. The original score was optimistic

The original XGBoost was scored on a random row split, where the same person's days sit in both
training and test. On unseen users the same model scores **R² 0.725** instead of 0.757.

## 2. Every model reaches the same ceiling

| Model | Test R² | 95% interval (resampling users) | RMSE (pp) | MAE (pp) |
| --- | --- | --- | --- | --- |
| Mean baseline | −0.002 | −0.059 to 0.000 | 8.57 | 7.18 |
| Ridge (steps as a straight line) | 0.672 | 0.622 to 0.710 | 4.90 | 3.90 |
| Original XGBoost (7 features) | 0.725 | 0.671 to 0.763 | 4.49 | 3.45 |
| Two-stage: spline + XGBoost | 0.725 | 0.672 to 0.764 | 4.49 | 3.44 |
| **Two-stage: spline + Ridge** | **0.726** | 0.673 to 0.764 | 4.48 | 3.44 |
| SleepNet (5-seed ensemble) | 0.726 | 0.672 to 0.764 | 4.49 | 3.42 |
| SleepNet + decorrelation penalty | 0.726 | 0.672 to 0.764 | 4.49 | 3.42 |
| SleepNet + GRU history | 0.726 | 0.672 to 0.764 | 4.48 | 3.42 |

The spread between the best models (about 0.001) is far smaller than the uncertainty from
having only 45 test users (an interval about 0.09 wide). They are indistinguishable. Single
SleepNet seeds vary by at most ±0.0007. Neither the GRU history branch nor the decorrelation penalty
improved validation R² by the pre-set 0.002, so neither is kept. The app uses the two-stage
model because it scored best on validation users and its habit attributions are exact.

The remaining error of about 4.5 points looks like noise in the generator rather than signal
the models miss: a spline with a linear habits stage, gradient boosting and a neural network
with a recurrent history branch all stop at the same place. (Plain Ridge, with steps as a
straight line, stops lower at 0.672 because it cannot bend with the steps curve.)

## 3. Steps carry almost all of the signal

- A spline on steps alone reaches grouped-CV R² **0.755** (a straight line reaches 0.699). On
  the test users, steps alone give R² **0.718**.
- Habits and history explain **3.1%** of the variance that steps leave unexplained. SleepNet,
  whose branches are trained jointly rather than in stages, splits the variance the same way:
  its activity branch alone gives R² 0.717 on test, and its habits branch explains 3.1% of the
  remainder.
- The relationship is not linear: efficiency rises with steps up to about 15,000 and then flattens
  against a hard cap (`figures/stage1_steps_curve.png`).

## 4. The target is capped at 99%

17.8% of training days record exactly 0.99, and 0.09% record exactly 0.60. These look like
limits in the generator. A capped day's true value is unknown, which shows as a vertical stripe
at 99% in `figures/pred_vs_actual.png`. Predictions are clipped to the training range, and a
first version of the steps spline that curved above 100% past the data was replaced.

## 5. Efficiency lines up with the same row's activity, and no other day's

Within a person, efficiency on day t correlates with that row's steps (r = 0.68) and stress
(−0.19). The previous and next days' values show essentially nothing (|r| ≤ 0.015), so there
is no carry-over between days (`figures/timing_check.png`). A same-row correlation has no
direction in time, though: this cannot show whether a row's night comes after its day's
activity or before it. The dataset does not document it either. Like the original project,
the app assumes the night that follows the day. The pattern may simply be how the data was
generated.

The lag features behave accordingly. The 7-day steps average and last night's efficiency
correlate with steps **across** people (0.75 and 0.53) but hardly **within** a person (0.04 and
0.005). They describe how active someone typically is, not what happened yesterday, and they
add almost nothing once today's steps are known.

## 6. Features dropped by the original Lasso stay dropped

Adding HRV, resting heart rate, workout minutes, mindfulness minutes, age or BMI changes
grouped-CV R² by at most 0.0002 in either direction (`results/feature_retest.csv`). Age helps
in all 5 folds, but only by about 0.0001, far below the 0.002 needed to keep it.

## 7. Counterintuitive signs are artefacts of how steps are modelled

The original notebook found caffeine and screen time associated with *higher* efficiency.
Those signs only appear when steps enter the model as a straight line:

| Habit (points per unit) | Steps as a straight line | Steps as a spline (two-stage) |
| --- | --- | --- |
| Caffeine (per mg) | +0.0029 | −0.0002 |
| Screen time (per minute) | +0.0024 | −0.0002 |
| Stress (per point) | −0.028 | +0.0025 |
| Alcohol (per unit) | −1.17 | −1.69 |

Once the curve and the cap are modelled, caffeine and screen time turn slightly negative and
negligible: about −0.06 points at 300 mg, and −0.13 points at 600 minutes. The likely reason,
not tested directly: a straight-line steps term leaves the curve and the cap in the
residuals, and weakly correlated features absorb some of that. Either way, the small signs for caffeine, screen time and stress all depend on how steps is modelled, so none of
them should be read as an effect.

Stress is the one sign that stays counterintuitive. Its raw within-person association with
efficiency is negative (−0.19), but stress also moves with steps (−0.34 across people, −0.23
within a person). Once steps are in the model, stress adds almost nothing, with a very slightly
positive coefficient. Fitting habits jointly with the steps spline instead of on residuals gives
the same coefficients, so the two-stage fitting order does not produce it. In this data,
steps accounts for stress's association with sleep. The app flags stress as
counterintuitive. Its contribution never reaches the 0.25-point advice threshold (at most
about 0.12 points), and if it did, the app would label it an artefact rather than a
recommendation.

**Alcohol** is the only habit with a consistent, sizeable association: about −1.7 points per
unit, up to −5 points on the heaviest test days, in both the two-stage model and SleepNet
(`figures/shap_importance.png`). Workout type shifts predictions by at most about 0.15 points.

## 8. What this means for the app

The app shows the steps-only prediction and the habits shift separately. Its advice names only
habits that pull a prediction down by at least 0.25 points, and it flags any whose direction
contradicts common sleep guidance. In practice that means alcohol.
