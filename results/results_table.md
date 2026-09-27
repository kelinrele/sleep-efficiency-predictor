| model | n_features | val_r2 | test_r2 | test_r2_ci_low | test_r2_ci_high | test_rmse_pp | test_mae_pp | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mean baseline | 15 | -0.0763 | -0.0016 | -0.0585 | 0.0000 | 8.57 | 7.18 |  |
| Ridge (steps + habits + lags) | 15 | 0.6542 | 0.6720 | 0.6217 | 0.7102 | 4.90 | 3.90 |  |
| Original XGBoost (7 features) | 7 | 0.7716 | 0.7250 | 0.6712 | 0.7634 | 4.49 | 3.45 | was R2 0.7568 on a random row split |
| Two-stage (spline + XGBoost) | 15 | 0.7715 | 0.7253 | 0.6719 | 0.7636 | 4.49 | 3.44 |  |
| Two-stage (spline + Ridge) | 15 | 0.7718 | 0.7263 | 0.6729 | 0.7637 | 4.48 | 3.44 | chosen Stage 2 |
| SleepNet | 15 | 0.7707 | 0.7256 | 0.6721 | 0.7637 | 4.49 | 3.42 | single-seed test R2 0.7252 ± 0.0007 over 5 seeds; row is the seed ensemble |
| SleepNet + decorrelation penalty | 15 | 0.7707 | 0.7256 | 0.6723 | 0.7637 | 4.49 | 3.42 | single-seed test R2 0.7251 ± 0.0005 over 5 seeds; row is the seed ensemble |
| SleepNet + GRU history | 15 | 0.7705 | 0.7258 | 0.6723 | 0.7638 | 4.48 | 3.42 | single-seed test R2 0.7254 ± 0.0005 over 5 seeds; row is the seed ensemble |
