"""Regression metrics, with errors reported in percentage points of sleep efficiency."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_report(y_true, y_pred):
    """R², RMSE and MAE. The target is a 0-1 fraction, so errors are scaled by 100 into points."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_pp": float(np.sqrt(mean_squared_error(y_true, y_pred)) * 100),
        "mae_pp": float(mean_absolute_error(y_true, y_pred) * 100),
    }
