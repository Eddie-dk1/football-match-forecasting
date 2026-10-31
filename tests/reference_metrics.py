import numpy as np
import pandas as pd
from scipy.special import xlogy

from quant_football.scoring import PREDICTION_COLUMNS


def numpy_scores(truth: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Independent reference: no calls to sklearn metrics or scoring helpers."""
    y, p = truth.to_numpy(dtype=float), pred.to_numpy(dtype=float)
    result = {}
    for name, columns in [("total_goals_mae", [0, 1]), ("total_corners_mae", [2, 3])]:
        ok = np.isfinite(y[:, columns]).all(axis=1)
        result[name] = {
            "n": int(ok.sum()),
            "value": float(np.abs(y[ok][:, columns].sum(axis=1) - p[ok][:, columns].sum(axis=1)).mean())
            if ok.any()
            else None,
        }
    for i, name in enumerate(PREDICTION_COLUMNS):
        ok = np.isfinite(y[:, i])
        a, b = y[ok, i], p[ok, i]
        result[f"{name}_mae"] = {"n": len(a), "value": float(np.abs(a - b).mean()) if len(a) else None}
        result[f"{name}_deviance"] = {
            "n": len(a),
            "value": float((2 * (xlogy(a, a / b) - a + b)).mean()) if len(a) else None,
        }
    return result
