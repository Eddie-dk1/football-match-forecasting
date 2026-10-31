"""The ten assignment metrics, with explicit row masks and no target filling."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from .config import TARGETS

PREDICTION_COLUMNS = tuple(c.removesuffix("_90") for c in TARGETS)
METRIC_NAMES = ("total_goals_mae", "total_corners_mae") + tuple(
    f"{c}_{metric}" for c in PREDICTION_COLUMNS for metric in ("mae", "deviance")
)


def validate_predictions(predictions: pd.DataFrame, expected_index: pd.Index) -> None:
    if tuple(predictions.columns) != TARGETS:
        raise ValueError("Prediction columns differ from the four required targets")
    if not expected_index.is_unique or expected_index.name != "match_id":
        raise ValueError("Expected unique match_id index")
    if not pd.api.types.is_integer_dtype(expected_index.dtype) or (expected_index <= 0).any():
        raise ValueError("match_id values must be positive integers")
    if not predictions.index.equals(expected_index):
        raise ValueError("Prediction match_id order or coverage differs from schedule")
    values = predictions.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Predicted expectations must be finite and strictly positive")


def score_predictions(truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    validate_predictions(predictions, truth.index)
    if tuple(truth.columns) != TARGETS:
        raise ValueError("Truth must contain the four ordered targets")
    values = truth.to_numpy(dtype="float64")
    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all() or (observed < 0).any() or (observed % 1 != 0).any():
        raise ValueError("Invalid observed count targets")
    result = {}
    for label, cols in [("total_goals_mae", TARGETS[:2]), ("total_corners_mae", TARGETS[2:])]:
        ok = truth[list(cols)].notna().all(axis=1)
        result[label] = {
            "value": float(
                mean_absolute_error(
                    truth.loc[ok, list(cols)].sum(axis=1), predictions.loc[ok, list(cols)].sum(axis=1)
                )
            )
            if ok.any()
            else None,
            "n": int(ok.sum()),
        }
    for target, name in zip(TARGETS, PREDICTION_COLUMNS):
        ok = truth[target].notna()
        for metric, function in [("mae", mean_absolute_error), ("deviance", mean_poisson_deviance)]:
            result[f"{name}_{metric}"] = {
                "value": float(function(truth.loc[ok, target], predictions.loc[ok, target]))
                if ok.any()
                else None,
                "n": int(ok.sum()),
            }
    return result


def relative_score(candidate: dict, baseline: dict) -> float:
    """Internal equal-weight average ratio; the employer supplies no weighting."""
    ratios = []
    if set(candidate) != set(METRIC_NAMES) or set(baseline) != set(METRIC_NAMES):
        raise ValueError("Expected all ten metrics")
    for name in METRIC_NAMES:
        c, b = candidate[name], baseline[name]
        if c["n"] != b["n"] or c["n"] <= 0 or c["value"] is None or b["value"] is None:
            raise ValueError(f"No comparable observations for {name}")
        if not np.isfinite([c["value"], b["value"]]).all() or b["value"] <= 0:
            raise ValueError(f"Undefined metric ratio for {name}")
        ratios.append(c["value"] / b["value"])
    return float(np.mean(ratios))
