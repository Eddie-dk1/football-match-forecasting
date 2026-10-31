"""End-to-end frozen-model replay, also used by the historical spring smoke test."""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .baseline import build_last_features
from .config import METADATA_COLUMNS, TRAIN_CUTOFF, AvailabilityPolicy
from .contracts import merge_histories, normalize_matches, schedule_only, utc_timestamp
from .features import build_features
from .io import write_parquet
from .modeling import PoissonConfig, fit_poisson
from .scoring import PREDICTION_COLUMNS, validate_predictions


def write_submission(path: Path, predictions: pd.DataFrame, expected_index: pd.Index) -> dict:
    validate_predictions(predictions, expected_index)
    output = predictions.set_axis(PREDICTION_COLUMNS, axis=1).reset_index()
    write_parquet(path, output)
    saved = pd.read_parquet(path)
    pd.testing.assert_frame_equal(saved, output)
    return {"rows": len(saved), "columns": saved.columns.tolist(), "parquet_roundtrip": "pass"}


def train_predict(
    training_core: pd.DataFrame,
    training_history: pd.DataFrame,
    forecast_schedule: pd.DataFrame,
    forecast_history: pd.DataFrame,
    cutoff=TRAIN_CUTOFF,
    config: PoissonConfig | None = None,
    policy: AvailabilityPolicy | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fit exactly once on training files, then replay a potentially labelled future.

    The future files are never given to any fit or transform.fit. Their outcomes
    are visible only via the feature engine's strict availability gate.
    """
    cutoff, policy = utc_timestamp(cutoff), policy or AvailabilityPolicy()
    training_core = normalize_matches(training_core, "core")
    training_history = normalize_matches(training_history, "open")
    if (training_core.kickoff_utc >= cutoff).any() or (training_history.kickoff_utc >= cutoff).any():
        raise ValueError("Training files contain matches at/after the frozen cutoff")
    from .contracts import assert_core_open_consistency

    assert_core_open_consistency(training_core, training_history)
    schedule = schedule_only(forecast_schedule)
    if (schedule.kickoff_utc < cutoff).any():
        raise ValueError("Forecast schedule starts before the frozen cutoff")
    if schedule.match_id.isin(training_core.match_id).any():
        raise ValueError("Training and forecast core IDs overlap")
    config = config or PoissonConfig()
    training_rows = training_history if config.kind == "last_match" else training_core
    feature_builder = build_last_features if config.kind == "last_match" else build_features
    train_features = feature_builder(training_rows, training_history, policy)
    model = fit_poisson(train_features, training_rows, cutoff, config, policy)
    before = model.fingerprint()
    del train_features
    history = merge_histories([training_history, forecast_history])
    # Check only forecast metadata. Own target columns may be hidden or absent.
    history_schedule = schedule_only(history).set_index("match_id")
    indexed_schedule = schedule.set_index("match_id")
    if not indexed_schedule.index.isin(history_schedule.index).all():
        raise ValueError("Forecast core contains match_id absent from combined open history")
    matched = history_schedule.reindex(indexed_schedule.index)
    for column in METADATA_COLUMNS[1:]:
        a, b = indexed_schedule[column], matched[column]
        if not (a.eq(b) | (a.isna() & b.isna())).all():
            raise ValueError(f"Forecast schedule/open metadata disagree: {column}")
    features = feature_builder(schedule, history, policy)
    predictions = model.predict(features)
    if before != model.fingerprint():
        raise RuntimeError("Frozen model changed during forecast replay")
    return predictions, {
        "cutoff_utc": cutoff,
        "fit_report": model.fit_report,
        "config": asdict(model.config),
        "feature_metadata_sha256": model.feature_metadata.fingerprint(),
        "fingerprint_before": before,
        "fingerprint_after": model.fingerprint(),
        "model_fit_calls": 4,
        "test_period_fit_calls": 0,
        "forecast_rows": len(predictions),
        "max_history_available_at": features.provenance.history_max_available_at.max(),
    }


def train_predict_boost(
    training_core,
    training_history,
    forecast_schedule,
    forecast_history,
    cutoff=TRAIN_CUTOFF,
    config=None,
    policy=None,
):
    """CatBoost counterpart: four fits exclusively before merging forecast files."""
    from .boosted import BoostConfig, fit_boost
    from .contracts import assert_core_open_consistency
    from .open_stats import build_open_features

    cutoff, policy, config = utc_timestamp(cutoff), policy or AvailabilityPolicy(), config or BoostConfig()
    training_core = normalize_matches(training_core, "core")
    training_history = normalize_matches(training_history, "open")
    if (training_core.kickoff_utc >= cutoff).any() or (training_history.kickoff_utc >= cutoff).any():
        raise ValueError("Training files contain matches at/after frozen cutoff")
    assert_core_open_consistency(training_core, training_history)
    schedule = schedule_only(forecast_schedule)
    if (schedule.kickoff_utc < cutoff).any():
        raise ValueError("Forecast schedule precedes frozen cutoff")
    if schedule.match_id.isin(training_core.match_id).any():
        raise ValueError("Training and forecast core IDs overlap")
    builder = build_open_features if config.open_stats else build_features
    train_features = builder(training_history, training_history, policy)
    model = fit_boost(train_features, training_core, training_history, cutoff, config, policy)
    before = model.fingerprint()
    del train_features
    # Every fit has completed before these files are read by the history engine.
    history = merge_histories([training_history, forecast_history])
    indexed, matched = schedule.set_index("match_id"), schedule_only(history).set_index("match_id")
    if not indexed.index.isin(matched.index).all():
        raise ValueError("Forecast core IDs absent from combined open history")
    matched = matched.reindex(indexed.index)
    for column in METADATA_COLUMNS[1:]:
        a, b = indexed[column], matched[column]
        if not (a.eq(b) | (a.isna() & b.isna())).all():
            raise ValueError(f"Forecast schedule/open metadata disagree: {column}")
    features = builder(schedule, history, policy)
    predictions = model.predict(features)
    if model.fingerprint() != before:
        raise RuntimeError("Frozen CatBoost changed during replay")
    return predictions, {
        "cutoff_utc": cutoff,
        "fit_report": model.fit_report,
        "config": asdict(config),
        "feature_metadata_sha256": model.feature_metadata.fingerprint(),
        "fingerprint_before": before,
        "fingerprint_after": model.fingerprint(),
        "model_fit_calls": 4,
        "test_period_fit_calls": 0,
        "forecast_rows": len(predictions),
    }
