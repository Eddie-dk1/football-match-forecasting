import numpy as np
import pandas as pd
import pytest

from quant_football.config import METADATA_COLUMNS, TARGETS, AvailabilityPolicy, FeatureConfig
from quant_football.contracts import (
    DataContractError,
    assert_core_open_consistency,
    merge_histories,
    normalize_matches,
    schedule_only,
    utc_timestamp,
)


def test_utc_and_normalization_are_non_mutating(history):
    raw = history.copy(deep=True)
    raw.kickoff_utc = raw.kickoff_utc.dt.tz_localize(None)
    before = raw.copy(deep=True)
    got = normalize_matches(raw, "open")
    pd.testing.assert_frame_equal(raw, before)
    assert str(got.kickoff_utc.dtype) == "datetime64[ns, UTC]"
    pd.testing.assert_series_equal(got.kickoff_utc, history.kickoff_utc.dt.as_unit("ns"))


def test_schedule_discards_current_outcomes_and_untrusted_extra_columns(history):
    raw = history.copy()
    for c in TARGETS:
        raw[c] = "not even a numeric outcome"
    raw["xg_h"] = float("inf")
    raw["future_event_count"] = 100_000
    schedule = schedule_only(raw)
    assert list(schedule.columns) == list(METADATA_COLUMNS)
    assert set(TARGETS).isdisjoint(schedule.columns)


def test_minimal_schedule_is_supported(history):
    minimal = history[["match_id", "kickoff_utc", "league_id", "home_team_id", "away_team_id"]]
    result = schedule_only(minimal)
    assert result.league_country.isna().all()
    assert result["round"].isna().all()


@pytest.mark.parametrize(
    "column,value",
    [
        ("home_goals_90", -1.0),
        ("away_goals_90", 0.5),
        ("home_corners_90", np.inf),
        ("away_corners_90", -np.inf),
    ],
)
def test_invalid_targets_fail(history, column, value):
    history.loc[0, column] = value
    with pytest.raises(DataContractError):
        normalize_matches(history, "open")


def test_zero_and_missing_are_distinct(history):
    result = normalize_matches(history, "open")
    assert result.loc[1, "home_goals_90"] == 0
    assert pd.isna(result.loc[1, "home_corners_90"])


@pytest.mark.parametrize("name", ["match_id", "league_id", "home_team_id", "away_team_id"])
def test_float_identifiers_fail(history, name):
    history[name] = history[name].astype(float)
    with pytest.raises(DataContractError):
        schedule_only(history)


def test_duplicate_ids_and_duplicate_columns_fail(history):
    with pytest.raises(DataContractError, match="Duplicate match_id"):
        normalize_matches(pd.concat([history, history.iloc[:1]]), "open")
    duplicate_columns = pd.concat([history, history[["match_id"]]], axis=1)
    with pytest.raises(DataContractError, match="Duplicate column"):
        schedule_only(duplicate_columns)


def test_missing_numeric_and_invalid_dates_fail(history):
    for bad in [None, "not-a-time", 1000]:
        raw = history.copy()
        raw["kickoff_utc"] = raw.kickoff_utc.astype(object)
        raw.loc[0, "kickoff_utc"] = bad
        with pytest.raises(DataContractError):
            normalize_matches(raw, "open")


def test_same_team_both_sides_fails(history):
    history.loc[0, "away_team_id"] = history.loc[0, "home_team_id"]
    with pytest.raises(DataContractError):
        schedule_only(history)


def test_core_open_consistency(history):
    core = normalize_matches(history.drop(columns="extra_time"), "core")
    normal = normalize_matches(history, "open")
    assert_core_open_consistency(core, normal)
    core.loc[0, "home_goals_90"] = 10
    with pytest.raises(DataContractError, match="disagree"):
        assert_core_open_consistency(core, normal)
    with pytest.raises(DataContractError, match="absent"):
        assert_core_open_consistency(core, normal.iloc[1:])


def test_overlapping_history_is_idempotent(history):
    result = merge_histories([history, history.iloc[:2], history])
    expected = normalize_matches(history, "open").sort_values("match_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(result.sort_values("match_id").reset_index(drop=True), expected)


def test_history_conflicts_fail_and_missing_may_be_complemented(history):
    duplicate = history.iloc[:1].copy()
    duplicate.loc[0, "home_goals_90"] = 9
    with pytest.raises(DataContractError, match="Conflicting"):
        merge_histories([history, duplicate])
    duplicate.loc[0, "home_goals_90"] = np.nan
    result = merge_histories([history, duplicate])
    assert result.set_index("match_id").loc[1, "home_goals_90"] == 2


def test_boolean_and_metadata_contract(history):
    history.extra_time = history.extra_time.astype(int)
    with pytest.raises(DataContractError, match="boolean"):
        normalize_matches(history, "open")
    history.extra_time = False
    history.loc[0, "round"] = np.inf
    with pytest.raises(DataContractError, match="metadata"):
        schedule_only(history)


@pytest.mark.parametrize("lag", [0, -1, np.inf, np.nan])
def test_invalid_availability_policy(lag):
    with pytest.raises(ValueError):
        AvailabilityPolicy(lag)


def test_feature_config_validation():
    for kwargs in [
        {"windows": (5, 5)},
        {"windows": (10, 5)},
        {"windows": ()},
        {"windows": (1.5,)},
        {"team_prior_strength": 0},
    ]:
        with pytest.raises(ValueError):
            FeatureConfig(**kwargs)
    assert utc_timestamp("2025-01-01").tzinfo is not None
