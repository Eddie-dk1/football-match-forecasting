import math

import numpy as np
import pandas as pd
import pytest
from conftest import make_matches

from quant_football.config import TARGETS, FeatureConfig
from quant_football.features import CausalFeatureEngine, build_features, feature_columns, training_labels


def query(at, mid=100, home=1, away=2):
    return make_matches([{"match_id": mid, "kickoff_utc": at, "home_team_id": home, "away_team_id": away}])


def test_strict_release_boundary_to_nanosecond(history):
    before = build_features(query("2025-01-02T11:59:59.999999999Z"), history)
    equal = build_features(query("2025-01-02T12:00:00Z"), history)
    after = build_features(query("2025-01-02T12:00:00.000000001Z"), history)
    assert before.X.loc[100, "h_matches_seen"] == 0
    assert equal.X.loc[100, "h_matches_seen"] == 0
    assert after.X.loc[100, "h_matches_seen"] == 1
    assert after.provenance.loc[100, "history_max_available_at"] == pd.Timestamp("2025-01-02T12:00:00Z")
    assert after.provenance.loc[100, "kickoff_utc"].nanosecond == 1


def test_timestamp_provenance_never_rounds_through_float64():
    h = query("2025-01-01T12:00:00.000000019Z", mid=1)
    t = "2025-01-02T12:00:00.000000020Z"
    result = build_features(query(t), h)
    assert result.provenance.loc[100, "history_max_available_at"].nanosecond == 19
    assert result.provenance.loc[100, "kickoff_utc"].nanosecond == 20


def test_current_match_outcome_and_stats_cannot_change_own_features(history):
    original = build_features(history, history)
    for mid in history.match_id:
        altered = history.copy()
        mask = altered.match_id.eq(mid)
        altered.loc[mask, list(TARGETS)] = [19, 17, 25, 23]
        altered["xg_h"] = 1000.0
        changed = build_features(altered, altered)
        pd.testing.assert_series_equal(original.X.loc[mid], changed.X.loc[mid])
        pd.testing.assert_series_equal(original.provenance.loc[mid], changed.provenance.loc[mid])


def test_future_outcomes_and_appended_future_rows_cannot_change_past(history):
    s = query("2025-01-03T13:00:00Z")
    a = build_features(s, history)
    altered = history.copy()
    altered.loc[altered.kickoff_utc >= pd.Timestamp("2025-01-03T12:00:00Z"), list(TARGETS)] = [20, 21, 22, 23]
    altered = pd.concat([altered, query("2027-01-01T00:00:00Z", mid=200)])
    b = build_features(s, altered)
    pd.testing.assert_frame_equal(a.X, b.X)
    pd.testing.assert_frame_equal(a.provenance, b.provenance)


def test_full_history_equals_available_prefix(history):
    for at in ["2024-12-01", "2025-01-02T13:00:00Z", "2025-01-04T13:00:00Z", "2025-02-01"]:
        s = query(at)
        prefix = history[(history.kickoff_utc + pd.Timedelta(hours=24)) < pd.Timestamp(at, tz="UTC")]
        full = build_features(s, history)
        clipped = build_features(s, prefix)
        pd.testing.assert_frame_equal(full.X, clipped.X)
        pd.testing.assert_frame_equal(full.provenance, clipped.provenance)


def test_source_and_schedule_permutations_are_invariant(history):
    original = build_features(history, history)
    altered = build_features(history.sample(frac=1, random_state=9), history.sample(frac=1, random_state=10))
    pd.testing.assert_frame_equal(original.X.sort_index(), altered.X.sort_index())
    pd.testing.assert_frame_equal(original.provenance.sort_index(), altered.provenance.sort_index())


def test_source_index_and_match_ids_are_not_positionally_aligned(history):
    history.index = [50, 40, 30, 20, 10]
    order = history.iloc[[4, 0, 3, 1, 2]]
    result = build_features(order, history)
    assert result.X.index.tolist() == order.match_id.tolist()
    assert result.provenance.index.tolist() == order.match_id.tolist()


def test_home_away_direction_and_shrinkage_are_mathematically_correct(history):
    result = build_features(query("2025-01-02T13:00:00Z"), history)
    row = result.X.loc[100]
    # After one 2:1 game each hierarchy level has 2 team observations, sum 3.
    p = 1.4
    for _ in range(4):
        p = (3 + 50 * p) / 52
    assert row.league_prior_gf == pytest.approx(p, rel=1e-6)
    assert row.h_gf_all_mean == pytest.approx((2 + 10 * p) / 11, rel=1e-6)
    assert row.a_ga_all_mean == pytest.approx(row.h_gf_all_mean)
    assert row.a_gf_all_mean == pytest.approx((1 + 10 * p) / 11, rel=1e-6)
    assert row.h_ga_all_mean == pytest.approx(row.a_gf_all_mean)
    assert row.h_venue_matches_seen == row.a_venue_matches_seen == 1
    assert row.h_rest_days == pytest.approx(25 / 24)


def test_missing_corners_do_not_become_zero_or_remove_goal_history(history):
    c = FeatureConfig(windows=(1, 2), venue_windows=(1, 2), ewm_half_lives=(1, 2))
    row = build_features(query("2025-01-03T13:00:00Z"), history, config=c).X.loc[100]
    assert row.h_matches_seen == 2
    assert row.h_gf_all_n == 2
    assert row.h_cf_all_n == 1
    assert row.h_cf_last1_n == 0
    assert row.h_cf_last2_n == 1
    assert row.h_cf_last1_mean == row.league_prior_cf
    assert row.h_cf_last1_mean > 0
    assert pd.isna(row.h_cf_last1_std)


def test_ewma_uses_normalized_weights_and_effective_sample_size(history):
    c = FeatureConfig(windows=(2,), venue_windows=(2,), ewm_half_lives=(1,))
    row = build_features(query("2025-01-03T13:00:00Z"), history, config=c).X.loc[100]
    # GF sequence is 2,0. Half-life 1 -> weights .5,1 -> mean 2/3.
    effective_n = 1.5**2 / (0.5**2 + 1)
    expected = ((2 / 3) * effective_n + 10 * row.league_prior_gf) / (effective_n + 10)
    assert row.h_gf_ewm1_effective_n == pytest.approx(effective_n)
    assert row.h_gf_ewm1_mean == pytest.approx(expected, rel=1e-6)
    # Only one observed corner value: missing values neither decay nor add zero.
    assert row.h_cf_ewm1_effective_n == 1


def test_unknown_teams_and_leagues_use_past_priors(history):
    s = query("2025-01-03T13:00:00Z", home=999, away=1000)
    s.league_id = 999
    s.league_country = None
    s.league_type = None
    row = build_features(s, history).X.loc[100]
    assert row.h_matches_seen == row.a_matches_seen == 0
    assert pd.isna(row.h_rest_days)
    for metric in ("gf", "ga", "cf", "ca"):
        assert row[f"h_{metric}_all_mean"] == row[f"league_prior_{metric}"]
        assert math.isfinite(row[f"h_{metric}_all_mean"])


def test_all_unknown_outcomes_are_not_released_as_a_completed_match(history):
    history.loc[0, list(TARGETS)] = np.nan
    result = build_features(query("2025-01-02T13:00:00Z"), history)
    assert result.X.loc[100, "h_matches_seen"] == 0
    assert result.provenance.loc[100, "history_rows_released"] == 0


def test_partial_results_are_kept_per_metric(history):
    history.loc[0, "home_goals_90"] = np.nan
    row = build_features(query("2025-01-02T13:00:00Z"), history).X.loc[100]
    assert row.h_gf_all_n == 0
    assert row.h_ga_all_n == 1
    assert row.h_cf_all_n == 1
    assert row.a_gf_all_n == 1
    assert row.a_ga_all_n == 0


def test_engine_is_idempotent_and_rejects_backward_and_mixed_batches(history):
    engine = CausalFeatureEngine(history)
    s = query("2025-01-04T13:00:00Z")
    a = engine.features_before(s)
    released = engine.released_matches
    b = engine.features_before(s)
    pd.testing.assert_frame_equal(a.X, b.X)
    assert engine.released_matches == released == 4
    with pytest.raises(ValueError, match="backward"):
        engine.features_before(query("2025-01-03T00:00:00Z"))
    with pytest.raises(ValueError, match="one kickoff"):
        engine.features_before(history)


def test_sequential_api_equals_complete_replay(history):
    engine = CausalFeatureEngine(history)
    pieces = [engine.features_before(group).X for _, group in history.groupby("kickoff_utc", sort=True)]
    expected = build_features(history, history).X
    pd.testing.assert_frame_equal(pd.concat(pieces).reindex(expected.index), expected)


def test_empty_schedule_and_history_have_stable_schema(history):
    a = build_features(history.iloc[:0], history)
    assert a.X.empty and a.provenance.empty
    assert list(a.X.columns) == feature_columns(FeatureConfig())
    b = build_features(history, history.iloc[:0])
    assert b.X.h_matches_seen.eq(0).all()
    assert b.provenance.history_max_available_at.isna().all()


def test_training_mask_respects_availability_not_just_kickoff():
    core = make_matches(
        [
            {"match_id": 1, "kickoff_utc": "2026-02-27T23:59:59.999999999Z"},
            {"match_id": 2, "kickoff_utc": "2026-02-28T00:00:00Z"},
            {"match_id": 3, "kickoff_utc": "2026-02-28T23:30:00Z"},
        ]
    )
    core.loc[0, "home_corners_90"] = np.nan
    y, mask = training_labels(core)
    assert mask.tolist() == [True, False, False]
    assert pd.isna(y.loc[1, "home_corners_90"])


def test_schedule_own_outcomes_can_be_hidden_or_arbitrary(history):
    original = build_features(history, history)
    schedule = history.copy()
    for c in TARGETS:
        schedule[c] = "HIDDEN"
    changed = build_features(schedule, history)
    pd.testing.assert_frame_equal(original.X, changed.X)


def test_extra_time_and_auxiliary_statistics_never_replace_regulation_targets(history):
    original = build_features(query("2025-01-02T13:00:00Z"), history)
    changed = history.copy()
    changed["extra_time"] = True
    changed["corn_h"] = 100.0
    changed["corn_a"] = 99.0
    changed["xg_h"] = 80.0
    changed["xg_a"] = 70.0
    actual = build_features(query("2025-01-02T13:00:00Z"), changed)
    pd.testing.assert_frame_equal(original.X, actual.X)


def test_venue_history_tracks_role_in_past_matches(history):
    row = build_features(query("2025-01-04T13:00:00Z", home=1, away=4), history).X.loc[100]
    assert row.h_matches_seen == 4
    assert row.h_venue_matches_seen == 3
    assert row.a_matches_seen == 1
    assert row.a_venue_matches_seen == 0
    assert row.a_gf_venue10_n == 0


def test_no_available_history_uses_only_declared_initial_constants(history):
    result = build_features(query("2023-01-01T00:00:00Z"), history).X.loc[100]
    assert result.h_gf_all_mean == pytest.approx(1.4)
    assert result.h_cf_all_mean == pytest.approx(4.7)
    assert result.h_matches_seen == 0


@pytest.mark.parametrize("seed", [3, 19, 42])
def test_randomized_prefix_invariance(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(35):
        home, away = rng.choice(np.arange(1, 9), size=2, replace=False)
        rows.append(
            {
                "match_id": i + 1,
                "kickoff_utc": pd.Timestamp("2025-01-01", tz="UTC")
                + pd.Timedelta(hours=int(rng.integers(0, 400))),
                "home_team_id": int(home),
                "away_team_id": int(away),
                "home_goals_90": float(rng.integers(0, 6)),
                "away_goals_90": float(rng.integers(0, 6)),
                "home_corners_90": np.nan if i % 4 == 0 else float(rng.integers(0, 12)),
            }
        )
    history = make_matches(rows)
    s = query("2025-01-11T00:00:00Z", home=3, away=8)
    prefix = history[(history.kickoff_utc + pd.Timedelta(hours=24)) < s.kickoff_utc.iloc[0]]
    a = build_features(s, history.sample(frac=1, random_state=seed))
    b = build_features(s, prefix)
    pd.testing.assert_frame_equal(a.X, b.X)
    pd.testing.assert_frame_equal(a.provenance, b.provenance)


def test_workload_windows_are_based_on_match_kickoff_and_released_history():
    at = pd.Timestamp("2025-02-01T12:00:00Z")
    rows = [
        {"match_id": i + 1, "kickoff_utc": at - pd.Timedelta(days=days)}
        for i, days in enumerate([29, 28, 14, 7, 2, 1, 0])
    ]
    h = make_matches(rows)
    row = build_features(query(at), h).X.loc[100]
    # The game exactly 24h ago is not yet released at the strict boundary.
    assert row.h_matches_seen == 5
    assert row.h_matches_last7d == 2
    assert row.h_matches_last14d == 3
    assert row.h_matches_last28d == 4
