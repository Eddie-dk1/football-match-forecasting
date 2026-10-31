import numpy as np
import pandas as pd
import pytest
from conftest import make_matches

from quant_football.baseline import build_last_features
from quant_football.config import TARGETS, AvailabilityPolicy
from quant_football.features import build_features
from quant_football.forecasting import train_predict, write_submission
from quant_football.modeling import PoissonConfig, fit_poisson
from quant_football.scoring import METRIC_NAMES, relative_score, score_predictions


@pytest.fixture
def modeling_data():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(100):
        y = rng.poisson([1.7, 1.2, 5.3, 4.2]).astype(float)
        if i % 7 == 0:
            y[2] = np.nan
        if i % 9 == 0:
            y[3] = np.nan
        rows.append(
            {
                "match_id": i + 1,
                "kickoff_utc": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=3 * i),
                "home_team_id": i % 6 + 1,
                "away_team_id": (i + 2) % 6 + 1,
                **dict(zip(TARGETS, y)),
            }
        )
    history = make_matches(rows)
    cutoff = history.kickoff_utc.iloc[75]
    return history, cutoff


def test_metrics_hand_computed_masks_and_zero_truth():
    idx = pd.Index([1, 2, 3], name="match_id")
    truth = pd.DataFrame(
        [[0, 2, 4, 5], [2, np.nan, 1, np.nan], [3, 1, np.nan, 2]], index=idx, columns=TARGETS
    )
    pred = pd.DataFrame([[1, 2, 3, 6], [3, 4, 2, 1], [2, 2, 4, 3]], index=idx, columns=TARGETS)
    scores = score_predictions(truth, pred)
    assert set(scores) == set(METRIC_NAMES)
    assert scores["total_goals_mae"] == {"value": 0.5, "n": 2}
    assert scores["total_corners_mae"] == {"value": 0.0, "n": 1}
    assert scores["home_goals_mae"]["value"] == 1
    expected = 2 * (1 + (2 * np.log(2 / 3) - 2 + 3) + (3 * np.log(3 / 2) - 3 + 2)) / 3
    assert scores["home_goals_deviance"]["value"] == pytest.approx(expected)
    empty = score_predictions(truth * np.nan, pred)
    assert all(v == {"value": None, "n": 0} for v in empty.values())
    positive = {k: {"value": 2.0, "n": 3} for k in METRIC_NAMES}
    assert relative_score(positive, positive) == 1


@pytest.mark.parametrize("bad", ["order", "missing", "zero", "nan", "infinity"])
def test_metrics_refuse_invalid_submission(bad):
    truth = pd.DataFrame([[1] * 4, [2] * 4], index=pd.Index([1, 2], name="match_id"), columns=TARGETS)
    pred = truth.astype(float).copy()
    if bad == "order":
        pred = pred.iloc[::-1]
    elif bad == "missing":
        pred = pred.iloc[:1]
    else:
        pred.iloc[0, 0] = {"zero": 0.0, "nan": np.nan, "infinity": np.inf}[bad]
    with pytest.raises(ValueError):
        score_predictions(truth, pred)


def test_last_match_boundary_missing_orientation_and_order(history):
    schedule = make_matches([{"match_id": 99, "kickoff_utc": "2025-01-03T12:00:00Z"}])
    at_boundary = build_last_features(schedule, history)
    assert at_boundary.X.iloc[0].h_gf == 2
    schedule.kickoff_utc += pd.Timedelta(nanoseconds=1)
    after = build_last_features(schedule, history)
    assert after.X.iloc[0].h_gf == 0
    assert np.isnan(after.X.iloc[0].h_cf)
    pd.testing.assert_frame_equal(after.X, build_last_features(schedule, history.iloc[::-1]).X)
    future = history.copy()
    mask = future.kickoff_utc >= pd.Timestamp("2025-01-03T12:00Z")
    future.loc[mask, list(TARGETS)] += 100
    pd.testing.assert_frame_equal(after.X, build_last_features(schedule, future).X)


@pytest.mark.parametrize("kind", ["history_glm", "last_match"])
def test_fit_excludes_unavailable_rows_and_freezes_preprocessor(modeling_data, kind):
    history, cutoff = modeling_data
    builder = build_features if kind == "history_glm" else build_last_features
    features = builder(history, history)
    config = PoissonConfig(kind=kind, alpha=0.01)
    model = fit_poisson(features, history, cutoff, config)
    eligible = history.kickoff_utc + AvailabilityPolicy().lag < cutoff
    for t in TARGETS:
        assert model.fit_report[t]["n_train"] == int((eligible & history[t].notna()).sum())
        assert model.fit_report[t]["max_label_available_at"] < cutoff
    changed = history.copy()
    changed.loc[~eligible, list(TARGETS)] += 100
    changed.loc[~eligible, "league_id"] = 99999
    changed_features = builder(changed, changed)
    second = fit_poisson(changed_features, changed, cutoff, config)
    assert model.fingerprint() == second.fingerprint()
    # Vocabulary, imputation and scaling ignore the entire unavailable suffix.
    test_ids = pd.Index(history.loc[~eligible, "match_id"], name="match_id")
    test = changed_features.subset(test_ids)
    before = model.fingerprint()
    prediction = model.predict(test)
    assert model.fingerprint() == before
    assert np.isfinite(prediction.to_numpy()).all()
    assert (prediction > 0).all().all()
    pd.testing.assert_frame_equal(prediction, model.predict(test))
    with pytest.raises(ValueError, match="precedes"):
        model.predict(features)


@pytest.mark.parametrize("kind", ["history_glm", "last_match"])
def test_end_to_end_future_target_perturbation_and_submission(modeling_data, tmp_path, kind):
    history, cutoff = modeling_data
    train = history[history.kickoff_utc < cutoff]
    future = history[history.kickoff_utc >= cutoff].copy()
    config = PoissonConfig(kind=kind, alpha=0.01)
    pred, report = train_predict(train, train, future, future, cutoff, config)
    hidden = future.copy()
    for t in TARGETS:
        hidden[t] = "not known to forecast"
    # Own labels on the schedule are ignored even if not numeric.
    same, same_report = train_predict(train, train, hidden, future, cutoff, config)
    pd.testing.assert_frame_equal(pred, same)
    assert report["fingerprint_before"] == report["fingerprint_after"] == same_report["fingerprint_before"]
    altered = future.copy()
    altered.loc[altered.index[-1], list(TARGETS)] += 100
    other, _ = train_predict(train, train, future, altered, cutoff, config)
    pd.testing.assert_frame_equal(pred, other)  # last outcome cannot affect any earlier forecast
    changed = future.copy()
    changed.loc[changed.index[0], list(TARGETS)] += 3
    updated, updated_report = train_predict(train, train, future, changed, cutoff, config)
    pd.testing.assert_series_equal(pred.iloc[0], updated.iloc[0])
    assert updated_report["fingerprint_before"] == report["fingerprint_before"]
    assert not np.allclose(pred.iloc[1:].to_numpy(), updated.iloc[1:].to_numpy())
    path = tmp_path / "predictions_spring.parquet"
    output_report = write_submission(path, pred, pred.index)
    assert output_report["rows"] == len(future)
    assert pd.read_parquet(path).match_id.tolist() == future.match_id.tolist()
    with pytest.raises(ValueError, match="at/after"):
        train_predict(history, history, future, future, cutoff, config)


def test_fit_rejects_bad_alignment_and_empty_target(modeling_data):
    history, cutoff = modeling_data
    features = build_features(history, history)
    with pytest.raises(ValueError, match="order"):
        fit_poisson(features, history.iloc[::-1], cutoff)
    history.home_corners_90 = np.nan
    with pytest.raises(ValueError, match="Insufficient"):
        fit_poisson(features, history, cutoff)


def test_predict_never_calls_fit_and_training_order_is_stable(modeling_data, monkeypatch):
    from sklearn.pipeline import Pipeline

    history, cutoff = modeling_data
    config = PoissonConfig(alpha=0.01)
    first = fit_poisson(build_features(history, history), history, cutoff, config)
    shuffled = history.sample(frac=1, random_state=123)
    second = fit_poisson(build_features(shuffled, shuffled), shuffled, cutoff, config)
    assert first.fingerprint() == second.fingerprint()
    future = history[history.kickoff_utc >= cutoff]
    features = build_features(future, history)

    def prohibited(*args, **kwargs):
        raise AssertionError("fit called during prediction")

    monkeypatch.setattr(Pipeline, "fit", prohibited)
    prediction = first.predict(features)
    assert len(prediction) == len(future)


def test_model_training_exact_release_boundary_is_excluded(modeling_data):
    history, _ = modeling_data
    cutoff = history.kickoff_utc.iloc[75] + AvailabilityPolicy().lag
    model = fit_poisson(build_features(history, history), history, cutoff, PoissonConfig(alpha=0.01))
    assert model.fit_report[TARGETS[0]]["n_train"] == 75
    assert model.fit_report[TARGETS[0]]["max_label_available_at"] < cutoff


def test_forecast_rejects_inconsistent_future_metadata(modeling_data):
    history, cutoff = modeling_data
    train = history[history.kickoff_utc < cutoff]
    future = history[history.kickoff_utc >= cutoff].copy()
    broken = future.copy()
    broken.loc[broken.index[0], "home_team_id"] = 9999
    with pytest.raises(ValueError, match="metadata disagree"):
        train_predict(train, train, broken, future, cutoff, PoissonConfig(alpha=0.01))


def test_constant_count_poisson_fit_matches_analytic_mean(modeling_data):
    history, cutoff = modeling_data
    # All historical predictors are constant; the optimum is the observed count mean.
    for target, value in zip(TARGETS, [2.0, 1.0, 6.0, 4.0]):
        history[target] = value
    features = build_last_features(history, history)
    features.X.loc[:, :] = 0.0
    model = fit_poisson(features, history, cutoff, PoissonConfig(kind="last_match", alpha=0.01))
    future_ids = history.loc[history.kickoff_utc >= cutoff, "match_id"]
    test = features.subset(future_ids)
    prediction = model.predict(test)
    np.testing.assert_allclose(
        prediction.to_numpy(), np.tile([2.0, 1.0, 6.0, 4.0], (len(test.X), 1)), rtol=1e-7
    )


def test_independent_numpy_metrics_match_hand_and_sklearn():
    from reference_metrics import numpy_scores

    rng = np.random.default_rng(17)
    y = rng.poisson(2.0, size=(30, 4)).astype(float)
    y[::3, 1] = np.nan
    y[::5, 2:] = np.nan
    index = pd.Index(range(1, 31), name="match_id")
    truth = pd.DataFrame(y, index=index, columns=TARGETS)
    pred = pd.DataFrame(rng.uniform(0.05, 6.0, size=(30, 4)), index=index, columns=TARGETS)
    measured, independent = score_predictions(truth, pred), numpy_scores(truth, pred)
    for name in METRIC_NAMES:
        assert measured[name]["n"] == independent[name]["n"]
        assert measured[name]["value"] == pytest.approx(independent[name]["value"], rel=1e-12)


def test_numeric_round_metadata_can_have_negative_qualifier_codes(modeling_data):
    history, cutoff = modeling_data
    history.loc[history.index[::3], "round"] = -2.0
    model = fit_poisson(build_features(history, history), history, cutoff, PoissonConfig(alpha=0.01))
    test = history[history.kickoff_utc >= cutoff]
    prediction = model.predict(build_features(test, history))
    assert np.isfinite(prediction.to_numpy()).all()


def test_nonfinite_optimizer_gradient_cannot_be_accepted(modeling_data, monkeypatch):
    from sklearn.linear_model import PoissonRegressor

    history, cutoff = modeling_data
    monkeypatch.setattr(PoissonRegressor, "predict", lambda self, X: np.full(len(X), np.nan))
    with pytest.raises(RuntimeError, match="gradient"):
        fit_poisson(build_last_features(history, history), history, cutoff, PoissonConfig(kind="last_match"))
