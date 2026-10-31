import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import make_matches

from quant_football import stage6_model
from quant_football.boosted import BoostConfig
from quant_football.config import TARGETS, AvailabilityPolicy
from quant_football.feature_contract import FeatureMetadata
from quant_football.features import FeatureDataset, build_features
from quant_football.forecasting import write_submission
from quant_football.modeling import PoissonConfig
from quant_football.open_stats import STAT_INPUTS, build_open_features
from quant_football.stage6_model import (
    FinalConfig,
    blend_predictions,
    fit_final_model,
    train_predict_final,
)


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(814)
    h = make_matches(
        [
            {
                "match_id": i + 1,
                "kickoff_utc": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=3 * i),
                "home_team_id": i % 6 + 1,
                "away_team_id": (i + 2) % 6 + 1,
                **dict(zip(TARGETS, rng.poisson([1.5, 1.2, 5, 4]).astype(float), strict=True)),
            }
            for i in range(64)
        ]
    )
    for i, name in enumerate(STAT_INPUTS):
        h[name] = (h.index % (i + 3)).astype(float)
    h.loc[h.index[::9], ["home_corners_90", "away_corners_90"]] = np.nan
    return h, h.kickoff_utc.iloc[54]


def config(weight=0.25):
    return FinalConfig(
        BoostConfig(depth=2, iterations=6, l2_leaf_reg=30, cohort="all_open", open_stats=True),
        glm_weight=weight,
    )


@pytest.fixture(scope="module")
def fitted(synthetic):
    h, cutoff = synthetic
    core = h.iloc[::2].copy()
    cb, glm = build_open_features(h, h), build_features(core, h)
    model = fit_final_model(cb, glm, core, h, cutoff, config())
    ids = core.loc[core.kickoff_utc >= cutoff, "match_id"]
    return model, cb.subset(ids), glm.subset(ids)


def test_config_json_roundtrip_and_strict_schema():
    cfg = config()
    encoded = json.loads(json.dumps(cfg.to_dict(), allow_nan=False))
    assert FinalConfig.from_dict(encoded) == cfg
    assert cfg.glm == PoissonConfig(alpha=0.01)
    bad_values = []
    for key in encoded:
        bad = copy.deepcopy(encoded)
        del bad[key]
        bad_values.append(bad)
    bad_values.append({**encoded, "extra": 1})
    for component, key, value in (
        ("catboost", "extra", 1),
        ("catboost", "seed", True),
        ("catboost", "l2_leaf_reg", True),
        ("catboost", "depth", 2.0),
        ("catboost", "cohort", "core"),
        ("catboost", "open_stats", False),
        ("glm", "kind", "last_match"),
        ("glm", "max_iter", True),
        ("glm", "alpha", "0.01"),
    ):
        bad = copy.deepcopy(encoded)
        bad[component][key] = value
        bad_values.append(bad)
    for value in (True, "0.2", None, float("nan"), float("inf"), -0.1, 1.1):
        bad_values.append({**encoded, "glm_weight": value})
    for bad in bad_values:
        with pytest.raises(ValueError):
            FinalConfig.from_dict(bad)
    with pytest.raises(ValueError, match="non-event"):
        FinalConfig(object())


def test_blend_exact_endpoints_and_alignment():
    index = pd.Index([4, 2], name="match_id")
    cb = pd.DataFrame([[1, 2, 4, 6], [2, 1, 6, 4]], index=index, columns=TARGETS, dtype=float)
    glm = cb * 1.3
    pd.testing.assert_frame_equal(blend_predictions(cb, None, 0), cb, check_exact=True)
    pd.testing.assert_frame_equal(blend_predictions(cb, glm, 0), cb, check_exact=True)
    pd.testing.assert_frame_equal(blend_predictions(cb, glm, 1), glm, check_exact=True)
    pd.testing.assert_frame_equal(blend_predictions(cb, glm, 0.25), cb * 0.75 + glm * 0.25)
    for bad in (True, np.bool_(True), "0.2", [0.2], np.nan, np.inf, -0.1, 1.1):
        with pytest.raises(ValueError, match="weight"):
            blend_predictions(cb, glm, bad)
    for bad in (
        glm.iloc[::-1],
        glm.iloc[:1],
        glm.rename_axis("wrong_name"),
        glm.set_axis(pd.Index([4.0, 2.0], name="match_id")),
        glm.rename(columns={TARGETS[0]: "home_goals"}),
        glm * 0,
    ):
        with pytest.raises(ValueError):
            blend_predictions(cb, bad, 0.25)
    bad = glm.copy()
    bad.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        blend_predictions(cb, bad, 0)
    with pytest.raises(ValueError, match="requires"):
        blend_predictions(cb, None, 0.25)


@pytest.mark.parametrize("weight", [0, 0.25])
def test_api_all_fits_precede_future_merge_and_json_audit(synthetic, monkeypatch, weight):
    from catboost import CatBoostRegressor
    from sklearn.linear_model import PoissonRegressor

    h, cutoff = synthetic
    old, future = h[h.kickoff_utc < cutoff], h[h.kickoff_utc >= cutoff]
    core = old.iloc[::2]
    calls = {"cb": 0, "glm": 0}
    original_cb, original_glm = CatBoostRegressor.fit, PoissonRegressor.fit
    original_merge = stage6_model.merge_histories

    def cb_fit(self, *args, **kwargs):
        calls["cb"] += 1
        return original_cb(self, *args, **kwargs)

    def glm_fit(self, *args, **kwargs):
        calls["glm"] += 1
        return original_glm(self, *args, **kwargs)

    def merge(frames):
        assert calls == {"cb": 4, "glm": 4 if weight else 0}
        return original_merge(frames)

    monkeypatch.setattr(CatBoostRegressor, "fit", cb_fit)
    monkeypatch.setattr(PoissonRegressor, "fit", glm_fit)
    monkeypatch.setattr(stage6_model, "merge_histories", merge)
    schedule = future.copy()
    schedule[TARGETS[0]] = "hidden"
    schedule["shots_h"] = "hidden"
    pred, audit = train_predict_final(core, old, schedule, future, cutoff, config(weight))
    assert audit["model_fit_calls"] == (8 if weight else 4)
    assert audit["test_period_fit_calls"] == 0
    assert audit["forecast_rows"] == len(future)
    assert audit["fingerprint_before"] == audit["fingerprint_after"]
    assert audit == json.loads(json.dumps(audit, allow_nan=False))
    assert audit["components"]["catboost"]["feature_metadata"]["builder"] == "history_stats_v1"
    assert audit["components"]["catboost"]["fit_report"][TARGETS[0]]["n_core"] < len(old)
    if weight:
        glm = audit["components"]["glm"]
        assert glm["feature_metadata"]["builder"] == "history"
        assert len(glm["feature_metadata"]["columns"]) == 185
        assert glm["fit_report"][TARGETS[0]]["n_train"] == len(core)
    else:
        assert audit["components"]["glm"] is None
    assert np.isfinite(pred.to_numpy()).all() and (pred.to_numpy() > 0).all()


def test_frozen_predict_separate_metadata_and_export(fitted, monkeypatch, tmp_path):
    from catboost import CatBoostRegressor
    from sklearn.linear_model import PoissonRegressor

    model, cb, glm = fitted
    before = model.fingerprint()

    def forbidden(*args, **kwargs):
        raise AssertionError("fit during predict")

    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)
    monkeypatch.setattr(PoissonRegressor, "fit", forbidden)
    prediction = model.predict(cb, glm)
    expected = blend_predictions(model.cb.predict(cb), model.glm.predict(glm), 0.25)
    pd.testing.assert_frame_equal(prediction, expected, check_exact=True)
    assert model.fingerprint() == before
    with pytest.raises(ValueError, match="separate"):
        model.predict(cb)
    with pytest.raises(ValueError, match="metadata"):
        model.predict(cb, cb)
    with pytest.raises(ValueError, match="order"):
        model.predict(cb, glm.subset(glm.X.index[::-1]))
    with pytest.raises(ValueError, match="metadata"):
        model.predict(glm, glm)
    bad = replace(cb, metadata=replace(cb.metadata, policy=AvailabilityPolicy(1)))
    with pytest.raises(ValueError, match="metadata"):
        model.predict(bad, glm)
    output = tmp_path / "predictions.parquet"
    write_submission(output, prediction, prediction.index)
    saved = pd.read_parquet(output)
    assert saved.columns.tolist() == ["match_id", *(t.removesuffix("_90") for t in TARGETS)]
    assert (saved.drop(columns="match_id").to_numpy() > 0).all()
    endpoint = replace(model, config=replace(model.config, glm_weight=1))
    pd.testing.assert_frame_equal(endpoint.predict(cb, glm), model.glm.predict(glm), check_exact=True)
    assert endpoint.fingerprint() != model.fingerprint()


def test_cutoff_boundary_and_unavailable_outcomes_invariance(synthetic):
    h, cutoff = synthetic
    # Even row 52 belongs to core, so equality must be excluded by both fits.
    h = h.copy()
    boundary_id = int(h.iloc[52].match_id)
    h.loc[h.index[52], "kickoff_utc"] = cutoff - pd.Timedelta(hours=24)
    core = h.iloc[::2].copy()
    assert boundary_id in core.match_id.to_numpy()
    assert h.loc[h.match_id.eq(boundary_id), list(TARGETS)].notna().all().all()
    first = fit_final_model(build_open_features(h, h), build_features(core, h), core, h, cutoff, config())
    changed = h.copy()
    mask = changed.kickoff_utc + AvailabilityPolicy().lag >= cutoff
    changed.loc[mask, list(TARGETS)] += 123
    changed.loc[mask, list(STAT_INPUTS)] += 123
    changed_core = changed.iloc[::2].copy()
    second = fit_final_model(
        build_open_features(changed, changed),
        build_features(changed_core, changed),
        changed_core,
        changed,
        cutoff,
        config(),
    )
    assert first.fingerprint() == second.fingerprint()
    for component, rows in ((first.cb, h), (first.glm, core)):
        eligible = rows.kickoff_utc + AvailabilityPolicy().lag < cutoff
        inclusive = rows.kickoff_utc + AvailabilityPolicy().lag <= cutoff
        for target in TARGETS:
            expected = rows.loc[eligible & rows[target].notna()].sort_values(["kickoff_utc", "match_id"])
            ids = expected.match_id.to_numpy(dtype="int64")
            audit = component.fit_report[target]
            assert boundary_id not in ids
            # Every boundary target is observed: an inclusive gate would add it.
            assert int((inclusive & rows[target].notna()).sum()) == len(ids) + 1
            assert audit["n_train"] == len(ids)
            assert audit["train_match_ids_sha256"] == hashlib.sha256(ids.tobytes()).hexdigest()
            assert audit["max_label_available_at"] < cutoff
    # Current/future outcomes cannot affect forecasts before their release.
    schedule = h.iloc[[54]]
    a = first.predict(build_open_features(schedule, h), build_features(schedule, h))
    b = first.predict(build_open_features(schedule, changed), build_features(schedule, changed))
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    with pytest.raises(ValueError, match="precedes"):
        first.predict(build_open_features(h.iloc[[0]], h), build_features(h.iloc[[0]], h))


def test_fit_rejects_statistics_reused_for_glm_and_arbitrary_history_columns(synthetic):
    h, cutoff = synthetic
    cb = build_open_features(h, h)
    with pytest.raises(ValueError, match="separate history"):
        fit_final_model(cb, cb, h, h, cutoff, config())
    glm = build_features(h, h)
    altered = glm.X.assign(arbitrary=0.0)
    wrong = FeatureDataset(
        altered,
        glm.provenance,
        FeatureMetadata.for_frame(altered, "history", glm.metadata.policy, glm.metadata.config),
    )
    with pytest.raises(ValueError, match="inventory"):
        fit_final_model(cb, wrong, h, h, cutoff, config())


def test_replay_rejects_training_future_and_schedule_mismatch(synthetic):
    h, cutoff = synthetic
    old, future = h[h.kickoff_utc < cutoff], h[h.kickoff_utc >= cutoff]
    with pytest.raises(ValueError, match="at/after"):
        train_predict_final(h, h, future, future, cutoff, config(0))
    with pytest.raises(ValueError, match="precedes"):
        train_predict_final(old, old, old.iloc[:1], future, cutoff, config(0))
    changed = future.copy()
    changed["league_name"] = "different"
    with pytest.raises(ValueError, match="metadata disagree"):
        train_predict_final(old, old, changed, future, cutoff, config(0))
