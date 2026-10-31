"""Frozen composition of the selected statistics CatBoost and history GLM."""

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from numbers import Real

import numpy as np
import pandas as pd

from .boosted import BoostConfig, FrozenBoost, fit_boost
from .config import METADATA_COLUMNS, TRAIN_CUTOFF, AvailabilityPolicy
from .contracts import (
    assert_core_open_consistency,
    merge_histories,
    normalize_matches,
    schedule_only,
    utc_timestamp,
)
from .features import FeatureDataset, build_features, feature_columns
from .io import _json_default
from .modeling import FrozenPoisson, PoissonConfig, fit_poisson
from .open_stats import build_open_features
from .scoring import validate_predictions


def _weight(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("GLM weight must be a finite real scalar in [0, 1]")
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("GLM weight must be a finite real scalar in [0, 1]")
    return float(value)


def _native(value):
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


@dataclass(frozen=True)
class FinalConfig:
    catboost: BoostConfig
    glm: PoissonConfig = field(default_factory=lambda: PoissonConfig(alpha=0.01))
    glm_weight: float = 0.0

    def __post_init__(self):
        # Exact types deliberately exclude event configs and ad hoc subclasses.
        if type(self.catboost) is not BoostConfig:
            raise ValueError("Final CatBoost must use a non-event BoostConfig")
        if not self.catboost.open_stats or self.catboost.cohort != "all_open":
            raise ValueError("Final CatBoost requires all_open statistics features")
        if type(self.glm) is not PoissonConfig or self.glm.kind != "history_glm":
            raise ValueError("Final GLM must use history_glm")
        object.__setattr__(self, "glm_weight", _weight(self.glm_weight))
        # Also validate native scalar types for a strict, round-trippable schema.
        self._check_scalars(asdict(self.catboost), asdict(self.glm))

    @staticmethod
    def _check_scalars(catboost, glm):
        for name in ("depth", "iterations", "seed"):
            if type(catboost[name]) is not int:
                raise ValueError(f"CatBoost {name} must be a JSON integer")
        for name in ("l2_leaf_reg", "learning_rate"):
            if type(catboost[name]) not in (int, float):
                raise ValueError(f"CatBoost {name} must be a JSON number")
        if type(catboost["cohort"]) is not str or type(catboost["open_stats"]) is not bool:
            raise ValueError("Invalid CatBoost JSON scalar types")
        if type(glm["kind"]) is not str or type(glm["max_iter"]) is not int:
            raise ValueError("Invalid GLM JSON scalar types")
        for name in ("alpha", "tol"):
            if type(glm[name]) not in (int, float):
                raise ValueError(f"GLM {name} must be a JSON number")

    def to_dict(self):
        return {
            "catboost": asdict(self.catboost),
            "glm": asdict(self.glm),
            "glm_weight": self.glm_weight,
        }

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {"catboost", "glm", "glm_weight"}:
            raise ValueError("Invalid final configuration schema")
        cb, glm = value["catboost"], value["glm"]
        if type(cb) is not dict or set(cb) != {f.name for f in fields(BoostConfig)}:
            raise ValueError("Invalid final CatBoost configuration schema")
        if type(glm) is not dict or set(glm) != {f.name for f in fields(PoissonConfig)}:
            raise ValueError("Invalid final GLM configuration schema")
        if type(value["glm_weight"]) not in (int, float):
            raise ValueError("GLM weight must be a JSON number")
        cls._check_scalars(cb, glm)
        return cls(BoostConfig(**cb), PoissonConfig(**glm), value["glm_weight"])


def blend_predictions(cb, glm, weight):
    """Blend conditional means with exact endpoints and no implicit alignment."""
    weight = _weight(weight)
    if not isinstance(cb, pd.DataFrame):
        raise ValueError("CatBoost predictions must be a DataFrame")
    validate_predictions(cb, cb.index)
    if glm is not None:
        if not isinstance(glm, pd.DataFrame):
            raise ValueError("GLM predictions must be a DataFrame")
        if (
            not glm.index.equals(cb.index)
            or glm.index.name != cb.index.name
            or glm.index.dtype != cb.index.dtype
        ):
            raise ValueError("Prediction match_id index type, name, order or coverage differs")
        validate_predictions(glm, cb.index)
    elif weight > 0:
        raise ValueError("Positive GLM weight requires GLM predictions")
    if weight == 0:
        return cb.copy(deep=True)
    if weight == 1:
        return glm.copy(deep=True)
    blended = pd.DataFrame(
        (1 - weight) * cb.to_numpy(dtype="float64") + weight * glm.to_numpy(dtype="float64"),
        index=cb.index,
        columns=cb.columns,
    )
    validate_predictions(blended, cb.index)
    return blended


@dataclass(frozen=True)
class FrozenFinal:
    cutoff: pd.Timestamp
    config: FinalConfig
    policy: AvailabilityPolicy
    cb: FrozenBoost
    glm: FrozenPoisson | None = None

    def __post_init__(self):
        if type(self.config) is not FinalConfig or not isinstance(self.policy, AvailabilityPolicy):
            raise ValueError("Final model requires its frozen configuration and policy")
        object.__setattr__(self, "cutoff", utc_timestamp(self.cutoff))
        if not isinstance(self.cb, FrozenBoost):
            raise ValueError("Final model requires a fitted CatBoost component")
        if (self.config.glm_weight > 0) != (self.glm is not None):
            raise ValueError("GLM component presence differs from its frozen weight")
        if self.glm is not None and not isinstance(self.glm, FrozenPoisson):
            raise ValueError("Invalid fitted GLM component")
        for component, config, builder in (
            (self.cb, self.config.catboost, "history_stats_v1"),
            (self.glm, self.config.glm, "history"),
        ):
            if component is None:
                continue
            if (
                component.cutoff != self.cutoff
                or component.policy != self.policy
                or component.config != config
            ):
                raise ValueError("Component cutoff/configuration/policy differs from final model")
            if component.feature_metadata.builder != builder:
                raise ValueError("Component feature builder differs from its required contract")
        if self.glm is not None and self.cb.feature_metadata.config != self.glm.feature_metadata.config:
            raise ValueError("Component history feature settings differ")

    def fingerprint(self):
        state = {
            "cutoff_utc": self.cutoff.isoformat(),
            "config": self.config.to_dict(),
            "availability_policy": asdict(self.policy),
            "catboost": self.cb.fingerprint(),
            "glm": None if self.glm is None else self.glm.fingerprint(),
        }
        return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @property
    def fit_report(self):
        def component_report(component):
            if component is None:
                return None
            return {
                "config": asdict(component.config),
                "fit_report": component.fit_report,
                "feature_metadata": component.feature_metadata.to_dict(),
                "feature_metadata_sha256": component.feature_metadata.fingerprint(),
                "model_fit_calls": 4,
            }

        return _native({"catboost": component_report(self.cb), "glm": component_report(self.glm)})

    def predict(self, cb_dataset: FeatureDataset, glm_dataset: FeatureDataset | None = None):
        cb_dataset.assert_valid()
        if self.glm is None:
            if glm_dataset is not None:
                raise ValueError("Pure CatBoost final model does not accept a GLM feature dataset")
        elif glm_dataset is None:
            raise ValueError("Positive GLM weight requires a separate GLM feature dataset")
        else:
            glm_dataset.assert_valid()
            if (
                not cb_dataset.X.index.equals(glm_dataset.X.index)
                or cb_dataset.X.index.name != glm_dataset.X.index.name
                or cb_dataset.X.index.dtype != glm_dataset.X.index.dtype
            ):
                raise ValueError("Component feature match_id order or coverage differs")
            if not cb_dataset.provenance.kickoff_utc.equals(glm_dataset.provenance.kickoff_utc):
                raise ValueError("Component feature forecast kickoff differs")
        before = self.fingerprint()
        cb_prediction = self.cb.predict(cb_dataset)
        glm_prediction = None if self.glm is None else self.glm.predict(glm_dataset)
        prediction = blend_predictions(cb_prediction, glm_prediction, self.config.glm_weight)
        if self.fingerprint() != before:
            raise RuntimeError("Prediction mutated a frozen final component")
        return prediction


def fit_final_model(cb_dataset, glm_dataset, core, history, cutoff, config, policy=None):
    """Finish every component fit before any future files are merged for replay."""
    if type(config) is not FinalConfig:
        raise ValueError("FinalConfig is required")
    cutoff, policy = utc_timestamp(cutoff), policy or AvailabilityPolicy()
    core, history = normalize_matches(core, "core"), normalize_matches(history, "open")
    assert_core_open_consistency(core, history)
    cb_dataset.assert_valid()
    if cb_dataset.metadata.builder != "history_stats_v1":
        raise ValueError("CatBoost requires its separate statistics feature dataset")
    if config.glm_weight > 0:
        if glm_dataset is None:
            raise ValueError("Positive GLM weight requires a separate GLM feature dataset")
        glm_dataset.assert_valid()
        if glm_dataset.metadata.builder != "history":
            raise ValueError("GLM requires its separate history feature dataset")
        if glm_dataset.metadata.columns != tuple(feature_columns(glm_dataset.metadata.config)):
            raise ValueError("GLM feature inventory differs from its history contract")
        if cb_dataset.metadata.config != glm_dataset.metadata.config:
            raise ValueError("Component history feature settings differ")
    cb = fit_boost(cb_dataset, core, history, cutoff, config.catboost, policy)
    glm = None
    if config.glm_weight > 0:
        glm = fit_poisson(glm_dataset, core, cutoff, config.glm, policy)
    return FrozenFinal(cutoff, config, policy, cb, glm)


def train_predict_final(
    training_core,
    training_history,
    forecast_schedule,
    forecast_history,
    cutoff=TRAIN_CUTOFF,
    config=None,
    policy=None,
):
    """Train on old files, then replay future history with frozen components."""
    config = config or FinalConfig(
        BoostConfig(depth=6, l2_leaf_reg=30.0, iterations=300, cohort="all_open", open_stats=True)
    )
    if type(config) is not FinalConfig:
        raise ValueError("FinalConfig is required")
    cutoff, policy = utc_timestamp(cutoff), policy or AvailabilityPolicy()
    core = normalize_matches(training_core, "core")
    old = normalize_matches(training_history, "open")
    if (core.kickoff_utc >= cutoff).any() or (old.kickoff_utc >= cutoff).any():
        raise ValueError("Training files contain matches at/after the frozen cutoff")
    assert_core_open_consistency(core, old)
    schedule = schedule_only(forecast_schedule)
    if (schedule.kickoff_utc < cutoff).any():
        raise ValueError("Forecast schedule precedes the frozen cutoff")
    if schedule.match_id.isin(core.match_id).any():
        raise ValueError("Training and forecast core IDs overlap")
    cb_features = build_open_features(old, old, policy)
    glm_features = build_features(core, old, policy) if config.glm_weight > 0 else None
    model = fit_final_model(cb_features, glm_features, core, old, cutoff, config, policy)
    before = model.fingerprint()
    components = model.fit_report
    for name, component in (("catboost", model.cb), ("glm", model.glm)):
        if component is not None:
            components[name]["fingerprint_before"] = component.fingerprint()
    del cb_features, glm_features
    # No future-history normalization, merge, or feature read occurs before all fits.
    history = merge_histories([old, forecast_history])
    indexed, matched = schedule.set_index("match_id"), schedule_only(history).set_index("match_id")
    if not indexed.index.isin(matched.index).all():
        raise ValueError("Forecast core IDs absent from combined open history")
    matched = matched.reindex(indexed.index)
    for column in METADATA_COLUMNS[1:]:
        a, b = indexed[column], matched[column]
        if not (a.eq(b) | (a.isna() & b.isna())).all():
            raise ValueError(f"Forecast schedule/open metadata disagree: {column}")
    cb_features = build_open_features(schedule, history, policy)
    glm_features = build_features(schedule, history, policy) if config.glm_weight > 0 else None
    prediction = model.predict(cb_features, glm_features)
    after = model.fingerprint()
    if before != after:
        raise RuntimeError("Frozen final model changed during forecast replay")
    for name, component in (("catboost", model.cb), ("glm", model.glm)):
        if component is not None:
            components[name]["fingerprint_after"] = component.fingerprint()
    return prediction, _native(
        {
            "cutoff_utc": cutoff,
            "config": config.to_dict(),
            "availability_policy": asdict(policy),
            "components": components,
            "fingerprint_before": before,
            "fingerprint_after": after,
            "model_fit_calls": 4 if model.glm is None else 8,
            "test_period_fit_calls": 0,
            "forecast_rows": len(prediction),
        }
    )
