"""Four independent Poisson heads with train-only preprocessing and a frozen fit.

No estimator serialization or refitting during prediction. Categorical vocabulary,
missing-value fills, scaling, coefficients and per-head counts belong to one cutoff.
"""

import hashlib
import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .config import TARGETS, AvailabilityPolicy, ResourceConfig
from .contracts import normalize_matches, utc_timestamp
from .feature_contract import FeatureMetadata
from .features import CATEGORICAL_FEATURES, FeatureDataset
from .scoring import validate_predictions


@dataclass(frozen=True)
class PoissonConfig:
    kind: str = "history_glm"
    alpha: float = 0.001
    max_iter: int = 3000
    tol: float = 1e-5

    def __post_init__(self):
        if self.kind not in ("last_match", "history_glm"):
            raise ValueError("Unknown model kind")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if type(self.max_iter) is not int or self.max_iter <= 0 or not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("Invalid optimizer settings")


def model_columns(target: str, kind: str) -> tuple[list[str], list[str]]:
    if target not in TARGETS or kind not in ("last_match", "history_glm"):
        raise ValueError("Unknown target or model kind")
    metrics = ("gf", "ga") if "goals" in target else ("cf", "ca")
    if kind == "last_match":
        return [], [f"{p}_{m}" for p in ("h", "a") for m in metrics]
    numeric = ["round", "month_sin", "month_cos"] + [f"league_prior_{m}" for m in metrics]
    for p in ("h", "a"):
        numeric += [f"{p}_{s}" for s in ("matches_seen", "rest_days", "matches_last7d", "matches_last28d")]
        for m in metrics:
            numeric += [
                f"{p}_{m}_{s}"
                for s in (
                    "all_mean",
                    "last5_mean",
                    "last20_mean",
                    "venue20_mean",
                    "ewm5_mean",
                    "ewm15_mean",
                    "all_n",
                )
            ]
    return list(CATEGORICAL_FEATURES), numeric


def design_frame(X: pd.DataFrame, target: str, kind: str) -> pd.DataFrame:
    categorical, numeric = model_columns(target, kind)
    # Explicit column selection: no arbitrary open stats, targets or odds are admitted.
    frame = X.loc[:, categorical + numeric].copy()
    if np.isinf(frame[numeric].to_numpy(dtype="float64")).any():
        raise ValueError("Infinite model inputs")
    if kind == "history_glm":
        for name in numeric:
            if name in ("month_sin", "month_cos"):
                continue
            values = frame[name].astype("float64")
            if name == "round":
                # Round identifiers are numeric metadata, not counts. Qualifying
                # rounds may have negative codes under the schedule contract.
                frame[name] = np.sign(values) * np.log1p(np.abs(values))
                continue
            if (values.dropna() < 0).any():
                raise ValueError(f"Negative count/context feature: {name}")
            if name.endswith("_mean") or name.startswith("league_prior_"):
                if (values.dropna() <= 0).any():
                    raise ValueError(f"Nonpositive smoothed mean: {name}")
                frame[name] = np.log(values)
            else:
                if name.endswith("rest_days"):
                    values = values.clip(upper=60)
                frame[name] = np.log1p(values)
        for name in categorical:
            frame[name] = frame[name].fillna("__MISSING__").astype(str)
    return frame


def _pipeline(categorical, numeric, config):
    numeric_steps = [
        (
            "impute",
            SimpleImputer(
                strategy="mean" if config.kind == "last_match" else "median",
                keep_empty_features=True,
                add_indicator=config.kind != "last_match",
            ),
        )
    ]
    # Match the supplied baseline's raw four inputs; regularized history GLM uses scaling.
    if config.kind != "last_match":
        numeric_steps.append(("scale", StandardScaler()))
    transforms = [("numeric", Pipeline(numeric_steps), numeric)]
    if categorical:
        transforms.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical)
        )
    return Pipeline(
        [
            ("prepare", ColumnTransformer(transforms, sparse_threshold=1.0)),
            ("poisson", PoissonRegressor(alpha=config.alpha, max_iter=config.max_iter, tol=config.tol)),
        ]
    )


def _digest_value(value, digest):
    if isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        if value.dtype.kind in "OUS":
            digest.update(json.dumps(value.tolist(), ensure_ascii=False).encode())
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (list, tuple)):
        for item in value:
            _digest_value(item, digest)
    else:
        digest.update(repr(value).encode())


@dataclass(frozen=True)
class FrozenPoisson:
    cutoff: pd.Timestamp
    config: PoissonConfig
    policy: AvailabilityPolicy
    heads: dict
    fit_report: dict
    feature_metadata: FeatureMetadata

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        _digest_value((self.cutoff.isoformat(), repr(self.config), repr(self.policy)), digest)
        _digest_value(self.feature_metadata.fingerprint(), digest)
        for target in TARGETS:
            pipe = self.heads[target]
            _digest_value(target, digest)
            prepare, model = pipe["prepare"], pipe["poisson"]
            _digest_value((model.coef_, model.intercept_, model.n_iter_), digest)
            numeric = prepare.named_transformers_["numeric"]
            imputer = numeric["impute"]
            _digest_value(
                (imputer.statistics_, imputer.indicator_.features_ if imputer.add_indicator else ()), digest
            )
            if "scale" in numeric.named_steps:
                scale = numeric["scale"]
                _digest_value((scale.mean_, scale.var_, scale.scale_, scale.n_samples_seen_), digest)
            if "categorical" in prepare.named_transformers_:
                _digest_value(prepare.named_transformers_["categorical"].categories_, digest)
        return digest.hexdigest()

    def predict(self, dataset: FeatureDataset) -> pd.DataFrame:
        dataset.assert_valid()
        if dataset.metadata != self.feature_metadata:
            raise ValueError("Prediction feature metadata differs from fitted feature contract")
        if (dataset.provenance.kickoff_utc < self.cutoff).any():
            raise ValueError("Forecast schedule precedes frozen model cutoff")
        if dataset.X.empty:
            return pd.DataFrame(index=dataset.X.index, columns=TARGETS, dtype="float64")
        before = self.fingerprint()
        with threadpool_limits(limits=ResourceConfig().threads), warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            predictions = pd.DataFrame(
                {
                    target: self.heads[target].predict(design_frame(dataset.X, target, self.config.kind))
                    for target in TARGETS
                },
                index=dataset.X.index,
            )
        # Finite, positive means are required; no rounding or arbitrary upper clipping.
        if not np.isfinite(predictions.to_numpy()).all():
            raise ValueError("Nonfinite Poisson predictions")
        predictions = predictions.clip(lower=1e-8)
        validate_predictions(predictions, dataset.X.index)
        if self.fingerprint() != before:
            raise RuntimeError("Prediction mutated fitted model state")
        return predictions


def fit_poisson(
    dataset: FeatureDataset,
    matches: pd.DataFrame,
    cutoff,
    config: PoissonConfig | None = None,
    policy: AvailabilityPolicy | None = None,
) -> FrozenPoisson:
    config, policy = config or PoissonConfig(), policy or AvailabilityPolicy()
    cutoff = utc_timestamp(cutoff)
    # Direct library use must respect the same budget as the CLI/notebook.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(ResourceConfig().threads))
    dataset.assert_valid()
    if dataset.metadata.policy != policy:
        raise ValueError("Feature availability policy differs from model policy")
    expected_builder = "history" if config.kind == "history_glm" else "last_match"
    if dataset.metadata.builder != expected_builder:
        raise ValueError("Feature builder differs from model kind")
    matches = normalize_matches(matches, "core").set_index("match_id")
    if not dataset.X.index.equals(matches.index):
        raise ValueError("Training features and match rows must have identical match_id order")
    if not dataset.provenance.kickoff_utc.equals(matches.kickoff_utc):
        raise ValueError("Training provenance kickoff differs from match metadata")
    eligible = matches.kickoff_utc + policy.lag < cutoff
    heads, report = {}, {}
    for target in TARGETS:
        mask = eligible & matches[target].notna()
        ids = matches.loc[mask].reset_index().sort_values(["kickoff_utc", "match_id"]).match_id
        y = matches.loc[ids, target]
        if len(y) < 2 or y.sum() <= 0:
            raise ValueError(f"Insufficient nonzero training outcomes: {target}")
        categorical, numeric = model_columns(target, config.kind)
        pipe = _pipeline(categorical, numeric, config)
        # Fit both preprocessor and regressor exclusively on this head's observed past labels.
        with threadpool_limits(limits=ResourceConfig().threads), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", RuntimeWarning)
            pipe.fit(design_frame(dataset.X.loc[y.index], target, config.kind), y)
        encoded = pipe["prepare"].transform(design_frame(dataset.X.loc[y.index], target, config.kind))
        fitted = pipe["poisson"].predict(encoded)
        residual = (fitted - y.to_numpy()) / len(y)
        gradient = np.asarray(encoded.T @ residual).ravel() + config.alpha * pipe["poisson"].coef_
        gradient_norm = float(max(np.abs(gradient).max(), abs(residual.sum())))
        if not np.isfinite(gradient_norm) or gradient_norm > 10 * config.tol:
            raise RuntimeError(f"Optimizer gradient too large for {target}: {gradient_norm}")
        heads[target] = pipe
        report[target] = {
            "n_train": int(mask.sum()),
            "gradient_max_abs": gradient_norm,
            "max_label_available_at": (matches.loc[mask].kickoff_utc + policy.lag).max(),
            "train_match_ids_sha256": hashlib.sha256(y.index.to_numpy(dtype="int64").tobytes()).hexdigest(),
            "iterations": int(pipe["poisson"].n_iter_),
            "encoded_features": int(pipe["poisson"].n_features_in_),
        }
    return FrozenPoisson(cutoff, config, policy, heads, report, dataset.metadata)
