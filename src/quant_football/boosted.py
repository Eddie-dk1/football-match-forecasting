"""Four frozen CPU CatBoost Poisson heads. No eval_set or test-period fitting."""

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .config import TARGETS, AvailabilityPolicy, ResourceConfig
from .contracts import assert_core_open_consistency, normalize_matches, utc_timestamp
from .feature_contract import FeatureMetadata
from .features import CATEGORICAL_FEATURES, FeatureDataset, feature_columns
from .open_stats import BUILDER, stat_columns
from .scoring import validate_predictions

COHORTS = ("core", "open_core_leagues", "all_open")


@dataclass(frozen=True)
class BoostConfig:
    depth: int = 6
    l2_leaf_reg: float = 10.0
    iterations: int = 750
    learning_rate: float = 0.04
    seed: int = 42
    cohort: str = "core"
    open_stats: bool = False

    def __post_init__(self):
        if type(self.depth) is not int or not 1 <= self.depth <= 8:
            raise ValueError("Invalid tree depth")
        if type(self.iterations) is not int or not 1 <= self.iterations <= 1500:
            raise ValueError("Invalid iteration budget")
        if not np.isfinite(self.l2_leaf_reg) or self.l2_leaf_reg <= 0:
            raise ValueError("Invalid L2 regularization")
        if not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError("Invalid learning rate")
        if type(self.seed) is not int or self.cohort not in COHORTS or type(self.open_stats) is not bool:
            raise ValueError("Invalid research configuration")


def select_cohort(core, history, cutoff, cohort, policy=None):
    """Membership and league vocabulary learned exclusively from eligible old core."""
    if cohort not in COHORTS:
        raise ValueError("Unknown training cohort")
    policy, cutoff = policy or AvailabilityPolicy(), utc_timestamp(cutoff)
    core, history = normalize_matches(core, "core"), normalize_matches(history, "open")
    assert_core_open_consistency(core, history)
    old_core = core.loc[core.kickoff_utc + policy.lag < cutoff]
    old_open = history.loc[history.kickoff_utc + policy.lag < cutoff]
    if cohort == "core":
        rows = old_core.copy()
    elif cohort == "open_core_leagues":
        rows = old_open.loc[old_open.league_id.isin(old_core.league_id.unique())].copy()
    else:
        rows = old_open.copy()
    rows = rows.sort_values(["kickoff_utc", "match_id"], kind="stable").reset_index(drop=True)
    # core-only is unweighted. Mixed cohorts: core 3, non-core 1.
    weights = pd.Series(
        np.where(rows.match_id.isin(old_core.match_id), 3.0, 1.0) if cohort != "core" else 1.0,
        index=pd.Index(rows.match_id, name="match_id"),
        dtype="float64",
    )
    return rows, weights


def boost_frame(dataset, config):
    dataset.assert_valid()
    event_book = getattr(config, "event_codebook", None)
    expected = BUILDER if config.open_stats else "history"
    if event_book is not None:
        from .events import BUILDER_PREFIX

        expected = BUILDER_PREFIX + event_book.fingerprint()
    if dataset.metadata.builder != expected:
        raise ValueError("Feature builder differs from CatBoost configuration")
    columns = feature_columns(dataset.metadata.config)
    if config.open_stats:
        columns += list(stat_columns())
    if event_book is not None:
        from .events import event_columns

        columns += list(event_columns())
    if tuple(columns) != dataset.metadata.columns:
        raise ValueError("Unexpected CatBoost feature inventory")
    frame = dataset.X.loc[:, columns].copy()
    for c in CATEGORICAL_FEATURES:
        frame[c] = frame[c].fillna("__MISSING__").astype(str)
    # Native missing-value handling, no fitted imputer/scaler.
    return frame


def pool_for(dataset, config, label=None, weight=None):
    return Pool(
        boost_frame(dataset, config),
        label=label,
        weight=weight,
        cat_features=list(CATEGORICAL_FEATURES),
        thread_count=ResourceConfig().threads,
    )


def _state_digest(head):
    # Public JSON export contains all prediction state, including categorical
    # tables. Exclude volatile training metadata (GUID, time and logs) only.
    with tempfile.TemporaryDirectory(prefix="quant-model-state-") as directory:
        path = Path(directory) / "state.json"
        head.save_model(str(path), format="json")
        state = json.loads(path.read_text())
    state.pop("model_info", None)
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class FrozenBoost:
    cutoff: pd.Timestamp
    config: BoostConfig
    policy: AvailabilityPolicy
    heads: dict
    fit_report: dict
    feature_metadata: FeatureMetadata

    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update(json.dumps(asdict(self.config), sort_keys=True).encode())
        digest.update(self.cutoff.isoformat().encode())
        digest.update(self.feature_metadata.fingerprint().encode())
        for target in TARGETS:
            digest.update(_state_digest(self.heads[target]).encode())
        return digest.hexdigest()

    def predict(self, dataset: FeatureDataset, iterations=None):
        dataset.assert_valid()
        if dataset.metadata != self.feature_metadata:
            raise ValueError("Prediction feature metadata differs from fitted feature contract")
        if (dataset.provenance.kickoff_utc < self.cutoff).any():
            raise ValueError("Forecast schedule precedes frozen model cutoff")
        trees = self.config.iterations if iterations is None else iterations
        if type(trees) is not int or not 1 <= trees <= self.config.iterations:
            raise ValueError("Invalid prediction tree count")
        if dataset.X.empty:
            return pd.DataFrame(index=dataset.X.index, columns=TARGETS, dtype=float)
        before = self.fingerprint()
        pool = pool_for(dataset, self.config)
        predictions = pd.DataFrame(
            {
                t: self.heads[t].predict(
                    pool, prediction_type="Exponent", ntree_end=trees, thread_count=ResourceConfig().threads
                )
                for t in TARGETS
            },
            index=dataset.X.index,
        )
        if not np.isfinite(predictions.to_numpy()).all():
            raise ValueError("Nonfinite CatBoost predictions")
        predictions = predictions.clip(lower=1e-8)
        validate_predictions(predictions, dataset.X.index)
        if self.fingerprint() != before:
            raise RuntimeError("Prediction mutated fitted CatBoost")
        return predictions


def catboost_parameters(config):
    return dict(
        loss_function="Poisson",
        iterations=config.iterations,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        learning_rate=config.learning_rate,
        random_seed=config.seed,
        has_time=True,
        boosting_type="Plain",
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        one_hot_max_size=2,
        max_ctr_complexity=1,
        nan_mode="Min",
        task_type="CPU",
        thread_count=ResourceConfig().threads,
        allow_writing_files=False,
        verbose=False,
        use_best_model=False,
    )


def fit_boost(dataset, core, history, cutoff, config=None, policy=None):
    config, policy, cutoff = config or BoostConfig(), policy or AvailabilityPolicy(), utc_timestamp(cutoff)
    dataset.assert_valid()
    book = getattr(config, "event_codebook", None)
    if book is not None and (utc_timestamp(book.cutoff) != cutoff or book.lag_hours != policy.lag_hours):
        raise ValueError("Event decoder cutoff/policy differs from model fit")
    if dataset.metadata.policy != policy:
        raise ValueError("Feature availability policy differs from model policy")
    # Validate builder even when no rows can be fitted.
    boost_frame(dataset.subset(dataset.X.index[:0]), config)
    rows, weights = select_cohort(core, history, cutoff, config.cohort, policy)
    ids = pd.Index(rows.match_id, name="match_id")
    if not ids.isin(dataset.X.index).all():
        raise ValueError("Missing cohort features")
    if not dataset.provenance.loc[ids, "kickoff_utc"].equals(rows.set_index("match_id").kickoff_utc):
        raise ValueError("Training provenance kickoff differs from match metadata")
    heads, report = {}, {}
    indexed = rows.set_index("match_id")
    for target in TARGETS:
        selected = indexed.loc[indexed[target].notna()]
        if len(selected) < 2 or selected[target].sum() <= 0:
            raise ValueError(f"Insufficient nonzero training outcomes: {target}")
        fit_ids = selected.index
        pool = pool_for(dataset.subset(fit_ids), config, selected[target], weights.loc[fit_ids])
        model = CatBoostRegressor(**catboost_parameters(config))
        model.fit(pool)
        if model.tree_count_ != config.iterations:
            raise RuntimeError("Unexpected tree count; model was not fully fitted")
        heads[target] = model
        report[target] = {
            "n_train": len(selected),
            "n_core": int(selected.index.isin(core.match_id).sum()),
            "weight_sum": float(weights.loc[fit_ids].sum()),
            "max_label_available_at": (selected.kickoff_utc + policy.lag).max(),
            "train_match_ids_sha256": hashlib.sha256(fit_ids.to_numpy(dtype="int64").tobytes()).hexdigest(),
            "weights_sha256": hashlib.sha256(
                weights.loc[fit_ids].to_numpy(dtype="float64").tobytes()
            ).hexdigest(),
            "trees": model.tree_count_,
            "n_features": len(model.feature_names_),
            "training_parameters": model.get_params(),
        }
    return FrozenBoost(cutoff, config, policy, heads, report, dataset.metadata)
