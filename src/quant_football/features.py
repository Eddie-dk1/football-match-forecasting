"""Causal replay of completed match history, independent of model fitting.

The queue can contain future outcomes, but a single release gate controls every
state mutation. A prediction-facing request is projected to schedule metadata.
Events and open-source statistics are deliberately absent at stages 1–2.
"""

import json
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import METRICS, TARGETS, TRAIN_CUTOFF, AvailabilityPolicy, FeatureConfig
from .contracts import normalize_matches, schedule_only, utc_timestamp
from .feature_contract import FeatureMetadata
from .io import ArtifactStore, contained_file, sha256_file, verify_hashes

DAY_NS = 86_400_000_000_000
CATEGORICAL_FEATURES = (
    "league_id",
    "league_country",
    "league_type",
    "season_id",
    "home_team_id",
    "away_team_id",
)
CONTEXT_NUMERIC = ("round", "month_sin", "month_cos")


def _category(value: object) -> str:
    if pd.isna(value):
        return "__MISSING__"
    if isinstance(value, (float, np.floating)) and value.is_integer():
        return str(int(value))
    return str(value)


@dataclass(slots=True)
class _Running:
    sums: list[float] = field(default_factory=lambda: [0.0] * 4)
    squares: list[float] = field(default_factory=lambda: [0.0] * 4)
    counts: list[int] = field(default_factory=lambda: [0] * 4)

    def add(self, values: tuple[float, ...]) -> None:
        for i, value in enumerate(values):
            if not math.isnan(value):
                self.sums[i] += value
                self.squares[i] += value * value
                self.counts[i] += 1

    def raw(self, i: int) -> tuple[float, int, float]:
        n = self.counts[i]
        if not n:
            return math.nan, 0, math.nan
        mean = self.sums[i] / n
        variance = max(0.0, self.squares[i] / n - mean * mean)
        return mean, n, math.sqrt(variance) if n >= 2 else math.nan

    def shrink(self, priors: list[float], strength: float) -> list[float]:
        return [(s + strength * p) / (n + strength) for s, n, p in zip(self.sums, self.counts, priors)]


@dataclass(slots=True)
class _Ewm:
    decay: float
    sums: list[float] = field(default_factory=lambda: [0.0] * 4)
    weights: list[float] = field(default_factory=lambda: [0.0] * 4)
    squared_weights: list[float] = field(default_factory=lambda: [0.0] * 4)

    def add(self, values: tuple[float, ...]) -> None:
        for i, value in enumerate(values):
            if not math.isnan(value):
                self.sums[i] = self.decay * self.sums[i] + value
                self.weights[i] = self.decay * self.weights[i] + 1.0
                self.squared_weights[i] = self.decay**2 * self.squared_weights[i] + 1.0

    def raw(self, i: int) -> tuple[float, float]:
        w = self.weights[i]
        if not w:
            return math.nan, 0.0
        return self.sums[i] / w, w * w / self.squared_weights[i]


@dataclass(frozen=True, slots=True)
class _Result:
    match_id: int
    kickoff_ns: int
    available_ns: int
    league: int
    country: str
    league_type: str
    home: int
    away: int
    values: tuple[float, ...]


class _Team:
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.all = _Running()
        self.matches_seen = 0
        self.venue_seen = {"home": 0, "away": 0}
        self.last_kickoff_ns: int | None = None
        self.last_available_ns: int | None = None
        self.recent: deque[int] = deque()
        self.records: deque[tuple[float, ...]] = deque(maxlen=max(config.windows))
        self.venue_records = {v: deque(maxlen=max(config.venue_windows)) for v in ("home", "away")}
        self.ewm = {h: _Ewm(0.5 ** (1.0 / h)) for h in config.ewm_half_lives}
        self._cache: dict[str, dict] = {}

    def add(self, values: tuple[float, ...], venue: str, kickoff_ns: int, available_ns: int) -> None:
        self.matches_seen += 1
        self.venue_seen[venue] += 1
        self.all.add(values)
        self.records.append(values)
        self.venue_records[venue].append(values)
        for ewm in self.ewm.values():
            ewm.add(values)
        self.last_kickoff_ns = kickoff_ns
        self.last_available_ns = available_ns
        self.recent.append(kickoff_ns)
        while self.recent and self.recent[0] < kickoff_ns - 28 * DAY_NS:
            self.recent.popleft()
        self._cache.clear()

    @staticmethod
    def _window(records: deque, n: int, i: int) -> tuple[float, int, float]:
        values = [v[i] for v in list(records)[-n:] if not math.isnan(v[i])]
        count = len(values)
        if not count:
            return math.nan, 0, math.nan
        mean = sum(values) / count
        variance = max(0.0, sum(v * v for v in values) / count - mean * mean)
        return mean, count, math.sqrt(variance) if count >= 2 else math.nan

    def _raw(self, venue: str) -> dict:
        if venue not in self._cache:
            raw = {}
            for i, metric in enumerate(METRICS):
                raw[(metric, "all")] = self.all.raw(i)
                for n in self.config.windows:
                    raw[(metric, f"last{n}")] = self._window(self.records, n, i)
                for n in self.config.venue_windows:
                    raw[(metric, f"venue{n}")] = self._window(self.venue_records[venue], n, i)
                for h in self.config.ewm_half_lives:
                    raw[(metric, f"ewm{h}")] = self.ewm[h].raw(i)
            self._cache[venue] = raw
        return self._cache[venue]

    def describe(self, prefix: str, venue: str, t_ns: int, priors: list[float]) -> dict:
        values = {
            f"{prefix}_matches_seen": self.matches_seen,
            f"{prefix}_venue_matches_seen": self.venue_seen[venue],
            f"{prefix}_rest_days": math.nan
            if self.last_kickoff_ns is None
            else (t_ns - self.last_kickoff_ns) / DAY_NS,
        }
        for days in (7, 14, 28):
            values[f"{prefix}_matches_last{days}d"] = sum(
                t_ns - days * DAY_NS <= t < t_ns for t in self.recent
            )
        raw = self._raw(venue)
        strength = self.config.team_prior_strength
        for i, metric in enumerate(METRICS):
            for label in ("all", *(f"last{n}" for n in self.config.windows)):
                mean, n, std = raw[(metric, label)]
                values[f"{prefix}_{metric}_{label}_mean"] = (
                    priors[i] if not n else (mean * n + strength * priors[i]) / (n + strength)
                )
                values[f"{prefix}_{metric}_{label}_n"] = n
                values[f"{prefix}_{metric}_{label}_std"] = std
            for n in self.config.venue_windows:
                mean, count, _ = raw[(metric, f"venue{n}")]
                values[f"{prefix}_{metric}_venue{n}_mean"] = (
                    priors[i] if not count else (mean * count + strength * priors[i]) / (count + strength)
                )
                values[f"{prefix}_{metric}_venue{n}_n"] = count
            for h in self.config.ewm_half_lives:
                mean, effective_n = raw[(metric, f"ewm{h}")]
                values[f"{prefix}_{metric}_ewm{h}_mean"] = (
                    priors[i]
                    if not effective_n
                    else (mean * effective_n + strength * priors[i]) / (effective_n + strength)
                )
                values[f"{prefix}_{metric}_ewm{h}_effective_n"] = effective_n
        return values


def feature_columns(config: FeatureConfig) -> list[str]:
    columns = [*CATEGORICAL_FEATURES, *CONTEXT_NUMERIC, *(f"league_prior_{m}" for m in METRICS)]
    neutral = _Team(config)
    for prefix, venue in (("h", "home"), ("a", "away")):
        columns.extend(
            neutral.describe(
                prefix, venue, 0, [config.initial_goal_mean] * 2 + [config.initial_corner_mean] * 2
            )
        )
    if len(set(columns)) != len(columns):
        raise AssertionError("Duplicate generated feature names")
    return columns


@dataclass(frozen=True)
class FeatureDataset:
    X: pd.DataFrame
    provenance: pd.DataFrame
    metadata: FeatureMetadata | None = None

    def assert_valid(self) -> None:
        if not self.X.index.is_unique or not self.X.index.equals(self.provenance.index):
            raise AssertionError("Feature/provenance index mismatch")
        if self.X.index.name != "match_id":
            raise AssertionError("Expected match_id index")
        if any(c in self.X for c in (*TARGETS, "kickoff_utc", "extra_time", "match_id")):
            raise AssertionError("Outcome or audit columns entered the model feature frame")
        if np.isinf(self.X.select_dtypes(include="number").to_numpy()).any():
            raise AssertionError("Infinite model features")
        required_times = {
            "history_max_available_at",
            "h_last_available_at",
            "a_last_available_at",
            "kickoff_utc",
        }
        if not required_times.issubset(self.provenance.columns):
            raise ValueError("Missing required feature provenance")
        for name in (c for c in self.provenance.columns if c.endswith("_available_at")):
            valid = self.provenance[name].notna()
            if not self.provenance.loc[valid, name].lt(self.provenance.loc[valid, "kickoff_utc"]).all():
                raise AssertionError(f"Non-causal provenance in {name}")

        if self.metadata is None:
            raise ValueError("Feature metadata is required; rebuild legacy artifacts")
        self.metadata.validate_frame(self.X)

    def subset(self, ids) -> "FeatureDataset":
        result = FeatureDataset(self.X.loc[ids], self.provenance.loc[ids], self.metadata)
        result.assert_valid()
        return result

    @staticmethod
    def artifact_names(prefix="core") -> tuple[str, ...]:
        if not prefix or not prefix.replace("_", "").isalnum():
            raise ValueError("Invalid feature artifact prefix")
        return (
            f"{prefix}_features.parquet",
            f"{prefix}_feature_provenance.parquet",
            f"{prefix}_feature_metadata.json",
        )

    def save(self, directory: Path, prefix="core", *, protected: Path | None = None) -> None:
        self.assert_valid()
        store = ArtifactStore(directory, protected)
        features, provenance, metadata = self.artifact_names(prefix)
        store.preflight((features, provenance, metadata))
        store.parquet(features, self.X, index=True)
        store.parquet(provenance, self.provenance, index=True)
        store.json(
            metadata,
            {
                "metadata": self.metadata.to_dict(),
                "files": {name: sha256_file(store.path(name)) for name in (features, provenance)},
            },
        )

    @classmethod
    def load(cls, directory: Path, prefix="core") -> "FeatureDataset":
        features, provenance, metadata = cls.artifact_names(prefix)
        path = contained_file(directory, metadata)
        if not path.is_file():
            raise ValueError("Missing feature metadata; rebuild legacy artifacts")
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or set(record) != {"metadata", "files"}:
            raise ValueError("Invalid feature sidecar inventory")
        verify_hashes(directory, record["files"], (features, provenance))
        result = cls(
            pd.read_parquet(contained_file(directory, features)),
            pd.read_parquet(contained_file(directory, provenance)),
            FeatureMetadata.from_dict(record["metadata"]),
        )
        result.assert_valid()
        return result


class _FeatureAccumulator:
    """Preallocate float32 output: never retain a dict per historical match.

    Provenance uses int64 nanoseconds, including the NaT sentinel. It must not
    pass through float64, which would lose timestamp precision at boundaries.
    """

    TIME_COLUMNS = ("kickoff_utc", "history_max_available_at", "h_last_available_at", "a_last_available_at")

    def __init__(self, schedule: pd.DataFrame, config: FeatureConfig, policy: AvailabilityPolicy):
        self.config, self.policy = config, policy
        self.order = pd.Index(schedule.match_id, name="match_id")
        self.positions = {int(mid): i for i, mid in enumerate(self.order)}
        self.columns = feature_columns(config)
        self.numeric = [c for c in self.columns if c not in CATEGORICAL_FEATURES]
        self.numbers = np.empty((len(schedule), len(self.numeric)), dtype="float32")
        self.times = np.full((len(schedule), 4), np.iinfo("int64").min, dtype="int64")
        self.counts = np.zeros(len(schedule), dtype="int64")
        self.seen = np.zeros(len(schedule), dtype=bool)
        self.categories = pd.DataFrame(
            {c: pd.array(schedule[c].map(_category), dtype="string") for c in CATEGORICAL_FEATURES},
            index=self.order,
        )

    def add(self, records: list[dict], traces: list[dict]) -> None:
        if len(records) != len(traces):
            raise AssertionError("Record/provenance batch length mismatch")
        for record, trace in zip(records, traces):
            pos = self.positions[record["match_id"]]
            if self.seen[pos] or record["match_id"] != trace["match_id"]:
                raise AssertionError("Duplicate or misaligned feature row")
            self.numbers[pos] = [record[c] for c in self.numeric]
            for j, c in enumerate(self.TIME_COLUMNS):
                value = trace[c]
                self.times[pos, j] = np.iinfo("int64").min if value is None else value
            self.counts[pos] = trace["history_rows_released"]
            self.seen[pos] = True

    def finish(self) -> FeatureDataset:
        if not self.seen.all():
            raise AssertionError("Missing feature rows")
        numeric = pd.DataFrame(self.numbers, columns=self.numeric, index=self.order)
        X = pd.concat([self.categories, numeric], axis=1).loc[:, self.columns]
        trace = pd.DataFrame(
            {
                c: pd.to_datetime(self.times[:, j], unit="ns", utc=True)
                for j, c in enumerate(self.TIME_COLUMNS)
            },
            index=self.order,
        )
        trace["history_rows_released"] = self.counts
        result = FeatureDataset(X, trace, FeatureMetadata.for_frame(X, "history", self.policy, self.config))
        result.assert_valid()
        return result


class CausalFeatureEngine:
    """Replay a fixed history queue in increasing prediction time.

    `features_before` accepts one kickoff batch and automatically advances the
    release gate. Repeated reads at the same time are idempotent. Going backward
    is an error. Equal release times are ordered by kickoff, then match_id.
    """

    def __init__(
        self,
        history: pd.DataFrame,
        policy: AvailabilityPolicy | None = None,
        config: FeatureConfig | None = None,
    ):
        self.policy = policy or AvailabilityPolicy()
        self.config = config or FeatureConfig()
        history = normalize_matches(history, "open")
        self.history_rows = len(history)
        self.skipped_no_result = int(history[list(TARGETS)].isna().all(axis=1).sum())
        playable = history.loc[~history[list(TARGETS)].isna().all(axis=1)].copy()
        playable["available_at"] = playable.kickoff_utc + self.policy.lag
        playable = playable.sort_values(["available_at", "kickoff_utc", "match_id"], kind="stable")
        columns = [
            "match_id",
            "kickoff_utc",
            "available_at",
            "league_id",
            "league_country",
            "league_type",
            "home_team_id",
            "away_team_id",
            *TARGETS,
        ]
        self._queue = [
            _Result(
                int(mid),
                start.value,
                available.value,
                int(league),
                _category(country),
                _category(lt),
                int(home),
                int(away),
                tuple(float(v) for v in vals),
            )
            for mid, start, available, league, country, lt, home, away, *vals in playable[columns].itertuples(
                index=False, name=None
            )
        ]
        self._cursor = 0
        self._clock_ns: int | None = None
        self._max_available_ns: int | None = None
        self._teams: dict[int, _Team] = {}
        self._global = _Running()
        self._types: dict[str, _Running] = {}
        self._countries: dict[tuple[str, str], _Running] = {}
        self._leagues: dict[int, _Running] = {}

    @property
    def released_matches(self) -> int:
        return self._cursor

    def advance_to(self, value: str | pd.Timestamp) -> None:
        t_ns = utc_timestamp(value).value
        if self._clock_ns is not None and t_ns < self._clock_ns:
            raise ValueError("CausalFeatureEngine cannot move backward in time")
        while self._cursor < len(self._queue) and self._queue[self._cursor].available_ns < t_ns:
            result = self._queue[self._cursor]
            home = result.values
            away = (home[1], home[0], home[3], home[2])
            for team_id, venue, values in ((result.home, "home", home), (result.away, "away", away)):
                team = self._teams.get(team_id)
                if team is None:
                    team = self._teams[team_id] = _Team(self.config)
                team.add(values, venue, result.kickoff_ns, result.available_ns)
                self._global.add(values)
                self._types.setdefault(result.league_type, _Running()).add(values)
                self._countries.setdefault((result.country, result.league_type), _Running()).add(values)
                self._leagues.setdefault(result.league, _Running()).add(values)
            self._max_available_ns = result.available_ns
            self._cursor += 1
        self._clock_ns = t_ns

    def _priors(self, league: int, country: str, lt: str) -> list[float]:
        prior = [self.config.initial_goal_mean] * 2 + [self.config.initial_corner_mean] * 2
        strength = self.config.group_prior_strength
        # All four levels are past-only, including the global prior.
        prior = self._global.shrink(prior, strength)
        for group in (self._types.get(lt), self._countries.get((country, lt)), self._leagues.get(league)):
            if group is not None:
                prior = group.shrink(prior, strength)
        return prior

    def _records_for_validated_batch(self, batch: pd.DataFrame) -> tuple[list[dict], list[dict]]:
        t = batch.kickoff_utc.iloc[0]
        self.advance_to(t)
        records, traces = [], []
        for row in batch.itertuples(index=False):
            priors = self._priors(
                int(row.league_id), _category(row.league_country), _category(row.league_type)
            )
            values = {c: _category(getattr(row, c)) for c in CATEGORICAL_FEATURES}
            values.update(
                {
                    "match_id": int(row.match_id),
                    "round": row.round,
                    "month_sin": math.sin(2 * math.pi * (t.month - 1) / 12),
                    "month_cos": math.cos(2 * math.pi * (t.month - 1) / 12),
                }
            )
            values.update({f"league_prior_{m}": p for m, p in zip(METRICS, priors)})
            trace = {
                "match_id": int(row.match_id),
                "kickoff_utc": t.value,
                "history_max_available_at": self._max_available_ns,
                "history_rows_released": self.released_matches,
            }
            for prefix, venue, team_id in (("h", "home", row.home_team_id), ("a", "away", row.away_team_id)):
                team = self._teams.get(int(team_id))
                if team is None:
                    team = _Team(self.config)
                values.update(team.describe(prefix, venue, t.value, priors))
                trace[f"{prefix}_last_available_at"] = team.last_available_ns
            records.append(values)
            traces.append(trace)
        return records, traces

    def features_before(self, matches: pd.DataFrame) -> FeatureDataset:
        batch = schedule_only(matches)
        accumulator = _FeatureAccumulator(batch, self.config, self.policy)
        if batch.empty:
            return accumulator.finish()
        if batch.kickoff_utc.nunique() != 1:
            raise ValueError(
                "features_before expects one kickoff batch; use build_features for a full schedule"
            )
        records, provenance = self._records_for_validated_batch(batch)
        accumulator.add(records, provenance)
        return accumulator.finish()


def build_features(
    schedule: pd.DataFrame,
    history: pd.DataFrame,
    policy: AvailabilityPolicy | None = None,
    config: FeatureConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> FeatureDataset:
    """Build a complete pre-match matrix, preserving the requested match order."""
    schedule = schedule_only(schedule)
    config = config or FeatureConfig()
    engine = CausalFeatureEngine(history, policy, config)
    accumulator = _FeatureAccumulator(schedule, config, engine.policy)
    done = 0
    for _, batch in schedule.groupby("kickoff_utc", sort=True):
        rows, provenance = engine._records_for_validated_batch(batch)
        accumulator.add(rows, provenance)
        done += len(batch)
        if progress is not None:
            progress(done, len(schedule))
    return accumulator.finish()


def training_labels(
    core: pd.DataFrame,
    policy: AvailabilityPolicy | None = None,
    cutoff: str | pd.Timestamp = TRAIN_CUTOFF,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate outcomes and a time-eligibility mask. Per-target nulls remain null.

    A future model must additionally select y[target].notna() independently for
    each target. No model fitting is performed by this function.
    """
    core = normalize_matches(core, "core")
    policy = policy or AvailabilityPolicy()
    y = core.set_index("match_id")[list(TARGETS)]
    available = core.kickoff_utc + policy.lag
    eligible = pd.Series(
        (available < utc_timestamp(cutoff)).to_numpy(), index=y.index, name="labels_available"
    )
    return y, eligible
