"""Versioned allowlist of past open statistics, behind the same strict release gate.

Auxiliary totals have no regulation-time guarantee: every extra-time match is
excluded from this component. Null is unknown. No current-match coverage flag.
"""

from collections import deque

import numpy as np
import pandas as pd

from .config import TARGETS, AvailabilityPolicy, FeatureConfig
from .contracts import DataContractError, normalize_matches, schedule_only
from .feature_contract import FeatureMetadata
from .features import FeatureDataset, build_features

STATS = ("xg", "shots", "sot")
STAT_INPUTS = tuple(f"{s}_{v}" for s in STATS for v in ("h", "a"))
BUILDER = "history_stats_v1"
# Configuration is fixed by the versioned builder; no fitted transforms.
WINDOW = 10
DECAY = 0.5 ** (1 / 5)


def stat_columns() -> tuple[str, ...]:
    return tuple(
        f"{team}_open_{stat}_{direction}_{summary}"
        for team in ("h", "a")
        for stat in STATS
        for direction in ("for", "against")
        for summary in ("all_mean", "all_n", "last10_mean", "last10_n", "ewm5_mean", "ewm5_n")
    )


def validated_stats(history: pd.DataFrame) -> pd.DataFrame:
    """Missing source columns are unknown; malformed known values fail visibly."""
    result = pd.DataFrame(index=history.index)
    for name in STAT_INPUTS:
        values = history[name] if name in history else pd.Series(np.nan, index=history.index)
        try:
            values = pd.to_numeric(values, errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            raise DataContractError(f"Invalid auxiliary statistic: {name}") from exc
        known = values.dropna().to_numpy()
        if not np.isfinite(known).all() or (known < 0).any():
            raise DataContractError(f"Invalid auxiliary statistic: {name}")
        if name.startswith(("shots_", "sot_")) and (known % 1 != 0).any():
            raise DataContractError(f"Noninteger auxiliary count: {name}")
        result[name] = values
    return result


class _StatsTeam:
    def __init__(self):
        self.records = deque(maxlen=WINDOW)
        self.sums = np.zeros(6)
        self.counts = np.zeros(6)
        self.ewm_sum = np.zeros(6)
        self.ewm_weight = np.zeros(6)
        self.ewm_squared = np.zeros(6)
        self.last_available = None
        self.cache = None

    def add(self, values, available):
        values = np.asarray(values)
        known = ~np.isnan(values)
        self.records.append(values)
        self.sums[known] += values[known]
        self.counts[known] += 1
        self.ewm_sum[known] = DECAY * self.ewm_sum[known] + values[known]
        self.ewm_weight[known] = DECAY * self.ewm_weight[known] + 1
        self.ewm_squared[known] = DECAY**2 * self.ewm_squared[known] + 1
        if known.any():
            self.last_available = available
        self.cache = None

    def describe(self):
        if self.cache is None:
            recent = np.asarray(self.records).reshape(-1, 6)
            counts = np.isfinite(recent).sum(axis=0)
            means = np.divide(self.sums, self.counts, out=np.full(6, np.nan), where=self.counts > 0)
            rolling = np.divide(np.nansum(recent, axis=0), counts, out=np.full(6, np.nan), where=counts > 0)
            ewm = np.divide(self.ewm_sum, self.ewm_weight, out=np.full(6, np.nan), where=self.ewm_weight > 0)
            effective = np.divide(
                self.ewm_weight**2, self.ewm_squared, out=np.zeros(6), where=self.ewm_squared > 0
            )
            self.cache = np.column_stack((means, self.counts, rolling, counts, ewm, effective)).ravel()
        return self.cache


def build_open_features(schedule, history, policy=None, config=None, progress=None) -> FeatureDataset:
    policy, config = policy or AvailabilityPolicy(), config or FeatureConfig()
    schedule = schedule_only(schedule)
    # Existing outcome engine is unchanged and retains its own provenance.
    base = build_features(schedule, history, policy, config, progress)
    history = normalize_matches(history, "open")
    stats = validated_stats(history)
    playable = ~history[list(TARGETS)].isna().all(axis=1)
    selected = history.loc[playable].copy()
    selected["available_at"] = selected.kickoff_utc + policy.lag
    selected = selected.sort_values(["available_at", "kickoff_utc", "match_id"], kind="stable")
    # ET statistics are deliberately set unknown, but occupy a rolling match slot.
    stats.loc[history.extra_time, :] = np.nan
    ordered_stats = stats.loc[selected.index].to_numpy()
    queue = list(
        zip(selected.available_at, selected.home_team_id, selected.away_team_id, ordered_stats, strict=True)
    )
    ncols = len(stat_columns())
    numbers = np.empty((len(schedule), ncols), dtype="float32")
    times = np.full((len(schedule), 2), np.iinfo("int64").min, dtype="int64")
    positions = {int(v): i for i, v in enumerate(schedule.match_id)}
    teams, cursor = {}, 0
    neutral = _StatsTeam().describe()
    for kickoff, batch in schedule.groupby("kickoff_utc", sort=True):
        while cursor < len(queue) and queue[cursor][0] < kickoff:
            available, home, away, values = queue[cursor]
            # Input pairs [xg_h,xg_a,shots_h,shots_a,sot_h,sot_a].
            for team, oriented in ((home, values), (away, values.reshape(3, 2)[:, ::-1].ravel())):
                teams.setdefault(int(team), _StatsTeam()).add(oriented, available)
            cursor += 1
        for row in batch.itertuples(index=False):
            pos = positions[row.match_id]
            for j, team in enumerate((row.home_team_id, row.away_team_id)):
                state = teams.get(team)
                numbers[pos, j * (ncols // 2) : (j + 1) * (ncols // 2)] = (
                    neutral if state is None else state.describe()
                )
                if state is not None and state.last_available is not None:
                    times[pos, j] = state.last_available.value
    extra = pd.DataFrame(numbers, columns=stat_columns(), index=base.X.index)
    X = pd.concat([base.X, extra], axis=1)
    provenance = base.provenance.copy()
    for j, side in enumerate(("h", "a")):
        name = f"{side}_open_last_available_at"
        provenance[name] = pd.to_datetime(times[:, j], unit="ns", utc=True)
        if not (provenance[name].isna() | provenance[name].lt(provenance.kickoff_utc)).all():
            raise ValueError("Noncausal open-stat history")
    result = FeatureDataset(X, provenance, FeatureMetadata.for_frame(X, BUILDER, policy, config))
    result.assert_valid()
    return result
