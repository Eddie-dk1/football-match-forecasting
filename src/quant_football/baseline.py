"""Supplied last-match baseline, repaired to respect strict history availability.

The original notebook uses shift(1) by kickoff, which does not handle overlap.
This version releases complete records only at kickoff + policy lag < forecast.
A missing metric in the last confirmed match stays missing (not last non-null).
"""

import numpy as np
import pandas as pd

from .config import TARGETS, AvailabilityPolicy
from .contracts import normalize_matches, schedule_only
from .feature_contract import FeatureMetadata
from .features import FeatureDataset

LAST_COLUMNS = tuple(f"{prefix}_{metric}" for prefix in ("h", "a") for metric in ("gf", "ga", "cf", "ca"))


def build_last_features(
    schedule: pd.DataFrame, history: pd.DataFrame, policy: AvailabilityPolicy | None = None
) -> FeatureDataset:
    policy = policy or AvailabilityPolicy()
    schedule = schedule_only(schedule)
    history = normalize_matches(history, "open")
    history = history.loc[history[list(TARGETS)].notna().any(axis=1)].copy()
    history["available_at"] = history.kickoff_utc + policy.lag
    queue = list(
        history.sort_values(["available_at", "kickoff_utc", "match_id"])[
            ["available_at", "home_team_id", "away_team_id", *TARGETS]
        ].itertuples(index=False, name=None)
    )
    index = pd.Index(schedule.match_id, name="match_id")
    positions = {mid: i for i, mid in enumerate(index)}
    output = np.full((len(index), 8), np.nan)
    nat = np.iinfo("int64").min
    times = np.full((len(index), 4), nat, dtype="int64")
    counts = np.zeros(len(index), dtype="int64")
    teams, cursor = {}, 0
    for kickoff, batch in schedule.groupby("kickoff_utc", sort=True):
        while cursor < len(queue) and queue[cursor][0] < kickoff:
            available, home, away, hg, ag, hc, ac = queue[cursor]
            teams[home] = ((hg, ag, hc, ac), available.value)
            teams[away] = ((ag, hg, ac, hc), available.value)
            cursor += 1
        for row in batch.itertuples(index=False):
            pos = positions[row.match_id]
            times[pos, 0] = kickoff.value
            times[pos, 1] = queue[cursor - 1][0].value if cursor else nat
            counts[pos] = cursor
            for j, team in enumerate((row.home_team_id, row.away_team_id)):
                if team in teams:
                    output[pos, 4 * j : 4 * j + 4] = teams[team][0]
                    times[pos, j + 2] = teams[team][1]
    provenance = pd.DataFrame(
        {
            name: pd.to_datetime(times[:, j], utc=True, unit="ns")
            for j, name in enumerate(
                ("kickoff_utc", "history_max_available_at", "h_last_available_at", "a_last_available_at")
            )
        },
        index=index,
    )
    provenance["history_rows_released"] = counts
    X = pd.DataFrame(output, index=index, columns=LAST_COLUMNS)
    result = FeatureDataset(X, provenance, FeatureMetadata.for_frame(X, "last_match", policy))
    result.assert_valid()
    return result
