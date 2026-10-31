"""Explicit, immutable research settings; no learned parameters live here."""

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TRAIN_CUTOFF = pd.Timestamp("2026-03-01T00:00:00Z")
TARGETS = ("home_goals_90", "away_goals_90", "home_corners_90", "away_corners_90")
METRICS = ("gf", "ga", "cf", "ca")
IDENTIFIERS = ("match_id", "league_id", "home_team_id", "away_team_id")
MIN_SCHEDULE_COLUMNS = (*IDENTIFIERS, "kickoff_utc")
METADATA_COLUMNS = (
    "match_id",
    "kickoff_utc",
    "league_id",
    "league_name",
    "league_country",
    "league_type",
    "season_id",
    "round",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
)
INPUT_FILES = (
    "core_matches.parquet",
    "open_matches.parquet",
    "feed1_events.parquet",
    "feed2_events.parquet",
    "feed3_events.parquet",
)
EVENT_COLUMNS = (
    "match_id",
    "event_time",
    "clock_time",
    "minute",
    "period",
    "seq",
    "event",
    "side",
    "zone",
)


@dataclass(frozen=True)
class AvailabilityPolicy:
    # Publication timestamps are absent from open_matches. This is an assumption,
    # NOT an observed completion/publication time. Exact equality is unavailable.
    lag_hours: float = 24.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.lag_hours) or self.lag_hours <= 0:
            raise ValueError("lag_hours must be finite and strictly positive")

    @property
    def lag(self) -> pd.Timedelta:
        return pd.Timedelta(hours=self.lag_hours)


@dataclass(frozen=True)
class FeatureConfig:
    windows: tuple[int, ...] = (5, 10, 20)
    venue_windows: tuple[int, ...] = (10, 20)
    ewm_half_lives: tuple[int, ...] = (5, 15)
    team_prior_strength: float = 10.0
    group_prior_strength: float = 50.0
    # Neutral starting constants, specified before fitting any model. Hierarchical
    # priors subsequently use only released history. They are not fit on all data.
    initial_goal_mean: float = 1.4
    initial_corner_mean: float = 4.7

    def __post_init__(self) -> None:
        for name in ("windows", "venue_windows", "ewm_half_lives"):
            values = getattr(self, name)
            if not values or any(type(v) is not int or v <= 0 for v in values):
                raise ValueError(f"{name} must contain positive integers")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be unique and increasing")
        for name in (
            "team_prior_strength",
            "group_prior_strength",
            "initial_goal_mean",
            "initial_corner_mean",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")


@dataclass(frozen=True)
class ResourceConfig:
    threads: int = 6
    duckdb_memory_limit: str = "2GB"
    arrow_batch_rows: int = 131_072


def project_root(root: str | Path) -> Path:
    """Require an explicit project directory; never depend on a developer's path."""
    path = Path(root).expanduser().resolve()
    if not (path / "data").is_dir():
        raise FileNotFoundError(f"Missing data directory: {path / 'data'}")
    return path
