"""Input contracts and schedule/outcome separation.

Only `schedule_only` may pass a match to the prediction-facing feature code.
Core/open normalization validates supplied outcomes but never fills missing ones.
"""

from collections.abc import Sequence
from numbers import Number
from typing import Literal

import numpy as np
import pandas as pd

from .config import IDENTIFIERS, METADATA_COLUMNS, MIN_SCHEDULE_COLUMNS, TARGETS


class DataContractError(ValueError):
    """A contradictory or malformed input must fail visibly."""


def utc_series(values: pd.Series, name: str) -> pd.Series:
    mixed_numeric = (
        not pd.api.types.is_datetime64_any_dtype(values.dtype)
        and values.dropna().map(lambda v: isinstance(v, Number)).any()
    )
    if pd.api.types.is_numeric_dtype(values.dtype) or mixed_numeric:
        raise DataContractError(f"{name}: numeric timestamps are ambiguous")
    try:
        result = pd.to_datetime(values, utc=True, errors="raise", format="mixed").dt.as_unit("ns")
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataContractError(f"{name}: invalid timestamp") from exc
    if result.isna().any():
        raise DataContractError(f"{name}: missing timestamps")
    return result


def utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(value, Number):
        raise DataContractError("Numeric timestamps are ambiguous")
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise DataContractError("Timestamp must not be NaT")
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _base(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.columns.is_unique:
        raise DataContractError("Duplicate column names")
    missing = set(MIN_SCHEDULE_COLUMNS) - set(frame.columns)
    if missing:
        raise DataContractError(f"Missing schedule columns: {sorted(missing)}")
    out = frame.copy()
    for name in IDENTIFIERS:
        s = out[name]
        if s.isna().any() or not pd.api.types.is_integer_dtype(s.dtype):
            raise DataContractError(f"{name}: identifiers must be non-null integers")
        if len(s) and ((s <= 0).any() or (s > np.iinfo(np.int64).max).any()):
            raise DataContractError(f"{name}: identifiers must fit positive int64")
        out[name] = s.astype("int64")
    if out.match_id.duplicated().any():
        bad = out.loc[out.match_id.duplicated(keep=False), "match_id"].head().tolist()
        raise DataContractError(f"Duplicate match_id values: {bad}")
    out["kickoff_utc"] = utc_series(out.kickoff_utc, "kickoff_utc")
    if out.home_team_id.eq(out.away_team_id).any():
        raise DataContractError("A match cannot have the same team on both sides")
    return out


def schedule_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Project BEFORE validation: own outcomes/events/stats cannot enter features.

    Minimal schedule columns are required. Optional context is explicitly
    allow-listed. Future metadata is considered known only as provided by task.
    """
    if not frame.columns.is_unique:
        raise DataContractError("Duplicate column names")
    columns = [c for c in METADATA_COLUMNS if c in frame.columns]
    out = _base(frame.loc[:, columns])
    for c in METADATA_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan if c in ("round", "season_id") else None
    for c in ("round", "season_id"):
        try:
            out[c] = pd.to_numeric(out[c], errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            raise DataContractError(f"{c}: expected numeric metadata") from exc
        if np.isinf(out[c]).any():
            raise DataContractError(f"{c}: infinite metadata")
    for c in ("league_name", "league_country", "league_type", "home_team_name", "away_team_name"):
        valid = out[c].dropna()
        if not valid.map(lambda x: isinstance(x, str)).all():
            raise DataContractError(f"{c}: expected string metadata")
    return out.loc[:, list(METADATA_COLUMNS)].reset_index(drop=True)


def normalize_matches(frame: pd.DataFrame, kind: Literal["core", "open"]) -> pd.DataFrame:
    if kind not in ("core", "open"):
        raise ValueError("kind must be core or open")
    if not frame.columns.is_unique:
        raise DataContractError("Duplicate column names")
    required = set(METADATA_COLUMNS) | set(TARGETS)
    if kind == "open":
        required.add("extra_time")
    missing = required - set(frame.columns)
    if missing:
        raise DataContractError(f"{kind}: missing columns {sorted(missing)}")
    out = _base(frame).reset_index(drop=True)
    schedule = schedule_only(out)
    for c in METADATA_COLUMNS:
        out[c] = schedule[c]
    for c in TARGETS:
        if not pd.api.types.is_numeric_dtype(out[c].dtype):
            raise DataContractError(f"{c}: expected numeric outcomes")
        values = out[c].dropna().to_numpy(dtype="float64")
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise DataContractError(f"{c}: outcomes must be finite non-negative integers or null")
        out[c] = out[c].astype("float64")
    if kind == "open":
        if out.extra_time.isna().any() or not pd.api.types.is_bool_dtype(out.extra_time.dtype):
            raise DataContractError("extra_time must be non-null boolean")
        out["extra_time"] = out.extra_time.astype(bool)
    return out


def assert_core_open_consistency(core: pd.DataFrame, history: pd.DataFrame) -> None:
    """The task promises core is a subset of open; contradictions are errors."""
    if not core.match_id.isin(history.match_id).all():
        raise DataContractError("core contains match_id absent from open")
    left = core.set_index("match_id")
    right = history.set_index("match_id").reindex(left.index)
    for c in (*METADATA_COLUMNS[1:], *TARGETS):
        a, b = left[c], right[c]
        different = ~(a.eq(b) | (a.isna() & b.isna()))
        if different.any():
            raise DataContractError(f"core/open disagree in {c}: {int(different.sum())} rows")


def merge_histories(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Merge overlapping open files once; conflicting non-null values fail.

    Missing fields may be complemented; changed non-null outcomes are never
    silently preferred. Exact repeated input is idempotent.
    """
    if not frames:
        raise ValueError("At least one history frame is required")
    normalized = [normalize_matches(f, "open") for f in frames]
    combined = pd.concat(normalized, ignore_index=True, sort=False)
    duplicate = combined.match_id.duplicated(keep=False)
    if not duplicate.any():
        return normalize_matches(combined, "open")
    unique = combined.loc[~duplicate]
    repeated = combined.loc[duplicate]
    for column in repeated.columns:
        if column == "match_id":
            continue
        counts = repeated.groupby("match_id", sort=False)[column].nunique(dropna=True)
        if counts.gt(1).any():
            bad = counts.index[counts.gt(1)].tolist()[:5]
            raise DataContractError(f"Conflicting repeated history in {column}: {bad}")
    coalesced = repeated.groupby("match_id", sort=False, as_index=False).first()
    return normalize_matches(pd.concat([unique, coalesced], ignore_index=True, sort=False), "open")
