"""Frozen provider-specific proxies and bounded SQL aggregation of delivered messages.

Codes are statistical hypotheses, not asserted football semantics. Providers stay
separate; only identical messages are duplicates. Raw counts select candidates on
old rows, then exact deduplication validates the chosen code before it can be used.
"""

import hashlib
import json
from collections import deque
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import EVENT_COLUMNS, TARGETS, AvailabilityPolicy, FeatureConfig
from .contracts import DataContractError, normalize_matches, schedule_only, utc_timestamp
from .feature_contract import FeatureMetadata
from .features import FeatureDataset
from .open_stats import build_open_features

FEEDS = ("feed1", "feed2", "feed3")
FAMILIES = ("goals", "corners", "shots", "sot")
PROXY_TARGETS = {
    "goals": ("home_goals_90", "away_goals_90"),
    "corners": ("home_corners_90", "away_corners_90"),
    "shots": ("shots_h", "shots_a"),
    "sot": ("sot_h", "sot_a"),
}
BUILDER_PREFIX = "history_stats_events_v1:"
SUMMARY = ("last10_mean", "last10_n", "ewm5_mean", "ewm5_n")


def event_columns():
    return tuple(
        f"{team}_{feed}_{family}_proxy_{direction}_{summary}"
        for team in ("h", "a")
        for feed in FEEDS
        for family in FAMILIES
        for direction in ("for", "against")
        for summary in SUMMARY
    )


def _connect():
    con = duckdb.connect()
    con.execute("SET threads=6")
    con.execute("SET TimeZone='UTC'")
    con.execute("SET memory_limit='2GB'")
    return con


def aggregate_feed(path, history, feed, *, codes=None, policy=None):
    """Return compact counts and coverage; raw events never materialize in pandas.

    codes=None gives raw first-pass counts. A code list gives exact-deduplicated
    counts and a duplicate audit. Sequence is part of a message, not a unique key.
    Missing files are an error; an empty correctly typed feed is supported.
    """
    if feed not in FEEDS:
        raise ValueError("Unknown provider")
    history = normalize_matches(history, "open")
    policy = policy or AvailabilityPolicy()
    schema = pq.read_schema(path)
    if set(EVENT_COLUMNS) - set(schema.names):
        raise DataContractError("Missing event columns")
    if not pa.types.is_integer(schema.field("match_id").type):
        raise DataContractError("Event match_id must be integer")
    for name in ("event_time", "clock_time"):
        if not pa.types.is_timestamp(schema.field(name).type):
            raise DataContractError("Event timestamps must be timestamps")
    for name in ("event", "minute", "period", "side"):
        if not (
            pa.types.is_string(schema.field(name).type)
            or pa.types.is_large_string(schema.field(name).type)
            or pa.types.is_null(schema.field(name).type)
        ):
            raise DataContractError("Event labels must be strings")
    meta = history[["match_id", "kickoff_utc", "extra_time", *TARGETS]].copy()
    meta["release"] = meta.kickoff_utc + policy.lag
    con = _connect()
    try:
        con.register("history", meta)
        con.read_parquet(str(path)).create_view("events")
        invalid = con.execute("""SELECT count(*) FROM events e LEFT JOIN history h USING(match_id)
            WHERE h.match_id IS NULL OR event_time IS NULL OR clock_time IS NULL OR event IS NULL
            OR (side IS NOT NULL AND side NOT IN ('home','away'))""").fetchone()[0]
        if invalid:
            raise DataContractError("Unknown match, missing timestamp/code or invalid side in event feed")
        period = "AND e.period IN ('first_half','second_half')" if feed == "feed3" else ""
        # All-null outcomes do not certify completion; unknown/ET data never become zeros.
        con.execute(f"""CREATE TEMP VIEW delivered AS SELECT e.* FROM events e JOIN history h USING(match_id)
            WHERE NOT h.extra_time
            AND (home_goals_90 IS NOT NULL OR away_goals_90 IS NOT NULL
                 OR home_corners_90 IS NOT NULL OR away_corners_90 IS NOT NULL)
            AND e.event_time>=h.kickoff_utc AND e.clock_time>=h.kickoff_utc
            AND e.event_time<h.release AND e.clock_time<h.release {period}""")
        # Parse elapsed minute MM:SS (including M>=90). Unknown formats cannot certify coverage.
        coverage = (
            con.sql("""SELECT match_id,count(*) AS delivered_rows,
            max(try_cast(regexp_extract(minute,'^([0-9]+):[0-5][0-9]$',1) AS INTEGER)) AS last_minute,
            min(try_cast(regexp_extract(minute,'^([0-9]+):[0-5][0-9]$',1) AS INTEGER)) AS first_minute,
            max(event_time) AS last_event_time,max(clock_time) AS last_clock_time,
            count(*) FILTER(WHERE side='home') AS home_messages,
            count(*) FILTER(WHERE side='away') AS away_messages
            FROM delivered GROUP BY match_id ORDER BY match_id""")
            .to_arrow_table()
            .to_pandas()
        )
        coverage["complete"] = (
            coverage.last_minute.ge(85)
            & coverage.first_minute.le(5)
            & coverage.home_messages.gt(0)
            & coverage.away_messages.gt(0)
        )
        con.register("coverage", coverage)
        extra_join, expression = "", "count(*)"
        if codes is not None:
            codes = sorted(set(codes))
            if any(not isinstance(v, str) or not v for v in codes):
                raise ValueError("Invalid event codes")
            con.register("selected_codes", pd.DataFrame({"event": pd.Series(codes, dtype="str")}))
            extra_join = "JOIN selected_codes USING(event)"
            expression = (
                "count(DISTINCT struct_pack(event_time,clock_time,minute,period,seq,event,side,zone))"
            )
        counts = (
            con.sql(f"""SELECT match_id,event,side,{expression} AS n,count(*) AS raw_n
            FROM delivered JOIN coverage USING(match_id) {extra_join}
            WHERE complete AND side IN ('home','away') GROUP BY 1,2,3 ORDER BY 1,2,3""")
            .to_arrow_table()
            .to_pandas()
        )
        audit = {
            "feed": feed,
            "complete_matches": int(coverage.complete.sum()),
            "delivered_matches": len(coverage),
            "count_rows": len(counts),
            "exact_duplicate_rows_removed": int((counts.raw_n - counts.n).sum()),
            "deduplicated": codes is not None,
            "selected_codes": codes,
        }
        return counts, coverage, audit
    finally:
        con.close()


def code_diagnostics(counts, coverage, history, cutoff, *, candidates=None, policy=None):
    """Fit-time selection statistics; no future labels or coverage enter this fit."""
    policy, cutoff = policy or AvailabilityPolicy(), utc_timestamp(cutoff)
    history = normalize_matches(history, "open")
    complete = coverage.loc[coverage.complete, "match_id"]
    old = history.loc[(history.kickoff_utc + policy.lag < cutoff) & history.match_id.isin(complete)]
    old = old.set_index("match_id")
    counts = counts.loc[counts.match_id.isin(old.index)]
    records = []
    for side_index, side in enumerate(("home", "away")):
        table = counts.loc[counts.side.eq(side)].pivot(index="match_id", columns="event", values="n")
        table = table.reindex(old.index).fillna(0)
        for family in FAMILIES:
            name = PROXY_TARGETS[family][side_index]
            if name not in old:
                continue
            y = pd.to_numeric(old[name], errors="raise").astype(float)
            valid = y.notna()
            if not np.isfinite(y[valid]).all() or (y[valid] < 0).any():
                raise DataContractError("Invalid proxy calibration target")
            if int(valid.sum()) < 200 or y[valid].std() == 0:
                continue
            allowed = table.columns if candidates is None else candidates.get(f"{family}_{side}", [])
            for code in allowed:
                if code not in table:
                    continue
                x, v = table.loc[valid, code].to_numpy(), y[valid].to_numpy()
                if int((x > 0).sum()) < 100 or np.std(x) == 0:
                    continue
                corr = float(np.corrcoef(x, v)[0, 1])
                mae, exact = float(np.mean(abs(x - v))), float(np.mean(x == v))
                # Statistical proxies may differ from true outcomes. Strict semantic
                # validation is reported separately and never assumed from correlation.
                min_corr, max_mae = (0.80, 1.0) if family in ("goals", "corners") else (0.60, 5.0)
                records.append(
                    {
                        "family": family,
                        "side": side,
                        "event": str(code),
                        "n": len(v),
                        "nonzero": int((x > 0).sum()),
                        "corr": corr,
                        "mae": mae,
                        "exact": exact,
                        "eligible_proxy": corr >= min_corr and mae <= max_mae,
                        "strict_semantic_match": mae <= 0.10 and exact >= 0.95,
                    }
                )
    return sorted(records, key=lambda r: (r["family"], r["side"], r["mae"], -r["corr"], r["event"]))


@dataclass(frozen=True)
class Codebook:
    cutoff: str
    # (feed, family, side, event), one code per family/side/provider at most.
    entries: tuple[tuple[str, str, str, str], ...]
    lag_hours: float = 24.0

    def __post_init__(self):
        utc_timestamp(self.cutoff)
        AvailabilityPolicy(self.lag_hours)
        if not isinstance(self.entries, tuple) or any(
            not isinstance(e, tuple) or len(e) != 4 for e in self.entries
        ):
            raise ValueError("Codebook entries must be immutable four-tuples")
        keys = []
        for feed, family, side, code in self.entries:
            if (
                any(not isinstance(v, str) or not v for v in (feed, family, side, code))
                or feed not in FEEDS
                or family not in FAMILIES
                or side not in ("home", "away")
            ):
                raise ValueError("Invalid event codebook")
            keys.append((feed, family, side))
        if len(set(keys)) != len(keys) or tuple(sorted(self.entries)) != self.entries:
            raise ValueError("Duplicate or unsorted event codebook")

    def to_dict(self):
        return {
            "version": 1,
            "cutoff": self.cutoff,
            "lag_hours": self.lag_hours,
            "entries": [list(e) for e in self.entries],
        }

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def choose_codes(diagnostics):
    """One candidate per slot; exact ranking ties remain unassigned."""
    chosen = {}
    for r in diagnostics:
        if r["eligible_proxy"]:
            chosen.setdefault(f"{r['family']}_{r['side']}", []).append(r)
    result = {}
    for key, rows in chosen.items():
        rows = sorted(rows, key=lambda r: (r["mae"], -r["exact"], -r["corr"], r["event"]))
        if len(rows) > 1 and all(rows[0][k] == rows[1][k] for k in ("mae", "exact", "corr")):
            continue
        result[key] = rows[0]
    return result


def proxy_table(aggregates, history, book):
    """Null is unavailable feed/code/side; a known complete code can have true zero."""
    history = normalize_matches(history, "open")
    result = history[["match_id"]].copy().set_index("match_id")
    if set(aggregates) != set(FEEDS):
        raise DataContractError("Exactly three provider aggregates required")
    usable = set(history.loc[~history.extra_time & ~history[list(TARGETS)].isna().all(axis=1), "match_id"])
    for feed in FEEDS:
        counts, coverage = aggregates[feed]
        if not {"match_id", "event", "side", "n"}.issubset(counts) or not {"match_id", "complete"}.issubset(
            coverage
        ):
            raise DataContractError("Missing aggregate columns")
        for frame in (counts, coverage):
            if (
                not pd.api.types.is_integer_dtype(frame.match_id)
                or frame.match_id.isna().any()
                or (frame.match_id <= 0).any()
            ):
                raise DataContractError("Invalid aggregate match ids")
        if counts.duplicated(["match_id", "event", "side"]).any() or coverage.match_id.duplicated().any():
            raise DataContractError("Duplicate event aggregate keys")
        if not pd.api.types.is_bool_dtype(coverage.complete) or coverage.complete.isna().any():
            raise DataContractError("Invalid aggregate coverage")
        if counts.event.isna().any() or not counts.side.isin(("home", "away")).all():
            raise DataContractError("Invalid event aggregate labels")
        try:
            values = counts.n.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataContractError("Invalid aggregate counts") from exc
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise DataContractError("Invalid aggregate counts")
        complete_ids = coverage.loc[coverage.complete & coverage.match_id.isin(usable), "match_id"]
        for family in FAMILIES:
            for side in ("home", "away"):
                name = f"{feed}_{family}_{side}"
                result[name] = np.nan
                mapping = [e[3] for e in book.entries if e[:3] == (feed, family, side)]
                if not mapping:
                    continue
                part = counts.loc[counts.event.eq(mapping[0]) & counts.side.eq(side)].set_index("match_id")
                ids = result.index.intersection(complete_ids)
                result.loc[ids, name] = part.n.reindex(ids).fillna(0).to_numpy()
    return result


class _EventTeam:
    def __init__(self):
        self.records = deque(maxlen=10)
        self.sums = np.zeros(24)
        self.weights = np.zeros(24)
        self.squared = np.zeros(24)
        self.last = None
        self.cache = None

    def add(self, values, available):
        values = np.asarray(values, dtype=float)
        self.records.append(values)
        ok = np.isfinite(values)
        decay = 0.5 ** (1 / 5)
        self.sums[ok] = decay * self.sums[ok] + values[ok]
        self.weights[ok] = decay * self.weights[ok] + 1
        self.squared[ok] = decay**2 * self.squared[ok] + 1
        if ok.any():
            self.last = available
        self.cache = None

    def describe(self):
        if self.cache is None:
            a = np.asarray(self.records).reshape(-1, 24)
            n = np.isfinite(a).sum(axis=0)
            mean = np.divide(np.nansum(a, axis=0), n, out=np.full(24, np.nan), where=n > 0)
            ewm = np.divide(self.sums, self.weights, out=np.full(24, np.nan), where=self.weights > 0)
            effective = np.divide(self.weights**2, self.squared, out=np.zeros(24), where=self.squared > 0)
            self.cache = np.column_stack((mean, n, ewm, effective)).ravel()
        return self.cache


def build_event_features(schedule, history, aggregates, book, policy=None, config=None, *, base=None):
    policy, config = policy or AvailabilityPolicy(), config or FeatureConfig()
    schedule, history = schedule_only(schedule), normalize_matches(history, "open")
    if book.lag_hours != policy.lag_hours:
        raise ValueError("Event codebook availability policy differs")
    if base is None:
        base = build_open_features(schedule, history, policy, config)
    if not base.X.index.equals(pd.Index(schedule.match_id, name="match_id")):
        raise ValueError("Base feature schedule differs")
    if (
        base.metadata.policy != policy
        or base.metadata.config != config
        or base.metadata.builder != "history_stats_v1"
    ):
        raise ValueError("Wrong base feature contract")
    proxies = proxy_table(aggregates, history, book)
    playable = history.loc[~history[list(TARGETS)].isna().all(axis=1)].copy()
    playable["release"] = playable.kickoff_utc + policy.lag
    playable = playable.sort_values(["release", "kickoff_utc", "match_id"], kind="stable")
    queue = list(
        zip(
            playable.release,
            playable.home_team_id,
            playable.away_team_id,
            proxies.loc[playable.match_id].to_numpy(),
            strict=True,
        )
    )
    numbers = np.empty((len(schedule), len(event_columns())), dtype="float32")
    times = np.full((len(schedule), 2), np.iinfo("int64").min, dtype="int64")
    positions, teams, cursor = {int(v): i for i, v in enumerate(schedule.match_id)}, {}, 0
    neutral = _EventTeam().describe()
    for kickoff, batch in schedule.groupby("kickoff_utc", sort=True):
        while cursor < len(queue) and queue[cursor][0] < kickoff:
            available, home, away, values = queue[cursor]
            for team, oriented in ((home, values), (away, values.reshape(12, 2)[:, ::-1].ravel())):
                teams.setdefault(int(team), _EventTeam()).add(oriented, available)
            cursor += 1
        for row in batch.itertuples(index=False):
            pos = positions[row.match_id]
            for j, team in enumerate((row.home_team_id, row.away_team_id)):
                state = teams.get(team)
                numbers[pos, j * 96 : (j + 1) * 96] = neutral if state is None else state.describe()
                if state is not None and state.last is not None:
                    times[pos, j] = state.last.value
    X = pd.concat([base.X, pd.DataFrame(numbers, index=base.X.index, columns=event_columns())], axis=1)
    provenance = base.provenance.copy()
    for j, side in enumerate(("h", "a")):
        provenance[f"{side}_event_last_available_at"] = pd.to_datetime(times[:, j], unit="ns", utc=True)
    metadata = FeatureMetadata.for_frame(X, BUILDER_PREFIX + book.fingerprint(), policy, config)
    result = FeatureDataset(X, provenance, metadata)
    result.assert_valid()
    return result
