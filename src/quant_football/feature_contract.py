"""Immutable, serializable producer contract for a feature matrix."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields

import pandas as pd

from .config import AvailabilityPolicy, FeatureConfig


@dataclass(frozen=True)
class FeatureMetadata:
    builder: str
    policy: AvailabilityPolicy
    config: FeatureConfig | None
    columns: tuple[str, ...]
    dtypes: tuple[str, ...]

    def __post_init__(self):
        event_builder = re.fullmatch(r"history_stats_events_v1:[0-9a-f]{64}", self.builder) is not None
        if self.builder not in ("history", "last_match", "history_stats_v1") and not event_builder:
            raise ValueError("Unknown feature builder")
        if not isinstance(self.policy, AvailabilityPolicy):
            raise ValueError("Feature availability policy is required")
        if (
            (self.builder in ("history", "history_stats_v1") or event_builder)
            and not isinstance(self.config, FeatureConfig)
        ) or (self.builder == "last_match" and self.config is not None):
            raise ValueError("Feature configuration does not match builder")
        if not isinstance(self.columns, tuple) or not isinstance(self.dtypes, tuple):
            raise ValueError("Feature schema must be immutable tuples")
        if (
            not self.columns
            or len(self.columns) != len(self.dtypes)
            or len(set(self.columns)) != len(self.columns)
        ):
            raise ValueError("Invalid feature schema")
        if any(not isinstance(v, str) or not v for v in (*self.columns, *self.dtypes)):
            raise ValueError("Invalid schema names/types")

    @classmethod
    def for_frame(cls, frame, builder, policy, config=None):
        return cls(builder, policy, config, tuple(frame.columns), tuple(map(str, frame.dtypes)))

    def validate_frame(self, frame: pd.DataFrame) -> None:
        if tuple(frame.columns) != self.columns or tuple(map(str, frame.dtypes)) != self.dtypes:
            raise ValueError("Feature schema differs from its producer metadata")

    def to_dict(self) -> dict:
        # JSON-native lists, independent of caller serialization choices.
        return json.loads(
            json.dumps(
                {
                    "version": 1,
                    "builder": self.builder,
                    "policy": asdict(self.policy),
                    "config": None if self.config is None else asdict(self.config),
                    "columns": self.columns,
                    "dtypes": self.dtypes,
                }
            )
        )

    @classmethod
    def from_dict(cls, value: dict):
        expected = {"version", "builder", "policy", "config", "columns", "dtypes"}
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or type(value["version"]) is not int
            or value["version"] != 1
        ):
            raise ValueError("Invalid feature metadata schema/version")
        policy = value["policy"]
        if not isinstance(policy, dict) or set(policy) != {"lag_hours"}:
            raise ValueError("Invalid feature availability policy")
        config = value["config"]
        if config is not None:
            if not isinstance(config, dict) or set(config) != {f.name for f in fields(FeatureConfig)}:
                raise ValueError("Invalid feature configuration inventory")
            config = dict(config)
            for name in ("windows", "venue_windows", "ewm_half_lives"):
                if not isinstance(config[name], list):
                    raise ValueError("Feature windows must be JSON lists")
                config[name] = tuple(config[name])
            config = FeatureConfig(**config)
        if not isinstance(value["columns"], list) or not isinstance(value["dtypes"], list):
            raise ValueError("Feature schema must contain JSON lists")
        return cls(
            value["builder"],
            AvailabilityPolicy(**policy),
            config,
            tuple(value["columns"]),
            tuple(value["dtypes"]),
        )

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
