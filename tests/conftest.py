import numpy as np
import pandas as pd
import pytest

from quant_football.config import METADATA_COLUMNS, TARGETS


def make_matches(rows):
    result = []
    for source in rows:
        d = dict(source)
        home, away = d.get("home_team_id", 1), d.get("away_team_id", 2)
        row = {
            "match_id": d["match_id"],
            "kickoff_utc": d["kickoff_utc"],
            "league_id": 10,
            "league_name": "Test League",
            "league_country": "Test Country",
            "league_type": "league",
            "season_id": 100.0,
            "round": 1.0,
            "home_team_id": home,
            "home_team_name": f"Team {home}",
            "away_team_id": away,
            "away_team_name": f"Team {away}",
            "home_goals_90": 2.0,
            "away_goals_90": 1.0,
            "home_corners_90": 6.0,
            "away_corners_90": 4.0,
            "extra_time": False,
        }
        row.update(d)
        result.append(row)
    frame = pd.DataFrame(result, columns=[*METADATA_COLUMNS, *TARGETS, "extra_time"])
    for name in ("match_id", "league_id", "home_team_id", "away_team_id"):
        frame[name] = frame[name].astype("int64")
    for name in (*TARGETS, "season_id", "round"):
        frame[name] = frame[name].astype("float64")
    frame["extra_time"] = frame["extra_time"].astype(bool)
    frame["kickoff_utc"] = pd.to_datetime(frame.kickoff_utc, utc=True, format="mixed")
    return frame


@pytest.fixture
def history():
    return make_matches(
        [
            {"match_id": 1, "kickoff_utc": "2025-01-01T12:00:00Z"},
            {
                "match_id": 2,
                "kickoff_utc": "2025-01-02T12:00:00Z",
                "away_team_id": 3,
                "home_goals_90": 0.0,
                "away_goals_90": 2.0,
                "home_corners_90": np.nan,
                "away_corners_90": np.nan,
            },
            {
                "match_id": 3,
                "kickoff_utc": "2025-01-03T12:00:00Z",
                "home_team_id": 4,
                "away_team_id": 1,
                "home_goals_90": 1.0,
                "away_goals_90": 3.0,
                "home_corners_90": 5.0,
                "away_corners_90": 7.0,
            },
            {
                "match_id": 4,
                "kickoff_utc": "2025-01-03T12:00:00Z",
                "away_team_id": 5,
                "home_goals_90": 2.0,
                "away_goals_90": 0.0,
                "home_corners_90": 3.0,
                "away_corners_90": 2.0,
            },
            {"match_id": 5, "kickoff_utc": "2025-01-06T13:00:00Z"},
        ]
    )
