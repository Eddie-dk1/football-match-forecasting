"""Создать искусственный пример для проверки запуска, не для оценки качества."""

from pathlib import Path
from sys import path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
path.insert(0, str(ROOT / "src"))

from quant_football.config import TARGETS  # noqa: E402


def main():
    data = ROOT / "data"
    if data.exists():
        raise FileExistsError("Каталог data уже существует. Пример не перезаписывает ваши данные.")
    rng = np.random.default_rng(42)
    n = 300
    home = rng.integers(1, 9, size=n)
    away = (home - 1 + rng.integers(1, 8, size=n)) % 8 + 1
    goals = rng.poisson([1.6, 1.2], size=(n, 2))
    corners = rng.poisson([5.1, 4.4], size=(n, 2))
    history = pd.DataFrame(
        {
            "match_id": np.arange(1, n + 1, dtype="int64"),
            "kickoff_utc": pd.date_range("2024-01-01", periods=n, freq="2D", tz="UTC"),
            "league_id": 1,
            "league_name": "Искусственная лига",
            "league_country": "Демонстрация",
            "league_type": "league",
            "season_id": 2024.0,
            "round": (np.arange(n) % 30 + 1).astype(float),
            "home_team_id": home,
            "away_team_id": away,
            "home_team_name": [f"Команда {x}" for x in home],
            "away_team_name": [f"Команда {x}" for x in away],
            **dict(zip(TARGETS, np.column_stack([goals, corners]).astype(float).T, strict=True)),
            "extra_time": False,
            "xg_h": rng.gamma(2, 0.8, n),
            "xg_a": rng.gamma(2, 0.6, n),
            "shots_h": rng.poisson(12, n).astype(float),
            "shots_a": rng.poisson(10, n).astype(float),
            "sot_h": rng.poisson(4, n).astype(float),
            "sot_a": rng.poisson(3, n).astype(float),
        }
    )
    future = history.tail(6).copy().reset_index(drop=True)
    future["match_id"] = np.arange(1001, 1007, dtype="int64")
    future["kickoff_utc"] = pd.date_range("2026-03-03", periods=6, freq="3D", tz="UTC")
    for name in (*TARGETS, "xg_h", "xg_a", "shots_h", "shots_a", "sot_h", "sot_a"):
        future[name] = np.nan
    # Atomic directory claim prevents overwriting an existing data directory.
    data.mkdir(exist_ok=False)
    (data / "spring").mkdir()
    history.to_parquet(data / "open_matches.parquet", index=False)
    history.iloc[::3].to_parquet(data / "core_matches.parquet", index=False)
    future.to_parquet(data / "spring/open_matches.parquet", index=False)
    future.to_parquet(data / "spring/core_matches.parquet", index=False)
    print("Искусственный пример создан: 300 исторических матчей, 100 core и 6 будущих.")
    print("Теперь выполните solution.ipynb. Этот пример не воспроизводит метрики из отчёта.")


if __name__ == "__main__":
    main()
