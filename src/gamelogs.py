"""Player game logs and the league schedule (nba_api, one request per season each).

- Game logs: data/raw/gamelogs/gamelog_YYYY.csv (YYYY = season end year), regular season,
  one row per player per game played. Completed seasons are fetched once and cached; the
  current season is refreshed every run. Used to measure how teammates' minutes and usage
  change when a player sits (src/opportunity.py) and to count games already missed.
- Schedule: data/raw/schedule_YYYY.csv for the current season, regular-season games only
  (GAME_ID starting 002), with home and away team abbreviations.

A failed fetch keeps the saved file. `python src/gamelogs.py` refreshes both.
"""

from __future__ import annotations

import os
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, scheduleleaguev2

import ingest  # noqa: F401  (applies the stats.nba.com headers)
from ingest import _fetch_with_retry

RAW_DATA_DIR = os.path.join("data", "raw")
GAMELOG_DIR = os.path.join(RAW_DATA_DIR, "gamelogs")
CURRENT_SEASON = "2026-27"
FIRST_GAMELOG_END_YEAR = 2014  # matches the held-out window used elsewhere (2014-26)
GAMELOG_COLUMNS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE",
    "MATCHUP", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PTS",
]
SCHEDULE_COLUMNS = ["GAME_ID", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM"]
REQUEST_PAUSE_SECONDS = 1.5


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{end_year % 100:02d}"


def season_end_year(season: str) -> int:
    return int(season[:4]) + 1


def gamelog_path(end_year: int) -> str:
    return os.path.join(GAMELOG_DIR, f"gamelog_{end_year}.csv")


def schedule_path(end_year: int) -> str:
    return os.path.join(RAW_DATA_DIR, f"schedule_{end_year}.csv")


def fetch_gamelog(end_year: int, force: bool = False) -> pd.DataFrame:
    path = gamelog_path(end_year)
    current = end_year == season_end_year(CURRENT_SEASON)
    if os.path.exists(path) and not force and not current:
        return pd.read_csv(path)
    os.makedirs(GAMELOG_DIR, exist_ok=True)
    try:
        raw = _fetch_with_retry(
            lambda: leaguegamelog.LeagueGameLog(
                player_or_team_abbreviation="P",
                season=season_label(end_year),
                season_type_all_star="Regular Season",
                timeout=60,
            ).get_data_frames()[0],
            label=f"Game log {season_label(end_year)}",
        )
        missing = set(GAMELOG_COLUMNS) - set(raw.columns)
        if missing or raw.empty:
            raise ValueError(f"game log missing columns {sorted(missing)} or empty")
    except Exception as exc:
        if os.path.exists(path):
            print(f"Game log fetch failed ({exc}); keeping {path}.")
            return pd.read_csv(path)
        raise
    df = raw[GAMELOG_COLUMNS].copy()
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df)} player-games, {df['GAME_ID'].nunique()} games).")
    return df


def fetch_all_gamelogs(first_end_year: int = FIRST_GAMELOG_END_YEAR) -> None:
    last = season_end_year(CURRENT_SEASON)
    for end_year in range(first_end_year, last + 1):
        cached = os.path.exists(gamelog_path(end_year)) and end_year != last
        fetch_gamelog(end_year)
        if not cached:
            time.sleep(REQUEST_PAUSE_SECONDS)


def load_gamelogs(end_years) -> pd.DataFrame:
    frames = []
    for end_year in end_years:
        path = gamelog_path(end_year)
        if os.path.exists(path):
            frames.append(pd.read_csv(path).assign(SEASON_END=end_year))
    if not frames:
        return pd.DataFrame(columns=GAMELOG_COLUMNS + ["SEASON_END"])
    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    return df


def fetch_schedule(season: str = CURRENT_SEASON) -> pd.DataFrame:
    end_year = season_end_year(season)
    path = schedule_path(end_year)
    try:
        raw = _fetch_with_retry(
            lambda: scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=60).get_data_frames()[0],
            label=f"Schedule {season}",
        )
        df = pd.DataFrame(
            {
                "GAME_ID": raw["gameId"].astype(str),
                "GAME_DATE": pd.to_datetime(raw["gameDateEst"]).dt.date,
                "HOME_TEAM": raw["homeTeam_teamTricode"],
                "AWAY_TEAM": raw["awayTeam_teamTricode"],
            }
        )
        df = df[df["GAME_ID"].str.startswith("002")].drop_duplicates("GAME_ID")
        if len(df) < 1000:
            raise ValueError(f"only {len(df)} regular-season games in the schedule")
    except Exception as exc:
        if os.path.exists(path):
            print(f"Schedule fetch failed ({exc}); keeping {path}.")
            return load_schedule(season)
        raise
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df)} regular-season games).")
    return df


def load_schedule(season: str = CURRENT_SEASON) -> pd.DataFrame:
    path = schedule_path(season_end_year(season))
    if not os.path.exists(path):
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    df = pd.read_csv(path, dtype={"GAME_ID": str})
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    return df


def team_game_dates(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: TEAM_ABBREVIATION, GAME_DATE, GAME_ID."""
    home = schedule.rename(columns={"HOME_TEAM": "TEAM_ABBREVIATION"})[["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]]
    away = schedule.rename(columns={"AWAY_TEAM": "TEAM_ABBREVIATION"})[["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]]
    return pd.concat([home, away], ignore_index=True).sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])


if __name__ == "__main__":
    fetch_all_gamelogs()
    fetch_schedule()
