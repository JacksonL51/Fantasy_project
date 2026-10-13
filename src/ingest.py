import os
import time
from typing import Iterable, List

import pandas as pd
from nba_api.stats.endpoints import commonteamroster, leaguedashplayerstats
from nba_api.stats.static import teams
from nba_api.stats.library.parameters import Season
from nba_api.stats.library import http as stats_http
from nba_api.stats.endpoints import commonallplayers, commonteamroster, leaguedashplayerstats

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
HISTORICAL_STATS_PATH = os.path.join(RAW_DATA_DIR, "historical_stats.csv")
CURRENT_ROSTERS_PATH = os.path.join(RAW_DATA_DIR, "current_rosters.csv")

# Custom browser headers to avoid stats.nba.com silent drops
CUSTOM_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Pragma": "no-cache",
    "Referer": "https://www.nba.com/",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}

# Apply headers globally across nba_api
stats_http.STATS_HEADERS = CUSTOM_HEADERS
stats_http.NBAStatsHTTP.headers = CUSTOM_HEADERS

REQUIRED_HISTORICAL_COLUMNS = {
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "GP",
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FTM",
    "FTA",
    "REB",
    "AST",
    "STL",
    "BLK",
    "PTS",
    "TOV",
}

REQUIRED_ROSTER_COLUMNS = {"PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"}

SELECTED_HISTORICAL_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "SEASON",
    "AGE",
    "GP",
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FTM",
    "FTA",
    "REB",
    "AST",
    "STL",
    "BLK",
    "PTS",
    "TOV",
]

SELECTED_ROSTER_COLUMNS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "SEASON"]


def ensure_dirs() -> None:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip().upper().replace(" ", "_").replace("-", "_") for col in cleaned.columns]

    if "MP" in cleaned.columns and "MIN" not in cleaned.columns:
        cleaned = cleaned.rename(columns={"MP": "MIN"})
    if "TOV" not in cleaned.columns and "TO" in cleaned.columns:
        cleaned = cleaned.rename(columns={"TO": "TOV"})
    if "FG_PCT" in cleaned.columns and "FG_PERCENTAGE" not in cleaned.columns:
        cleaned = cleaned.rename(columns={"FG_PCT": "FG_PERCENTAGE"})
    if "FT_PCT" in cleaned.columns and "FT_PERCENTAGE" not in cleaned.columns:
        cleaned = cleaned.rename(columns={"FT_PCT": "FT_PERCENTAGE"})

    return cleaned


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for col in columns:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    return cleaned


def _validate_columns(df: pd.DataFrame, required_columns: set, label: str) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _load_cached_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    return df


def _get_current_roster_player_ids() -> set[int] | None:
    if not os.path.exists(CURRENT_ROSTERS_PATH):
        return None

    roster_df = _load_cached_dataframe(CURRENT_ROSTERS_PATH)
    if roster_df.empty or "PLAYER_ID" not in roster_df.columns:
        return None

    roster_df["PLAYER_ID"] = pd.to_numeric(roster_df["PLAYER_ID"], errors="coerce")
    roster_ids = roster_df["PLAYER_ID"].dropna().astype(int).unique().tolist()
    return set(roster_ids)


def _fetch_with_retry(operation, label: str, max_retries: int = 3, base_delay: float = 2.0):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as exc:  # pragma: no cover - defensive network retry path
            last_error = exc
            if attempt == max_retries:
                raise
            delay = base_delay * attempt
            print(f"{label} attempt {attempt}/{max_retries} failed: {exc}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise last_error


def fetch_historical_stats(
    seasons: List[str] | None = None,
    force_refresh: bool = False,
    roster_player_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Fetches multi-season player box-score totals and stores a normalized raw snapshot to data/raw/historical_stats.csv.
    If a cache already exists and force_refresh is False, the cached file is reused.
    Optionally filters the saved dataset to the current roster player IDs.
    """
    ensure_dirs()

    if seasons is None:
        seasons = ["2022-23", "2023-24", "2024-25", "2025-26"]

    if roster_player_ids is None:
        roster_player_ids = _get_current_roster_player_ids()

    if not force_refresh and os.path.exists(HISTORICAL_STATS_PATH):
        print(f"Loading cached historical stats from {HISTORICAL_STATS_PATH}...")
        cached_df = _load_cached_dataframe(HISTORICAL_STATS_PATH)
        if not cached_df.empty:
            cleaned_df = _normalize_columns(cached_df)
            if roster_player_ids is not None:
                cleaned_df["PLAYER_ID"] = pd.to_numeric(cleaned_df["PLAYER_ID"], errors="coerce")
                cleaned_df = cleaned_df[cleaned_df["PLAYER_ID"].isin(roster_player_ids)].copy()
                cleaned_df.to_csv(HISTORICAL_STATS_PATH, index=False)
            return cleaned_df

    all_dfs = []
    for season in seasons:
        print(f"Fetching stats for season {season}...")
        try:
            season_df = _fetch_with_retry(
                lambda: leaguedashplayerstats.LeagueDashPlayerStats(
                    season=season,
                    per_mode_detailed="Totals",
                    timeout=60,
                ).get_data_frames()[0],
                label=f"Historical stats for season {season}",
            )
            season_df = _normalize_columns(season_df)
            season_df["SEASON"] = season
            _validate_columns(season_df, REQUIRED_HISTORICAL_COLUMNS | {"SEASON"}, f"Historical stats for {season}")
            all_dfs.append(season_df)
            time.sleep(1.5)
        except Exception as exc:
            print(f"Error fetching season {season}: {exc}")

    if not all_dfs:
        print("No historical stats fetched.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = _coerce_numeric_columns(
        combined_df,
        [
            "PLAYER_ID",
            "TEAM_ID",
            "GP",
            "MIN",
            "FGM",
            "FGA",
            "FG3M",
            "FTM",
            "FTA",
            "REB",
            "AST",
            "STL",
            "BLK",
            "PTS",
            "TOV",
            "AGE",
        ],
    )
    combined_df = combined_df.drop_duplicates(subset=["PLAYER_ID", "SEASON"], keep="last").reset_index(drop=True)

    if roster_player_ids is not None:
        combined_df["PLAYER_ID"] = pd.to_numeric(combined_df["PLAYER_ID"], errors="coerce")
        combined_df = combined_df[combined_df["PLAYER_ID"].isin(roster_player_ids)].copy()

    # Keep only player and stat fields needed for the fantasy model.
    selected_columns = [col for col in SELECTED_HISTORICAL_COLUMNS if col in combined_df.columns]
    combined_df = combined_df[selected_columns]
    combined_df.to_csv(HISTORICAL_STATS_PATH, index=False)
    print(f"Saved historical stats to {HISTORICAL_STATS_PATH} ({len(combined_df)} rows).")
    return combined_df


def fetch_current_rosters(season: str = "2026-27", force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches active roster assignments in a single API call and caches the result.
    """
    ensure_dirs()

    if not force_refresh and os.path.exists(CURRENT_ROSTERS_PATH):
        print(f"Loading cached current rosters from {CURRENT_ROSTERS_PATH}...")
        cached_df = _load_cached_dataframe(CURRENT_ROSTERS_PATH)
        if not cached_df.empty:
            return _normalize_columns(cached_df)

    print(f"Fetching active rosters for season {season} via CommonAllPlayers...")

    raw_df = _fetch_with_retry(
        lambda: commonallplayers.CommonAllPlayers(
            is_only_current_season=1,
            season=season,
            timeout=60,
        ).get_data_frames()[0],
        label="Active rosters fetch",
    )

    df = _normalize_columns(raw_df)

    # Filter out free agents / unassigned players (TEAM_ID == 0)
    df = df[df["TEAM_ID"] != 0].copy()

    # Normalize column names to match REQUIRED_ROSTER_COLUMNS
    column_mappings = {
        "PERSON_ID": "PLAYER_ID",
        "DISPLAY_FIRST_LAST": "PLAYER_NAME",
        "TEAM_ABBREVIATION": "TEAM_ABBREVIATION",
        "TEAM_NAME": "TEAM_NAME",
    }
    df = df.rename(columns=column_mappings)
    df["SEASON"] = season

    _validate_columns(df, REQUIRED_ROSTER_COLUMNS | {"SEASON"}, "Active rosters")

    df = df[list(REQUIRED_ROSTER_COLUMNS) + ["SEASON"]]
    df = df.drop_duplicates(subset=["PLAYER_ID", "SEASON"], keep="last").reset_index(drop=True)

    df = df[SELECTED_ROSTER_COLUMNS]
    df.to_csv(CURRENT_ROSTERS_PATH, index=False)
    print(f"Saved current rosters to {CURRENT_ROSTERS_PATH} ({len(df)} rows).")
    return df

def tidy_raw_stats(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    # 1. Round minutes to 1 decimal place
    if "MIN" in cleaned.columns:
        cleaned["MIN"] = cleaned["MIN"].round(1)

    # 2. Filter out extreme small-sample noise (optional, but cleans up garbage rows)
    # e.g., require at least 5 games or 50 minutes played
    cleaned = cleaned[cleaned["MIN"] >= 50.0]

    # 3. Drop redundant IDs not needed for rate calculations
    columns_to_drop = ["TEAM_ID"]
    cleaned = cleaned.drop(columns=[col for col in columns_to_drop if col in cleaned.columns])

    # 4. Enforce integer types on counting stats
    int_cols = ["GP", "FGM", "FGA", "FG3M", "FTM", "FTA", "REB", "AST", "STL", "BLK", "PTS", "TOV"]
    for col in int_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna(0).astype(int)

    return cleaned


if __name__ == "__main__":
    print("Starting data ingestion phase...")
    roster_df = fetch_current_rosters()
    fetch_historical_stats(roster_player_ids=roster_df["PLAYER_ID"].unique())
    print("Data ingestion complete.")
