import os
import time
from typing import Iterable, List

import pandas as pd
from nba_api.stats.endpoints import (
    commonallplayers,
    commonplayerinfo,
    drafthistory,
    leaguedashplayerstats,
    playerindex,
)
from nba_api.stats.library.parameters import Season
from nba_api.stats.library import http as stats_http

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
HISTORICAL_STATS_PATH = os.path.join(RAW_DATA_DIR, "historical_stats.csv")
CURRENT_ROSTERS_PATH = os.path.join(RAW_DATA_DIR, "current_rosters.csv")
DRAFT_HISTORY_PATH = os.path.join(RAW_DATA_DIR, "draft_history.csv")
ROOKIE_AGES_PATH = os.path.join(RAW_DATA_DIR, "rookie_ages.csv")
PLAYER_POSITIONS_PATH = os.path.join(RAW_DATA_DIR, "player_positions.csv")
NBA_STATS_FIRST_SEASON_START_YEAR = 1996
DRAFT_HISTORY_COLUMNS = [
    "PERSON_ID", "PLAYER_NAME", "SEASON", "ROUND_NUMBER", "ROUND_PICK",
    "OVERALL_PICK", "DRAFT_TYPE", "TEAM_ABBREVIATION", "ORGANIZATION",
    "ORGANIZATION_TYPE",
]

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


def _completed_seasons() -> List[str]:
    """Return API-supported seasons through the last completed season."""
    last_season = Season.previous_season
    last_start_year = int(last_season[:4])
    return [
        f"{start_year}-{(start_year + 1) % 100:02d}"
        for start_year in range(
            NBA_STATS_FIRST_SEASON_START_YEAR, last_start_year + 1
        )
    ]


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
    By default, fetches every season supported by the endpoint through the last
    completed season. The complete league cache is retained, while an optional
    roster_player_ids argument filters only the returned DataFrame.
    """
    ensure_dirs()

    if seasons is None:
        seasons = _completed_seasons()
    else:
        seasons = list(seasons)

    if roster_player_ids is None:
        roster_player_ids = _get_current_roster_player_ids()
    if roster_player_ids is not None:
        roster_player_ids = set(roster_player_ids)

    cached_df = pd.DataFrame()
    if not force_refresh and os.path.exists(HISTORICAL_STATS_PATH):
        print(f"Loading cached historical stats from {HISTORICAL_STATS_PATH}...")
        cached_df = _load_cached_dataframe(HISTORICAL_STATS_PATH)
        if not cached_df.empty:
            cleaned_df = _normalize_columns(cached_df)
            cached_seasons = set(cleaned_df["SEASON"].dropna().astype(str))
            requested_seasons = set(seasons)
            if requested_seasons.issubset(cached_seasons):
                if roster_player_ids is not None:
                    player_ids = pd.to_numeric(
                        cleaned_df["PLAYER_ID"], errors="coerce"
                    )
                    cleaned_df = cleaned_df.loc[
                        player_ids.isin(roster_player_ids)
                    ].copy()
                print(
                    f"Using cached historical stats through "
                    f"{max(cached_seasons)} ({len(cleaned_df)} rows)."
                )
                return cleaned_df

            # Older cache files were limited to recent seasons and current-roster
            # players. Rebuild them completely once so earlier player histories
            # are not permanently missing from the expanded cache.
            if seasons[0] not in cached_seasons:
                print("Cached history does not cover the requested range; rebuilding it.")
                cached_df = pd.DataFrame()
            else:
                seasons = [season for season in seasons if season not in cached_seasons]

    all_dfs = []
    failed_seasons = []
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
            if season_df.empty:
                print(f"No stats returned for season {season}.")
                failed_seasons.append(season)
                continue
            all_dfs.append(season_df)
            time.sleep(1.5)
        except Exception as exc:
            print(f"Error fetching season {season}: {exc}")
            failed_seasons.append(season)

    if not all_dfs and cached_df.empty:
        print("No historical stats fetched.")
        return pd.DataFrame()

    frames = [frame for frame in (cached_df, *all_dfs) if not frame.empty]
    combined_df = pd.concat(frames, ignore_index=True)
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

    # Keep the cache complete and unfiltered; return a roster-scoped copy below.
    # This avoids losing history when the current roster changes between runs.
    selected_columns = [col for col in SELECTED_HISTORICAL_COLUMNS if col in combined_df.columns]
    combined_df = combined_df[selected_columns]
    combined_df.to_csv(HISTORICAL_STATS_PATH, index=False)

    if roster_player_ids is not None:
        player_ids = pd.to_numeric(combined_df["PLAYER_ID"], errors="coerce")
        result_df = combined_df.loc[player_ids.isin(roster_player_ids)].copy()
    else:
        result_df = combined_df

    if failed_seasons:
        print(
            "Warning: historical stats could not be fetched for seasons: "
            + ", ".join(failed_seasons)
        )
    print(
        f"Saved complete historical stats to {HISTORICAL_STATS_PATH} "
        f"({len(combined_df)} rows); returning {len(result_df)} rows."
    )
    return result_df


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

def fetch_draft_history() -> pd.DataFrame:
    """Every NBA draft pick (one request). Refreshed each run; a failed fetch keeps the saved file."""
    ensure_dirs()
    try:
        raw_df = _fetch_with_retry(
            lambda: drafthistory.DraftHistory(timeout=60).get_data_frames()[0],
            label="Draft history fetch",
        )
        _validate_columns(raw_df, set(DRAFT_HISTORY_COLUMNS), "Draft history")
    except Exception as exc:
        if os.path.exists(DRAFT_HISTORY_PATH):
            print(f"Draft history fetch failed ({exc}); keeping {DRAFT_HISTORY_PATH}.")
            return pd.read_csv(DRAFT_HISTORY_PATH)
        raise
    df = raw_df[DRAFT_HISTORY_COLUMNS].copy()
    df.to_csv(DRAFT_HISTORY_PATH, index=False)
    print(
        f"Saved draft history to {DRAFT_HISTORY_PATH} ({len(df)} picks, "
        f"{df['SEASON'].min()}-{df['SEASON'].max()})."
    )
    return df


def _height_inches(height) -> float:
    """'6-10' -> 82.0; blank or malformed -> NaN."""
    parts = str(height).split("-")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        return float("nan")
    return float(int(parts[0]) * 12 + int(parts[1]))


def fetch_player_positions(season: str = "2026-27") -> pd.DataFrame:
    """Listed position, height and weight for every player, current and retired (one request).

    POSITION is the NBA's coarse label (G, F, C, G-F, F-G, F-C, C-F), one per player
    for the whole career. COLLEGE and COUNTRY are kept for later college-stat matching.
    Refreshed each run; a failed fetch keeps the saved file.
    """
    ensure_dirs()
    try:
        raw_df = _fetch_with_retry(
            lambda: playerindex.PlayerIndex(
                season=season, historical_nullable=1, timeout=60
            ).get_data_frames()[0],
            label="Player index fetch",
        )
        _validate_columns(raw_df, {"PERSON_ID", "POSITION", "HEIGHT", "WEIGHT"}, "Player index")
    except Exception as exc:
        if os.path.exists(PLAYER_POSITIONS_PATH):
            print(f"Player index fetch failed ({exc}); keeping {PLAYER_POSITIONS_PATH}.")
            return pd.read_csv(PLAYER_POSITIONS_PATH)
        raise
    df = pd.DataFrame(
        {
            "PLAYER_ID": pd.to_numeric(raw_df["PERSON_ID"], errors="coerce"),
            "PLAYER_NAME": (
                raw_df["PLAYER_FIRST_NAME"].fillna("") + " " + raw_df["PLAYER_LAST_NAME"].fillna("")
            ).str.strip(),
            "POSITION": raw_df["POSITION"].replace("", pd.NA),
            "HEIGHT_IN": raw_df["HEIGHT"].map(_height_inches),
            "WEIGHT_LB": pd.to_numeric(raw_df["WEIGHT"], errors="coerce"),
            "COLLEGE": raw_df.get("COLLEGE"),
            "COUNTRY": raw_df.get("COUNTRY"),
        }
    ).dropna(subset=["PLAYER_ID"])
    df["PLAYER_ID"] = df["PLAYER_ID"].astype("int64")
    df = df.drop_duplicates("PLAYER_ID", keep="last")
    df.to_csv(PLAYER_POSITIONS_PATH, index=False)
    print(
        f"Saved positions to {PLAYER_POSITIONS_PATH} ({len(df)} players, "
        f"{int(df['POSITION'].notna().sum())} with a position)."
    )
    return df


def _age_on_feb_1(birthdate: str, season_end_year: int) -> float:
    """Whole-year age on 1 February of the season, close to the NBA's AGE column."""
    born = pd.to_datetime(birthdate, errors="coerce")
    if pd.isna(born):
        return float("nan")
    reference = pd.Timestamp(year=season_end_year, month=2, day=1)
    years = reference.year - born.year - ((reference.month, reference.day) < (born.month, born.day))
    return float(years)


def fetch_rookie_ages(roster_df: pd.DataFrame, season: str = "2026-27") -> pd.DataFrame:
    """Ages for rostered players with no completed NBA season (current rookies).

    Uses the in-progress season's league stats first (same AGE definition as the
    history), then one player-info call per rookie who has not played yet.
    Writes data/raw/rookie_ages.csv; a failed fetch keeps the saved file.
    """
    ensure_dirs()
    known = set()
    if os.path.exists(HISTORICAL_STATS_PATH):
        known = set(pd.to_numeric(pd.read_csv(HISTORICAL_STATS_PATH, usecols=["PLAYER_ID"])["PLAYER_ID"], errors="coerce").dropna().astype(int))
    roster_ids = pd.to_numeric(roster_df["PLAYER_ID"], errors="coerce").dropna().astype(int)
    rookies = sorted(set(roster_ids) - known)
    season_end_year = int(season[:4]) + 1
    print(f"Fetching ages for {len(rookies)} rostered players with no NBA history...")

    try:
        current = _fetch_with_retry(
            lambda: leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, per_mode_detailed="Totals", timeout=60
            ).get_data_frames()[0],
            label=f"In-progress {season} stats",
        )
        current_age = dict(zip(current["PLAYER_ID"].astype(int), pd.to_numeric(current["AGE"], errors="coerce")))
    except Exception as exc:
        print(f"In-progress season stats failed ({exc}); using player-info calls only.")
        current_age = {}

    records = []
    for player_id in rookies:
        age = current_age.get(player_id, float("nan"))
        source = "season_stats"
        if pd.isna(age):
            source = "birthdate"
            try:
                info = _fetch_with_retry(
                    lambda: commonplayerinfo.CommonPlayerInfo(
                        player_id=player_id, timeout=60
                    ).get_data_frames()[0],
                    label=f"Player info {player_id}",
                )
                age = _age_on_feb_1(info["BIRTHDATE"].iloc[0], season_end_year)
                time.sleep(0.8)
            except Exception as exc:
                print(f"No birth date for player {player_id}: {exc}")
                age, source = float("nan"), "missing"
        records.append({"PLAYER_ID": player_id, "AGE": age, "AGE_SOURCE": source, "SEASON": season})

    df = pd.DataFrame(records, columns=["PLAYER_ID", "AGE", "AGE_SOURCE", "SEASON"])
    if df["AGE"].isna().all() and len(df) and os.path.exists(ROOKIE_AGES_PATH):
        print(f"No rookie ages fetched; keeping {ROOKIE_AGES_PATH}.")
        return pd.read_csv(ROOKIE_AGES_PATH)
    df.to_csv(ROOKIE_AGES_PATH, index=False)
    print(f"Saved rookie ages to {ROOKIE_AGES_PATH} ({df['AGE'].notna().sum()} of {len(df)} found).")
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
    fetch_draft_history()
    fetch_rookie_ages(roster_df)
    fetch_player_positions()
    print("Data ingestion complete.")
