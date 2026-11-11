"""Current injury reports, matched to NBA player IDs.

Two sources, combined by `build_current_injuries`:
  1. ESPN's injuries feed (one request, unofficial endpoint) saved to
     data/raw/injuries_espn.csv by `fetch_espn_injuries`. A failed fetch keeps the saved file.
  2. data/raw/injury_overrides.csv, edited by hand. An override wins over ESPN for that
     player. STATUS values:
        OUT    - out until RETURN_DATE (YYYY-MM-DD; blank = no timetable)
        OFS    - out for the season
        GTD    - day-to-day / game-time decision
        ACTIVE - healthy: drops any ESPN entry for that player
     Rows need PLAYER_ID or PLAYER_NAME (PLAYER_ID wins when both are given).

Output: data/processed/current_injuries.csv, one row per injured rostered player.
Nothing here changes projections; project.py decides what a status means for games.

Run on its own with `python src/injuries.py` (also runs at the end of ingest.py).
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import date

import pandas as pd
import requests

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
ROSTERS_PATH = os.path.join(RAW_DATA_DIR, "current_rosters.csv")
ESPN_INJURIES_PATH = os.path.join(RAW_DATA_DIR, "injuries_espn.csv")
OVERRIDES_PATH = os.path.join(RAW_DATA_DIR, "injury_overrides.csv")
CURRENT_INJURIES_PATH = os.path.join(PROCESSED_DATA_DIR, "current_injuries.csv")

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
REQUEST_HEADERS = {"User-Agent": "nba-fantasy-projections/1.0", "Accept": "application/json"}

ESPN_COLUMNS = [
    "ESPN_ID", "PLAYER_NAME", "ESPN_TEAM", "STATUS", "FANTASY_STATUS", "RETURN_DATE",
    "INJURY_TYPE", "REPORT_DATE", "SHORT_COMMENT", "FETCHED_AT",
]
OVERRIDE_COLUMNS = ["PLAYER_ID", "PLAYER_NAME", "STATUS", "RETURN_DATE", "NOTE"]
OVERRIDE_STATUSES = {"OUT", "OFS", "GTD", "ACTIVE"}
OUTPUT_COLUMNS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "STATUS", "RETURN_DATE",
    "INJURY_TYPE", "SOURCE", "REPORT_DATE", "NOTE",
]
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


# --------------------------------------------------------------------------
# ESPN feed
# --------------------------------------------------------------------------
def _espn_status(feed_status: str, fantasy_status: str) -> str:
    """One status per player: OFS, OUT or GTD.

    ESPN's fantasy status (OFS / OUT / GTD) is the useful one; the plain status only
    says Out or Day-To-Day, and many "Out" players are listed GTD for fantasy.
    """
    fantasy = str(fantasy_status or "").upper()
    if fantasy in ("OFS", "OUT", "GTD"):
        return fantasy
    return "GTD" if "day" in str(feed_status or "").lower() else "OUT"


def parse_espn_feed(payload: dict, fetched_at: str) -> pd.DataFrame:
    rows = []
    for team in payload.get("injuries", []):
        for entry in team.get("injuries", []):
            athlete = entry.get("athlete") or {}
            details = entry.get("details") or {}
            fantasy = (details.get("fantasyStatus") or {}).get("abbreviation", "")
            rows.append(
                {
                    "ESPN_ID": entry.get("id"),
                    "PLAYER_NAME": athlete.get("displayName", ""),
                    "ESPN_TEAM": team.get("displayName", ""),
                    "STATUS": _espn_status(entry.get("status"), fantasy),
                    "FANTASY_STATUS": fantasy,
                    "RETURN_DATE": details.get("returnDate"),
                    "INJURY_TYPE": details.get("type"),
                    "REPORT_DATE": str(entry.get("date") or "")[:10],
                    "SHORT_COMMENT": entry.get("shortComment"),
                    "FETCHED_AT": fetched_at,
                }
            )
    return pd.DataFrame(rows, columns=ESPN_COLUMNS)


def fetch_espn_injuries() -> pd.DataFrame:
    """Download ESPN's injury list (one request). A failed fetch keeps the saved file."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    try:
        response = requests.get(ESPN_INJURIES_URL, headers=REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        df = parse_espn_feed(response.json(), date.today().isoformat())
        if df.empty or df["PLAYER_NAME"].eq("").all():
            raise ValueError("feed returned no injury entries")
    except Exception as exc:
        if os.path.exists(ESPN_INJURIES_PATH):
            print(f"ESPN injury fetch failed ({exc}); keeping {ESPN_INJURIES_PATH}.")
            return pd.read_csv(ESPN_INJURIES_PATH)
        raise
    df.to_csv(ESPN_INJURIES_PATH, index=False)
    counts = df["STATUS"].value_counts().to_dict()
    print(f"Saved ESPN injuries to {ESPN_INJURIES_PATH} ({len(df)} players: {counts}).")
    return df


# --------------------------------------------------------------------------
# Matching to NBA player IDs
# --------------------------------------------------------------------------
def normalize_name(name) -> str:
    """'Dereck Lively II' -> 'dereck lively'; accents, punctuation and suffixes dropped."""
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z ]", " ", text.lower().replace(".", "").replace("'", ""))
    words = [word for word in text.split() if word not in NAME_SUFFIXES]
    return " ".join(words)


def _team_lookup(roster: pd.DataFrame) -> dict[str, str]:
    """ESPN's full team name -> roster abbreviation, by the roster's team nickname."""
    nicknames = roster.drop_duplicates("TEAM_ABBREVIATION")[["TEAM_NAME", "TEAM_ABBREVIATION"]]
    return {
        str(name).lower(): abbreviation
        for name, abbreviation in zip(nicknames["TEAM_NAME"], nicknames["TEAM_ABBREVIATION"])
    }


def _espn_team_abbreviation(espn_team: str, lookup: dict[str, str]) -> str | None:
    team = str(espn_team or "").lower()
    for nickname, abbreviation in lookup.items():
        if team.endswith(nickname):
            return abbreviation
    return None


def match_players(names: pd.Series, teams: pd.Series, roster: pd.DataFrame) -> pd.Series:
    """PLAYER_ID for each (name, team abbreviation); NaN when no unique match.

    Order: exact normalized name on that team; normalized name anywhere on a roster
    (unique only, covers trades the feed hasn't caught up with); last name on that team
    (unique only).
    """
    keyed = roster.assign(
        KEY=roster["PLAYER_NAME"].map(normalize_name),
        LAST=roster["PLAYER_NAME"].map(lambda n: normalize_name(n).split(" ")[-1]),
    )
    result = []
    for name, team in zip(names, teams):
        key = normalize_name(name)
        last = key.split(" ")[-1] if key else ""
        candidates = keyed[(keyed["KEY"] == key) & (keyed["TEAM_ABBREVIATION"] == team)]
        if len(candidates) != 1:
            candidates = keyed[keyed["KEY"] == key]
        if len(candidates) != 1:
            candidates = keyed[(keyed["LAST"] == last) & (keyed["TEAM_ABBREVIATION"] == team)]
        result.append(candidates["PLAYER_ID"].iloc[0] if len(candidates) == 1 else pd.NA)
    return pd.Series(result, index=names.index, dtype="Int64")


# --------------------------------------------------------------------------
# Manual overrides
# --------------------------------------------------------------------------
def load_overrides(path: str = OVERRIDES_PATH) -> pd.DataFrame:
    """The hand-edited override file; created with just a header if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(columns=OVERRIDE_COLUMNS).to_csv(path, index=False)
        print(f"Created empty override file {path}.")
    df = pd.read_csv(path, dtype={"PLAYER_NAME": str, "STATUS": str, "RETURN_DATE": str, "NOTE": str})
    for column in OVERRIDE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df["STATUS"] = df["STATUS"].fillna("").str.strip().str.upper()
    bad = df[~df["STATUS"].isin(OVERRIDE_STATUSES)]
    if not bad.empty:
        raise ValueError(
            f"{path}: STATUS must be one of {sorted(OVERRIDE_STATUSES)}; "
            f"bad rows: {bad[['PLAYER_NAME', 'STATUS']].to_dict('records')}"
        )
    df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce").astype("Int64")
    return df[OVERRIDE_COLUMNS]


# --------------------------------------------------------------------------
# Combined table
# --------------------------------------------------------------------------
def build_current_injuries(
    roster: pd.DataFrame,
    espn: pd.DataFrame | None = None,
    overrides: pd.DataFrame | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """One row per injured rostered player: ESPN entries, then overrides on top."""
    espn = espn if espn is not None else (
        pd.read_csv(ESPN_INJURIES_PATH) if os.path.exists(ESPN_INJURIES_PATH)
        else pd.DataFrame(columns=ESPN_COLUMNS)
    )
    overrides = overrides if overrides is not None else load_overrides()
    roster = roster.copy()
    roster["PLAYER_ID"] = pd.to_numeric(roster["PLAYER_ID"]).astype("int64")
    team_of = roster.set_index("PLAYER_ID")["TEAM_ABBREVIATION"]
    name_of = roster.set_index("PLAYER_ID")["PLAYER_NAME"]

    lookup = _team_lookup(roster)
    espn = espn.copy()
    espn_teams = espn["ESPN_TEAM"].map(lambda t: _espn_team_abbreviation(t, lookup))
    espn["PLAYER_ID"] = match_players(espn["PLAYER_NAME"], espn_teams, roster)
    unmatched = espn[espn["PLAYER_ID"].isna()]
    if not unmatched.empty:
        listed = ", ".join(f"{n} ({t})" for n, t in zip(unmatched["PLAYER_NAME"], unmatched["ESPN_TEAM"]))
        print(f"ESPN injuries not on a current roster (skipped, {len(unmatched)}): {listed}")
    from_espn = espn.dropna(subset=["PLAYER_ID"]).assign(
        SOURCE="espn", NOTE=espn["SHORT_COMMENT"]
    )

    overrides = overrides.copy()
    missing_id = overrides["PLAYER_ID"].isna()
    if missing_id.any():
        overrides.loc[missing_id, "PLAYER_ID"] = match_players(
            overrides.loc[missing_id, "PLAYER_NAME"],
            pd.Series(pd.NA, index=overrides.index[missing_id]),
            roster,
        )
    bad = overrides[overrides["PLAYER_ID"].isna() | ~overrides["PLAYER_ID"].isin(team_of.index)]
    if not bad.empty:
        print(f"Overrides not matched to a rostered player (skipped): {bad['PLAYER_NAME'].tolist()}")
    overrides = overrides.drop(bad.index).assign(SOURCE="override", REPORT_DATE=pd.NA, INJURY_TYPE=pd.NA)

    combined = pd.concat([from_espn, overrides], ignore_index=True)
    combined["PLAYER_ID"] = combined["PLAYER_ID"].astype("int64")
    # Overrides come last, so keep="last" lets them win; ACTIVE then removes the player.
    combined = combined.drop_duplicates("PLAYER_ID", keep="last")
    combined = combined[combined["STATUS"] != "ACTIVE"].copy()
    combined["PLAYER_NAME"] = combined["PLAYER_ID"].map(name_of)
    combined["TEAM_ABBREVIATION"] = combined["PLAYER_ID"].map(team_of)
    combined["RETURN_DATE"] = pd.to_datetime(combined["RETURN_DATE"], errors="coerce").dt.date
    combined = combined[OUTPUT_COLUMNS].sort_values(["TEAM_ABBREVIATION", "PLAYER_NAME"])
    if write:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        combined.to_csv(CURRENT_INJURIES_PATH, index=False)
        print(
            f"Saved {len(combined)} injured rostered players to {CURRENT_INJURIES_PATH} "
            f"({combined['STATUS'].value_counts().to_dict()}; "
            f"{int(combined['SOURCE'].eq('override').sum())} from overrides)."
        )
    return combined.reset_index(drop=True)


def load_current_injuries(path: str = CURRENT_INJURIES_PATH) -> pd.DataFrame:
    """The saved combined table; empty when it hasn't been built yet."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.read_csv(path)
    df["RETURN_DATE"] = pd.to_datetime(df["RETURN_DATE"], errors="coerce").dt.date
    return df


# --------------------------------------------------------------------------
# Games: played so far, known games out, out for season
# --------------------------------------------------------------------------
SEASON_GAMES = 82
# A player whose return leaves fewer than this many team games counts as out for season.
OUT_FOR_SEASON_MIN_GAMES_LEFT = 10
# Judgment calls (no data to fit them): a day-to-day listing costs one game; "out" with no
# return date costs five.
GTD_GAMES_OUT = 1.0
OUT_NO_DATE_GAMES = 5.0


def injury_games(
    roster: pd.DataFrame,
    injuries: pd.DataFrame,
    schedule: pd.DataFrame,
    season_log: pd.DataFrame,
    today: date | None = None,
) -> pd.DataFrame:
    """Per rostered player: games played so far, team games left, known games out ahead
    and whether he is out for the season.

    Team games left = 82 - team games played (the schedule can miss Cup games not set yet);
    games out ahead = the team's scheduled games from today up to the day before his
    return date, capped at games left.
    """
    today = today or date.today()
    roster = roster[["PLAYER_ID", "TEAM_ABBREVIATION"]].copy()
    roster["PLAYER_ID"] = pd.to_numeric(roster["PLAYER_ID"]).astype("int64")
    if len(season_log):
        played_log = season_log[pd.to_datetime(season_log["GAME_DATE"]).dt.date < today]
    else:
        played_log = season_log
    team_played = played_log.groupby("TEAM_ABBREVIATION")["GAME_ID"].nunique()
    player_played = played_log.groupby("PLAYER_ID")["GAME_ID"].nunique()
    from gamelogs import team_game_dates

    team_dates = team_game_dates(schedule) if len(schedule) else pd.DataFrame(
        columns=["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"])
    team_dates = team_dates[team_dates["GAME_DATE"] >= today]

    out = roster.set_index("PLAYER_ID")
    out["GAMES_PLAYED"] = player_played.reindex(out.index).fillna(0).astype(int)
    out["TEAM_GAMES_PLAYED"] = out["TEAM_ABBREVIATION"].map(team_played).fillna(0).astype(int)
    out["TEAM_GAMES_LEFT"] = (SEASON_GAMES - out["TEAM_GAMES_PLAYED"]).clip(lower=0)
    out["INJURY_STATUS"] = pd.NA
    out["RETURN_DATE"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    out["GAMES_OUT_AHEAD"] = 0.0
    inj = injuries.set_index("PLAYER_ID")
    inj = inj[inj.index.isin(out.index)]
    for pid, row in inj.iterrows():
        team = out.at[pid, "TEAM_ABBREVIATION"]
        left = out.at[pid, "TEAM_GAMES_LEFT"]
        status = row["STATUS"]
        ret = row["RETURN_DATE"]
        ret = None if pd.isna(ret) else pd.Timestamp(ret).date()
        out.at[pid, "INJURY_STATUS"] = status
        out.at[pid, "RETURN_DATE"] = ret
        if status == "OFS":
            games = left
        elif ret is not None and ret > today:
            team_games = team_dates.loc[team_dates["TEAM_ABBREVIATION"] == team, "GAME_DATE"]
            if team_games.empty or ret > team_games.max():
                games = left  # back after the team's last scheduled game
            else:
                games = int((team_games < ret).sum())
            if status == "GTD":
                games = min(games, GTD_GAMES_OUT)
        elif status == "GTD":
            games = GTD_GAMES_OUT
        elif ret is None:
            games = OUT_NO_DATE_GAMES
        else:  # return date today or past but still listed out
            games = GTD_GAMES_OUT
        out.at[pid, "GAMES_OUT_AHEAD"] = float(min(games, left))
    out["OUT_FOR_SEASON"] = out["INJURY_STATUS"].eq("OFS") | (
        out["INJURY_STATUS"].notna()
        & (out["TEAM_GAMES_LEFT"] - out["GAMES_OUT_AHEAD"] < OUT_FOR_SEASON_MIN_GAMES_LEFT)
    )
    out["OUT_FOR_SEASON"] = out["OUT_FOR_SEASON"].fillna(False).astype(bool)
    out.loc[out["OUT_FOR_SEASON"], "GAMES_OUT_AHEAD"] = out.loc[out["OUT_FOR_SEASON"], "TEAM_GAMES_LEFT"]
    out["AS_OF"] = today.isoformat()
    return out.reset_index()


def adjusted_games(model_gp: pd.Series, games: pd.DataFrame) -> pd.Series:
    """Full-season GP = games played so far + the model's availability rate (model GP / 82)
    x the team games he isn't known to miss. `model_gp` is indexed like `games`."""
    availability = (model_gp / SEASON_GAMES).clip(0.0, 1.0)
    available = (games["TEAM_GAMES_LEFT"] - games["GAMES_OUT_AHEAD"]).clip(lower=0)
    new_gp = games["GAMES_PLAYED"] + availability * available
    return new_gp.clip(upper=SEASON_GAMES)


def load_injury_games(today: date | None = None, roster: pd.DataFrame | None = None) -> pd.DataFrame:
    """injury_games from the saved files (current season log and schedule)."""
    from gamelogs import CURRENT_SEASON, load_gamelogs, load_schedule, season_end_year

    roster = roster if roster is not None else pd.read_csv(ROSTERS_PATH)
    season_log = load_gamelogs([season_end_year(CURRENT_SEASON)])
    return injury_games(roster, load_current_injuries(), load_schedule(), season_log, today)


if __name__ == "__main__":
    fetch_espn_injuries()
    build_current_injuries(pd.read_csv(ROSTERS_PATH))
