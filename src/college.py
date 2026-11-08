"""College stats for NBA rookies (rookie handling layer 3).

Reads Bart Torvik's per-season player tables from data/raw/torvik/torvik_YEAR.csv
(downloaded by the user with src/fetch_college.py; YEAR = season ending year) and
keeps only the final college season of players who reached the NBA.

Matching to NBA player IDs:
  1. Drafted players: Torvik's NBA pick column plus the season (= draft year)
     against draft_history.csv. The names must also agree loosely, which guards
     against pick-column errors.
  2. Everyone else (mostly undrafted rookies): normalized name in the season
     before the player's NBA debut (or up to MAX_GAP_YEARS earlier), and only if
     that name is unique in that season.
Unmatched players (international, G League Ignite, Overtime Elite, or name
mismatches) simply have no college line.

Run from the project root to rebuild data/processed/college_stats.csv:
    python src/college.py
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORVIK_DIR = PROJECT_ROOT / "data" / "raw" / "torvik"
COLLEGE_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "college_stats.csv"
FIRST_COLLEGE_YEAR = 2008
MAX_GAP_YEARS = 3  # undrafted players may debut a few seasons after college

# Torvik's season tables have no header row; columns checked against known players.
TORVIK_COLUMNS = {
    0: "NAME", 1: "TEAM", 2: "CONF", 3: "GP", 4: "MIN_PCT", 5: "ORTG", 6: "USG",
    7: "EFG", 8: "TS", 9: "ORB_PCT", 10: "DRB_PCT", 11: "AST_PCT", 12: "TO_PCT",
    13: "FTM", 14: "FTA", 15: "FT_PCT", 16: "FG2M", 17: "FG2A", 18: "FG2_PCT",
    19: "FG3M", 20: "FG3A", 21: "FG3_PCT", 22: "BLK_PCT", 23: "STL_PCT", 24: "FTR",
    25: "CLASS", 26: "HT", 31: "YEAR", 32: "TORVIK_ID", 45: "PICK", 50: "BPM",
    54: "MPG", 59: "TREB", 60: "AST", 61: "STL", 62: "BLK", 63: "PTS", 64: "ROLE",
    66: "DOB",
}


def normalize_name(name) -> str:
    """'Nikola Jokić Jr.' -> 'nikola jokic'; punctuation and suffixes dropped."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[^a-z ]", " ", text.replace("'", "").replace(".", ""))
    words = [w for w in text.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(words)


def _height_inches(height) -> float:
    parts = str(height).split("-")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        return np.nan
    return float(int(parts[0]) * 12 + int(parts[1]))


def load_torvik(folder=TORVIK_DIR) -> pd.DataFrame:
    """All saved Torvik seasons with named columns (empty if none are saved)."""
    frames = []
    for path in sorted(Path(folder).glob("torvik_*.csv")):
        raw = pd.read_csv(path, header=None, low_memory=False)
        if raw.shape[1] < max(TORVIK_COLUMNS) + 1:
            raise ValueError(f"{path.name} has {raw.shape[1]} columns; expected about 78.")
        frames.append(raw[list(TORVIK_COLUMNS)].rename(columns=TORVIK_COLUMNS))
    if not frames:
        return pd.DataFrame(columns=list(TORVIK_COLUMNS.values()))
    table = pd.concat(frames, ignore_index=True)
    table["YEAR"] = pd.to_numeric(table["YEAR"], errors="coerce")
    table = table.loc[table["YEAR"] >= FIRST_COLLEGE_YEAR].copy()
    table["HEIGHT_IN"] = table["HT"].map(_height_inches)
    table["NAME_KEY"] = table["NAME"].map(normalize_name)
    return table.reset_index(drop=True)


def _names_agree(a: str, b: str) -> bool:
    """Loose check: same last word or same first word (nicknames, suffixes, typos)."""
    a_words, b_words = a.split(), b.split()
    if not a_words or not b_words:
        return False
    return a_words[-1] == b_words[-1] or a_words[0] == b_words[0]


def match_players(
    torvik: pd.DataFrame,
    draft: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Final college season for each NBA player who can be matched.

    `draft`: PLAYER_ID, DRAFT_YEAR, OVERALL_PICK (rookies.load_draft_history).
    `players`: PLAYER_ID, PLAYER_NAME, FIRST_SEASON (season-ending year of the
    NBA debut; for current rookies, the current season).
    """
    players = players.copy()
    players["NAME_KEY"] = players["PLAYER_NAME"].map(normalize_name)
    matches = []

    # 1. Drafted: pick and draft year.
    picked = torvik.loc[torvik["PICK"].notna()].copy()
    picked["PICK"] = picked["PICK"].astype(int)
    by_pick = picked.merge(
        draft, left_on=["YEAR", "PICK"], right_on=["DRAFT_YEAR", "OVERALL_PICK"], how="inner"
    ).merge(players[["PLAYER_ID", "NAME_KEY"]], on="PLAYER_ID", suffixes=("", "_NBA"))
    by_pick = by_pick.loc[
        [_names_agree(a, b) for a, b in zip(by_pick["NAME_KEY"], by_pick["NAME_KEY_NBA"])]
    ]
    by_pick = by_pick.sort_values("MPG", ascending=False).drop_duplicates("PLAYER_ID")
    matches.append(by_pick.assign(MATCH_METHOD="draft_pick"))

    # 2. Name in the seasons just before the NBA debut, unique in that season.
    remaining = players.loc[~players["PLAYER_ID"].isin(by_pick["PLAYER_ID"])]
    taken = set(zip(by_pick["YEAR"], by_pick["TORVIK_ID"]))
    unique = torvik.loc[~torvik.duplicated(["YEAR", "NAME_KEY"], keep=False)]
    candidates = remaining.merge(unique, on="NAME_KEY", how="inner")
    gap = candidates["FIRST_SEASON"] - candidates["YEAR"]
    candidates = candidates.loc[(gap >= 1) & (gap <= MAX_GAP_YEARS)]
    candidates = candidates.loc[
        [(y, t) not in taken for y, t in zip(candidates["YEAR"], candidates["TORVIK_ID"])]
    ]
    by_name = candidates.sort_values("YEAR").drop_duplicates("PLAYER_ID", keep="last")
    matches.append(by_name.assign(MATCH_METHOD="name"))

    result = pd.concat(matches, ignore_index=True)
    keep = ["PLAYER_ID", "MATCH_METHOD"] + [
        c for c in TORVIK_COLUMNS.values() if c not in ("NAME",)
    ] + ["NAME", "HEIGHT_IN"]
    return result[keep].rename(columns={"NAME": "COLLEGE_NAME", "YEAR": "COLLEGE_YEAR"})


def build_college_stats(history: pd.DataFrame, roster: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """Match every NBA player who debuted from 2009 on, plus current rookies."""
    torvik = load_torvik()
    if torvik.empty:
        raise FileNotFoundError(
            f"No Torvik files in {TORVIK_DIR}. Run python src/fetch_college.py first."
        )
    first = history.sort_values("SEASON_YEAR").drop_duplicates("PLAYER_ID")
    debut = first[["PLAYER_ID", "PLAYER_NAME", "SEASON_YEAR"]].rename(columns={"SEASON_YEAR": "FIRST_SEASON"})
    current_season = int(history["SEASON_YEAR"].max()) + 1
    rookies = roster.loc[~roster["PLAYER_ID"].isin(history["PLAYER_ID"])]
    rookies = pd.DataFrame(
        {
            "PLAYER_ID": rookies["PLAYER_ID"],
            "PLAYER_NAME": rookies["ROSTER_NAME"],
            "FIRST_SEASON": current_season,
        }
    )
    players = pd.concat([debut, rookies], ignore_index=True)
    players = players.loc[players["FIRST_SEASON"] > FIRST_COLLEGE_YEAR]
    return match_players(torvik, draft, players)


def load_college_stats(path=COLLEGE_STATS_PATH) -> pd.DataFrame:
    """Matched college seasons keyed by PLAYER_ID (empty if not built)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["PLAYER_ID"])
    return pd.read_csv(path)


if __name__ == "__main__":
    import project as P
    import rookies as R

    history = P.load_history()
    stats = build_college_stats(history, P.load_roster(), R.load_draft_history())
    COLLEGE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(COLLEGE_STATS_PATH, index=False)
    print(stats["MATCH_METHOD"].value_counts().to_string())
    print(f"Wrote {len(stats)} matched players to {COLLEGE_STATS_PATH}")
