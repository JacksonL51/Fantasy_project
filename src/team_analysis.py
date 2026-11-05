"""Team stats, feedback and trade comparisons for the user's fantasy team.

No Streamlit here, so everything can be tested directly. `players` is the ranked table the
app loads: one row per player, with RANK, VALUE, Z_* category scores, TOTAL_* and PROJ_*
stats, and PROJECTED_GP.

There is no team-level baseline yet. A category's SCORE is the roster's average player score
from rankings.py (compared with players projected for 500+ minutes). Above 0 is better,
for every category.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROSTER_SIZE = 13
DEFAULT_LEAGUE_SIZE = 10
MIN_LEAGUE_SIZE = 6
MAX_LEAGUE_SIZE = 16

COUNT_CATEGORIES = {
    "PTS": "Points",
    "FG3M": "Threes",
    "REB": "Rebounds",
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks",
    "TOV": "Turnovers",
}
PERCENT_CATEGORIES = {"FG": ("FG%", "FGM", "FGA"), "FT": ("FT%", "FTM", "FTA")}
TOTAL_STATS = [*COUNT_CATEGORIES, "FGM", "FGA", "FTM", "FTA"]

# These bands are judgment calls: nothing here predicts league results.
FEEDBACK_CATEGORIES = 3
PUNT_SCORE = -0.75
PUNT_REST_OF_TEAM = 0.25
LOW_GAMES_PLAYER = 55.0
TRADE_EVEN_BAND = 0.05
TRADE_CATEGORY_MOVE = 0.1
REPLACEMENT_POOL = 10


@dataclass
class TeamReport:
    size: int
    categories: pd.DataFrame  # by category: LABEL, TOTAL, PER_GAME, SCORE
    average_gp: float


@dataclass
class TradeReport:
    before: pd.DataFrame  # by category: LABEL, TOTAL, PER_GAME
    after: pd.DataFrame
    moves: pd.Series  # season-total change per category in team units; above 0 helps
    per_game_moves: pd.Series  # the same for per-game stats
    giving: pd.DataFrame
    getting: pd.DataFrame
    dropped: pd.DataFrame
    replacements: int
    value_out: float
    value_in: float
    overall_change: float
    verdict: str
    improved: list[str] = field(default_factory=list)
    declined: list[str] = field(default_factory=list)


def _team_totals(rows: pd.DataFrame) -> pd.DataFrame:
    """Each category's team number.

    TOTAL adds up projected seasons. PER_GAME adds up each player's per-game projection.
    FG% and FT% come from total makes and attempts, so they are the same in both.
    """
    records = {
        stat: {
            "LABEL": label,
            "TOTAL": rows[f"TOTAL_{stat}"].sum(),
            "PER_GAME": rows[f"PROJ_{stat}"].sum(),
        }
        for stat, label in COUNT_CATEGORIES.items()
    }
    for key, (label, made, attempts) in PERCENT_CATEGORIES.items():
        team_attempts = rows[f"TOTAL_{attempts}"].sum()
        percent = 100 * rows[f"TOTAL_{made}"].sum() / team_attempts if team_attempts else 0.0
        records[key] = {"LABEL": label, "TOTAL": percent, "PER_GAME": percent}
    return pd.DataFrame.from_dict(records, orient="index")


def _rows(players: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    return players.set_index("PLAYER_ID").loc[list(ids)].reset_index()


def evaluate_team(players: pd.DataFrame, roster: list[int]) -> TeamReport:
    if not roster:
        raise ValueError("A team needs at least one player to evaluate.")
    rows = _rows(players, roster)
    categories = _team_totals(rows)
    categories["SCORE"] = [rows[f"Z_{stat}"].mean() for stat in categories.index]
    return TeamReport(
        size=len(rows), categories=categories, average_gp=float(rows["PROJECTED_GP"].mean())
    )


def team_feedback(report: TeamReport, roster_rows: pd.DataFrame) -> list[str]:
    """Plain-language notes on the team's shape, most important first."""
    ordered = report.categories.sort_values("SCORE", ascending=False)
    notes = [
        f"Strongest categories: {', '.join(ordered['LABEL'].head(FEEDBACK_CATEGORIES))}.",
        f"Weakest categories: {', '.join(ordered['LABEL'].tail(FEEDBACK_CATEGORIES).iloc[::-1])}.",
    ]
    lowest = ordered.iloc[-1]
    if lowest["SCORE"] <= PUNT_SCORE and ordered["SCORE"].iloc[:-1].mean() >= PUNT_REST_OF_TEAM:
        notes.append(
            f"{lowest['LABEL']} is far behind your other categories. Punting it and building "
            "on the rest is a real option."
        )
    notes.append(f"Your players average {report.average_gp:.0f} projected games.")
    low_games = roster_rows[roster_rows["PROJECTED_GP"] < LOW_GAMES_PLAYER].sort_values("RANK")
    if not low_games.empty:
        names = ", ".join(
            f"{row.PLAYER_NAME} ({row.PROJECTED_GP:.0f})" for row in low_games.head(5).itertuples()
        )
        notes.append(f"Projected for under {LOW_GAMES_PLAYER:.0f} games: {names}.")
    return notes


def _replacement_row(players: pd.DataFrame, league_size: int) -> pd.Series:
    """The average of the best few players a full league leaves unrostered."""
    start = league_size * ROSTER_SIZE
    pool = players.sort_values("RANK").iloc[start : start + REPLACEMENT_POOL]
    columns = [
        f"{prefix}_{stat}" for prefix in ("TOTAL", "PROJ") for stat in TOTAL_STATS
    ] + ["PROJECTED_GP", "VALUE"]
    return pool[columns].mean()


def _with_replacements(rows: pd.DataFrame, count: int, replacement: pd.Series) -> pd.DataFrame:
    if count <= 0:
        return rows
    return pd.concat([rows, pd.DataFrame([replacement] * count)], ignore_index=True)


def _impact(
    rows: pd.DataFrame, made: str, attempts: str, percent: float, prefix: str
) -> pd.Series:
    """Makes above what the given percentage would produce on the same attempts."""
    return rows[f"{prefix}_{made}"] - percent * rows[f"{prefix}_{attempts}"]


def _category_moves(
    before: pd.DataFrame,
    after: pd.DataFrame,
    players: pd.DataFrame,
    league_size: int,
    prefix: str = "TOTAL",
) -> pd.Series:
    """Each category's change, sized by how much rostered players differ in it.

    This puts points and steals on one scale, so a trade's categories can be weighed
    together. It measures only the change, so it needs no team-level baseline.
    """
    pool = players.sort_values("RANK").head(league_size * ROSTER_SIZE)
    root = np.sqrt(len(after))
    moves = {}
    for stat in COUNT_CATEGORIES:
        column = f"{prefix}_{stat}"
        move = (after[column].sum() - before[column].sum()) / (root * pool[column].std())
        moves[stat] = -move if stat == "TOV" else move
    for key, (_, made, attempts) in PERCENT_CATEGORIES.items():
        percent = pool[f"{prefix}_{made}"].sum() / pool[f"{prefix}_{attempts}"].sum()
        spread = root * _impact(pool, made, attempts, percent, prefix).std()
        change = (
            _impact(after, made, attempts, percent, prefix).sum()
            - _impact(before, made, attempts, percent, prefix).sum()
        )
        moves[key] = change / spread
    return pd.Series(moves)


def analyze_trade(
    players: pd.DataFrame,
    roster: list[int],
    giving: list[int],
    getting: list[int],
    league_size: int,
) -> TradeReport:
    """How a trade changes the user's team.

    Taking more players than fit drops the lowest-value players you are keeping. If your
    roster ends up with fewer players than it started, the empty spots go to a
    replacement-level pickup, so uneven trades compare equal roster sizes.
    """
    kept = [player for player in roster if player not in giving]
    overflow = max(0, len(kept) + len(getting) - ROSTER_SIZE)
    kept_rows = _rows(players, kept).sort_values("VALUE", ascending=False)
    dropped_rows = kept_rows.tail(overflow) if overflow else kept_rows.iloc[0:0]
    after_rows = pd.concat(
        [kept_rows.iloc[: len(kept_rows) - overflow], _rows(players, getting)], ignore_index=True
    )
    before_rows = _rows(players, roster)
    size = max(len(before_rows), len(after_rows))
    replacement = _replacement_row(players, league_size)
    replacements = size - len(after_rows)

    before_team = _with_replacements(before_rows, size - len(before_rows), replacement)
    after_team = _with_replacements(after_rows, replacements, replacement)
    moves = _category_moves(before_team, after_team, players, league_size)
    change = float(moves.mean())
    if change >= TRADE_EVEN_BAND:
        verdict = "Helps your team"
    elif change <= -TRADE_EVEN_BAND:
        verdict = "Hurts your team"
    else:
        verdict = "About even for your team"

    before, after = _team_totals(before_team), _team_totals(after_team)
    giving_rows, getting_rows = _rows(players, giving), _rows(players, getting)
    return TradeReport(
        before=before,
        after=after,
        moves=moves,
        per_game_moves=_category_moves(before_team, after_team, players, league_size, "PROJ"),
        giving=giving_rows,
        getting=getting_rows,
        dropped=dropped_rows,
        replacements=replacements,
        value_out=float(giving_rows["VALUE"].sum() + dropped_rows["VALUE"].sum()),
        value_in=float(getting_rows["VALUE"].sum() + replacements * replacement["VALUE"]),
        overall_change=change,
        verdict=verdict,
        improved=before.loc[moves >= TRADE_CATEGORY_MOVE, "LABEL"].tolist(),
        declined=before.loc[moves <= -TRADE_CATEGORY_MOVE, "LABEL"].tolist(),
    )
