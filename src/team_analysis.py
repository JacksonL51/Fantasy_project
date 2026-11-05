"""Team feedback and trade comparisons for the user's fantasy team.

No Streamlit here, so everything can be tested directly. `players` is the ranked table the
app loads: one row per player, with RANK, VALUE, TOTAL_* stats and PROJECTED_GP.

Team scores use projected season totals, so games played count in full. A score says how
far a team sits from a typical team of the same size in this league, in units of how much
same-size teams normally differ. Above 0 is better than typical, for every category.
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

# Feedback bands are judgment calls: nothing here predicts league results. Grades are sized
# to simulated snake drafts on real projections, where teams' overall scores had a spread
# (standard deviation) of about 0.15, so most drafted teams land near average.
RATING_BANDS = ((1.0, "Strong"), (0.35, "Above average"), (-0.35, "Average"), (-1.0, "Below average"))
LOWEST_RATING = "Weak"
GRADE_BANDS = ((0.3, "Excellent"), (0.1, "Above average"), (-0.1, "About average"), (-0.3, "Below average"))
LOWEST_GRADE = "Weak"
PUNT_SCORE = -1.25
LOW_GAMES_GAP = 4.0
LOW_GAMES_PLAYER = 55.0
TRADE_EVEN_BAND = 0.05
TRADE_CATEGORY_MOVE = 0.1
REPLACEMENT_POOL = 10


@dataclass
class TeamReport:
    size: int
    categories: pd.DataFrame  # by category: LABEL, TEAM, TYPICAL, SCORE, RATING
    overall: float
    grade: str
    average_gp: float
    typical_gp: float


@dataclass
class TradeReport:
    before: TeamReport
    after: TeamReport
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


def _band(score: float, bands: tuple, lowest: str) -> str:
    for cutoff, label in bands:
        if score >= cutoff:
            return label
    return lowest


def rating(score: float) -> str:
    return _band(score, RATING_BANDS, LOWEST_RATING)


def grade(score: float) -> str:
    return _band(score, GRADE_BANDS, LOWEST_GRADE)


def _impact(rows: pd.DataFrame, made: str, attempts: str, percent: float) -> pd.Series:
    """Makes above what the given percentage would produce on the same attempts."""
    return rows[f"TOTAL_{made}"] - percent * rows[f"TOTAL_{attempts}"]


def _evaluate(rows: pd.DataFrame, players: pd.DataFrame, league_size: int) -> TeamReport:
    """Score rows against a typical team of the same size.

    In a league of L teams, after S players each, the league holds about the top L x S
    players. So a typical S-player team is S average players from that group, which lets a
    team be judged during a draft without tracking other teams. Spreads always come from
    the full rostered pool, because the very top of the league alone is too small to measure.
    """
    size = len(rows)
    ordered = players.sort_values("RANK")
    typical_pool = ordered.head(league_size * size)
    full_pool = ordered.head(league_size * ROSTER_SIZE)
    root = np.sqrt(size)

    records = {}
    for stat, label in COUNT_CATEGORIES.items():
        column = f"TOTAL_{stat}"
        team, typical = rows[column].sum(), size * typical_pool[column].mean()
        score = (team - typical) / (root * full_pool[column].std())
        records[stat] = {"LABEL": label, "TEAM": team, "TYPICAL": typical,
                         "SCORE": -score if stat == "TOV" else score}
    for key, (label, made, attempts) in PERCENT_CATEGORIES.items():
        made_col, attempts_col = f"TOTAL_{made}", f"TOTAL_{attempts}"
        typical_pct = typical_pool[made_col].sum() / typical_pool[attempts_col].sum()
        full_pct = full_pool[made_col].sum() / full_pool[attempts_col].sum()
        spread = root * _impact(full_pool, made, attempts, full_pct).std()
        team_attempts = rows[attempts_col].sum()
        records[key] = {
            "LABEL": label,
            "TEAM": 100 * rows[made_col].sum() / team_attempts if team_attempts else 0.0,
            "TYPICAL": 100 * typical_pct,
            "SCORE": _impact(rows, made, attempts, typical_pct).sum() / spread,
        }
    categories = pd.DataFrame.from_dict(records, orient="index")
    categories["RATING"] = categories["SCORE"].map(rating)
    overall = float(categories["SCORE"].mean())
    return TeamReport(
        size=size,
        categories=categories,
        overall=overall,
        grade=grade(overall),
        average_gp=float(rows["PROJECTED_GP"].mean()),
        typical_gp=float(typical_pool["PROJECTED_GP"].mean()),
    )


def _rows(players: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    return players.set_index("PLAYER_ID").loc[list(ids)].reset_index()


def evaluate_team(players: pd.DataFrame, roster: list[int], league_size: int) -> TeamReport:
    if not roster:
        raise ValueError("A team needs at least one player to evaluate.")
    return _evaluate(_rows(players, roster), players, league_size)


def team_feedback(report: TeamReport, roster_rows: pd.DataFrame) -> list[str]:
    """Plain-language notes on the team's shape, most important first."""
    categories = report.categories
    labels = lambda frame: ", ".join(frame["LABEL"])
    notes = []
    if report.size < ROSTER_SIZE:
        notes.append(
            f"Compared with a typical team after {report.size} of {ROSTER_SIZE} players. "
            "Early picks swing this a lot."
        )
    strong = categories[categories["SCORE"] >= 0.35].sort_values("SCORE", ascending=False)
    weak = categories[categories["SCORE"] <= -0.35].sort_values("SCORE")
    if not strong.empty:
        notes.append(f"Best categories: {labels(strong)}.")
    if not weak.empty:
        notes.append(f"Needs help: {labels(weak)}.")
    if strong.empty and weak.empty:
        notes.append("Balanced: every category is close to a typical team.")

    lowest = categories.sort_values("SCORE").iloc[0]
    if lowest["SCORE"] <= PUNT_SCORE and report.overall >= 0:
        notes.append(
            f"{lowest['LABEL']} is far behind, but the rest of the team is good. Punting it "
            "and building on your other categories is a real option."
        )
    if report.typical_gp - report.average_gp >= LOW_GAMES_GAP:
        notes.append(
            f"Availability risk: your players average {report.average_gp:.0f} projected games, "
            f"against {report.typical_gp:.0f} for a typical team."
        )
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
    columns = [f"TOTAL_{stat}" for stat in TOTAL_STATS] + ["PROJECTED_GP", "VALUE"]
    return pool[columns].mean()


def _with_replacements(rows: pd.DataFrame, count: int, replacement: pd.Series) -> pd.DataFrame:
    if count <= 0:
        return rows
    return pd.concat([rows, pd.DataFrame([replacement] * count)], ignore_index=True)


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

    before = _evaluate(
        _with_replacements(before_rows, size - len(before_rows), replacement), players, league_size
    )
    after = _evaluate(_with_replacements(after_rows, replacements, replacement), players, league_size)

    giving_rows, getting_rows = _rows(players, giving), _rows(players, getting)
    change = after.overall - before.overall
    moves = after.categories["SCORE"] - before.categories["SCORE"]
    if change >= TRADE_EVEN_BAND:
        verdict = "Helps your team"
    elif change <= -TRADE_EVEN_BAND:
        verdict = "Hurts your team"
    else:
        verdict = "About even for your team"
    return TradeReport(
        before=before,
        after=after,
        giving=giving_rows,
        getting=getting_rows,
        dropped=dropped_rows,
        replacements=replacements,
        value_out=float(giving_rows["VALUE"].sum() + dropped_rows["VALUE"].sum()),
        value_in=float(getting_rows["VALUE"].sum() + replacements * replacement["VALUE"]),
        overall_change=float(change),
        verdict=verdict,
        improved=after.categories.loc[moves >= TRADE_CATEGORY_MOVE, "LABEL"].tolist(),
        declined=after.categories.loc[moves <= -TRADE_CATEGORY_MOVE, "LABEL"].tolist(),
    )
