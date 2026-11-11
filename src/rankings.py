"""Fantasy rankings built from the projections written by src/project.py.

Run from the project root:
    python src/rankings.py

Input: data/processed/projections_by_team.csv.
Output: data/processed/player_rankings.csv, with each player's rank, value and
category scores. The backtest still scores its own plain season-total value, so the
two values differ on purpose.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROJECTIONS_PATH = PROCESSED_DIR / "projections_by_team.csv"
RANKINGS_PATH = PROCESSED_DIR / "player_rankings.csv"
# Preseason view (injuries known heading into the season; see project.py).
PRESEASON_PROJECTIONS_PATH = PROCESSED_DIR / "projections_preseason.csv"
PRESEASON_RANKINGS_PATH = PROCESSED_DIR / "player_rankings_preseason.csv"

# Value baselines come from this pool, the same 500-minute cut the backtest scores on.
# Deep-bench players would otherwise drag the averages down and compress the top.
VALUE_POOL_MIN_MINUTES = 500

# How much projected games count in value: 0 = per game only, 1 = full season totals.
# 0.5 counts availability without letting it outweigh production.
GAMES_WEIGHT = 0.5

# A player's strength is the sum of their scores outside their worst category. At or below
# the low mark the worst category counts in full; at or above the high mark it is
# ignored; in between its share falls steadily. Judgment calls: no online board data is
# saved to test them against.
WORST_FULL_COUNT_STRENGTH = 2.0
WORST_IGNORED_STRENGTH = 8.0

COUNT_STATS = ["PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV"]
PERCENT_STATS = (("FG", "FGM", "FGA"), ("FT", "FTM", "FTA"))
TOP_TO_PRINT = 25


def category_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Z-score for each of the 9 categories (turnovers count against).

    Stats are scaled by GP ** GAMES_WEIGHT. FG% and FT% are scored by impact (makes
    above league average on the player's attempts), so volume counts.
    """
    scaled = pd.DataFrame(index=frame.index)
    scale = frame["PROJECTED_GP"] ** (GAMES_WEIGHT - 1.0)
    for stat in [*COUNT_STATS, "FGM", "FGA", "FTM", "FTA"]:
        scaled[stat] = frame[f"TOTAL_{stat}"] * scale
    in_pool = frame["PROJECTED_SEASON_MIN"] >= VALUE_POOL_MIN_MINUTES
    pool = scaled[in_pool]

    scores = pd.DataFrame(index=frame.index)
    for stat in COUNT_STATS:
        z = (scaled[stat] - pool[stat].mean()) / pool[stat].std()
        scores[stat] = -z if stat == "TOV" else z
    for label, made, attempts in PERCENT_STATS:
        league_pct = pool[made].sum() / pool[attempts].sum()
        impact = scaled[made] - league_pct * scaled[attempts]
        pool_impact = impact[in_pool]
        scores[label] = (impact - pool_impact.mean()) / pool_impact.std()
    return scores


def fantasy_value(scores: pd.DataFrame) -> pd.Series:
    """Sum of category z-scores, discounting a strong player's worst category.

    The stronger a player is elsewhere, the less their single worst negative score counts:
    a star can carry one bad category, while a weak player pays for it in full. Positive
    scores always count in full.
    """
    worst = scores.min(axis=1).clip(upper=0.0)
    strength = scores.sum(axis=1) - worst
    span = WORST_IGNORED_STRENGTH - WORST_FULL_COUNT_STRENGTH
    worst_share = ((WORST_IGNORED_STRENGTH - strength) / span).clip(0.0, 1.0)
    return strength + worst * worst_share


def rank_players(projections: pd.DataFrame) -> pd.DataFrame:
    """One row per player, best first, with rank, value and category scores."""
    scores = category_scores(projections)
    rankings = projections[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]].copy()
    rankings["VALUE"] = fantasy_value(scores)
    for category in scores.columns:
        rankings[f"Z_{category}"] = scores[category]
    rankings = rankings.sort_values("VALUE", ascending=False, ignore_index=True)
    rankings.insert(0, "RANK", rankings.index + 1)
    return rankings


def main() -> None:
    if not PROJECTIONS_PATH.exists():
        raise FileNotFoundError(
            f"No projections found at {PROJECTIONS_PATH}. Run `python src/project.py` first."
        )
    # Preseason first (if project.py wrote it), current last so its top list prints last.
    for label, source, target in (
        ("preseason", PRESEASON_PROJECTIONS_PATH, PRESEASON_RANKINGS_PATH),
        ("current", PROJECTIONS_PATH, RANKINGS_PATH),
    ):
        if not source.exists():
            print(f"No {source}; skipping {label} rankings.")
            continue
        rankings = rank_players(pd.read_csv(source))
        rankings.to_csv(target, index=False)
        print(f"[{label}] wrote {len(rankings)} players to {target}")
        print(
            rankings.head(TOP_TO_PRINT)[["RANK", "PLAYER_NAME", "TEAM_ABBREVIATION", "VALUE"]]
            .round(2)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
