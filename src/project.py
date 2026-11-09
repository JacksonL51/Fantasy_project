"""Phase 5: team minutes allocation and per-game stat synthesis.

Run from the project root:  python src/project.py

Inputs
------
data/raw/current_rosters.csv        active rosters (needs a player id column
                                    and TEAM_ABBREVIATION; ROSTERSTATUS is
                                    used when present)
data/processed/training_data.csv    one row per player per completed season
                                    (PLAYER_ID, PLAYER_NAME, SEASON_YEAR, AGE,
                                    GP, MIN as season TOTAL minutes, and the
                                    11 [STAT]_RATE columns)
data/processed/rate_models.joblib   bundle written by model.py

Output
------
data/processed/projections_by_team.csv   per-game projections, sorted by team
                                         and descending PROJECTED_MPG

Steps (one function each)
-------------------------
1. load_roster / load_history / latest_season_rows / build_inference_table
2. replacement_rates + predict_with_credibility (low-minute / rookie handling)
3. predict_rates (imported from model.py; applies the per-stat policy)
4. raw_minutes_and_gp + league_typical_games + blend_projected_games (model
   games estimate blended toward recent league starter/bench averages)
5. allocate_team_season_minutes / allocate_all_teams (top-nine core by
   healthy active MPG and fringe season-minute allocation within GP × MPG bounds)
6. synthesize_stats
7. validate_projections / print_review
"""

import sys

import numpy as np
import pandas as pd

from history import (
    age_mpg_adjustment,
    compute_history_features,
    fit_age_mpg_curve,
    fit_games_model,
    fit_role_curve,
    fit_minutes_shrink,
    fit_position_crowding,
    games_feature_table,
    load_positions,
    position_crowding,
    position_mpg_adjustment,
    predict_games,
    season_production_z as _season_production_z,
    shrink_baseline_mpg,
    team_mpg_adjustment,
    role_mpg_adjustment,
    season_production_z,
    season_usage,
    team_usage_changes,
    usage_change_history,
)
from history import load_history as _load_history
from model import (
    PROCESSED_DIR,
    TARGET_STATS,
    load_model_bundle,
    predict_rates,
)
import rookies as R

PROJECT_ROOT = PROCESSED_DIR.parents[1]
ROSTERS_PATH = PROJECT_ROOT / "data" / "raw" / "current_rosters.csv"
TRAINING_DATA_PATH = PROCESSED_DIR / "training_data.csv"
PROJECTIONS_PATH = PROCESSED_DIR / "projections_by_team.csv"

# Minutes allocation
GAMES_PER_SEASON = 82
TEAM_SEASON_MINUTES = GAMES_PER_SEASON * 240.0  # 19,680.0 minutes per team
MIN_HEALTHY_MPG = 4.0
MAX_HEALTHY_MPG = 37.0
CORE_ROTATION_SIZE = 9
TIER_1_MINUTES = 2000.0
TIER_2_MINUTES = 1400.0
TIER_3_MINUTES = 750.0
# Fringe (rank 10+) players keep their own credibility-blended active MPG
# instead of a flat 12; GP is whatever the allocated minutes imply at that MPG.
FRINGE_COHORT_MPG = 8.0  # what an unknown / low-credibility fringe player gets
FRINGE_MIN_MPG = 3.0
FRINGE_MAX_MPG = 24.0
FRINGE_MAX_GP = 78.0
# A player needs this many average season minutes to claim a top-nine core slot
# on healthy MPG alone (stops 300-minute small samples from outranking regulars).
CORE_MIN_WEIGHTED_MINUTES = 700.0
MINUTES_CREDIBILITY_CAREER_MINUTES = 1500.0
COHORT_PRIOR_MINUTES = {
    "young": 550.0,
    "general": 300.0,
    "veteran": 325.0,
}
CHRONIC_GP_CEILING = 58.0
FLUKE_GP_SPREAD = 30.0
FLUKE_GP_MINIMUM = 45.0
FLUKE_HEALTHY_GP = 58.0
FRAGILE_GP_BASELINE = 47.5
CHRONIC_INJURY_BLEND = 0.5
STARTER_GP_BASELINE = 66.5
STARTER_MPG_THRESHOLD = 24.0
STARTER_GP_MODEL_WEIGHT = 0.5
BENCH_GP_MODEL_WEIGHT = 0.2
RECENCY_WEIGHTS = {0: 0.60, 1: 0.27, 2: 0.13}

# Choices that the backtest recommends (see backtest_recommendations.json).
# MPG_ADJUSTMENT: "age"  = expected MPG change from age and current MPG
#                 "role" = also uses per-minute production (rotation-jump model)
#                 "none" = baseline MPG only
# GAMES_METHOD:   "blend"  = hand-built injury rules blended toward league typical
#                 "fitted" = ridge model of next-season games (history.py)
MPG_ADJUSTMENT = "age"
GAMES_METHOD = "fitted"
# POSITION_ADJUSTMENT: "fitted" = small MPG change from how crowded a player's position
#                                 is on his team versus the team overall (history.py)
#                      "none"   = no positional adjustment
POSITION_ADJUSTMENT = "fitted"
# TEAM_CROWD_ADJUSTMENT: "fitted" = also move raw MPG with overall team crowding (starters
#                        too; the budget alone only moves bench minutes), shrunk by
#                        TEAM_CROWD_SHRINK. "none" = budget only. Set from the allocation
#                        backtest (backtest_recommendations.json).
TEAM_CROWD_ADJUSTMENT = "none"
TEAM_CROWD_SHRINK = 0.5
# ALLOCATION_METHOD: how raw MPG x games become season minutes within a team.
#   "anchored"     = top nine keep their minutes; rank 10+ share the leftovers by
#                    cohort-blended baseline within 3-24 MPG (the original method)
#   "raw_priority" = as anchored, but leftovers follow raw season minutes, floor 1 MPG
#   "proportional" = everyone scaled by one team factor so the team totals 19,680
#   "soft"         = the same factor raised to ALLOCATION_SOFT_POWER (partway there)
#   "none"         = raw MPG x games, no team budget
ALLOCATION_METHODS = ("anchored", "raw_priority", "proportional", "soft", "quality", "mpg_weighted", "none")
ALLOCATION_METHOD = "mpg_weighted"
ALLOCATION_SOFT_POWER = 0.25
# "quality": the soft team total, but a team's cut (or extra minutes) is shared by
# per-minute production instead of evenly: player i loses C x raw_i x exp(-beta q_i) /
# sum(raw x exp(-beta q)), q = projected per-minute 9-cat value (league z-score).
ALLOCATION_QUALITY_BETA = 1.0
# "mpg_weighted": like "quality", but the cut is shared by each player's own MPG level
# (lower-minute players give up more). quality_beta in project_rosters sets gamma.
ALLOCATION_MPG_GAMMA = 2.0
# MINUTES_SHRINK: "career" = pull short-career baseline MPG toward a typical value
# (history.fit_minutes_shrink); "none" = off.
MINUTES_SHRINK = "career"
RAW_PRIORITY_MIN_MPG = 1.0
# ROOKIE_METHOD: "draft"       = draft slot and age prior (rookies.py) for rookies'
#                                rates, MPG and games, and as the shrinkage target
#                                for early-career players with small samples
#                "replacement" = old behavior (replacement rates, floor MPG, typical games)
ROOKIE_METHOD = "draft"
# Players who debuted in the latest this-many seasons shrink toward their rookie
# prior instead of replacement level when their sample is small.
EARLY_CAREER_SEASONS = 2
# Stats whose rookie rates come from the draft prior; the rest stay at replacement.
# User's choice: all stats. Rookie backtest, classes 2016-26 (per-36 error): the
# prior beat replacement in 8+ of 11 classes for FGM, FGA, FG3M, FTM, FTA, REB,
# BLK and PTS, but AST (1 of 11, 3.7% worse) and STL (3 of 11, 3.1% worse) lost
# and TOV was a toss-up (7 of 11). To drop those three, remove them from this list.
ROOKIE_RATE_STATS = list(TARGET_STATS)
# USAGE_BUDGET (team usage limit, Usage branch): after the minutes budget, each team's
# projected shot volume V = (FGA + 0.44 FTA + TOV) per team minute, relative to the
# average team, and veterans' shot rates (USAGE_BUDGET_STATS) are multiplied by
# (1 / V) ** USAGE_BUDGET_POWER. The pipeline spreads teams about twice as far apart as
# they really end up (2026: Washington +12% projected vs +4% actual). Allocation backtest
# 2024-26, 500+ min players: PTS/36 error 1.907/1.939/2.166 -> 1.887/1.908/2.120, FGA and
# FTA better all three years, TOV equal or better, value rank -0.001 to 0.
#   "veterans" = scale everyone but players with no prior data (rookies' college-based
#                rates got worse every year when scaled); "all" = everyone; "none" = off.
USAGE_BUDGET = "veterans"
USAGE_BUDGET_POWER = 0.5  # 1.0 (all the way to the average team) was worse than 0.5
USAGE_BUDGET_STATS = ["FGA", "FGM", "FG3M", "FTA", "FTM", "PTS", "TOV"]
# USAGE_REDISTRIBUTION (usage that left or arrived, Usage branch): returning players'
# rates move with the share of their current team's last-season possessions used by
# players who left (VACATED_REL) and by arrivals (ARRIVED_REL), both relative to the
# average team (training uses end-of-season teams, projection opening rosters). Fitted at
# run time on the rate model's residuals (fit_usage_redistribution); FGM and FTM follow
# FGA and FTA so percentages hold. Held-out 2014-26 (both features as model inputs):
# TOV better 13/13 years, PTS 12/13, FGA 11/13, about 1%; strongest for stayers on teams
# that lost half their usage. But in the allocation backtest on top of USAGE_BUDGET, all
# five stats made PTS worse in 2 of 3 seasons; turnovers alone pass (500+ min TOV/36
# error 0.310/0.333/0.366 -> 0.307/0.332/0.366, starters 0.301/0.329/0.350 ->
# 0.299/0.327/0.350, value same or +0.001, rookies unchanged).
#   "tov"    = turnovers only (current)
#   "fitted" = all of USAGE_REDISTRIBUTION_STATS (FGM/FTM follow FGA/FTA)
#   "none"   = off
USAGE_REDISTRIBUTION = "tov"
USAGE_REDISTRIBUTION_STATS = ["PTS", "FGA", "FTA", "AST", "TOV"]
USAGE_REDISTRIBUTION_METHODS = {"tov": ["TOV"], "fitted": USAGE_REDISTRIBUTION_STATS, "none": []}
USAGE_REDISTRIBUTION_FOLLOW = {"FGM": "FGA", "FTM": "FTA"}
USAGE_REDISTRIBUTION_MIN_MINUTES = 200.0  # both seasons of a training pair (training_pairs rule)

# Expected Games Played priors
DEFAULT_STARTER_GP = 68.0
DEFAULT_BENCH_GP = 55.0
DEFAULT_ROOKIE_GP = 45.0

# Low-sample handling. The models were trained only on players with >= 200
# minutes in both seasons, so smaller samples are shrunk toward replacement.
CREDIBILITY_MINUTES = 200.0
# Max model weight if the player missed last season. Age is as of the last season
# played. Past returners kept most of their rates at every age; 0.5 for veterans is a
# judgment call standing in for minutes and games risk, not something the rates show.
STALE_CREDIBILITY_CAP = 0.85
STALE_VETERAN_MIN_AGE = 31.0
STALE_VETERAN_CREDIBILITY_CAP = 0.5
# Short-season return: a young starter's short latest season (minutes held) is a weak
# signal. Rates and active MPG average two views: one where the season counts for little
# (history rates as the model input, MPG weight scaled by GP) and one where it is ignored.
# Past cases support this group only; other short-season players got worse. Games
# projections do not change because discounting them raised over-projection. The
# averaging helps points and assists, but slightly hurts steals and blocks.
SHORT_SEASON_IGNORE_SHARE = 0.5
SHORT_SEASON_GP_RANGE = (15, 29)
SHORT_SEASON_MIN_MINUTES = 200.0
SHORT_SEASON_MAX_AGE = 30.0
SHORT_SEASON_PRIOR_MIN_MINUTES = 1000.0
SHORT_SEASON_PRIOR_MIN_MPG = 28.0
SHORT_SEASON_MAX_MPG_DROP = 5.0
SHORT_SEASON_FULL_GP = 60.0  # the latest season's MPG weight scales by GP / this
REPLACEMENT_MIN_RANGE = (200.0, 800.0)  # bench-level pool for replacement rates

# Output checks
FG_PCT_BOUNDS = (0.30, 0.72)
FT_PCT_BOUNDS = (0.45, 0.96)
TEAM_PTS_WARN_RANGE = (100.0, 130.0)
EXPECTED_TEAMS = 30

OUTPUT_STATS = [
    "PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FTM", "FTA",
]


# --------------------------------------------------------------------------
# Step 1: inference table
# --------------------------------------------------------------------------
def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"Could not find a {label} column. Tried {list(candidates)}; "
        f"available columns: {list(df.columns)}"
    )


def load_roster(path=ROSTERS_PATH) -> pd.DataFrame:
    """Active players with a team: PLAYER_ID, TEAM_ABBREVIATION, ROSTER_NAME."""
    raw = pd.read_csv(path)
    id_column = _pick_column(raw, ("PLAYER_ID", "PERSON_ID"), "player id")
    team_column = _pick_column(raw, ("TEAM_ABBREVIATION",), "team abbreviation")
    name_column = next(
        (c for c in ("PLAYER_NAME", "DISPLAY_FIRST_LAST") if c in raw.columns), None
    )

    if "ROSTERSTATUS" in raw.columns:
        active = pd.to_numeric(raw["ROSTERSTATUS"], errors="coerce") == 1
    else:
        active = pd.Series(True, index=raw.index)

    roster = pd.DataFrame(
        {
            "PLAYER_ID": pd.to_numeric(raw[id_column], errors="coerce"),
            "TEAM_ABBREVIATION": raw[team_column].fillna("").astype(str).str.strip(),
            "ROSTER_NAME": raw[name_column] if name_column else pd.NA,
        }
    )
    roster = roster.loc[
        active & roster["PLAYER_ID"].notna() & (roster["TEAM_ABBREVIATION"] != "")
    ].copy()
    roster["PLAYER_ID"] = roster["PLAYER_ID"].astype("int64")

    duplicates = int(roster["PLAYER_ID"].duplicated().sum())
    if duplicates:
        print(f"Warning: dropping {duplicates} duplicate roster rows.")
        roster = roster.drop_duplicates("PLAYER_ID", keep="first")
    print(
        f"Roster: {len(roster)} active players on "
        f"{roster['TEAM_ABBREVIATION'].nunique()} teams"
    )
    return roster.reset_index(drop=True)


def load_history(path=TRAINING_DATA_PATH) -> pd.DataFrame:
    """Season-level history (shared loader in history.py)."""
    return _load_history(path, TARGET_STATS)


def _is_short_season_return(seasons: pd.DataFrame, latest_year: int) -> bool:
    """True for a young starter whose minutes held in a short latest season."""
    last = seasons.iloc[-1]
    before = seasons.loc[seasons["SEASON_YEAR"] == latest_year - 1]
    if last["SEASON_YEAR"] != latest_year or before.empty:
        return False
    before = before.iloc[-1]
    if not (
        SHORT_SEASON_GP_RANGE[0] <= last["GP"] <= SHORT_SEASON_GP_RANGE[1]
        and last["MIN"] >= SHORT_SEASON_MIN_MINUTES
        and last["AGE"] <= SHORT_SEASON_MAX_AGE
        and before["MIN"] >= SHORT_SEASON_PRIOR_MIN_MINUTES
        and before["GP"] > 0
    ):
        return False
    before_mpg = before["MIN"] / before["GP"]
    return bool(
        before_mpg >= SHORT_SEASON_PRIOR_MIN_MPG
        and last["MIN"] / last["GP"] >= before_mpg - SHORT_SEASON_MAX_MPG_DROP
    )


def latest_season_rows(history: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Build three-season recency-weighted player profiles."""
    latest_year = int(history["SEASON_YEAR"].max())
    ordered = history.sort_values(["PLAYER_ID", "SEASON_YEAR"]).drop_duplicates(
        ["PLAYER_ID", "SEASON_YEAR"], keep="last"
    )
    recent = ordered.loc[ordered["SEASON_YEAR"] >= latest_year - 2].copy()
    recent["RECENCY_WEIGHT"] = (
        latest_year - recent["SEASON_YEAR"]
    ).map(RECENCY_WEIGHTS)

    profile_rows = []
    career_minutes = ordered.groupby("PLAYER_ID")["MIN"].sum()
    career_seasons = ordered.groupby("PLAYER_ID")["SEASON_YEAR"].count()
    career_first_year = ordered.groupby("PLAYER_ID")["SEASON_YEAR"].min()
    for player_id, seasons in recent.groupby("PLAYER_ID", sort=False):
        latest_player_season = seasons.iloc[-1].copy()
        season_weights = seasons["RECENCY_WEIGHT"].to_numpy(
            dtype=float, copy=True
        )
        season_weights /= season_weights.sum()
        gp = seasons["GP"].to_numpy(dtype=float)
        latest_player_season["WEIGHTED_GP"] = float(
            np.average(gp, weights=season_weights)
        )
        latest_player_season["GP_MIN"] = float(np.min(gp))
        latest_player_season["GP_MAX"] = float(np.max(gp))
        latest_player_season["GP_MEDIAN"] = float(np.median(gp))
        latest_player_season["SEASONS_IN_WINDOW"] = len(seasons)
        latest_player_season["WEIGHTED_SEASON_MINUTES"] = float(
            np.average(
                seasons["MIN"].to_numpy(dtype=float),
                weights=season_weights,
            )
        )
        latest_player_season["PRIOR_CAREER_MINUTES"] = float(
            career_minutes.loc[player_id]
        )
        latest_player_season["CAREER_SEASONS"] = int(career_seasons.loc[player_id])
        latest_player_season["CAREER_FIRST_YEAR"] = int(
            career_first_year.loc[player_id]
        )

        short_return = _is_short_season_return(seasons, latest_year)
        latest_player_season["SHORT_SEASON_RETURN"] = short_return
        healthy_seasons = seasons.loc[seasons["GP"] >= 15]
        if healthy_seasons.empty:
            latest_player_season["BASELINE_ACTIVE_MPG"] = np.nan
            latest_player_season["ROTATION_BASELINE_MPG"] = np.nan
            latest_player_season["PRE_SHORT_BASELINE_MPG"] = np.nan
            latest_player_season["ELIGIBLE_MPG_SEASONS"] = 0
        else:
            healthy_weights = healthy_seasons["RECENCY_WEIGHT"].to_numpy(
                dtype=float, copy=True
            )
            healthy_weights /= healthy_weights.sum()
            season_mpg = (
                healthy_seasons["MIN"].to_numpy(dtype=float)
                / healthy_seasons["GP"].to_numpy(dtype=float)
            )
            latest_player_season["BASELINE_ACTIVE_MPG"] = float(
                np.average(season_mpg, weights=healthy_weights)
            )
            # BASELINE_ACTIVE_MPG stays unscaled: it also feeds the games projections.
            rotation_weights = healthy_weights.copy()
            if short_return:
                rotation_weights[-1] *= min(
                    healthy_seasons["GP"].iloc[-1] / SHORT_SEASON_FULL_GP, 1.0
                )
            latest_player_season["ROTATION_BASELINE_MPG"] = float(
                np.average(season_mpg, weights=rotation_weights)
            )
            latest_player_season["PRE_SHORT_BASELINE_MPG"] = (
                float(np.average(season_mpg[:-1], weights=healthy_weights[:-1]))
                if short_return
                else np.nan
            )
            latest_player_season["ELIGIBLE_MPG_SEASONS"] = len(healthy_seasons)
        latest_player_season["PLAYER_ID"] = player_id
        profile_rows.append(latest_player_season)

    profiles = pd.DataFrame(profile_rows).drop(columns=["RECENCY_WEIGHT"])
    # `[STAT]_RATE` stays the LAST season's rate (what the models were trained
    # on as the prior). The multi-season blend enters only through the
    # `[STAT]_RATE_hist` features, computed exactly as in training.
    features = compute_history_features(history, TARGET_STATS)
    profiles = profiles.merge(
        features, on=["PLAYER_ID", "SEASON_YEAR"], how="left", validate="one_to_one"
    )
    # The season before the latest one, for short-season returns (see SHORT_SEASON_*).
    rate_columns = [f"{stat}_RATE" for stat in TARGET_STATS]
    before_season = ordered.loc[
        ordered["SEASON_YEAR"] == latest_year - 1, ["PLAYER_ID", "MIN"] + rate_columns
    ].merge(
        features.loc[features["SEASON_YEAR"] == latest_year - 1].drop(columns="SEASON_YEAR"),
        on="PLAYER_ID", how="left", validate="one_to_one",
    )
    before_season = before_season.rename(
        columns={
            column: f"PRE_{column}" for column in before_season.columns if column != "PLAYER_ID"
        }
    )
    profiles = profiles.merge(before_season, on="PLAYER_ID", how="left", validate="one_to_one")
    # Production score (rotation-jump model) and games-played features, both
    # built by history.py exactly as they are for training.
    profiles = profiles.merge(
        season_production_z(history), on=["PLAYER_ID", "SEASON_YEAR"],
        how="left", validate="one_to_one",
    )
    profiles = profiles.merge(
        games_feature_table(history), on=["PLAYER_ID", "SEASON_YEAR"],
        how="left", validate="one_to_one",
    )
    return profiles.reset_index(drop=True), latest_year


def replacement_rates(
    latest: pd.DataFrame, latest_year: int, bundle: dict[str, object]
) -> pd.Series:
    """Minute-weighted bench-level rates, used for rookies and small samples."""
    low, high = REPLACEMENT_MIN_RANGE
    pool = latest.loc[
        (latest["SEASON_YEAR"] == latest_year)
        & (latest["MIN"] >= low)
        & (latest["MIN"] < high)
    ]
    if len(pool) < 10:
        print("Warning: tiny replacement pool; using the league training mean.")
        return pd.Series(bundle["prior_shrinkage_reference_by_stat"]).reindex(
            TARGET_STATS
        )
    return pd.Series(
        {
            stat: float(np.average(pool[f"{stat}_RATE"], weights=pool["MIN"]))
            for stat in TARGET_STATS
        }
    )


def build_inference_table(
    roster: pd.DataFrame,
    latest: pd.DataFrame,
    latest_year: int,
    replacement: pd.Series,
) -> pd.DataFrame:
    """Roster joined to latest stats, with the exact columns the models expect."""
    history_for_merge = latest.drop(
        columns=["TEAM_ABBREVIATION"], errors="ignore"
    )
    table = roster.merge(
        history_for_merge, on="PLAYER_ID", how="left", validate="one_to_one"
    )
    table["NO_PRIOR_DATA"] = table["SEASON_YEAR"].isna()
    table["STALE_DATA"] = ~table["NO_PRIOR_DATA"] & (table["SEASON_YEAR"] < latest_year)
    if "CAREER_SEASONS" not in table.columns:
        table["CAREER_SEASONS"] = 0

    for stat in TARGET_STATS:
        table[f"{stat}_RATE_prior"] = table[f"{stat}_RATE"].fillna(replacement[stat])
        table[f"{stat}_RATE_hist"] = table[f"{stat}_RATE_hist"].fillna(replacement[stat])
    table["HIST_MIN"] = table["HIST_MIN"].fillna(0.0)
    table["HIST_SEASONS"] = table["HIST_SEASONS"].fillna(0).astype(int)
    table["AGE_prior"] = table["AGE"].fillna(latest["AGE"].median())
    table["AGE_SQ"] = table["AGE_prior"] ** 2
    table["MIN_prior"] = table["MIN"].fillna(0.0)

    # A tiny last season (injury) with a credible multi-season history: use the
    # history rates as the prior instead of blending toward replacement level.
    table["HIST_SUBSTITUTED"] = (
        ~table["NO_PRIOR_DATA"]
        & (table["MIN_prior"] < CREDIBILITY_MINUTES)
        & (table["HIST_MIN"] >= CREDIBILITY_MINUTES)
    )
    for stat in TARGET_STATS:
        table.loc[table["HIST_SUBSTITUTED"], f"{stat}_RATE_prior"] = table.loc[
            table["HIST_SUBSTITUTED"], f"{stat}_RATE_hist"
        ]

    # Short-season return: history rates are the model's last-season input.
    table["SHORT_SEASON_RETURN"] = table["SHORT_SEASON_RETURN"].eq(True)
    for stat in TARGET_STATS:
        table.loc[table["SHORT_SEASON_RETURN"], f"{stat}_RATE_prior"] = table.loc[
            table["SHORT_SEASON_RETURN"], f"{stat}_RATE_hist"
        ]

    print(
        f"Matched to history: {int((~table['NO_PRIOR_DATA']).sum())}; "
        f"no prior data: {int(table['NO_PRIOR_DATA'].sum())}; "
        f"stale (missed {latest_year}): {int(table['STALE_DATA'].sum())}"
    )
    return table


# --------------------------------------------------------------------------
# Rookie priors (draft slot and age; see rookies.py)
# --------------------------------------------------------------------------
def rookie_priors(
    table: pd.DataFrame,
    history: pd.DataFrame,
    draft: pd.DataFrame | None = None,
    current_ages: pd.Series | None = None,
    use_team_quality: bool = True,
    use_playmaker: bool = True,
) -> pd.DataFrame:
    """Rookie prior rates, MPG and GAMES_82 for every row of `table`.

    Models are fit on rookie seasons in `history` only, and rates are scaled by
    the latest season's league rates, so a past-only history gives a leak-free
    prior. Rows with no usable prior have HAS_ROOKIE_PRIOR = False.
    """
    draft = R.load_draft_history() if draft is None else draft
    current_ages = R.load_rookie_ages() if current_ages is None else current_ages
    seasons = R.rookie_seasons(history, draft, TARGET_STATS)
    models = R.fit_rookie_models(
        seasons, TARGET_STATS, use_team_quality=use_team_quality, use_playmaker=use_playmaker
    )
    league = R.league_rates(history, TARGET_STATS).iloc[-1]
    features = R.prior_features(table["PLAYER_ID"], history, draft, current_ages)
    # Team opportunity on the current roster, as in training: teammates' latest-season
    # MPG (15+ games) and last season's minutes on this team by players now gone.
    latest_year = int(history["SEASON_YEAR"].max())
    last = history.loc[history["SEASON_YEAR"] == latest_year]
    prior_mpg = (last["MIN"] / last["GP"]).where(last["GP"] >= 15).set_axis(last["PLAYER_ID"])
    production = _season_production_z(last).set_index("PLAYER_ID")["PROD_Z"]
    prior_ast = (last["AST_RATE"] * 36.0).where(last["GP"] >= 15).set_axis(last["PLAYER_ID"])
    opportunity = R.team_opportunity(
        pd.DataFrame(
            {
                "PLAYER_ID": table["PLAYER_ID"],
                "TEAM": table["TEAM_ABBREVIATION"],
                "PRIOR_MPG": table["PLAYER_ID"].map(prior_mpg),
                "PRIOR_PROD": table["PLAYER_ID"].map(production),
                "PRIOR_AST36": table["PLAYER_ID"].map(prior_ast),
            }
        ),
        last.rename(columns={"TEAM_ABBREVIATION": "TEAM"})[["PLAYER_ID", "TEAM", "MIN"]],
    )
    features[R.OPPORTUNITY_FEATURES + ["TEAM_TOP_AST36_REL"]] = opportunity[
        R.OPPORTUNITY_FEATURES + ["TEAM_TOP_AST36_REL"]
    ]
    priors = R.predict_rookies(models, features, league)
    priors[["TEAM_CROWD", "VACATED_SHARE"]] = opportunity[["TOTAL_CROWD", "VACATED_SHARE"]]
    priors["OVERALL_PICK"] = features["OVERALL_PICK"]
    priors["HAS_ROOKIE_PRIOR"] = features["HAS_ROOKIE_PRIOR"]
    # A true rookie has no NBA season at all. "No prior data" in the table also
    # covers veterans back after 3+ seasons away (outside the recent window).
    priors["ROOKIE"] = table["NO_PRIOR_DATA"] & ~features["NBA_HISTORY"]
    print(
        f"Rookie models: {models['rows']} rookie seasons "
        f"({models['rate_rows']} with 200+ minutes for rates), classes through "
        f"{models['last_class']}"
    )
    return priors


def early_career(table: pd.DataFrame, latest_year: int) -> pd.Series:
    """Players who debuted in the latest EARLY_CAREER_SEASONS seasons."""
    first_year = pd.to_numeric(table.get("CAREER_FIRST_YEAR"), errors="coerce")
    return (
        ~table["NO_PRIOR_DATA"]
        & first_year.ge(latest_year - EARLY_CAREER_SEASONS + 1).fillna(False)
    )


def shrink_targets(
    table: pd.DataFrame,
    replacement: pd.Series,
    priors: pd.DataFrame | None,
    latest_year: int,
) -> pd.DataFrame:
    """Per-player rates that small samples are blended toward.

    Replacement level for everyone, except rookies and early-career players
    with a rookie prior, who use that prior for ROOKIE_RATE_STATS.
    Sets table["ROOKIE_PRIOR"].
    """
    targets = pd.DataFrame(
        np.tile(replacement.reindex(TARGET_STATS).to_numpy(), (len(table), 1)),
        index=table.index,
        columns=TARGET_STATS,
    )
    use_prior = pd.Series(False, index=table.index)
    if priors is not None:
        use_prior = priors["HAS_ROOKIE_PRIOR"] & (
            priors["ROOKIE"] | early_career(table, latest_year)
        )
        targets.loc[use_prior, ROOKIE_RATE_STATS] = priors.loc[use_prior, ROOKIE_RATE_STATS]
    table["ROOKIE_PRIOR"] = use_prior
    table["DRAFT_PICK"] = priors["OVERALL_PICK"] if priors is not None else np.nan
    return targets


def apply_rookie_minutes_and_games(
    table: pd.DataFrame,
    active_mpg: pd.Series,
    proj_gp: pd.Series,
    priors: pd.DataFrame | None,
    latest_year: int,
) -> tuple[pd.Series, pd.Series]:
    """Rookies get prior MPG and games; early-career players with no 15-game
    season get prior MPG. Also sets ALLOCATION_SEASON_MINUTES and
    ROOKIE_MINUTES_PRIOR for the team allocation.
    """
    active_mpg = active_mpg.copy()
    proj_gp = proj_gp.copy()
    table["ALLOCATION_SEASON_MINUTES"] = table["WEIGHTED_SEASON_MINUTES"].fillna(0.0)
    table["ROOKIE_MINUTES_PRIOR"] = False
    if priors is None:
        return active_mpg, proj_gp
    has_prior = priors["HAS_ROOKIE_PRIOR"]
    rookie = priors["ROOKIE"] & has_prior
    no_mpg_yet = (
        has_prior
        & early_career(table, latest_year)
        & table["ELIGIBLE_MPG_SEASONS"].fillna(0).eq(0)
    )
    mpg_rows = rookie | no_mpg_yet
    active_mpg.loc[mpg_rows] = priors.loc[mpg_rows, "MPG"].clip(
        MIN_HEALTHY_MPG, MAX_HEALTHY_MPG
    )
    proj_gp.loc[rookie] = priors.loc[rookie, "GAMES_82"]
    # Predicted season minutes let a high pick claim a top-nine slot; the
    # allocation takes them at full credibility (they are already a model prior).
    table.loc[rookie, "ALLOCATION_SEASON_MINUTES"] = (
        active_mpg.loc[rookie] * proj_gp.loc[rookie]
    )
    table.loc[rookie, "ROOKIE_MINUTES_PRIOR"] = True
    print(
        f"Rookie prior: {int(rookie.sum())} rookies; {int(no_mpg_yet.sum())} "
        "early-career players use prior MPG"
    )
    return active_mpg, proj_gp


# --------------------------------------------------------------------------
# Steps 2-3: rate predictions with credibility blending
# --------------------------------------------------------------------------
def predict_with_credibility(
    bundle: dict[str, object],
    table: pd.DataFrame,
    replacement: pd.Series | pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Model rates blended toward a shrinkage target for small or stale samples.

    weight = clip(MIN_prior / 200, 0, 1). Stale rows are capped at 0.85, or at 0.5
    for players aged 31+ in their last season played.
    `replacement` is one rate per stat (Series, same for everyone) or a per-row
    DataFrame (see shrink_targets). Rookies have MIN_prior = 0, so they get the
    target rates exactly.
    """
    predicted = predict_rates(bundle, table)
    short = table["SHORT_SEASON_RETURN"]
    if short.any():
        # Average with the prediction made as if the short season never happened.
        ignored = table.loc[short].copy()
        ignored["MIN_prior"] = ignored["PRE_MIN"]
        ignored["HIST_MIN"] = ignored["PRE_HIST_MIN"]
        ignored["HIST_SEASONS"] = ignored["PRE_HIST_SEASONS"]
        for stat in TARGET_STATS:
            ignored[f"{stat}_RATE_prior"] = ignored[f"PRE_{stat}_RATE"]
            ignored[f"{stat}_RATE_hist"] = ignored[f"PRE_{stat}_RATE_hist"]
        predicted.loc[short, TARGET_STATS] = (
            (1.0 - SHORT_SEASON_IGNORE_SHARE) * predicted.loc[short, TARGET_STATS]
            + SHORT_SEASON_IGNORE_SHARE * predict_rates(bundle, ignored)[TARGET_STATS]
        )
    effective_minutes = table["MIN_prior"].where(
        ~table["HIST_SUBSTITUTED"], CREDIBILITY_MINUTES
    )
    weight = np.clip(effective_minutes / CREDIBILITY_MINUTES, 0.0, 1.0)
    stale_cap = pd.Series(
        np.where(
            table["AGE_prior"] >= STALE_VETERAN_MIN_AGE,
            STALE_VETERAN_CREDIBILITY_CAP,
            STALE_CREDIBILITY_CAP,
        ),
        index=table.index,
    )
    weight = weight.where(~table["STALE_DATA"], np.minimum(weight, stale_cap))

    if isinstance(replacement, pd.DataFrame):
        target = replacement.loc[table.index, TARGET_STATS].to_numpy(dtype=float)
    else:
        target = np.tile(replacement.reindex(TARGET_STATS).to_numpy(dtype=float), (len(table), 1))
    replacement_matrix = pd.DataFrame(
        (1.0 - weight.to_numpy())[:, None] * target,
        index=table.index,
        columns=TARGET_STATS,
    )
    blended = (predicted[TARGET_STATS].mul(weight, axis=0) + replacement_matrix).clip(
        lower=0.0
    )
    print(f"Players blended toward their shrinkage target (weight < 1): {int((weight < 1).sum())}")
    return blended, weight


# --------------------------------------------------------------------------
# Usage: team usage that left or arrived, and the team usage limit
# --------------------------------------------------------------------------
def _usage_pairs(history: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-season pairs (both 200+ minutes) with the rate models' inputs as of t,
    the t+1 rates, and the t+1 end-of-season team's usage changes."""
    h = history.sort_values(["PLAYER_ID", "SEASON_YEAR"])
    rate_cols = [f"{stat}_RATE" for stat in TARGET_STATS]
    prior = h[["PLAYER_ID", "SEASON_YEAR", "AGE", "MIN", *rate_cols]].rename(
        columns={"AGE": "AGE_prior", "MIN": "MIN_prior", **{c: f"{c}_prior" for c in rate_cols}}
    )
    target = h[["PLAYER_ID", "SEASON_YEAR", "TEAM_ABBREVIATION", "MIN", *rate_cols]].rename(
        columns={"MIN": "MIN_target", **{c: f"{c}_target" for c in rate_cols}}
    )
    target["SEASON_YEAR"] -= 1
    pairs = prior.merge(target, on=["PLAYER_ID", "SEASON_YEAR"])
    pairs = pairs.loc[
        (pairs["MIN_prior"] >= USAGE_REDISTRIBUTION_MIN_MINUTES)
        & (pairs["MIN_target"] >= USAGE_REDISTRIBUTION_MIN_MINUTES)
        & pairs["AGE_prior"].notna()
    ].copy()
    pairs["AGE_SQ"] = pairs["AGE_prior"] ** 2
    pairs = pairs.merge(
        compute_history_features(history, TARGET_STATS), on=["PLAYER_ID", "SEASON_YEAR"], how="left"
    )
    changes = usage_change_history(history)
    changes["SEASON_YEAR"] -= 1
    return pairs.merge(changes, on=["SEASON_YEAR", "TEAM_ABBREVIATION"], how="inner")


def fit_usage_redistribution(
    history: pd.DataFrame, bundle: dict[str, object], stats: list[str] | None = None
) -> dict:
    """Per stat in `stats` (default USAGE_REDISTRIBUTION_STATS), minute-weighted least
    squares of the rate model's residual (actual next rate minus predict_rates) on
    VACATED_REL and ARRIVED_REL. Only the slopes are used. Pass past-only history and a
    bundle trained on it (the backtest does)."""
    stats = USAGE_REDISTRIBUTION_STATS if stats is None else stats
    pairs = _usage_pairs(history)
    if len(pairs) < 300:
        return {"active": False, "pairs": int(len(pairs))}
    predicted = predict_rates(bundle, pairs)
    design = np.column_stack([np.ones(len(pairs)), pairs["VACATED_REL"], pairs["ARRIVED_REL"]])
    sw = np.sqrt(pairs["MIN_target"].to_numpy(dtype=float))
    coef = {}
    for stat in stats:
        residual = (pairs[f"{stat}_RATE_target"] - predicted[stat]).to_numpy(dtype=float)
        b, *_ = np.linalg.lstsq(design * sw[:, None], residual * sw, rcond=None)
        coef[stat] = {"VACATED_REL": round(float(b[1]), 5), "ARRIVED_REL": round(float(b[2]), 5)}
    return {"active": True, "pairs": int(len(pairs)), "coef": coef}


def current_usage_changes(roster: pd.DataFrame, history: pd.DataFrame, latest_year: int) -> pd.DataFrame:
    """team_usage_changes for the roster's teams against the latest season."""
    usage = season_usage(history)
    current = roster.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["TEAM_ABBREVIATION"]
    return team_usage_changes(usage.loc[usage["SEASON_YEAR"] == latest_year], current)


def apply_usage_redistribution(
    table: pd.DataFrame, rates: pd.DataFrame, weight: pd.Series, model: dict, changes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Shift returning players' rates for each stat in the model by the fitted slopes x
    their team's usage changes, scaled by credibility weight. FGM/FTM follow FGA/FTA when
    those are shifted. Players with no prior data are unchanged."""
    team = table["TEAM_ABBREVIATION"]
    vacated = team.map(changes["VACATED_REL"]).astype(float).fillna(0.0)
    arrived = team.map(changes["ARRIVED_REL"]).astype(float).fillna(0.0)
    if not model.get("active", False):
        return rates, vacated, arrived
    scale = weight.where(~table["NO_PRIOR_DATA"], 0.0)
    out = rates.copy()
    for stat, c in model["coef"].items():
        out[stat] = rates[stat] + scale * (c["VACATED_REL"] * vacated + c["ARRIVED_REL"] * arrived)
    out = out.clip(lower=0.0)
    for stat, source in USAGE_REDISTRIBUTION_FOLLOW.items():
        if source not in model["coef"]:
            continue
        ratio = (out[source] / rates[source]).where(rates[source] > 0, 1.0)
        out[stat] = rates[stat] * ratio
    return out, vacated, arrived


def apply_usage_budget(
    table: pd.DataFrame, rates: pd.DataFrame, season_min: pd.Series,
    method: str = USAGE_BUDGET, power: float = USAGE_BUDGET_POWER,
) -> tuple[pd.DataFrame, pd.Series]:
    """Scale shot rates by (1 / team volume) ** power (see USAGE_BUDGET). Volume counts
    every rostered player; method "veterans" leaves players with no prior data alone."""
    if method not in ("veterans", "all", "none"):
        raise ValueError(f"Unknown USAGE_BUDGET: {method}")
    scale = pd.Series(1.0, index=table.index)
    if method == "none" or power == 0:
        return rates, scale
    team = table["TEAM_ABBREVIATION"]
    minutes = season_min.fillna(0.0)
    poss = (rates["FGA"] + 0.44 * rates["FTA"] + rates["TOV"]) * minutes
    volume = poss.groupby(team).sum() / minutes.groupby(team).sum()
    volume = volume / volume.mean()
    scale = team.map((1.0 / volume) ** power).astype(float)
    if method == "veterans":
        scale = scale.where(~table["NO_PRIOR_DATA"].astype(bool), 1.0)
    out = rates.copy()
    out[USAGE_BUDGET_STATS] = rates[USAGE_BUDGET_STATS].mul(scale, axis=0)
    return out, scale


# --------------------------------------------------------------------------
# Step 4: raw minutes per game
# --------------------------------------------------------------------------
def fit_minutes_curve(history: pd.DataFrame, method: str = MPG_ADJUSTMENT) -> dict | None:
    """Fit the MPG adjustment curve named by `method` ("age", "role", "none")."""
    if method == "age":
        return fit_age_mpg_curve(history)
    if method == "role":
        return fit_role_curve(history)
    if method == "none":
        return None
    raise ValueError(f"Unknown MPG_ADJUSTMENT: {method}")


def raw_minutes_and_gp(
    table: pd.DataFrame, curve: dict | None = None, shrink: dict | None = None
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return multi-season healthy MPG and the injury-adjusted model games estimate.
    `shrink` (history.fit_minutes_shrink) pulls small-sample baselines toward a
    typical MPG before the age adjustment.
    """
    def adjusted_mpg(baseline: pd.Series) -> pd.Series:
        mpg = baseline.fillna(MIN_HEALTHY_MPG)
        if shrink is not None and shrink.get("active", False):
            eligible = table["ELIGIBLE_MPG_SEASONS"].fillna(0) > 0
            mpg = mpg.where(
                ~eligible, shrink_baseline_mpg(shrink, mpg, table["PRIOR_CAREER_MINUTES"])
            )
        if curve is not None and curve.get("active", False):
            # Players with real history only; rookies / no-data rows stay at the floor.
            if curve.get("kind", "age") == "role":
                adjustment = role_mpg_adjustment(
                    curve, table["AGE_prior"], mpg, table["PROD_Z"]
                )
            else:
                adjustment = age_mpg_adjustment(curve, table["AGE_prior"], mpg)
            mpg = mpg + adjustment.where(table["ELIGIBLE_MPG_SEASONS"].fillna(0) > 0, 0.0)
        return mpg.clip(lower=MIN_HEALTHY_MPG, upper=MAX_HEALTHY_MPG)

    active_mpg = adjusted_mpg(table["ROTATION_BASELINE_MPG"])
    short = table["SHORT_SEASON_RETURN"]
    if short.any():
        ignored_mpg = adjusted_mpg(table["PRE_SHORT_BASELINE_MPG"])
        active_mpg = active_mpg.where(
            ~short,
            (1.0 - SHORT_SEASON_IGNORE_SHARE) * active_mpg
            + SHORT_SEASON_IGNORE_SHARE * ignored_mpg,
        )
    weighted_gp = table["WEIGHTED_GP"].fillna(DEFAULT_ROOKIE_GP)
    min_gp = table["GP_MIN"]
    max_gp = table["GP_MAX"]
    spread = max_gp - min_gp

    fluke_injury = (
        spread.ge(FLUKE_GP_SPREAD)
        & min_gp.lt(FLUKE_GP_MINIMUM)
        & max_gp.ge(FLUKE_HEALTHY_GP)
    )
    chronic_fragility = max_gp.lt(CHRONIC_GP_CEILING) & ~fluke_injury
    proj_gp = 0.70 * weighted_gp + 0.30 * STARTER_GP_BASELINE
    proj_gp = proj_gp.where(~fluke_injury, table["GP_MEDIAN"])
    chronic_projection = (
        (1 - CHRONIC_INJURY_BLEND) * weighted_gp
        + CHRONIC_INJURY_BLEND * FRAGILE_GP_BASELINE
    )
    proj_gp = proj_gp.where(~chronic_fragility, chronic_projection)
    proj_gp = proj_gp.fillna(DEFAULT_ROOKIE_GP).clip(lower=15.0, upper=78.0)

    blended_mpg = table["ELIGIBLE_MPG_SEASONS"].fillna(0).lt(3)
    return active_mpg, proj_gp, blended_mpg


def position_adjustment(
    table: pd.DataFrame,
    latest_year: int,
    model: dict,
    positions: pd.Series | None = None,
) -> pd.Series:
    """MPG change from positional crowding on each player's current team.

    Demand comes from teammates' latest-season MPG (15+ games), exactly as in the
    fit; only players with such a season are adjusted. Sets table["POSITION_CROWD"].
    """
    positions = load_positions() if positions is None else positions
    prior = (table["MIN"] / table["GP"]).where(
        table["SEASON_YEAR"].eq(latest_year) & table["GP"].ge(15)
    )
    crowd = position_crowding(
        pd.DataFrame(
            {"PLAYER_ID": table["PLAYER_ID"], "TEAM": table["TEAM_ABBREVIATION"], "PRIOR_MPG": prior}
        ),
        positions,
    )
    table["POSITION_CROWD"] = crowd["POSITION_CROWD"]
    table["TOTAL_CROWD"] = crowd["TOTAL_CROWD"]
    table["CROWD_ELIGIBLE"] = prior.notna()
    return position_mpg_adjustment(model, crowd["POSITION_CROWD"]).where(prior.notna(), 0.0)


def team_crowd_adjustment(table: pd.DataFrame, model: dict, shrink: float) -> pd.Series:
    """MPG change from overall team crowding (run position_adjustment first)."""
    change = team_mpg_adjustment(model, table["TOTAL_CROWD"], shrink)
    return change.where(table["CROWD_ELIGIBLE"], 0.0)


def league_typical_games(history: pd.DataFrame) -> tuple[float, float]:
    """Return starter and bench mean GP from 500+ minute player-seasons in the
    latest three seasons in the supplied, leakage-safe history frame.
    """
    latest_year = int(history["SEASON_YEAR"].max())
    recent = history.loc[history["SEASON_YEAR"] >= latest_year - 2].copy()
    recent["RECENCY_WEIGHT"] = (
        latest_year - recent["SEASON_YEAR"]
    ).map(RECENCY_WEIGHTS)
    recent["SEASON_MPG"] = recent["MIN"] / recent["GP"]
    baseline_mpg = {}
    for player_id, seasons in recent.groupby("PLAYER_ID", sort=False):
        mpg_seasons = seasons.loc[seasons["GP"] >= 15]
        if mpg_seasons.empty:
            mpg_seasons = seasons
        weights = mpg_seasons["RECENCY_WEIGHT"].to_numpy(dtype=float)
        baseline_mpg[player_id] = float(
            np.average(
                mpg_seasons["SEASON_MPG"].to_numpy(dtype=float),
                weights=weights,
            )
        )
    recent["BASELINE_ACTIVE_MPG"] = recent["PLAYER_ID"].map(baseline_mpg)
    recent = recent.loc[(recent["MIN"] >= 500) & (recent["GP"] > 0)].copy()
    if recent.empty:
        raise ValueError(
            "Cannot compute league-typical games: no 500+ minute player-seasons "
            "in the latest three seasons."
        )

    starters = recent.loc[
        recent["BASELINE_ACTIVE_MPG"] >= STARTER_MPG_THRESHOLD, "GP"
    ]
    bench = recent.loc[
        recent["BASELINE_ACTIVE_MPG"] < STARTER_MPG_THRESHOLD, "GP"
    ]
    overall_mean = float(recent["GP"].mean())
    starter_mean = float(starters.mean()) if not starters.empty else overall_mean
    bench_mean = float(bench.mean()) if not bench.empty else overall_mean
    print(
        f"League-typical GP from {latest_year - 2}-{latest_year} "
        f"(500+ minutes): starters={starter_mean:.1f} (n={len(starters)}), "
        f"others={bench_mean:.1f} (n={len(bench)})"
    )
    return starter_mean, bench_mean


def blend_projected_games(
    table: pd.DataFrame,
    model_gp: pd.Series,
    typical_starter_gp: float,
    typical_bench_gp: float,
    starter_model_weight: float = STARTER_GP_MODEL_WEIGHT,
    bench_model_weight: float = BENCH_GP_MODEL_WEIGHT,
) -> pd.Series:
    """Blend injury/history GP estimates toward their league-typical group mean.

    Rookies and players without history use the league typical with no model
    component. Group membership is based on the unadjusted baseline active MPG.
    """
    if not 0.0 <= starter_model_weight <= 1.0:
        raise ValueError("starter_model_weight must be between 0 and 1.")
    if not 0.0 <= bench_model_weight <= 1.0:
        raise ValueError("bench_model_weight must be between 0 and 1.")
    starter = table["BASELINE_ACTIVE_MPG"].ge(STARTER_MPG_THRESHOLD).fillna(False)
    typical = pd.Series(
        np.where(starter, typical_starter_gp, typical_bench_gp),
        index=table.index,
        dtype=float,
    )
    model_weight = pd.Series(
        np.where(starter, starter_model_weight, bench_model_weight),
        index=table.index,
        dtype=float,
    )
    rookie_or_unknown = table["NO_PRIOR_DATA"] | table["CAREER_SEASONS"].eq(1)
    blended = model_weight * model_gp + (1.0 - model_weight) * typical
    blended = blended.where(~rookie_or_unknown, typical)
    return blended.clip(lower=15.0, upper=78.0).rename("PROJECTED_GP")


def fitted_games(
    table: pd.DataFrame, games_model: dict, fallback: pd.Series
) -> pd.Series:
    """Fitted games for current, non-stale players; everyone else keeps `fallback`.

    Rookies, players with no history and players who missed the latest season
    are outside what the model was trained on, so they use the blended value.
    """
    result = fallback.copy()
    if not games_model.get("active", False):
        return result
    base_mpg = (
        table["BASELINE_ACTIVE_MPG"].fillna(table["MIN"] / table["GP"]).fillna(8.0)
    )
    frame = table[
        ["GF_LAST", "GF_W", "GF_MIN", "GF_MAX", "GF_N", "GF_5YR", "GF_5YR_N", "GF_75_PLUS"]
    ].copy()
    frame["BASE_MPG"] = base_mpg
    frame["AGE"] = table["AGE_prior"]
    usable = (
        ~table["NO_PRIOR_DATA"] & ~table["STALE_DATA"] & frame["GF_LAST"].notna()
    )
    if usable.any():
        result.loc[usable] = predict_games(games_model, frame.loc[usable])
    return result.rename("PROJECTED_GP")


# --------------------------------------------------------------------------
# Step 5: Healthy-MPG core and cohort-informed fringe allocation
# --------------------------------------------------------------------------

def allocate_team_season_minutes(
    active_mpg: np.ndarray,
    proj_gp: np.ndarray,
    weighted_season_minutes: np.ndarray,
    prior_career_minutes: np.ndarray,
    ages: np.ndarray,
    target_total: float = TEAM_SEASON_MINUTES,
    minutes_from_model: np.ndarray | None = None,
    fringe_weighting: str = "baseline",
    fringe_min_mpg: float = FRINGE_MIN_MPG,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Anchor the top-nine healthy-MPG core and distribute remaining minutes by
    weighted-minute baseline. This keeps injury-shortened seasons from demoting
    established high-MPG players into the 12-MPG fringe pool.
    `minutes_from_model` (bool per player) marks season minutes that are a model
    prior (rookies); those are not blended again toward the cohort prior.
    """
    active_mpg = np.asarray(active_mpg, dtype=float)
    proj_gp = np.asarray(proj_gp, dtype=float)
    weighted_season_minutes = np.asarray(weighted_season_minutes, dtype=float)
    prior_career_minutes = np.asarray(prior_career_minutes, dtype=float)
    ages = np.asarray(ages, dtype=float)
    arrays = (
        active_mpg,
        proj_gp,
        weighted_season_minutes,
        prior_career_minutes,
        ages,
    )
    if any(array.ndim != 1 for array in arrays) or len(
        {len(array) for array in arrays}
    ) != 1:
        raise ValueError("All team allocation inputs must be aligned one-dimensional arrays.")
    if len(active_mpg) == 0 or not np.isfinite(active_mpg).all():
        raise ValueError("Team allocation requires non-empty finite active MPG values.")
    if not np.isfinite(proj_gp).all() or (proj_gp <= 0).any():
        raise ValueError("Team allocation requires finite positive projected GP values.")
    if (
        not np.isfinite(weighted_season_minutes).all()
        or (weighted_season_minutes < 0).any()
    ):
        raise ValueError("Weighted season minutes must be finite and non-negative.")
    if (
        not np.isfinite(prior_career_minutes).all()
        or (prior_career_minutes < 0).any()
    ):
        raise ValueError("Prior career minutes must be finite and non-negative.")
    if target_total <= 0 or not np.isfinite(target_total):
        raise ValueError("target_total must be finite and positive.")

    player_order = np.lexsort((-weighted_season_minutes, -active_mpg))
    credible = weighted_season_minutes[player_order] >= CORE_MIN_WEIGHTED_MINUTES
    player_order = np.concatenate([player_order[credible], player_order[~credible]])
    core_indices = player_order[:CORE_ROTATION_SIZE]
    fringe_indices = player_order[CORE_ROTATION_SIZE:]
    season_minutes = np.zeros_like(active_mpg)
    projected_gp = proj_gp.copy()
    minute_baseline = weighted_season_minutes.copy()
    cohort_prior = np.zeros_like(active_mpg)
    credibility = np.ones_like(active_mpg)
    role_tier = np.select(
        [
            weighted_season_minutes > TIER_1_MINUTES,
            weighted_season_minutes >= TIER_2_MINUTES,
            weighted_season_minutes >= TIER_3_MINUTES,
        ],
        ["Tier 1", "Tier 2", "Tier 3"],
        default="Tier 4",
    )

    season_minutes[core_indices] = active_mpg[core_indices] * proj_gp[core_indices]
    remaining = target_total - season_minutes[core_indices].sum()
    if remaining < -1e-8:
        raise ValueError(
            "Anchored top-nine core workloads exceed the team's season minutes: "
            f"core={season_minutes[core_indices].sum():.2f}, target={target_total:.2f}."
        )
    tier4_indices = np.flatnonzero(role_tier == "Tier 4")
    known_ages = np.isfinite(ages[tier4_indices])
    tier4_ages = ages[tier4_indices]
    tier4_cohort_prior = np.full(
        len(tier4_indices), COHORT_PRIOR_MINUTES["general"]
    )
    tier4_cohort_prior[known_ages & (tier4_ages <= 22)] = (
        COHORT_PRIOR_MINUTES["young"]
    )
    tier4_cohort_prior[known_ages & (tier4_ages >= 29)] = (
        COHORT_PRIOR_MINUTES["veteran"]
    )
    cohort_prior[tier4_indices] = tier4_cohort_prior
    credibility[tier4_indices] = np.minimum(
        1.0,
        prior_career_minutes[tier4_indices]
        / MINUTES_CREDIBILITY_CAREER_MINUTES,
    )
    if minutes_from_model is not None:
        credibility[np.asarray(minutes_from_model, dtype=bool)] = 1.0
    minute_baseline[tier4_indices] = (
        credibility[tier4_indices] * weighted_season_minutes[tier4_indices]
        + (1.0 - credibility[tier4_indices]) * tier4_cohort_prior
    )
    if not len(fringe_indices):
        if abs(remaining) > 1.0:
            raise ValueError("No fringe players are available for minute reconciliation.")
        return (
            season_minutes / projected_gp,
            season_minutes,
            projected_gp,
            role_tier,
            minute_baseline,
            credibility,
            cohort_prior,
        )

    # Tier 2/3 fringe players retain their individual weighted-minute baseline.
    fringe_weights = minute_baseline[fringe_indices]
    if fringe_weighting == "raw":
        # Leftovers follow each player's own raw season minutes, not the cohort blend.
        fringe_weights = active_mpg[fringe_indices] * proj_gp[fringe_indices]
    if (fringe_weights <= 0).any():
        raise ValueError("Every fringe player must have a positive minute baseline.")

    fringe_gp = projected_gp[fringe_indices]
    lower = fringe_gp * fringe_min_mpg
    upper = fringe_gp * FRINGE_MAX_MPG
    if remaining < lower.sum() - 1e-8 or remaining > upper.sum() + 1e-8:
        raise ValueError(
            f"Team fringe minutes cannot be reconciled within the {fringe_min_mpg:g}-24 MPG "
            f"bounds: remaining={remaining:.2f}, "
            f"feasible=[{lower.sum():.2f}, {upper.sum():.2f}]."
        )

    low_scale, high_scale = 0.0, max(1.0, remaining / fringe_weights.sum())
    for _ in range(100):
        scale = (low_scale + high_scale) / 2.0
        allocated = np.clip(fringe_weights * scale, lower, upper)
        if allocated.sum() < remaining:
            low_scale = scale
        else:
            high_scale = scale
    fringe_minutes = np.clip(
        fringe_weights * ((low_scale + high_scale) / 2.0), lower, upper
    )
    residual = remaining - fringe_minutes.sum()
    if abs(residual) > 1e-8:
        capacities = (
            upper - fringe_minutes if residual > 0 else fringe_minutes - lower
        )
        candidates = np.flatnonzero(capacities > 1e-8)
        if not len(candidates):
            raise ValueError("Unable to reconcile team season minutes under fringe bounds.")
        for candidate in candidates:
            adjustment = np.sign(residual) * min(abs(residual), capacities[candidate])
            fringe_minutes[candidate] += adjustment
            residual -= adjustment
            if abs(residual) <= 1e-8:
                break

    season_minutes[fringe_indices] = fringe_minutes
    norm_active_mpg = season_minutes / projected_gp
    if (
        (norm_active_mpg[fringe_indices] < fringe_min_mpg - 1e-8).any()
        or (norm_active_mpg[fringe_indices] > FRINGE_MAX_MPG + 1e-8).any()
    ):
        raise ValueError("Fringe MPG is outside the configured allocation bounds.")
    total_error = abs(season_minutes.sum() - target_total)
    if total_error > 1.0:
        raise ValueError(
            f"Team minutes reconciliation error is {total_error:.3f}, over 1 minute."
        )
    return (
        norm_active_mpg,
        season_minutes,
        projected_gp,
        role_tier,
        minute_baseline,
        credibility,
        cohort_prior,
    )


def per_minute_value(rates: pd.DataFrame, minutes: pd.Series) -> pd.Series:
    """Projected per-minute 9-cat value, as a z-score against players with 500+ minutes.

    Same components as history.season_production_z: seven counting rates (turnovers
    negated) and FG/FT impact per minute, each standardized, then averaged.
    """
    pool = minutes >= 500
    if pool.sum() < 30:
        pool = minutes > 0
    fg = (rates.loc[pool, "FGM"] * minutes[pool]).sum() / (rates.loc[pool, "FGA"] * minutes[pool]).sum()
    ft = (rates.loc[pool, "FTM"] * minutes[pool]).sum() / (rates.loc[pool, "FTA"] * minutes[pool]).sum()
    parts = pd.DataFrame(
        {
            "PTS": rates["PTS"], "FG3M": rates["FG3M"], "REB": rates["REB"], "AST": rates["AST"],
            "STL": rates["STL"], "BLK": rates["BLK"], "TOV": -rates["TOV"],
            "FG": rates["FGM"] - fg * rates["FGA"], "FT": rates["FTM"] - ft * rates["FTA"],
        }
    )
    reference = parts.loc[pool]
    score = ((parts - reference.mean()) / reference.std(ddof=0).replace(0.0, 1.0)).mean(axis=1)
    return (score - score[pool].mean()) / score[pool].std(ddof=0)


def allocate_all_teams(
    table: pd.DataFrame,
    active_mpg: pd.Series,
    proj_gp: pd.Series,
    method: str | None = None,
    quality: pd.Series | None = None,
    quality_beta: float | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Season minutes per player under ALLOCATION_METHOD (see its comment)."""
    method = method or ALLOCATION_METHOD
    if method not in ALLOCATION_METHODS:
        raise ValueError(f"Unknown ALLOCATION_METHOD: {method}")
    fringe_options = (
        {"fringe_weighting": "raw", "fringe_min_mpg": RAW_PRIORITY_MIN_MPG}
        if method == "raw_priority"
        else {}
    )
    norm_mpg = pd.Series(np.nan, index=table.index, name="PROJECTED_MPG")
    norm_season_min = pd.Series(np.nan, index=table.index, name="PROJECTED_SEASON_MIN")
    final_gp = pd.Series(np.nan, index=table.index, name="PROJECTED_GP")

    minutes_column = (
        "ALLOCATION_SEASON_MINUTES"
        if "ALLOCATION_SEASON_MINUTES" in table.columns
        else "WEIGHTED_SEASON_MINUTES"
    )
    from_model = (
        table["ROOKIE_MINUTES_PRIOR"].eq(True)
        if "ROOKIE_MINUTES_PRIOR" in table.columns
        else pd.Series(False, index=table.index)
    )
    for team, index in table.groupby("TEAM_ABBREVIATION").groups.items():
        try:
            m, s, gp, tiers, baseline, credibility, cohort_prior = (
                allocate_team_season_minutes(
                    active_mpg.loc[index].to_numpy(),
                    proj_gp.loc[index].to_numpy(),
                    table.loc[index, minutes_column]
                    .fillna(0.0)
                    .to_numpy(),
                    table.loc[index, "PRIOR_CAREER_MINUTES"]
                    .fillna(0.0)
                    .to_numpy(),
                    pd.to_numeric(
                        table.loc[index, "AGE"], errors="coerce"
                    ).to_numpy(),
                    minutes_from_model=from_model.loc[index].to_numpy(),
                    **fringe_options,
                )
            )
        except ValueError as error:
            raise ValueError(f"Team {team}: {error}") from error
        norm_mpg.loc[index] = m
        norm_season_min.loc[index] = s
        final_gp.loc[index] = gp
        table.loc[index, "ROLE_TIER"] = tiers
        table.loc[index, "MINUTES_BASELINE"] = baseline
        table.loc[index, "MINUTES_CREDIBILITY"] = credibility
        table.loc[index, "COHORT_PRIOR_MINUTES"] = cohort_prior

    if method in ("quality", "mpg_weighted"):
        raw = active_mpg * proj_gp
        teams = table["TEAM_ABBREVIATION"]
        raw_total = raw.groupby(teams).transform("sum")
        soft_total = raw_total * (TEAM_SEASON_MINUTES / raw_total) ** ALLOCATION_SOFT_POWER
        change = soft_total - raw_total  # negative = minutes to cut
        if method == "quality":
            if quality is None:
                raise ValueError('ALLOCATION_METHOD "quality" needs per-minute values.')
            beta = ALLOCATION_QUALITY_BETA if quality_beta is None else quality_beta
            q = quality.reindex(raw.index).fillna(0.0).clip(-3.0, 3.0)
            # Cuts fall on lower-value players; extra minutes go to higher-value players.
            share = raw * np.exp(np.where(change < 0, -beta, beta) * q)
        else:
            # Cuts fall on lower-minute players: share ~ raw x (MPG / 25) ^ -gamma, so a
            # 35-MPG starter gives up less per minute than a 15-MPG reserve.
            gamma = ALLOCATION_MPG_GAMMA if quality_beta is None else quality_beta
            relative = (active_mpg / 25.0).clip(lower=0.2)
            share = raw * relative ** np.where(change < 0, -gamma, gamma)
        share = share / share.groupby(teams).transform("sum")
        floor = proj_gp * RAW_PRIORITY_MIN_MPG
        season = (raw + change * share).clip(lower=floor)
        final_gp = proj_gp.rename("PROJECTED_GP")
        norm_mpg = (season / final_gp).clip(upper=MAX_HEALTHY_MPG + 3).rename("PROJECTED_MPG")
        norm_season_min = (norm_mpg * final_gp).rename("PROJECTED_SEASON_MIN")
        return norm_mpg, norm_season_min, final_gp
    if method in ("proportional", "soft", "none"):
        # Every player's raw season minutes scaled by the same team factor (fully,
        # partly, or not at all), so no group takes the whole cut.
        power = {"proportional": 1.0, "soft": ALLOCATION_SOFT_POWER, "none": 0.0}[method]
        raw = active_mpg * proj_gp
        factor = TEAM_SEASON_MINUTES / raw.groupby(table["TEAM_ABBREVIATION"]).transform("sum")
        norm_season_min = (raw * factor**power).rename("PROJECTED_SEASON_MIN")
        final_gp = proj_gp.rename("PROJECTED_GP")
        norm_mpg = (norm_season_min / final_gp).rename("PROJECTED_MPG")
    return norm_mpg, norm_season_min, final_gp


# --------------------------------------------------------------------------
# Step 6: stat synthesis
# --------------------------------------------------------------------------
def synthesize_stats(
    table: pd.DataFrame,
    rates: pd.DataFrame,
    raw_mpg: pd.Series,
    proj_gp: pd.Series,
    mpg: pd.Series,
    season_min: pd.Series,
    weight: pd.Series,
    mpg_blended: pd.Series,
) -> pd.DataFrame:
    """Per-game stats = rate * normalized active MPG; Total stats = rate * season_min."""
    projections = pd.DataFrame(
        {
            "PLAYER_ID": table["PLAYER_ID"],
            "PLAYER_NAME": table["PLAYER_NAME"].fillna(table["ROSTER_NAME"]),
            "TEAM_ABBREVIATION": table["TEAM_ABBREVIATION"],
            "PROJECTED_GP": proj_gp,
            "PROJECTED_MPG": mpg.round(1),
            "PROJECTED_SEASON_MIN": season_min.round(1),
            "ROLE_TIER": table["ROLE_TIER"],
            "WEIGHTED_SEASON_MINUTES": table["WEIGHTED_SEASON_MINUTES"].fillna(0.0),
            "MINUTES_BASELINE": table["MINUTES_BASELINE"],
            "MINUTES_CREDIBILITY": table["MINUTES_CREDIBILITY"],
            "COHORT_PRIOR_MINUTES": table["COHORT_PRIOR_MINUTES"],
        }
    )
    
    # 1. Per-game stat projections (active fantasy production)
    for stat in OUTPUT_STATS:
        projections[f"PROJ_{stat}"] = rates[stat] * mpg

    # 2. Total season volume (for Roto / Total Points draft ranks)
    for stat in OUTPUT_STATS:
        projections[f"TOTAL_{stat}"] = rates[stat] * season_min

    # Shooting percentage splits derived from volume
    projections["PROJ_FG_PCT"] = (
        projections["TOTAL_FGM"] / projections["TOTAL_FGA"]
    ).where(projections["TOTAL_FGA"] > 0)
    projections["PROJ_FT_PCT"] = (
        projections["TOTAL_FTM"] / projections["TOTAL_FTA"]
    ).where(projections["TOTAL_FTA"] > 0)

    # Diagnostic metadata
    projections["RAW_MPG"] = raw_mpg
    projections["CREDIBILITY_WEIGHT"] = weight
    projections["DATA_SEASON"] = table["SEASON_YEAR"]
    projections["NO_PRIOR_DATA"] = table["NO_PRIOR_DATA"]
    projections["STALE_DATA"] = table["STALE_DATA"]
    projections["SHORT_SEASON_RETURN"] = table["SHORT_SEASON_RETURN"]
    projections["MPG_BLENDED"] = mpg_blended
    projections["ROOKIE_PRIOR"] = table.get("ROOKIE_PRIOR", False)
    projections["DRAFT_PICK"] = table.get("DRAFT_PICK", np.nan)
    projections["POSITION_CROWD"] = table.get("POSITION_CROWD", np.nan)
    projections["LAST_PTS_PG"] = table["PTS_RATE"] * table["MIN"] / table["GP"]
    
    return projections

# --------------------------------------------------------------------------
# Step 7: checks
# --------------------------------------------------------------------------
def validate_projections(projections: pd.DataFrame) -> None:
    # 1. Check finite values and non-negativity
    stat_columns = [f"PROJ_{stat}" for stat in OUTPUT_STATS] + [
        "PROJ_FG_PCT",
        "PROJ_FT_PCT",
        "PROJECTED_MPG",
        "PROJECTED_GP",
        "PROJECTED_SEASON_MIN",
    ]
    if not np.isfinite(projections[stat_columns].to_numpy(dtype=float)).all():
        raise ValueError("Projections contain missing or non-finite values.")
    if (projections[[f"PROJ_{stat}" for stat in OUTPUT_STATS]] < 0).any().any():
        raise ValueError("Projections contain negative counting stats.")
    if (projections[["PROJECTED_GP", "PROJECTED_SEASON_MIN"]] <= 0).any().any():
        raise ValueError("Projected games and season minutes must be positive.")

    # Check each team's full-season minutes total (not per-game minutes).
    # "soft" and "none" do not force teams to the budget, so they skip this check.
    team_season_minutes = projections.groupby("TEAM_ABBREVIATION")["PROJECTED_SEASON_MIN"].sum()
    off = team_season_minutes[(team_season_minutes - TEAM_SEASON_MINUTES).abs() > 1.0]
    if ALLOCATION_METHOD in ("soft", "quality", "mpg_weighted", "none"):
        off = off.iloc[0:0]
    if len(off):
        raise ValueError(
            f"Teams not summing to {TEAM_SEASON_MINUTES} total season minutes "
            f"within 1 minute:\n{off}"
        )

    if ALLOCATION_METHOD in ("soft", "quality", "mpg_weighted", "none"):
        print(
            "Hard checks passed: finite, non-negative. Team season minutes "
            f"({ALLOCATION_METHOD}): {team_season_minutes.min():,.0f} to "
            f"{team_season_minutes.max():,.0f}, mean {team_season_minutes.mean():,.0f}."
        )
    else:
        print(
            "Hard checks passed: finite, non-negative, teams sum to "
            "19,680 season minutes."
        )

def _print_console_table(table: pd.DataFrame) -> None:
    text = table.to_string(index=False)
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def print_review(projections: pd.DataFrame) -> None:
    team_points = projections.groupby("TEAM_ABBREVIATION")["PROJ_PTS"].sum()
    print("\nTeam projected PTS per game (sum over roster):")
    print(team_points.describe().round(1).to_string())

    columns = ["PLAYER_NAME", "TEAM_ABBREVIATION", "PROJECTED_MPG", "PROJ_PTS"]
    print("\nTop 10 by projected PTS:")
    _print_console_table(
        projections.nlargest(10, "PROJ_PTS")[columns].round(2)
    )

    movers = projections.assign(
        PTS_CHANGE=projections["PROJ_PTS"] - projections["LAST_PTS_PG"]
    )
    movers = movers.loc[~projections["NO_PRIOR_DATA"]]
    movers = movers.reindex(movers["PTS_CHANGE"].abs().sort_values(ascending=False).index)
    print("\nBiggest PTS movers vs last season's per-game line:")
    _print_console_table(
        movers.head(10)[columns + ["LAST_PTS_PG", "PTS_CHANGE"]].round(2)
    )
    print(
        f"\nFlags: no prior data={int(projections['NO_PRIOR_DATA'].sum())}, "
        f"stale={int(projections['STALE_DATA'].sum())}, "
        f"two-season MPG={int(projections['MPG_BLENDED'].sum())}"
    )


def project_rosters(
    bundle: dict[str, object],
    roster: pd.DataFrame,
    history: pd.DataFrame,
    latest: tuple[pd.DataFrame, int] | None = None,
    rookie_method: str | None = None,
    position_method: str | None = None,
    team_crowd_method: str | None = None,
    team_crowd_shrink: float | None = None,
    allocation_method: str | None = None,
    quality_beta: float | None = None,
    minutes_shrink: str | None = None,
    rookie_team_quality: bool = True,
    rookie_playmaker: bool = True,
    usage_budget: str | None = None,
    usage_redistribution: str | None = None,
    draft: pd.DataFrame | None = None,
    current_ages: pd.Series | None = None,
) -> pd.DataFrame:
    """The whole projection for `roster` from `history` (seasons that may be used).

    Methods default to the module switches. The backtest passes past-only
    history, rebuilt rosters and `latest` (precomputed latest_season_rows).
    """
    rookie_method = rookie_method or ROOKIE_METHOD
    position_method = position_method or POSITION_ADJUSTMENT
    team_crowd_method = team_crowd_method or TEAM_CROWD_ADJUSTMENT
    team_crowd_shrink = TEAM_CROWD_SHRINK if team_crowd_shrink is None else team_crowd_shrink
    latest, latest_year = latest if latest is not None else latest_season_rows(history)
    replacement = replacement_rates(latest, latest_year, bundle)
    table = build_inference_table(roster, latest, latest_year, replacement)
    if rookie_method == "draft":
        priors = rookie_priors(
            table, history, draft, current_ages, rookie_team_quality, rookie_playmaker
        )
    elif rookie_method == "replacement":
        priors = None
    else:
        raise ValueError(f"Unknown ROOKIE_METHOD: {rookie_method}")
    targets = shrink_targets(table, replacement, priors, latest_year)

    rates, weight = predict_with_credibility(bundle, table, targets)
    usage_redistribution = usage_redistribution or USAGE_REDISTRIBUTION
    if usage_redistribution not in USAGE_REDISTRIBUTION_METHODS:
        raise ValueError(f"Unknown USAGE_REDISTRIBUTION: {usage_redistribution}")
    changes = current_usage_changes(roster, history, latest_year)
    usage_model = {"active": False}
    if usage_redistribution != "none":
        usage_model = fit_usage_redistribution(
            history, bundle, USAGE_REDISTRIBUTION_METHODS[usage_redistribution]
        )
        print(f"Usage redistribution: {usage_model}")
    rates, vacated, arrived = apply_usage_redistribution(table, rates, weight, usage_model, changes)
    curve = fit_minutes_curve(history)
    print(f"MPG adjustment ({MPG_ADJUSTMENT}): {curve}")
    minutes_shrink = minutes_shrink or MINUTES_SHRINK
    if minutes_shrink not in ("career", "none"):
        raise ValueError(f"Unknown MINUTES_SHRINK: {minutes_shrink}")
    shrink = fit_minutes_shrink(history) if minutes_shrink == "career" else None
    if shrink is not None:
        print(f"Minutes shrink: {shrink}")
    active_mpg, model_gp, mpg_blended = raw_minutes_and_gp(table, curve, shrink)
    if position_method not in ("fitted", "none"):
        raise ValueError(f"Unknown POSITION_ADJUSTMENT: {position_method}")
    if team_crowd_method not in ("fitted", "none"):
        raise ValueError(f"Unknown TEAM_CROWD_ADJUSTMENT: {team_crowd_method}")
    if "fitted" in (position_method, team_crowd_method):
        crowding_model = fit_position_crowding(history)
        print(f"Crowding model: {crowding_model}")
        shift = position_adjustment(table, latest_year, crowding_model)
        if position_method == "none":
            shift = shift * 0.0
        if team_crowd_method == "fitted":
            shift = shift + team_crowd_adjustment(table, crowding_model, team_crowd_shrink)
        active_mpg = (active_mpg + shift).clip(MIN_HEALTHY_MPG, MAX_HEALTHY_MPG)
        print(
            f"Crowding MPG change: {int((shift != 0).sum())} players, "
            f"range {shift.min():+.2f} to {shift.max():+.2f}"
        )
    typical_starter_gp, typical_bench_gp = league_typical_games(history)
    proj_gp = blend_projected_games(
        table, model_gp, typical_starter_gp, typical_bench_gp
    )
    if GAMES_METHOD == "fitted":
        games_model = fit_games_model(history)
        print(f"Fitted games model: {games_model['pairs']} training pairs")
        proj_gp = fitted_games(table, games_model, proj_gp)
    elif GAMES_METHOD != "blend":
        raise ValueError(f"Unknown GAMES_METHOD: {GAMES_METHOD}")
    active_mpg, proj_gp = apply_rookie_minutes_and_games(
        table, active_mpg, proj_gp, priors, latest_year
    )
    method = allocation_method or ALLOCATION_METHOD
    quality = per_minute_value(rates, active_mpg * proj_gp) if method == "quality" else None
    mpg, season_min, proj_gp = allocate_all_teams(
        table, active_mpg, proj_gp, method, quality, quality_beta
    )
    rates, usage_scale = apply_usage_budget(table, rates, season_min, usage_budget or USAGE_BUDGET)
    if (usage_scale != 1.0).any():
        print(f"Usage budget: shot-rate scale {usage_scale.min():.3f} to {usage_scale.max():.3f}")
    projections = synthesize_stats(table, rates, active_mpg, proj_gp, mpg, season_min, weight, mpg_blended)
    projections["USAGE_BUDGET_SCALE"] = usage_scale
    projections["TEAM_VACATED_USAGE_REL"] = vacated
    projections["TEAM_ARRIVED_USAGE_REL"] = arrived
    return projections


def main() -> None:
    bundle = load_model_bundle()
    roster = load_roster()
    history = load_history()
    projections = project_rosters(bundle, roster, history)

    validate_projections(projections)
    projections = projections.sort_values(
        ["TEAM_ABBREVIATION", "PROJECTED_MPG"], ascending=[True, False]
    )
    csv_projections = projections.copy()
    csv_projections["PROJECTED_GP"] = csv_projections["PROJECTED_GP"].round().astype(
        "int64"
    )
    csv_projections[[f"PROJ_{stat}" for stat in OUTPUT_STATS]] = csv_projections[
        [f"PROJ_{stat}" for stat in OUTPUT_STATS]
    ].round(1)
    csv_projections[[f"TOTAL_{stat}" for stat in OUTPUT_STATS]] = csv_projections[
        [f"TOTAL_{stat}" for stat in OUTPUT_STATS]
    ].round().astype("int64")
    percentage_columns = ["PROJ_FG_PCT", "PROJ_FT_PCT"]
    csv_projections[percentage_columns] = (
        csv_projections[percentage_columns].mul(100).round(1)
    )
    metadata_columns = [
        "ROLE_TIER",
        "WEIGHTED_SEASON_MINUTES",
        "MINUTES_BASELINE",
        "MINUTES_CREDIBILITY",
        "COHORT_PRIOR_MINUTES",
        "RAW_MPG",
        "CREDIBILITY_WEIGHT",
        "DATA_SEASON",
        "NO_PRIOR_DATA",
        "STALE_DATA",
        "SHORT_SEASON_RETURN",
        "MPG_BLENDED",
        "ROOKIE_PRIOR",
        "DRAFT_PICK",
        "POSITION_CROWD",
        "LAST_PTS_PG",
        "USAGE_BUDGET_SCALE",
        "TEAM_VACATED_USAGE_REL",
        "TEAM_ARRIVED_USAGE_REL",
    ]
    metadata_columns = [
        column for column in metadata_columns if column in csv_projections.columns
    ]
    csv_projections = csv_projections[
        [column for column in csv_projections.columns if column not in metadata_columns]
        + metadata_columns
    ]
    csv_projections.to_csv(PROJECTIONS_PATH, index=False)
    print(f"\nWrote {len(projections)} rows to {PROJECTIONS_PATH}")
    print_review(projections)

if __name__ == "__main__":
    main()