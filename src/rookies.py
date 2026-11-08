"""Rookie priors from draft slot and age (Layer 1 of rookie handling).

Three separate models, matching how the pipeline splits veterans:
  * per-minute rates  - ridge on draft features, trained on rookies with 200+
                        minutes, weighted by minutes. Targets are each rate
                        divided by that season's league rate (an era index), so
                        the 3-point boom does not leak old-era levels into
                        today's rookies. Predictions are multiplied back by the
                        latest completed season's league rate.
  * MPG (per game played) and games per 82 - ridge on the same features,
                        trained on every rookie who appeared (no minute floor),
                        which avoids over-projecting late picks.

Features: log of overall pick and its square (undrafted = pick 61), age in the rookie season
(the NBA's AGE column, not draft-night age; nba_api has no birth dates in draft
history), and an undrafted flag. A rookie season is a player's first season in
the history, from 1997-98 on (1996-97 is the first season in the data, so its
"first seasons" include veterans).

No future data: fit_rookie_models only sees the rookie seasons it is given, and
callers pass rookie seasons before the target season.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from college import load_college_stats
from history import _games_per_82, position_crowding, season_production_z

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_HISTORY_PATH = PROJECT_ROOT / "data" / "raw" / "draft_history.csv"
ROOKIE_AGES_PATH = PROJECT_ROOT / "data" / "raw" / "rookie_ages.csv"
POSITIONS_PATH = PROJECT_ROOT / "data" / "raw" / "player_positions.csv"

ROOKIE_FIRST_SEASON = 1998  # first season whose first-appearance rows are real rookies
UNDRAFTED_PICK = 61
# LOG_PICK_SQ lets games bend (flat across the top picks, steep after pick 30):
# held-out classes 2016-26, games MAE 16.6 -> 16.1; rates and MPG unchanged.
# An age x pick interaction added nothing.
FEATURES = ["LOG_PICK", "LOG_PICK_SQ", "AGE", "UNDRAFTED"]
# Layer 2, team opportunity, for the MPG and games models only (rates unchanged):
# TOTAL_CROWD = teammates' prior-season MPG (15+ games) / 240, VACATED_SHARE = share
# of the team's minutes last season played by players no longer on it. Both are taken
# relative to that season's average team (_REL), because training uses end-of-season
# rosters (after trades and cuts) while projections use early-season rosters, which
# look less open across the board. Positional crowding added nothing.
OPPORTUNITY_FEATURES = ["TOTAL_CROWD_REL", "VACATED_SHARE_REL", "TEAM_QUALITY_REL"]
# TEAM_QUALITY_REL: returning teammates' last-season per-minute production (PROD_Z),
# weighted by their MPG, minus the average team. A thin roster of low producers leaves a
# rookie more room than the same minutes from good players. Held-out classes 2011-26:
# rookie MPG error better in 15 of 16 classes, season minutes 14 of 16, top-15 picks 13 of
# 16 (MPG MAE 5.13 -> 5.06, top-15 season-minute MAE 519 -> 512).
TEAM_QUALITY_FEATURE = "TEAM_QUALITY_REL"
# Playmaking need: a high-assist rookie on a team without a returning playmaker plays more.
# TEAM_TOP_AST36_REL = best assists per 36 among returning teammates who played 20+ MPG
# (15+ games) last season, minus the average team; C_AST_Z = the rookie's college assist
# rate, standardized on the training rookies (0 without college data); PLAYMAKER_GAP =
# C_AST_Z x -TEAM_TOP_AST36_REL. The NBA's G/F/C labels did not help (point and shooting
# guards share "G"). Held-out classes 2011-26: rookie MPG error better in 11 of 16 classes,
# top-15 picks 5.15 -> 5.06 MPG (8 of 16): a modest, borderline gain.
PLAYMAKER_FEATURES = ["TEAM_TOP_AST36_REL", "C_AST_Z", "PLAYMAKER_GAP"]
PLAYMAKER_MIN_MPG = 20.0
# Layer 3, college stats and size, for the per-minute rate models only. Each player's
# final college season (college.py, Torvik) plus NBA listed height. Players with no
# college line (international, G League Ignite, Overtime Elite) get HAS_COLLEGE = 0 and
# training means. The models train on classes from 2009 (college data starts 2007-08).
# Held-out classes 2016-26 (rookies with 200+ minutes, per-36 error vs layer 1): better
# in 9-11 of 11 classes for every stat; REB -43%, BLK -37%, AST -30%, 3PM -29%,
# STL -23%, FT% -16%, FG% -20%, PTS -7%. The NBA's listed position was left out (it can
# reflect the NBA role, a leak) and cost little. College stats did not help MPG or games.
COLLEGE_FIRST_CLASS = 2009
COLLEGE_FEATURES = [
    "C_AST", "C_TO", "C_STL", "C_BLK", "C_ORB", "C_DRB", "C_USG", "C_TS", "C_FTR",
    "C_FT", "C_3P", "C_3PA40", "C_PTS40", "C_BPM",
]
RATE_EXTRA_FEATURES = ["HEIGHT_IN", "HAS_COLLEGE"] + COLLEGE_FEATURES
MIN_COLLEGE_RATE_ROWS = 200
# Picks 4-15 correction: the MPG model under-projects these picks in almost every class
# (held-out 2011-26: about +2 MPG and +200 season minutes; picks 1-3 and later picks are
# close). A more flexible pick curve and recent-only training did not remove it. So their
# MPG gets the average held-out miss of earlier classes, recomputed on every fit from
# classes PICK_CORRECTION_FIRST_CLASS on (each predicted from the classes before it).
# Tested the same way on classes 2014-26: picks 4-15 season-minute error 537 -> 517
# (better in 10 of 13 classes), MPG error 5.43 -> 5.13.
PICK_CORRECTION_RANGE = (4, 15)
PICK_CORRECTION_FIRST_CLASS = 2008
PICK_CORRECTION_MIN_PLAYERS = 30
RATE_MIN_MINUTES = 200.0
RIDGE_ALPHA = 1.0
MPG_BOUNDS = (4.0, 36.0)
GAMES_BOUNDS = (10.0, 78.0)
# Pick buckets for a fallback age when a current rookie's age is unknown.
AGE_FALLBACK_BUCKETS = [0, 14, 30, 60]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_draft_history(path=DRAFT_HISTORY_PATH) -> pd.DataFrame:
    """One row per player: PLAYER_ID, DRAFT_YEAR, OVERALL_PICK (latest draft if re-drafted)."""
    raw = pd.read_csv(path)
    draft = pd.DataFrame(
        {
            "PLAYER_ID": pd.to_numeric(raw["PERSON_ID"], errors="coerce"),
            "DRAFT_YEAR": pd.to_numeric(raw["SEASON"], errors="coerce"),
            "OVERALL_PICK": pd.to_numeric(raw["OVERALL_PICK"], errors="coerce"),
        }
    ).dropna()
    draft = draft.loc[(draft["PLAYER_ID"] > 0) & (draft["OVERALL_PICK"] > 0)]
    draft = draft.sort_values("DRAFT_YEAR").drop_duplicates("PLAYER_ID", keep="last")
    return draft.astype({"PLAYER_ID": "int64", "DRAFT_YEAR": "int64"}).reset_index(drop=True)


def load_rookie_ages(path=ROOKIE_AGES_PATH) -> pd.Series:
    """Ages of current rookies keyed by PLAYER_ID (empty if the file is missing)."""
    if not Path(path).exists():
        return pd.Series(dtype=float)
    ages = pd.read_csv(path)
    return pd.Series(
        pd.to_numeric(ages["AGE"], errors="coerce").to_numpy(),
        index=pd.to_numeric(ages["PLAYER_ID"], errors="coerce").astype("int64"),
    ).dropna()


def load_heights(path=POSITIONS_PATH) -> pd.Series:
    """NBA listed height in inches keyed by PLAYER_ID (empty if the file is missing)."""
    if not Path(path).exists():
        return pd.Series(dtype=float)
    frame = pd.read_csv(path, usecols=["PLAYER_ID", "HEIGHT_IN"]).drop_duplicates("PLAYER_ID")
    return frame.set_index("PLAYER_ID")["HEIGHT_IN"]


def add_college_features(
    frame: pd.DataFrame,
    college: pd.DataFrame | None = None,
    heights: pd.Series | None = None,
) -> pd.DataFrame:
    """Add HEIGHT_IN, HAS_COLLEGE and the C_* college rates by PLAYER_ID.

    Small samples are shrunk: FT% toward .720 with 20 attempts, 3P% toward .330
    with 30 attempts. Players without a college line get NaN (filled with training
    means by the models) and HAS_COLLEGE = 0.
    """
    college = load_college_stats() if college is None else college
    heights = load_heights() if heights is None else heights
    out = frame.copy()
    out["HEIGHT_IN"] = out["PLAYER_ID"].map(heights).astype(float)
    if college.empty or "GP" not in college.columns:
        out["HAS_COLLEGE"] = 0.0
        for name in COLLEGE_FEATURES:
            out[name] = np.nan
        return out
    c = college.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID").reindex(out["PLAYER_ID"])
    c.index = out.index
    minutes = (c["GP"] * c["MPG"]).clip(lower=1.0)
    out["HAS_COLLEGE"] = c["GP"].notna().astype(float)
    out["C_AST"], out["C_TO"] = c["AST_PCT"], c["TO_PCT"]
    out["C_STL"], out["C_BLK"] = c["STL_PCT"], c["BLK_PCT"]
    out["C_ORB"], out["C_DRB"] = c["ORB_PCT"], c["DRB_PCT"]
    out["C_USG"], out["C_TS"], out["C_FTR"] = c["USG"], c["TS"], c["FTR"]
    out["C_FT"] = (c["FTM"] + 0.72 * 20) / (c["FTA"] + 20)
    out["C_3P"] = (c["FG3M"] + 0.33 * 30) / (c["FG3A"] + 30)
    out["C_3PA40"] = c["FG3A"] / minutes * 40
    out["C_PTS40"] = c["PTS"] * c["GP"] / minutes * 40
    out["C_BPM"] = c["BPM"]
    for name in COLLEGE_FEATURES:
        out[name] = pd.to_numeric(out[name], errors="coerce").astype(float)
    return out


def add_draft_features(frame: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """Add OVERALL_PICK, DRAFT_YEAR, UNDRAFTED and LOG_PICK by PLAYER_ID (AGE must exist)."""
    out = frame.drop(columns=["OVERALL_PICK", "DRAFT_YEAR"], errors="ignore").merge(
        draft, on="PLAYER_ID", how="left", validate="many_to_one"
    )
    out.index = frame.index
    out["OVERALL_PICK"] = pd.to_numeric(out["OVERALL_PICK"], errors="coerce").astype(float)
    out["UNDRAFTED"] = out["OVERALL_PICK"].isna().astype(float)
    out["LOG_PICK"] = np.log(out["OVERALL_PICK"].fillna(UNDRAFTED_PICK))
    out["LOG_PICK_SQ"] = out["LOG_PICK"] ** 2
    return out


def league_rates(history: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """Minute-weighted league rate per stat, indexed by SEASON_YEAR."""
    minutes = history["MIN"]
    made = history[[f"{stat}_RATE" for stat in stats]].mul(minutes, axis=0)
    made.columns = stats
    totals = made.groupby(history["SEASON_YEAR"]).sum()
    return totals.div(minutes.groupby(history["SEASON_YEAR"]).sum(), axis=0)


def rookie_seasons(history: pd.DataFrame, draft: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """One row per rookie season with draft features and targets.

    Targets: MPG (MIN / GP), GAMES_82 (games scaled to 82), and [STAT]_INDEX
    (rookie rate / league rate that season).
    """
    ordered = history.sort_values(["PLAYER_ID", "SEASON_YEAR"])
    first = ordered.drop_duplicates("PLAYER_ID", keep="first")
    first = first.loc[first["SEASON_YEAR"] >= ROOKIE_FIRST_SEASON].copy()
    first = add_draft_features(first, draft)
    # Drafted before the data starts and first seen 2+ years later: may have
    # played before 1996-97, so not a sure rookie season.
    early = (first["DRAFT_YEAR"] < ROOKIE_FIRST_SEASON - 2) & (
        first["SEASON_YEAR"] - first["DRAFT_YEAR"] > 1
    )
    first = first.loc[~early & first["AGE"].notna()].copy()
    first["MPG"] = first["MIN"] / first["GP"]
    first["GAMES_82"] = _games_per_82(first)
    league = league_rates(history, stats)
    for stat in stats:
        first[f"{stat}_INDEX"] = first[f"{stat}_RATE"] / first["SEASON_YEAR"].map(league[stat])
    first = first.merge(
        season_opportunity(history), on=["PLAYER_ID", "SEASON_YEAR"], how="left", validate="one_to_one"
    )
    return add_college_features(first).reset_index(drop=True)


def team_opportunity(
    players: pd.DataFrame, last_season: pd.DataFrame
) -> pd.DataFrame:
    """TOTAL_CROWD and VACATED_SHARE for each player on `players`' teams.

    `players`: PLAYER_ID, TEAM (the team now) and PRIOR_MPG (last season's MPG,
    NaN without a 15-game season). `last_season`: PLAYER_ID, TEAM, MIN for the
    previous season (each player credited to his last team). VACATED_SHARE is the
    share of a team's last-season minutes played by players not on it now.
    """
    crowd = position_crowding(players, pd.Series(dtype=object))["TOTAL_CROWD"]
    now = players[["PLAYER_ID", "TEAM"]].rename(columns={"TEAM": "TEAM_NOW"})
    before = last_season.merge(now, on="PLAYER_ID", how="left")
    before["GONE"] = before["MIN"].where(before["TEAM_NOW"] != before["TEAM"], 0.0)
    by_team = before.groupby("TEAM")[["GONE", "MIN"]].sum()
    vacated = by_team["GONE"] / by_team["MIN"]
    teams = players["TEAM"].unique()
    # Best returning playmaker: highest last-season AST per 36 among 20+ MPG teammates.
    if "PRIOR_AST36" in players:
        handlers = players.loc[players["PRIOR_MPG"] >= PLAYMAKER_MIN_MPG]
        top_ast = handlers.groupby("TEAM")["PRIOR_AST36"].max()
    else:
        top_ast = pd.Series(np.nan, index=teams)
    # Teammate quality: MPG-weighted last-season production of players with a prior season.
    if "PRIOR_PROD" in players:
        weight = players["PRIOR_MPG"].where(players["PRIOR_PROD"].notna()).fillna(0.0)
        weighted = (players["PRIOR_PROD"].fillna(0.0) * weight).groupby(players["TEAM"]).sum()
        quality = weighted / weight.groupby(players["TEAM"]).sum().replace(0.0, np.nan)
    else:
        quality = pd.Series(np.nan, index=teams)
    # Average team: full roster demand (as seen by a player with no prior season) and
    # vacated share, averaged over the teams on `players`.
    demand = players["PRIOR_MPG"].fillna(0.0).groupby(players["TEAM"]).sum() / 240.0
    out = pd.DataFrame(
        {"TOTAL_CROWD": crowd, "VACATED_SHARE": players["TEAM"].map(vacated).to_numpy()},
        index=players.index,
    )
    out["TOTAL_CROWD_REL"] = out["TOTAL_CROWD"] - float(demand.reindex(teams).mean())
    out["VACATED_SHARE_REL"] = out["VACATED_SHARE"] - float(vacated.reindex(teams).mean())
    out["TEAM_QUALITY"] = players["TEAM"].map(quality).to_numpy()
    out["TEAM_QUALITY_REL"] = out["TEAM_QUALITY"] - float(quality.reindex(teams).mean())
    out["TEAM_TOP_AST36"] = players["TEAM"].map(top_ast).to_numpy()
    out["TEAM_TOP_AST36_REL"] = out["TEAM_TOP_AST36"] - float(top_ast.reindex(teams).mean())
    return out


def season_opportunity(history: pd.DataFrame) -> pd.DataFrame:
    """Team opportunity for every player-season, on end-of-season teams."""
    if "TEAM_ABBREVIATION" not in history.columns:
        return pd.DataFrame(
            columns=["PLAYER_ID", "SEASON_YEAR", "TOTAL_CROWD", "VACATED_SHARE", "TEAM_QUALITY",
                     "TEAM_TOP_AST36", "TEAM_TOP_AST36_REL", *OPPORTUNITY_FEATURES]
        )
    h = history[["PLAYER_ID", "SEASON_YEAR", "TEAM_ABBREVIATION", "GP", "MIN", "AST_RATE"]].copy()
    h["MPG"] = h["MIN"] / h["GP"]
    production = season_production_z(history).set_index(["SEASON_YEAR", "PLAYER_ID"])["PROD_Z"]
    frames = []
    for year, season in h.groupby("SEASON_YEAR"):
        last = h.loc[h["SEASON_YEAR"] == year - 1]
        if last.empty:
            continue
        prior = last.loc[last["GP"] >= 15].set_index("PLAYER_ID")["MPG"]
        prior_ast = (last.loc[last["GP"] >= 15].set_index("PLAYER_ID")["AST_RATE"] * 36.0)
        prior_prod = production.xs(year - 1, level="SEASON_YEAR")
        players = pd.DataFrame(
            {
                "PLAYER_ID": season["PLAYER_ID"].to_numpy(),
                "TEAM": season["TEAM_ABBREVIATION"].to_numpy(),
                "PRIOR_MPG": season["PLAYER_ID"].map(prior).to_numpy(),
                "PRIOR_PROD": season["PLAYER_ID"].map(prior_prod).to_numpy(),
                "PRIOR_AST36": season["PLAYER_ID"].map(prior_ast).to_numpy(),
            }
        )
        opportunity = team_opportunity(
            players, last.rename(columns={"TEAM_ABBREVIATION": "TEAM"})[["PLAYER_ID", "TEAM", "MIN"]]
        )
        opportunity["PLAYER_ID"], opportunity["SEASON_YEAR"] = players["PLAYER_ID"].to_numpy(), year
        frames.append(opportunity)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def _ridge() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])


def fit_rookie_models(
    rookies: pd.DataFrame,
    stats: list[str],
    use_opportunity: bool = True,
    use_college: bool = True,
    use_pick_correction: bool = True,
    use_team_quality: bool = True,
    use_playmaker: bool = True,
) -> dict:
    """Fit rate-index, MPG and games models on the rookie seasons given.

    With `use_opportunity`, the MPG and games models add OPPORTUNITY_FEATURES
    (rookies without them get the training mean).
    """
    if len(rookies) < 50:
        raise ValueError(f"Only {len(rookies)} rookie seasons; too few to fit rookie models.")
    rate_rows = rookies.loc[rookies["MIN"] >= RATE_MIN_MINUTES]
    college_rows = rate_rows.loc[rate_rows["SEASON_YEAR"] >= COLLEGE_FIRST_CLASS] if "SEASON_YEAR" in rate_rows else rate_rows.iloc[0:0]
    use_college = (
        use_college
        and all(name in rate_rows for name in RATE_EXTRA_FEATURES)
        and float(college_rows.get("HAS_COLLEGE", pd.Series(dtype=float)).sum()) >= MIN_COLLEGE_RATE_ROWS
    )
    rate_features = FEATURES + (RATE_EXTRA_FEATURES if use_college else [])
    if use_college:
        rate_rows = college_rows
    rate_means = {name: float(rate_rows[name].mean()) for name in RATE_EXTRA_FEATURES if use_college}
    rate_X = rate_rows[rate_features].fillna(rate_means)
    weight = rate_rows["MIN"].to_numpy(dtype=float)
    weight = weight / weight.mean()
    rate_models = {}
    for stat in stats:
        model = _ridge()
        model.fit(rate_X, rate_rows[f"{stat}_INDEX"], ridge__sample_weight=weight)
        rate_models[stat] = model
    # Without team data (no TEAM_ABBREVIATION in the history) there is nothing to learn from.
    opportunity = [
        name for name in OPPORTUNITY_FEATURES if use_team_quality or name != TEAM_QUALITY_FEATURE
    ]
    use_opportunity = use_opportunity and all(
        name in rookies and rookies[name].notna().sum() >= 50 for name in opportunity
    )
    minutes_features = FEATURES + (opportunity if use_opportunity else [])
    opportunity_means = {
        name: float(rookies[name].mean()) for name in OPPORTUNITY_FEATURES if name in rookies
    }
    minutes_rows = rookies.copy()
    if use_opportunity:
        minutes_rows[opportunity] = minutes_rows[opportunity].fillna(opportunity_means)
    ast_mean, ast_sd = np.nan, np.nan
    use_playmaker = (
        use_playmaker
        and use_opportunity
        and "TEAM_TOP_AST36_REL" in minutes_rows
        and "C_AST" in minutes_rows
        and minutes_rows["C_AST"].notna().sum() >= 50
    )
    if use_playmaker:
        ast_mean = float(minutes_rows["C_AST"].mean())
        ast_sd = float(minutes_rows["C_AST"].std()) or 1.0
        minutes_rows = add_playmaker_features(minutes_rows, ast_mean, ast_sd)
        minutes_features = minutes_features + PLAYMAKER_FEATURES
    mpg_model = _ridge().fit(minutes_rows[minutes_features], minutes_rows["MPG"])
    games_model = _ridge().fit(minutes_rows[minutes_features], minutes_rows["GAMES_82"])
    pick_shift = held_out_pick_shift(minutes_rows, minutes_features) if use_pick_correction else 0.0

    drafted = rookies.loc[rookies["UNDRAFTED"] == 0]
    buckets = pd.cut(drafted["OVERALL_PICK"], AGE_FALLBACK_BUCKETS)
    fallback_age = drafted.groupby(buckets, observed=True)["AGE"].median()
    return {
        "rates": rate_models,
        "mpg": mpg_model,
        "games": games_model,
        "minutes_features": minutes_features,
        "c_ast_mean": ast_mean,
        "c_ast_sd": ast_sd,
        "pick_mpg_shift": pick_shift,
        "rate_features": rate_features,
        "rate_means": rate_means,
        "opportunity_means": opportunity_means,
        "stats": list(stats),
        "rate_rows": int(len(rate_rows)),
        "rows": int(len(rookies)),
        "last_class": int(rookies["SEASON_YEAR"].max()),
        "fallback_age": {str(k): float(v) for k, v in fallback_age.items()},
        "undrafted_age": float(rookies.loc[rookies["UNDRAFTED"] == 1, "AGE"].median()),
    }


def add_playmaker_features(frame: pd.DataFrame, ast_mean: float, ast_sd: float) -> pd.DataFrame:
    """Add C_AST_Z and PLAYMAKER_GAP (see PLAYMAKER_FEATURES); missing values count as 0."""
    out = frame.copy()
    need_column = out["TEAM_TOP_AST36_REL"] if "TEAM_TOP_AST36_REL" in out else pd.Series(0.0, index=out.index)
    team_need = -pd.to_numeric(need_column, errors="coerce").fillna(0.0)
    out["TEAM_TOP_AST36_REL"] = -team_need
    college_ast = pd.to_numeric(out.get("C_AST", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["C_AST_Z"] = ((college_ast - ast_mean) / ast_sd).fillna(0.0)
    out["PLAYMAKER_GAP"] = out["C_AST_Z"] * team_need
    return out


def held_out_pick_shift(rows: pd.DataFrame, features: list[str]) -> float:
    """Average MPG miss for picks 4-15 when each class is predicted from earlier classes.

    Uses only the rookie seasons given (so a past-only training set stays leak-free).
    Returns 0 when there are too few held-out picks to trust.
    """
    if "SEASON_YEAR" not in rows or "OVERALL_PICK" not in rows:
        return 0.0
    low, high = PICK_CORRECTION_RANGE
    misses = []
    for year in sorted(rows["SEASON_YEAR"].unique()):
        if year < PICK_CORRECTION_FIRST_CLASS:
            continue
        before, test = rows.loc[rows["SEASON_YEAR"] < year], rows.loc[rows["SEASON_YEAR"] == year]
        test = test.loc[test["OVERALL_PICK"].between(low, high)]
        if len(before) < 50 or test.empty:
            continue
        model = _ridge().fit(before[features], before["MPG"])
        predicted = np.clip(model.predict(test[features]), *MPG_BOUNDS)
        misses.extend((test["MPG"] - predicted).tolist())
    if len(misses) < PICK_CORRECTION_MIN_PLAYERS:
        return 0.0
    return float(np.mean(misses))


def fill_missing_age(models: dict, features: pd.DataFrame) -> pd.Series:
    """AGE, with the training median for the pick bucket where it is missing."""
    age = features["AGE"].astype(float).copy()
    missing = age.isna()
    if missing.any():
        buckets = pd.cut(features["OVERALL_PICK"], AGE_FALLBACK_BUCKETS).astype(str)
        fallback = buckets.map(models["fallback_age"]).astype(float)
        fallback = fallback.where(features["UNDRAFTED"] == 0, models["undrafted_age"])
        age = age.where(~missing, fallback.fillna(models["undrafted_age"]))
    return age


def predict_rookies(models: dict, features: pd.DataFrame, league_rate: pd.Series) -> pd.DataFrame:
    """Rookie prior: per-minute rates (one column per stat), MPG and GAMES_82.

    `features` needs OVERALL_PICK, UNDRAFTED, LOG_PICK, LOG_PICK_SQ and AGE, and
    optionally OPPORTUNITY_FEATURES (missing -> training mean); `league_rate` is
    the latest completed season's league rate per stat (no future seasons).
    """
    X = features.copy()
    X["AGE"] = fill_missing_age(models, features)
    for name in OPPORTUNITY_FEATURES:
        if name in models["minutes_features"]:
            column = X[name] if name in X else pd.Series(np.nan, index=X.index)
            X[name] = column.fillna(models["opportunity_means"][name])
    if "PLAYMAKER_GAP" in models["minutes_features"]:
        X = add_playmaker_features(X, models["c_ast_mean"], models["c_ast_sd"])
    out = pd.DataFrame(index=features.index)
    rate_features = models.get("rate_features", FEATURES)
    for name in rate_features:
        if name not in FEATURES:
            column = X[name] if name in X else pd.Series(np.nan, index=X.index)
            X[name] = column.fillna(models["rate_means"][name])
    for stat in models["stats"]:
        index = np.clip(models["rates"][stat].predict(X[rate_features]), 0.0, None)
        out[stat] = index * float(league_rate[stat])
    minutes_X = X[models["minutes_features"]]
    mpg = models["mpg"].predict(minutes_X)
    low, high = PICK_CORRECTION_RANGE
    corrected = features["OVERALL_PICK"].between(low, high).to_numpy()
    mpg = mpg + np.where(corrected, models.get("pick_mpg_shift", 0.0), 0.0)
    out["MPG"] = np.clip(mpg, *MPG_BOUNDS)
    out["GAMES_82"] = np.clip(models["games"].predict(minutes_X), *GAMES_BOUNDS)
    return out


def prior_features(
    player_ids: pd.Series,
    history: pd.DataFrame,
    draft: pd.DataFrame,
    current_ages: pd.Series,
) -> pd.DataFrame:
    """Draft features for each player, using rookie-season age.

    Players with history use the AGE of their first season (only if that season
    is a real rookie season, otherwise AGE is NaN and no prior applies). Players
    with no history use `current_ages` (this season's rookies).
    """
    frame = pd.DataFrame({"PLAYER_ID": player_ids.astype("int64")}, index=player_ids.index)
    first = history.sort_values("SEASON_YEAR").drop_duplicates("PLAYER_ID", keep="first")
    first = first.loc[first["SEASON_YEAR"] >= ROOKIE_FIRST_SEASON].set_index("PLAYER_ID")
    in_history = frame["PLAYER_ID"].isin(history["PLAYER_ID"])
    frame["AGE"] = np.where(
        in_history,
        frame["PLAYER_ID"].map(first["AGE"]),
        frame["PLAYER_ID"].map(current_ages),
    ).astype(float)
    frame["HAS_ROOKIE_PRIOR"] = ~in_history | frame["PLAYER_ID"].isin(first.index)
    frame["NBA_HISTORY"] = in_history
    return add_college_features(add_draft_features(frame, draft))
