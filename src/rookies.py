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

from history import _games_per_82, position_crowding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_HISTORY_PATH = PROJECT_ROOT / "data" / "raw" / "draft_history.csv"
ROOKIE_AGES_PATH = PROJECT_ROOT / "data" / "raw" / "rookie_ages.csv"

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
OPPORTUNITY_FEATURES = ["TOTAL_CROWD_REL", "VACATED_SHARE_REL"]
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
    return first.reset_index(drop=True)


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
    # Average team: full roster demand (as seen by a player with no prior season) and
    # vacated share, averaged over the teams on `players`.
    demand = players["PRIOR_MPG"].fillna(0.0).groupby(players["TEAM"]).sum() / 240.0
    out = pd.DataFrame(
        {"TOTAL_CROWD": crowd, "VACATED_SHARE": players["TEAM"].map(vacated).to_numpy()},
        index=players.index,
    )
    out["TOTAL_CROWD_REL"] = out["TOTAL_CROWD"] - float(demand.reindex(teams).mean())
    out["VACATED_SHARE_REL"] = out["VACATED_SHARE"] - float(vacated.reindex(teams).mean())
    return out


def season_opportunity(history: pd.DataFrame) -> pd.DataFrame:
    """Team opportunity for every player-season, on end-of-season teams."""
    if "TEAM_ABBREVIATION" not in history.columns:
        return pd.DataFrame(
            columns=["PLAYER_ID", "SEASON_YEAR", "TOTAL_CROWD", "VACATED_SHARE", *OPPORTUNITY_FEATURES]
        )
    h = history[["PLAYER_ID", "SEASON_YEAR", "TEAM_ABBREVIATION", "GP", "MIN"]].copy()
    h["MPG"] = h["MIN"] / h["GP"]
    frames = []
    for year, season in h.groupby("SEASON_YEAR"):
        last = h.loc[h["SEASON_YEAR"] == year - 1]
        if last.empty:
            continue
        prior = last.loc[last["GP"] >= 15].set_index("PLAYER_ID")["MPG"]
        players = pd.DataFrame(
            {
                "PLAYER_ID": season["PLAYER_ID"].to_numpy(),
                "TEAM": season["TEAM_ABBREVIATION"].to_numpy(),
                "PRIOR_MPG": season["PLAYER_ID"].map(prior).to_numpy(),
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


def fit_rookie_models(rookies: pd.DataFrame, stats: list[str], use_opportunity: bool = True) -> dict:
    """Fit rate-index, MPG and games models on the rookie seasons given.

    With `use_opportunity`, the MPG and games models add OPPORTUNITY_FEATURES
    (rookies without them get the training mean).
    """
    if len(rookies) < 50:
        raise ValueError(f"Only {len(rookies)} rookie seasons; too few to fit rookie models.")
    rate_rows = rookies.loc[rookies["MIN"] >= RATE_MIN_MINUTES]
    weight = rate_rows["MIN"].to_numpy(dtype=float)
    weight = weight / weight.mean()
    rate_models = {}
    for stat in stats:
        model = _ridge()
        model.fit(rate_rows[FEATURES], rate_rows[f"{stat}_INDEX"], ridge__sample_weight=weight)
        rate_models[stat] = model
    # Without team data (no TEAM_ABBREVIATION in the history) there is nothing to learn from.
    use_opportunity = use_opportunity and all(
        name in rookies and rookies[name].notna().sum() >= 50 for name in OPPORTUNITY_FEATURES
    )
    minutes_features = FEATURES + (OPPORTUNITY_FEATURES if use_opportunity else [])
    opportunity_means = {
        name: float(rookies[name].mean()) for name in OPPORTUNITY_FEATURES if name in rookies
    }
    minutes_rows = rookies.copy()
    if use_opportunity:
        minutes_rows[OPPORTUNITY_FEATURES] = minutes_rows[OPPORTUNITY_FEATURES].fillna(opportunity_means)
    mpg_model = _ridge().fit(minutes_rows[minutes_features], minutes_rows["MPG"])
    games_model = _ridge().fit(minutes_rows[minutes_features], minutes_rows["GAMES_82"])

    drafted = rookies.loc[rookies["UNDRAFTED"] == 0]
    buckets = pd.cut(drafted["OVERALL_PICK"], AGE_FALLBACK_BUCKETS)
    fallback_age = drafted.groupby(buckets, observed=True)["AGE"].median()
    return {
        "rates": rate_models,
        "mpg": mpg_model,
        "games": games_model,
        "minutes_features": minutes_features,
        "opportunity_means": opportunity_means,
        "stats": list(stats),
        "rate_rows": int(len(rate_rows)),
        "rows": int(len(rookies)),
        "last_class": int(rookies["SEASON_YEAR"].max()),
        "fallback_age": {str(k): float(v) for k, v in fallback_age.items()},
        "undrafted_age": float(rookies.loc[rookies["UNDRAFTED"] == 1, "AGE"].median()),
    }


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
    out = pd.DataFrame(index=features.index)
    for stat in models["stats"]:
        index = np.clip(models["rates"][stat].predict(X[FEATURES]), 0.0, None)
        out[stat] = index * float(league_rate[stat])
    minutes_X = X[models["minutes_features"]]
    out["MPG"] = np.clip(models["mpg"].predict(minutes_X), *MPG_BOUNDS)
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
    return add_draft_features(frame, draft)
