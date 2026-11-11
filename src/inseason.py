"""Up-to-date ("Current") projections: actual games so far + blended projection of the rest.

Settings come from the in-season backtests (analysis/inseason_minutes.py and
analysis/inseason_rates.py; seasons 2020-26 held out, cutoffs after ~10/20/41/60 games):

- Minutes per game for the remaining games: w = n / (n + MINUTES_K) on observed MPG,
  1 - w on the preseason MPG, n = games played. From RECENT_FROM games on, observed MPG is
  (1 - RECENT_WEIGHT) x season MPG + RECENT_WEIGHT x last-10-games MPG (only clearly
  better from ~40 games: 7/7 seasons at 41 and 60, 4-5/7 at 10-20). Blend beat preseason
  only and observed only in 7/7 seasons at every cutoff.
- Per-minute stats: (k x preseason rate + observed total) / (k + observed minutes), k per
  stat in minutes (RATE_K); FG% and FT%: (k x preseason % + makes) / (k + attempts), k in
  attempts (PCT_K); FGM and FTM = blended % x blended attempts. 6-7 of 7 seasons for
  every stat and cutoff. k values are from the 20-game cutoff (they drift a little lower
  later in the season).
- Not used (failed the every-season rule): a separate rookie k, stripping minutes gained
  from absent teammates, holding teams to 240 minutes a game.

Known injuries enter through project.apply_injuries: remaining games, and the teammate
opportunity on the preseason part of the blend (observed minutes already reflect the
current absences).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES_K = 10.0
RECENT_WEIGHT = 0.5
RECENT_FROM = 30
RECENT_GAMES = 10
RATE_K = {
    "FGA": 200.0, "AST": 300.0, "FG3M": 300.0, "REB": 500.0, "PTS": 500.0,
    "FTA": 500.0, "TOV": 750.0, "BLK": 1000.0, "STL": 1500.0,
}
PCT_K = {"FG": ("FGM", "FGA", 500.0), "FT": ("FTM", "FTA", 200.0)}
TOTAL_STATS = ["PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FTM", "FTA"]


def observed_stats(season_log: pd.DataFrame, today=None) -> pd.DataFrame:
    """Per player this season (games before `today`): GP, MIN, stat totals, last-10 MPG.

    The frame's attrs["as_of"] is the latest game date included.
    """
    log = season_log.copy()
    if log.empty:
        out = pd.DataFrame(columns=["GP", "MIN", "MPG_L10", *TOTAL_STATS])
        out.attrs["as_of"] = None
        return out
    log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"]).dt.date
    if today is not None:
        log = log[log["GAME_DATE"] < today]
    log = log.sort_values(["GAME_DATE", "GAME_ID"])
    out = log.groupby("PLAYER_ID")[["MIN", *TOTAL_STATS]].sum()
    out["GP"] = log.groupby("PLAYER_ID").size()
    out["MPG_L10"] = log.groupby("PLAYER_ID").tail(RECENT_GAMES).groupby("PLAYER_ID")["MIN"].mean()
    out.attrs["as_of"] = log["GAME_DATE"].max() if len(log) else None
    return out


def blend_minutes(prior_mpg: pd.Series, observed: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(MPG for the remaining games, weight on observed minutes). Indexed like prior_mpg."""
    n = observed["GP"].reindex(prior_mpg.index).fillna(0.0)
    minutes = observed["MIN"].reindex(prior_mpg.index).fillna(0.0)
    season = minutes / n.where(n > 0)
    recent = observed["MPG_L10"].reindex(prior_mpg.index).fillna(season)
    r = np.where(n >= RECENT_FROM, RECENT_WEIGHT, 0.0)
    obs_mpg = (1 - r) * season + r * recent
    w = n / (n + MINUTES_K)
    blended = ((1 - w) * prior_mpg + w * obs_mpg).where(n > 0, prior_mpg)
    return blended.rename("ROS_MPG"), w.rename("MINUTES_OBS_WEIGHT")


def blend_rates(prior_rates: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Per-minute rates for the remaining games. Columns as prior_rates (TOTAL_STATS)."""
    obs = observed.reindex(prior_rates.index)
    minutes = obs["MIN"].fillna(0.0)
    out = prior_rates.copy()
    for stat, k in RATE_K.items():
        out[stat] = (k * prior_rates[stat] + obs[stat].fillna(0.0)) / (k + minutes)
    for made, att, k in PCT_K.values():
        prior_pct = (prior_rates[made] / prior_rates[att].where(prior_rates[att] > 0)).fillna(0.0)
        attempts = obs[att].fillna(0.0)
        pct = (k * prior_pct + obs[made].fillna(0.0)) / (k + attempts)
        out[made] = pct * out[att]
    return out


def combine_with_actual(projections: pd.DataFrame, observed: pd.DataFrame, as_of) -> pd.DataFrame:
    """Turn a rest-of-season projection into a whole-season one: actual + remaining.

    Expects projections whose PROJECTED_GP / PROJECTED_SEASON_MIN / TOTAL_* cover only the
    remaining games. Per-game columns become whole-season averages.
    """
    p = projections.copy()
    obs = observed.reindex(p["PLAYER_ID"].to_numpy())
    obs.index = p.index
    p["ROS_GP"] = p["PROJECTED_GP"]
    p["ROS_MPG"] = p["PROJECTED_MPG"]
    p["ACTUAL_GP"] = obs["GP"].fillna(0).astype(int)
    p["ACTUAL_MIN"] = obs["MIN"].fillna(0.0)
    gp = p["ROS_GP"] + p["ACTUAL_GP"]
    season_min = p["PROJECTED_SEASON_MIN"] + p["ACTUAL_MIN"]
    for stat in TOTAL_STATS:
        p[f"TOTAL_{stat}"] = p[f"TOTAL_{stat}"] + obs[stat].fillna(0.0)
        p[f"PROJ_{stat}"] = p[f"TOTAL_{stat}"] / gp.where(gp > 0)
    p["PROJECTED_GP"] = gp
    p["PROJECTED_SEASON_MIN"] = season_min
    p["PROJECTED_MPG"] = (season_min / gp.where(gp > 0)).round(1)
    p["PROJ_FG_PCT"] = (p["TOTAL_FGM"] / p["TOTAL_FGA"]).where(p["TOTAL_FGA"] > 0)
    p["PROJ_FT_PCT"] = (p["TOTAL_FTM"] / p["TOTAL_FTA"]).where(p["TOTAL_FTA"] > 0)
    p["AS_OF"] = str(as_of) if as_of is not None else ""
    return p
