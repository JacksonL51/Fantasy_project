"""In-season rates study: how much should current per-minute stats and percentages count?

Read-only. Backtest for the up-to-date ("Current") projections, step 2 (rates).
Results in analysis/results/inseason_rates_*. Same seasons, cutoffs and cached preseason
priors as analysis/inseason_minutes.py.

- Counting stats (PTS, FG3M, REB, AST, STL, BLK, TOV, FGA, FTA): blended rate =
  (k * prior rate + observed minutes * observed rate) / (k + observed minutes), k in
  minutes, picked per stat on earlier seasons. Score: per-36 MAE weighted by
  rest-of-season minutes (players with 100+ minutes after the cutoff).
- FG% and FT%: (k * prior % + makes) / (k + attempts), k in attempts. Score: MAE in
  percentage points weighted by rest-of-season attempts (30+ attempts after the cutoff).
- Rookies (no NBA season before): a separate multiplier on k (k * m, m picked on earlier
  seasons), kept only if it beats one shared k every season.
- Full-season value: actual totals before the cutoff + projected rest of season (minutes
  from the step-1 blend x true games after the cutoff x rates), 9-category z-score sum,
  Spearman vs actual full-season value. Variants: preseason only, minutes blend only,
  minutes + rate blend. Players with 10+ games after the cutoff.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
sys.path.insert(0, "analysis")

import inseason_minutes as IM  # noqa: E402
import project as P  # noqa: E402
from gamelogs import load_gamelogs  # noqa: E402

COUNT_STATS = ["PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV", "FGA", "FTA"]
PCT_STATS = {"FG": ("FGM", "FGA"), "FT": ("FTM", "FTA")}
ALL_TOTALS = ["PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FTM", "FTA"]
K_MIN_GRID = [0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2500, 5000, 1e9]
K_ATT_GRID = [0, 25, 50, 100, 150, 200, 300, 500, 1000, 1e9]
ROOKIE_MULT_GRID = [0.25, 0.5, 1.0]
MIN_ROS_MINUTES = 100
MIN_ROS_ATTEMPTS = 30
# Step-1 minutes settings (chosen): k = 10 games, last-10 weight 0.5 after 30 games.
MINUTES_K, RECENT_R, RECENT_FROM = 10, 0.5, 30
RESULTS = IM.RESULTS


def snapshot(year: int, n_games: int, logs: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame | None:
    d = IM.cutoff_date(logs, n_games)
    if d is None:
        return None
    before, after = logs[logs["GAME_DATE"] < d], logs[logs["GAME_DATE"] >= d]
    obs = before.groupby("PLAYER_ID")[["MIN"] + ALL_TOTALS].sum().add_prefix("OBS_")
    obs["OBS_GP"] = before.groupby("PLAYER_ID").size()
    obs["OBS_L10_MPG"] = before.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(10).groupby("PLAYER_ID")["MIN"].mean()
    ros = after.groupby("PLAYER_ID")[["MIN"] + ALL_TOTALS].sum().add_prefix("ROS_")
    ros["ROS_GP"] = after.groupby("PLAYER_ID").size()
    pr = prior.set_index("PLAYER_ID")
    prior_rates = pr[[f"TOTAL_{s}" for s in ALL_TOTALS]].div(pr["PROJECTED_SEASON_MIN"], axis=0)
    prior_rates.columns = [f"PRIOR_{s}" for s in ALL_TOTALS]
    df = ros.join(obs, how="left").join(prior_rates, how="inner")
    df["PRIOR_MPG"] = pr["PROJECTED_MPG"]
    df["ROOKIE"] = pr["NO_PRIOR_DATA"].astype(bool) & pr.get("ROOKIE_PRIOR", False).astype(bool)
    df = df.fillna({c: 0 for c in df.columns if c.startswith("OBS_") and c != "OBS_L10_MPG"})
    df["SEASON"], df["CUTOFF"] = year, n_games
    return df.reset_index()


def blend_rate(df, stat, k, rookie_mult=1.0):
    kk = np.where(df["ROOKIE"], k * rookie_mult, k)
    num = kk * df[f"PRIOR_{stat}"] + df[f"OBS_{stat}"]
    den = kk + df["OBS_MIN"]
    out = np.where(den > 0, num / np.where(den > 0, den, 1), df[f"PRIOR_{stat}"])
    return pd.Series(np.where(np.isinf(kk), df[f"PRIOR_{stat}"], out), index=df.index)


def blend_pct(df, made, att, k, rookie_mult=1.0):
    prior = df[f"PRIOR_{made}"] / df[f"PRIOR_{att}"].replace(0, np.nan)
    prior = prior.fillna(prior.median())
    kk = np.where(df["ROOKIE"], k * rookie_mult, k)
    den = kk + df[f"OBS_{att}"]
    out = (kk * prior + df[f"OBS_{made}"]) / den.replace(0, np.nan)
    out = out.where(~np.isinf(kk), prior)
    return out.fillna(prior)


def rate_error(df, stat, pred):
    part = df["ROS_MIN"] >= MIN_ROS_MINUTES
    actual = df.loc[part, f"ROS_{stat}"] / df.loc[part, "ROS_MIN"]
    return float(np.average(((pred[part] - actual) * 36).abs(), weights=df.loc[part, "ROS_MIN"]))


def pct_error(df, made, att, pred):
    part = df[f"ROS_{att}"] >= MIN_ROS_ATTEMPTS
    actual = df.loc[part, f"ROS_{made}"] / df.loc[part, f"ROS_{att}"]
    return float(np.average((pred[part] - actual).abs() * 100, weights=df.loc[part, f"ROS_{att}"]))


def minutes_blend(df):
    n = df["OBS_GP"]
    w = n / (n + MINUTES_K)
    season = (df["OBS_MIN"] / n.replace(0, np.nan))
    recent = df["OBS_L10_MPG"]
    r = np.where(n >= RECENT_FROM, RECENT_R, 0.0)
    observed = ((1 - r) * season + r * recent.fillna(season)).fillna(df["PRIOR_MPG"])
    return ((1 - w) * df["PRIOR_MPG"] + w * observed).where(n > 0, df["PRIOR_MPG"])


def nine_cat_value(t: pd.DataFrame) -> pd.Series:
    parts = []
    for s in ["PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV"]:
        z = (t[s] - t[s].mean()) / t[s].std()
        parts.append(-z if s == "TOV" else z)
    for made, att in (("FGM", "FGA"), ("FTM", "FTA")):
        league = t[made].sum() / t[att].sum()
        impact = t[made] - league * t[att]
        parts.append((impact - impact.mean()) / impact.std())
    return sum(parts)


def full_season_value(df, mpg, rates: dict) -> float:
    proj_min = mpg * df["ROS_GP"]
    totals = pd.DataFrame({s: df[f"OBS_{s}"] + rates[s] * proj_min for s in ALL_TOTALS})
    actual = pd.DataFrame({s: df[f"OBS_{s}"] + df[f"ROS_{s}"] for s in ALL_TOTALS})
    return float(spearmanr(nine_cat_value(totals), nine_cat_value(actual)).correlation)


def main() -> None:
    history = P.load_history()
    frames = []
    for year in IM.SEASONS:
        logs = load_gamelogs([year])
        prior = IM.preseason_prior(year, history, logs)
        for n in IM.CUTOFFS:
            snap = snapshot(year, n, logs, prior)
            if snap is not None:
                frames.append(snap)
    data = pd.concat(frames, ignore_index=True)
    data = data[data["ROS_GP"] >= IM.MIN_ROS_GAMES].copy()
    # FGM / FTM rates (for value) follow the blended FG% / FT% x blended attempts.
    rows, picks = [], []
    for cutoff, part in data.groupby("CUTOFF"):
        for s in IM.SEASONS[1:]:
            train, test = part[part["SEASON"] < s], part[part["SEASON"] == s]
            for stat in COUNT_STATS:
                k = min(K_MIN_GRID, key=lambda k: rate_error(train, stat, blend_rate(train, stat, k)))
                m = min(ROOKIE_MULT_GRID, key=lambda m: rate_error(train, stat, blend_rate(train, stat, k, m)))
                rows.append({
                    "cutoff": cutoff, "season": s, "stat": stat, "k": k, "rookie_mult": m,
                    "prior": rate_error(test, stat, blend_rate(test, stat, 1e9)),
                    "observed": rate_error(test, stat, blend_rate(test, stat, 0)),
                    "blend": rate_error(test, stat, blend_rate(test, stat, k)),
                    "blend_rookie": rate_error(test, stat, blend_rate(test, stat, k, m)),
                })
            for name, (made, att) in PCT_STATS.items():
                k = min(K_ATT_GRID, key=lambda k: pct_error(train, made, att, blend_pct(train, made, att, k)))
                m = min(ROOKIE_MULT_GRID, key=lambda m: pct_error(train, made, att, blend_pct(train, made, att, k, m)))
                rows.append({
                    "cutoff": cutoff, "season": s, "stat": f"{name}%", "k": k, "rookie_mult": m,
                    "prior": pct_error(test, made, att, blend_pct(test, made, att, 1e9)),
                    "observed": pct_error(test, made, att, blend_pct(test, made, att, 0)),
                    "blend": pct_error(test, made, att, blend_pct(test, made, att, k)),
                    "blend_rookie": pct_error(test, made, att, blend_pct(test, made, att, k, m)),
                })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, "inseason_rates_by_season.csv"), index=False)

    summary = res.groupby(["stat", "cutoff"])[["prior", "observed", "blend", "blend_rookie"]].mean().round(3)
    res["blend_beats_prior"] = res["blend"] < res["prior"]
    res["blend_beats_observed"] = res["blend"] < res["observed"]
    res["rookie_beats_blend"] = res["blend_rookie"] < res["blend"]
    wins = res.groupby(["stat", "cutoff"])[["blend_beats_prior", "blend_beats_observed", "rookie_beats_blend"]].sum()
    k2026 = res[res["season"] == IM.SEASONS[-1]].set_index(["stat", "cutoff"])[["k", "rookie_mult"]]
    table = summary.join(wins).join(k2026)
    print("Per-36 error (count stats) / pct-point error (FG%, FT%), test seasons 2020-26, wins of 7:\n")
    print(table.to_string())

    # Full-season value with each test season's fitted k (k per stat from training seasons).
    value_rows = []
    for (cutoff, s), test in data[data["SEASON"] >= IM.SEASONS[1]].groupby(["CUTOFF", "SEASON"]):
        ks = res[(res["cutoff"] == cutoff) & (res["season"] == s)].set_index("stat")
        prior_rates = {st: test[f"PRIOR_{st}"] for st in ALL_TOTALS}
        blended = {st: blend_rate(test, st, ks.at[st, "k"]) for st in COUNT_STATS}
        blended["FGM"] = blend_pct(test, "FGM", "FGA", ks.at["FG%", "k"]) * blended["FGA"]
        blended["FTM"] = blend_pct(test, "FTM", "FTA", ks.at["FT%", "k"]) * blended["FTA"]
        mpg_blend = minutes_blend(test)
        value_rows.append({
            "cutoff": cutoff, "season": s,
            "preseason_only": full_season_value(test, test["PRIOR_MPG"], prior_rates),
            "minutes_blend": full_season_value(test, mpg_blend, prior_rates),
            "minutes_and_rates": full_season_value(test, mpg_blend, blended),
        })
    val = pd.DataFrame(value_rows)
    val.to_csv(os.path.join(RESULTS, "inseason_value_by_season.csv"), index=False)
    print("\nFull-season value Spearman (actual so far + projected rest vs actual season):")
    print(val.groupby("cutoff")[["preseason_only", "minutes_blend", "minutes_and_rates"]].mean().round(4).to_string())
    val["rates_win"] = val["minutes_and_rates"] > val["minutes_blend"]
    val["minutes_win"] = val["minutes_blend"] > val["preseason_only"]
    print(val.groupby("cutoff")[["minutes_win", "rates_win"]].sum().to_string())

    report = {
        "summary": table.reset_index().to_dict("records"),
        "value": val.groupby("cutoff")[["preseason_only", "minutes_blend", "minutes_and_rates"]].mean().round(4).reset_index().to_dict("records"),
    }
    with open(os.path.join(RESULTS, "inseason_rates.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    _ = itertools


if __name__ == "__main__":
    main()
