"""In-season minutes study: how much should current minutes count?

Read-only. Backtest for the up-to-date ("Current") projections, step 1 (minutes).
Results in analysis/results/inseason_minutes_*.

Setup, per season 2019-2026:
- Prior: the preseason projection from seasons before it (project.project_rosters, as the
  allocation backtest builds it), on opening rosters (each player's first team in the game
  logs). No injuries are passed. Cached in data/processed/backtest_cache/.
- Cutoffs: the first date on which the average team has played N games, N = 10, 20, 41, 60.
  Observed = game logs before the cutoff; truth = game logs on or after it.
- Scored: players with 10+ games after the cutoff. Target = rest-of-season MPG (per game
  played). Error = MAE weighted by rest-of-season games. True games after the cutoff are
  used where a variant needs games (the team-240 hold), as agreed (hindsight on games only).

Variants:
- prior / season / last10: one source only.
- blend(k, r): w = n / (n + k), n = games played before the cutoff;
  pred = (1 - w) * prior + w * ((1 - r) * season MPG + r * last-10 MPG).
- absence correction: observed MPG minus the minutes gained from absent teammates in the
  games he played (opportunity model coefficients), then plus the gain expected from
  teammates' future absence rate (1 - their games / team games so far). No hindsight.
- team hold (power p): per team at the cutoff, scale predicted minutes so expected team
  minutes per game = 241.5 (OT included) x (target / raw) ^ p.
k, r and p are picked on earlier seasons only (expanding window from 2019) and scored on
each later season; each step must beat the simpler version every test season.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import model as M  # noqa: E402
import opportunity as O  # noqa: E402
import project as P  # noqa: E402
import rookies as R  # noqa: E402
from gamelogs import load_gamelogs  # noqa: E402

SEASONS = list(range(2019, 2027))
CUTOFFS = [10, 20, 41, 60]
MIN_ROS_GAMES = 10
TEAM_MINUTES_PER_GAME = 241.5
CACHE_DIR = os.path.join("data", "processed", "backtest_cache")
RESULTS = os.path.join("analysis", "results")
K_GRID = [0, 2, 5, 10, 15, 20, 30, 50, 100]
R_GRID = [0.0, 0.25, 0.5, 0.75]
P_GRID = [0.0, 0.25, 0.5, 1.0]


# --------------------------------------------------------------------------
# Priors
# --------------------------------------------------------------------------
def preseason_prior(year: int, history: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"prior_{year}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    past = history[history["SEASON_YEAR"] < year]
    season = history[history["SEASON_YEAR"] == year].set_index("PLAYER_ID")
    first = logs.sort_values(["GAME_DATE", "GAME_ID"]).drop_duplicates("PLAYER_ID")
    roster = pd.DataFrame(
        {"PLAYER_ID": first["PLAYER_ID"], "TEAM_ABBREVIATION": first["TEAM_ABBREVIATION"],
         "ROSTER_NAME": first["PLAYER_NAME"]}
    )
    ages = season.loc[~season.index.isin(past["PLAYER_ID"]), "AGE"]
    decisions = json.loads(M.MODEL_METADATA_PATH.read_text())["policy_decisions"]
    bundle = M.build_bundle(M._train_slice(M.load_training_pairs(), year), decisions)
    pred = P.project_rosters(
        bundle, roster, past, latest=P.latest_season_rows(past),
        draft=R.load_draft_history(), current_ages=ages,
    )
    os.makedirs(CACHE_DIR, exist_ok=True)
    pred.to_csv(path, index=False)
    return pred


# --------------------------------------------------------------------------
# Cutoff snapshots
# --------------------------------------------------------------------------
def cutoff_date(logs: pd.DataFrame, n_games: int):
    team_games = logs[["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"]].drop_duplicates()
    dates = sorted(team_games["GAME_DATE"].unique())
    teams = team_games["TEAM_ABBREVIATION"].nunique()
    for d in dates:
        if (team_games["GAME_DATE"] < d).sum() / teams >= n_games:
            return d
    return None


def absence_gain(before: pd.DataFrame, prior_mpg: pd.Series, positions: pd.Series, model: dict) -> pd.DataFrame:
    """Per player: average minutes gained per game played from absent teammates (past),
    and the expected gain per game from teammates' absence rate so far (future)."""
    rows = []
    edges, coef = model["band_edges"], model["band_coef"]
    beta, small = model["same_slot_beta"], model.get("small_absence_share", 0.0)
    for team, tg in before.groupby("TEAM_ABBREVIATION"):
        games = tg[["GAME_ID", "GAME_DATE"]].drop_duplicates().sort_values("GAME_DATE")["GAME_ID"].tolist()
        order = {g: i for i, g in enumerate(games)}
        tg = tg.assign(ORDER=tg["GAME_ID"].map(order))
        per = tg.groupby("PLAYER_ID").agg(GP=("MIN", "size"), MPG=("MIN", "mean"),
                                          FIRST=("ORDER", "min"), LAST=("ORDER", "max"))
        weight = np.where(per["MPG"] >= 20, 1.0, np.where(per["MPG"] >= 10, small, 0.0))
        per = per.assign(W=weight)
        players = per.index.to_numpy()
        slots = {p: O.slot_vector(positions.get(p)) for p in players}
        band = pd.cut(prior_mpg.reindex(players).fillna(per["MPG"]), edges, labels=False, include_lowest=True)
        a = pd.Series(band, index=players).map(coef).fillna(0.0)
        played = set(zip(tg["GAME_ID"], tg["PLAYER_ID"]))
        movers = per[per["W"] > 0]
        past_gain = {p: 0.0 for p in players}
        for p in players:
            total = 0.0
            mine = tg.loc[tg["PLAYER_ID"] == p, "GAME_ID"]
            for gid in mine:
                n = order[gid]
                for q, r in movers.iterrows():
                    if q != p and r.FIRST <= n <= r.LAST and (gid, q) not in played:
                        total += r.W * r.MPG * (1 + beta * slots[p] @ slots[q])
            past_gain[p] = a[p] * total / max(len(mine), 1)
        team_n = len(games)
        future = {}
        for p in players:
            total = 0.0
            for q, r in movers.iterrows():
                if q == p:
                    continue
                span = r.LAST - r.FIRST + 1
                rate = 1 - r.GP / span if span > 0 else 0.0
                total += r.W * r.MPG * rate * (1 + beta * slots[p] @ slots[q])
            future[p] = a[p] * total
        for p in players:
            rows.append((p, team, past_gain[p], future[p], team_n))
    return pd.DataFrame(rows, columns=["PLAYER_ID", "TEAM_NOW", "PAST_GAIN", "FUTURE_GAIN", "TEAM_GP"])


def snapshot(year, n_games, logs, prior, positions, model) -> pd.DataFrame | None:
    d = cutoff_date(logs, n_games)
    if d is None:
        return None
    before, after = logs[logs["GAME_DATE"] < d], logs[logs["GAME_DATE"] >= d]
    obs = before.sort_values("GAME_DATE").groupby("PLAYER_ID").agg(
        GP_OBS=("MIN", "size"), MPG_OBS=("MIN", "mean"))
    last10 = before.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(10).groupby("PLAYER_ID")["MIN"].mean()
    obs["MPG_L10"] = last10
    ros = after.groupby("PLAYER_ID").agg(GP_ROS=("MIN", "size"), MPG_ROS=("MIN", "mean"))
    # Team at the cutoff = team of the last game before it (else first game after).
    last_team = before.sort_values("GAME_DATE").groupby("PLAYER_ID")["TEAM_ABBREVIATION"].last()
    first_after = after.sort_values("GAME_DATE").groupby("PLAYER_ID")["TEAM_ABBREVIATION"].first()
    team_games_after = after.groupby("TEAM_ABBREVIATION")["GAME_ID"].nunique()
    pr = prior.set_index("PLAYER_ID")
    df = ros.join(obs, how="left").join(pr[["PROJECTED_MPG", "PROJECTED_GP"]], how="inner")
    df["TEAM_NOW"] = last_team.reindex(df.index).fillna(first_after.reindex(df.index))
    df["TEAM_GAMES_ROS"] = df["TEAM_NOW"].map(team_games_after)
    df[["GP_OBS"]] = df[["GP_OBS"]].fillna(0)
    gains = absence_gain(before, pr["PROJECTED_MPG"], positions, model).set_index("PLAYER_ID")
    df = df.join(gains[["PAST_GAIN", "FUTURE_GAIN"]], how="left")
    df[["PAST_GAIN", "FUTURE_GAIN"]] = df[["PAST_GAIN", "FUTURE_GAIN"]].fillna(0.0)
    df["SEASON"], df["CUTOFF"], df["CUTOFF_DATE"] = year, n_games, str(d)
    return df.reset_index()


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------
def predict(df: pd.DataFrame, k: float, r: float, corrected: bool, power: float = 0.0) -> pd.Series:
    season = df["MPG_OBS"]
    recent = df["MPG_L10"]
    if corrected:
        season = season - df["PAST_GAIN"] + df["FUTURE_GAIN"]
        recent = recent - df["PAST_GAIN"] + df["FUTURE_GAIN"]  # past gain is a season average
    observed = (1 - r) * season + r * recent
    n = df["GP_OBS"]
    w = n / (n + k) if k > 0 else (n > 0).astype(float)
    pred = (1 - w) * df["PROJECTED_MPG"] + w * observed.fillna(df["PROJECTED_MPG"])
    pred = pred.where(n > 0, df["PROJECTED_MPG"])
    if power > 0:
        share = df["GP_ROS"] / df["TEAM_GAMES_ROS"]  # hindsight availability (games only)
        per_game = (pred * share).groupby([df["SEASON"], df["CUTOFF"], df["TEAM_NOW"]]).transform("sum")
        pred = pred * (TEAM_MINUTES_PER_GAME / per_game) ** power
    return pred.clip(0, 48)


def mae(df: pd.DataFrame, pred: pd.Series) -> float:
    return float(np.average((pred - df["MPG_ROS"]).abs(), weights=df["GP_ROS"]))


def per_season(df: pd.DataFrame, fn) -> dict:
    return {s: mae(part, fn(part)) for s, part in df.groupby("SEASON")}


def pick(train: pd.DataFrame, grid, fn):
    return min(grid, key=lambda params: mae(train, fn(train, *params)))


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    history = P.load_history()
    positions = O.load_positions()
    model = O.load_opportunity_model()
    frames = []
    for year in SEASONS:
        logs = load_gamelogs([year])
        prior = preseason_prior(year, history, logs)
        for n in CUTOFFS:
            snap = snapshot(year, n, logs, prior, positions, model)
            if snap is not None:
                frames.append(snap)
        print(f"{year}: prior {len(prior)} players, snapshots done", flush=True)
    data = pd.concat(frames, ignore_index=True)
    data = data[data["GP_ROS"] >= MIN_ROS_GAMES].copy()
    data.to_csv(os.path.join(CACHE_DIR, "inseason_minutes_snapshots.csv"), index=False)

    report = {"n_rows": int(len(data))}
    variants = {
        "prior": lambda d: d["PROJECTED_MPG"],
        "season": lambda d: d["MPG_OBS"].fillna(d["PROJECTED_MPG"]),
        "last10": lambda d: d["MPG_L10"].fillna(d["PROJECTED_MPG"]),
    }
    rows = []
    for cutoff in CUTOFFS:
        part = data[data["CUTOFF"] == cutoff]
        for name, fn in variants.items():
            for s, v in per_season(part, fn).items():
                rows.append({"cutoff": cutoff, "variant": name, "season": s, "mae": v})
        # Expanding-window fits: each test season uses parameters picked on earlier seasons.
        for s in SEASONS[1:]:
            train, test = part[part["SEASON"] < s], part[part["SEASON"] == s]
            fits = {
                "blend_season": ((pick(train, [(k,) for k in K_GRID], lambda d, k: predict(d, k, 0.0, False))[0], 0.0, False, 0.0)),
                "blend_recent": (*pick(train, list(itertools.product(K_GRID, R_GRID)), lambda d, k, r: predict(d, k, r, False)), False, 0.0),
                "blend_recent_corrected": (*pick(train, list(itertools.product(K_GRID, R_GRID)), lambda d, k, r: predict(d, k, r, True)), True, 0.0),
            }
            k, r, _, _ = fits["blend_recent_corrected"]
            pw = pick(train, [(p,) for p in P_GRID], lambda d, p: predict(d, k, r, True, p))[0]
            fits["blend_recent_corrected_hold"] = (k, r, True, pw)
            for name, (k, r, corr, pw) in fits.items():
                rows.append({"cutoff": cutoff, "variant": name, "season": s, "mae": mae(test, predict(test, k, r, corr, pw)),
                             "k": k, "r": r, "power": pw})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, "inseason_minutes_by_season.csv"), index=False)

    test = res[res["season"] >= SEASONS[1]]
    summary = test.pivot_table(index=["cutoff"], columns="variant", values="mae", aggfunc="mean").round(3)
    print("\nMean rest-of-season MPG error (GP-weighted), test seasons 2020-26:\n", summary.to_string())
    order = ["prior", "season", "blend_season", "blend_recent", "blend_recent_corrected", "blend_recent_corrected_hold"]
    wins = {}
    for cutoff, part in test.groupby("cutoff"):
        piv = part.pivot_table(index="season", columns="variant", values="mae")
        wins[int(cutoff)] = {
            "blend_season vs prior": int((piv["blend_season"] < piv["prior"]).sum()),
            "blend_season vs season": int((piv["blend_season"] < piv["season"]).sum()),
            "recent vs blend_season": int((piv["blend_recent"] < piv["blend_season"]).sum()),
            "corrected vs recent": int((piv["blend_recent_corrected"] < piv["blend_recent"]).sum()),
            "hold vs corrected": int((piv["blend_recent_corrected_hold"] < piv["blend_recent_corrected"]).sum()),
            "of": int(len(piv)),
        }
    print("\nSeasons won (of 7):\n", pd.DataFrame(wins).T.to_string())
    last = res[(res["season"] == SEASONS[-1]) & res["k"].notna()][["cutoff", "variant", "k", "r", "power"]]
    print("\nParameters picked for 2026 (trained on 2019-25):\n", last.to_string(index=False))
    report["summary"] = summary.reset_index().to_dict("records")
    report["wins"] = wins
    report["params_2026"] = last.to_dict("records")
    # Groups, at cutoff 20, with 2026 parameters: big prior misses.
    with open(os.path.join(RESULTS, "inseason_minutes.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    _ = order


if __name__ == "__main__":
    main()
