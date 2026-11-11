"""How do teammates' minutes and per-minute stats change when a regular sits?

Read-only analysis on data/raw/gamelogs (2014-26). Writes analysis/results/injury_opportunity_*.

Definitions (per team-season, using only that team's games):
- Regular: averaged 20+ MPG over 20+ games for the team.
- Absent: a regular who did not appear in one of the team's games between his first and
  last game for the team (so trades and late signings don't count as absences).
- Freed minutes F_g: sum of absent regulars' average MPG in game g.
- Baseline b_i: player i's average minutes in the team's games with no regular absent
  (needs 5+ such games). Gain y = minutes in game g - b_i.

Share model (to be used in projection): gain_i = k * F_g * w_i / sum(w over teammates who
played), w_i = b_i ^ gamma * (1 + beta * same_slot_i), where same_slot_i is the overlap of
guard/forward/center slots with the absent players (G/F/C labels, hybrids split 2/3, 1/3).
Fitted on 2014-2022, scored on each of 2023-2026 (MAE of per-game minutes, games with F > 0).
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from gamelogs import load_gamelogs  # noqa: E402

RESULTS = os.path.join("analysis", "results")
REGULAR_MPG = 20.0
REGULAR_GP = 20
RATE_STATS = ["PTS", "FGA", "FG3A", "FTA", "AST", "REB", "TOV", "STL", "BLK"]
SLOTS = {
    "G": (1, 0, 0), "F": (0, 1, 0), "C": (0, 0, 1),
    "G-F": (2 / 3, 1 / 3, 0), "F-G": (1 / 3, 2 / 3, 0),
    "F-C": (0, 2 / 3, 1 / 3), "C-F": (0, 1 / 3, 2 / 3),
}


def slot_vector(position) -> np.ndarray:
    return np.array(SLOTS.get(str(position), (1 / 3, 1 / 3, 1 / 3)))


def build_games(logs: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """One row per (team-game, player who played) with gain, freed minutes and slots."""
    rows = []
    for (season, team), tg in logs.groupby(["SEASON_END", "TEAM_ABBREVIATION"]):
        games = tg[["GAME_ID", "GAME_DATE"]].drop_duplicates().sort_values("GAME_DATE")
        game_order = {gid: n for n, gid in enumerate(games["GAME_ID"])}
        tg = tg.assign(ORDER=tg["GAME_ID"].map(game_order))
        per_player = tg.groupby("PLAYER_ID").agg(GP=("MIN", "size"), MPG=("MIN", "mean"),
                                                 FIRST=("ORDER", "min"), LAST=("ORDER", "max"))
        regulars = per_player[(per_player["MPG"] >= REGULAR_MPG) & (per_player["GP"] >= REGULAR_GP)]
        played = set(zip(tg["GAME_ID"], tg["PLAYER_ID"]))
        absent_rows = []
        for pid, r in regulars.iterrows():
            for gid, n in game_order.items():
                if r.FIRST <= n <= r.LAST and (gid, pid) not in played:
                    absent_rows.append((gid, pid, r.MPG))
        absent = pd.DataFrame(absent_rows, columns=["GAME_ID", "ABSENT_ID", "ABSENT_MPG"])
        freed = absent.groupby("GAME_ID")["ABSENT_MPG"].sum() if len(absent) else pd.Series(dtype=float)
        tg = tg.assign(FREED=tg["GAME_ID"].map(freed).fillna(0.0))
        clean = tg[tg["FREED"] == 0]
        base = clean.groupby("PLAYER_ID").agg(BASE_N=("MIN", "size"), BASE_MPG=("MIN", "mean"),
                                              **{f"BASE_{s}": (s, "sum") for s in RATE_STATS},
                                              BASE_MIN=("MIN", "sum"))
        tg = tg.join(base, on="PLAYER_ID")
        # Absent players' slot mix, MPG-weighted, per game.
        if len(absent):
            absent = absent.assign(SLOT=absent["ABSENT_ID"].map(lambda p: slot_vector(positions.get(p))))
            absent_slot = absent.groupby("GAME_ID").apply(
                lambda a: np.average(np.vstack(a["SLOT"]), axis=0, weights=a["ABSENT_MPG"]),
                include_groups=False,
            )
        else:
            absent_slot = pd.Series(dtype=object)
        tg["SAME_SLOT"] = [
            float(slot_vector(positions.get(pid)) @ absent_slot[gid]) if gid in absent_slot.index else 0.0
            for pid, gid in zip(tg["PLAYER_ID"], tg["GAME_ID"])
        ]
        tg["SEASON_END"] = season
        rows.append(tg)
    df = pd.concat(rows, ignore_index=True)
    return df[df["BASE_N"] >= 5].copy()


def predict_gain(df: pd.DataFrame, gamma: float, beta: float) -> np.ndarray:
    """Share of freed minutes per player (before k), normalized within each team-game."""
    w = df["BASE_MPG"].clip(lower=1.0) ** gamma * (1 + beta * df["SAME_SLOT"])
    total = w.groupby([df["GAME_ID"], df["TEAM_ABBREVIATION"]]).transform("sum")
    return (df["FREED"] * w / total).to_numpy()


def fit_k(x: np.ndarray, y: np.ndarray) -> float:
    return float((x * y).sum() / (x * x).sum())


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    positions = pd.read_csv("data/raw/player_positions.csv").set_index("PLAYER_ID")["POSITION"]
    logs = load_gamelogs(range(2014, 2027))
    df = build_games(logs, positions)
    df["GAIN"] = df["MIN"] - df["BASE_MPG"]
    out = df[df["FREED"] > 0]
    report = {}

    # 1. How much of the freed time do teammates who played get back (team-game sums)?
    team_game = out.groupby(["SEASON_END", "GAME_ID", "TEAM_ABBREVIATION"]).agg(
        FREED=("FREED", "first"), GAIN=("GAIN", "sum"))
    recovered = float(team_game["GAIN"].sum() / team_game["FREED"].sum())
    report["share_of_freed_minutes_recovered"] = round(recovered, 3)
    by_season = team_game.groupby("SEASON_END").apply(
        lambda t: round(t["GAIN"].sum() / t["FREED"].sum(), 3), include_groups=False)
    report["recovered_by_season"] = by_season.to_dict()
    print(f"Teammates who played recover {recovered:.1%} of freed minutes")

    # 2. Share model grid: fit on 2014-2022, score each later season.
    train = out[out["SEASON_END"] <= 2022]
    grid = []
    for gamma, beta in itertools.product([0, 0.5, 1, 1.5, 2, 3], [0, 0.5, 1, 2]):
        x = predict_gain(train, gamma, beta)
        k = fit_k(x, train["GAIN"].to_numpy())
        row = {"gamma": gamma, "beta": beta, "k": round(k, 3)}
        for season in range(2023, 2027):
            test = out[out["SEASON_END"] == season]
            pred = k * predict_gain(test, gamma, beta)
            row[f"mae_{season}"] = round(float(np.abs(test["GAIN"] - pred).mean()), 4)
        row["mae_none"] = {s: round(float(out[out.SEASON_END == s]["GAIN"].abs().mean()), 4) for s in range(2023, 2027)}
        grid.append(row)
    grid_df = pd.DataFrame(grid)
    grid_df["mae_mean"] = grid_df[[f"mae_{s}" for s in range(2023, 2027)]].mean(axis=1)
    grid_df = grid_df.sort_values("mae_mean")
    grid_df.drop(columns="mae_none").to_csv(os.path.join(RESULTS, "injury_opportunity_grid.csv"), index=False)
    report["no_boost_mae"] = grid[0]["mae_none"]
    report["best"] = grid_df.drop(columns="mae_none").iloc[0].to_dict()
    flat = grid_df[(grid_df.gamma == 0) & (grid_df.beta == 0)].drop(columns="mae_none").iloc[0].to_dict()
    report["even_split"] = flat
    print(grid_df.drop(columns="mae_none").head(8).to_string(index=False))
    print("no boost:", report["no_boost_mae"])

    # 2b. Band model (no normalization): gain_i = F * a(band of b_i) * (1 + beta * same_slot).
    #     Expanding window: each test season fit on all earlier seasons (2014 on).
    band_edges = [0, 10, 20, 28, 34, 60]
    out = out.assign(BAND=pd.cut(out["BASE_MPG"], band_edges, labels=False))
    band_rows = []
    for beta in [0, 0.5, 1, 2]:
        row = {"beta": beta}
        for season in range(2018, 2027):
            tr = out[out["SEASON_END"] < season]
            te = out[out["SEASON_END"] == season]
            coef = {}
            for b, part in tr.groupby("BAND"):
                x = part["FREED"] * (1 + beta * part["SAME_SLOT"])
                coef[b] = fit_k(x.to_numpy(), part["GAIN"].to_numpy())
            pred = te["FREED"] * (1 + beta * te["SAME_SLOT"]) * te["BAND"].map(coef)
            row[f"mae_{season}"] = round(float(np.abs(te["GAIN"] - pred).mean()), 4)
            even_k = fit_k(predict_gain(tr, 0, 1), tr["GAIN"].to_numpy())
            row[f"even_{season}"] = round(float(np.abs(te["GAIN"] - even_k * predict_gain(te, 0, 1)).mean()), 4)
            row[f"none_{season}"] = round(float(te["GAIN"].abs().mean()), 4)
        x_all = out["FREED"] * (1 + beta * out["SAME_SLOT"])
        row["coef_all"] = {int(b): round(fit_k((x_all[out.BAND == b]).to_numpy(), out.loc[out.BAND == b, "GAIN"].to_numpy()), 4)
                           for b in sorted(out["BAND"].dropna().unique())}
        band_rows.append(row)
    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(os.path.join(RESULTS, "injury_opportunity_band_model.csv"), index=False)
    report["band_model"] = band_rows
    for r in band_rows:
        wins = sum(r[f"mae_{s}"] < r[f"even_{s}"] for s in range(2018, 2027))
        mean_b = np.mean([r[f"mae_{s}"] for s in range(2018, 2027)])
        mean_e = np.mean([r[f"even_{s}"] for s in range(2018, 2027)])
        mean_n = np.mean([r[f"none_{s}"] for s in range(2018, 2027)])
        print(f"band beta={r['beta']}: MAE {mean_b:.4f} vs even {mean_e:.4f} vs none {mean_n:.4f}; "
              f"beats even in {wins}/9; coef {r['coef_all']}")

    # 3. Gain by teammate's baseline MPG band (does the bench or the rotation absorb it?).
    best = report["best"]
    out = out.assign(PRED=best["k"] * predict_gain(out, best["gamma"], best["beta"]))
    bands = pd.cut(out["BASE_MPG"], [0, 10, 20, 28, 34, 48])
    band = out.groupby(bands, observed=True).agg(n=("GAIN", "size"), gain=("GAIN", "mean"), pred=("PRED", "mean"),
                                                 per_freed=("GAIN", lambda g: g.sum() / out.loc[g.index, "FREED"].sum()))
    print(band.round(3))
    report["by_baseline_band"] = band.round(3).reset_index().astype(str).to_dict("records")

    # 4. Per-minute stats: teammates' rate in absence games vs clean games, by freed minutes.
    #    Pooled ratio per band of F (sum of stat / sum of minutes), relative to the same
    #    players' clean-game rate.
    rates = {}
    for low, high in [(0.1, 25), (25, 45), (45, 200)]:
        part = out[(out["FREED"] >= low) & (out["FREED"] < high) & (out["BASE_MIN"] >= 200)]
        row = {"n": int(len(part))}
        for stat in RATE_STATS:
            observed = part[stat].sum()
            expected = (part["BASE_" + stat] / part["BASE_MIN"] * part["MIN"]).sum()
            row[stat] = round(float(observed / expected - 1), 4)
        rates[f"freed_{low}-{high}"] = row
    report["rate_change_vs_clean_games"] = rates
    print(pd.DataFrame(rates).T)

    # 5. Rate change per freed minute, by season (stability check for a linear term).
    per_season = {}
    for season, part in out[out["BASE_MIN"] >= 200].groupby("SEASON_END"):
        row = {}
        for stat in ["PTS", "FGA", "FTA", "AST", "TOV", "REB"]:
            expected = part["BASE_" + stat] / part["BASE_MIN"] * part["MIN"]
            # minutes-weighted average freed minutes
            f = np.average(part["FREED"], weights=expected.clip(lower=1e-6))
            row[stat] = round(float((part[stat].sum() / expected.sum() - 1) / f * 10), 4)
        per_season[int(season)] = row
    report["rate_change_per_10_freed_by_season"] = per_season
    print("rate change per 10 freed minutes:\n", pd.DataFrame(per_season).T)

    # 6. Rate change per unit of freed shot volume (absent players' FGA + 0.44 FTA + TOV
    #    per game), fitted on earlier seasons, sign check per season.
    vol = {}
    for (season, team), tg in logs.groupby(["SEASON_END", "TEAM_ABBREVIATION"]):
        pg = tg.groupby("PLAYER_ID").agg(GP=("MIN", "size"), MPG=("MIN", "mean"),
                                         V=("FGA", "mean"), FTA=("FTA", "mean"), TOV=("TOV", "mean"))
        pg["VOL"] = pg["V"] + 0.44 * pg["FTA"] + pg["TOV"]
        for pid, r in pg[(pg.MPG >= REGULAR_MPG) & (pg.GP >= REGULAR_GP)].iterrows():
            vol[(season, team, pid)] = r.VOL
    # Freed volume per team-game: rebuild from absences (same rule as build_games).
    freed_vol = {}
    for (season, team), tg in logs.groupby(["SEASON_END", "TEAM_ABBREVIATION"]):
        games = tg[["GAME_ID", "GAME_DATE"]].drop_duplicates().sort_values("GAME_DATE")["GAME_ID"].tolist()
        order = {g: n for n, g in enumerate(games)}
        span = tg.assign(O=tg["GAME_ID"].map(order)).groupby("PLAYER_ID")["O"].agg(["min", "max"])
        played = set(zip(tg["GAME_ID"], tg["PLAYER_ID"]))
        for (s2, t2, pid), v in ((k, v) for k, v in vol.items() if k[0] == season and k[1] == team):
            lo, hi = span.loc[pid]
            for g in games[lo:hi + 1]:
                if (g, pid) not in played:
                    freed_vol[(g, team)] = freed_vol.get((g, team), 0.0) + v
    out = out.assign(FREED_VOL=[freed_vol.get((g, t), 0.0) for g, t in zip(out["GAME_ID"], out["TEAM_ABBREVIATION"])])
    part_all = out[out["BASE_MIN"] >= 200]
    vol_rows = {}
    for season, part in part_all.groupby("SEASON_END"):
        row = {}
        for stat in ["PTS", "FGA", "FG3A", "FTA", "AST", "TOV", "REB", "STL", "BLK"]:
            expected = part["BASE_" + stat] / part["BASE_MIN"] * part["MIN"]
            f = np.average(part["FREED_VOL"], weights=expected.clip(lower=1e-6))
            row[stat] = round(float((part[stat].sum() / expected.sum() - 1) / f * 10), 4)
        vol_rows[int(season)] = row
    vol_df = pd.DataFrame(vol_rows).T
    print("rate change per 10 freed shot volume:\n", vol_df)
    print("positive seasons:", (vol_df > 0).sum().to_dict())
    print("mean:", vol_df.mean().round(4).to_dict())
    report["rate_change_per_10_freed_volume_by_season"] = vol_rows

    with open(os.path.join(RESULTS, "injury_opportunity.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
