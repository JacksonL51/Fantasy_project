"""Usage tests A-C (read-only): python analysis/usage_tests.py [A] [B] [C]

Each target season 2014-2026 is predicted from earlier seasons only and scored on
returning players (both seasons 200+ minutes, the training_pairs rule).
A = minutes, B = player rate models, C = usage leaving a team.
Results go to analysis/results/.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import model as M
import project as P
from history import (
    age_mpg_adjustment, fit_age_mpg_curve, fit_minutes_shrink, role_feature_table,
    season_usage, shrink_baseline_mpg, usage_feature_table,
)

YEARS = list(range(2014, 2027))
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261108)
USAGE_FEATURES = ["USG_S", "USG_HIST", "AST_PCT_S", "AST_PCT_HIST"]


def ridge(alpha):
    return Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=alpha))])


def boot_ci(diff, weight, n=2000):
    idx = RNG.integers(0, len(diff), size=(n, len(diff)))
    stats = (diff[idx] * weight[idx]).sum(1) / weight[idx].sum(1)
    return np.percentile(stats, [2.5, 97.5])


def load():
    hist = P.load_history()
    feats = usage_feature_table(hist)
    pairs = M.load_training_pairs()
    pairs = pairs.merge(
        feats.rename(columns={"SEASON_YEAR": "YEAR_CURRENT"}), on=["PLAYER_ID", "YEAR_CURRENT"], how="left"
    )
    return hist, feats, pairs


def rate_test(pairs, stats, extra, label, subgroup=None):
    """Selected policy per stat (as in model.py) with and without `extra` features."""
    decisions = json.loads(M.MODEL_METADATA_PATH.read_text())["policy_decisions"]
    preds = []
    for stat in stats:
        d = decisions[stat]
        form, alpha, fs = d["selected_policy"], float(d["tuned_alpha"]), d["tuned_feature_set"]
        base_cols = M.features_for(stat, fs)
        tgt = f"{stat}_RATE_target"
        for year in YEARS:
            train = pairs.loc[pairs["YEAR_NEXT"] < year]
            test = pairs.loc[(pairs["YEAR_NEXT"] == year) & (pairs["YEAR_CURRENT"] == year - 1)]
            offset_tr = train[f"{stat}_RATE_prior"] if form == "delta" else 0.0
            offset_te = test[f"{stat}_RATE_prior"].to_numpy() if form == "delta" else 0.0
            w = (train["MIN_target"] / train["MIN_target"].mean()).to_numpy()
            out = {}
            for name, cols in (("base", base_cols), ("usage", base_cols + extra)):
                m = ridge(alpha).fit(train[cols], train[tgt] - offset_tr, r__sample_weight=w)
                out[name] = m.predict(test[cols]) + offset_te
            frame = test[["PLAYER_ID", "PLAYER_NAME", "YEAR_NEXT", "MIN_target", tgt]].copy()
            frame = frame.rename(columns={tgt: "actual"})
            frame["stat"], frame["pred_base"], frame["pred_usage"] = stat, out["base"], out["usage"]
            if subgroup is not None:
                frame["sub"] = subgroup.loc[test.index].to_numpy()
            preds.append(frame)
    preds = pd.concat(preds, ignore_index=True)
    preds.to_csv(OUT / f"test_{label}_predictions.csv", index=False)
    return summarize_rates(preds, label), preds


def summarize_rates(preds, label, sub_only=False):
    lines = []
    for stat, g in preds.groupby("stat", sort=False):
        if sub_only:
            g = g.loc[g["sub"].astype(bool)]
        e_b = (g["pred_base"] - g["actual"]).abs() * 36
        e_u = (g["pred_usage"] - g["actual"]).abs() * 36
        w = g["MIN_target"]
        by_year = pd.DataFrame({"b": e_b * w, "u": e_u * w, "w": w, "y": g["YEAR_NEXT"]}).groupby("y").sum()
        mae_b, mae_u = by_year["b"] / by_year["w"], by_year["u"] / by_year["w"]
        wins = int((mae_u < mae_b).sum())
        lo, hi = boot_ci((e_b - e_u).to_numpy(), w.to_numpy())
        lines.append({
            "test": label, "stat": stat, "n": len(g),
            "MAE36_base": np.average(e_b, weights=w), "MAE36_usage": np.average(e_u, weights=w),
            "wins": f"{wins}/{len(by_year)}", "gain_CI_lo": lo, "gain_CI_hi": hi,
            "bias36_base": np.average((g["pred_base"] - g["actual"]) * 36, weights=w),
            "bias36_usage": np.average((g["pred_usage"] - g["actual"]) * 36, weights=w),
            "passes_rule": bool(wins == len(by_year) and lo > 0),
        })
    return pd.DataFrame(lines)


# ---------------------------------------------------------------- Test B
def test_b(pairs):
    stats = ["PTS", "FGA", "FTA", "FG3M", "TOV", "AST", "FGM", "FTM"]
    res, _ = rate_test(pairs, stats, USAGE_FEATURES, "B")
    return res


# ---------------------------------------------------------------- Test C
def team_usage_changes(hist):
    """Per (team, target year Y): share of the team's Y-1 possessions used by players gone
    in Y (VACATED) and the Y-1 possessions of arrivals (ARRIVED), on end-of-season teams.
    _REL columns subtract that season's average team."""
    u = season_usage(hist)
    rows = []
    for year in range(1998, int(hist["SEASON_YEAR"].max()) + 1):
        prev = u.loc[u["SEASON_YEAR"] == year - 1, ["PLAYER_ID", "TEAM_ABBREVIATION", "POSS"]]
        cur = u.loc[u["SEASON_YEAR"] == year, ["PLAYER_ID", "TEAM_ABBREVIATION"]]
        cur_team = cur.set_index("PLAYER_ID")["TEAM_ABBREVIATION"]
        prev = prev.assign(NOW=prev["PLAYER_ID"].map(cur_team))
        team_poss = prev.groupby("TEAM_ABBREVIATION")["POSS"].sum()
        moved = prev["NOW"] != prev["TEAM_ABBREVIATION"]
        vac = prev.loc[moved].groupby("TEAM_ABBREVIATION")["POSS"].sum()
        arr = prev.loc[prev["NOW"].notna() & moved].groupby("NOW")["POSS"].sum()
        for team in cur["TEAM_ABBREVIATION"].unique():
            if team not in team_poss.index:
                continue
            rows.append({"YEAR": year, "TEAM": team, "VACATED": vac.get(team, 0.0) / team_poss[team],
                         "ARRIVED": arr.get(team, 0.0) / team_poss[team]})
    t = pd.DataFrame(rows)
    for c in ("VACATED", "ARRIVED"):
        t[f"{c}_REL"] = t[c] - t.groupby("YEAR")[c].transform("mean")
    return t


def test_c(hist, pairs):
    changes = team_usage_changes(hist)
    team_now = hist[["PLAYER_ID", "SEASON_YEAR", "TEAM_ABBREVIATION"]].rename(
        columns={"SEASON_YEAR": "YEAR_NEXT", "TEAM_ABBREVIATION": "TEAM_NEXT"})
    team_prev = hist[["PLAYER_ID", "SEASON_YEAR", "TEAM_ABBREVIATION"]].rename(
        columns={"SEASON_YEAR": "YEAR_CURRENT", "TEAM_ABBREVIATION": "TEAM_PREV"})
    p = pairs.merge(team_now, on=["PLAYER_ID", "YEAR_NEXT"], how="left").merge(
        team_prev, on=["PLAYER_ID", "YEAR_CURRENT"], how="left")
    p = p.merge(changes.rename(columns={"YEAR": "YEAR_NEXT", "TEAM": "TEAM_NEXT"}),
                on=["YEAR_NEXT", "TEAM_NEXT"], how="left")
    p = p.loc[p["VACATED_REL"].notna()].reset_index(drop=True)
    p["USG_DEV"] = p["USG_S"] - 0.2
    p["VAC_X_USG"] = p["VACATED_REL"] * p["USG_DEV"]
    p["ARR_X_USG"] = p["ARRIVED_REL"] * p["USG_DEV"]
    stayed = p["TEAM_NEXT"] == p["TEAM_PREV"]
    big_loss = p["VACATED"] >= 0.25
    recent = changes.loc[changes["YEAR"] >= 2014]
    print(f"Pairs: {len(p)}; stayed on team {stayed.mean():.0%}; "
          f"stayers on teams that lost 25%+ of usage {(big_loss & stayed).mean():.0%}")
    print("Team vacated share 2014+:", recent["VACATED"].describe().round(3).to_dict())
    print("Team arrived share 2014+:", recent["ARRIVED"].describe().round(3).to_dict())
    stats = ["PTS", "FGA", "FTA", "AST", "FG3M", "TOV"]
    feats = ["VACATED_REL", "ARRIVED_REL", "VAC_X_USG", "ARR_X_USG"]
    allres, preds = rate_test(p, stats, feats, "C", subgroup=big_loss & stayed)
    sub = summarize_rates(preds, "C stayers, team lost 25%+", sub_only=True)
    return pd.concat([allres, sub]), changes, p


# ---------------------------------------------------------------- Test A
def test_a(hist, feats):
    role = role_feature_table(hist)  # BASE_MPG (60/27/13 over GP>=15), AGE, PROD_Z as of t
    h = hist[["PLAYER_ID", "SEASON_YEAR", "GP", "MIN"]].sort_values(["PLAYER_ID", "SEASON_YEAR"]).copy()
    h["CAREER_MIN"] = h.groupby("PLAYER_ID")["MIN"].cumsum()
    t = role.merge(h, on=["PLAYER_ID", "SEASON_YEAR"]).merge(feats, on=["PLAYER_ID", "SEASON_YEAR"])
    nxt = hist[["PLAYER_ID", "SEASON_YEAR", "GP", "MIN"]].rename(columns={"GP": "GP_N", "MIN": "MIN_N"})
    nxt["SEASON_YEAR"] -= 1
    d = t.merge(nxt, on=["PLAYER_ID", "SEASON_YEAR"])
    d = d.loc[(d["MIN"] >= 200) & (d["MIN_N"] >= 200) & (d["GP"] >= 15) & d["BASE_MPG"].notna()].copy()
    d["MPG_N"] = d["MIN_N"] / d["GP_N"]
    d["MPG_LAST"] = d["MIN"] / d["GP"]
    d["TARGET_YEAR"] = d["SEASON_YEAR"] + 1
    # Pipeline-style start: career shrink then age curve, both fit on seasons before the target year.
    d["START"] = np.nan
    for year in sorted(d["TARGET_YEAR"].unique()):
        past = hist.loc[hist["SEASON_YEAR"] < year]
        rows = d["TARGET_YEAR"] == year
        curve, shrink = fit_age_mpg_curve(past), fit_minutes_shrink(past)
        mpg = shrink_baseline_mpg(shrink, d.loc[rows, "BASE_MPG"], d.loc[rows, "CAREER_MIN"])
        d.loc[rows, "START"] = (mpg + age_mpg_adjustment(curve, d.loc[rows, "AGE"], mpg)).clip(4, 37)
    d["RESID"] = d["MPG_N"] - d["START"]
    base_feats = ["BASE_MPG", "AGE", "PROD_Z"]
    use_feats = base_feats + USAGE_FEATURES
    out = []
    for year in YEARS:
        train, test = d.loc[d["TARGET_YEAR"] < year], d.loc[d["TARGET_YEAR"] == year].copy()
        w = (train["GP_N"] / train["GP_N"].mean()).to_numpy()
        test["P_START"] = test["START"]
        for name, cols in (("P_ROLE", base_feats), ("P_USAGE", use_feats)):
            m = ridge(10.0).fit(train[cols], train["RESID"], r__sample_weight=w)
            test[name] = test["START"] + m.predict(test[cols])
        out.append(test)
    r = pd.concat(out)
    r.to_csv(OUT / "test_A_predictions.csv", index=False)
    groups = {
        "all": pd.Series(True, index=r.index), "bench <20": r["BASE_MPG"] < 20,
        "rotation 20-28": r["BASE_MPG"].between(20, 28, inclusive="left"), "starters 28+": r["BASE_MPG"] >= 28,
    }
    lines = []
    for g, mask in groups.items():
        s = r.loc[mask]
        err = {k: (s[k] - s["MPG_N"]).abs() for k in ("P_START", "P_ROLE", "P_USAGE")}
        yr = pd.DataFrame(err).groupby(s["TARGET_YEAR"]).mean()
        lines.append({
            "group": g, "n": len(s), "MAE_pipeline_start": err["P_START"].mean(),
            "MAE_plus_production": err["P_ROLE"].mean(), "MAE_plus_usage": err["P_USAGE"].mean(),
            "usage_better_years": f"{int((yr['P_USAGE'] < yr['P_ROLE']).sum())}/{len(yr)}",
            "bias_production": (s["P_ROLE"] - s["MPG_N"]).mean(), "bias_usage": (s["P_USAGE"] - s["MPG_N"]).mean(),
        })
    r["ACT_CHG"] = r["MPG_N"] - r["MPG_LAST"]
    jumps = []
    for year, s in r.groupby("TARGET_YEAR"):
        row = {"year": year}
        for k, short in (("P_ROLE", "prod"), ("P_USAGE", "usage")):
            top = (s[k] - s["MPG_LAST"]).rank(ascending=False) <= 20
            row[f"{short}_top20_lift"] = s.loc[top, "ACT_CHG"].mean() - s.loc[~top, "ACT_CHG"].mean()
            row[f"{short}_jumps6_in_top20"] = int((s.loc[top, "ACT_CHG"] >= 6).sum())
        jumps.append(row)
    return pd.DataFrame(lines), pd.DataFrame(jumps)


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C"]
    pd.set_option("display.width", 220)
    hist, feats, pairs = load()
    if "B" in which:
        res = test_b(pairs)
        print("\n=== Test B: own usage in the rate models (per-36 MAE, minute-weighted) ===")
        print(res.round(3).to_string(index=False))
        res.to_csv(OUT / "test_B_summary.csv", index=False)
    if "C" in which:
        res, changes, p = test_c(hist, pairs)
        print("\n=== Test C: usage leaving a team ===")
        print(res.round(3).to_string(index=False))
        res.to_csv(OUT / "test_C_summary.csv", index=False)
        changes.to_csv(OUT / "team_usage_changes.csv", index=False)
    if "A" in which:
        res, jumps = test_a(hist, feats)
        print("\n=== Test A: minutes (MPG MAE) ===")
        print(res.round(3).to_string(index=False))
        print(jumps.round(2).to_string(index=False))
        print("Mean over years:", jumps.drop(columns="year").mean().round(2).to_dict())
        res.to_csv(OUT / "test_A_summary.csv", index=False)
