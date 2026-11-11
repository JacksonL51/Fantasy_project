"""Usage limit refinements (read-only): python analysis/usage_test_d2.py

Allocation-backtest setup (2024-26, end-of-season rosters, data through the season before),
usage limit off inside the pipeline; rates are captured and each variant applied here.
Variants:
  current      veterans, power 0.5 on FGA/FGM/FG3M/FTA/FTM/PTS/TOV (what is on today)
  tov_P        turnovers get their own power P (shots stay 0.5)
  ast_same     assists also scaled by the shot factor
  ast_budget_P assists scaled by team AST per team FGM relative to the average team, power P
  share_K      a team's change shared by projected usage: player change x (usage / team
               average) ** K, rescaled so the team's total change is unchanged.
               K > 0: high-usage players absorb more; K < 0: role players absorb more.
  redistrib_tov  usage-left/arrived fit (USAGE_REDISTRIBUTION) applied to TOV only
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

import backtest as B
import model as M
import project as P
import rookies as R

SHOTS = ["FGA", "FGM", "FG3M", "FTA", "FTM", "PTS"]
PER36 = ["PTS", "FGA", "FTA", "FG3M", "TOV", "AST"]
OUT = Path(__file__).resolve().parent / "results"

captured = {}
_original = P.synthesize_stats


def _capture(*args):
    captured["args"] = args
    return _original(*args)


P.synthesize_stats = _capture


def team_rel(team, num, den):
    v = num.groupby(team).sum() / den.groupby(team).sum()
    return v / v.mean()


def apply(variant, table, rates, minutes, usage_fit, changes, weight):
    team = table["TEAM_ABBREVIATION"]
    vet = ~table["NO_PRIOR_DATA"].astype(bool)
    poss_rate = rates["FGA"] + 0.44 * rates["FTA"] + rates["TOV"]
    volume = team_rel(team, poss_rate * minutes, minutes)
    base = team.map(1.0 / volume).astype(float)  # full-strength factor, power applied below
    out = rates.copy()

    def scaled(power, which=None):
        s = base ** power
        return s.where(vet, 1.0)

    shot_power, tov_power = 0.5, 0.5
    if variant.startswith("tov_"):
        tov_power = float(variant.split("_")[1])
    if variant.startswith("redistrib_tov_"):
        tov_power = float(variant.split("_")[2])
    if variant.startswith("share_"):
        k = float(variant.split("_")[1])
        s = scaled(0.5)
        rel = (poss_rate / (poss_rate * minutes).groupby(team).transform("sum")
               * minutes.groupby(team).transform("sum")).clip(lower=0.2)
        change = (s - 1.0) * poss_rate * minutes  # team's change in possessions, player by player
        w = (rel ** k).where(vet, 0.0) * poss_rate * minutes
        team_change = change.groupby(team).transform("sum")
        new_change = team_change * w / w.groupby(team).transform("sum")
        s2 = 1.0 + (new_change / (poss_rate * minutes)).where(poss_rate * minutes > 0, 0.0)
        s2 = s2.where(vet, 1.0).clip(0.8, 1.2)
        out[SHOTS + ["TOV"]] = rates[SHOTS + ["TOV"]].mul(s2, axis=0)
        return out
    if variant == "none":
        return out
    out[SHOTS] = rates[SHOTS].mul(scaled(shot_power), axis=0)
    out["TOV"] = rates["TOV"] * scaled(tov_power)
    if variant == "ast_same":
        out["AST"] = rates["AST"] * scaled(shot_power)
    if variant.startswith("ast_budget_"):
        power = float(variant.split("_")[2])
        ratio = team_rel(team, rates["AST"] * minutes, out["FGM"] * minutes)
        out["AST"] = rates["AST"] * team.map(1.0 / ratio).astype(float).pow(power).where(vet, 1.0)
    if variant.startswith("redistrib_tov"):
        tmp, _, _ = P.apply_usage_redistribution(table, rates, weight, usage_fit, changes)
        out["TOV"] = out["TOV"] + (tmp["TOV"] - rates["TOV"])
    return out


VARIANTS = ["none", "current", "tov_0", "tov_0.25", "tov_0.75", "tov_1", "ast_same",
            "ast_budget_0.5", "ast_budget_1", "share_1", "share_-1", "share_2", "redistrib_tov", "redistrib_tov_0.75", "redistrib_tov_1"]


def main():
    history = P.load_history()
    decisions = json.loads(M.MODEL_METADATA_PATH.read_text())["policy_decisions"]
    pairs = M.load_training_pairs()
    draft = R.load_draft_history()
    records = []
    for year in (2024, 2025, 2026):
        past = history.loc[history["SEASON_YEAR"] < year]
        season = history.loc[history["SEASON_YEAR"] == year].set_index("PLAYER_ID")
        roster = pd.DataFrame({"PLAYER_ID": season.index.to_numpy(),
                               "TEAM_ABBREVIATION": season["TEAM_ABBREVIATION"].to_numpy(),
                               "ROSTER_NAME": season["PLAYER_NAME"].to_numpy()})
        newcomers = ~season.index.isin(past["PLAYER_ID"])
        bundle = M.build_bundle(M._train_slice(pairs, year), decisions)
        latest = P.latest_season_rows(past)
        P.project_rosters(bundle, roster, past, latest=latest, draft=draft,
                          current_ages=season.loc[newcomers, "AGE"], usage_budget="none",
                          **B.ALLOCATION_VARIANTS["current"])
        table, rates, raw_mpg, proj_gp, mpg, season_min, weight, mpg_blended = captured["args"]
        usage_fit = P.fit_usage_redistribution(past, bundle)
        changes = P.current_usage_changes(roster, past, latest[1])
        minutes = season_min.fillna(0.0)
        for variant in VARIANTS:
            r2 = apply(variant, table, rates, minutes, usage_fit, changes, weight)
            pred = _original(table, r2, raw_mpg, proj_gp, mpg, season_min, weight, mpg_blended).set_index("PLAYER_ID")
            actual = season.loc[pred.index]
            r2.index = pred.index
            usg_last = (actual["MIN"] > 0)
            groups = {
                "500+ actual minutes": actual["MIN"] >= 500,
                "starters_24_plus_actual": (actual["MIN"] / actual["GP"] >= 24) & (actual["GP"] >= 20),
                "rookies": pd.Series(newcomers & True, index=season.index).reindex(pred.index),
            }
            for group, mask in groups.items():
                mask = mask.fillna(False).astype(bool)
                m = B._allocation_metrics(pred, actual, mask, "PROJECTED_MPG", "PROJECTED_SEASON_MIN")
                w = actual.loc[mask, "MIN"]
                for s in PER36:
                    err = (r2.loc[mask, s] - actual.loc[mask, f"{s}_RATE"]) * 36
                    m[f"{s}36"] = float(np.average(err.abs(), weights=w))
                    m[f"{s}36_bias"] = float(np.average(err, weights=w))
                records.append({"year": year, "variant": variant, "group": group, **m})
    res = pd.DataFrame(records)
    res.to_csv(OUT / "test_D2_results.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["PTS36", "FGA36", "FTA36", "FG3M36", "TOV36", "AST36", "VALUE_Spearman"]
    for group in ("500+ actual minutes", "starters_24_plus_actual"):
        print(f"\n{group}")
        t = res.loc[res["group"] == group].pivot_table(index="variant", columns="year", values=cols)
        print(t.reindex(VARIANTS).round(3).to_string())


if __name__ == "__main__":
    main()
