"""Rankings before the Usage branch vs now (read-only): python analysis/usage_rank_changes.py
Old = data/processed/baseline_pre_usage/projections_by_team.csv, new = current projections;
both ranked with rankings.rank_players. Writes analysis/results/usage_rank_changes.csv."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd

import rankings as RK

old = pd.read_csv(ROOT / "data/processed/baseline_pre_usage/projections_by_team.csv")
new = pd.read_csv(ROOT / "data/processed/projections_by_team.csv")
ro, rn = RK.rank_players(old), RK.rank_players(new)
m = rn[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "RANK", "VALUE"]].merge(
    ro[["PLAYER_ID", "RANK", "VALUE"]], on="PLAYER_ID", suffixes=("_NEW", "_OLD"), how="outer")
stats = ["PROJ_PTS", "PROJ_FG3M", "PROJ_AST", "PROJ_TOV"]
m = m.merge(new[["PLAYER_ID", *stats, "USAGE_BUDGET_SCALE", "TEAM_VACATED_USAGE_REL",
                 "TEAM_ARRIVED_USAGE_REL", "NO_PRIOR_DATA"]], on="PLAYER_ID", how="left")
m = m.merge(old[["PLAYER_ID", *stats]].rename(columns={s: f"{s}_OLD" for s in stats}), on="PLAYER_ID", how="left")
m["RANK_CHANGE"] = m["RANK_OLD"] - m["RANK_NEW"]  # positive = moved up
m["VALUE_CHANGE"] = m["VALUE_NEW"] - m["VALUE_OLD"]
for s in ("PTS", "TOV"):
    m[f"{s}_PG_CHANGE"] = m[f"PROJ_{s}"] - m[f"PROJ_{s}_OLD"]
cols = ["RANK_NEW", "RANK_OLD", "RANK_CHANGE", "PLAYER_NAME", "TEAM_ABBREVIATION", "VALUE_NEW", "VALUE_OLD",
        "VALUE_CHANGE", "PROJ_PTS_OLD", "PROJ_PTS", "PTS_PG_CHANGE", "PROJ_TOV_OLD", "PROJ_TOV", "TOV_PG_CHANGE",
        "PROJ_FG3M_OLD", "PROJ_FG3M", "PROJ_AST_OLD", "PROJ_AST", "USAGE_BUDGET_SCALE",
        "TEAM_VACATED_USAGE_REL", "TEAM_ARRIVED_USAGE_REL", "NO_PRIOR_DATA", "PLAYER_ID"]
m = m[cols].sort_values("RANK_NEW")
m.round(3).to_csv(ROOT / "analysis/results/usage_rank_changes.csv", index=False)
top = m.loc[m["RANK_OLD"] <= 150]
print(f"{len(m)} players; top 150: unchanged {int((top['RANK_CHANGE'] == 0).sum())}, "
      f"median |move| {top['RANK_CHANGE'].abs().median():.0f}, max {top['RANK_CHANGE'].abs().max():.0f}")
