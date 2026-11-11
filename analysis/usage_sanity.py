"""Usage measures: sanity checks (read-only). python analysis/usage_sanity.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import project as P
from history import season_usage, usage_feature_table

hist = P.load_history()
u = season_usage(hist)
u = u.merge(hist[["PLAYER_ID", "SEASON_YEAR", "PLAYER_NAME", "GP"]], on=["PLAYER_ID", "SEASON_YEAR"])
tm = u.groupby(["SEASON_YEAR", "TEAM_ABBREVIATION"])
team_min = tm["MIN"].sum()
for c in ["USG", "REB_SHARE", "USG_S"]:
    s = (u[c] * u["MIN"]).groupby([u["SEASON_YEAR"], u["TEAM_ABBREVIATION"]]).sum() / (team_min / 5)
    print(f"Team sum of {c} (minute-weighted, 1.00 = 100%): min {s.min():.3f} mean {s.mean():.3f} max {s.max():.3f}")
full = {1999: 50, 2012: 66, 2020: 72, 2021: 72}
ratio = team_min / [full.get(y, 82) * 240 for y, _ in team_min.index]
recent = ratio[ratio.index.get_level_values(0) >= 2014]
print("\nTeam minutes / (games x 240), a trade-distortion proxy (2014+; COVID 2020 seasons vary):")
print(recent.describe().round(3).to_string())
print("Most distorted team-seasons:", recent.sort_values().iloc[[0, 1, 2, -3, -2, -1]].round(3).to_dict())
print("\nLeague average AST_PCT by season (recent):", u.groupby("SEASON_YEAR")["AST_PCT_LEAGUE"].first().tail(3).round(3).to_dict())
names = ["Luka Dončić", "Joel Embiid", "Giannis Antetokounmpo", "Shai Gilgeous-Alexander", "Nikola Jokić",
         "Stephen Curry", "Trae Young", "Jalen Brunson", "Anthony Edwards", "Tyrese Haliburton",
         "Rudy Gobert", "Jayson Tatum", "LeBron James", "Domantas Sabonis"]
show = u.loc[u["PLAYER_NAME"].isin(names) & u["SEASON_YEAR"].isin([2024, 2025, 2026])]
show = show.assign(USG_pct=100 * show["USG"], AST_pct=100 * show["AST_PCT"], REB_share_pct=100 * show["REB_SHARE"])
print("\nCheck against published USG% / AST% (e.g. Basketball-Reference):")
print(show.pivot_table(index="PLAYER_NAME", columns="SEASON_YEAR", values=["USG_pct", "AST_pct"]).round(1).to_string())
f = usage_feature_table(hist)
print("\nFeature table rows:", len(f), "NaNs:", int(f.isna().sum().sum()))
# No-future-data check: features for seasons <= 2020 must not change when later seasons are dropped.
early = usage_feature_table(hist.loc[hist["SEASON_YEAR"] <= 2020])
both = early.merge(f, on=["PLAYER_ID", "SEASON_YEAR"], suffixes=("_a", "_b"))
cols = [c for c in early.columns if c not in ("PLAYER_ID", "SEASON_YEAR")]
print("Max change in <=2020 features after adding later seasons:",
      max(float(np.nanmax(np.abs(both[c + "_a"] - both[c + "_b"]))) for c in cols))
