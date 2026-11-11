"""Test C follow-up (read-only): which features carry the signal, and the biggest losses.
python analysis/usage_test_c_detail.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

import usage_tests as T

hist, feats, pairs = T.load()
res, changes, p = T.test_c(hist, pairs)
stayed = p["TEAM_NEXT"] == p["TEAM_PREV"]
top_q = p["VACATED"] >= 0.50  # top quarter of teams since 2014
stats = ["PTS", "FGA", "FTA", "AST", "TOV"]
variants = {
    "vacated only": ["VACATED_REL", "VAC_X_USG"],
    "arrived only": ["ARRIVED_REL", "ARR_X_USG"],
    "both, no usage interaction": ["VACATED_REL", "ARRIVED_REL"],
    "both + own-usage interaction": ["VACATED_REL", "ARRIVED_REL", "VAC_X_USG", "ARR_X_USG"],
    "both + interaction + own usage": ["VACATED_REL", "ARRIVED_REL", "VAC_X_USG", "ARR_X_USG"] + T.USAGE_FEATURES,
}
rows = []
for name, cols in variants.items():
    summary, preds = T.rate_test(p, stats, cols, "C_detail", subgroup=top_q & stayed)
    rows.append(summary.assign(variant=name, group="all"))
    rows.append(T.summarize_rates(preds, "x", sub_only=True).assign(variant=name, group="stayers, team lost 50%+"))
out = pd.concat(rows)
pd.set_option("display.width", 220)
cols = ["variant", "group", "stat", "n", "MAE36_base", "MAE36_usage", "wins", "gain_CI_lo", "bias36_base", "bias36_usage", "passes_rule"]
print(out[cols].round(3).to_string(index=False))
out.to_csv(T.OUT / "test_C_detail.csv", index=False)
