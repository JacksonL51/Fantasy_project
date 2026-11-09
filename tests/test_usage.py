"""Run from the project root: python -m unittest discover tests"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import history as H
import project as P

STATS = ["FGM", "FGA", "FG3M", "FTM", "FTA", "REB", "AST", "STL", "BLK", "PTS", "TOV"]


def fake_history(seed: int = 0, years=range(2010, 2016), teams=("A", "B", "C"), per_team: int = 8) -> pd.DataFrame:
    """Player-seasons on a few teams, with some players changing teams each year."""
    rng = np.random.default_rng(seed)
    rows = []
    n_players = len(teams) * per_team
    for year in years:
        for pid in range(1, n_players + 1):
            team = teams[(pid + (year if pid % 4 == 0 else 0)) % len(teams)]
            minutes = float(rng.uniform(150, 2800))
            row = {"PLAYER_ID": pid, "PLAYER_NAME": f"P{pid}", "SEASON_YEAR": year,
                   "TEAM_ABBREVIATION": team, "AGE": 22 + (pid % 12), "GP": 60, "MIN": minutes}
            for stat in STATS:
                row[f"{stat}_RATE"] = float(rng.uniform(0.01, 0.6))
            row["FGM_RATE"] = row["FGA_RATE"] * 0.45
            row["AST_RATE"] = min(row["AST_RATE"], 0.15)
            rows.append(row)
    return pd.DataFrame(rows)


class UsageMeasureTests(unittest.TestCase):
    def test_team_usage_shares_sum_to_one(self):
        usage = H.season_usage(fake_history())
        team_min = usage.groupby(["SEASON_YEAR", "TEAM_ABBREVIATION"])["MIN"].transform("sum")
        share = (usage["USG"] * usage["MIN"] / (team_min / 5.0)).groupby(
            [usage["SEASON_YEAR"], usage["TEAM_ABBREVIATION"]]).sum()
        np.testing.assert_allclose(share.to_numpy(), 1.0)
        reb = (usage["REB_SHARE"] * usage["MIN"] / (team_min / 5.0)).groupby(
            [usage["SEASON_YEAR"], usage["TEAM_ABBREVIATION"]]).sum()
        np.testing.assert_allclose(reb.to_numpy(), 1.0)

    def test_usage_formula_on_a_tiny_team(self):
        # Two players, equal minutes: the one using twice the possessions gets 2/3 of 5 x 1/2.
        history = pd.DataFrame({
            "PLAYER_ID": [1, 2], "SEASON_YEAR": [2020, 2020], "TEAM_ABBREVIATION": ["A", "A"],
            "MIN": [1000.0, 1000.0], "FGA_RATE": [0.4, 0.2], "FTA_RATE": [0.0, 0.0], "TOV_RATE": [0.0, 0.0],
            "AST_RATE": [0.1, 0.1], "FGM_RATE": [0.2, 0.1], "REB_RATE": [0.1, 0.1],
        })
        usage = H.season_usage(history).set_index("PLAYER_ID")
        # Team minutes 2000 -> each player on court 1000 / 400 = 2.5 "games" of 5 slots.
        self.assertAlmostEqual(usage.loc[1, "USG"], 400 / (2.5 * 600))
        self.assertAlmostEqual(usage.loc[2, "USG"], 200 / (2.5 * 600))

    def test_features_ignore_later_seasons(self):
        history = fake_history()
        full = H.usage_feature_table(history)
        early = H.usage_feature_table(history.loc[history["SEASON_YEAR"] <= 2012])
        merged = early.merge(full, on=["PLAYER_ID", "SEASON_YEAR"], suffixes=("_a", "_b"))
        for column in early.columns.drop(["PLAYER_ID", "SEASON_YEAR"]):
            np.testing.assert_allclose(merged[f"{column}_a"], merged[f"{column}_b"])
        changes_full = H.usage_change_history(history)
        changes_early = H.usage_change_history(history.loc[history["SEASON_YEAR"] <= 2012])
        both = changes_early.merge(changes_full, on=["SEASON_YEAR", "TEAM_ABBREVIATION"])
        np.testing.assert_allclose(both["VACATED_REL_x"], both["VACATED_REL_y"])


class TeamUsageChangeTests(unittest.TestCase):
    def test_vacated_and_arrived_and_centering(self):
        previous = pd.DataFrame({"PLAYER_ID": [1, 2, 3, 4], "TEAM_ABBREVIATION": ["A", "A", "B", "B"],
                                 "POSS": [300.0, 100.0, 200.0, 200.0]})
        # Player 1 (A) moves to B, player 3 (B) retires, player 9 is a rookie on A.
        current = pd.Series({1: "B", 2: "A", 4: "B", 9: "A"})
        out = H.team_usage_changes(previous, current)
        self.assertAlmostEqual(out.loc["A", "VACATED"], 0.75)
        self.assertAlmostEqual(out.loc["A", "ARRIVED"], 0.0)
        self.assertAlmostEqual(out.loc["B", "VACATED"], 0.5)
        self.assertAlmostEqual(out.loc["B", "ARRIVED"], 0.75)
        self.assertAlmostEqual(out["VACATED_REL"].sum(), 0.0)
        self.assertAlmostEqual(out["ARRIVED_REL"].sum(), 0.0)


class UsageBudgetTests(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame({"TEAM_ABBREVIATION": ["A", "A", "B", "B"],
                                   "NO_PRIOR_DATA": [False, True, False, False]})
        self.rates = pd.DataFrame({s: [0.2, 0.2, 0.1, 0.1] for s in STATS})
        self.minutes = pd.Series([1000.0, 1000.0, 1000.0, 1000.0])

    def test_high_volume_team_scaled_down_and_rookies_kept(self):
        out, scale = P.apply_usage_budget(self.table, self.rates, self.minutes, "veterans", 0.5)
        # Volumes 0.2+0.088+0.2 vs half that: relative 4/3 and 2/3.
        self.assertAlmostEqual(scale[0], (3 / 4) ** 0.5)
        self.assertAlmostEqual(scale[1], 1.0)  # no prior data
        self.assertAlmostEqual(scale[2], (3 / 2) ** 0.5)
        self.assertAlmostEqual(out.loc[0, "PTS"], 0.2 * (3 / 4) ** 0.5)
        self.assertEqual(out.loc[0, "REB"], 0.2)  # not a shot stat
        # Percentages hold: makes and attempts scale together.
        self.assertAlmostEqual(out.loc[0, "FGM"] / out.loc[0, "FGA"], 1.0)

    def test_off_and_power_zero_change_nothing(self):
        for method, power in (("none", 0.5), ("all", 0.0)):
            out, scale = P.apply_usage_budget(self.table, self.rates, self.minutes, method, power)
            pd.testing.assert_frame_equal(out, self.rates)
            self.assertTrue((scale == 1.0).all())

    def test_redistribution_shifts_and_keeps_percentages(self):
        changes = pd.DataFrame({"VACATED_REL": [0.2, -0.2], "ARRIVED_REL": [0.0, 0.0]}, index=["A", "B"])
        model = {"active": True, "coef": {s: {"VACATED_REL": 0.05, "ARRIVED_REL": -0.05}
                                          for s in P.USAGE_REDISTRIBUTION_STATS}}
        weight = pd.Series(1.0, index=self.table.index)
        out, vacated, _ = P.apply_usage_redistribution(self.table, self.rates, weight, model, changes)
        self.assertAlmostEqual(out.loc[0, "PTS"], 0.2 + 0.01)
        self.assertAlmostEqual(out.loc[1, "PTS"], 0.2)  # no prior data: unchanged
        self.assertAlmostEqual(out.loc[2, "PTS"], 0.1 - 0.01)
        self.assertAlmostEqual(out.loc[0, "FGM"] / out.loc[0, "FGA"], 1.0)
        self.assertAlmostEqual(out.loc[0, "FG3M"], 0.2)  # not shifted

    def test_turnover_only_model_moves_only_turnovers(self):
        changes = pd.DataFrame({"VACATED_REL": [0.2, -0.2], "ARRIVED_REL": [0.1, 0.0]}, index=["A", "B"])
        model = {"active": True, "coef": {"TOV": {"VACATED_REL": 0.01, "ARRIVED_REL": -0.02}}}
        weight = pd.Series([1.0, 1.0, 0.5, 1.0], index=self.table.index)
        out, _, _ = P.apply_usage_redistribution(self.table, self.rates, weight, model, changes)
        self.assertAlmostEqual(out.loc[0, "TOV"], 0.2 + 0.01 * 0.2 - 0.02 * 0.1)
        self.assertAlmostEqual(out.loc[2, "TOV"], 0.1 + 0.5 * (0.01 * -0.2))  # credibility 0.5
        pd.testing.assert_frame_equal(out.drop(columns="TOV"), self.rates.drop(columns="TOV"))
        self.assertIn("tov", P.USAGE_REDISTRIBUTION_METHODS)
        self.assertEqual(P.USAGE_REDISTRIBUTION_METHODS["tov"], ["TOV"])


if __name__ == "__main__":
    unittest.main()
