"""Offline tests for src/inseason.py."""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import inseason as IS  # noqa: E402


def _log(rows):
    columns = ["PLAYER_ID", "GAME_ID", "GAME_DATE", "MIN", *IS.TOTAL_STATS]
    return pd.DataFrame(rows, columns=columns)


class InseasonTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for g in range(10):  # player 1: 10 games x 30 min, 15 pts, 12 FGA (6 made), 4 FTA (3 made)
            rows.append([1, f"g{g}", f"2026-10-{20 + g}", 30, 15, 1, 5, 4, 1, 0, 2, 6, 12, 3, 4])
        self.obs = IS.observed_stats(_log(rows))

    def test_observed_and_as_of(self):
        self.assertEqual(self.obs.at[1, "GP"], 10)
        self.assertEqual(self.obs.at[1, "MIN"], 300)
        self.assertEqual(str(self.obs.attrs["as_of"]), "2026-10-29")

    def test_minutes_blend_weights_games(self):
        prior = pd.Series([20.0, 25.0], index=[1, 2])
        mpg, w = IS.blend_minutes(prior, self.obs)
        self.assertAlmostEqual(w[1], 10 / (10 + IS.MINUTES_K))
        self.assertAlmostEqual(mpg[1], 0.5 * 20 + 0.5 * 30)
        self.assertEqual(mpg[2], 25.0)  # no games: preseason

    def test_rate_blend_and_percentages(self):
        prior = pd.DataFrame({s: [0.0, 0.0] for s in IS.TOTAL_STATS}, index=[1, 2])
        prior.loc[:, "PTS"] = 0.3
        prior.loc[:, "FGA"] = 0.4
        prior.loc[:, "FGM"] = 0.16  # 40%
        prior.loc[:, "FTA"] = 0.1
        prior.loc[:, "FTM"] = 0.08
        out = IS.blend_rates(prior, self.obs)
        k = IS.RATE_K["PTS"]
        self.assertAlmostEqual(out.at[1, "PTS"], (k * 0.3 + 150) / (k + 300))
        k_fg = IS.PCT_K["FG"][2]
        fg_pct = (k_fg * 0.4 + 60) / (k_fg + 120)
        self.assertAlmostEqual(out.at[1, "FGM"] / out.at[1, "FGA"], fg_pct)
        self.assertAlmostEqual(out.at[2, "PTS"], 0.3)

    def test_combine_adds_actual_games(self):
        proj = pd.DataFrame(
            {"PLAYER_ID": [1], "PROJECTED_GP": [60.0], "PROJECTED_MPG": [25.0], "PROJECTED_SEASON_MIN": [1500.0],
             **{f"TOTAL_{s}": [100.0] for s in IS.TOTAL_STATS},
             **{f"PROJ_{s}": [0.0] for s in IS.TOTAL_STATS}}
        )
        out = IS.combine_with_actual(proj, self.obs, self.obs.attrs["as_of"])
        self.assertEqual(out.at[0, "PROJECTED_GP"], 70)
        self.assertEqual(out.at[0, "TOTAL_PTS"], 250)
        self.assertAlmostEqual(out.at[0, "PROJECTED_MPG"], round(1800 / 70, 1))
        self.assertAlmostEqual(out.at[0, "PROJ_PTS"], 250 / 70)
        self.assertEqual(out.at[0, "AS_OF"], "2026-10-29")


if __name__ == "__main__":
    unittest.main()
