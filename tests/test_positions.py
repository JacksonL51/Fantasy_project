"""Run from the project root: python -m unittest discover tests"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import history as H


class PositionCrowdingTests(unittest.TestCase):
    def test_crowded_position_scores_higher(self):
        positions = pd.Series({1: "C", 2: "C", 3: "G", 4: "G", 5: "C", 6: "G"})
        players = pd.DataFrame(
            {
                "PLAYER_ID": [1, 2, 3, 4, 5, 6],
                # Team A has two centers; team B has one center and one guard.
                "TEAM": ["A", "A", "A", "A", "B", "B"],
                "PRIOR_MPG": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            }
        )
        crowd = H.position_crowding(players, positions)
        # Center 1 shares 48 center minutes with a 30-MPG center; team total is 90 others.
        self.assertAlmostEqual(crowd.loc[0, "TOTAL_CROWD"], 90 / 240)
        self.assertAlmostEqual(crowd.loc[0, "POSITION_CROWD"], 30 / 48 - 90 / 240)
        # Team B's center has no other center: less crowded than the team overall.
        self.assertLess(crowd.loc[4, "POSITION_CROWD"], 0)
        self.assertGreater(crowd.loc[0, "POSITION_CROWD"], crowd.loc[2, "POSITION_CROWD"])

    def test_unknown_position_and_no_prior(self):
        positions = pd.Series({1: "F", 2: "G-F"})
        players = pd.DataFrame(
            {"PLAYER_ID": [1, 2, 3], "TEAM": ["A", "A", "A"], "PRIOR_MPG": [20.0, np.nan, 25.0]}
        )
        crowd = H.position_crowding(players, positions)
        self.assertTrue(np.isnan(crowd.loc[2, "POSITION_CROWD"]))  # no position
        # Player 2 adds no demand (no prior season); player 3 still counts in the total.
        self.assertAlmostEqual(crowd.loc[0, "TOTAL_CROWD"], 25 / 240)
        self.assertAlmostEqual(crowd.loc[1, "TOTAL_CROWD"], 45 / 240)

    def test_adjustment_sign_clip_and_inactive(self):
        crowd = pd.Series([1.0, -1.0, 10.0, np.nan])
        model = {"active": True, "coef": -0.5}
        shift = H.position_mpg_adjustment(model, crowd)
        self.assertAlmostEqual(shift[0], -0.5 * H.POSITION_SHRINK)
        self.assertAlmostEqual(shift[1], 0.5 * H.POSITION_SHRINK)
        self.assertEqual(shift[2], -H.POSITION_CLIP)
        self.assertEqual(shift[3], 0.0)
        self.assertTrue((H.position_mpg_adjustment({"active": False}, crowd) == 0).all())

    def test_fit_inactive_without_positions(self):
        history = pd.DataFrame(
            {"PLAYER_ID": [1], "SEASON_YEAR": [2020], "TEAM_ABBREVIATION": ["A"],
             "AGE": [25], "GP": [50], "MIN": [1000.0]}
        )
        self.assertFalse(H.fit_position_crowding(history, pd.Series(dtype=object))["active"])


if __name__ == "__main__":
    unittest.main()
