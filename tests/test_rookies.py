"""Run from the project root: python -m unittest discover tests"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import project as P
import rookies as R

STATS = ["FGM", "FGA", "FG3M", "FTM", "FTA", "REB", "AST", "STL", "BLK", "PTS", "TOV"]


def fake_history(classes=range(1998, 2021), per_class: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rookie seasons where better picks play more and score more, plus a draft table."""
    rng = np.random.default_rng(seed)
    rows, draft = [], []
    player_id = 1
    for year in classes:
        for slot in range(per_class):
            pick = slot + 1 if slot < 35 else None  # the last few are undrafted
            strength = 1.0 - (slot / per_class)
            gp = int(np.clip(20 + 50 * strength + rng.normal(0, 5), 1, 82))
            mpg = float(np.clip(8 + 20 * strength + rng.normal(0, 2), 2, 36))
            row = {
                "PLAYER_ID": player_id, "PLAYER_NAME": f"P{player_id}", "SEASON_YEAR": year,
                "AGE": 19 + slot % 5, "GP": gp, "MIN": gp * mpg,
            }
            for stat in STATS:
                row[f"{stat}_RATE"] = 0.1 + 0.2 * strength + rng.normal(0, 0.01)
            rows.append(row)
            if pick is not None:
                draft.append({"PLAYER_ID": player_id, "DRAFT_YEAR": year - 1, "OVERALL_PICK": pick})
            player_id += 1
    return pd.DataFrame(rows), pd.DataFrame(draft)


class DraftDataTests(unittest.TestCase):
    def test_load_keeps_latest_draft_and_drops_missing_ids(self):
        raw = pd.DataFrame(
            {
                "PERSON_ID": [5, 5, 0, 7],
                "SEASON": [1985, 1986, 1990, 2020],
                "OVERALL_PICK": [77, 24, 3, 10],
            }
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "draft.csv"
            raw.to_csv(path, index=False)
            draft = R.load_draft_history(path).set_index("PLAYER_ID")
        self.assertEqual(sorted(draft.index), [5, 7])
        self.assertEqual(draft.loc[5, "OVERALL_PICK"], 24)

    def test_undrafted_features(self):
        frame = pd.DataFrame({"PLAYER_ID": [1, 2], "AGE": [20, 23]})
        draft = pd.DataFrame({"PLAYER_ID": [1], "DRAFT_YEAR": [2020], "OVERALL_PICK": [3]})
        out = R.add_draft_features(frame, draft)
        self.assertEqual(out["UNDRAFTED"].tolist(), [0.0, 1.0])
        self.assertAlmostEqual(out.loc[1, "LOG_PICK"], np.log(R.UNDRAFTED_PICK))
        self.assertAlmostEqual(out.loc[0, "LOG_PICK_SQ"], np.log(3) ** 2)

    def test_rookie_seasons_use_first_season_from_1998(self):
        history = pd.DataFrame(
            {
                "PLAYER_ID": [1, 1, 2, 2],
                "SEASON_YEAR": [1997, 1998, 1999, 2000],
                "AGE": [30, 31, 21, 22],
                "GP": [50, 50, 40, 60],
                "MIN": [1000.0, 1000.0, 800.0, 1500.0],
                **{f"{s}_RATE": [0.2, 0.2, 0.1, 0.3] for s in STATS},
            }
        )
        rookies = R.rookie_seasons(history, pd.DataFrame(columns=["PLAYER_ID", "DRAFT_YEAR", "OVERALL_PICK"]), STATS)
        self.assertEqual(rookies["PLAYER_ID"].tolist(), [2])  # player 1 debuted before the data
        self.assertEqual(int(rookies["SEASON_YEAR"].iloc[0]), 1999)
        self.assertAlmostEqual(rookies["MPG"].iloc[0], 20.0)
        # Only player 2 played in 1999, so the rate equals the league rate.
        self.assertAlmostEqual(rookies["PTS_INDEX"].iloc[0], 1.0)


class RookieModelTests(unittest.TestCase):
    def setUp(self):
        self.history, self.draft = fake_history()
        self.rookies = R.rookie_seasons(self.history, self.draft, STATS)
        self.models = R.fit_rookie_models(self.rookies, STATS)
        self.league = R.league_rates(self.history, STATS).iloc[-1]

    def test_better_picks_project_higher(self):
        features = pd.DataFrame({"PLAYER_ID": [0, 0, 0], "AGE": [21.0, 21.0, 21.0]})
        draft = pd.DataFrame({"PLAYER_ID": [0], "DRAFT_YEAR": [2020], "OVERALL_PICK": [1]})
        features = R.add_draft_features(features.iloc[:1], draft)
        later = features.copy()
        later["OVERALL_PICK"], later["LOG_PICK"] = 30, np.log(30)
        later["LOG_PICK_SQ"] = later["LOG_PICK"] ** 2
        both = pd.concat([features, later], ignore_index=True)
        pred = R.predict_rookies(self.models, both, self.league)
        self.assertGreater(pred.loc[0, "MPG"], pred.loc[1, "MPG"])
        self.assertGreater(pred.loc[0, "GAMES_82"], pred.loc[1, "GAMES_82"])
        self.assertGreater(pred.loc[0, "PTS"], pred.loc[1, "PTS"])
        self.assertTrue(pred["MPG"].between(*R.MPG_BOUNDS).all())
        self.assertTrue(pred["GAMES_82"].between(*R.GAMES_BOUNDS).all())

    def test_missing_age_uses_pick_bucket_median(self):
        features = pd.DataFrame({"PLAYER_ID": [0, 99], "AGE": [np.nan, np.nan]})
        draft = pd.DataFrame({"PLAYER_ID": [0], "DRAFT_YEAR": [2020], "OVERALL_PICK": [5]})
        features = R.add_draft_features(features, draft)
        age = R.fill_missing_age(self.models, features)
        self.assertFalse(age.isna().any())
        self.assertEqual(age.iloc[1], self.models["undrafted_age"])

    def test_prior_features_ages(self):
        current_ages = pd.Series({5000: 19.0})
        old_vet = pd.DataFrame({"PLAYER_ID": [4000], "SEASON_YEAR": [1997], "AGE": [28]})
        history = pd.concat([self.history, old_vet], ignore_index=True)
        ids = pd.Series([1, 5000, 4000])
        features = R.prior_features(ids, history, self.draft, current_ages)
        first_age = self.history.loc[self.history["PLAYER_ID"] == 1, "AGE"].iloc[0]
        self.assertEqual(features["AGE"].tolist()[:2], [first_age, 19.0])
        self.assertEqual(features["HAS_ROOKIE_PRIOR"].tolist(), [True, True, False])
        self.assertEqual(features["NBA_HISTORY"].tolist(), [True, False, True])


class TeamOpportunityTests(unittest.TestCase):
    def test_crowding_and_vacated_share(self):
        players = pd.DataFrame(
            {
                "PLAYER_ID": [1, 2, 3, 4],
                "TEAM": ["A", "A", "B", "B"],
                "PRIOR_MPG": [30.0, np.nan, 20.0, np.nan],  # 2 and 4 are rookies
            }
        )
        last_season = pd.DataFrame(
            {
                "PLAYER_ID": [1, 5, 3, 6],
                "TEAM": ["A", "A", "B", "B"],
                "MIN": [2000.0, 2000.0, 1000.0, 3000.0],  # 5 and 6 have left
            }
        )
        out = R.team_opportunity(players, last_season)
        self.assertAlmostEqual(out.loc[1, "TOTAL_CROWD"], 30 / 240)
        self.assertAlmostEqual(out.loc[3, "TOTAL_CROWD"], 20 / 240)
        self.assertAlmostEqual(out.loc[1, "VACATED_SHARE"], 0.5)
        self.assertAlmostEqual(out.loc[3, "VACATED_SHARE"], 0.75)
        # Relative to the average team: A is more crowded, B more open.
        self.assertAlmostEqual(out.loc[1, "TOTAL_CROWD_REL"], 30 / 240 - 25 / 240)
        self.assertAlmostEqual(out.loc[3, "VACATED_SHARE_REL"], 0.75 - 0.625)

    def test_open_team_projects_more_minutes(self):
        history, draft = fake_history()
        rookies = R.rookie_seasons(history, draft, STATS)
        rng = np.random.default_rng(1)
        rookies["TOTAL_CROWD_REL"] = rng.normal(0, 0.2, len(rookies))
        rookies["VACATED_SHARE_REL"] = rng.normal(0, 0.2, len(rookies))
        rookies["MPG"] = rookies["MPG"] - 10 * rookies["TOTAL_CROWD_REL"]
        models = R.fit_rookie_models(rookies, STATS)
        self.assertIn("TOTAL_CROWD_REL", models["minutes_features"])
        league = R.league_rates(history, STATS).iloc[-1]
        same = rookies.iloc[[0, 0]].reset_index(drop=True)
        same["TOTAL_CROWD_REL"] = [-0.3, 0.3]
        pred = R.predict_rookies(models, same, league)
        self.assertGreater(pred.loc[0, "MPG"], pred.loc[1, "MPG"])
        # Missing opportunity falls back to the training mean instead of failing.
        bare = same.drop(columns=R.OPPORTUNITY_FEATURES)
        self.assertFalse(R.predict_rookies(models, bare, league)["MPG"].isna().any())

    def test_no_team_data_skips_opportunity(self):
        history, draft = fake_history()
        models = R.fit_rookie_models(R.rookie_seasons(history, draft, STATS), STATS)
        self.assertEqual(models["minutes_features"], R.FEATURES)


class ProjectionHookTests(unittest.TestCase):
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "PLAYER_ID": [1, 2, 3, 4],
                "NO_PRIOR_DATA": [True, True, False, False],
                "CAREER_FIRST_YEAR": [np.nan, np.nan, 2026, 2015],
                "ELIGIBLE_MPG_SEASONS": [0, 0, 0, 3],
                "WEIGHTED_SEASON_MINUTES": [np.nan, np.nan, 100.0, 2000.0],
            }
        )

    def priors(self) -> pd.DataFrame:
        priors = pd.DataFrame({stat: [0.5] * 4 for stat in STATS})
        priors["MPG"], priors["GAMES_82"] = 30.0, 70.0
        priors["OVERALL_PICK"] = [1, 5, 10, 20]
        priors["HAS_ROOKIE_PRIOR"] = True
        # Player 2 is a veteran back after years away, not a rookie.
        priors["ROOKIE"] = [True, False, False, False]
        return priors

    def test_shrink_targets(self):
        table = self.table()
        replacement = pd.Series(0.1, index=STATS)
        targets = P.shrink_targets(table, replacement, self.priors(), latest_year=2026)
        self.assertEqual(table["ROOKIE_PRIOR"].tolist(), [True, False, True, False])
        for stat in STATS:
            expected = 0.5 if stat in P.ROOKIE_RATE_STATS else 0.1
            self.assertEqual(targets.loc[0, stat], expected)
            self.assertEqual(targets.loc[2, stat], expected)  # early career
            self.assertEqual(targets.loc[1, stat], 0.1)
            self.assertEqual(targets.loc[3, stat], 0.1)

    def test_rookie_minutes_and_games(self):
        table = self.table()
        mpg = pd.Series([4.0, 4.0, 4.0, 25.0])
        gp = pd.Series([58.0, 58.0, 60.0, 70.0])
        new_mpg, new_gp = P.apply_rookie_minutes_and_games(table, mpg, gp, self.priors(), 2026)
        self.assertEqual(new_mpg.tolist(), [30.0, 4.0, 30.0, 25.0])
        self.assertEqual(new_gp.tolist(), [70.0, 58.0, 60.0, 70.0])
        self.assertEqual(table.loc[0, "ALLOCATION_SEASON_MINUTES"], 2100.0)
        self.assertEqual(table["ROOKIE_MINUTES_PRIOR"].tolist(), [True, False, False, False])

    def test_replacement_method_leaves_everyone_alone(self):
        table = self.table()
        mpg, gp = pd.Series([4.0] * 4), pd.Series([58.0] * 4)
        new_mpg, new_gp = P.apply_rookie_minutes_and_games(table, mpg, gp, None, 2026)
        self.assertTrue(new_mpg.equals(mpg) and new_gp.equals(gp))
        targets = P.shrink_targets(table, pd.Series(0.1, index=STATS), None, 2026)
        self.assertTrue((targets == 0.1).all().all())

    def test_model_minutes_skip_cohort_blend(self):
        count = 12
        mpg = np.array([30.0] * 9 + [10.0, 10.0, 10.0])
        gp = np.full(count, 60.0)
        weighted = np.array([1800.0] * 9 + [300.0, 300.0, 300.0])
        career = np.array([5000.0] * 9 + [0.0, 0.0, 0.0])
        ages = np.full(count, 22.0)
        from_model = np.array([False] * 9 + [True, False, False])
        *_, baseline, credibility, _ = P.allocate_team_season_minutes(
            mpg, gp, weighted, career, ages, minutes_from_model=from_model
        )
        self.assertEqual(credibility[9], 1.0)
        self.assertEqual(baseline[9], 300.0)
        self.assertEqual(credibility[10], 0.0)
        self.assertEqual(baseline[10], P.COHORT_PRIOR_MINUTES["young"])


if __name__ == "__main__":
    unittest.main()
