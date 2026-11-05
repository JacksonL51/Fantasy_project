"""Run from the project root: python -m unittest discover tests"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from league_store import MyTeam, load_team, save_team
from team_analysis import ROSTER_SIZE, analyze_trade, evaluate_team, grade, team_feedback

STATS = ["PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FTM", "FTA"]


def fake_players(count: int = 300) -> pd.DataFrame:
    """Players whose stats all fall with rank, with some noise so spreads are never zero."""
    random = np.random.default_rng(1)
    strength = np.linspace(1.0, 0.2, count)
    frame = pd.DataFrame(
        {
            "PLAYER_ID": np.arange(1000, 1000 + count),
            "PLAYER_NAME": [f"Player {i}" for i in range(count)],
            "RANK": np.arange(1, count + 1),
            "VALUE": np.linspace(15, -5, count),
            "PROJECTED_GP": random.integers(60, 75, count).astype(float),
        }
    )
    bases = dict(PTS=1500, FG3M=150, REB=500, AST=300, STL=80, BLK=50, TOV=150,
                 FGM=550, FGA=1200, FTM=250, FTA=320)
    for stat in STATS:
        frame[f"TOTAL_{stat}"] = bases[stat] * strength * random.uniform(0.85, 1.15, count)
    return frame


class TeamScores(unittest.TestCase):
    def setUp(self):
        self.players = fake_players()
        self.ids = self.players["PLAYER_ID"].tolist()

    def test_league_average_team_scores_zero_at_every_size(self):
        # A roster holding the whole typical pool, scaled back down, is exactly typical.
        for size in (1, 5, ROSTER_SIZE):
            pool = self.players.head(10 * size)
            typical = pool.mean(numeric_only=True).to_dict()
            typical["PLAYER_ID"] = 9999
            typical["PLAYER_NAME"] = "Typical"
            players = pd.concat([self.players] + [pd.DataFrame([typical])], ignore_index=True)
            players.loc[players["PLAYER_ID"] == 9999, "RANK"] = 10_000
            roster = [9999] * size
            report = evaluate_team(players, roster, 10)
            self.assertLess(report.categories["SCORE"].abs().max(), 0.01, size)
            self.assertEqual(report.size, size)

    def test_better_team_scores_higher(self):
        best = evaluate_team(self.players, self.ids[:13], 10)
        worst = evaluate_team(self.players, self.ids[-13:], 10)
        self.assertGreater(best.overall, worst.overall)

    def test_turnovers_count_against(self):
        players = self.players.copy()
        roster = self.ids[60:73]
        base = evaluate_team(players, roster, 10).categories.loc["TOV", "SCORE"]
        players.loc[players["PLAYER_ID"].isin(roster), "TOTAL_TOV"] += 100
        worse = evaluate_team(players, roster, 10).categories.loc["TOV", "SCORE"]
        self.assertLess(worse, base)

    def test_team_percent_comes_from_total_makes_and_attempts(self):
        roster = self.ids[:2]
        rows = self.players[self.players["PLAYER_ID"].isin(roster)]
        expected = 100 * rows["TOTAL_FGM"].sum() / rows["TOTAL_FGA"].sum()
        report = evaluate_team(self.players, roster, 10)
        self.assertAlmostEqual(report.categories.loc["FG", "TEAM"], expected)

    def test_grade_bands(self):
        self.assertEqual(grade(1.0), "Excellent")
        self.assertEqual(grade(0.0), "About average")
        self.assertEqual(grade(-1.0), "Weak")

    def test_feedback_mentions_low_games_and_partial_roster(self):
        players = self.players.copy()
        players.loc[players["PLAYER_ID"] == self.ids[0], "PROJECTED_GP"] = 40.0
        roster = self.ids[:3]
        rows = players[players["PLAYER_ID"].isin(roster)]
        notes = " ".join(team_feedback(evaluate_team(players, roster, 10), rows))
        self.assertIn("after 3 of 13", notes)
        self.assertIn("Player 0 (40)", notes)


class Trades(unittest.TestCase):
    def setUp(self):
        self.players = fake_players()
        self.ids = self.players["PLAYER_ID"].tolist()
        self.roster = self.ids[50:63]

    def test_star_for_bench_player_helps(self):
        report = analyze_trade(self.players, self.roster, [self.roster[-1]], [self.ids[0]], 10)
        self.assertGreater(report.overall_change, 0)
        self.assertEqual(report.verdict, "Helps your team")
        self.assertEqual(report.before.size, report.after.size)

    def test_reverse_trade_hurts_by_the_same_amount(self):
        forward = analyze_trade(self.players, self.roster, [self.roster[0]], [self.ids[200]], 10)
        back_roster = [p for p in self.roster if p != self.roster[0]] + [self.ids[200]]
        back = analyze_trade(self.players, back_roster, [self.ids[200]], [self.roster[0]], 10)
        self.assertAlmostEqual(forward.overall_change, -back.overall_change)
        self.assertEqual(forward.verdict, "Hurts your team")

    def test_two_for_one_adds_a_replacement(self):
        report = analyze_trade(self.players, self.roster, self.roster[:2], [self.ids[0]], 10)
        self.assertEqual(report.replacements, 1)
        self.assertTrue(report.dropped.empty)
        self.assertEqual(report.after.size, ROSTER_SIZE)

    def test_one_for_two_drops_the_lowest_value_kept_player(self):
        report = analyze_trade(self.players, self.roster, [self.roster[0]], self.ids[:2], 10)
        self.assertEqual(report.dropped["PLAYER_ID"].tolist(), [self.roster[-1]])
        self.assertEqual(report.replacements, 0)
        self.assertEqual(report.after.size, ROSTER_SIZE)

    def test_short_roster_is_compared_at_equal_size(self):
        short = self.roster[:5]
        report = analyze_trade(self.players, short, [], [self.ids[0]], 10)
        self.assertEqual(report.before.size, 6)
        self.assertEqual(report.after.size, 6)
        self.assertEqual(report.replacements, 0)

    def test_value_totals_include_drops_and_replacements(self):
        report = analyze_trade(self.players, self.roster, [self.roster[0]], self.ids[:2], 10)
        rows = self.players.set_index("PLAYER_ID")["VALUE"]
        expected = rows[self.roster[0]] + rows[self.roster[-1]]
        self.assertAlmostEqual(report.value_out, expected)


class Saving(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "league" / "my_team.json"

    def tearDown(self):
        self.folder.cleanup()

    def test_round_trip_and_default(self):
        self.assertEqual(load_team(self.path), MyTeam())
        save_team(MyTeam(12, [3, 1, 2]), self.path)
        self.assertEqual(load_team(self.path), MyTeam(12, [3, 1, 2]))

    def test_bad_values_are_corrected(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text('{"league_size": 40, "roster": [1, 1, 2]}', encoding="utf-8")
        self.assertEqual(load_team(self.path), MyTeam(16, [1, 2]))

    def test_unreadable_file_reports_a_problem(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("not json", encoding="utf-8")
        team = load_team(self.path)
        self.assertEqual(team.roster, [])
        self.assertIsNotNone(team.problem)


if __name__ == "__main__":
    unittest.main()
