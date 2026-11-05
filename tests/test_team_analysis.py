"""Run from the project root: python -m unittest discover tests"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from league_store import MyTeam, load_team, save_team
from team_analysis import ROSTER_SIZE, analyze_trade, evaluate_team, team_feedback

STATS = ["PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FTM", "FTA"]
CATEGORIES = ["PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV", "FG", "FT"]


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
        frame[f"PROJ_{stat}"] = frame[f"TOTAL_{stat}"] / frame["PROJECTED_GP"]
    for category in CATEGORIES:
        frame[f"Z_{category}"] = random.normal(0, 1, count)
    return frame


class TeamStats(unittest.TestCase):
    def setUp(self):
        self.players = fake_players()
        self.ids = self.players["PLAYER_ID"].tolist()
        self.roster = self.ids[:3]
        self.rows = self.players[self.players["PLAYER_ID"].isin(self.roster)]

    def test_totals_and_per_game_add_up_each_players_stats(self):
        categories = evaluate_team(self.players, self.roster).categories
        self.assertAlmostEqual(categories.loc["PTS", "TOTAL"], self.rows["TOTAL_PTS"].sum())
        self.assertAlmostEqual(categories.loc["PTS", "PER_GAME"], self.rows["PROJ_PTS"].sum())

    def test_team_percent_comes_from_total_makes_and_attempts(self):
        categories = evaluate_team(self.players, self.roster).categories
        expected = 100 * self.rows["TOTAL_FGM"].sum() / self.rows["TOTAL_FGA"].sum()
        self.assertAlmostEqual(categories.loc["FG", "TOTAL"], expected)
        self.assertAlmostEqual(categories.loc["FG", "PER_GAME"], expected)

    def test_score_is_the_average_player_score(self):
        categories = evaluate_team(self.players, self.roster).categories
        self.assertAlmostEqual(categories.loc["REB", "SCORE"], self.rows["Z_REB"].mean())

    def test_feedback_names_categories_and_low_games(self):
        players = self.players.copy()
        for category in CATEGORIES:
            players[f"Z_{category}"] = 0.5
        players["Z_AST"] = 2.0
        players["Z_BLK"] = -2.0
        players.loc[players["PLAYER_ID"] == self.ids[0], "PROJECTED_GP"] = 40.0
        rows = players[players["PLAYER_ID"].isin(self.roster)]
        notes = " ".join(team_feedback(evaluate_team(players, self.roster), rows))
        self.assertIn("Strongest categories: Assists", notes)
        self.assertIn("Weakest categories: Blocks", notes)
        self.assertIn("Punting it", notes)
        self.assertIn("Player 0 (40)", notes)
        self.assertNotIn("typical", notes.lower())


class Trades(unittest.TestCase):
    def setUp(self):
        self.players = fake_players()
        self.ids = self.players["PLAYER_ID"].tolist()
        self.roster = self.ids[50:63]

    def test_star_for_bench_player_helps(self):
        report = analyze_trade(self.players, self.roster, [self.roster[-1]], [self.ids[0]], 10)
        self.assertGreater(report.overall_change, 0)
        self.assertEqual(report.verdict, "Helps your team")
        self.assertIn("Points", report.improved)

    def test_reverse_trade_hurts_by_the_same_amount(self):
        forward = analyze_trade(self.players, self.roster, [self.roster[0]], [self.ids[200]], 10)
        back_roster = [p for p in self.roster if p != self.roster[0]] + [self.ids[200]]
        back = analyze_trade(self.players, back_roster, [self.ids[200]], [self.roster[0]], 10)
        self.assertAlmostEqual(forward.overall_change, -back.overall_change)
        self.assertEqual(forward.verdict, "Hurts your team")

    def test_more_turnovers_count_against(self):
        players = self.players.copy()
        give, get = self.roster[0], self.ids[200]
        same = {f"TOTAL_{stat}": players.loc[players["PLAYER_ID"] == give, f"TOTAL_{stat}"].iloc[0]
                for stat in STATS}
        for column, value in same.items():
            players.loc[players["PLAYER_ID"] == get, column] = value
        players.loc[players["PLAYER_ID"] == get, "TOTAL_TOV"] += 100
        report = analyze_trade(players, self.roster, [give], [get], 10)
        self.assertLess(report.moves["TOV"], 0)
        self.assertEqual(report.declined, ["Turnovers"])

    def test_two_for_one_adds_a_replacement(self):
        report = analyze_trade(self.players, self.roster, self.roster[:2], [self.ids[0]], 10)
        self.assertEqual(report.replacements, 1)
        self.assertTrue(report.dropped.empty)
        pool = self.players.sort_values("RANK").iloc[130:140]
        rows = self.players.set_index("PLAYER_ID").loc[[*self.roster[2:], self.ids[0]]]
        self.assertAlmostEqual(
            report.after.loc["PTS", "PER_GAME"], rows["PROJ_PTS"].sum() + pool["PROJ_PTS"].mean()
        )

    def test_trade_reports_both_views(self):
        report = analyze_trade(self.players, self.roster, [self.roster[0]], [self.ids[0]], 10)
        rows = self.players.set_index("PLAYER_ID").loc[self.roster]
        self.assertAlmostEqual(report.before.loc["AST", "PER_GAME"], rows["PROJ_AST"].sum())
        self.assertAlmostEqual(report.before.loc["AST", "TOTAL"], rows["TOTAL_AST"].sum())
        self.assertEqual(report.before.loc["FG", "PER_GAME"], report.before.loc["FG", "TOTAL"])
        self.assertEqual(sorted(report.per_game_moves.index), sorted(report.moves.index))

    def test_per_game_and_season_moves_can_disagree(self):
        # A better per-game player who misses most of the season.
        players = self.players.copy()
        give, get = self.roster[0], self.ids[200]
        given = players["PLAYER_ID"] == give
        taken = players["PLAYER_ID"] == get
        players.loc[taken, "PROJ_PTS"] = players.loc[given, "PROJ_PTS"].iloc[0] + 5
        players.loc[taken, "TOTAL_PTS"] = players.loc[given, "TOTAL_PTS"].iloc[0] / 2
        report = analyze_trade(players, self.roster, [give], [get], 10)
        self.assertGreater(report.per_game_moves["PTS"], 0)
        self.assertLess(report.moves["PTS"], 0)

    def test_one_for_two_drops_the_lowest_value_kept_player(self):
        report = analyze_trade(self.players, self.roster, [self.roster[0]], self.ids[:2], 10)
        self.assertEqual(report.dropped["PLAYER_ID"].tolist(), [self.roster[-1]])
        self.assertEqual(report.replacements, 0)
        self.assertEqual(len(self.roster), ROSTER_SIZE)

    def test_short_roster_is_compared_at_equal_size(self):
        short = self.roster[:5]
        report = analyze_trade(self.players, short, [], [self.ids[0]], 10)
        self.assertEqual(report.replacements, 0)
        self.assertGreater(report.overall_change, 0)

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
