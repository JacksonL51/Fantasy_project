"""Run from the project root: python -m unittest discover tests"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from league_store import (
    League,
    Team,
    add_team,
    default_team_name,
    delete_team,
    load_league,
    name_problem,
    save_league,
)
from team_analysis import ROSTER_SIZE, analyze_trade, average_player, evaluate_team, team_feedback

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
            "PROJECTED_MPG": random.uniform(15, 36, count),
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

    def test_totals_add_up_and_per_game_is_the_team_as_one_player(self):
        categories = evaluate_team(self.players, self.roster).categories
        self.assertAlmostEqual(categories.loc["PTS", "TOTAL"], self.rows["TOTAL_PTS"].sum())
        self.assertAlmostEqual(categories.loc["PTS", "PER_GAME"], self.rows["PROJ_PTS"].mean())

    def test_average_player_weights_percentages_by_volume(self):
        average = average_player(self.rows)
        self.assertAlmostEqual(average["PROJ_REB"], self.rows["PROJ_REB"].mean())
        self.assertAlmostEqual(average["PROJECTED_GP"], self.rows["PROJECTED_GP"].mean())
        self.assertAlmostEqual(
            average["PROJ_FT_PCT"], 100 * self.rows["TOTAL_FTM"].sum() / self.rows["TOTAL_FTA"].sum()
        )

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
        self.assertTrue(report.improved)

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

    def test_trade_reports_team_season_totals(self):
        report = analyze_trade(self.players, self.roster, [self.roster[0]], [self.ids[0]], 10)
        rows = self.players.set_index("PLAYER_ID").loc[self.roster]
        self.assertAlmostEqual(report.before.loc["AST", "TOTAL"], rows["TOTAL_AST"].sum())

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
        folder = Path(self.folder.name) / "league"
        self.path = folder / "teams.json"
        self.legacy = folder / "my_team.json"
        folder.mkdir()

    def tearDown(self):
        self.folder.cleanup()

    def load(self) -> League:
        return load_league(self.path, self.legacy)

    def test_starts_empty(self):
        self.assertEqual(self.load(), League())

    def test_teams_round_trip_with_names_and_their_own_rosters(self):
        league = self.load()
        first = add_team(league, Team("Team1", 12, [3, 1, 2]))
        second = add_team(league, Team("  Dynasty  ", 8, [4]))
        save_league(league, self.path)
        loaded = self.load()
        self.assertEqual([t.name for t in loaded.teams], ["Team1", "Dynasty"])
        self.assertEqual(loaded.team(first.id).roster, [3, 1, 2])
        self.assertEqual(loaded.team(second.id).league_size, 8)
        self.assertEqual(loaded.active_id, second.id)

    def test_older_saved_team_becomes_team1_and_is_left_in_place(self):
        self.legacy.write_text('{"league_size": 12, "roster": [7, 8]}', encoding="utf-8")
        league = self.load()
        self.assertEqual(league.teams, [Team("Team1", 12, [7, 8], 1)])
        self.assertEqual(league.active_id, 1)
        self.assertEqual(add_team(league, Team("Team2")).id, 2)
        save_league(league, self.path)
        self.assertTrue(self.legacy.exists())
        self.assertEqual(len(self.load().teams), 2)

    def test_deleted_older_team_does_not_come_back(self):
        self.legacy.write_text('{"league_size": 12, "roster": [7]}', encoding="utf-8")
        league = self.load()
        delete_team(league, 1)
        save_league(league, self.path)
        self.assertEqual(self.load().teams, [])
        self.assertIsNone(self.load().active_id)

    def test_deleted_team_ids_are_not_reused(self):
        league = self.load()
        first = add_team(league, Team("A"))
        delete_team(league, first.id)
        self.assertNotEqual(add_team(league, Team("B")).id, first.id)

    def test_bad_values_are_corrected(self):
        self.path.write_text(
            '{"teams": [{"id": 3, "name": "A", "league_size": 40, "roster": [1, 1, 2]}]}',
            encoding="utf-8",
        )
        league = self.load()
        self.assertEqual(league.teams, [Team("A", 16, [1, 2], 3)])
        self.assertEqual(league.next_id, 4)
        self.assertEqual(league.active_id, 3)

    def test_unreadable_file_is_kept_and_reported(self):
        self.path.write_text("not json", encoding="utf-8")
        league = self.load()
        self.assertEqual(league.teams, [])
        self.assertIsNotNone(league.problem)
        kept = self.path.with_name("teams.unreadable.json")
        self.assertEqual(kept.read_text(encoding="utf-8"), "not json")

    def test_default_names_use_the_first_free_number(self):
        self.assertEqual(default_team_name([]), "Team1")
        teams = [Team("Team1"), Team("team3")]
        self.assertEqual(default_team_name(teams), "Team2")
        self.assertEqual(default_team_name([*teams, Team("TEAM2")]), "Team4")

    def test_name_rules(self):
        teams = [Team("Dynasty", id=1), Team("Team2", id=2)]
        self.assertIsNone(name_problem("Keepers", teams))
        self.assertIn("blank", name_problem("   ", teams))
        self.assertIn("already", name_problem(" dynasty ", teams))
        self.assertIsNone(name_problem("Dynasty", teams, ignoring_id=1))
        self.assertIn("up to", name_problem("x" * 41, teams))


if __name__ == "__main__":
    unittest.main()
