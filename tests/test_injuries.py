"""Offline tests for src/injuries.py (no network, nothing written to data/)."""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import injuries as inj  # noqa: E402

ROSTER = pd.DataFrame(
    {
        "PLAYER_ID": [1, 2, 3, 4],
        "PLAYER_NAME": ["Dereck Lively II", "Nikola Jokić", "Jalen Williams", "Jaylin Williams"],
        "TEAM_ABBREVIATION": ["DAL", "DEN", "OKC", "OKC"],
        "TEAM_NAME": ["Mavericks", "Nuggets", "Thunder", "Thunder"],
    }
)


def _feed(entries):
    return {"injuries": [{"displayName": team, "injuries": rows} for team, rows in entries.items()]}


def _entry(name, status, fantasy, return_date):
    return {
        "id": name,
        "status": status,
        "date": "2026-11-10T12:00Z",
        "athlete": {"displayName": name},
        "details": {"fantasyStatus": {"abbreviation": fantasy}, "returnDate": return_date, "type": "Knee"},
        "shortComment": "note",
    }


class InjuryTests(unittest.TestCase):
    def test_normalize_name(self):
        self.assertEqual(inj.normalize_name("Dereck Lively II"), "dereck lively")
        self.assertEqual(inj.normalize_name("Nikola Jokić"), "nikola jokic")
        self.assertEqual(inj.normalize_name("T.J. McConnell"), "tj mcconnell")

    def test_parse_status_prefers_fantasy_flag(self):
        feed = _feed(
            {
                "Dallas Mavericks": [_entry("Dereck Lively II", "Out", "GTD", "2026-11-12")],
                "Denver Nuggets": [_entry("Nikola Jokic", "Out", "OFS", "2027-07-01")],
                "Oklahoma City Thunder": [_entry("Jalen Williams", "Day-To-Day", "", None)],
            }
        )
        df = inj.parse_espn_feed(feed, "2026-11-11")
        self.assertEqual(df["STATUS"].tolist(), ["GTD", "OFS", "GTD"])

    def test_match_does_not_confuse_similar_names(self):
        ids = inj.match_players(
            pd.Series(["Jalen Williams", "Jaylin Williams", "Nikola Jokic"]),
            pd.Series(["OKC", "OKC", "DEN"]),
            ROSTER,
        )
        self.assertEqual(ids.tolist(), [3, 4, 2])

    def test_overrides_win_and_active_removes(self):
        espn = inj.parse_espn_feed(
            _feed(
                {
                    "Dallas Mavericks": [_entry("Dereck Lively II", "Out", "OUT", "2026-12-01")],
                    "Oklahoma City Thunder": [_entry("Jalen Williams", "Out", "OUT", "2027-03-02")],
                }
            ),
            "2026-11-11",
        )
        overrides = pd.DataFrame(
            {
                "PLAYER_ID": pd.array([pd.NA, 3], dtype="Int64"),
                "PLAYER_NAME": ["Dereck Lively", None],
                "STATUS": ["ACTIVE", "OFS"],
                "RETURN_DATE": [None, None],
                "NOTE": [None, "season over"],
            }
        )
        out = inj.build_current_injuries(ROSTER, espn, overrides, write=False)
        self.assertEqual(out["PLAYER_ID"].tolist(), [3])
        self.assertEqual(out["STATUS"].iloc[0], "OFS")
        self.assertEqual(out["SOURCE"].iloc[0], "override")


class InjuryGamesTests(unittest.TestCase):
    def setUp(self):
        from datetime import date, timedelta

        self.today = date(2026, 11, 11)
        days = [self.today + timedelta(days=2 * n) for n in range(40)]
        self.schedule = pd.DataFrame(
            {
                "GAME_ID": [f"002{n:05d}" for n in range(40)],
                "GAME_DATE": days,
                "HOME_TEAM": ["OKC"] * 40,
                "AWAY_TEAM": ["DAL"] * 40,
            }
        )
        past = [self.today - timedelta(days=d) for d in (1, 3, 5)]
        self.log = pd.DataFrame(
            {
                "PLAYER_ID": [3, 3, 3, 4, 4],
                "TEAM_ABBREVIATION": ["OKC"] * 5,
                "GAME_ID": ["p1", "p2", "p3", "p1", "p2"],
                "GAME_DATE": [past[0], past[1], past[2], past[0], past[1]],
            }
        )

    def _injuries(self, rows):
        return pd.DataFrame(rows, columns=["PLAYER_ID", "STATUS", "RETURN_DATE"])

    def test_games_out_and_out_for_season(self):
        from datetime import timedelta

        injuries = self._injuries(
            [
                (3, "OUT", self.today + timedelta(days=10)),  # games on days 0,2,4,6,8 -> 5
                (4, "OUT", self.today + timedelta(days=400)),  # past season end -> out for season
                (2, "GTD", None),
            ]
        )
        games = inj.injury_games(ROSTER, injuries, self.schedule, self.log, self.today).set_index("PLAYER_ID")
        self.assertEqual(games.at[3, "GAMES_PLAYED"], 3)
        self.assertEqual(games.at[3, "TEAM_GAMES_LEFT"], 79)
        self.assertEqual(games.at[3, "GAMES_OUT_AHEAD"], 5)
        self.assertFalse(games.at[3, "OUT_FOR_SEASON"])
        self.assertTrue(games.at[4, "OUT_FOR_SEASON"])
        self.assertEqual(games.at[2, "GAMES_OUT_AHEAD"], inj.GTD_GAMES_OUT)
        self.assertEqual(games.at[1, "GAMES_OUT_AHEAD"], 0)

    def test_adjusted_games(self):
        games = pd.DataFrame({"GAMES_PLAYED": [3, 0], "TEAM_GAMES_LEFT": [79, 79], "GAMES_OUT_AHEAD": [5, 79]})
        new = inj.adjusted_games(pd.Series([41.0, 70.0]), games)
        self.assertAlmostEqual(new.iloc[0], 3 + 0.5 * 74)
        self.assertEqual(new.iloc[1], 0)


class PreseasonTests(unittest.TestCase):
    def setUp(self):
        from datetime import date, timedelta

        self.today = date(2026, 11, 11)
        opening = date(2026, 10, 20)
        dates = [opening + timedelta(days=2 * n) for n in range(10)]
        rows = []
        for n, d in enumerate(dates):
            rows.append((1, d, f"g{n}"))  # Lively: played every game
            if n >= 4:
                rows.append((3, d, f"g{n}"))  # Jalen Williams: missed the first 4, then played
            if n >= 1:
                rows.append((2, d, f"g{n}"))  # Jokic: missed only the opener
        self.log = pd.DataFrame(rows, columns=["PLAYER_ID", "GAME_DATE", "GAME_ID"])
        self.log["TEAM_ABBREVIATION"] = self.log["PLAYER_ID"].map({1: "DAL", 2: "DAL", 3: "DAL", 4: "DAL"})
        self.roster = ROSTER.assign(TEAM_ABBREVIATION="DAL")
        # Jaylin Williams (4) never played: today 6 games out ahead.
        self.current = pd.DataFrame(
            {"PLAYER_ID": [1, 2, 3, 4], "GAMES_OUT_AHEAD": [0.0, 0.0, 0.0, 6.0],
             "OUT_FOR_SEASON": [False] * 4, "RETURN_DATE": [None, None, None, None]}
        )
        self.empty = pd.DataFrame(columns=inj.PRESEASON_OVERRIDE_COLUMNS)

    def test_streaks_and_still_out(self):
        g = inj.preseason_injury_games(self.roster, self.current, self.log, self.empty, self.today).set_index("PLAYER_ID")
        self.assertTrue(g["INJURY_STATUS"].isna()[[1, 2]].all())  # opener only is below the 3-game line
        self.assertEqual(g.at[3, "GAMES_OUT_AHEAD"], 4)
        self.assertEqual(g.at[4, "GAMES_OUT_AHEAD"], 10 + 6)
        self.assertTrue((g["TEAM_GAMES_LEFT"] == 82).all() and (g["GAMES_PLAYED"] == 0).all())

    def test_overrides(self):
        overrides = pd.DataFrame(
            {"PLAYER_ID": pd.array([3, 2], dtype="Int64"), "PLAYER_NAME": [None, None],
             "STATUS": ["ACTIVE", "OUT"], "GAMES_OUT": [None, 20.0], "NOTE": [None, None]}
        )
        g = inj.preseason_injury_games(self.roster, self.current, self.log, overrides, self.today).set_index("PLAYER_ID")
        self.assertTrue(pd.isna(g.at[3, "INJURY_STATUS"]))
        self.assertEqual(g.at[3, "GAMES_OUT_AHEAD"], 0)
        self.assertEqual(g.at[2, "GAMES_OUT_AHEAD"], 20)


class OpportunityTests(unittest.TestCase):
    def test_absence_moves_minutes_and_rates_to_teammates(self):
        import opportunity as opp

        model = {
            "active": True, "band_edges": [0, 10, 20, 28, 34, 60],
            "band_coef": {0: 0.05, 1: 0.04, 2: 0.02, 3: 0.01, 4: 0.0},
            "same_slot_beta": 2.0, "regular_mpg": 20.0,
            "rate_coef_per_10_volume": {"PTS": 0.01, "FGA": 0.01, "FTA": 0.02, "AST": 0.01},
        }
        frame = pd.DataFrame(
            {"PLAYER_ID": [10, 11, 12], "TEAM_ABBREVIATION": ["A"] * 3, "PROJECTED_MPG": [34.0, 15.0, 36.0]}
        )
        rates = pd.DataFrame({s: [0.5, 0.4, 0.5] for s in ["PTS", "FGA", "FGM", "FTA", "FTM", "AST", "TOV"]})
        positions = pd.Series({10: "G", 11: "G", 12: "C"})
        gain, new_rates = opp.apply_opportunity(
            frame, rates, pd.Series([70.0, 60.0, 70.0]), pd.Series([29.0, 60.0, 70.0]), model, positions
        )
        # Bench guard: 0.04 * (1 + 2 * 1) * 34 * 41 / 82 = 2.04; 36-MPG center gets nothing.
        self.assertAlmostEqual(gain.iloc[1], 0.04 * 3 * 34 * 41 / 82)
        self.assertEqual(gain.iloc[2], 0.0)
        self.assertEqual(gain.iloc[0], 0.0)
        self.assertGreater(new_rates.at[1, "PTS"], rates.at[1, "PTS"])
        self.assertEqual(new_rates.at[1, "TOV"], rates.at[1, "TOV"])
        self.assertAlmostEqual(new_rates.at[1, "FGM"] / new_rates.at[1, "FGA"], 1.0)


if __name__ == "__main__":
    unittest.main()
