"""Saves the user's named fantasy teams to a local file, so they survive app restarts.

The file is personal and git-ignored. The pipeline never reads or overwrites it. A team saved
by the older one-team version (my_team.json) is brought in once as Team1, and that older
file is left in place.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from team_analysis import DEFAULT_LEAGUE_SIZE, MAX_LEAGUE_SIZE, MIN_LEAGUE_SIZE, ROSTER_SIZE

LEAGUE_DIR = Path(__file__).resolve().parent.parent / "data" / "league"
TEAMS_PATH = LEAGUE_DIR / "teams.json"
LEGACY_TEAM_PATH = LEAGUE_DIR / "my_team.json"
MAX_NAME_LENGTH = 40


@dataclass
class Team:
    name: str
    league_size: int = DEFAULT_LEAGUE_SIZE
    roster: list[int] = field(default_factory=list)
    id: int | None = None  # None until the team is saved


@dataclass
class League:
    teams: list[Team] = field(default_factory=list)
    active_id: int | None = None
    next_id: int = 1
    problem: str | None = None  # set when a saved file could not be read

    def team(self, team_id: int | None) -> Team | None:
        return next((team for team in self.teams if team.id == team_id), None)


def _clean_team(saved: dict, team_id: int | None) -> Team:
    return Team(
        name=str(saved.get("name", "")).strip()[:MAX_NAME_LENGTH],
        league_size=min(max(int(saved["league_size"]), MIN_LEAGUE_SIZE), MAX_LEAGUE_SIZE),
        roster=list(dict.fromkeys(int(player) for player in saved["roster"]))[:ROSTER_SIZE],
        id=team_id,
    )


def default_team_name(teams: list[Team]) -> str:
    """The first free name of Team1, Team2 and so on."""
    taken = {team.name.casefold() for team in teams}
    number = 1
    while f"team{number}" in taken:
        number += 1
    return f"Team{number}"


def name_problem(name: str, teams: list[Team], ignoring_id: int | None = None) -> str | None:
    """Why a team can't use this name, or None if it can."""
    cleaned = name.strip()
    if not cleaned:
        return "Team names can't be blank."
    if len(cleaned) > MAX_NAME_LENGTH:
        return f"Team names can be up to {MAX_NAME_LENGTH} characters."
    if any(t.id != ignoring_id and t.name.casefold() == cleaned.casefold() for t in teams):
        return f'You already have a team called "{cleaned}".'
    return None


def _load_legacy(path: Path) -> League:
    """The older single saved team, if there is one, as Team1."""
    if not path.exists():
        return League()
    try:
        team = _clean_team(json.loads(path.read_text(encoding="utf-8")), 1)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return League(problem=f"Could not read the older saved team at {path}.")
    team.name = "Team1"
    return League(teams=[team], active_id=1, next_id=2)


def load_league(path: Path = TEAMS_PATH, legacy_path: Path = LEGACY_TEAM_PATH) -> League:
    if not path.exists():
        return _load_legacy(legacy_path)
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        teams = [_clean_team(team, int(team["id"])) for team in saved["teams"]]
        ids = [team.id for team in teams]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate team IDs.")
        for team in teams:
            if not team.name:
                team.name = default_team_name(teams)
        active = saved.get("active_id")
        return League(
            teams=teams,
            active_id=active if active in ids else (ids[0] if ids else None),
            next_id=max([int(saved.get("next_id", 1)), *[i + 1 for i in ids]]),
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # Keep the unreadable file, so saving a new team can't destroy what's in it.
        kept = path.with_name(f"{path.stem}.unreadable{path.suffix}")
        os.replace(path, kept)
        return League(problem=f"Could not read your saved teams. The file was kept at {kept}.")


def save_league(league: League, path: Path = TEAMS_PATH) -> None:
    """Write through a temporary file, so a crash cannot leave half-written teams."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "active_id": league.active_id,
                "next_id": league.next_id,
                "teams": [
                    {"id": team.id, "name": team.name, "league_size": team.league_size,
                     "roster": team.roster}
                    for team in league.teams
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def add_team(league: League, team: Team) -> Team:
    """Give a new team an ID, add it to the league and make it the active team."""
    team.id = league.next_id
    team.name = team.name.strip()
    league.next_id += 1
    league.teams.append(team)
    league.active_id = team.id
    return team


def delete_team(league: League, team_id: int) -> None:
    league.teams = [team for team in league.teams if team.id != team_id]
    if league.active_id == team_id:
        league.active_id = league.teams[0].id if league.teams else None
