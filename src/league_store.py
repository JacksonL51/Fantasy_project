"""Saves the user's league size and roster to a local file, so they survive app restarts.

The file is personal and git-ignored. The pipeline never reads or overwrites it.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from team_analysis import DEFAULT_LEAGUE_SIZE, MAX_LEAGUE_SIZE, MIN_LEAGUE_SIZE, ROSTER_SIZE

TEAM_PATH = Path(__file__).resolve().parent.parent / "data" / "league" / "my_team.json"


@dataclass
class MyTeam:
    league_size: int = DEFAULT_LEAGUE_SIZE
    roster: list[int] = field(default_factory=list)
    problem: str | None = None  # set when a saved file could not be read


def load_team(path: Path = TEAM_PATH) -> MyTeam:
    if not path.exists():
        return MyTeam()
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        league_size = min(max(int(saved["league_size"]), MIN_LEAGUE_SIZE), MAX_LEAGUE_SIZE)
        roster = list(dict.fromkeys(int(player) for player in saved["roster"]))[:ROSTER_SIZE]
    except (OSError, ValueError, KeyError, TypeError):
        return MyTeam(problem=f"Could not read the saved team at {path}. Starting empty.")
    return MyTeam(league_size=league_size, roster=roster)


def save_team(team: MyTeam, path: Path = TEAM_PATH) -> None:
    """Write through a temporary file, so a crash cannot leave a half-written team."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"league_size": team.league_size, "roster": team.roster}, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
