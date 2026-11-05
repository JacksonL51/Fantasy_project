"""My Team page: build a roster, during a draft or after it, and get feedback on it."""

import streamlit as st

from app_shared import (
    PER_GAME,
    category_table_html,
    player_label,
    player_row,
    player_search,
    require_players,
    table_html,
)
from league_store import MyTeam, load_team, save_team
from team_analysis import (
    MAX_LEAGUE_SIZE,
    MIN_LEAGUE_SIZE,
    ROSTER_SIZE,
    evaluate_team,
    team_feedback,
)

LEAGUE_KEY = "my_team_league_size"
SEARCH_KEY = "my_team_search"


def save_league_size() -> None:
    team = load_team()
    save_team(MyTeam(st.session_state[LEAGUE_KEY], team.roster))


def add_player(player_id: int, search_key: str) -> None:
    team = load_team()
    if player_id not in team.roster and len(team.roster) < ROSTER_SIZE:
        save_team(MyTeam(team.league_size, [*team.roster, player_id]))
    st.session_state[search_key] = ""


def remove_player(player_id: int) -> None:
    team = load_team()
    save_team(MyTeam(team.league_size, [p for p in team.roster if p != player_id]))


def clear_roster() -> None:
    save_team(MyTeam(load_team().league_size, []))


def render() -> None:
    st.title("My Team")
    players, _ = require_players()
    team = load_team()  # The saved file is the source of truth, so other pages stay in step.
    if team.problem:
        st.warning(team.problem)

    known = set(players["PLAYER_ID"])
    roster = [player for player in team.roster if player in known]
    missing = len(team.roster) - len(roster)
    if missing:
        st.warning(
            f"{missing} saved player(s) are no longer in the projections, so they aren't "
            "counted here. Your saved team still lists them."
        )

    st.session_state[LEAGUE_KEY] = team.league_size
    size_col, _ = st.columns([1, 3])
    league_size = size_col.number_input(
        "Teams in league", MIN_LEAGUE_SIZE, MAX_LEAGUE_SIZE, step=1,
        key=LEAGUE_KEY, on_change=save_league_size,
    )

    add_col, roster_col = st.columns(2, gap="large")
    with add_col:
        st.subheader("Add players")
        player_search(
            players, roster, SEARCH_KEY, "Search player", add_player,
            disabled=len(roster) >= ROSTER_SIZE,
        )
    with roster_col:
        st.subheader(f"Your roster ({len(roster)} of {ROSTER_SIZE})")
        if not roster:
            st.caption("Add your first player to see team feedback. Your team saves automatically.")
        describe = player_label(players)
        by_rank = players.loc[players["PLAYER_ID"].isin(roster), "PLAYER_ID"].tolist()
        for player_id in by_rank:
            player_row(
                describe(player_id), "Remove", f"my_team_remove_{player_id}",
                remove_player, (player_id,),
            )
        if roster:
            st.button("Clear roster", on_click=clear_roster)

    if not roster:
        return

    report = evaluate_team(players, roster, league_size)
    roster_rows = players[players["PLAYER_ID"].isin(roster)].sort_values("RANK")

    st.divider()
    grade_col, score_col, games_col = st.columns(3)
    grade_col.metric("Team grade", report.grade)
    score_col.metric(
        "Overall score", f"{report.overall:+.2f}",
        help="Average of the nine category scores. 0 is a typical team in your league.",
    )
    games_col.metric(
        "Average projected games", f"{report.average_gp:.0f}",
        delta=f"{report.average_gp - report.typical_gp:+.0f} vs typical",
    )

    st.subheader("Feedback")
    st.markdown("\n".join(f"- {note}" for note in team_feedback(report, roster_rows)))

    st.subheader("Categories")
    st.html(category_table_html(report.categories, "Typical team"))

    st.subheader("Roster stats")
    st.html(table_html(roster_rows, PER_GAME))

    with st.expander("How this works"):
        st.markdown(
            f"- **Typical team:** in a {league_size}-team league after {report.size} player(s) "
            f"each, the league holds about the top {league_size * report.size} players. A "
            "typical team has the average of that group, so it stays fair during a draft "
            "without tracking other teams.\n"
            "- **Score:** how far you are from typical, sized by how much teams normally "
            "differ. Above 0 is better, including for turnovers.\n"
            "- **Season totals:** games played count in full here, so a team score can "
            "disagree with player Value, which counts games only partly.\n"
            "- **Grades and ratings:** these bands are judgment calls and don't predict "
            "league results.\n"
            "- **Stats:** projections cover a full season and don't know about current injuries."
        )
