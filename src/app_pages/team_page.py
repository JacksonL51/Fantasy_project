"""My Team page: build a roster, during a draft or after it, and get feedback on it."""

import streamlit as st

from app_shared import (
    PER_GAME,
    SEASON_TOTALS,
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
    size_col.number_input(
        "Teams in league", MIN_LEAGUE_SIZE, MAX_LEAGUE_SIZE, step=1,
        key=LEAGUE_KEY, on_change=save_league_size,
        help="Used by the Trade Analyzer to judge trades and replacement-level pickups.",
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

    report = evaluate_team(players, roster)
    roster_rows = players[players["PLAYER_ID"].isin(roster)].sort_values("RANK")

    st.divider()
    st.subheader("Feedback")
    st.markdown("\n".join(f"- {note}" for note in team_feedback(report, roster_rows)))

    view = st.segmented_control(
        "Stats", [PER_GAME, SEASON_TOTALS], default=PER_GAME, required=True
    )

    st.subheader("Categories")
    st.html(category_table_html(report.categories, view))

    st.subheader("Roster stats")
    st.html(table_html(roster_rows, view))

    with st.expander("How this works"):
        st.markdown(
            "- **Per game:** adds up each player's per-game projection, as if everyone plays "
            "that night.\n"
            "- **Season totals:** adds up each player's projected season, so games played "
            "count in full.\n"
            "- **FG% and FT%:** come from the team's total makes and attempts.\n"
            "- **Average player score:** the average of your players' category scores from the "
            "rankings, compared with players projected for 500+ minutes. Above 0 is better, "
            "including for turnovers. It isn't a comparison with other fantasy teams.\n"
            "- **Strongest and weakest categories:** ranked against each other within your "
            "team.\n"
            "- **Stats:** projections cover a full season and don't know about current injuries."
        )
