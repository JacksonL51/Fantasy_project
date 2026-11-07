"""My Team page: create named teams, build their rosters and get feedback on them."""

import streamlit as st

from app_shared import (
    DRAFT_KEY,
    DRAFT_CHOICE,
    PER_GAME,
    SEASON_TOTALS,
    category_table_html,
    player_label,
    player_row,
    player_search,
    require_players,
    table_html,
    team_choice_label,
    team_choices,
)
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
from team_analysis import (
    MAX_LEAGUE_SIZE,
    MIN_LEAGUE_SIZE,
    ROSTER_SIZE,
    evaluate_team,
    team_feedback,
)

CHOICE_KEY = "my_team_choice"  # the team on screen; kept across pages
CHOICE_WIDGET = "my_team_choice_picker"
NAME_WIDGET = "my_team_name"
LEAGUE_WIDGET = "my_team_league_size"
SEARCH_KEY = "my_team_search"
MESSAGE_KEY = "my_team_message"


def show_later(kind: str, text: str) -> None:
    st.session_state[MESSAGE_KEY] = (kind, text)


def change_team(change) -> None:
    """Apply a change to the team on screen. Saved teams save right away."""
    if st.session_state.get(CHOICE_KEY) == DRAFT_CHOICE:
        draft = st.session_state.get(DRAFT_KEY)
        if draft:
            change(draft)
        return
    league = load_league()
    team = league.team(st.session_state.get(CHOICE_KEY))
    if team:
        change(team)
        save_league(league)


def choose_team() -> None:
    choice = st.session_state[CHOICE_WIDGET]
    st.session_state[CHOICE_KEY] = choice
    st.session_state[SEARCH_KEY] = ""
    if choice != DRAFT_CHOICE:
        league = load_league()
        league.active_id = choice
        save_league(league)


def start_new_team() -> None:
    if DRAFT_KEY not in st.session_state:
        st.session_state[DRAFT_KEY] = Team(default_team_name(load_league().teams))
    st.session_state[CHOICE_KEY] = DRAFT_CHOICE
    st.session_state[SEARCH_KEY] = ""


def save_new_team() -> None:
    draft = st.session_state.get(DRAFT_KEY)
    if not draft:
        return
    league = load_league()
    problem = name_problem(draft.name, league.teams)
    if problem:
        show_later("error", problem)
        return
    saved = add_team(league, draft)
    save_league(league)
    del st.session_state[DRAFT_KEY]
    st.session_state[CHOICE_KEY] = saved.id
    show_later("success", f"Saved {saved.name}.")


def discard_new_team() -> None:
    st.session_state.pop(DRAFT_KEY, None)
    st.session_state.pop(CHOICE_KEY, None)


def remove_saved_team() -> None:
    league = load_league()
    team = league.team(st.session_state.get(CHOICE_KEY))
    if team:
        delete_team(league, team.id)
        save_league(league)
        show_later("success", f"Deleted {team.name}.")
    st.session_state.pop(CHOICE_KEY, None)


def rename_team() -> None:
    new_name = st.session_state[NAME_WIDGET]
    if st.session_state.get(CHOICE_KEY) == DRAFT_CHOICE:
        st.session_state[DRAFT_KEY].name = new_name
        return
    league = load_league()
    team = league.team(st.session_state.get(CHOICE_KEY))
    if not team:
        return
    problem = name_problem(new_name, league.teams, ignoring_id=team.id)
    if problem:
        show_later("error", problem)
        return
    team.name = new_name.strip()
    save_league(league)


def set_league_size() -> None:
    size = st.session_state[LEAGUE_WIDGET]
    change_team(lambda team: setattr(team, "league_size", size))


def add_player(player_id: int, search_key: str) -> None:
    def change(team: Team) -> None:
        if player_id not in team.roster and len(team.roster) < ROSTER_SIZE:
            team.roster.append(player_id)

    change_team(change)
    st.session_state[search_key] = ""


def remove_player(player_id: int) -> None:
    change_team(lambda team: team.roster.remove(player_id) if player_id in team.roster else None)


def clear_roster() -> None:
    change_team(lambda team: team.roster.clear())


def current_team(league: League) -> Team:
    """The team on screen. With nothing saved yet, it's a new unsaved team."""
    draft = st.session_state.get(DRAFT_KEY)
    if not league.teams and not draft:
        draft = st.session_state[DRAFT_KEY] = Team(default_team_name([]))
    choice = st.session_state.get(CHOICE_KEY)
    valid = [team.id for team in league.teams] + ([DRAFT_CHOICE] if draft else [])
    if choice not in valid:
        choice = league.active_id if league.active_id in valid else valid[0]
        st.session_state[CHOICE_KEY] = choice
    return draft if choice == DRAFT_CHOICE else league.team(choice)


def team_controls(league: League, team: Team) -> None:
    is_draft = team.id is None
    st.session_state[CHOICE_WIDGET] = st.session_state[CHOICE_KEY]
    st.session_state[NAME_WIDGET] = team.name
    st.session_state[LEAGUE_WIDGET] = team.league_size

    pick_col, name_col, size_col = st.columns([3, 3, 2])
    pick_col.selectbox(
        "Team", team_choices(league), format_func=team_choice_label(league),
        key=CHOICE_WIDGET, on_change=choose_team,
    )
    name_col.text_input(
        "Team name", key=NAME_WIDGET, on_change=rename_team, max_chars=40,
    )
    size_col.number_input(
        "Teams in league", MIN_LEAGUE_SIZE, MAX_LEAGUE_SIZE, step=1,
        key=LEAGUE_WIDGET, on_change=set_league_size,
        help="Used by the Trade Analyzer to judge trades and replacement-level pickups.",
    )

    buttons = st.container(horizontal=True)
    if is_draft:
        buttons.button("Save team", type="primary", on_click=save_new_team)
        buttons.button("Discard", on_click=discard_new_team)
        st.caption(
            "This team isn't saved yet. It'll be lost if you discard it, reload the page or "
            "close the app."
        )
    else:
        buttons.button("New team", on_click=start_new_team)
        with buttons.popover("Delete team"):
            st.write(f"Delete {team.name}? This can't be undone.")
            st.button("Yes, delete it", type="primary", on_click=remove_saved_team)
        st.caption("Changes to this team save automatically.")


def render() -> None:
    st.title("My Team")
    players, _ = require_players()
    league = load_league()
    if league.problem:
        st.warning(league.problem)
    message = st.session_state.pop(MESSAGE_KEY, None)
    if message:
        (st.error if message[0] == "error" else st.success)(message[1])

    team = current_team(league)
    team_controls(league, team)

    known = set(players["PLAYER_ID"])
    roster = [player for player in team.roster if player in known]
    missing = len(team.roster) - len(roster)
    if missing:
        st.warning(
            f"{missing} saved player(s) are no longer in the projections, so they aren't "
            "counted here. Your saved team still lists them."
        )

    add_col, roster_col = st.columns(2, gap="large")
    with add_col:
        st.subheader("Add players")
        player_search(
            players, roster, SEARCH_KEY, "Search player", add_player,
            disabled=len(roster) >= ROSTER_SIZE,
        )
    with roster_col:
        st.subheader(f"Roster ({len(roster)} of {ROSTER_SIZE})")
        if not roster:
            st.caption("Add your first player to see team feedback.")
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
            "- **Per game:** your team as one player, the average of your players' per-game "
            "lines.\n"
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
