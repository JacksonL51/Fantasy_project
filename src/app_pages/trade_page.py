"""Trade Analyzer page: how a trade changes the user's own team."""

import streamlit as st

from app_shared import (
    DRAFT_CHOICE,
    DRAFT_KEY,
    PER_GAME,
    SEASON_TOTALS,
    player_label,
    player_row,
    player_search,
    require_players,
    show_as_of,
    team_choice_label,
    team_choices,
    trade_table_html,
)
from league_store import load_league
from team_analysis import analyze_trade

VERDICT_STYLES = {
    "Helps your team": st.success,
    "Hurts your team": st.error,
    "About even for your team": st.info,
}
GETTING_KEY = "trade_getting"
SEARCH_KEY = "trade_search"
CHOICE_KEY = "trade_team_choice"  # the chosen team; kept across pages
CHOICE_WIDGET = "trade_team_picker"
MY_TEAM_CHOICE_KEY = "my_team_choice"


def add_incoming(player_id: int, search_key: str) -> None:
    st.session_state[GETTING_KEY] = [*st.session_state.get(GETTING_KEY, []), player_id]
    st.session_state[search_key] = ""


def remove_incoming(player_id: int) -> None:
    st.session_state[GETTING_KEY] = [
        player for player in st.session_state.get(GETTING_KEY, []) if player != player_id
    ]


def choose_team() -> None:
    st.session_state[CHOICE_KEY] = st.session_state[CHOICE_WIDGET]
    st.session_state[GETTING_KEY] = []
    st.session_state[SEARCH_KEY] = ""


def render() -> None:
    st.title("Trade Analyzer")
    players, _ = require_players()
    show_as_of(players)
    league = load_league()
    if league.problem:
        st.warning(league.problem)
    choices = team_choices(league)
    if not choices:
        st.info("Create a team on the My Team page first. Trades are measured against a team.")
        return

    # Start on the team open on the My Team page; after that, keep this page's own choice.
    choice = st.session_state.get(CHOICE_KEY)
    if choice not in choices:
        choice = st.session_state.get(MY_TEAM_CHOICE_KEY)
    if choice not in choices:
        choice = league.active_id if league.active_id in choices else choices[0]
    st.session_state[CHOICE_KEY] = choice
    st.session_state[CHOICE_WIDGET] = choice
    pick_col, _ = st.columns([1, 2])
    pick_col.selectbox(
        "Your team", choices, format_func=team_choice_label(league),
        key=CHOICE_WIDGET, on_change=choose_team,
    )
    team = st.session_state[DRAFT_KEY] if choice == DRAFT_CHOICE else league.team(choice)

    known = set(players["PLAYER_ID"])
    roster = [player for player in team.roster if player in known]
    if not roster:
        st.info(f"{team.name} has no players yet. Add some on the My Team page.")
        return

    # Someone added to the roster since they were chosen can't also be incoming.
    getting = [p for p in st.session_state.get(GETTING_KEY, []) if p in known and p not in roster]
    describe = player_label(players)
    give_col, get_col = st.columns(2, gap="large")
    with give_col:
        st.subheader("You give")
        by_rank = players.loc[players["PLAYER_ID"].isin(roster), "PLAYER_ID"].tolist()
        giving = [
            player_id
            for player_id in by_rank
            if st.checkbox(describe(player_id), key=f"trade_give_{choice}_{player_id}")
        ]
    with get_col:
        st.subheader("You get")
        for player_id in getting:
            player_row(
                describe(player_id), "Remove", f"trade_remove_{player_id}",
                remove_incoming, (player_id,),
            )
        player_search(players, [*roster, *getting], SEARCH_KEY, "Search player", add_incoming)

    if not giving and not getting:
        st.info("Choose the players on each side of the trade.")
        return

    report = analyze_trade(players, roster, giving, getting, team.league_size)

    st.divider()
    VERDICT_STYLES[report.verdict](
        f"**{report.verdict}** · Average category change {report.overall_change:+.2f} "
        f"in your {team.league_size}-team league"
    )
    out_col, in_col, net_col = st.columns(3)
    out_col.metric("Value going out", f"{report.value_out:.1f}")
    in_col.metric("Value coming in", f"{report.value_in:.1f}")
    net_col.metric(
        "Value change", f"{report.value_in - report.value_out:+.1f}",
        help="Player Value counts games only partly, and it doesn't add up exactly across "
        "players, so it can disagree with the team verdict.",
    )

    notes = []
    if report.improved:
        notes.append(f"Better at: {', '.join(report.improved)}.")
    if report.declined:
        notes.append(f"Worse at: {', '.join(report.declined)}.")
    if not report.dropped.empty:
        names = ", ".join(report.dropped["PLAYER_NAME"])
        notes.append(f"Your roster would be over the limit, so this assumes you drop {names}.")
    if report.replacements:
        notes.append(
            f"You'd have {report.replacements} open roster spot(s), so this assumes a "
            "replacement-level pickup for each."
        )
    if notes:
        st.markdown("\n".join(f"- {note}" for note in notes))

    title_col, view_col = st.columns([3, 1], vertical_alignment="bottom")
    title_col.subheader("Trade details")
    view = view_col.segmented_control(
        "Stats", [PER_GAME, SEASON_TOTALS], default=PER_GAME, required=True
    )
    st.html(
        trade_table_html(
            report.giving, report.getting, report.before, report.after, report.moves, view
        )
    )

    explanation = (
        "Average shows each side as one player: the average of their per-game lines."
        if view == PER_GAME
        else "On the Change row, blue helps your team and red hurts it, including for turnovers."
    )
    st.caption(
        f"{explanation} The verdict and the better and worse lists use season totals, so they "
        "count games played. Stats are full-season projections that include games already "
        "played and known injury absences, so a hurt player's season total is lower."
    )
