"""Rookies page: every rostered rookie's projection, plus draft and college details."""

import pandas as pd
import streamlit as st

from app_pages.rankings_page import (
    SORTABLE_TABLE,
    TINT_LEGEND,
    filter_and_sort,
    next_sort,
    sort_column,
)
from app_shared import (
    PER_GAME,
    SEASON_TOTALS,
    height_text,
    is_rookie,
    require_players,
    rookie_details,
    table_html,
)

SORT_STATE_KEY = "rookie_table_sort"
TABLE_KEY = "rookie_table"
DEFAULT_SORT = "VALUE"
DRAFT_FILTERS = ("All", "Drafted", "Undrafted")


def on_header_click() -> None:
    clicked = st.session_state[TABLE_KEY].get("sort")
    if clicked:
        current_key, descending = st.session_state.get(SORT_STATE_KEY, (DEFAULT_SORT, True))
        st.session_state[SORT_STATE_KEY] = next_sort(current_key, descending, clicked)


def detail_table(shown: pd.DataFrame) -> pd.DataFrame:
    """Draft and college columns for the rookies shown, in the same order."""
    details = shown[["RANK", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "DRAFT_PICK"]].merge(
        rookie_details(), on="PLAYER_ID", how="left"
    )
    college_season = details["COLLEGE_YEAR"].map(
        lambda year: f"{int(year) - 1}-{int(year) % 100:02d}" if pd.notna(year) else ""
    )
    return pd.DataFrame(
        {
            "Rank": details["RANK"],
            "Player": details["PLAYER_NAME"],
            "Team": details["TEAM_ABBREVIATION"],
            "Pick": details["DRAFT_PICK"],
            "Age": details["AGE"],
            "Pos": details["POSITION"].fillna(""),
            "Height": details["HEIGHT_IN"].map(height_text),
            "College": details["TEAM"].fillna("No college data"),
            "Final season": college_season,
            "Class": details["CLASS"].fillna(""),
            "PTS": details["PTS"],
            "REB": details["TREB"],
            "AST": details["AST"],
            "STL": details["STL"],
            "BLK": details["BLK"],
            "BPM": details["BPM"],
        }
    )


@st.fragment
def rookie_tables(rookies: pd.DataFrame, show_colors: bool) -> None:
    search_col, team_col, draft_col, view_col = st.columns([4, 3, 3, 3])
    search = search_col.text_input("Search rookie", type="search", placeholder="Player name", live=True)
    teams = team_col.multiselect(
        "Team", sorted(rookies["TEAM_ABBREVIATION"].unique()), placeholder="All teams"
    )
    draft_filter = draft_col.segmented_control("Draft", DRAFT_FILTERS, default="All", required=True)
    view = view_col.segmented_control("Stats", [PER_GAME, SEASON_TOTALS], default=PER_GAME, required=True)

    pool = rookies
    if draft_filter == "Drafted":
        pool = pool[pool["DRAFT_PICK"].notna()]
    elif draft_filter == "Undrafted":
        pool = pool[pool["DRAFT_PICK"].isna()]
    sort_key, descending = st.session_state.get(SORT_STATE_KEY, (DEFAULT_SORT, True))
    shown = filter_and_sort(pool, search, teams, sort_column(sort_key, view), descending)

    st.caption(
        f"Showing {len(shown)} of {len(rookies)} rookies · # is the overall fantasy rank · "
        "Click a column header to sort"
    )
    SORTABLE_TABLE(
        key=TABLE_KEY,
        data={
            "html": table_html(
                shown, view, sort_key, descending,
                fixed_height=True, show_colors=show_colors, sortable=True,
            ),
            "rows_key": f"{search}|{teams}|{draft_filter}|{view}|{sort_key}|{descending}",
        },
        on_sort_change=on_header_click,
    )

    st.subheader("Draft and college")
    st.caption("Final college season per game (Bart Torvik). BPM is box plus-minus.")
    st.dataframe(
        detail_table(shown),
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn(format="%d"),
            "Pick": st.column_config.NumberColumn(format="%d"),
            "Age": st.column_config.NumberColumn(format="%d"),
            **{
                stat: st.column_config.NumberColumn(format="%.1f")
                for stat in ("PTS", "REB", "AST", "STL", "BLK", "BPM")
            },
        },
    )


def render() -> None:
    st.title("Rookies")
    players, modified = require_players()
    rookies = players[is_rookie(players)]
    if rookies.empty:
        st.info("No rookies in the current projections.")
        return
    drafted = int(rookies["DRAFT_PICK"].notna().sum())
    st.caption(
        f"{len(rookies)} rostered rookies ({drafted} drafted, {len(rookies) - drafted} undrafted) · "
        f"Projections updated {modified:%b %d, %Y}"
    )

    toggle_col, legend_col = st.columns([1, 4], vertical_alignment="center")
    show_colors = toggle_col.toggle("Category colors", value=True, key="rookie_colors")
    if show_colors:
        legend_col.html(TINT_LEGEND)
    rookie_tables(rookies, show_colors)

    with st.expander("How rookie projections are made"):
        st.markdown(
            "- **Per-minute stats:** draft pick and age, plus the player's final college season "
            "(assist, steal, block and rebound rates, shooting, usage) and listed height. "
            "Players without college data (international, G League) lean on pick, age and height.\n"
            "- **Minutes and games:** draft pick and age, plus how much room the team has: how "
            "much of last season's playing time left the roster, and how many minutes the "
            "returning players take.\n"
            "- **Colors:** compared with every player projected for 500+ minutes, not just rookies.\n"
            "- **This season's games so far are not used yet.**"
        )
