"""Rankings page: the top 10 and the searchable, sortable full rankings."""

import pandas as pd
import streamlit as st

from app_shared import (
    PER_GAME,
    SEASON_TOTALS,
    STAT_LABELS,
    PRESEASON,
    plain_text,
    projection_view_control,
    require_players,
    show_as_of,
    stat_prefix,
    table_html,
)

# Sort keys that are not stats. The first click on these sorts low to high (A-Z, rank 1
# first); the first click on any stat sorts high to low.
SORT_COLUMNS = {"RANK": "RANK", "PLAYER": "SEARCH_NAME", "TEAM": "TEAM_ABBREVIATION"}
DEFAULT_SORT = "VALUE"
SORT_STATE_KEY = "table_sort"
TABLE_KEY = "full_rankings_table"

# Draws the table HTML from Python and reports header clicks back. The scroll position is
# kept when only the colors change, and returns to the top when the rows change.
SORTABLE_TABLE_JS = """
export default function ({ data, parentElement, setTriggerValue }) {
  const slot = parentElement.querySelector(".table-slot");
  const previous = slot.querySelector(".stat-table");
  const scrollTop = previous ? previous.scrollTop : 0;
  const sameRows = slot.dataset.rowsKey === data.rows_key;
  slot.innerHTML = data.html;
  slot.dataset.rowsKey = data.rows_key;
  const current = slot.querySelector(".stat-table");
  if (current && sameRows) current.scrollTop = scrollTop;

  const sortFrom = (event) => {
    const header = event.target.closest("th[data-sort]");
    if (header) setTriggerValue("sort", header.dataset.sort);
  };
  slot.onclick = sortFrom;
  slot.onkeydown = (event) => {
    if (event.key === "Enter" || event.key === " ") {
      if (event.target.closest("th[data-sort]")) event.preventDefault();
      sortFrom(event);
    }
  };
}
"""
SORTABLE_TABLE = st.components.v2.component(
    "sortable_stat_table",
    html='<div class="table-slot"></div>',
    js=SORTABLE_TABLE_JS,
    isolate_styles=False,
)

TINT_LEGEND = (
    '<div class="tint-legend">'
    '<span class="key"><span class="swatch bad2"></span><span class="swatch bad1"></span>'
    "Weak for the category</span>"
    '<span class="key"><span class="swatch good1"></span><span class="swatch good2"></span>'
    "Strong for the category</span>"
    "</div>"
)


def sort_column(sort_key: str, view: str) -> str:
    """Data column for a sort key. Stat keys follow the per-game or season-total view."""
    if sort_key in SORT_COLUMNS:
        return SORT_COLUMNS[sort_key]
    if sort_key in STAT_LABELS:
        return f"{stat_prefix(view)}_{sort_key}"
    return sort_key


def next_sort(current_key: str, descending: bool, clicked: str) -> tuple[str, bool]:
    """Clicking the sorted column flips its order; a new column starts at its natural order."""
    if clicked == current_key:
        return clicked, not descending
    return clicked, clicked not in SORT_COLUMNS


def filter_and_sort(
    frame: pd.DataFrame,
    search: str,
    teams: list[str],
    column: str,
    descending: bool,
) -> pd.DataFrame:
    shown = frame
    if search.strip():
        needle = plain_text(search.strip())
        shown = shown[shown["SEARCH_NAME"].str.contains(needle, regex=False)]
    if teams:
        shown = shown[shown["TEAM_ABBREVIATION"].isin(teams)]
    # Sort by rank first so ties in the chosen stat stay in fantasy-value order.
    shown = shown.sort_values("RANK")
    return shown.sort_values(column, ascending=not descending, kind="stable")


def on_header_click() -> None:
    clicked = st.session_state[TABLE_KEY].get("sort")
    if clicked:
        current_key, descending = st.session_state.get(SORT_STATE_KEY, (DEFAULT_SORT, True))
        st.session_state[SORT_STATE_KEY] = next_sort(current_key, descending, clicked)


@st.fragment
def full_rankings(players: pd.DataFrame, show_colors: bool, projection_view: str = "") -> None:
    """Search, filters and table. A fragment, so typing does not redraw the rest of the page."""
    search_col, team_col, view_col = st.columns([4, 4, 3])
    search = search_col.text_input(
        "Search player", type="search", placeholder="Player name", live=True
    )
    teams = team_col.multiselect(
        "Team",
        sorted(players["TEAM_ABBREVIATION"].unique()),
        placeholder="All teams",
    )
    view = view_col.segmented_control(
        "Stats", [PER_GAME, SEASON_TOTALS], default=PER_GAME, required=True
    )

    sort_key, descending = st.session_state.get(SORT_STATE_KEY, (DEFAULT_SORT, True))
    shown = filter_and_sort(players, search, teams, sort_column(sort_key, view), descending)
    st.caption(
        f"Showing {len(shown)} of {len(players)} players · Click a column header to sort"
    )
    SORTABLE_TABLE(
        key=TABLE_KEY,
        data={
            "html": table_html(
                shown, view, sort_key, descending,
                fixed_height=True, show_colors=show_colors, sortable=True,
            ),
            "rows_key": f"{projection_view}|{search}|{teams}|{view}|{sort_key}|{descending}",
        },
        on_sort_change=on_header_click,
    )


def render() -> None:
    st.title("NBA Fantasy Projections")
    projection_view = projection_view_control()
    players, modified = require_players(projection_view)
    show_as_of(players, projection_view)
    detail = (
        "82-game season with injuries known heading into it"
        if projection_view == PRESEASON
        else "actual games so far plus a projection of the rest"
    )
    st.caption(
        f"{len(players)} rostered players · {projection_view}: {detail} · "
        f"Projections updated {modified:%b %d, %Y}"
    )

    st.subheader("Top 10")
    toggle_col, legend_col = st.columns([1, 4], vertical_alignment="center")
    show_colors = toggle_col.toggle("Category colors", value=True)
    if show_colors:
        legend_col.html(TINT_LEGEND)
    st.html(table_html(players.head(10), PER_GAME, show_colors=show_colors))

    st.subheader("Full rankings")
    full_rankings(players, show_colors, projection_view)

    with st.expander("How to read these numbers"):
        st.markdown(
            "- **Rank and Value:** overall fantasy value across all 9 categories. Rank stays "
            "the same whatever the sort or filter. A player's worst category counts less the "
            "stronger they are elsewhere.\n"
            "- **Colors:** how the player compares with others projected for 500+ minutes in "
            "that category. For turnovers, blue means fewer.\n"
            "- **Bench players:** their minutes are what remains after each team's top nine, "
            "so read their lines loosely.\n"
            "- **Rookies:** projected from draft pick, age, college stats, height and team "
            "opportunity; see the Rookies page.\n"
            "- **No recent NBA seasons:** players back after years away get replacement-level "
            "stats.\n"
            "- **Current:** the whole season. Games already played count as they happened; the "
            "remaining games blend the preseason projection with this season's minutes "
            "(weighted strongly: about half after 10 games) and per-minute stats (weighted by "
            "stat: shot attempts and assists move fastest, steals and blocks slowest), minus "
            "known injury absences. Per-game numbers are whole-season averages.\n"
            "- **Preseason:** a full 82 games minus injuries known heading into the season, "
            "found from who missed their team's opening games (so it uses hindsight, and a "
            "player who wasn't in the rotation can look injured). "
            "Correct those in data/raw/injury_overrides_preseason.csv. Players out for the "
            "season are left out of both. Teammates of injured players get the extra minutes."
        )
