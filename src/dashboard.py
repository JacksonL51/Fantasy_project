"""Read-only dashboard for the projections written by src/project.py.

Run from the project root:
    streamlit run src/dashboard.py

Input: data/processed/projections_by_team.csv. Nothing is written. Ranks, fantasy value and
category scores come from src/rankings.py, which owns that logic; this file only displays
them. Rerun project.py, then refresh the page, to see new projections.
"""

import html
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from rankings import rank_players

PROJECTIONS_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "projections_by_team.csv"
)

STAT_LABELS = {
    "PTS": "PTS",
    "FG3M": "3PM",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "TOV": "TOV",
}
STAT_TITLES = {
    "PTS": "Points",
    "FG3M": "Three-pointers made",
    "REB": "Rebounds",
    "AST": "Assists",
    "STL": "Steals",
    "BLK": "Blocks",
    "TOV": "Turnovers (fewer is better)",
}
PER_GAME = "Per game"
SEASON_TOTALS = "Season totals"

# Sort keys that are not stats. The first click on these sorts low to high (A-Z, rank 1
# first); the first click on any stat sorts high to low.
SORT_COLUMNS = {"RANK": "RANK", "PLAYER": "SEARCH_NAME", "TEAM": "TEAM_ABBREVIATION"}
DEFAULT_SORT = "VALUE"
SORT_STATE_KEY = "table_sort"
TABLE_KEY = "full_rankings_table"

# Category tint: blue is a strength, red is a weakness, no tint is near average. The bands
# are category z-scores from rankings.py, with turnovers already flipped so blue is good.
TINT_LIGHT_FROM = 1.0
TINT_STRONG_FROM = 2.0

# The full table keeps this height whatever the filter returns, so the page never changes
# length (and the browser never has to jump) while someone searches.
FULL_TABLE_HEIGHT_PX = 640

THEMES = {
    "light": {
        "surface": "#ffffff",
        "hover": "#f4f5f7",
        "ink": "#1a1c20",
        "muted": "#646973",
        "line": "#e6e7ea",
        "line_strong": "#c9ccd2",
        "tag": "#eef0f3",
        "good_light": "rgba(42, 120, 214, 0.12)",
        "good_strong": "rgba(42, 120, 214, 0.28)",
        "bad_light": "rgba(227, 73, 72, 0.11)",
        "bad_strong": "rgba(227, 73, 72, 0.26)",
    },
    "dark": {
        "surface": "#171a21",
        "hover": "#20242d",
        "ink": "#f0f1f3",
        "muted": "#a0a5b0",
        "line": "#2a2e38",
        "line_strong": "#434855",
        "tag": "#2a2e38",
        "good_light": "rgba(57, 135, 229, 0.20)",
        "good_strong": "rgba(57, 135, 229, 0.42)",
        "bad_light": "rgba(230, 103, 103, 0.17)",
        "bad_strong": "rgba(230, 103, 103, 0.38)",
    },
}

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


def plain_text(text: str) -> str:
    """Lowercase and strip accents, so 'jokic' finds 'Jokić'."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def data_note(row: pd.Series) -> str:
    if row["NO_PRIOR_DATA"]:
        return "No NBA history"
    if row["STALE_DATA"]:
        return "Missed last season"
    return ""


@st.cache_data
def load_players(path: str, modified: float) -> pd.DataFrame:
    """Load projections with ranks, best first. `modified` refreshes the cache on file change."""
    projections = pd.read_csv(path)
    rankings = rank_players(projections).drop(columns=["PLAYER_NAME", "TEAM_ABBREVIATION"])
    frame = rankings.merge(projections, on="PLAYER_ID", how="left", validate="one_to_one")
    frame["NOTE"] = frame.apply(data_note, axis=1)
    frame["SEARCH_NAME"] = frame["PLAYER_NAME"].map(plain_text)
    return frame


def stat_prefix(view: str) -> str:
    return "TOTAL" if view == SEASON_TOTALS else "PROJ"


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


def table_columns(view: str) -> list[dict[str, str]]:
    """Display order of the numeric columns. `group` marks the first column of a group."""
    prefix = stat_prefix(view)
    stat_format = "{:,.0f}" if view == SEASON_TOTALS else "{:.1f}"
    columns = [
        {"key": "VALUE", "sort": "VALUE", "label": "Value", "format": "{:.1f}", "group": "1",
         "title": "Fantasy value across all 9 categories", "class": "value"},
        {"key": "PROJECTED_GP", "sort": "PROJECTED_GP", "label": "GP", "format": "{:.0f}",
         "group": "1", "title": "Projected games played"},
        {"key": "PROJECTED_MPG", "sort": "PROJECTED_MPG", "label": "MPG", "format": "{:.1f}",
         "title": "Projected minutes per game played"},
    ]
    for index, (stat, label) in enumerate(STAT_LABELS.items()):
        column = {"key": f"{prefix}_{stat}", "sort": stat, "label": label,
                  "format": stat_format, "title": STAT_TITLES[stat], "score": f"Z_{stat}"}
        if index == 0:
            column["group"] = "1"
        columns.append(column)
    columns.append({"key": "PROJ_FG_PCT", "sort": "PROJ_FG_PCT", "label": "FG%",
                    "format": "{:.1f}", "group": "1", "title": "Field-goal percentage",
                    "score": "Z_FG"})
    columns.append({"key": "PROJ_FT_PCT", "sort": "PROJ_FT_PCT", "label": "FT%",
                    "format": "{:.1f}", "title": "Free-throw percentage", "score": "Z_FT"})
    return columns


def tint_class(score: float) -> str:
    if score >= TINT_STRONG_FROM:
        return "good2"
    if score >= TINT_LIGHT_FROM:
        return "good1"
    if score <= -TINT_STRONG_FROM:
        return "bad2"
    if score <= -TINT_LIGHT_FROM:
        return "bad1"
    return ""


def header_cell(
    label: str,
    classes: list[str],
    title: str,
    sort_key: str,
    current_sort: str | None,
    descending: bool,
    sortable: bool,
) -> str:
    attributes = [f'title="{html.escape(title)}"'] if title else []
    if sortable:
        attributes += [f'data-sort="{sort_key}"', 'tabindex="0"', 'role="button"']
    if sort_key == current_sort:
        classes = [*classes, "sorted"]
        label = f"{label} {'▼' if descending else '▲'}"
        attributes.append(f'aria-sort="{"descending" if descending else "ascending"}"')
    return f'<th class="{" ".join(classes)}" {" ".join(attributes)}>{label}</th>'


def table_html(
    frame: pd.DataFrame,
    view: str,
    current_sort: str | None = None,
    descending: bool = True,
    fixed_height: bool = False,
    show_colors: bool = True,
    sortable: bool = False,
) -> str:
    """The stat table as HTML. Player names are escaped; everything else is numeric."""
    columns = table_columns(view)

    def head_cell(label: str, classes: list[str], title: str, sort_key: str) -> str:
        return header_cell(label, classes, title, sort_key, current_sort, descending, sortable)

    head = [
        head_cell("#", ["rank"], "Overall fantasy rank", "RANK"),
        head_cell("Player", ["player"], "", "PLAYER"),
        head_cell("Team", ["team"], "", "TEAM"),
    ]
    for column in columns:
        classes = ["num"]
        if column.get("group"):
            classes.append("group")
        if column.get("class"):
            classes.append(column["class"])
        head.append(head_cell(column["label"], classes, column["title"], column["sort"]))

    rows = []
    for player in frame.to_dict("records"):
        note = (
            f'<span class="tag">{html.escape(player["NOTE"])}</span>' if player["NOTE"] else ""
        )
        name_sorted = " sorted" if current_sort == "PLAYER" else ""
        cells = [
            f'<td class="rank">{int(player["RANK"])}</td>',
            f'<td class="player{name_sorted}"><span class="name">'
            f'{html.escape(str(player["PLAYER_NAME"]))}</span>{note}</td>',
            f'<td class="team">{html.escape(str(player["TEAM_ABBREVIATION"]))}</td>',
        ]
        for column in columns:
            classes = ["num"]
            if column.get("group"):
                classes.append("group")
            if column.get("class"):
                classes.append(column["class"])
            if column["sort"] == current_sort:
                classes.append("sorted")
            if show_colors and column.get("score"):
                tint = tint_class(player[column["score"]])
                if tint:
                    classes.append(tint)
            text = column["format"].format(player[column["key"]])
            cells.append(f'<td class="{" ".join(classes)}">{text}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")

    wrap_class = "stat-table fixed" if fixed_height else "stat-table"
    empty = (
        '<p class="no-matches">No players match this search and team filter.</p>'
        if frame.empty
        else ""
    )
    return (
        f'<div class="{wrap_class}" tabindex="0"><table>'
        f"<thead><tr>{''.join(head)}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{empty}</div>"
    )


def page_css(theme_name: str) -> str:
    t = THEMES[theme_name]
    return f"""<style>
.stMainBlockContainer {{ padding-top: 2.5rem; max-width: 1320px; }}
.stat-table {{
  overflow: auto; background: {t["surface"]}; color: {t["ink"]};
  border: 1px solid {t["line"]}; border-radius: 10px; font-size: 14px;
  font-variant-numeric: tabular-nums;
}}
.stat-table.fixed {{
  height: {FULL_TABLE_HEIGHT_PX}px; box-sizing: border-box; scrollbar-gutter: stable;
}}
/* Fixed column widths: the columns hold still however many rows a search returns. */
.stat-table table {{
  width: 100%; min-width: 1120px; table-layout: fixed;
  border-collapse: separate; border-spacing: 0; margin: 0;
}}
.stat-table th, .stat-table td {{
  padding: 9px 8px; white-space: nowrap; text-align: left; border: 0;
  border-bottom: 1px solid {t["line"]}; box-sizing: border-box;
}}
.stat-table tbody tr:last-child td {{ border-bottom: 0; }}
.stat-table th {{
  position: sticky; top: 0; z-index: 2; background: {t["surface"]}; color: {t["muted"]};
  font-size: 12px; font-weight: 600; letter-spacing: 0.02em;
  border-bottom: 1px solid {t["line_strong"]};
}}
.stat-table th[data-sort] {{ cursor: pointer; user-select: none; }}
.stat-table th[data-sort]:hover {{ color: {t["ink"]}; background: {t["hover"]}; }}
.stat-table th[data-sort]:focus-visible {{ outline: 2px solid {t["muted"]}; outline-offset: -2px; }}
.stat-table .num {{ text-align: right; width: 60px; }}
.stat-table .num.value {{ width: 76px; }}
.stat-table .group {{ border-left: 1px solid {t["line_strong"]}; }}
.stat-table .rank {{
  position: sticky; left: 0; z-index: 1; width: 44px;
  background: {t["surface"]}; color: {t["muted"]}; text-align: right;
}}
.stat-table .player {{
  position: sticky; left: 44px; z-index: 1; background: {t["surface"]};
  box-shadow: 1px 0 0 {t["line"]}; overflow: hidden; text-overflow: ellipsis;
}}
.stat-table th.rank, .stat-table th.player {{ z-index: 3; }}
.stat-table .team {{ color: {t["muted"]}; width: 60px; }}
.stat-table .name {{ font-weight: 600; }}
.stat-table .value {{ font-weight: 700; }}
.stat-table th.sorted {{ color: {t["ink"]}; font-weight: 700; }}
.stat-table td.sorted {{ font-weight: 700; }}
.stat-table tbody tr:hover td {{ background-color: {t["hover"]}; }}
.stat-table .tag {{
  margin-left: 8px; padding: 1px 7px; border-radius: 999px; background: {t["tag"]};
  color: {t["muted"]}; font-size: 11px; font-weight: 500;
}}
.stat-table .no-matches {{ margin: 0; padding: 32px 16px; text-align: center; color: {t["muted"]}; }}
.stat-table td.good1, .stat-table tbody tr:hover td.good1, .swatch.good1 {{ background-color: {t["good_light"]}; }}
.stat-table td.good2, .stat-table tbody tr:hover td.good2, .swatch.good2 {{ background-color: {t["good_strong"]}; }}
.stat-table td.bad1, .stat-table tbody tr:hover td.bad1, .swatch.bad1 {{ background-color: {t["bad_light"]}; }}
.stat-table td.bad2, .stat-table tbody tr:hover td.bad2, .swatch.bad2 {{ background-color: {t["bad_strong"]}; }}
.tint-legend {{
  display: flex; flex-wrap: wrap; gap: 4px 0; align-items: center;
  color: {t["muted"]}; font-size: 13px; margin: 0;
}}
.tint-legend .swatch {{ width: 22px; height: 14px; border-radius: 3px; display: inline-block; }}
.tint-legend .swatch + .swatch {{ margin-left: -3px; }}
.tint-legend .key {{ display: inline-flex; align-items: center; gap: 6px; margin-right: 14px; }}
@media (max-width: 640px) {{
  .stat-table table {{ min-width: 1060px; }}
  .stat-table .rank, .stat-table .player {{ position: static; box-shadow: none; }}
  .stat-table th.rank, .stat-table th.player {{ position: sticky; left: auto; }}
  .stat-table .tag {{ display: none; }}
}}
</style>"""


TINT_LEGEND = (
    '<div class="tint-legend">'
    '<span class="key"><span class="swatch bad2"></span><span class="swatch bad1"></span>'
    "Weak for the category</span>"
    '<span class="key"><span class="swatch good1"></span><span class="swatch good2"></span>'
    "Strong for the category</span>"
    "</div>"
)


def current_theme() -> str:
    theme_type = st.context.theme.type
    return theme_type if theme_type in THEMES else "light"


def on_header_click() -> None:
    clicked = st.session_state[TABLE_KEY].get("sort")
    if clicked:
        current_key, descending = st.session_state.get(SORT_STATE_KEY, (DEFAULT_SORT, True))
        st.session_state[SORT_STATE_KEY] = next_sort(current_key, descending, clicked)


@st.fragment
def full_rankings(players: pd.DataFrame, show_colors: bool) -> None:
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
            "rows_key": f"{search}|{teams}|{view}|{sort_key}|{descending}",
        },
        on_sort_change=on_header_click,
    )


def main() -> None:
    st.set_page_config(page_title="NBA Fantasy Projections", layout="wide")
    st.html(page_css(current_theme()))
    st.title("NBA Fantasy Projections")

    if not PROJECTIONS_PATH.exists():
        st.error(
            f"No projections found at {PROJECTIONS_PATH}. "
            "Run `python src/project.py` first."
        )
        st.stop()

    modified = PROJECTIONS_PATH.stat().st_mtime
    players = load_players(str(PROJECTIONS_PATH), modified)
    st.caption(
        f"{len(players)} rostered players · Projections updated "
        f"{datetime.fromtimestamp(modified):%b %d, %Y}"
    )

    st.subheader("Top 10")
    toggle_col, legend_col = st.columns([1, 4], vertical_alignment="center")
    show_colors = toggle_col.toggle("Category colors", value=True)
    if show_colors:
        legend_col.html(TINT_LEGEND)
    st.html(table_html(players.head(10), PER_GAME, show_colors=show_colors))

    st.subheader("Full rankings")
    full_rankings(players, show_colors)

    with st.expander("How to read these numbers"):
        st.markdown(
            "- **Rank and Value:** overall fantasy value across all 9 categories. Rank stays "
            "the same whatever the sort or filter. A player's worst category counts less the "
            "stronger they are elsewhere.\n"
            "- **Colors:** how the player compares with others projected for 500+ minutes in "
            "that category. For turnovers, blue means fewer.\n"
            "- **Bench players:** their minutes are what remains after each team's top nine, "
            "so read their lines loosely.\n"
            "- **No NBA history:** these players get replacement-level stats."
        )


if __name__ == "__main__":
    main()
