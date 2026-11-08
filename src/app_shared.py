"""Data loading, tables and styling shared by the app's pages."""

import html
import unicodedata
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
import streamlit as st

from rankings import rank_players
from team_analysis import average_player

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

# A new team stays in the session until it's saved. Pages can use it while it's unsaved.
DRAFT_KEY = "unsaved_team"
DRAFT_CHOICE = "unsaved"

# Category tint: blue is a strength, red is a weakness, no tint is near average. The bands
# are category z-scores from rankings.py, with turnovers already flipped so blue is good.
TINT_LIGHT_FROM = 1.0
TINT_STRONG_FROM = 2.0

# The full table keeps this height whatever the filter returns, so the page never changes
# length (and the browser never has to jump) while someone searches.
FULL_TABLE_HEIGHT_PX = 640

PLAYER_LIST_HEIGHT_PX = 420

# Draws the player list from Python and reports Add clicks back. The scroll position is kept
# while the search stays the same, so someone can keep adding from further down the list.
PLAYER_LIST_JS = """
export default function ({ data, parentElement, setTriggerValue }) {
  const slot = parentElement.querySelector(".list-slot");
  const previous = slot.querySelector(".player-list");
  const scrollTop = previous ? previous.scrollTop : 0;
  const sameSearch = slot.dataset.listKey === data.list_key;
  slot.innerHTML = data.html;
  slot.dataset.listKey = data.list_key;
  const current = slot.querySelector(".player-list");
  if (current && sameSearch) current.scrollTop = scrollTop;
  slot.onclick = (event) => {
    const button = event.target.closest("button[data-add]");
    if (button && !button.disabled) setTriggerValue("add", button.dataset.add);
  };
}
"""
PLAYER_LIST = st.components.v2.component(
    "player_add_list",
    html='<div class="list-slot"></div>',
    js=PLAYER_LIST_JS,
    isolate_styles=False,
)

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


def plain_text(text: str) -> str:
    """Lowercase and strip accents, so 'jokic' finds 'Jokić'."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def data_note(row: pd.Series) -> str:
    if row["NO_PRIOR_DATA"]:
        if row.get("ROOKIE_PRIOR", False) == True:  # noqa: E712 (numpy bools)
            pick = row.get("DRAFT_PICK")
            return f"Rookie, pick {int(pick)}" if pd.notna(pick) else "Rookie, undrafted"
        return "No recent NBA seasons"
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


def require_players() -> tuple[pd.DataFrame, datetime]:
    """The ranked players and when projections were made. Stops the page if there are none."""
    if not PROJECTIONS_PATH.exists():
        st.error(
            f"No projections found at {PROJECTIONS_PATH}. "
            "Run `python src/project.py` first."
        )
        st.stop()
    modified = PROJECTIONS_PATH.stat().st_mtime
    return load_players(str(PROJECTIONS_PATH), modified), datetime.fromtimestamp(modified)


DATA_DIR = PROJECTIONS_PATH.parent.parent
ROOKIE_DETAIL_FILES = {
    "college": DATA_DIR / "processed" / "college_stats.csv",
    "ages": DATA_DIR / "raw" / "rookie_ages.csv",
    "positions": DATA_DIR / "raw" / "player_positions.csv",
}


def is_rookie(frame: pd.DataFrame) -> pd.Series:
    """Rostered players with no NBA season who got the rookie projection."""
    flag = frame["ROOKIE_PRIOR"] if "ROOKIE_PRIOR" in frame else pd.Series(False, index=frame.index)
    return frame["NO_PRIOR_DATA"].eq(True) & flag.eq(True)


@st.cache_data
def _rookie_details(stamps: tuple) -> pd.DataFrame:
    """Age, height, position and final college season keyed by PLAYER_ID."""
    details = pd.DataFrame(columns=["PLAYER_ID"])
    paths = ROOKIE_DETAIL_FILES
    if paths["ages"].exists():
        ages = pd.read_csv(paths["ages"], usecols=["PLAYER_ID", "AGE"])
        details = details.merge(ages, on="PLAYER_ID", how="outer")
    if paths["positions"].exists():
        positions = pd.read_csv(paths["positions"], usecols=["PLAYER_ID", "POSITION", "HEIGHT_IN"])
        details = details.merge(positions.drop_duplicates("PLAYER_ID"), on="PLAYER_ID", how="outer")
    if paths["college"].exists():
        college = pd.read_csv(paths["college"]).drop_duplicates("PLAYER_ID")
        college = college[["PLAYER_ID", "TEAM", "COLLEGE_YEAR", "CLASS", "GP", "PTS", "TREB", "AST", "STL", "BLK", "BPM"]]
        details = details.merge(college, on="PLAYER_ID", how="outer")
    return details


def rookie_details() -> pd.DataFrame:
    """Rookie details, refreshed when any source file changes. Missing files leave blanks."""
    stamps = tuple(path.stat().st_mtime if path.exists() else 0 for path in ROOKIE_DETAIL_FILES.values())
    return _rookie_details(stamps)


def height_text(inches: float) -> str:
    return f"{int(inches) // 12}'{int(inches) % 12}\"" if pd.notna(inches) else ""


def team_choices(league) -> list:
    """Team picker options: saved team IDs, plus the unsaved new team if there is one."""
    choices = [team.id for team in league.teams]
    if DRAFT_KEY in st.session_state:
        choices.append(DRAFT_CHOICE)
    return choices


def team_choice_label(league):
    """A function that shows a team picker option as the team's name."""
    names = {team.id: team.name for team in league.teams}

    def label(choice) -> str:
        if choice == DRAFT_CHOICE:
            return f"{st.session_state[DRAFT_KEY].name or 'New team'} (not saved)"
        return names.get(choice, str(choice))

    return label


def player_label(players: pd.DataFrame):
    """A function that shows a player ID as 'Name · Team · #rank' in pickers."""
    labels = {
        row.PLAYER_ID: f"{row.PLAYER_NAME} · {row.TEAM_ABBREVIATION} · #{row.RANK}"
        for row in players.itertuples()
    }
    return lambda player_id: labels.get(player_id, str(player_id))


def player_row(label: str, button: str, key: str, on_click, args: tuple, disabled: bool = False) -> None:
    """One player's line with an action button at the end."""
    text_col, button_col = st.columns([3, 1], vertical_alignment="center")
    text_col.markdown(label)
    button_col.button(
        button, key=key, on_click=on_click, args=args, disabled=disabled, width="stretch"
    )


def player_search(
    players: pd.DataFrame,
    exclude: list[int],
    key: str,
    label: str,
    on_add,
    disabled: bool = False,
) -> None:
    """Search box and a scrollable list of Add buttons, best available first.

    Accents are optional, so 'jokic' finds 'Jokić'. Pickers with built-in search were not
    used: they don't ignore accents and can add the wrong player on Enter. The list is drawn
    as one piece of HTML because hundreds of Streamlit buttons would slow every click.
    """
    search = st.text_input(
        label, key=key, type="search", placeholder="Player name", live=True, disabled=disabled
    )
    available = players[~players["PLAYER_ID"].isin(exclude)]
    if search.strip():
        needle = plain_text(search.strip())
        available = available[available["SEARCH_NAME"].str.contains(needle, regex=False)]
    noun = "player" if len(available) == 1 else "players"
    st.caption(f"{len(available)} available {noun}, best first · Scroll for more")
    list_key = f"{key}_list"
    PLAYER_LIST(
        key=list_key,
        data={"html": player_list_html(available, disabled), "list_key": search.strip()},
        on_add_change=partial(_on_list_click, list_key, on_add, key),
    )


def _on_list_click(list_key: str, on_add, search_key: str) -> None:
    clicked = st.session_state[list_key].get("add")
    if clicked:
        on_add(int(clicked), search_key)


def player_list_html(available: pd.DataFrame, disabled: bool) -> str:
    """The scrollable list of players with Add buttons. Names and teams are escaped."""
    state = " disabled" if disabled else ""
    rows = [
        '<div class="player-list-row">'
        f'<span class="who"><span class="name">{html.escape(str(row.PLAYER_NAME))}</span>'
        f'<span class="detail">{html.escape(str(row.TEAM_ABBREVIATION))} · #{row.RANK}</span>'
        f'</span><button type="button" data-add="{row.PLAYER_ID}"{state} '
        f'aria-label="Add {html.escape(str(row.PLAYER_NAME))}">Add</button></div>'
        for row in available.itertuples()
    ]
    body = "".join(rows) or '<p class="no-matches">No available players match this search.</p>'
    return f'<div class="player-list">{body}</div>'


def stat_prefix(view: str) -> str:
    return "TOTAL" if view == SEASON_TOTALS else "PROJ"


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


def player_row_html(
    player: dict,
    columns: list[dict[str, str]],
    current_sort: str | None = None,
    show_colors: bool = True,
    row_class: str = "",
) -> str:
    """One player's table row. Names and teams are escaped; everything else is numeric."""
    note = f'<span class="tag">{html.escape(player["NOTE"])}</span>' if player["NOTE"] else ""
    name_sorted = " sorted" if current_sort == "PLAYER" else ""
    rank = "" if player["RANK"] is None else int(player["RANK"])
    cells = [
        f'<td class="rank">{rank}</td>',
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
    return f'<tr class="{row_class}">{"".join(cells)}</tr>'


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

    rows = [
        player_row_html(player, columns, current_sort, show_colors)
        for player in frame.to_dict("records")
    ]

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


def category_number(key: str, value: float, per_game: bool = False) -> str:
    if key in ("FG", "FT"):
        return f"{value:.1f}%"
    return f"{value:.1f}" if per_game else f"{value:,.0f}"


def score_tint(score: float) -> str:
    """Tint on a narrower range than player scores, because averages and changes are smaller."""
    return tint_class(score * 2.0)


def category_table_html(categories: pd.DataFrame, view: str) -> str:
    """The team's nine categories: as one average player per game, or in season totals."""
    per_game = view == PER_GAME
    column = "PER_GAME" if per_game else "TOTAL"
    heading = "Per game (as one player)" if per_game else view
    rows = []
    for key, row in categories.iterrows():
        rows.append(
            f'<tr><td class="label">{html.escape(row["LABEL"])}</td>'
            f'<td class="num">{category_number(key, row[column], per_game)}</td>'
            f'<td class="num {score_tint(row["SCORE"])}">{row["SCORE"]:+.2f}</td></tr>'
        )
    return (
        '<div class="stat-table compact" tabindex="0"><table><thead><tr>'
        f'<th class="label">Category</th><th class="num">{html.escape(heading)}</th>'
        '<th class="num" title="Average of your players\' category scores, compared with players '
        'projected for 500+ minutes. Above 0 is better, including for turnovers">'
        f'Average player score</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def trade_table_html(
    giving: pd.DataFrame,
    getting: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
    moves: pd.Series,
    view: str,
) -> str:
    """One table holding the players in the trade.

    Per game adds each side as one average player. Season totals adds the team's before,
    after and change under the same stats.
    """
    per_game = view == PER_GAME
    columns = table_columns(view)
    width = len(columns) + 3

    def section(title: str) -> str:
        return f'<tr class="section"><td colspan="{width}">{html.escape(title)}</td></tr>'

    def team_row(label: str, row_class: str, text, tint=None) -> str:
        cells = [
            '<td class="rank"></td>',
            f'<td class="player"><span class="name">{html.escape(label)}</span></td>',
            '<td class="team"></td>',
        ]
        for column in columns:
            classes = ["num"]
            if column.get("group"):
                classes.append("group")
            content = ""
            if column.get("score"):
                category = column["score"].removeprefix("Z_")
                content = text(category)
                if tint:
                    classes.append(tint(category))
            cells.append(f'<td class="{" ".join(classes)}">{content}</td>')
        return f'<tr class="{row_class}">{"".join(cells)}</tr>'

    def plain(frame: pd.DataFrame):
        def show(category: str) -> str:
            value = frame.loc[category, "TOTAL"]
            return f"{value:.1f}" if category in ("FG", "FT") else f"{value:,.0f}"
        return show

    def change(category: str) -> str:
        value = after.loc[category, "TOTAL"] - before.loc[category, "TOTAL"]
        return f"{value:+.1f}" if category in ("FG", "FT") else f"{value:+,.0f}"

    def side(title: str, rows: pd.DataFrame) -> list[str]:
        if rows.empty:
            return []
        lines = [section(title)]
        lines += [player_row_html(player, columns) for player in rows.to_dict("records")]
        if per_game and len(rows) > 1:
            average = {"RANK": None, "PLAYER_NAME": "Average", "TEAM_ABBREVIATION": "", "NOTE": ""}
            lines.append(
                player_row_html(
                    {**average, **average_player(rows)}, columns, row_class="team-row change"
                )
            )
        return lines

    head = [
        '<th class="rank" title="Overall fantasy rank">#</th>',
        '<th class="player">Player</th>',
        '<th class="team">Team</th>',
    ]
    for column in columns:
        classes = ["num"]
        if column.get("group"):
            classes.append("group")
        if column.get("class"):
            classes.append(column["class"])
        head.append(
            f'<th class="{" ".join(classes)}" title="{html.escape(column["title"])}">'
            f'{column["label"]}</th>'
        )

    body = [*side("You give", giving), *side("You get", getting)]
    if not per_game:
        body.append(section("Your team"))
        body.append(team_row("Before", "team-row", plain(before)))
        body.append(team_row("After", "team-row", plain(after)))
        body.append(
            team_row(
                "Change", "team-row change", change,
                tint=lambda category: score_tint(moves[category] * 2.0),
            )
        )
    return (
        '<div class="stat-table trade" tabindex="0"><table>'
        f'<thead><tr>{"".join(head)}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
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
.stat-table.compact table {{ min-width: 520px; table-layout: auto; }}
.stat-table.compact .num {{ width: auto; }}
.stat-table.compact .label {{ font-weight: 600; }}
.stat-table.compact .muted {{ color: {t["muted"]}; }}
.stat-table.trade .section td {{
  padding: 7px 8px; color: {t["muted"]}; font-size: 12px; font-weight: 600;
  letter-spacing: 0.02em; background: {t["tag"]};
}}
.stat-table.trade .change td {{ font-weight: 700; }}
.player-list {{
  height: {PLAYER_LIST_HEIGHT_PX}px; overflow-y: auto; box-sizing: border-box;
  background: {t["surface"]}; color: {t["ink"]}; font-size: 14px;
  border: 1px solid {t["line"]}; border-radius: 10px; scrollbar-gutter: stable;
}}
.player-list-row {{
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 7px 8px 7px 12px; border-bottom: 1px solid {t["line"]};
}}
.player-list-row:last-child {{ border-bottom: 0; }}
.player-list-row:hover {{ background: {t["hover"]}; }}
.player-list-row .who {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.player-list-row .name {{ font-weight: 600; }}
.player-list-row .detail {{ margin-left: 8px; color: {t["muted"]}; font-size: 13px; }}
.player-list-row button {{
  flex: none; padding: 4px 14px; font: inherit; font-size: 13px; cursor: pointer;
  color: {t["ink"]}; background: transparent;
  border: 1px solid {t["line_strong"]}; border-radius: 6px;
}}
.player-list-row button:hover:enabled {{ background: {t["tag"]}; }}
.player-list-row button:focus-visible {{ outline: 2px solid {t["muted"]}; outline-offset: 1px; }}
.player-list-row button:disabled {{ color: {t["muted"]}; cursor: not-allowed; opacity: 0.55; }}
.player-list .no-matches {{ margin: 0; padding: 32px 16px; text-align: center; color: {t["muted"]}; }}
.tint-legend {{
  display: flex; flex-wrap: wrap; gap: 4px 0; align-items: center;
  color: {t["muted"]}; font-size: 13px; margin: 0;
}}
.tint-legend .swatch {{ width: 22px; height: 14px; border-radius: 3px; display: inline-block; }}
.tint-legend .swatch + .swatch {{ margin-left: -3px; }}
.tint-legend .key {{ display: inline-flex; align-items: center; gap: 6px; margin-right: 14px; }}
@media (max-width: 640px) {{
  .stat-table table {{ min-width: 1060px; }}
  .stat-table.compact table {{ min-width: 520px; }}
  .stat-table .rank, .stat-table .player {{ position: static; box-shadow: none; }}
  .stat-table th.rank, .stat-table th.player {{ position: sticky; left: auto; }}
  .stat-table .tag {{ display: none; }}
}}
</style>"""


def current_theme() -> str:
    theme_type = st.context.theme.type
    return theme_type if theme_type in THEMES else "light"
