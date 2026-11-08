"""The fantasy app: rankings, rookies, a team builder and a trade analyzer.

Run from the project root:
    streamlit run src/dashboard.py

Input: data/processed/projections_by_team.csv. Ranks, fantasy value and category scores come
from src/rankings.py, and team feedback from src/team_analysis.py; the pages in
src/app_pages/ only display them. The only file the app writes is the saved team (see
src/league_store.py). Rerun project.py, then refresh the page, to see new projections.
"""

import streamlit as st

from app_pages import rankings_page, rookies_page, team_page, trade_page
from app_shared import current_theme, page_css


def main() -> None:
    st.set_page_config(page_title="NBA Fantasy Projections", layout="wide")
    st.html(page_css(current_theme()))
    page = st.navigation(
        [
            st.Page(rankings_page.render, title="Rankings", url_path="rankings", default=True),
            st.Page(rookies_page.render, title="Rookies", url_path="rookies"),
            st.Page(team_page.render, title="My Team", url_path="my-team"),
            st.Page(trade_page.render, title="Trade Analyzer", url_path="trade"),
        ],
        position="top",
    )
    page.run()


if __name__ == "__main__":
    main()
