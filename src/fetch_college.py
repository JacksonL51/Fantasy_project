"""Download Bart Torvik's per-season college player tables (for rookie layer 3).

Run it yourself from the project root:   python src/fetch_college.py
Only some seasons:                       python src/fetch_college.py 2025 2026

Saves data/raw/torvik/torvik_YEAR.csv, one file per season (YEAR = the season's
ending year, so 2024 is 2023-24). Seasons already saved are skipped. Default
seasons are 2008-2026, each rookie class's last college season from 2009 on.

Polite by design: one request per season, 10 seconds apart (the site's
robots.txt crawl delay), an honest user agent, and it only uses the per-season
CSV download (per-player pages are off limits in robots.txt). If the site
answers with anything other than a CSV (for example its browser check), the
script stops and you can download the remaining seasons by hand in a browser
from the printed links. It never tries to get around that check.

If you plan to share or publish anything built from this data, check Torvik's
terms or ask him first.
"""

import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "torvik"
URL = "https://barttorvik.com/getadvstats.php?year={year}&csv=1"
DEFAULT_YEARS = list(range(2008, 2027))
DELAY_SECONDS = 10.0  # robots.txt Crawl-Delay
USER_AGENT = "personal-fantasy-projection-script/1.0 (one CSV per season, 10s apart)"
MIN_ROWS = 1000  # a real season has thousands of players
MIN_COLUMNS = 60  # the table has about 78 columns


def check_csv(text: str) -> str | None:
    """Return a problem description, or None if `text` looks like a season table."""
    head = text.lstrip()[:200].lower()
    if head.startswith("<") or "<html" in head:
        return "the site sent a web page instead of a CSV (probably its browser check)"
    try:
        table = pd.read_csv(StringIO(text), header=None, low_memory=False)
    except Exception as error:  # malformed CSV
        return f"could not read the CSV ({error})"
    if len(table) < MIN_ROWS or table.shape[1] < MIN_COLUMNS:
        return f"unexpected table size {table.shape}"
    return None


def download(years: list[int]) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    todo = [year for year in years if not (OUTPUT_DIR / f"torvik_{year}.csv").exists()]
    skipped = len(years) - len(todo)
    if skipped:
        print(f"Skipping {skipped} season(s) already saved in {OUTPUT_DIR}.")
    for position, year in enumerate(todo):
        if position:
            time.sleep(DELAY_SECONDS)
        url = URL.format(year=year)
        print(f"{year}: downloading ...", end=" ", flush=True)
        try:
            response = session.get(url, timeout=60)
        except requests.RequestException as error:
            print(f"failed ({error}).")
            return _stop(todo[position:])
        if response.status_code != 200:
            print(f"failed (HTTP {response.status_code}).")
            return _stop(todo[position:])
        problem = check_csv(response.text)
        if problem:
            print(f"stopped: {problem}.")
            return _stop(todo[position:])
        path = OUTPUT_DIR / f"torvik_{year}.csv"
        path.write_text(response.text, encoding="utf-8")
        rows = response.text.count("\n")
        print(f"saved {path.name} ({rows:,} players).")
    print(f"Done. Files are in {OUTPUT_DIR}.")
    return 0


def _stop(remaining: list[int]) -> int:
    print("\nNot downloaded. Open these in your browser and save each as torvik_YEAR.csv in")
    print(f"{OUTPUT_DIR}:")
    for year in remaining:
        print(f"  {URL.format(year=year)}")
    return 1


if __name__ == "__main__":
    chosen = [int(arg) for arg in sys.argv[1:]] or DEFAULT_YEARS
    sys.exit(download(chosen))
