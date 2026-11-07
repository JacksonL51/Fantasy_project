# NBA Fantasy Projections

Season projections for every rostered NBA player, built for 9-category fantasy basketball
(points, threes, rebounds, assists, steals, blocks, turnovers, FG% and FT%), plus a small app
for rankings, building your own teams and checking trades.

The pipeline pulls NBA stats with [`nba_api`](https://github.com/swar/nba_api), learns how
players' per-minute production changes from one season to the next, estimates minutes and
games played, and fits every team to a realistic minutes budget. The results feed a
Streamlit app with three pages:

- **Rankings:** every player ranked by 9-category fantasy value, with search, a team filter,
  click-to-sort columns and color-coded category strengths.
- **My Team:** create and name your own teams of 13 players, see each team's strongest and
  weakest categories, and view stats per game or as season totals.
- **Trade Analyzer:** pick one of your teams and the players on each side of a trade, then
  see whether it helps your team and how each category changes.

## Getting started

You need Python 3.11 or newer.

```bash
python -m venv venv
```

Activate it (`venv\Scripts\activate` on Windows, `source venv/bin/activate` on macOS and
Linux), then install the packages:

```bash
pip install -r requirements.txt
```

Data files aren't stored in git, so a fresh copy has to run the pipeline once before the app
has anything to show. Run every command from the project root.

## Running the pipeline

Run these in order. The first run of `ingest.py` downloads every season since 1996-97, which
takes a while. Later runs reuse the saved files and only download seasons that are missing.
Rosters are saved too, so to get current rosters, delete `data/raw/current_rosters.csv`
before running it.

| Step | Command | What it does |
|---|---|---|
| 1 | `python src/ingest.py` | Downloads player-season stats and current rosters into `data/raw/` |
| 2 | `python src/prep_training_data.py` | Combines traded players' seasons, adds per-minute rates and builds season-to-season training pairs |
| 3 | `python src/model.py` | Trains and selects the per-minute rate models |
| 4 | `python src/project.py` | Writes this season's projections to `data/processed/projections_by_team.csv` |
| 5 | `python src/rankings.py` | Optional: writes `player_rankings.csv` with ranks and category scores |
| 6 | `python src/backtest.py` | Optional: tests the pipeline on the last 3 finished seasons |

To pick up roster moves during the season, delete the saved rosters file, then rerun steps 1
and 4. Rerun steps 2 and 3 when a new season of history has been added. The backtest can also score a single season, for example
`python src/backtest.py 2026`.

## Using the app

```bash
streamlit run src/dashboard.py
```

The app needs Streamlit 1.66 or newer. It reads the projections file, so rerun
`src/project.py` and refresh the page to see new projections. Restart the app after editing
any of its code.

**My Team.** Use **New team** to start a team. It gets the next free name (Team1, Team2 and
so on), which you can change. Add players from the scrollable list or by searching, then
choose **Save team** or **Discard**. A new team isn't saved until you save it, and it's lost
if you reload the page. After that, every change to a saved team saves right away. Each team
keeps its own roster and league size (6 to 16 teams, 10 by default).

**Trade Analyzer.** Choose one of your teams, tick the players you'd give and search for the
players you'd get. The verdict uses season totals, so games played count. If a trade leaves
you with too many players, it assumes you drop your lowest-value player; if it leaves an
empty spot, it assumes a replacement-level pickup.

Your teams are saved to `data/league/teams.json` on your own computer. That file is ignored
by git, and the pipeline never reads or changes it.

## How the projections work

- **Per-minute rates:** ridge regression models predict each stat per minute from a player's
  last season, a blend of his last three seasons, his age and his minutes. A model is used
  only if it beats simply repeating last season in every test year.
- **Small samples:** rookies and players with too few minutes are pulled toward
  replacement-level rates. Players who missed last season, and young starters coming back
  from a short injury-hit season, have their own rules.
- **Minutes:** a recency-weighted average of the last three seasons, adjusted for age.
- **Games played:** a model of next season's games that uses up to five seasons of
  durability history for rotation players. Even very durable players project well below 82,
  because season totals need the average outcome, including the occasional major injury.
- **Team budget:** each team's projected minutes add up to a full season (82 games of 240
  minutes). The top nine players keep their own minutes, and the rest share what's left.
- **Fantasy value:** the sum of each player's 9 category scores, compared with players
  projected for 500+ minutes. A strong player's single worst category counts less.

### How accurate is it?

The backtest predicts each of the last three seasons using only earlier data. Ranking players
by projected fantasy value against how they actually finished (Spearman correlation, players
with 500+ minutes):

| Season | This pipeline | Repeating last season |
|---|---|---|
| 2023-24 | 0.735 | 0.713 |
| 2024-25 | 0.700 | 0.661 |
| 2025-26 | 0.570 | 0.532 |

It beats simply repeating last season every year. Games played remain the hardest part to
predict, and major injuries and sudden role changes can't be forecast from season totals.

## Project layout

```
src/
  ingest.py              download stats and rosters
  prep_training_data.py  build training data
  model.py               train the rate models
  history.py             shared history features and the games model
  project.py             make this season's projections
  backtest.py            test the pipeline on past seasons
  rankings.py            fantasy value and rankings
  team_analysis.py       team stats, feedback and trade comparisons
  league_store.py        save and load your teams
  dashboard.py           app entry point
  app_shared.py          app data loading, tables and styling
  app_pages/             the Rankings, My Team and Trade Analyzer pages
tests/                   unit tests for team analysis and saving
data/raw/                downloaded data (not in git)
data/processed/          pipeline outputs (not in git)
data/league/             your saved teams (not in git)
```

## Tests

```bash
python -m unittest discover tests
```

The tests cover team stats, trade comparisons (including uneven trades), saving teams and
team names. The projection pipeline is checked by the backtest instead.

## Known limitations

- Projections cover a full season, so the Trade Analyzer doesn't adjust for games already
  played or for current injuries.
- Rookies and players with no NBA history get replacement-level stats.
- Bench players' per-game lines are what's left after each team's top nine, so read them
  loosely.
- Minutes don't yet account for teammates leaving or arriving, so some role changes are
  missed.
- Team feedback compares categories within your team, not against other fantasy teams.
