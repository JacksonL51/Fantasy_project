# NBA Fantasy 9-Category Projection MVP — project context

Python pipeline that projects per-game and season-total stats for every rostered
NBA player (9-cat fantasy: PTS, FG3M, REB, AST, STL, BLK, TOV, FG%, FT%).
Run everything from the project root.

## Pipeline and run order
1. `src/ingest.py` — pulls all historical player-season totals (1996-97 on) and current
   rosters via nba_api into `data/raw/` (full league cache kept; return value filtered).
2. `src/prep_training_data.py` — resolves traded players (TOT rows), adds `[STAT]_RATE`
   (per-minute) columns, writes `training_data.csv` and `training_pairs.csv`
   (t -> t+1 transitions, both seasons >= 200 min). Flag `RESTRICT_TO_CURRENT_ROSTER`
   (False = keep retired players, fixes survivor bias; True = old behavior).
3. `src/model.py` — Phase 4. Ridge rate models (11 stats) tuned over feature set
   ("base" = last-season only, "history" = plus 3-season blend), formulation
   (level/delta) and alpha, using expanding-window folds. A tuned model replaces the
   naive prior only if it beats naive in EVERY fold and the pooled bootstrap CI excludes
   zero. Writes `rate_models.joblib`, `policy_decisions.csv`, etc. `TRAIN_WINDOW_YEARS`
   (None default) can limit old eras.
4. `src/project.py` — Phase 5. Rate predictions with credibility blending toward
   replacement level, baseline MPG and games, team minute allocation, stat synthesis.
   Writes `data/processed/projections_by_team.csv`.
5. `src/backtest.py` — predicts a past season using only earlier data and scores it.
   `python src/backtest.py` (no arg) runs the latest 3 seasons. Writes
   `backtest_results.csv`, `backtest_players.csv`, `games_weight_selection.json`,
   `backtest_minutes_test.csv`, `backtest_recommendations.json`.
6. `src/rankings.py` — fantasy rankings from `projections_by_team.csv`; writes
   `player_rankings.csv` (rank, value, category scores). `python src/rankings.py`. Value:
   z-scores against players projected for 500+ minutes, stats scaled by GP^0.5. The worst
   negative category is discounted by strength (sum of the other eight scores): full count at
   strength 2 or below, ignored at 8 or above, falling steadily between. So stars can carry
   one bad category (Giannis's FT) and weak players cannot (Mitchell Robinson). Both marks are
   judgment calls, untested against online boards (no board data saved). Stars' turnovers often
   go uncounted as a result; never discounting turnovers was tried and rejected because it
   moved guards too far (Cunningham 7 -> 23). The backtest still scores the plain season-total
   sum, so the two values differ on purpose. Still to be refined.
7. `src/dashboard.py` — Streamlit entry point and page menu; needs Streamlit 1.66+.
   `streamlit run src/dashboard.py`. Pages live in `src/app_pages/`; `src/app_shared.py` holds
   their data loading, tables and styling. Pages only display results, so tune logic in the
   modules below, not in pages. Restart Streamlit after editing imported modules.
   - Rankings: top 10, full rankings, search, team filter, click-a-header sorting, category
     color switch. The full table keeps a fixed height so searching never moves the page. It
     calls `rank_players` from `src/rankings.py` on the current projections.
   - My Team: draft tracker and team builder for the user's own named teams (other league
     teams are not tracked). Roster of 13; league size is per team, defaults to 10, from 6 to 16.
   - Trade Analyzer: how a trade changes the user's team. One table holds the players in the
     trade. Per game adds an Average row for a side with 2+ players (the side as one player,
     with percentages from combined makes and attempts); season totals adds the team's before,
     after and change. The verdict always uses season totals. My Team's per-game categories
     also show the team as one average player.
8. `src/team_analysis.py` — team stats, feedback and trade comparisons, with no Streamlit.
   There is deliberately no team-level baseline, grade or "typical team" (removed; the user may
   add one later with a real baseline). A category's score is the roster's average player score
   from `rankings.py`, and strongest/weakest categories are ranked within the team. Trade
   verdicts weigh each category's season-total change by how much rostered players differ in
   it, which is where league size matters. Uneven trades: extras drop the lowest-value kept
   player; open spots get the average of the next 10 unrostered players. All bands are judgment
   calls. No rest-of-season scaling or current injuries yet.
9. `src/league_store.py` — saves named teams to `data/league/teams.json` (git-ignored; the
   pipeline never touches it). The older `my_team.json` is read once as Team1 if `teams.json`
   doesn't exist yet. New teams stay in the session until saved; saved teams then save every
   change right away. Default names are the first free Team1, Team2 and so on, and names are
   unique. The Trade Analyzer picks which team to use, including an unsaved one. Tests: `python -m unittest discover tests`.
`src/history.py` is shared (rate-history features, age/role MPG curves, production
score, fitted games model). Its features must be computed identically in training and
projection.

## Key design decisions
- Season-budget minutes: every team totals 19,680 minutes (82 x 240). Season minutes =
  active MPG x GP. PROJECTED_MPG is per game played (active); PROJECTED_GP separate;
  TOTAL_* stats = rate x season minutes (for roto/totals).
- Top-nine core per team is anchored at its own healthy MPG x projected GP (a player needs
  ~700 avg season minutes to claim a core slot). Remaining minutes go to rank 10+ players,
  bounded to 3-24 MPG per game played; bench GP follows from allocated minutes.
- Baseline MPG = 60/27/13 recency-weighted active MPG over last 3 seasons (seasons with
  GP >= 15). Age adjustment to MPG is fitted from data (halved, capped).
- Games played: currently fitted ridge models of next-season games per 82 (`GAMES_METHOD =
  "fitted"`, in `history.py`, capped at 82). Players with 20+ baseline MPG use a model
  trained only on such players, with five-season durability features; others use a model of
  all players. Shortened seasons are scaled to 82. Expected games stay well below 82 even for
  iron men, on purpose: totals need the average, including the odd major injury. The
  alternative, "blend", mixes hand rules (fluke vs chronic injury) with
  league-typical GP (starters 0.5 model weight, bench 0.2). Bench low GP is mostly role,
  not injury.
- Rate models train on all players, including retired ones
  (`RESTRICT_TO_CURRENT_ROSTER = False`), which removes survivor bias.
- Rookies / no-history players get replacement-level rates and typical games.
- Players who missed last season blend model rates toward replacement: model weight 0.85,
  or 0.5 if aged 31+ in their last season played. Past returners support 0.85 (rate error
  about 26% below 0.5, 61 players aged 30 or under). The 0.5 for veterans is a judgment
  call for minutes and games risk; rates alone do not support it (7 players).
- Short-season return (`SHORT_SEASON_RETURN`): latest season of 15-29 games, age 30 or under,
  1,000+ minutes and 28+ MPG the season before, and MPG down no more than 5. Rates and MPG
  average two views (`SHORT_SEASON_IGNORE_SHARE = 0.5`): the season counting for little (history
  blend as the model's last-season input, MPG weight scaled by GP/60) and the season ignored.
  Games are unchanged. In 31 past cases, points-per-game error fell from 2.98 to 2.51, with
  steals and blocks slightly worse; the group was picked to resemble Tatum, so that is somewhat
  optimistic, and the share is a judgment call. Other short-season players got worse with this
  kind of change, so do not widen it without new evidence.
- Selection of model policies uses the same folds that score them, so reported rate
  metrics are slightly optimistic.

## Switches in project.py (set from backtest_recommendations.json)
- `GAMES_METHOD`: "fitted" (current; ridge model of next-season games) or "blend".
- `MPG_ADJUSTMENT`: "age" (current), "role" (adds per-minute production to model rotation
  jumps), or "none".
Recommendations are only trusted when the backtest ran >= 2 seasons. The 3-season backtest
on real data recommends the current settings, "fitted" and "age".

## Backtest findings so far (real data, seasons 2024/2025/2026, 500+ minute players)
- VALUE_Spearman: fitted 0.735/0.700/0.570, blend 0.733/0.694/0.540, naive (repeat last
  year) 0.713/0.661/0.532. The pipeline beats naive every year. Age adjustment helps every
  year (no-age 0.725/0.687/0.531). Rate model adds ~+0.005 consistently.
- Games MAE: fitted 10.5/11.5/11.5, blend 11.2/11.6/11.8, constant-66 baseline
  11.3/12.4/11.4, naive 13.9/15.8/16.6. Fitted misses the constant-66 baseline in 2026.
  Blend weight grid is flat; keep 0.5/0.2.
- Fitted passes the recommendation test in all 3 seasons.
- Games-played ranking is still weak. GP_Spearman: fitted 0.355/0.323/0.192, blend
  0.331/0.281/0.031, naive 0.346/0.306/0.223. Fitted beats blend every year and naive except
  in 2026, a hard season for everyone. Starters in 2024-26 stay about 2.5 games over-projected.
- Tried on top of the games model, no meaningful gain: one-bad-year and career durability
  signals, player role, recent-era level (worse errors), a 12-year training window, trees and
  separate injury/healthy estimates. Real gains likely need game logs.
- Role curve fails the recommendation test all 3 years. It adds a small VALUE_Spearman gain
  (0.735/0.696/0.545 vs age 0.733/0.694/0.540) and bench MAE barely moves, so keep "age".
- Known unforecastable: major injuries (Tatum 2026). Rotation jumps (e.g. Ryan Rollins
  14.6 -> 32.1 MPG) are only partly predictable from age, minutes and per-minute production.
  Jumps driven by teammates leaving need team context (not built).

## Known limitations / next steps
1. Try `TRAIN_WINDOW_YEARS = 12` (currently None), rerun model.py and compare
   `policy_decisions.csv`.
2. Team-context minutes: keep each player's end-of-season team (prep drops it for traded
   players) to measure vacated minutes and give them to the best candidates.
3. Backtest cannot test the team minute budget (no historical rosters) or rookies.
4. Bench per-game lines are allocation leftovers (many pinned at 3 or 24 MPG); do not read
   them closely. Consider limiting rosters to ~15 plus anyone with real minutes.
5. Optional: manual injury-override file for known long-term injuries.
6. `src/ingest.py` refreshes `data/raw/current_rosters.csv` on every run (`FETCHED_AT` records
   the date; a failed fetch keeps the saved file). `src/export.py` is an empty placeholder.

## Working preferences
- The user asks for analysis first and wants to approve before building on bigger changes.
- Explain results in plain language, name the files changed, and say what was and was not
  tested. Deliver complete files that can drop into `src/`.
- No future data may leak into features (history features use seasons <= t only).
