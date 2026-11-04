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
- Games played: currently a fitted ridge model of next-season games (`GAMES_METHOD =
  "fitted"`). The alternative, "blend", mixes hand rules (fluke vs chronic injury) with
  league-typical GP (starters 0.5 model weight, bench 0.2). Bench low GP is mostly role,
  not injury.
- Rate models train on all players, including retired ones
  (`RESTRICT_TO_CURRENT_ROSTER = False`), which removes survivor bias.
- Rookies / no-history players get replacement-level rates and typical games.
- Selection of model policies uses the same folds that score them, so reported rate
  metrics are slightly optimistic.

## Switches in project.py (set from backtest_recommendations.json)
- `GAMES_METHOD`: "fitted" (current; ridge model of next-season games) or "blend".
- `MPG_ADJUSTMENT`: "age" (current), "role" (adds per-minute production to model rotation
  jumps), or "none".
Recommendations are only trusted when the backtest ran >= 2 seasons. The 3-season backtest
on real data recommends the current settings, "fitted" and "age".

## Backtest findings so far (real data, seasons 2024/2025/2026, 500+ minute players)
- VALUE_Spearman: fitted 0.734/0.699/0.567, blend 0.733/0.694/0.540, naive (repeat last
  year) 0.713/0.661/0.532. The pipeline beats naive every year. Age adjustment helps every
  year (no-age 0.725/0.687/0.531). Rate model adds ~+0.005 consistently.
- Games MAE: fitted 10.8/11.6/11.6, blend 11.2/11.6/11.8, constant-66 baseline
  11.3/12.4/11.4, naive 13.9/15.8/16.6. Fitted misses the constant-66 baseline in 2026.
  Blend weight grid is flat; keep 0.5/0.2.
- Fitted passes the recommendation test in 2024 and 2026 but not 2025 (slightly worse MAE).
- Weakness: games-played ranking is still worse than last season's raw games. GP_Spearman:
  fitted 0.307/0.268/0.161, blend 0.331/0.281/0.031, naive 0.346/0.306/0.223. Fitted
  improves on blend only in 2026, a hard season for everyone.
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
