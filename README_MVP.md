# NBA Fantasy Basketball Prediction Model MVP: Architecture & Roadmap

## 1. Project Context & Objectives

### Core Purpose

The primary objective of this Minimum Viable Product (MVP) is to generate deterministic, full-season individual player statistical projections for standard **9-category (9-cat)** fantasy basketball formats. The final output provides player-level per-game projections organized and sorted by NBA team depth.

The 9 standard categories evaluated are:

* **Points (PTS)**

* **Three-Pointers Made (3PM)**

* **Total Rebounds (REB)**

* **Assists (AST)**

* **Steals (STL)**

* **Blocks (BLK)**

* **Field Goal Percentage (FG%)**

* **Free Throw Percentage (FT%)**

* **Turnovers (TO)**

### Key Analytical Challenges Solved

1. **Separating Skill from Opportunity:** Counting stats are primarily driven by playing time. Modeling counting stats directly across roster or role changes introduces severe variance. This system projects **per-minute efficiency rates** and scales them by projected playing time.

2. **The Zero-Sum Roster Constraint:** Every standard regulation NBA game contains exactly **240 total player-minutes** (48 minutes $\times$ 5 positions). Evaluating players in isolation leads to over-projecting total team minutes and stats. This pipeline enforces a strict 240.0 minutes per game (MPG) ceiling per team.

3. **Shooting Percentage Volume Weighting:** Directly averaging shooting percentages (e.g., $FG\%$ or $FT\%$) distorts player value. A player shooting 85% on 2 free throws per game has vastly different fantasy impact than one shooting 85% on 10 attempts. The model predicts **makes and attempts separately** ($FGM, FGA, FTM, FTA$) and derives true volume-weighted percentages.

4. **Data Sourcing Stability:** Scraping dynamic websites (like Basketball-Reference) introduces rate-limit bans (e.g., 20 requests/minute rules) and parser fragility. The pipeline utilizes structured API pulls cached locally.

## 2. Mathematical Framework

### Rate-Based Calculations

For every player $i$ and counting metric $m \in \{PTS, 3PM, FGM, FGA, FTM, FTA, REB, AST, STL, BLK, TO\}$:

$$
\text{Rate}_{m} = \frac{\text{Total Metric}_m}{\text{Total Minutes Played}}
$$

### 3-Year Exponential Decay Baseline

Historical performance uses exponential decay weighting to prioritize recent seasons:

$$
\text{Baseline Rate}_m = (0.60 \times R_{t-1}) + (0.27 \times R_{t-2}) + (0.13 \times R_{t-3})
$$


*Where* $R_{t-1}$ *is the most recent season,* $R_{t-2}$ *is two seasons prior, and* $R_{t-3}$ *is three seasons prior.*

If a player has fewer than 3 seasons of historical data, weights are normalized:

* **2 Seasons Available:** Season $t-1 = 0.69$ ($\frac{0.60}{0.87}$), Season $t-2 = 0.31$ ($\frac{0.27}{0.87}$)

* **1 Season Available:** Season $t-1 = 1.00$

### Age Curve Multipliers

A progression/regression factor ($\Delta_{\text{age}}$) adjusts baseline efficiency rates based on player age:

$$
\text{Adjusted Rate}_m = \text{Baseline Rate}_m \times (1 + \Delta_{\text{age}})
$$

| Age Bracket | Lifecycle Phase | Rate Multiplier ($\Delta_{\text{age}}$) | 
 | ----- | ----- | ----- | 
| $\le 24$ | Developmental Growth | $+3\%$ ($+0.03$) | 
| $25 - 28$ | Peak Prime | $0\%$ ($0.00$) | 
| $29 - 32$ | Early Career Decline | $-3\%$ ($-0.03$) | 
| $\ge 33$ | Late Career Regression | $-6\%$ ($-0.06$) | 

### Team Minutes Allocation (240-Minute Normalization)

Each active player on a team's roster is assigned an initial minute estimate ($\text{MPG}_{\text{init}}$), defaulted to their prior season's MPG (or 12.0 MPG for rookies/deep reserves).

For a team roster with $N$ players:

$$
\text{Team Total MPG} = \sum_{j=1}^{N} \text{MPG}_{\text{init}, j}
$$

$$
\text{Projected MPG}_i = \left( \frac{\text{MPG}_{\text{init}, i}}{\text{Team Total MPG}} \right) \times 240.0
$$

### Final Per-Game Projections

* **Counting Stats:** $\text{Projected Stat}_m = \text{Adjusted Rate}_m \times \text{Projected MPG}$

* **Shooting Percentages:**
  

  $$
  \text{Projected FG\%} = \frac{\text{Projected FGM}}{\text{Projected FGA}}
  $$

  $$
  \text{Projected FT\%} = \frac{\text{Projected FTM}}{\text{Projected FTA}}
  $$

## 3. Directory & Architecture Structure

```
nba-fantasy-mvp/
├── data/
│   ├── raw/                  # Offline historical dumps & roster pulls from nba_api
│   └── processed/            # Final consolidated projections and intermediate tables
├── src/
│   ├── __init__.py
│   ├── ingest.py             # Pulls and caches NBA API data locally
│   ├── model.py              # Rate math, age curves, 240-minute normalization
│   └── export.py             # Formats and writes projections grouped by team
├── .gitignore                # Excludes venv/, data/raw/, and caches
├── main.py                   # Master script executing ingest -> model -> export
└── requirements.txt          # Python dependencies

```

## 4. Step-by-Step Implementation Roadmap

```
+-------------------------------------------------------------------------+
|                      PHASE 1: ENVIRONMENT SETUP                         |
|  - Initialize Python virtual environment                                |
|  - Install dependencies: pandas, numpy, nba_api, scikit-learn           |
|  - Configure .gitignore                                                 |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                      PHASE 2: DATA EXTRACTION                           |
|  - src/ingest.py: Fetch 3 seasons of player box-score totals            |
|  - Query active 30-team roster assignments                              |
|  - Cache tables to data/raw/historical_stats.csv & current_rosters.csv  |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  PHASE 3: STATISTICAL MODELING ENGINE                   |
|  - src/model.py: Convert raw stats to per-minute rates                  |
|  - Apply 3-year exponential decay weights (0.60 / 0.27 / 0.13)          |
|  - Apply age curve multipliers                                          |
|  - Normalize team MPG to exactly 240.0 minutes                          |
|  - Compute final counting stats and derive FG% & FT%                    |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    PHASE 4: EXPORT & VALIDATION                         |
|  - src/export.py: Sort players by Team and descending Projected MPG     |
|  - Run sanity tests: sum(team_mpg) == 240.0                             |
|  - Export to data/processed/fantasy_projections_by_team.csv             |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                   PHASE 5: POST-MVP EXTENSIONS                          |
|  - Convert 9-cat stats to Z-Scores & VORP draft rankings                |
|  - Model team-level Usage Rate (USG%) redistribution                    |
|  - Factor injury history & projected Games Played (GP)                  |
+-------------------------------------------------------------------------+

```

## 5. Technical Requirements & Dependencies

Create a `requirements.txt` file containing:

```
pandas>=2.0.0
numpy>=1.24.0
nba_api>=1.4.0
scikit-learn>=1.3.0

```

Install via terminal:

```
pip install -r requirements.txt

```