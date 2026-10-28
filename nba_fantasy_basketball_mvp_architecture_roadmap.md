# NBA Fantasy Basketball Prediction Model (MVP)
## System Architecture, Mathematical Framework & Implementation Roadmap

---

## 1. Executive Summary & Core Objective

The primary objective of this project is to develop an automated, deterministic machine learning and statistical pipeline in Python to generate full-season player-level projections for standard **9-Category (9-Cat)** Fantasy Basketball formats.

The standard 9 categories evaluated are:
1. **Points (PTS)**
2. **Three-Pointers Made (3PM / FG3M)**
3. **Total Rebounds (REB)**
4. **Assists (AST)**
5. **Steals (STL)**
6. **Blocks (BLK)**
7. **Field Goal Percentage (FG%)**
8. **Free Throw Percentage (FT%)**
9. **Turnovers (TOV)**

Projections are generated as per-game statistical outputs, grouped and sorted by active NBA team depth to preserve environmental context.

---

## 2. Core Architectural Principles & Methodological Solutions

### A. The Opportunity vs. Skill Separation (Rate-Based Modeling)
* **Problem:** Modeling raw counting totals directly across seasons introduces extreme noise and bias caused by fluctuations in playing time (injuries, role promotions, benchings).
* **Solution:** Deconstruct player production into **Per-Minute Rates** ($Stat / MIN$) and **Projected Minutes Per Game ($MPG$)**. The model predicts rates of production per unit of court time, which are subsequently scaled by allocated playing time.

### B. Accurate Percentage Derivation (Volume Weighting)
* **Problem:** In fantasy basketball, shooting percentages cannot be treated as isolated rates. Shooting 85% on 2 free throws per game provides vastly less impact than 85% on 10 free throws per game. Furthermore, arithmetic averages of percentages are mathematically invalid over varying volumes.
* **Solution:** Never predict $FG\%$ or $FT\%$ directly. The pipeline independently models makes and attempts:
  * Field Goals: $\text{FGM}$ and $\text{FGA}$
  * Free Throws: $\text{FTM}$ and $\text{FTA}$
  * Composite efficiency is derived explicitly from predicted volume:
    $$\text{Projected } FG\% = \frac{\text{Projected } FGM}{\text{Projected } FGA}, \quad \text{Projected } FT\% = \frac{\text{Projected } FTM}{\text{Projected } FTA}$$

### C. The Zero-Sum Team Minutes Constraint ($240.0$ MPG Rule)
* **Problem:** Evaluating 15 players on a roster independently will over-project total team playing time (often summing to 280+ minutes).
* **Solution:** An NBA regulation game has exactly **240 total player-minutes** ($48 \text{ minutes} \times 5 \text{ positions}$). All individual baseline minutes estimates ($MPG_{\text{raw}}$) within an NBA roster must be rescaled using a zero-sum constraint:
  $$MPG_i = \left( \frac{MPG_{\text{raw}, i}}{\sum_{j \in \text{Team}} MPG_{\text{raw}, j}} \right) \times 240.0$$

### D. The Lagged Transition Paradigm ($t \to t+1$)
* **Problem:** Static machine learning models cannot infer career aging curves or regression to the mean without explicit temporal transitions.
* **Solution:** Models are trained on consecutive-season paired observations where the feature vector ($X$) describes player context in Year $t$, and the target vector ($y$) represents observed production in Year $t+1$.

---

## 3. Data Flow & System Schemas

```
                       +-----------------------------+
                       |      stats.nba.com API      |
                       +-----------------------------+
                                      |
                         (ingest.py - via nba_api)
                                      v
                       +-----------------------------+
                       |  data/raw/                  |
                       |  - historical_stats.csv     |
                       |  - current_rosters.csv      |
                       +-----------------------------+
                                      |
                     (prepare_training_data.py)
                                      v
                       +-----------------------------+
                       |  data/processed/            |
                       |  - training_data.csv        |
                       |  - training_pairs.csv       |
                       +-----------------------------+
                                      |
                           (model.py: Training)
                                      v
                       +-----------------------------+
                       |  11 Fitted Rate Regressors  |
                       |  (Ridge / Regularized ML)   |
                       +-----------------------------+
                                      |
                      (model.py: Inference Pipeline)
                       - Load active rosters
                       - Extract latest season stats (t)
                       - Predict rates (t+1)
                       - Allocate & normalize to 240m
                       - Scale Rates x Normalized MPG
                                      v
                       +-----------------------------+
                       |  data/processed/            |
                       |  projections_by_team.csv    |
                       +-----------------------------+
```

### Processed Dataset Schemas

#### 1. `data/processed/training_data.csv`
* **Granularity:** 1 row per player per completed NBA season.
* **Key Columns:**
  * `PLAYER_ID`, `PLAYER_NAME`, `SEASON`, `AGE`, `GP`, `MIN`
  * Per-minute rate features: `PTS_RATE`, `FG3M_RATE`, `FGM_RATE`, `FGA_RATE`, `FTM_RATE`, `FTA_RATE`, `REB_RATE`, `AST_RATE`, `STL_RATE`, `BLK_RATE`, `TOV_RATE`

#### 2. `data/processed/training_pairs.csv`
* **Granularity:** 1 row per consecutive player-season transition ($t \to t+1$).
* **Filter Conditions:** $MIN_{\text{prior}} \ge 200.0$ and $MIN_{\text{target}} \ge 200.0$ (eliminates small-sample garbage-time distortions).
* **Key Columns:**
  * Identifiers: `PLAYER_ID`, `PLAYER_NAME`, `YEAR_CURRENT`, `YEAR_NEXT`
  * Features ($X$): `AGE_prior`, `MIN_prior`, and all 11 `[STAT]_RATE_prior` columns
  * Target Labels ($y$): `MIN_target` (for sample weighting), and all 11 `[STAT]_RATE_target` columns

---

## 4. Detailed Implementation Roadmap

### Phase 1: Environment & Tooling Setup `[COMPLETED]`
* Initialize Python 3.10+ virtual environment (`venv`).
* Install core dependencies: `pandas`, `numpy`, `nba_api`, `scikit-learn`.
* Configure `.gitignore` to protect raw API caches and environment binaries.

### Phase 2: Offline Data Ingestion & Caching `[COMPLETED]`
* Script: `src/ingest.py`
* Configure custom desktop browser headers and global session patching for `nba_api` to prevent read timeouts.
* Query 4 historical seasons of box-score totals via `leaguedashplayerstats.LeagueDashPlayerStats`.
* Deduplicate mid-season multi-team player entries (retaining unified `"TOT"` season records).
* Fetch active NBA team rosters via `commonallplayers.CommonAllPlayers` in a single request.
* Persist locally to `data/raw/historical_stats.csv` and `data/raw/current_rosters.csv`.

### Phase 3: Dataset Preparation & Pairing `[COMPLETED]`
* Script: `src/prepare_training_data.py`
* Compute per-minute rates for all 11 statistical components.
* Construct lagged temporal pairs ($t \to t+1$) by joining on `PLAYER_ID` and `YEAR_CURRENT + 1 == YEAR_NEXT`.
* Apply the 200-minute sample cutoff across both prior and target seasons.
* Persist locally to `data/processed/training_data.csv` and `data/processed/training_pairs.csv`.

### Phase 4: Rate Regression Modeling Engine `[CURRENT STEP]`
* Script: `src/model.py`
* Train 11 independent regularized linear regression models (`Ridge(alpha=10.0)`) or Gradient Boosters.
* For each target stat $m \in \{FGM, FGA, FG3M, FTM, FTA, REB, AST, STL, BLK, PTS, TOV\}$:
  * Inputs ($X$): `[STAT]_RATE_prior`, `AGE_prior`, `MIN_prior`
  * Target ($y$): `[STAT]_RATE_target`
  * Weight ($w$): Sample weighting proportional to `MIN_target`
* Execute out-of-sample temporal validation:
  * Train set: Transitions up to 2024.
  * Validation set: Most recent 2024-to-2025 transition.
  * Metric: Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) per stat.

### Phase 5: Team Minutes Allocation & Stat Synthesis `[NEXT STEP]`
* Join trained model estimators with `data/raw/current_rosters.csv`.
* Derive baseline playing time ($MPG_{\text{raw}}$) from each player's latest available season.
* Implement hierarchical minute floors/ceilings (e.g., minimum 4.0 MPG, maximum 36.0 MPG).
* Apply the 240.0 team minute normalization function.
* Compute projected counting stats:
  $$\text{Stat}_{\text{projected}} = \text{Predicted Rate} \times \text{Normalized MPG}$$
* Compute derived shooting splits:
  $$\text{Projected } FG\% = \frac{\text{Projected } FGM}{\text{Projected } FGA}, \quad \text{Projected } FT\% = \frac{\text{Projected } FTM}{\text{Projected } FTA}$$

### Phase 6: Export & Quality Verification `[PENDING]`
* Script: `src/export.py` / `main.py`
* Aggregate full-season projections into a single dataset sorted by `TEAM_ABBREVIATION` and descending `PROJECTED_MPG`.
* Verification Suite:
  * Unit check: Assert all 30 teams have $\sum MPG = 240.0 \pm 0.01$.
  * Anomaly bounds: Verify shooting percentages reside within realistic boundaries ($0.30 \le FG\% \le 0.72$; $0.45 \le FT\% \le 0.96$).
* Export final human-readable artifact: `data/processed/projections_by_team.csv`.

---

## 5. Post-MVP Evolutionary Extensions

1. **Rookie & Zero-Sample Prior Models:** Introduce an empirical Bayes prior or specialized sub-model driven by draft position and college/G-League conversion translation matrices.
2. **Usage Rate ($USG\%$) Possession Re-allocation:** Dynamically rebalance shot attempts ($FGA, FTA$) and turnovers ($TOV$) when high-usage teammates depart or join rosters.
3. **9-Cat Category Normalization (Z-Scores / VORP):** Convert absolute stat lines into category Z-scores scaled against top-150 replacement levels to generate automated draft rankings and auction values.
4. **Availability & Games Played (GP) Discounting:** Model games played probability based on historical games missed, age, and back-to-back schedule density.