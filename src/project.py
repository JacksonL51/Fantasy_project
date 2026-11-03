"""Phase 5: team minutes allocation and per-game stat synthesis.

Run from the project root:  python src/project.py

Inputs
------
data/raw/current_rosters.csv        active rosters (needs a player id column
                                    and TEAM_ABBREVIATION; ROSTERSTATUS is
                                    used when present)
data/processed/training_data.csv    one row per player per completed season
                                    (PLAYER_ID, PLAYER_NAME, SEASON_YEAR, AGE,
                                    GP, MIN as season TOTAL minutes, and the
                                    11 [STAT]_RATE columns)
data/processed/rate_models.joblib   bundle written by model.py

Output
------
data/processed/projections_by_team.csv   per-game projections, sorted by team
                                         and descending PROJECTED_MPG

Steps (one function each)
-------------------------
1. load_roster / load_history / latest_season_rows / build_inference_table
2. replacement_rates + predict_with_credibility (low-minute / rookie handling)
3. predict_rates (imported from model.py; applies the per-stat policy)
4. raw_minutes_and_gp (three-season recency and durability profiles)
5. allocate_team_season_minutes / allocate_all_teams (top-nine core by
   healthy active MPG and cohort-informed fringe allocation)
6. synthesize_stats
7. validate_projections / print_review
"""

import sys

import numpy as np
import pandas as pd

from model import (
    PROCESSED_DIR,
    TARGET_STATS,
    load_model_bundle,
    predict_rates,
)

PROJECT_ROOT = PROCESSED_DIR.parents[1]
ROSTERS_PATH = PROJECT_ROOT / "data" / "raw" / "current_rosters.csv"
TRAINING_DATA_PATH = PROCESSED_DIR / "training_data.csv"
PROJECTIONS_PATH = PROCESSED_DIR / "projections_by_team.csv"

# Minutes allocation
GAMES_PER_SEASON = 82
TEAM_SEASON_MINUTES = GAMES_PER_SEASON * 240.0  # 19,680.0 minutes per team
MIN_HEALTHY_MPG = 4.0
MAX_HEALTHY_MPG = 37.0
CORE_ROTATION_SIZE = 9
TIER_1_MINUTES = 2000.0
TIER_2_MINUTES = 1400.0
TIER_3_MINUTES = 750.0
# Fringe (rank 10+) players keep their own credibility-blended active MPG
# instead of a flat 12; GP is whatever the allocated minutes imply at that MPG.
FRINGE_COHORT_MPG = 8.0  # what an unknown / low-credibility fringe player gets
FRINGE_MIN_MPG = 5.0
FRINGE_MAX_MPG = 24.0
FRINGE_MIN_GP = 8.0
FRINGE_MAX_GP = 78.0
FRINGE_CAP_STRETCH = (1.0, 1.25, 1.5, 2.0)  # widen per-player caps only if needed
# A player needs this many average season minutes to claim a top-nine core slot
# on healthy MPG alone (stops 300-minute small samples from outranking regulars).
CORE_MIN_WEIGHTED_MINUTES = 700.0
MINUTES_CREDIBILITY_CAREER_MINUTES = 1500.0
COHORT_PRIOR_MINUTES = {
    "young": 550.0,
    "general": 300.0,
    "veteran": 325.0,
}
CHRONIC_GP_CEILING = 58.0
FLUKE_GP_SPREAD = 30.0
FLUKE_GP_MINIMUM = 45.0
FLUKE_HEALTHY_GP = 58.0
FRAGILE_GP_BASELINE = 47.5
CHRONIC_INJURY_BLEND = 0.5
STARTER_GP_BASELINE = 66.5
RECENCY_WEIGHTS = {0: 0.60, 1: 0.27, 2: 0.13}

# Expected Games Played priors
DEFAULT_STARTER_GP = 68.0
DEFAULT_BENCH_GP = 55.0
DEFAULT_ROOKIE_GP = 45.0

# Low-sample handling. The models were trained only on players with >= 200
# minutes in both seasons, so smaller samples are shrunk toward replacement.
CREDIBILITY_MINUTES = 200.0
STALE_CREDIBILITY_CAP = 0.5  # max model weight if the player missed last season
REPLACEMENT_MIN_RANGE = (200.0, 800.0)  # bench-level pool for replacement rates

# Output checks
FG_PCT_BOUNDS = (0.30, 0.72)
FT_PCT_BOUNDS = (0.45, 0.96)
TEAM_PTS_WARN_RANGE = (100.0, 130.0)
EXPECTED_TEAMS = 30

OUTPUT_STATS = [
    "PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FTM", "FTA",
]


# --------------------------------------------------------------------------
# Step 1: inference table
# --------------------------------------------------------------------------
def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"Could not find a {label} column. Tried {list(candidates)}; "
        f"available columns: {list(df.columns)}"
    )


def load_roster(path=ROSTERS_PATH) -> pd.DataFrame:
    """Active players with a team: PLAYER_ID, TEAM_ABBREVIATION, ROSTER_NAME."""
    raw = pd.read_csv(path)
    id_column = _pick_column(raw, ("PLAYER_ID", "PERSON_ID"), "player id")
    team_column = _pick_column(raw, ("TEAM_ABBREVIATION",), "team abbreviation")
    name_column = next(
        (c for c in ("PLAYER_NAME", "DISPLAY_FIRST_LAST") if c in raw.columns), None
    )

    if "ROSTERSTATUS" in raw.columns:
        active = pd.to_numeric(raw["ROSTERSTATUS"], errors="coerce") == 1
    else:
        active = pd.Series(True, index=raw.index)

    roster = pd.DataFrame(
        {
            "PLAYER_ID": pd.to_numeric(raw[id_column], errors="coerce"),
            "TEAM_ABBREVIATION": raw[team_column].fillna("").astype(str).str.strip(),
            "ROSTER_NAME": raw[name_column] if name_column else pd.NA,
        }
    )
    roster = roster.loc[
        active & roster["PLAYER_ID"].notna() & (roster["TEAM_ABBREVIATION"] != "")
    ].copy()
    roster["PLAYER_ID"] = roster["PLAYER_ID"].astype("int64")

    duplicates = int(roster["PLAYER_ID"].duplicated().sum())
    if duplicates:
        print(f"Warning: dropping {duplicates} duplicate roster rows.")
        roster = roster.drop_duplicates("PLAYER_ID", keep="first")
    print(
        f"Roster: {len(roster)} active players on "
        f"{roster['TEAM_ABBREVIATION'].nunique()} teams"
    )
    return roster.reset_index(drop=True)


def _season_year(series: pd.Series) -> pd.Series:
    """Integer season year. "2025-26" -> 2026; a bare number is used as is."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype("int64")
    start = pd.to_numeric(
        series.astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
    )
    if start.isna().any():
        raise ValueError("Could not parse every SEASON value into a year.")
    return (start + 1).astype("int64")


def load_history(path=TRAINING_DATA_PATH) -> pd.DataFrame:
    """Season-level history with valid minutes, games and finite rates."""
    history = pd.read_csv(path)
    rate_columns = [f"{stat}_RATE" for stat in TARGET_STATS]
    required = ["PLAYER_ID", "SEASON", "AGE", "GP", "MIN", *rate_columns]
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise ValueError("training_data.csv is missing columns: " + ", ".join(missing))

    if "SEASON_YEAR" in history.columns:
        history["SEASON_YEAR"] = pd.to_numeric(
            history["SEASON_YEAR"], errors="coerce"
        )
    else:
        history["SEASON_YEAR"] = _season_year(history["SEASON"])
    for column in ["PLAYER_ID", "AGE", "GP", "MIN", *rate_columns]:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    valid = (
        history["PLAYER_ID"].notna()
        & history["SEASON_YEAR"].notna()
        & (history["MIN"] > 0)
        & (history["GP"] > 0)
        & np.isfinite(history[rate_columns].to_numpy(dtype=float)).all(axis=1)
    )
    history = history.loc[valid].copy()
    history["PLAYER_ID"] = history["PLAYER_ID"].astype("int64")
    if "PLAYER_NAME" not in history.columns:
        history["PLAYER_NAME"] = pd.NA
    return history


def latest_season_rows(history: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Build three-season recency-weighted player profiles."""
    latest_year = int(history["SEASON_YEAR"].max())
    ordered = history.sort_values(["PLAYER_ID", "SEASON_YEAR"]).drop_duplicates(
        ["PLAYER_ID", "SEASON_YEAR"], keep="last"
    )
    recent = ordered.loc[ordered["SEASON_YEAR"] >= latest_year - 2].copy()
    recent["RECENCY_WEIGHT"] = (
        latest_year - recent["SEASON_YEAR"]
    ).map(RECENCY_WEIGHTS)

    profile_rows = []
    rate_columns = [f"{stat}_RATE" for stat in TARGET_STATS]
    career_minutes = ordered.groupby("PLAYER_ID")["MIN"].sum()
    for player_id, seasons in recent.groupby("PLAYER_ID", sort=False):
        latest_player_season = seasons.iloc[-1].copy()
        season_weights = seasons["RECENCY_WEIGHT"].to_numpy(
            dtype=float, copy=True
        )
        season_weights /= season_weights.sum()
        for column in rate_columns:
            latest_player_season[column] = float(
                np.average(seasons[column].to_numpy(dtype=float), weights=season_weights)
            )

        gp = seasons["GP"].to_numpy(dtype=float)
        latest_player_season["WEIGHTED_GP"] = float(
            np.average(gp, weights=season_weights)
        )
        latest_player_season["GP_MIN"] = float(np.min(gp))
        latest_player_season["GP_MAX"] = float(np.max(gp))
        latest_player_season["GP_MEDIAN"] = float(np.median(gp))
        latest_player_season["SEASONS_IN_WINDOW"] = len(seasons)
        latest_player_season["WEIGHTED_SEASON_MINUTES"] = float(
            np.average(
                seasons["MIN"].to_numpy(dtype=float),
                weights=season_weights,
            )
        )
        latest_player_season["PRIOR_CAREER_MINUTES"] = float(
            career_minutes.loc[player_id]
        )

        healthy_seasons = seasons.loc[seasons["GP"] >= 15]
        if healthy_seasons.empty:
            latest_player_season["BASELINE_ACTIVE_MPG"] = np.nan
            latest_player_season["ELIGIBLE_MPG_SEASONS"] = 0
        else:
            healthy_weights = healthy_seasons["RECENCY_WEIGHT"].to_numpy(
                dtype=float, copy=True
            )
            healthy_weights /= healthy_weights.sum()
            season_mpg = (
                healthy_seasons["MIN"].to_numpy(dtype=float)
                / healthy_seasons["GP"].to_numpy(dtype=float)
            )
            latest_player_season["BASELINE_ACTIVE_MPG"] = float(
                np.average(season_mpg, weights=healthy_weights)
            )
            latest_player_season["ELIGIBLE_MPG_SEASONS"] = len(healthy_seasons)
        latest_player_season["PLAYER_ID"] = player_id
        profile_rows.append(latest_player_season)

    profiles = pd.DataFrame(profile_rows).drop(columns=["RECENCY_WEIGHT"])
    return profiles.reset_index(drop=True), latest_year


def replacement_rates(
    latest: pd.DataFrame, latest_year: int, bundle: dict[str, object]
) -> pd.Series:
    """Minute-weighted bench-level rates, used for rookies and small samples."""
    low, high = REPLACEMENT_MIN_RANGE
    pool = latest.loc[
        (latest["SEASON_YEAR"] == latest_year)
        & (latest["MIN"] >= low)
        & (latest["MIN"] < high)
    ]
    if len(pool) < 10:
        print("Warning: tiny replacement pool; using the league training mean.")
        return pd.Series(bundle["prior_shrinkage_reference_by_stat"]).reindex(
            TARGET_STATS
        )
    return pd.Series(
        {
            stat: float(np.average(pool[f"{stat}_RATE"], weights=pool["MIN"]))
            for stat in TARGET_STATS
        }
    )


def build_inference_table(
    roster: pd.DataFrame,
    latest: pd.DataFrame,
    latest_year: int,
    replacement: pd.Series,
) -> pd.DataFrame:
    """Roster joined to latest stats, with the exact columns the models expect."""
    history_for_merge = latest.drop(
        columns=["TEAM_ABBREVIATION"], errors="ignore"
    )
    table = roster.merge(
        history_for_merge, on="PLAYER_ID", how="left", validate="one_to_one"
    )
    table["NO_PRIOR_DATA"] = table["SEASON_YEAR"].isna()
    table["STALE_DATA"] = ~table["NO_PRIOR_DATA"] & (table["SEASON_YEAR"] < latest_year)

    for stat in TARGET_STATS:
        table[f"{stat}_RATE_prior"] = table[f"{stat}_RATE"].fillna(replacement[stat])
    table["AGE_prior"] = table["AGE"].fillna(latest["AGE"].median())
    table["AGE_SQ"] = table["AGE_prior"] ** 2
    table["MIN_prior"] = table["MIN"].fillna(0.0)

    print(
        f"Matched to history: {int((~table['NO_PRIOR_DATA']).sum())}; "
        f"no prior data: {int(table['NO_PRIOR_DATA'].sum())}; "
        f"stale (missed {latest_year}): {int(table['STALE_DATA'].sum())}"
    )
    return table


# --------------------------------------------------------------------------
# Steps 2-3: rate predictions with credibility blending
# --------------------------------------------------------------------------
def predict_with_credibility(
    bundle: dict[str, object], table: pd.DataFrame, replacement: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Model rates blended toward replacement for small or stale samples.

    weight = clip(MIN_prior / 200, 0, 1), capped at 0.5 for stale rows.
    Rookies have MIN_prior = 0, so they get pure replacement rates.
    """
    predicted = predict_rates(bundle, table)
    weight = np.clip(table["MIN_prior"] / CREDIBILITY_MINUTES, 0.0, 1.0)
    weight = weight.where(~table["STALE_DATA"], weight.clip(upper=STALE_CREDIBILITY_CAP))

    replacement_matrix = pd.DataFrame(
        np.outer(1.0 - weight.to_numpy(), replacement.reindex(TARGET_STATS).to_numpy()),
        index=table.index,
        columns=TARGET_STATS,
    )
    blended = (predicted[TARGET_STATS].mul(weight, axis=0) + replacement_matrix).clip(
        lower=0.0
    )
    print(f"Players blended toward replacement (weight < 1): {int((weight < 1).sum())}")
    return blended, weight


# --------------------------------------------------------------------------
# Step 4: raw minutes per game
# --------------------------------------------------------------------------
def raw_minutes_and_gp(table: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return multi-season healthy MPG and injury-adjusted projected games played.
    """
    active_mpg = table["BASELINE_ACTIVE_MPG"].fillna(MIN_HEALTHY_MPG).clip(
        lower=MIN_HEALTHY_MPG, upper=MAX_HEALTHY_MPG
    )
    weighted_gp = table["WEIGHTED_GP"].fillna(DEFAULT_ROOKIE_GP)
    min_gp = table["GP_MIN"]
    max_gp = table["GP_MAX"]
    spread = max_gp - min_gp

    fluke_injury = (
        spread.ge(FLUKE_GP_SPREAD)
        & min_gp.lt(FLUKE_GP_MINIMUM)
        & max_gp.ge(FLUKE_HEALTHY_GP)
    )
    chronic_fragility = max_gp.lt(CHRONIC_GP_CEILING) & ~fluke_injury
    proj_gp = 0.70 * weighted_gp + 0.30 * STARTER_GP_BASELINE
    proj_gp = proj_gp.where(~fluke_injury, table["GP_MEDIAN"])
    chronic_projection = (
        (1 - CHRONIC_INJURY_BLEND) * weighted_gp
        + CHRONIC_INJURY_BLEND * FRAGILE_GP_BASELINE
    )
    proj_gp = proj_gp.where(~chronic_fragility, chronic_projection)
    proj_gp = proj_gp.fillna(DEFAULT_ROOKIE_GP).clip(lower=15.0, upper=78.0)

    blended_mpg = table["ELIGIBLE_MPG_SEASONS"].fillna(0).lt(3)
    return active_mpg, proj_gp, blended_mpg


# --------------------------------------------------------------------------
# Step 5: Healthy-MPG core and cohort-informed fringe allocation
# --------------------------------------------------------------------------

def allocate_team_season_minutes(
    active_mpg: np.ndarray,
    proj_gp: np.ndarray,
    weighted_season_minutes: np.ndarray,
    prior_career_minutes: np.ndarray,
    ages: np.ndarray,
    target_total: float = TEAM_SEASON_MINUTES,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Anchor the top-nine healthy-MPG core and distribute remaining minutes by
    weighted-minute baseline. This keeps injury-shortened seasons from demoting
    established high-MPG players into the 12-MPG fringe pool.
    """
    active_mpg = np.asarray(active_mpg, dtype=float)
    proj_gp = np.asarray(proj_gp, dtype=float)
    weighted_season_minutes = np.asarray(weighted_season_minutes, dtype=float)
    prior_career_minutes = np.asarray(prior_career_minutes, dtype=float)
    ages = np.asarray(ages, dtype=float)
    arrays = (
        active_mpg,
        proj_gp,
        weighted_season_minutes,
        prior_career_minutes,
        ages,
    )
    if any(array.ndim != 1 for array in arrays) or len(
        {len(array) for array in arrays}
    ) != 1:
        raise ValueError("All team allocation inputs must be aligned one-dimensional arrays.")
    if len(active_mpg) == 0 or not np.isfinite(active_mpg).all():
        raise ValueError("Team allocation requires non-empty finite active MPG values.")
    if not np.isfinite(proj_gp).all() or (proj_gp <= 0).any():
        raise ValueError("Team allocation requires finite positive projected GP values.")
    if (
        not np.isfinite(weighted_season_minutes).all()
        or (weighted_season_minutes < 0).any()
    ):
        raise ValueError("Weighted season minutes must be finite and non-negative.")
    if (
        not np.isfinite(prior_career_minutes).all()
        or (prior_career_minutes < 0).any()
    ):
        raise ValueError("Prior career minutes must be finite and non-negative.")
    if target_total <= 0 or not np.isfinite(target_total):
        raise ValueError("target_total must be finite and positive.")

    player_order = np.lexsort((-weighted_season_minutes, -active_mpg))
    credible = weighted_season_minutes[player_order] >= CORE_MIN_WEIGHTED_MINUTES
    player_order = np.concatenate([player_order[credible], player_order[~credible]])
    core_indices = player_order[:CORE_ROTATION_SIZE]
    fringe_indices = player_order[CORE_ROTATION_SIZE:]
    season_minutes = np.zeros_like(active_mpg)
    projected_gp = proj_gp.copy()
    minute_baseline = weighted_season_minutes.copy()
    cohort_prior = np.zeros_like(active_mpg)
    credibility = np.ones_like(active_mpg)
    role_tier = np.select(
        [
            weighted_season_minutes > TIER_1_MINUTES,
            weighted_season_minutes >= TIER_2_MINUTES,
            weighted_season_minutes >= TIER_3_MINUTES,
        ],
        ["Tier 1", "Tier 2", "Tier 3"],
        default="Tier 4",
    )

    season_minutes[core_indices] = active_mpg[core_indices] * proj_gp[core_indices]
    remaining = target_total - season_minutes[core_indices].sum()
    if remaining < -1e-8:
        raise ValueError(
            "Anchored top-nine core workloads exceed the team's season minutes: "
            f"core={season_minutes[core_indices].sum():.2f}, target={target_total:.2f}."
        )
    tier4_indices = np.flatnonzero(role_tier == "Tier 4")
    known_ages = np.isfinite(ages[tier4_indices])
    tier4_ages = ages[tier4_indices]
    tier4_cohort_prior = np.full(
        len(tier4_indices), COHORT_PRIOR_MINUTES["general"]
    )
    tier4_cohort_prior[known_ages & (tier4_ages <= 22)] = (
        COHORT_PRIOR_MINUTES["young"]
    )
    tier4_cohort_prior[known_ages & (tier4_ages >= 29)] = (
        COHORT_PRIOR_MINUTES["veteran"]
    )
    cohort_prior[tier4_indices] = tier4_cohort_prior
    credibility[tier4_indices] = np.minimum(
        1.0,
        prior_career_minutes[tier4_indices]
        / MINUTES_CREDIBILITY_CAREER_MINUTES,
    )
    minute_baseline[tier4_indices] = (
        credibility[tier4_indices] * weighted_season_minutes[tier4_indices]
        + (1.0 - credibility[tier4_indices]) * tier4_cohort_prior
    )
    if not len(fringe_indices):
        if abs(remaining) > 1.0:
            raise ValueError("No fringe players are available for minute reconciliation.")
        return (
            season_minutes / projected_gp,
            season_minutes,
            projected_gp,
            role_tier,
            minute_baseline,
            credibility,
            cohort_prior,
        )

    # Tier 2/3 fringe players retain their individual weighted-minute baseline.
    fringe_weights = minute_baseline[fringe_indices]
    if (fringe_weights <= 0).any():
        raise ValueError("Every fringe player must have a positive minute baseline.")

    # Each fringe player's own playing level: active MPG blended toward the
    # cohort MPG by career-minutes credibility.
    fringe_credibility = np.minimum(
        1.0, prior_career_minutes[fringe_indices] / MINUTES_CREDIBILITY_CAREER_MINUTES
    )
    fringe_mpg = np.clip(
        fringe_credibility * active_mpg[fringe_indices]
        + (1.0 - fringe_credibility) * FRINGE_COHORT_MPG,
        FRINGE_MIN_MPG,
        FRINGE_MAX_MPG,
    )

    base_upper = fringe_mpg * FRINGE_MAX_GP
    for stretch in FRINGE_CAP_STRETCH:
        upper = base_upper * stretch
        if remaining <= upper.sum() + 1e-8:
            break
    else:
        raise ValueError(
            "Team fringe pool cannot absorb the remaining minutes: "
            f"remaining={remaining:.2f}, maximum={upper.sum():.2f}."
        )

    low_scale, high_scale = 0.0, max(1.0, remaining / fringe_weights.sum())
    for _ in range(100):
        scale = (low_scale + high_scale) / 2.0
        allocated = np.minimum(fringe_weights * scale, upper)
        if allocated.sum() < remaining:
            low_scale = scale
        else:
            high_scale = scale
    fringe_minutes = np.minimum(
        fringe_weights * ((low_scale + high_scale) / 2.0), upper
    )
    residual = remaining - fringe_minutes.sum()
    if abs(residual) > 1e-8:
        capacities = (
            upper - fringe_minutes if residual > 0 else fringe_minutes
        )
        candidates = np.flatnonzero(capacities > 1e-8)
        if not len(candidates):
            raise ValueError("Unable to reconcile team season minutes under fringe bounds.")
        for candidate in candidates:
            adjustment = np.sign(residual) * min(abs(residual), capacities[candidate])
            fringe_minutes[candidate] += adjustment
            residual -= adjustment
            if abs(residual) <= 1e-8:
                break

    season_minutes[fringe_indices] = fringe_minutes
    # Games follow from minutes at the player's own MPG; if that would exceed a
    # full season, the extra minutes show up as a higher MPG instead.
    projected_gp[fringe_indices] = np.clip(
        fringe_minutes / fringe_mpg, FRINGE_MIN_GP, FRINGE_MAX_GP
    )
    norm_active_mpg = season_minutes / projected_gp
    total_error = abs(season_minutes.sum() - target_total)
    if total_error > 1.0:
        raise ValueError(
            f"Team minutes reconciliation error is {total_error:.3f}, over 1 minute."
        )
    return (
        norm_active_mpg,
        season_minutes,
        projected_gp,
        role_tier,
        minute_baseline,
        credibility,
        cohort_prior,
    )


def allocate_all_teams(
    table: pd.DataFrame,
    active_mpg: pd.Series,
    proj_gp: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series]:
    norm_mpg = pd.Series(np.nan, index=table.index, name="PROJECTED_MPG")
    norm_season_min = pd.Series(np.nan, index=table.index, name="PROJECTED_SEASON_MIN")
    final_gp = pd.Series(np.nan, index=table.index, name="PROJECTED_GP")

    for team, index in table.groupby("TEAM_ABBREVIATION").groups.items():
        try:
            m, s, gp, tiers, baseline, credibility, cohort_prior = (
                allocate_team_season_minutes(
                    active_mpg.loc[index].to_numpy(),
                    proj_gp.loc[index].to_numpy(),
                    table.loc[index, "WEIGHTED_SEASON_MINUTES"]
                    .fillna(0.0)
                    .to_numpy(),
                    table.loc[index, "PRIOR_CAREER_MINUTES"]
                    .fillna(0.0)
                    .to_numpy(),
                    pd.to_numeric(
                        table.loc[index, "AGE"], errors="coerce"
                    ).to_numpy(),
                )
            )
        except ValueError as error:
            raise ValueError(f"Team {team}: {error}") from error
        norm_mpg.loc[index] = m
        norm_season_min.loc[index] = s
        final_gp.loc[index] = gp
        table.loc[index, "ROLE_TIER"] = tiers
        table.loc[index, "MINUTES_BASELINE"] = baseline
        table.loc[index, "MINUTES_CREDIBILITY"] = credibility
        table.loc[index, "COHORT_PRIOR_MINUTES"] = cohort_prior

    return norm_mpg, norm_season_min, final_gp


# --------------------------------------------------------------------------
# Step 6: stat synthesis
# --------------------------------------------------------------------------
def synthesize_stats(
    table: pd.DataFrame,
    rates: pd.DataFrame,
    raw_mpg: pd.Series,
    proj_gp: pd.Series,
    mpg: pd.Series,
    season_min: pd.Series,
    weight: pd.Series,
    mpg_blended: pd.Series,
) -> pd.DataFrame:
    """Per-game stats = rate * normalized active MPG; Total stats = rate * season_min."""
    projections = pd.DataFrame(
        {
            "PLAYER_ID": table["PLAYER_ID"],
            "PLAYER_NAME": table["PLAYER_NAME"].fillna(table["ROSTER_NAME"]),
            "TEAM_ABBREVIATION": table["TEAM_ABBREVIATION"],
            "PROJECTED_GP": proj_gp,
            "PROJECTED_MPG": mpg.round(1),
            "PROJECTED_SEASON_MIN": season_min.round(1),
            "ROLE_TIER": table["ROLE_TIER"],
            "WEIGHTED_SEASON_MINUTES": table["WEIGHTED_SEASON_MINUTES"].fillna(0.0),
            "MINUTES_BASELINE": table["MINUTES_BASELINE"],
            "MINUTES_CREDIBILITY": table["MINUTES_CREDIBILITY"],
            "COHORT_PRIOR_MINUTES": table["COHORT_PRIOR_MINUTES"],
        }
    )
    
    # 1. Per-game stat projections (active fantasy production)
    for stat in OUTPUT_STATS:
        projections[f"PROJ_{stat}"] = rates[stat] * mpg

    # 2. Total season volume (for Roto / Total Points draft ranks)
    for stat in OUTPUT_STATS:
        projections[f"TOTAL_{stat}"] = rates[stat] * season_min

    # Shooting percentage splits derived from volume
    projections["PROJ_FG_PCT"] = (
        projections["TOTAL_FGM"] / projections["TOTAL_FGA"]
    ).where(projections["TOTAL_FGA"] > 0)
    projections["PROJ_FT_PCT"] = (
        projections["TOTAL_FTM"] / projections["TOTAL_FTA"]
    ).where(projections["TOTAL_FTA"] > 0)

    # Diagnostic metadata
    projections["RAW_MPG"] = raw_mpg
    projections["CREDIBILITY_WEIGHT"] = weight
    projections["DATA_SEASON"] = table["SEASON_YEAR"]
    projections["NO_PRIOR_DATA"] = table["NO_PRIOR_DATA"]
    projections["STALE_DATA"] = table["STALE_DATA"]
    projections["MPG_BLENDED"] = mpg_blended
    projections["LAST_PTS_PG"] = table["PTS_RATE"] * table["MIN"] / table["GP"]
    
    return projections

# --------------------------------------------------------------------------
# Step 7: checks
# --------------------------------------------------------------------------
def validate_projections(projections: pd.DataFrame) -> None:
    # 1. Check finite values and non-negativity
    stat_columns = [f"PROJ_{stat}" for stat in OUTPUT_STATS] + [
        "PROJ_FG_PCT",
        "PROJ_FT_PCT",
        "PROJECTED_MPG",
        "PROJECTED_GP",
        "PROJECTED_SEASON_MIN",
    ]
    if not np.isfinite(projections[stat_columns].to_numpy(dtype=float)).all():
        raise ValueError("Projections contain missing or non-finite values.")
    if (projections[[f"PROJ_{stat}" for stat in OUTPUT_STATS]] < 0).any().any():
        raise ValueError("Projections contain negative counting stats.")
    if (projections[["PROJECTED_GP", "PROJECTED_SEASON_MIN"]] <= 0).any().any():
        raise ValueError("Projected games and season minutes must be positive.")

    # Check each team's full-season minutes total (not per-game minutes).
    team_season_minutes = projections.groupby("TEAM_ABBREVIATION")["PROJECTED_SEASON_MIN"].sum()
    off = team_season_minutes[(team_season_minutes - TEAM_SEASON_MINUTES).abs() > 1.0]
    if len(off):
        raise ValueError(
            f"Teams not summing to {TEAM_SEASON_MINUTES} total season minutes "
            f"within 1 minute:\n{off}"
        )

    print(
        "Hard checks passed: finite, non-negative, teams sum to "
        "19,680 season minutes."
    )

def _print_console_table(table: pd.DataFrame) -> None:
    text = table.to_string(index=False)
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def print_review(projections: pd.DataFrame) -> None:
    team_points = projections.groupby("TEAM_ABBREVIATION")["PROJ_PTS"].sum()
    print("\nTeam projected PTS per game (sum over roster):")
    print(team_points.describe().round(1).to_string())

    columns = ["PLAYER_NAME", "TEAM_ABBREVIATION", "PROJECTED_MPG", "PROJ_PTS"]
    print("\nTop 10 by projected PTS:")
    _print_console_table(
        projections.nlargest(10, "PROJ_PTS")[columns].round(2)
    )

    movers = projections.assign(
        PTS_CHANGE=projections["PROJ_PTS"] - projections["LAST_PTS_PG"]
    )
    movers = movers.loc[~projections["NO_PRIOR_DATA"]]
    movers = movers.reindex(movers["PTS_CHANGE"].abs().sort_values(ascending=False).index)
    print("\nBiggest PTS movers vs last season's per-game line:")
    _print_console_table(
        movers.head(10)[columns + ["LAST_PTS_PG", "PTS_CHANGE"]].round(2)
    )
    print(
        f"\nFlags: no prior data={int(projections['NO_PRIOR_DATA'].sum())}, "
        f"stale={int(projections['STALE_DATA'].sum())}, "
        f"two-season MPG={int(projections['MPG_BLENDED'].sum())}"
    )


def main() -> None:
    bundle = load_model_bundle()
    roster = load_roster()
    history = load_history()
    latest, latest_year = latest_season_rows(history)
    replacement = replacement_rates(latest, latest_year, bundle)
    table = build_inference_table(roster, latest, latest_year, replacement)

    rates, weight = predict_with_credibility(bundle, table, replacement)
    active_mpg, proj_gp, mpg_blended = raw_minutes_and_gp(table)
    mpg, season_min, proj_gp = allocate_all_teams(table, active_mpg, proj_gp)
    projections = synthesize_stats(table, rates, active_mpg, proj_gp, mpg, season_min, weight, mpg_blended)

    validate_projections(projections)
    projections = projections.sort_values(
        ["TEAM_ABBREVIATION", "PROJECTED_MPG"], ascending=[True, False]
    )
    csv_projections = projections.copy()
    csv_projections["PROJECTED_GP"] = csv_projections["PROJECTED_GP"].round().astype(
        "int64"
    )
    csv_projections[[f"PROJ_{stat}" for stat in OUTPUT_STATS]] = csv_projections[
        [f"PROJ_{stat}" for stat in OUTPUT_STATS]
    ].round(1)
    csv_projections[[f"TOTAL_{stat}" for stat in OUTPUT_STATS]] = csv_projections[
        [f"TOTAL_{stat}" for stat in OUTPUT_STATS]
    ].round().astype("int64")
    percentage_columns = ["PROJ_FG_PCT", "PROJ_FT_PCT"]
    csv_projections[percentage_columns] = (
        csv_projections[percentage_columns].mul(100).round(1)
    )
    metadata_columns = [
        "ROLE_TIER",
        "WEIGHTED_SEASON_MINUTES",
        "MINUTES_BASELINE",
        "MINUTES_CREDIBILITY",
        "COHORT_PRIOR_MINUTES",
        "RAW_MPG",
        "CREDIBILITY_WEIGHT",
        "DATA_SEASON",
        "NO_PRIOR_DATA",
        "STALE_DATA",
        "MPG_BLENDED",
        "LAST_PTS_PG",
    ]
    metadata_columns = [
        column for column in metadata_columns if column in csv_projections.columns
    ]
    csv_projections = csv_projections[
        [column for column in csv_projections.columns if column not in metadata_columns]
        + metadata_columns
    ]
    csv_projections.to_csv(PROJECTIONS_PATH, index=False)
    print(f"\nWrote {len(projections)} rows to {PROJECTIONS_PATH}")
    print_review(projections)

if __name__ == "__main__":
    main()