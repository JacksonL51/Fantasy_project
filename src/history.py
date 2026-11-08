"""Shared multi-season player history utilities.

Used by model.py (training features) and project.py (inference features and
the minutes / games baselines) so identical numbers are computed in both
places. This module has no dependency on model.py.

Two kinds of history are built here:

1. Rate history (`compute_history_features`): for every player-season t, a
   recency- and minutes-weighted average of that player's per-minute rates
   over the seasons ending at t, plus how many minutes / seasons back it.
   It only looks at seasons <= t, so it is safe to use as a training feature
   for the t -> t+1 transition.

2. Playing-time baselines (`player_baselines`): a recency-weighted active MPG
   and the raw games-played history for the last few completed seasons.
"""

import numpy as np
import pandas as pd

# Rate-history features
HISTORY_WINDOW_YEARS = 3  # seasons ending at t that feed the history features
HISTORY_DECAY = 0.7  # weight multiplier per year back (1.0, 0.7, 0.49)

# Playing-time baselines
BASELINE_WEIGHTS = (0.60, 0.27, 0.13)  # most recent completed season first
MIN_GP_FOR_STABLE_MPG = 20  # seasons with fewer games are ignored for MPG


def season_year(series: pd.Series) -> pd.Series:
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


def history_feature_columns(stats: list[str]) -> list[str]:
    return [f"{stat}_RATE_hist" for stat in stats] + ["HIST_MIN", "HIST_SEASONS"]


def load_history(path, stats: list[str]) -> pd.DataFrame:
    """One row per player per season with valid minutes, games and rates."""
    history = pd.read_csv(path)
    rate_columns = [f"{stat}_RATE" for stat in stats]
    required = ["PLAYER_ID", "SEASON", "AGE", "GP", "MIN", *rate_columns]
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise ValueError("training_data.csv is missing columns: " + ", ".join(missing))

    history["SEASON_YEAR"] = season_year(history["SEASON"])
    for column in ["PLAYER_ID", "AGE", "GP", "MIN", *rate_columns]:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    valid = (
        history["PLAYER_ID"].notna()
        & (history["MIN"] > 0)
        & (history["GP"] > 0)
        & np.isfinite(history[rate_columns].to_numpy(dtype=float)).all(axis=1)
    )
    history = history.loc[valid].copy()
    history["PLAYER_ID"] = history["PLAYER_ID"].astype("int64")
    if "PLAYER_NAME" not in history.columns:
        history["PLAYER_NAME"] = pd.NA
    return (
        history.sort_values(["PLAYER_ID", "SEASON_YEAR"])
        .drop_duplicates(["PLAYER_ID", "SEASON_YEAR"], keep="last")
        .reset_index(drop=True)
    )


def compute_history_features(
    history: pd.DataFrame,
    stats: list[str],
    window_years: int = HISTORY_WINDOW_YEARS,
    decay: float = HISTORY_DECAY,
) -> pd.DataFrame:
    """Rolling multi-season rate features, one row per (PLAYER_ID, SEASON_YEAR).

    For season t the feature is the average of each rate over seasons
    t, t-1, ..., t-(window_years-1), weighted by decay**years_back * MINUTES,
    so recent seasons and bigger samples count more. Later seasons are never
    used. HIST_MIN is the unweighted minutes in the window and HIST_SEASONS the
    number of seasons found in it (calendar gaps simply shrink the window).
    """
    rate_columns = [f"{stat}_RATE" for stat in stats]
    ordered = history.sort_values(["PLAYER_ID", "SEASON_YEAR"])
    ids, season_years, rate_rows, minute_totals, season_counts = [], [], [], [], []
    for player_id, group in ordered.groupby("PLAYER_ID", sort=False):
        years = group["SEASON_YEAR"].to_numpy()
        minutes = group["MIN"].to_numpy(dtype=float)
        rates = group[rate_columns].to_numpy(dtype=float)
        for i in range(len(group)):
            years_back = years[i] - years[: i + 1]
            keep = years_back < window_years
            weight = (decay ** years_back[keep]) * minutes[: i + 1][keep]
            rate_rows.append(
                (rates[: i + 1][keep] * weight[:, None]).sum(axis=0) / weight.sum()
            )
            ids.append(player_id)
            season_years.append(years[i])
            minute_totals.append(minutes[: i + 1][keep].sum())
            season_counts.append(int(keep.sum()))

    features = pd.DataFrame(
        np.vstack(rate_rows), columns=[f"{stat}_RATE_hist" for stat in stats]
    )
    features.insert(0, "SEASON_YEAR", np.asarray(season_years, dtype="int64"))
    features.insert(0, "PLAYER_ID", np.asarray(ids, dtype="int64"))
    features["HIST_MIN"] = minute_totals
    features["HIST_SEASONS"] = season_counts
    return features


def _weighted_nan_mean(frame: pd.DataFrame, weights) -> pd.Series:
    """Row-wise weighted mean that skips NaN cells (weights renormalized)."""
    values = frame.to_numpy(dtype=float)
    valid = ~np.isnan(values)
    cell_weights = np.where(valid, np.asarray(weights, dtype=float)[None, :], 0.0)
    denominator = cell_weights.sum(axis=1)
    numerator = (np.where(valid, values, 0.0) * cell_weights).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = numerator / denominator
    return pd.Series(np.where(denominator > 0, result, np.nan), index=frame.index)


def player_baselines(
    history: pd.DataFrame,
    latest_year: int,
    weights=BASELINE_WEIGHTS,
    min_gp: int = MIN_GP_FOR_STABLE_MPG,
) -> pd.DataFrame:
    """Playing-time baselines over the last completed seasons, one row per player.

    Active MPG (MIN / GP) is averaged with recency weights over seasons with
    at least `min_gp` games; if a player has none in the window, the most
    recent season is used and MPG_FALLBACK is set.

    Games played are tracked per calendar season of the window. A season the
    player missed entirely counts as 0 games if he was already in the league
    (first season <= that year); seasons before his debut are not counted.
    """
    window = len(weights)
    first_year = history.groupby("PLAYER_ID")["SEASON_YEAR"].min()
    players = first_year.index
    gp_wide = history.pivot(index="PLAYER_ID", columns="SEASON_YEAR", values="GP")
    min_wide = history.pivot(index="PLAYER_ID", columns="SEASON_YEAR", values="MIN")

    gp_columns, mpg_columns = {}, {}
    empty = pd.Series(np.nan, index=players)
    for k in range(window):
        year = latest_year - k
        gp_k = gp_wide[year].reindex(players) if year in gp_wide.columns else empty
        min_k = min_wide[year].reindex(players) if year in min_wide.columns else empty
        in_league = first_year <= year
        gp_columns[k] = gp_k.fillna(0.0).where(in_league)
        mpg_columns[k] = (min_k / gp_k).where(gp_k >= min_gp)
    gp_frame = pd.DataFrame(gp_columns)
    mpg_frame = pd.DataFrame(mpg_columns)

    last_row = history.groupby("PLAYER_ID").tail(1).set_index("PLAYER_ID")
    last_mpg = (last_row["MIN"] / last_row["GP"]).reindex(players)
    weighted_mpg = _weighted_nan_mean(mpg_frame, weights)
    fallback = weighted_mpg.isna()

    return pd.DataFrame(
        {
            "PLAYER_ID": players.to_numpy(),
            "BASE_MPG": weighted_mpg.fillna(last_mpg).to_numpy(),
            "MPG_SEASONS_USED": mpg_frame.notna().sum(axis=1).to_numpy(),
            "MPG_FALLBACK": fallback.to_numpy(),
            "GP_WEIGHTED": _weighted_nan_mean(gp_frame, weights).to_numpy(),
            "GP_MEDIAN": gp_frame.median(axis=1).to_numpy(),
            "GP_MAX": gp_frame.max(axis=1).to_numpy(),
            "GP_SEASONS_COUNTED": gp_frame.notna().sum(axis=1).to_numpy(),
            "FIRST_SEASON": first_year.to_numpy(),
        }
    )


# --------------------------------------------------------------------------
# Age effect on playing time
# --------------------------------------------------------------------------
AGE_CURVE_MIN_GP = 40  # both seasons need this many games to enter the fit
AGE_CURVE_MIN_PAIRS = 300  # fewer transitions than this -> no adjustment
AGE_CURVE_SHRINK = 0.5  # fraction of the fitted change that is applied
AGE_CURVE_CLIP = 2.5  # max +/- MPG adjustment


def fit_age_mpg_curve(history: pd.DataFrame, min_gp: int = AGE_CURVE_MIN_GP) -> dict:
    """Expected next-season change in active MPG, from age and current MPG.

    Least squares on consecutive-season pairs (both seasons with >= min_gp
    games): delta = b0 + b1*a + b2*a^2 + b3*m with a = age - 26 and
    m = MPG - 20. The MPG term captures regression to the mean in minutes, so
    the age terms are not confused with it. Pass a history that already ends at
    the last season you may use (the backtest does this to avoid leakage).
    """
    h = history.loc[history["GP"] >= min_gp, ["PLAYER_ID", "SEASON_YEAR", "AGE", "GP", "MIN"]]
    h = h.assign(MPG=h["MIN"] / h["GP"])
    nxt = h[["PLAYER_ID", "SEASON_YEAR", "MPG"]].copy()
    nxt["SEASON_YEAR"] -= 1
    pairs = h.merge(nxt, on=["PLAYER_ID", "SEASON_YEAR"], suffixes=("", "_NEXT"))
    pairs = pairs.dropna(subset=["AGE", "MPG", "MPG_NEXT"])
    if len(pairs) < AGE_CURVE_MIN_PAIRS:
        return {"kind": "age", "coef": [0.0, 0.0, 0.0, 0.0], "pairs": int(len(pairs)), "active": False}
    a = pairs["AGE"].to_numpy(dtype=float) - 26.0
    m = pairs["MPG"].to_numpy(dtype=float) - 20.0
    design = np.column_stack([np.ones(len(a)), a, a**2, m])
    delta = (pairs["MPG_NEXT"] - pairs["MPG"]).to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(design, delta, rcond=None)
    return {"kind": "age", "coef": [round(float(c), 5) for c in coef], "pairs": int(len(pairs)), "active": True}


def age_mpg_adjustment(curve: dict, age: pd.Series, mpg: pd.Series) -> pd.Series:
    """Shrunk, clipped expected MPG change for each player (0 if curve inactive)."""
    if not curve.get("active", False):
        return pd.Series(0.0, index=age.index)
    b0, b1, b2, b3 = curve["coef"]
    a = pd.to_numeric(age, errors="coerce") - 26.0
    m = mpg - 20.0
    predicted = b0 + b1 * a + b2 * a**2 + b3 * m
    return (AGE_CURVE_SHRINK * predicted).clip(-AGE_CURVE_CLIP, AGE_CURVE_CLIP).fillna(0.0)


# --------------------------------------------------------------------------
# Production score, role (rotation-change) curve, fitted games model
# --------------------------------------------------------------------------
PROD_POOL_MINUTES = 500.0  # players with at least this many minutes set the scale
PROD_SHRINK_MINUTES = 400.0  # score is multiplied by MIN / (MIN + this)
ROLE_WEIGHTS = (0.60, 0.27, 0.13)  # must match project.py's RECENCY_WEIGHTS
ROLE_BASELINE_MIN_GP = 15  # must match project.py's BASELINE_ACTIVE_MPG rule
ROLE_CURVE_MIN_GP = 20  # next season needs this many games to count
ROLE_CURVE_MIN_PAIRS = 300
ROLE_SHRINK = 0.5  # fraction of the fitted MPG change that is applied
ROLE_CLIP = 3.0  # max +/- MPG adjustment
ROLE_LOW_MPG = 22.0  # production matters most below this baseline MPG
GAMES_RIDGE_ALPHA = 10.0
GAMES_MIN_NEXT_MINUTES = 200.0  # next-season minutes needed to enter the fit
GAMES_BOUNDS = (15.0, 82.0)
GAMES_ROTATION_MIN_MPG = 20.0  # at or above: durability model; below: low GP is mostly role
GAMES_MIN_ROTATION_PAIRS = 300
GAMES_FULL_SEASON = 82.0
# Shortened seasons are scaled to 82 games so 72 in 2021 does not read as missed time.
GAMES_SHORTENED_SEASONS = {1999: 50.0, 2012: 66.0, 2020: 72.0, 2021: 72.0}
GAMES_DURABLE_GP = 75.0
GAMES_FEATURES = [
    "GF_LAST", "GF_W", "GF_MIN", "GF_MAX", "GF_N", "BASE_MPG", "AGE", "AGE_SQ", "GF_W_X_MPG",
]
GAMES_ROTATION_FEATURES = GAMES_FEATURES + ["GF_5YR", "GF_5YR_N", "GF_75_PLUS"]


def season_production_z(history: pd.DataFrame) -> pd.DataFrame:
    """Per-minute 9-category production score for every player-season.

    Within each season, players with >= 500 minutes define the mean and spread
    of seven per-minute counting rates (turnovers negated) plus FG and FT
    efficiency impact per minute (made minus league percentage times attempts).
    The score is the average z-score, shrunk toward 0 by MIN / (MIN + 400) so
    tiny samples cannot look like stars. Only the same season's data is used.
    """
    rows = []
    for year, group in history.groupby("SEASON_YEAR"):
        pool = group.loc[group["MIN"] >= PROD_POOL_MINUTES]
        if len(pool) < 30:
            rows.append(
                pd.DataFrame(
                    {"PLAYER_ID": group["PLAYER_ID"], "SEASON_YEAR": year, "PROD_Z": 0.0}
                )
            )
            continue
        fg = (pool["FGM_RATE"] * pool["MIN"]).sum() / (pool["FGA_RATE"] * pool["MIN"]).sum()
        ft = (pool["FTM_RATE"] * pool["MIN"]).sum() / (pool["FTA_RATE"] * pool["MIN"]).sum()

        def components(frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "PTS": frame["PTS_RATE"], "FG3M": frame["FG3M_RATE"],
                    "REB": frame["REB_RATE"], "AST": frame["AST_RATE"],
                    "STL": frame["STL_RATE"], "BLK": frame["BLK_RATE"],
                    "TOV": -frame["TOV_RATE"],
                    "FG": frame["FGM_RATE"] - fg * frame["FGA_RATE"],
                    "FT": frame["FTM_RATE"] - ft * frame["FTA_RATE"],
                }
            )

        reference = components(pool)
        spread = reference.std(ddof=0).replace(0.0, 1.0)
        z = ((components(group) - reference.mean()) / spread).mean(axis=1)
        shrink = group["MIN"] / (group["MIN"] + PROD_SHRINK_MINUTES)
        rows.append(
            pd.DataFrame(
                {"PLAYER_ID": group["PLAYER_ID"].to_numpy(), "SEASON_YEAR": year,
                 "PROD_Z": (z * shrink).to_numpy()}
            )
        )
    return pd.concat(rows, ignore_index=True)


def _wide(history: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """Players x consecutive calendar years (NaN where a season is missing)."""
    frame = pd.DataFrame(
        {"PLAYER_ID": history["PLAYER_ID"].to_numpy(),
         "SEASON_YEAR": history["SEASON_YEAR"].to_numpy(),
         "V": values.to_numpy(dtype=float)}
    ).pivot(index="PLAYER_ID", columns="SEASON_YEAR", values="V")
    years = range(int(frame.columns.min()), int(frame.columns.max()) + 1)
    return frame.reindex(columns=list(years))


def role_feature_table(history: pd.DataFrame) -> pd.DataFrame:
    """Per (PLAYER_ID, SEASON_YEAR t): BASE_MPG as of t, AGE and PROD_Z.

    BASE_MPG is the 60/27/13 recency-weighted active MPG over seasons t, t-1,
    t-2 that have >= 15 games, using only seasons <= t (same rule as
    project.py's BASELINE_ACTIVE_MPG for a player whose latest season is t).
    """
    mpg = (history["MIN"] / history["GP"]).where(history["GP"] >= ROLE_BASELINE_MIN_GP)
    wide = _wide(history, mpg)
    numerator = pd.DataFrame(0.0, index=wide.index, columns=wide.columns)
    denominator = numerator.copy()
    for lag, weight in enumerate(ROLE_WEIGHTS):
        shifted = wide.shift(lag, axis=1)
        numerator += shifted.fillna(0.0) * weight
        denominator += shifted.notna() * weight
    base = (numerator / denominator.where(denominator > 0)).stack().rename("BASE_MPG")
    base = base.reset_index()
    out = history[["PLAYER_ID", "SEASON_YEAR", "AGE"]].merge(
        base, on=["PLAYER_ID", "SEASON_YEAR"], how="left", validate="one_to_one"
    )
    return out.merge(
        season_production_z(history), on=["PLAYER_ID", "SEASON_YEAR"],
        how="left", validate="one_to_one",
    )


def _role_design(age, base_mpg, prod) -> np.ndarray:
    a = np.asarray(age, dtype=float) - 26.0
    m = np.asarray(base_mpg, dtype=float) - 20.0
    p = np.asarray(prod, dtype=float)
    low = np.maximum(0.0, ROLE_LOW_MPG - np.asarray(base_mpg, dtype=float)) / 10.0
    return np.column_stack([np.ones(len(a)), a, a**2, m, m**2, p, p * low])


def fit_role_curve(history: pd.DataFrame) -> dict:
    """Expected next-season MPG change from age, baseline MPG and production.

    Least squares over consecutive-season pairs where next season has >= 20
    games: next MPG minus baseline MPG on age, age^2, MPG, MPG^2, production
    score, and production x how far the baseline is below 22 MPG. Pass a
    history that already ends at the last season you may use.
    """
    features = role_feature_table(history)
    following = history.loc[
        history["GP"] >= ROLE_CURVE_MIN_GP, ["PLAYER_ID", "SEASON_YEAR", "GP", "MIN"]
    ].copy()
    following["MPG_NEXT"] = following["MIN"] / following["GP"]
    following["SEASON_YEAR"] -= 1
    pairs = features.merge(
        following[["PLAYER_ID", "SEASON_YEAR", "MPG_NEXT"]],
        on=["PLAYER_ID", "SEASON_YEAR"],
    ).dropna(subset=["AGE", "BASE_MPG", "PROD_Z", "MPG_NEXT"])
    if len(pairs) < ROLE_CURVE_MIN_PAIRS:
        return {"kind": "role", "coef": [0.0] * 7, "pairs": int(len(pairs)), "active": False}
    design = _role_design(pairs["AGE"], pairs["BASE_MPG"], pairs["PROD_Z"])
    target = (pairs["MPG_NEXT"] - pairs["BASE_MPG"]).to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return {"kind": "role", "coef": [round(float(c), 5) for c in coef],
            "pairs": int(len(pairs)), "active": True}


def role_mpg_adjustment(curve: dict, age: pd.Series, mpg: pd.Series, prod: pd.Series) -> pd.Series:
    """Shrunk, clipped expected MPG change; 0 if the curve is inactive."""
    if not curve.get("active", False):
        return pd.Series(0.0, index=age.index)
    design = _role_design(
        pd.to_numeric(age, errors="coerce").fillna(26.0), mpg, prod.fillna(0.0)
    )
    predicted = pd.Series(design @ np.asarray(curve["coef"], dtype=float), index=age.index)
    return (ROLE_SHRINK * predicted).clip(-ROLE_CLIP, ROLE_CLIP).fillna(0.0)


def _games_per_82(history: pd.DataFrame) -> pd.Series:
    """Games played, scaled to 82 for shortened seasons and capped at 82."""
    full_season = history["SEASON_YEAR"].map(GAMES_SHORTENED_SEASONS).fillna(GAMES_FULL_SEASON)
    return (history["GP"] * GAMES_FULL_SEASON / full_season).clip(upper=GAMES_FULL_SEASON)


def games_feature_table(history: pd.DataFrame) -> pd.DataFrame:
    """Games-played features per (PLAYER_ID, SEASON_YEAR t), seasons <= t only.

    All use games per 82. GF_LAST = games in t; GF_W = recency-weighted mean of
    games over seasons t, t-1, t-2 that exist; GF_MIN / GF_MAX = lowest / highest
    of those; GF_N = how many of the three exist. GF_5YR = mean over up to five
    calendar seasons since debut, with a season missed entirely counted as 0;
    GF_5YR_N = how many seasons that covers; GF_75_PLUS = how many of the latest
    three of them reached 75 games.
    """
    wide = _wide(history, _games_per_82(history))
    stack = np.stack([wide.shift(lag, axis=1).to_numpy() for lag in range(3)])
    weights = np.asarray(ROLE_WEIGHTS)[:, None, None]
    valid = ~np.isnan(stack)
    weight_sum = (weights * valid).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted = (np.where(valid, stack, 0.0) * weights).sum(axis=0) / weight_sum
        lowest = np.where(valid, stack, np.inf).min(axis=0)
        highest = np.where(valid, stack, -np.inf).max(axis=0)
    debut = history.groupby("PLAYER_ID")["SEASON_YEAR"].min().reindex(wide.index).to_numpy()
    years = wide.columns.to_numpy()
    five = np.stack(
        [
            np.where(
                (years - lag)[None, :] >= debut[:, None],
                np.nan_to_num(wide.shift(lag, axis=1).to_numpy()),
                np.nan,
            )
            for lag in range(5)
        ]
    )
    five_count = (~np.isnan(five)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        five_mean = np.nansum(five, axis=0) / five_count
    durable_count = (np.nan_to_num(five[:3]) >= GAMES_DURABLE_GP).sum(axis=0)
    table = pd.DataFrame(
        {
            "GF_LAST": wide.to_numpy().ravel(),
            "GF_W": weighted.ravel(),
            "GF_MIN": lowest.ravel(),
            "GF_MAX": highest.ravel(),
            "GF_N": valid.sum(axis=0).ravel(),
            "GF_5YR": five_mean.ravel(),
            "GF_5YR_N": five_count.ravel(),
            "GF_75_PLUS": durable_count.ravel(),
        },
        index=pd.MultiIndex.from_product(
            [wide.index, wide.columns], names=["PLAYER_ID", "SEASON_YEAR"]
        ),
    )
    return table.dropna(subset=["GF_LAST"]).reset_index()


def _games_matrix(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    age = pd.to_numeric(frame["AGE"], errors="coerce").fillna(26.0).to_numpy(dtype=float)
    columns = []
    for name in names:
        if name == "AGE":
            columns.append(age)
        elif name == "AGE_SQ":
            columns.append(age**2)
        elif name == "GF_W_X_MPG":
            columns.append(
                frame["GF_W"].to_numpy(dtype=float) * frame["BASE_MPG"].to_numpy(dtype=float)
            )
        else:
            columns.append(frame[name].to_numpy(dtype=float))
    return np.column_stack(columns)


def _fit_games_ridge(pairs: pd.DataFrame, names: list[str], alpha: float) -> dict:
    design = _games_matrix(pairs, names)
    mean, scale = design.mean(axis=0), design.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (design - mean) / scale
    target = pairs["GP_NEXT"].to_numpy(dtype=float)
    intercept = float(target.mean())
    gram = standardized.T @ standardized + alpha * np.eye(standardized.shape[1])
    coef = np.linalg.solve(gram, standardized.T @ (target - intercept))
    return {
        "intercept": intercept, "mean": mean.tolist(), "scale": scale.tolist(),
        "coef": coef.tolist(), "features": names,
    }


def _predict_games_ridge(model: dict, frame: pd.DataFrame) -> np.ndarray:
    design = _games_matrix(frame, model["features"])
    standardized = (design - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return model["intercept"] + standardized @ np.asarray(model["coef"])


def fit_games_model(history: pd.DataFrame, alpha: float = GAMES_RIDGE_ALPHA) -> dict:
    """Ridge regressions of next-season games per 82.

    Pairs are (season t features) -> games in t+1, where the player had >= 200
    minutes in t+1 (the same population the projections are scored on). Players
    with GAMES_ROTATION_MIN_MPG or more get a model trained only on such players,
    with longer-term durability features; everyone else uses a model of all pairs.
    """
    features = games_feature_table(history).merge(
        role_feature_table(history)[["PLAYER_ID", "SEASON_YEAR", "AGE", "BASE_MPG"]],
        on=["PLAYER_ID", "SEASON_YEAR"], how="left", validate="one_to_one",
    )
    last = history[["PLAYER_ID", "SEASON_YEAR", "GP", "MIN"]]
    features = features.merge(last, on=["PLAYER_ID", "SEASON_YEAR"], how="left")
    features["BASE_MPG"] = features["BASE_MPG"].fillna(features["MIN"] / features["GP"])
    following = history.loc[
        history["MIN"] >= GAMES_MIN_NEXT_MINUTES, ["PLAYER_ID", "SEASON_YEAR"]
    ].copy()
    following["GP_NEXT"] = _games_per_82(history).loc[following.index]
    following["SEASON_YEAR"] -= 1
    required = [
        name for name in GAMES_ROTATION_FEATURES if name not in ("AGE_SQ", "GF_W_X_MPG")
    ]
    pairs = features.merge(following, on=["PLAYER_ID", "SEASON_YEAR"]).dropna(
        subset=required + ["GP_NEXT"]
    )
    rotation = pairs.loc[pairs["BASE_MPG"] >= GAMES_ROTATION_MIN_MPG]
    if len(pairs) < 300 or len(rotation) < GAMES_MIN_ROTATION_PAIRS:
        return {"active": False, "pairs": int(len(pairs))}
    return {
        "active": True, "pairs": int(len(pairs)), "rotation_pairs": int(len(rotation)),
        "rotation_min_mpg": GAMES_ROTATION_MIN_MPG,
        "rotation": _fit_games_ridge(rotation, GAMES_ROTATION_FEATURES, alpha),
        "general": _fit_games_ridge(pairs, GAMES_FEATURES, alpha),
    }


def predict_games(model: dict, frame: pd.DataFrame) -> pd.Series:
    """Fitted games, clipped to 15-82. `frame` needs the games features, BASE_MPG and AGE."""
    predicted = np.where(
        frame["BASE_MPG"].to_numpy(dtype=float) >= model["rotation_min_mpg"],
        _predict_games_ridge(model["rotation"], frame),
        _predict_games_ridge(model["general"], frame),
    )
    return pd.Series(np.clip(predicted, *GAMES_BOUNDS), index=frame.index)


# --------------------------------------------------------------------------
# Positional crowding (soft MPG adjustment)
# --------------------------------------------------------------------------
# Positions are the NBA's coarse career labels from data/raw/player_positions.csv.
# Each label is split across guard / forward / center slots; a team's floor time
# per game is 96 guard, 96 forward and 48 center minutes.
POSITIONS_PATH_PARTS = ("data", "raw", "player_positions.csv")
POSITION_MEMBERSHIP = {
    "G": (1.0, 0.0, 0.0),
    "G-F": (2 / 3, 1 / 3, 0.0),
    "F-G": (1 / 3, 2 / 3, 0.0),
    "F": (0.0, 1.0, 0.0),
    "F-C": (0.0, 2 / 3, 1 / 3),
    "C-F": (0.0, 1 / 3, 2 / 3),
    "C": (0.0, 0.0, 1.0),
}
POSITION_CAPACITY = np.array([96.0, 96.0, 48.0])
POSITION_PRIOR_MIN_GP = 15  # a season counts toward teammates' demand with this many games
POSITION_TARGET_MIN_GP = 20  # next season needs this many games to enter the fit
POSITION_FIT_YEARS = 14  # fit on the latest target seasons only (2013-26 today): the
#                          effect is about a third as strong in older seasons
POSITION_MIN_PAIRS = 300
POSITION_SHRINK = 1.0  # fraction of the fitted change that is applied
POSITION_CLIP = 1.5  # max +/- MPG adjustment


def load_positions(path=None) -> pd.Series:
    """POSITION keyed by PLAYER_ID (empty if the file is missing)."""
    from pathlib import Path

    path = Path(path) if path else Path(__file__).resolve().parents[1].joinpath(*POSITIONS_PATH_PARTS)
    if not path.exists():
        return pd.Series(dtype=object)
    frame = pd.read_csv(path, usecols=["PLAYER_ID", "POSITION"])
    return frame.dropna().drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")["POSITION"]


def position_crowding(players: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """Team and positional crowding for each player.

    `players` needs PLAYER_ID, TEAM and PRIOR_MPG (last season's MPG, NaN if the
    player did not have a 15-game season; such players add no demand). Returns
    TOTAL_CROWD (teammates' prior MPG / 240) and POSITION_CROWD (positional crowding
    minus TOTAL_CROWD: how much more crowded the player's own position is than the
    team overall). POSITION_CROWD is NaN without a known position.
    """
    labels = players["PLAYER_ID"].map(positions)
    known = labels.isin(POSITION_MEMBERSHIP)
    membership = np.array(
        [POSITION_MEMBERSHIP.get(label, (np.nan, np.nan, np.nan)) for label in labels]
    )
    prior = players["PRIOR_MPG"].to_numpy(dtype=float)
    demand_weight = np.where(np.isfinite(prior), prior, 0.0)
    # Players without a position still use team minutes; count them only in the total.
    own = np.nan_to_num(membership) * demand_weight[:, None]
    frame = pd.DataFrame(own, columns=["G", "F", "C"], index=players.index)
    frame["ALL"] = demand_weight
    team_sums = frame.groupby(players["TEAM"].to_numpy()).transform("sum")
    others = team_sums - frame
    positional = (membership * others[["G", "F", "C"]].to_numpy() / POSITION_CAPACITY).sum(axis=1)
    total = others["ALL"].to_numpy() / 240.0
    return pd.DataFrame(
        {
            "TOTAL_CROWD": total,
            "POSITION_CROWD": np.where(known, positional - total, np.nan),
        },
        index=players.index,
    )


def _crowding_pairs(history: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """t -> t+1 pairs with crowding on the t+1 (end-of-season) team."""
    h = history[["PLAYER_ID", "SEASON_YEAR", "TEAM_ABBREVIATION", "AGE", "GP", "MIN"]].copy()
    h["MPG"] = h["MIN"] / h["GP"]
    prior = h.loc[h["GP"] >= POSITION_PRIOR_MIN_GP, ["PLAYER_ID", "SEASON_YEAR", "MPG"]].copy()
    prior["SEASON_YEAR"] += 1
    season = h.merge(
        prior.rename(columns={"MPG": "PRIOR_MPG"}), on=["PLAYER_ID", "SEASON_YEAR"], how="left"
    )
    season = season.loc[season["SEASON_YEAR"] > season["SEASON_YEAR"].min()]
    season["TEAM"] = season["SEASON_YEAR"].astype(str) + "_" + season["TEAM_ABBREVIATION"].astype(str)
    season = season.join(position_crowding(season, positions))
    return season.loc[
        season["PRIOR_MPG"].notna()
        & (season["GP"] >= POSITION_TARGET_MIN_GP)
        & season["POSITION_CROWD"].notna()
        & season["AGE"].notna()
    ]


def _crowding_design(frame: pd.DataFrame) -> np.ndarray:
    a = frame["AGE"].to_numpy(dtype=float) - 26.0
    return np.column_stack(
        [
            np.ones(len(frame)),
            frame["PRIOR_MPG"].to_numpy(dtype=float) - 20.0,
            a,
            a**2,
            frame["TOTAL_CROWD"].to_numpy(dtype=float),
            frame["POSITION_CROWD"].to_numpy(dtype=float),
        ]
    )


def fit_position_crowding(history: pd.DataFrame, positions: pd.Series | None = None) -> dict:
    """Least squares of next-season MPG change on prior MPG, age, team crowding and
    positional crowding, using end-of-season teams. Only the positional term is
    applied (the team minutes budget already handles overall crowding). Pass a
    history that ends at the last season you may use.
    """
    positions = load_positions() if positions is None else positions
    inactive = {"kind": "position", "coef": 0.0, "pairs": 0, "active": False}
    if positions.empty or "TEAM_ABBREVIATION" not in history.columns:
        return inactive
    pairs = _crowding_pairs(history, positions)
    last_year = int(history["SEASON_YEAR"].max())
    pairs = pairs.loc[pairs["SEASON_YEAR"] > last_year - POSITION_FIT_YEARS]
    if len(pairs) < POSITION_MIN_PAIRS:
        return {**inactive, "pairs": int(len(pairs))}
    delta = (pairs["MPG"] - pairs["PRIOR_MPG"]).to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(_crowding_design(pairs), delta, rcond=None)
    return {
        "kind": "position",
        "coef": round(float(coef[-1]), 5),
        "team_coef": round(float(coef[-2]), 5),
        "team_mean": round(float(pairs["TOTAL_CROWD"].mean()), 5),
        "pairs": int(len(pairs)),
        "active": True,
    }


def position_mpg_adjustment(model: dict, crowd: pd.Series) -> pd.Series:
    """Shrunk, clipped MPG change from positional crowding (0 if inactive or unknown)."""
    if not model.get("active", False):
        return pd.Series(0.0, index=crowd.index)
    change = POSITION_SHRINK * model["coef"] * pd.to_numeric(crowd, errors="coerce")
    return change.clip(-POSITION_CLIP, POSITION_CLIP).fillna(0.0)


TEAM_CROWD_CLIP = 3.0  # max +/- MPG from team crowding


def team_mpg_adjustment(model: dict, total: pd.Series, shrink: float) -> pd.Series:
    """MPG change from overall team crowding, centered on the fit's average crowding.

    The team minutes budget already moves bench minutes with crowding, so this is
    an optional extra (project.py TEAM_CROWD_ADJUSTMENT) and is shrunk.
    """
    if not model.get("active", False) or "team_mean" not in model:
        return pd.Series(0.0, index=total.index)
    centered = pd.to_numeric(total, errors="coerce") - model["team_mean"]
    change = shrink * model["team_coef"] * centered
    return change.clip(-TEAM_CROWD_CLIP, TEAM_CROWD_CLIP).fillna(0.0)


# --------------------------------------------------------------------------
# Small-sample shrink for baseline MPG
# --------------------------------------------------------------------------
# A baseline from a short career (a stint on a tanking team, a partial season) overstates
# next season's MPG. Baseline MPG is pulled toward the typical next-season MPG of players
# whose baseline rests on few games, with weight min(1, career minutes / 1,500). Career
# minutes, not recent games, so stars back from injury are left alone. Held-out seasons
# 2014-26 (returning players' MPG error): none 4.56, this 4.48; 28+ MPG baselines 3.06 ->
# 3.06; 20+ MPG from under 60 games 6.96 -> 6.47. A games-based weight (games / (games +
# 20)) hurt stars (3.06 -> 3.17; starters' season-minute error 457 -> 478 on rebuilt
# rosters). 1,500 matches the old anchored allocation's credibility, which the soft
# budget had dropped.
MPG_SHRINK_CAREER_MINUTES = 1500.0
MPG_SHRINK_FEW_GAMES = 60  # baselines from fewer games set the shrink target


def fit_minutes_shrink(history: pd.DataFrame, career_minutes: float = MPG_SHRINK_CAREER_MINUTES) -> dict:
    """Shrink target: average next-season MPG of players whose baseline rests on few games.

    Uses consecutive seasons in `history` only (pass past-only history in backtests).
    """
    h = history[["PLAYER_ID", "SEASON_YEAR", "GP", "MIN"]].copy()
    h["MPG"] = h["MIN"] / h["GP"]
    years = sorted(h["SEASON_YEAR"].unique())
    targets = []
    for year in years[3:]:
        window = h.loc[(h["SEASON_YEAR"] >= year - 3) & (h["SEASON_YEAR"] < year) & (h["GP"] >= 15)]
        played = window.groupby("PLAYER_ID")["GP"].sum()
        few = played.index[played < MPG_SHRINK_FEW_GAMES]
        nxt = h.loc[(h["SEASON_YEAR"] == year) & h["PLAYER_ID"].isin(few), "MPG"]
        targets.extend(nxt.tolist())
    if career_minutes <= 0 or len(targets) < 100:
        return {"active": False, "career_minutes": career_minutes, "target": float("nan"), "n": len(targets)}
    return {
        "active": True, "career_minutes": float(career_minutes),
        "target": round(float(np.mean(targets)), 3), "n": len(targets),
    }


def shrink_baseline_mpg(model: dict, mpg: pd.Series, career_minutes: pd.Series) -> pd.Series:
    """Baseline MPG pulled toward the shrink target by min(1, career minutes / K)."""
    if not model.get("active", False):
        return mpg
    career = pd.to_numeric(career_minutes, errors="coerce").fillna(0.0)
    weight = (career / model["career_minutes"]).clip(0.0, 1.0)
    return weight * mpg + (1.0 - weight) * model["target"]
