"""Teammate opportunity when a player misses games, fitted from game logs.

Fit (`fit_opportunity_model`, writes data/processed/opportunity_model.json):
  Per team-season (2014 on), a regular averaged 20+ MPG over 20+ games for the team. A
  regular is absent from a team game between his first and last game for the team when he
  doesn't appear in it. For each teammate who played, gain = minutes - his average minutes
  in the team's games with no regular absent (5+ such games needed).
  - Minutes: gain = F * a(band) * (1 + beta * same_slot), F = absent regulars' average MPG,
    band = teammate's usual MPG (<10, 10-20, 20-28, 28-34, 34+), same_slot = overlap of
    guard/forward/center slots with the absent players. beta = 2. Held-out (each season
    2018-26 fit on earlier ones): per-game minutes MAE 5.88 with no boost, 5.54 for an even
    split, 5.42 for this (better than the even split in 9 of 9 seasons).
  - Rates: teammates' per-minute PTS/FGA/FTA/AST in absence games vs clean games, per 10
    units of freed shot volume (absent players' FGA + 0.44 FTA + TOV per game). Positive in
    13/13 seasons (PTS, FGA), 12/13 (FTA, AST). 3PA (9/13), TOV (8/13), REB, STL and BLK
    showed no reliable change and are left alone. FGM and FTM follow FGA and FTA.
  The analysis is in analysis/injury_opportunity.py (results in analysis/results/).

Apply (`apply_opportunity`): each player j's expected extra absences dGP_j = old projected
GP - injury-adjusted GP (regulars only: projected 20+ MPG). Teammate i gains
  dMPG_i = sum_j a(band_i) * (1 + beta * same_slot_ij) * MPG_j * dGP_j / 82
per game he plays (j is out in dGP_j / 82 of i's games, on average), and his rates move by
  c_stat * sum_j VOL_j * dGP_j / 82 / 10.
dGP can be negative (a regular who has played every game so far), which takes a little away.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from gamelogs import FIRST_GAMELOG_END_YEAR, load_gamelogs

PROCESSED_DIR = os.path.join("data", "processed")
OPPORTUNITY_MODEL_PATH = os.path.join(PROCESSED_DIR, "opportunity_model.json")
POSITIONS_PATH = os.path.join("data", "raw", "player_positions.csv")

REGULAR_MPG = 20.0
REGULAR_GP = 20
# Absences of 10-20 MPG players pass a fitted share of a regular's effect (see
# fit_small_absence_share). Held-out 2018-26: share ~0.1 beat ignoring them in 7 of 9
# seasons (MAE 5.0569 -> 5.0562, tiny); the full share (1.0) was worse in 9 of 9.
SMALL_ABSENCE_MPG = 10.0
MIN_CLEAN_GAMES = 5
BAND_EDGES = [0, 10, 20, 28, 34, 60]
SAME_SLOT_BETA = 2.0
RATE_STATS = ["PTS", "FGA", "FTA", "AST"]
RATE_FOLLOW = {"FGM": "FGA", "FTM": "FTA"}
RATE_CHECK_STATS = ["PTS", "FGA", "FG3A", "FTA", "AST", "TOV", "REB", "STL", "BLK"]
SEASON_GAMES = 82.0
SLOTS = {
    "G": (1, 0, 0), "F": (0, 1, 0), "C": (0, 0, 1),
    "G-F": (2 / 3, 1 / 3, 0), "F-G": (1 / 3, 2 / 3, 0),
    "F-C": (0, 2 / 3, 1 / 3), "C-F": (0, 1 / 3, 2 / 3),
}


def slot_vector(position) -> np.ndarray:
    return np.array(SLOTS.get(str(position), (1 / 3, 1 / 3, 1 / 3)), dtype=float)


def load_positions() -> pd.Series:
    if not os.path.exists(POSITIONS_PATH):
        return pd.Series(dtype=object)
    return pd.read_csv(POSITIONS_PATH).set_index("PLAYER_ID")["POSITION"]


def build_games(logs: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """One row per (team game, player who played) with freed minutes, freed shot volume,
    same-slot overlap and the player's clean-game baseline."""
    frames = []
    for (season, team), tg in logs.groupby(["SEASON_END", "TEAM_ABBREVIATION"]):
        games = tg[["GAME_ID", "GAME_DATE"]].drop_duplicates().sort_values("GAME_DATE")["GAME_ID"].tolist()
        order = {gid: n for n, gid in enumerate(games)}
        tg = tg.assign(ORDER=tg["GAME_ID"].map(order))
        per = tg.groupby("PLAYER_ID").agg(
            GP=("MIN", "size"), MPG=("MIN", "mean"), FIRST=("ORDER", "min"), LAST=("ORDER", "max"),
            FGA_PG=("FGA", "mean"), FTA_PG=("FTA", "mean"), TOV_PG=("TOV", "mean"),
        )
        per["VOL"] = per["FGA_PG"] + 0.44 * per["FTA_PG"] + per["TOV_PG"]
        regulars = per[(per["MPG"] >= REGULAR_MPG) & (per["GP"] >= REGULAR_GP)]
        played = set(zip(tg["GAME_ID"], tg["PLAYER_ID"]))
        absent = [
            (gid, pid, r.MPG, r.VOL)
            for pid, r in regulars.iterrows()
            for gid in games[int(r.FIRST): int(r.LAST) + 1]
            if (gid, pid) not in played
        ]
        absent = pd.DataFrame(absent, columns=["GAME_ID", "ABSENT_ID", "ABSENT_MPG", "ABSENT_VOL"])
        if len(absent):
            freed = absent.groupby("GAME_ID")[["ABSENT_MPG", "ABSENT_VOL"]].sum()
            slots = np.vstack(absent["ABSENT_ID"].map(lambda p: slot_vector(positions.get(p))))
            weighted = pd.DataFrame(slots * absent[["ABSENT_MPG"]].to_numpy(), columns=["SG", "SF", "SC"])
            weighted["GAME_ID"] = absent["GAME_ID"].to_numpy()
            absent_slot = weighted.groupby("GAME_ID").sum().div(freed["ABSENT_MPG"], axis=0)
        else:
            freed = pd.DataFrame(columns=["ABSENT_MPG", "ABSENT_VOL"])
            absent_slot = pd.DataFrame(columns=["SG", "SF", "SC"])
        tg = tg.join(freed.rename(columns={"ABSENT_MPG": "FREED", "ABSENT_VOL": "FREED_VOL"}), on="GAME_ID")
        tg[["FREED", "FREED_VOL"]] = tg[["FREED", "FREED_VOL"]].fillna(0.0)
        own = np.vstack(tg["PLAYER_ID"].map(lambda p: slot_vector(positions.get(p))))
        other = absent_slot.reindex(tg["GAME_ID"]).fillna(0.0).to_numpy(dtype=float)
        tg["SAME_SLOT"] = (own * other).sum(axis=1)
        clean = tg[tg["FREED"] == 0]
        base = clean.groupby("PLAYER_ID").agg(
            BASE_N=("MIN", "size"), BASE_MPG=("MIN", "mean"), BASE_MIN=("MIN", "sum"),
            **{f"BASE_{s}": (s, "sum") for s in RATE_CHECK_STATS},
        )
        frames.append(tg.join(base, on="PLAYER_ID").assign(SEASON_END=season))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["BASE_N"] >= MIN_CLEAN_GAMES].copy()
    df["GAIN"] = df["MIN"] - df["BASE_MPG"]
    df["BAND"] = pd.cut(df["BASE_MPG"], BAND_EDGES, labels=False, include_lowest=True)
    return df


def _ls(x: np.ndarray, y: np.ndarray) -> float:
    return float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else 0.0


def fit_band_coefficients(out: pd.DataFrame, beta: float = SAME_SLOT_BETA) -> dict[int, float]:
    coef = {}
    for band, part in out.groupby("BAND"):
        x = (part["FREED"] * (1 + beta * part["SAME_SLOT"])).to_numpy()
        coef[int(band)] = _ls(x, part["GAIN"].to_numpy())
    return coef


def fit_rate_coefficients(out: pd.DataFrame, stats=RATE_CHECK_STATS) -> dict[str, float]:
    """Pooled relative rate change per 10 freed shot volume (teammates with 200+ clean minutes)."""
    part = out[out["BASE_MIN"] >= 200]
    coef = {}
    for stat in stats:
        expected = part["BASE_" + stat] / part["BASE_MIN"] * part["MIN"]
        f = np.average(part["FREED_VOL"], weights=expected.clip(lower=1e-6))
        coef[stat] = float((part[stat].sum() / expected.sum() - 1) / f * 10) if f > 0 else 0.0
    return coef


def fit_small_absence_share(
    logs: pd.DataFrame, positions: pd.Series, regular_games: pd.DataFrame, bands: dict[int, float]
) -> float:
    """How much of a 10-20 MPG player's absence reaches teammates, relative to a regular's.

    Rebuilds absences with a 10-MPG regular line, keeps the 20+ baselines and bands, and fits
    gain = a(band) * (X20 + share * X_small) for the share by least squares.
    """
    global REGULAR_MPG
    saved = REGULAR_MPG
    try:
        REGULAR_MPG = SMALL_ABSENCE_MPG
        wide = build_games(logs, positions)
    finally:
        REGULAR_MPG = saved
    keys = ["SEASON_END", "GAME_ID", "PLAYER_ID"]
    m = regular_games.merge(
        wide[keys + ["FREED", "SAME_SLOT"]].rename(columns={"FREED": "F_WIDE", "SAME_SLOT": "S_WIDE"}),
        on=keys, how="left",
    )
    m = m[m["F_WIDE"] > 0]
    x_regular = m["FREED"] * (1 + SAME_SLOT_BETA * m["SAME_SLOT"])
    x_small = (m["F_WIDE"] * (1 + SAME_SLOT_BETA * m["S_WIDE"]) - x_regular).clip(lower=0)
    a = m["BAND"].map(bands)
    residual = m["GAIN"] - a * x_regular
    return max(0.0, _ls((a * x_small).to_numpy(), residual.to_numpy()))


def fit_opportunity_model(last_end_year: int, first_end_year: int = FIRST_GAMELOG_END_YEAR, write: bool = True) -> dict:
    """Fit on completed seasons first_end_year..last_end_year and save the coefficients."""
    logs = load_gamelogs(range(first_end_year, last_end_year + 1))
    if logs.empty:
        return {"active": False, "reason": "no game logs"}
    positions = load_positions()
    df = build_games(logs, positions)
    out = df[df["FREED"] > 0]
    bands = fit_band_coefficients(out)
    small_share = fit_small_absence_share(logs, positions, df, bands)
    # A negative band coefficient (34+ MPG starters gain nothing) is set to 0.
    bands = {b: max(0.0, c) for b, c in bands.items()}
    rates = fit_rate_coefficients(out)
    model = {
        "active": True,
        "seasons": [first_end_year, last_end_year],
        "band_edges": BAND_EDGES,
        "band_coef": bands,
        "same_slot_beta": SAME_SLOT_BETA,
        "rate_coef_per_10_volume": {s: rates[s] for s in RATE_STATS},
        "rate_coef_checked": rates,
        "regular_mpg": REGULAR_MPG,
        "small_absence_mpg": SMALL_ABSENCE_MPG,
        "small_absence_share": small_share,
        "team_games": int(out.groupby(["SEASON_END", "TEAM_ABBREVIATION", "GAME_ID"]).ngroups),
    }
    if write:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with open(OPPORTUNITY_MODEL_PATH, "w") as f:
            json.dump(model, f, indent=2)
        print(f"Saved {OPPORTUNITY_MODEL_PATH}: bands {({b: round(c, 4) for b, c in bands.items()})}, "
              f"10-20 MPG absence share {small_share:.3f}, "
              f"rates {({s: round(rates[s], 4) for s in RATE_STATS})}")
    return model


def load_opportunity_model(path: str = OPPORTUNITY_MODEL_PATH) -> dict:
    if not os.path.exists(path):
        return {"active": False, "reason": f"{path} missing (run python src/opportunity.py)"}
    with open(path) as f:
        model = json.load(f)
    model["band_coef"] = {int(k): v for k, v in model["band_coef"].items()}
    return model


def apply_opportunity(
    projections: pd.DataFrame,
    rates: pd.DataFrame,
    old_gp: pd.Series,
    new_gp: pd.Series,
    model: dict,
    positions: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """MPG gain per player and rates after teammates' expected absences change.

    `projections` needs PLAYER_ID, TEAM_ABBREVIATION, PROJECTED_MPG (before this change);
    rates are per minute, same index. Returns (MPG change, new rates).
    """
    mpg_gain = pd.Series(0.0, index=projections.index)
    rates = rates.copy()
    if not model.get("active", False):
        return mpg_gain, rates
    positions = positions if positions is not None else load_positions()
    mpg = projections["PROJECTED_MPG"].astype(float)
    # Regulars' absences count in full; 10-20 MPG players' at the fitted share (~0.1: their
    # minutes mostly go to players outside the usual rotation); under 10 MPG not at all.
    weight = pd.Series(
        np.where(
            mpg >= model["regular_mpg"], 1.0,
            np.where(mpg >= model.get("small_absence_mpg", np.inf), model.get("small_absence_share", 0.0), 0.0),
        ),
        index=mpg.index,
    )
    delta = ((old_gp - new_gp) * weight).fillna(0.0)
    volume = (rates["FGA"] + 0.44 * rates["FTA"] + rates["TOV"]) * mpg
    bands = pd.cut(mpg, model["band_edges"], labels=False, include_lowest=True)
    coef = bands.map(model["band_coef"]).fillna(0.0)
    beta = model["same_slot_beta"]
    slots = np.vstack(projections["PLAYER_ID"].map(lambda p: slot_vector(positions.get(p))))
    rate_coef = model["rate_coef_per_10_volume"]
    freed_volume = pd.Series(0.0, index=projections.index)
    for _, index in projections.groupby("TEAM_ABBREVIATION").groups.items():
        idx = projections.index.get_indexer(index)
        d = delta.iloc[idx].to_numpy()
        if not np.any(d):
            continue
        absent_share = d / SEASON_GAMES  # share of the season each teammate is missing
        overlap = slots[idx] @ slots[idx].T  # same_slot between every pair
        freed_min = mpg.iloc[idx].to_numpy() * absent_share
        gain = (1 + beta * overlap) @ freed_min - (1 + beta * np.diag(overlap)) * freed_min
        mpg_gain.iloc[idx] = coef.iloc[idx].to_numpy() * gain
        vol = volume.iloc[idx].to_numpy() * absent_share
        freed_volume.iloc[idx] = vol.sum() - vol
    for stat, c in rate_coef.items():
        rates[stat] = rates[stat] * (1 + c * freed_volume / 10)
    for stat, base in RATE_FOLLOW.items():
        if base in rate_coef:
            rates[stat] = rates[stat] * (1 + rate_coef[base] * freed_volume / 10)
    return mpg_gain, rates


if __name__ == "__main__":
    from gamelogs import CURRENT_SEASON, season_end_year

    fit_opportunity_model(season_end_year(CURRENT_SEASON) - 1)
