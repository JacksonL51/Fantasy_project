"""Backtest games-played blends and fantasy rank accuracy by target season.

Run from the project root with no year to evaluate the latest three seasons,
or pass one season-ending year (for example 2024). Both forms also run the
rookie backtest; `python src/backtest.py rookies` runs only that (each rookie
class from 2016 on predicted from earlier classes, against replacement-level
rates, a typical-rookie MPG and games line, and pick order).

Rate-policy choices are loaded from the project metadata and may have been
selected using folds that include a backtest target year; treat value-rank
comparisons as slightly optimistic.
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import model as M
import project as P
import rookies as R
from history import (
    fit_age_mpg_curve,
    fit_games_model,
    fit_position_crowding,
    fit_role_curve,
    load_positions,
    position_crowding,
    position_mpg_adjustment,
)

MIN_ACTUAL_MINUTES = 500.0
EXPANDED_MINUTES_THRESHOLD = 200.0
GAMES_MODEL_WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0)
CONSTANT_GP_BASELINE = 66.0
TARGET_GP_MAE = 11.4
TOP_N = (50, 100)
COUNT_CATS = ["PTS", "FG3M", "REB", "AST", "STL", "BLK", "TOV"]
RESULTS_PATH = M.PROCESSED_DIR / "backtest_results.csv"
PLAYERS_PATH = M.PROCESSED_DIR / "backtest_players.csv"
WEIGHT_SELECTION_PATH = M.PROCESSED_DIR / "games_weight_selection.json"
MINUTES_TEST_PATH = M.PROCESSED_DIR / "backtest_minutes_test.csv"
RECOMMENDATIONS_PATH = M.PROCESSED_DIR / "backtest_recommendations.json"
RISER_TOP_N = 20
MIN_BENCH_MAE_GAIN = 0.2  # role must cut bench-player MPG error by this much
MIN_RISER_LIFT = 1.0  # predicted top risers must gain this much more MPG than the rest
VALUE_TOLERANCE = 0.002  # allowed drop in VALUE_Spearman versus the comparison
MPG_GROUPS = {
    "all": (0.0, 99.0),
    "bench_under_20": (0.0, 20.0),
    "rotation_20_28": (20.0, 28.0),
    "stars_28_plus": (28.0, 99.0),
}


def _spearman(a, b) -> float:
    return float(spearmanr(a, b).statistic)


def _nine_cat_value(frame: pd.DataFrame) -> pd.Series:
    """Sum of 9-category z-scores from season totals."""
    parts = []
    for stat in COUNT_CATS:
        z = (frame[f"T_{stat}"] - frame[f"T_{stat}"].mean()) / frame[
            f"T_{stat}"
        ].std()
        parts.append(-z if stat == "TOV" else z)
    for made, att in (("FGM", "FGA"), ("FTM", "FTA")):
        league = frame[f"T_{made}"].sum() / frame[f"T_{att}"].sum()
        impact = frame[f"T_{made}"] - league * frame[f"T_{att}"]
        parts.append((impact - impact.mean()) / impact.std())
    return sum(parts)


def _totals(rates: pd.DataFrame, mpg: np.ndarray, gp: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=rates.index)
    for stat in M.TARGET_STATS:
        out[f"PG_{stat}"] = rates[stat].to_numpy() * mpg
        out[f"T_{stat}"] = rates[stat].to_numpy() * mpg * gp
    out["MPG"] = mpg
    out["GP"] = gp
    return out


def _weight_variant(starter_weight: float, bench_weight: float) -> str:
    return f"starter_{starter_weight:g}_bench_{bench_weight:g}"


def _minutes_test(
    year: int,
    table: pd.DataFrame,
    act_mpg: np.ndarray,
    act_gp: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict]:
    """Minutes error by role group and a test of whether predicted risers rise.

    Uses players with a real baseline and >= 20 actual games. "Change" means
    MPG minus the player's baseline MPG. The riser lift is the mean actual
    change of the 20 players each method predicts to rise most, minus the mean
    actual change of everyone else.
    """
    baseline = table["BASELINE_ACTIVE_MPG"].to_numpy(dtype=float)
    eligible = (
        (act_gp >= 20)
        & np.isfinite(baseline)
        & (table["ELIGIBLE_MPG_SEASONS"].fillna(0).to_numpy() > 0)
    )
    actual_change = act_mpg - baseline
    records = []
    for group, (low, high) in MPG_GROUPS.items():
        mask = eligible & (baseline >= low) & (baseline < high)
        if int(mask.sum()) < 10:
            continue
        for method, predicted in predictions.items():
            predicted_change = predicted - baseline
            order = np.argsort(-predicted_change[mask])
            top = order[: min(RISER_TOP_N, len(order))]
            rest = np.setdiff1d(np.arange(int(mask.sum())), top)
            lift = float(
                actual_change[mask][top].mean() - actual_change[mask][rest].mean()
            ) if len(rest) else np.nan
            records.append(
                {
                    "Target_Year": year,
                    "Group": group,
                    "Method": method,
                    "Players": int(mask.sum()),
                    "MPG_MAE": float(np.mean(np.abs(predicted[mask] - act_mpg[mask]))),
                    "MPG_Bias": float(np.mean(predicted[mask] - act_mpg[mask])),
                    "Change_Spearman": _spearman(
                        predicted_change[mask], actual_change[mask]
                    ),
                    "Top_Risers_Actual_Change": float(actual_change[mask][top].mean()),
                    "Riser_Lift": lift,
                }
            )
    return records


def _position_mpg(
    year: int, history: pd.DataFrame, table: pd.DataFrame, mpg: pd.Series
) -> tuple[pd.Series, dict]:
    """Age-adjusted MPG plus the positional crowding change.

    Teams are the target season's end-of-season teams (no historical opening-day
    rosters exist), with demand from every player on that team who had a 15-game
    season the year before, the same as the fit. The fit itself uses seasons
    before `year` only. Slightly optimistic: those rosters are known in hindsight.
    """
    past = history.loc[history["SEASON_YEAR"] < year]
    model = fit_position_crowding(past)
    season = history.loc[history["SEASON_YEAR"] == year, ["PLAYER_ID", "TEAM_ABBREVIATION"]]
    before = past.loc[(past["SEASON_YEAR"] == year - 1) & (past["GP"] >= 15)]
    prior = (before["MIN"] / before["GP"]).set_axis(before["PLAYER_ID"])
    frame = pd.DataFrame(
        {
            "PLAYER_ID": season["PLAYER_ID"].to_numpy(),
            "TEAM": season["TEAM_ABBREVIATION"].to_numpy(),
            "PRIOR_MPG": season["PLAYER_ID"].map(prior).to_numpy(),
        }
    )
    crowd = position_crowding(frame, load_positions()).set_axis(frame["PLAYER_ID"])
    player_crowd = table["PLAYER_ID"].map(crowd["POSITION_CROWD"])
    has_prior = table["PLAYER_ID"].map(prior).notna()
    shift = position_mpg_adjustment(model, player_crowd).where(has_prior, 0.0)
    return (mpg + shift).clip(P.MIN_HEALTHY_MPG, P.MAX_HEALTHY_MPG), model


def _score_year(
    year: int,
    history: pd.DataFrame,
    pairs: pd.DataFrame,
    decisions: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    print(f"\n=== Backtest: predict {year} from data through {year - 1} ===")
    train = M._train_slice(pairs, year)
    if train.empty:
        raise ValueError(f"No training transitions end before {year}.")
    bundle = M.build_bundle(train, decisions)

    past = history.loc[history["SEASON_YEAR"] < year].copy()
    actual = history.loc[history["SEASON_YEAR"] == year].set_index("PLAYER_ID")
    latest, latest_year = P.latest_season_rows(past)
    replacement = P.replacement_rates(latest, latest_year, bundle)
    typical_starter_gp, typical_bench_gp = P.league_typical_games(past)
    ids = [
        player_id
        for player_id in latest["PLAYER_ID"]
        if player_id in actual.index
        and actual.loc[player_id, "MIN"] >= EXPANDED_MINUTES_THRESHOLD
    ]
    roster = pd.DataFrame(
        {"PLAYER_ID": ids, "TEAM_ABBREVIATION": "BT", "ROSTER_NAME": pd.NA}
    )
    table = P.build_inference_table(roster, latest, latest_year, replacement)
    rates, _ = P.predict_with_credibility(bundle, table, replacement)
    curve = fit_age_mpg_curve(past)
    mpg, model_gp, _ = P.raw_minutes_and_gp(table, curve)
    mpg_flat, model_gp_flat, _ = P.raw_minutes_and_gp(table, None)
    role_curve = fit_role_curve(past)
    mpg_role, _, _ = P.raw_minutes_and_gp(table, role_curve)
    games_model = fit_games_model(past)
    print(f"Role curve: {role_curve}")
    print(f"Games model: {games_model['pairs']} training pairs")

    last_mpg = (table["MIN"] / table["GP"]).to_numpy(dtype=float)
    last_gp = table["GP"].to_numpy(dtype=float)
    naive_rates = pd.DataFrame(
        {stat: table[f"{stat}_RATE_prior"].to_numpy() for stat in M.TARGET_STATS},
        index=table.index,
    )

    variants: dict[str, pd.DataFrame] = {}
    default_gp = P.blend_projected_games(
        table,
        model_gp,
        typical_starter_gp,
        typical_bench_gp,
        P.STARTER_GP_MODEL_WEIGHT,
        P.BENCH_GP_MODEL_WEIGHT,
    )
    variants["default_blend"] = _totals(
        rates, mpg.to_numpy(), default_gp.to_numpy()
    )
    for starter_weight in GAMES_MODEL_WEIGHT_GRID:
        for bench_weight in GAMES_MODEL_WEIGHT_GRID:
            name = _weight_variant(starter_weight, bench_weight)
            games = P.blend_projected_games(
                table,
                model_gp,
                typical_starter_gp,
                typical_bench_gp,
                starter_weight,
                bench_weight,
            )
            variants[name] = _totals(rates, mpg.to_numpy(), games.to_numpy())

    no_age_gp = P.blend_projected_games(
        table, model_gp_flat, typical_starter_gp, typical_bench_gp
    )
    fitted_gp = P.fitted_games(table, games_model, default_gp)
    variants["role_mpg"] = _totals(rates, mpg_role.to_numpy(), default_gp.to_numpy())
    variants["fitted_games"] = _totals(rates, mpg.to_numpy(), fitted_gp.to_numpy())
    variants["fitted_games_role"] = _totals(
        rates, mpg_role.to_numpy(), fitted_gp.to_numpy()
    )
    mpg_position, position_model = _position_mpg(year, history, table, mpg)
    print(f"Positional crowding: {position_model}")
    variants["fitted_games_position"] = _totals(
        rates, mpg_position.to_numpy(), fitted_gp.to_numpy()
    )
    variants["pipeline_no_age"] = _totals(
        rates, mpg_flat.to_numpy(), no_age_gp.to_numpy()
    )
    variants["naive"] = _totals(naive_rates, last_mpg, last_gp)
    variants["naive_rates"] = _totals(
        naive_rates, mpg.to_numpy(), default_gp.to_numpy()
    )

    act = actual.loc[table["PLAYER_ID"]]
    act_mpg = (act["MIN"] / act["GP"]).to_numpy(dtype=float)
    act_gp = act["GP"].to_numpy(dtype=float)
    actual_rates = pd.DataFrame(
        {stat: act[f"{stat}_RATE"].to_numpy() for stat in M.TARGET_STATS},
        index=table.index,
    )
    truth_all = _totals(actual_rates, act_mpg, act_gp)
    actual_minutes = act["MIN"].to_numpy(dtype=float)

    players = pd.DataFrame(
        {
            "Target_Year": year,
            "PLAYER_ID": table["PLAYER_ID"].to_numpy(),
            "PLAYER_NAME": table["PLAYER_NAME"].to_numpy(),
            "AGE": table["AGE_prior"].to_numpy(),
            "ACTUAL_MIN": actual_minutes,
            "ACTUAL_MPG": act_mpg,
            "ACTUAL_GP": act_gp,
            "BASELINE_ACTIVE_MPG": table["BASELINE_ACTIVE_MPG"].to_numpy(),
            "MODEL_GP": model_gp.to_numpy(),
            "TYPICAL_GP": np.where(
                table["BASELINE_ACTIVE_MPG"].ge(P.STARTER_MPG_THRESHOLD),
                typical_starter_gp,
                typical_bench_gp,
            ),
        }
    )
    for name, variant in variants.items():
        players[f"PRED_GP_{name}"] = variant["GP"].to_numpy()

    records = []
    evaluation_sets = (
        ("200+ actual minutes", actual_minutes >= EXPANDED_MINUTES_THRESHOLD),
        ("500+ actual minutes", actual_minutes >= MIN_ACTUAL_MINUTES),
    )
    for evaluation_set, eligible in evaluation_sets:
        if int(eligible.sum()) < 2:
            raise ValueError(
                f"Only {int(eligible.sum())} players in {year} meet "
                f"{evaluation_set}; cannot calculate rank metrics."
            )
        truth = truth_all.loc[eligible].copy()
        truth["VALUE"] = _nine_cat_value(truth)
        actual_gp = act_gp[eligible]
        baseline_66_mae = float(
            np.mean(np.abs(CONSTANT_GP_BASELINE - actual_gp))
        )

        for name, variant in variants.items():
            scored = variant.loc[eligible].copy()
            scored["VALUE"] = _nine_cat_value(scored)
            gp_values = scored["GP"].to_numpy(dtype=float)
            metrics = {
                "Target_Year": year,
                "Evaluation_Set": evaluation_set,
                "Variant": name,
                "GP_MAE": float(np.mean(np.abs(gp_values - actual_gp))),
                "GP_Bias": float(np.mean(gp_values - actual_gp)),
                "GP_Spearman": _spearman(gp_values, actual_gp),
                "GP_MAE_66_Baseline": baseline_66_mae,
                "Beats_66_GP_Baseline": bool(
                    np.mean(np.abs(gp_values - actual_gp)) < baseline_66_mae
                ),
                "Beats_11_4_GP_MAE": bool(
                    np.mean(np.abs(gp_values - actual_gp)) < TARGET_GP_MAE
                ),
                "VALUE_Spearman": _spearman(scored["VALUE"], truth["VALUE"]),
                "VALUE_MAE_z": float(
                    np.mean(np.abs(scored["VALUE"] - truth["VALUE"]))
                ),
                "Players": int(eligible.sum()),
            }
            for stat in M.TARGET_STATS:
                metrics[f"PG_{stat}_MAE"] = float(
                    np.mean(
                        np.abs(
                            scored[f"PG_{stat}"].to_numpy()
                            - truth[f"PG_{stat}"].to_numpy()
                        )
                    )
                )
                metrics[f"PG_{stat}_Spearman"] = _spearman(
                    scored[f"PG_{stat}"], truth[f"PG_{stat}"]
                )
                metrics[f"TOTAL_{stat}_Spearman"] = _spearman(
                    scored[f"T_{stat}"], truth[f"T_{stat}"]
                )
            for top in TOP_N:
                pred_top = set(scored["VALUE"].nlargest(top).index)
                true_top = set(truth["VALUE"].nlargest(top).index)
                metrics[f"Top{top}_Overlap"] = len(pred_top & true_top) / top
            records.append(metrics)

    minutes_records = _minutes_test(
        year,
        table,
        act_mpg,
        act_gp,
        {
            "none": mpg_flat.to_numpy(),
            "age": mpg.to_numpy(),
            "role": mpg_role.to_numpy(),
            "age_position": mpg_position.to_numpy(),
            "last_season": last_mpg,
        },
    )
    players["BASELINE_MPG"] = table["BASELINE_ACTIVE_MPG"].to_numpy()
    players["PRED_MPG_AGE"] = mpg.to_numpy()
    players["PRED_MPG_ROLE"] = mpg_role.to_numpy()
    players["PRED_MPG_POSITION"] = mpg_position.to_numpy()
    players["PROD_Z"] = table["PROD_Z"].to_numpy()
    return pd.DataFrame(records), players, minutes_records


def _select_weights(results: pd.DataFrame, years: list[int]) -> dict:
    """Select weights that preserve value rank and pass checks in two years."""
    selection_rows = []
    candidate_names = [
        _weight_variant(starter, bench)
        for starter in GAMES_MODEL_WEIGHT_GRID
        for bench in GAMES_MODEL_WEIGHT_GRID
    ]
    for year in years:
        year_results = results.loc[results["Target_Year"] == year]
        primary = year_results.loc[
            year_results["Evaluation_Set"] == "500+ actual minutes"
        ].set_index("Variant")
        expanded = year_results.loc[
            year_results["Evaluation_Set"] == "200+ actual minutes"
        ].set_index("Variant")
        default = primary.loc["default_blend"]
        default_expanded = expanded.loc["default_blend"]
        candidates = primary.loc[primary.index.isin(candidate_names)].join(
            expanded[
                [
                    "GP_MAE",
                    "GP_Spearman",
                    "GP_MAE_66_Baseline",
                    "VALUE_Spearman",
                ]
            ],
            how="inner",
            lsuffix="_500",
            rsuffix="_200",
        )
        candidates = candidates.loc[
            (candidates["VALUE_Spearman_500"] >= default["VALUE_Spearman"] - 1e-12)
            & (
                candidates["VALUE_Spearman_200"]
                >= default_expanded["VALUE_Spearman"] - 1e-12
            )
            & (candidates["GP_Spearman_500"] > 0)
            & (candidates["GP_Spearman_200"] > 0)
            & (candidates["GP_MAE_500"] < candidates["GP_MAE_66_Baseline_500"])
            & (candidates["GP_MAE_200"] < candidates["GP_MAE_66_Baseline_200"])
            & (candidates["GP_MAE_500"] < TARGET_GP_MAE)
        ].sort_values(
            ["GP_MAE_500", "GP_MAE_200", "VALUE_Spearman_500"],
            ascending=[True, True, False],
        )
        if candidates.empty:
            selection_rows.append(
                {
                    "Target_Year": year,
                    "Winner": None,
                    "Reason": (
                        "No candidate preserved value rank on both sets while "
                        "beating the 66-GP baselines and 11.4 GP-MAE target."
                    ),
                }
            )
        else:
            winner = candidates.iloc[0]
            selection_rows.append(
                {
                    "Target_Year": year,
                    "Winner": winner.name,
                    "GP_MAE_500": float(winner["GP_MAE_500"]),
                    "GP_MAE_200": float(winner["GP_MAE_200"]),
                    "GP_MAE_66_Baseline_500": float(
                        winner["GP_MAE_66_Baseline_500"]
                    ),
                    "GP_Spearman_500": float(winner["GP_Spearman_500"]),
                    "GP_Spearman_200": float(winner["GP_Spearman_200"]),
                    "VALUE_Spearman_500": float(
                        winner["VALUE_Spearman_500"]
                    ),
                    "VALUE_Spearman_200": float(
                        winner["VALUE_Spearman_200"]
                    ),
                }
            )

    winners = [
        row["Winner"] for row in selection_rows if row["Winner"] is not None
    ]
    win_counts = pd.Series(winners).value_counts()
    qualifying = win_counts[win_counts >= 2]
    if qualifying.empty:
        selected_starter = P.STARTER_GP_MODEL_WEIGHT
        selected_bench = P.BENCH_GP_MODEL_WEIGHT
        selected_variant = "default_blend"
        reason = "No candidate won at least two years; retained initial weights."
    else:
        rows = results.loc[
            (results["Evaluation_Set"] == "500+ actual minutes")
            & results["Variant"].isin(qualifying.index)
        ].copy()
        winner_pairs = pd.DataFrame(selection_rows).dropna(subset=["Winner"])
        rows = rows.merge(
            winner_pairs[["Target_Year", "Winner"]],
            left_on=["Target_Year", "Variant"],
            right_on=["Target_Year", "Winner"],
            how="inner",
        )
        summary = (
            rows.groupby("Variant")
            .agg(Wins=("Target_Year", "nunique"), Mean_GP_MAE=("GP_MAE", "mean"))
            .sort_values(["Mean_GP_MAE", "Wins"], ascending=[True, False])
        )
        selected_variant = str(summary.index[0])
        starter_text, bench_text = selected_variant.split("_bench_")
        selected_starter = float(starter_text.removeprefix("starter_"))
        selected_bench = float(bench_text)
        reason = "Selected lowest mean GP error among variants winning at least two years."

    return {
        "selected_starter_model_weight": selected_starter,
        "selected_bench_model_weight": selected_bench,
        "selected_variant": selected_variant,
        "reason": reason,
        "yearly_winners": selection_rows,
    }


def _recommend(
    results: pd.DataFrame, minutes: pd.DataFrame, years: list[int]
) -> dict:
    """Turn the backtest into settings advice for project.py.

    Games: "fitted" is recommended only if, in at least two seasons, it has a
    lower games error AND a value rank no worse (within a small tolerance) than
    the blend on BOTH the 200+ and 500+ minute sets, and its mean games rank
    correlation is higher.
    Minutes: "role" replaces "age" only if, in at least two seasons, bench
    players' MPG error falls by >= 0.2, the 20 predicted top risers gain >= 1.0
    MPG more than everyone else, and value rank is no worse than the age
    version. Otherwise "age" is kept if it beats "none" in at least two
    seasons, else "none".
    """
    sets = ("200+ actual minutes", "500+ actual minutes")

    def pick(year: int, evaluation_set: str, variant: str) -> pd.Series:
        rows = results.loc[
            (results["Target_Year"] == year)
            & (results["Evaluation_Set"] == evaluation_set)
            & (results["Variant"] == variant)
        ]
        return rows.iloc[0]

    games_rows, role_rows, age_rows = [], [], []
    for year in years:
        passes = True
        detail = {"Target_Year": year}
        for evaluation_set in sets:
            blend = pick(year, evaluation_set, "default_blend")
            fitted = pick(year, evaluation_set, "fitted_games")
            better_error = fitted["GP_MAE"] < blend["GP_MAE"]
            keeps_rank = fitted["VALUE_Spearman"] >= blend["VALUE_Spearman"] - VALUE_TOLERANCE
            passes &= bool(better_error and keeps_rank)
            detail[evaluation_set] = {
                "GP_MAE_blend": float(blend["GP_MAE"]),
                "GP_MAE_fitted": float(fitted["GP_MAE"]),
                "GP_Spearman_blend": float(blend["GP_Spearman"]),
                "GP_Spearman_fitted": float(fitted["GP_Spearman"]),
                "VALUE_Spearman_blend": float(blend["VALUE_Spearman"]),
                "VALUE_Spearman_fitted": float(fitted["VALUE_Spearman"]),
            }
        detail["passes"] = bool(passes)
        games_rows.append(detail)
    mean_rank_gain = float(
        np.mean(
            [
                row[s]["GP_Spearman_fitted"] - row[s]["GP_Spearman_blend"]
                for row in games_rows
                for s in sets
            ]
        )
    )
    games_wins = sum(row["passes"] for row in games_rows)
    games_choice = (
        "fitted" if len(years) >= 2 and games_wins >= 2 and mean_rank_gain > 0 else "blend"
    )

    for year in years:
        yearly = minutes.loc[minutes["Target_Year"] == year]
        bench = yearly.loc[yearly["Group"] == "bench_under_20"].set_index("Method")
        if not {"age", "role", "none"}.issubset(bench.index):
            continue
        value_age = pick(year, "500+ actual minutes", "default_blend")["VALUE_Spearman"]
        value_role = pick(year, "500+ actual minutes", "role_mpg")["VALUE_Spearman"]
        value_none = pick(year, "500+ actual minutes", "pipeline_no_age")["VALUE_Spearman"]
        gain = float(bench.loc["age", "MPG_MAE"] - bench.loc["role", "MPG_MAE"])
        lift = float(bench.loc["role", "Riser_Lift"])
        role_rows.append(
            {
                "Target_Year": year,
                "Bench_MAE_gain_vs_age": gain,
                "Riser_Lift_role": lift,
                "Riser_Lift_age": float(bench.loc["age", "Riser_Lift"]),
                "VALUE_Spearman_role": float(value_role),
                "VALUE_Spearman_age": float(value_age),
                "passes": bool(
                    gain >= MIN_BENCH_MAE_GAIN
                    and lift >= MIN_RISER_LIFT
                    and value_role >= value_age - VALUE_TOLERANCE
                ),
            }
        )
        age_rows.append(
            {"Target_Year": year, "VALUE_Spearman_age": float(value_age),
             "VALUE_Spearman_none": float(value_none),
             "passes": bool(value_age > value_none)}
        )
    role_wins = sum(row["passes"] for row in role_rows)
    age_wins = sum(row["passes"] for row in age_rows)
    if len(years) >= 2 and role_wins >= 2:
        minutes_choice = "role"
    elif len(years) >= 2 and age_wins >= 2:
        minutes_choice = "age"
    elif len(years) >= 2:
        minutes_choice = "none"
    else:
        minutes_choice = "age"
    # Positional crowding: "fitted" only if, in at least two seasons, it lowers
    # MPG error for all players and keeps value rank (500+ minutes) within tolerance.
    position_rows = []
    for year in years:
        yearly = minutes.loc[(minutes["Target_Year"] == year) & (minutes["Group"] == "all")]
        by_method = yearly.set_index("Method")
        if not {"age", "age_position"}.issubset(by_method.index):
            continue
        value_age = pick(year, "500+ actual minutes", "fitted_games")["VALUE_Spearman"]
        value_position = pick(year, "500+ actual minutes", "fitted_games_position")["VALUE_Spearman"]
        mae_age = float(by_method.loc["age", "MPG_MAE"])
        mae_position = float(by_method.loc["age_position", "MPG_MAE"])
        position_rows.append(
            {
                "Target_Year": year,
                "MPG_MAE_age": mae_age,
                "MPG_MAE_age_position": mae_position,
                "VALUE_Spearman_age": float(value_age),
                "VALUE_Spearman_age_position": float(value_position),
                "passes": bool(
                    mae_position < mae_age and value_position >= value_age - VALUE_TOLERANCE
                ),
            }
        )
    position_wins = sum(row["passes"] for row in position_rows)
    position_choice = "fitted" if len(years) >= 2 and position_wins >= 2 else "none"
    if len(years) < 2:
        position_choice = P.POSITION_ADJUSTMENT
    note = (
        "" if len(years) >= 2
        else "Only one season was run; recommendations need at least two "
        "(run with no year argument). Defaults kept."
    )
    return {
        "GAMES_METHOD": games_choice,
        "MPG_ADJUSTMENT": minutes_choice,
        "POSITION_ADJUSTMENT": position_choice,
        "note": note,
        "position_evidence": position_rows,
        "games_evidence": games_rows,
        "games_mean_rank_gain": mean_rank_gain,
        "role_evidence": role_rows,
        "age_vs_none_evidence": age_rows,
    }


def run(year: int | None = None) -> None:
    history = P.load_history()
    last_year = int(history["SEASON_YEAR"].max())
    years = list(range(last_year - 2, last_year + 1)) if year is None else [year]
    for target_year in years:
        if (
            target_year > last_year
            or target_year - 1 < history["SEASON_YEAR"].min() + 3
        ):
            raise ValueError(
                f"Cannot backtest {target_year}; data covers through {last_year}."
            )

    decisions = json.loads(M.MODEL_METADATA_PATH.read_text())["policy_decisions"]
    pairs = M.load_training_pairs()
    scored_frames = []
    player_frames = []
    minutes_records: list[dict] = []
    for target_year in years:
        results, players, minutes = _score_year(target_year, history, pairs, decisions)
        scored_frames.append(results)
        player_frames.append(players)
        minutes_records.extend(minutes)
    all_results = pd.concat(scored_frames, ignore_index=True)
    all_results.to_csv(RESULTS_PATH, index=False)
    pd.concat(player_frames, ignore_index=True).to_csv(PLAYERS_PATH, index=False)

    selection = _select_weights(all_results, years)
    WEIGHT_SELECTION_PATH.write_text(json.dumps(selection, indent=2) + "\n")
    candidate_results = all_results.loc[
        all_results["Variant"].str.startswith("starter_")
    ]
    for (target_year, evaluation_set), group in candidate_results.groupby(
        ["Target_Year", "Evaluation_Set"], sort=True
    ):
        print(f"\n{target_year} / {evaluation_set}, sorted by games error")
        print(
            group.sort_values("GP_MAE")[
                [
                    "Variant",
                    "GP_MAE",
                    "GP_MAE_66_Baseline",
                    "Beats_11_4_GP_MAE",
                    "GP_Spearman",
                    "VALUE_Spearman",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )
    print("\nSelected games weights:")
    print(json.dumps(selection, indent=2))

    minutes_table = pd.DataFrame(minutes_records)
    minutes_table.to_csv(MINUTES_TEST_PATH, index=False)
    compare = [
        "default_blend", "fitted_games", "role_mpg", "fitted_games_role", "fitted_games_position",
        "pipeline_no_age", "naive",
    ]
    shown = all_results.loc[all_results["Variant"].isin(compare)]
    for (target_year, evaluation_set), group in shown.groupby(
        ["Target_Year", "Evaluation_Set"], sort=True
    ):
        print(f"\n{target_year} / {evaluation_set}: new variants")
        print(
            group.set_index("Variant")
            .loc[compare, ["GP_MAE", "GP_Spearman", "VALUE_Spearman", "Top50_Overlap", "Top100_Overlap"]]
            .round(3)
            .to_string()
        )
    print("\nMinutes by role group (age vs role vs none vs last season):")
    print(
        minutes_table.loc[minutes_table["Group"].isin(["all", "bench_under_20"])]
        .round(3)
        .to_string(index=False)
    )

    recommendations = _recommend(all_results, minutes_table, years)
    RECOMMENDATIONS_PATH.write_text(json.dumps(recommendations, indent=2) + "\n")
    print("\nRecommended settings for project.py:")
    print(f'  GAMES_METHOD = "{recommendations["GAMES_METHOD"]}"')
    print(f'  MPG_ADJUSTMENT = "{recommendations["MPG_ADJUSTMENT"]}"')
    print(f'  POSITION_ADJUSTMENT = "{recommendations["POSITION_ADJUSTMENT"]}"')
    if recommendations["note"]:
        print("  " + recommendations["note"])
    print(
        f"\nWrote {RESULTS_PATH.name}, {PLAYERS_PATH.name}, "
        f"{WEIGHT_SELECTION_PATH.name}, {MINUTES_TEST_PATH.name} and "
        f"{RECOMMENDATIONS_PATH.name} to {M.PROCESSED_DIR}"
    )


# --------------------------------------------------------------------------
# Rookie backtest: hold out each rookie class, predict it from earlier classes
# --------------------------------------------------------------------------
ROOKIE_FIRST_CLASS = 2016
ROOKIE_RESULTS_PATH = M.PROCESSED_DIR / "rookie_backtest_results.csv"
ROOKIE_PLAYERS_PATH = M.PROCESSED_DIR / "rookie_backtest_players.csv"
ROOKIE_SUMMARY_PATH = M.PROCESSED_DIR / "rookie_backtest_summary.json"
PER_36 = 36.0


def _weighted_mae(actual, predicted, weight) -> float:
    return float(np.average(np.abs(np.asarray(actual) - np.asarray(predicted)), weights=weight))


def _rate_metrics(actual: pd.DataFrame, rates: pd.DataFrame, weight: pd.Series, prefix: str) -> dict:
    """Per-36 rate MAE per stat, plus FG% and FT% error weighted by attempts."""
    out = {}
    for stat in M.TARGET_STATS:
        out[f"{prefix}{stat}_MAE36"] = PER_36 * _weighted_mae(
            actual[f"{stat}_RATE"], rates[stat], weight
        )
    for made, att, name in (("FGM", "FGA", "FG_PCT"), ("FTM", "FTA", "FT_PCT")):
        attempts = actual[f"{att}_RATE"] * actual["MIN"]
        usable = attempts > 0
        actual_pct = actual[f"{made}_RATE"] / actual[f"{att}_RATE"]
        predicted_pct = rates[made] / rates[att]
        out[f"{prefix}{name}_MAE"] = _weighted_mae(
            actual_pct[usable], predicted_pct[usable], attempts[usable]
        )
    return out


def _value_spearman(actual: pd.DataFrame, rates: pd.DataFrame, mpg, gp) -> float:
    """Rank predicted 9-cat season value against actual, within the rookie class."""
    predicted = _nine_cat_value(_totals(rates, np.asarray(mpg), np.asarray(gp)))
    if predicted.nunique() < 2:
        return float("nan")
    return _spearman(predicted, _value_spearman_target(actual))


def _second_year_test(
    year: int,
    history: pd.DataFrame,
    draft: pd.DataFrame,
    pairs: pd.DataFrame,
    decisions: dict,
    models: dict,
    replacement: pd.Series,
    league_prev: pd.Series,
) -> dict:
    """Second-year players with a rookie season under 200 minutes: shrink toward
    replacement (old) versus toward their rookie prior (new)."""
    past = history.loc[history["SEASON_YEAR"] < year]
    first_year = history.groupby("PLAYER_ID")["SEASON_YEAR"].min()
    rookie_rows = past.loc[past["SEASON_YEAR"] == year - 1].set_index("PLAYER_ID")
    actual = history.loc[history["SEASON_YEAR"] == year].set_index("PLAYER_ID")
    candidates = [
        player_id
        for player_id in rookie_rows.index
        if first_year.get(player_id) == year - 1
        and rookie_rows.loc[player_id, "MIN"] < P.CREDIBILITY_MINUTES
        and player_id in actual.index
        and actual.loc[player_id, "MIN"] >= EXPANDED_MINUTES_THRESHOLD
    ]
    if len(candidates) < 5:
        return {"SecondYear_N": len(candidates)}
    # Only these players' rows: their rate features are per player, so this
    # matches the full run (league-relative production scores are not used here).
    latest, latest_year = P.latest_season_rows(past.loc[past["PLAYER_ID"].isin(candidates)])
    roster = pd.DataFrame({"PLAYER_ID": candidates, "TEAM_ABBREVIATION": "BT", "ROSTER_NAME": pd.NA})
    table = P.build_inference_table(roster, latest, latest_year, replacement)
    features = R.prior_features(table["PLAYER_ID"], past, draft, pd.Series(dtype=float))
    priors = R.predict_rookies(models, features, league_prev)
    targets = pd.DataFrame(
        np.tile(replacement.reindex(M.TARGET_STATS).to_numpy(), (len(table), 1)),
        index=table.index, columns=M.TARGET_STATS,
    )
    has = features["HAS_ROOKIE_PRIOR"]
    targets.loc[has, M.TARGET_STATS] = priors.loc[has, M.TARGET_STATS]
    bundle = M.build_bundle(M._train_slice(pairs, year), decisions)
    rates_old, _ = P.predict_with_credibility(bundle, table, replacement)
    rates_new, _ = P.predict_with_credibility(bundle, table, targets)
    truth = actual.loc[table["PLAYER_ID"]].reset_index()
    truth.index = table.index
    out = {"SecondYear_N": len(candidates)}
    out.update(_rate_metrics(truth, rates_old, truth["MIN"], "SecondYear_replacement_"))
    out.update(_rate_metrics(truth, rates_new, truth["MIN"], "SecondYear_draft_"))
    return out


def _score_rookie_class(
    year: int,
    history: pd.DataFrame,
    draft: pd.DataFrame,
    rookies: pd.DataFrame,
    pairs: pd.DataFrame,
    decisions: dict,
) -> tuple[dict, pd.DataFrame]:
    print(f"\n=== Rookie backtest: class of {year} from classes before {year} ===")
    past = history.loc[history["SEASON_YEAR"] < year]
    train = rookies.loc[rookies["SEASON_YEAR"] < year]
    test = rookies.loc[rookies["SEASON_YEAR"] == year].copy()
    models = R.fit_rookie_models(train, M.TARGET_STATS)
    league_prev = R.league_rates(past, M.TARGET_STATS).iloc[-1]
    replacement = P.replacement_rates(past, year - 1, {})
    _, typical_bench_gp = P.league_typical_games(past)
    actual_min_82 = test["MIN"] * test["GAMES_82"] / test["GP"]  # shortened seasons scaled to 82

    pred = R.predict_rookies(models, test, league_prev)
    # Layer 1 alone (draft slot and age, no team opportunity), to show what layer 2 adds.
    slot_only = R.predict_rookies(
        R.fit_rookie_models(train, M.TARGET_STATS, use_opportunity=False), test, league_prev
    )
    # Without the picks 4-15 MPG correction, to show what it adds.
    no_pick_fix = R.predict_rookies(
        R.fit_rookie_models(train, M.TARGET_STATS, use_pick_correction=False), test, league_prev
    )
    low, high = R.PICK_CORRECTION_RANGE
    lottery = test["OVERALL_PICK"].between(low, high)
    # Layers 1-2 rates (no college stats or height), to show what layer 3 adds.
    no_college = R.predict_rookies(
        R.fit_rookie_models(train, M.TARGET_STATS, use_college=False), test, league_prev
    )
    replacement_rates = pd.DataFrame(
        np.tile(replacement.reindex(M.TARGET_STATS).to_numpy(), (len(test), 1)),
        index=test.index, columns=M.TARGET_STATS,
    )
    # What project.py uses: draft prior for ROOKIE_RATE_STATS, replacement otherwise.
    deployed_rates = replacement_rates.copy()
    deployed_rates[P.ROOKIE_RATE_STATS] = pred[P.ROOKIE_RATE_STATS]
    mean_mpg = float(train["MPG"].mean())
    mean_games = float(train["GAMES_82"].mean())

    rated = test["MIN"] >= R.RATE_MIN_MINUTES
    result = {
        "Class": year,
        "Rookies": len(test),
        "Rookies_200_Min": int(rated.sum()),
        "MPG_MAE_draft": float(np.mean(np.abs(test["MPG"] - pred["MPG"]))),
        "MPG_MAE_typical_rookie": float(np.mean(np.abs(test["MPG"] - mean_mpg))),
        "MPG_MAE_slot_only": float(np.mean(np.abs(test["MPG"] - slot_only["MPG"]))),
        "MPG_MAE_no_pick_fix": float(np.mean(np.abs(test["MPG"] - no_pick_fix["MPG"]))),
        "Picks4_15_MPG_MAE_draft": float(np.mean(np.abs(test["MPG"] - pred["MPG"])[lottery])),
        "Picks4_15_MPG_MAE_no_pick_fix": float(np.mean(np.abs(test["MPG"] - no_pick_fix["MPG"])[lottery])),
        "Picks4_15_SEASON_MIN_MAE_draft": float(
            np.mean(np.abs(actual_min_82 - pred["MPG"] * pred["GAMES_82"])[lottery])
        ),
        "Picks4_15_SEASON_MIN_MAE_no_pick_fix": float(
            np.mean(np.abs(actual_min_82 - no_pick_fix["MPG"] * no_pick_fix["GAMES_82"])[lottery])
        ),
        "Picks4_15_SEASON_MIN_miss_draft": float(
            np.mean((actual_min_82 - pred["MPG"] * pred["GAMES_82"])[lottery])
        ),
        "Picks4_15_SEASON_MIN_miss_no_pick_fix": float(
            np.mean((actual_min_82 - no_pick_fix["MPG"] * no_pick_fix["GAMES_82"])[lottery])
        ),
        "Pick_MPG_shift": float(models["pick_mpg_shift"]),
        "GAMES_MAE_slot_only": float(np.mean(np.abs(test["GAMES_82"] - slot_only["GAMES_82"]))),
        "SEASON_MIN_MAE_slot_only": float(
            np.mean(np.abs(actual_min_82 - slot_only["MPG"] * slot_only["GAMES_82"]))
        ),
        "GAMES_MAE_draft": float(np.mean(np.abs(test["GAMES_82"] - pred["GAMES_82"]))),
        "GAMES_MAE_current": float(np.mean(np.abs(test["GAMES_82"] - typical_bench_gp))),
        "GAMES_MAE_typical_rookie": float(np.mean(np.abs(test["GAMES_82"] - mean_games))),
        "SEASON_MIN_MAE_draft": float(np.mean(np.abs(actual_min_82 - pred["MPG"] * pred["GAMES_82"]))),
        "SEASON_MIN_MAE_typical_rookie": float(np.mean(np.abs(actual_min_82 - mean_mpg * mean_games))),
        "MPG_Spearman_draft": _spearman(pred["MPG"], test["MPG"]),
        "GAMES_Spearman_draft": _spearman(pred["GAMES_82"], test["GAMES_82"]),
        "VALUE_Spearman_draft": _value_spearman(test, deployed_rates, pred["MPG"], pred["GAMES_82"]),
        "VALUE_Spearman_draft_all_rates": _value_spearman(test, pred, pred["MPG"], pred["GAMES_82"]),
        "VALUE_Spearman_pick_order": _spearman(-test["LOG_PICK"], _value_spearman_target(test)),
    }
    result.update(_rate_metrics(test.loc[rated], pred.loc[rated], test.loc[rated, "MIN"], "draft_"))
    result.update(
        _rate_metrics(test.loc[rated], replacement_rates.loc[rated], test.loc[rated, "MIN"], "replacement_")
    )
    result.update(
        _rate_metrics(test.loc[rated], no_college.loc[rated], test.loc[rated, "MIN"], "no_college_")
    )
    result["VALUE_Spearman_no_college"] = _value_spearman(
        test, no_college, pred["MPG"], pred["GAMES_82"]
    )
    result.update(
        _second_year_test(year, history, draft, pairs, decisions, models, replacement, league_prev)
    )

    players = test[["PLAYER_ID", "PLAYER_NAME", "SEASON_YEAR", "OVERALL_PICK", "AGE", "MIN", "MPG", "GAMES_82"]].copy()
    players["PRED_MPG"] = pred["MPG"]
    players["PRED_GAMES_82"] = pred["GAMES_82"]
    for stat in ("PTS", "REB", "AST", "FG3M", "STL", "BLK"):
        players[f"{stat}_36"] = PER_36 * test[f"{stat}_RATE"]
        players[f"PRED_{stat}_36"] = PER_36 * pred[stat]
    return result, players


def _value_spearman_target(test: pd.DataFrame) -> pd.Series:
    real_rates = pd.DataFrame(
        {stat: test[f"{stat}_RATE"].to_numpy() for stat in M.TARGET_STATS}, index=test.index
    )
    return _nine_cat_value(_totals(real_rates, test["MPG"].to_numpy(), test["GP"].to_numpy()))


def _rookie_summary(results: pd.DataFrame) -> dict:
    """Mean of each metric over classes, and whether the draft prior beat each
    baseline in every class (the project's adoption test)."""
    summary = {"classes": [int(y) for y in results["Class"]], "metrics": {}, "beats_every_class": {}}
    for column in results.columns:
        if column != "Class":
            summary["metrics"][column] = round(float(results[column].mean()), 4)
    comparisons = [("MPG_MAE_draft", "MPG_MAE_typical_rookie"),
                   ("GAMES_MAE_draft", "GAMES_MAE_current"),
                   ("GAMES_MAE_draft", "GAMES_MAE_typical_rookie"),
                   ("SEASON_MIN_MAE_draft", "SEASON_MIN_MAE_typical_rookie"),
                   ("MPG_MAE_draft", "MPG_MAE_slot_only"),
                   ("MPG_MAE_draft", "MPG_MAE_no_pick_fix"),
                   ("Picks4_15_MPG_MAE_draft", "Picks4_15_MPG_MAE_no_pick_fix"),
                   ("Picks4_15_SEASON_MIN_MAE_draft", "Picks4_15_SEASON_MIN_MAE_no_pick_fix"),
                   ("GAMES_MAE_draft", "GAMES_MAE_slot_only"),
                   ("SEASON_MIN_MAE_draft", "SEASON_MIN_MAE_slot_only")]
    for stat in M.TARGET_STATS + ["FG_PCT", "FT_PCT"]:
        suffix = "_MAE" if stat.endswith("_PCT") else "_MAE36"
        comparisons.append((f"draft_{stat}{suffix}", f"replacement_{stat}{suffix}"))
        comparisons.append((f"draft_{stat}{suffix}", f"no_college_{stat}{suffix}"))
        if f"SecondYear_draft_{stat}{suffix}" in results.columns:
            comparisons.append((f"SecondYear_draft_{stat}{suffix}", f"SecondYear_replacement_{stat}{suffix}"))
    for new, old in comparisons:
        both = results[[new, old]].dropna()
        summary["beats_every_class"][f"{new} < {old}"] = bool(len(both) and (both[new] < both[old]).all())
    return summary


def run_rookies(first_class: int = ROOKIE_FIRST_CLASS) -> None:
    history = P.load_history()
    draft = R.load_draft_history()
    rookies = R.rookie_seasons(history, draft, M.TARGET_STATS)
    decisions = json.loads(M.MODEL_METADATA_PATH.read_text())["policy_decisions"]
    pairs = M.load_training_pairs()
    last_class = int(rookies["SEASON_YEAR"].max())
    records, player_frames = [], []
    for year in range(first_class, last_class + 1):
        result, players = _score_rookie_class(year, history, draft, rookies, pairs, decisions)
        records.append(result)
        player_frames.append(players)
    results = pd.DataFrame(records)
    results.to_csv(ROOKIE_RESULTS_PATH, index=False)
    pd.concat(player_frames, ignore_index=True).to_csv(ROOKIE_PLAYERS_PATH, index=False)
    summary = _rookie_summary(results)
    ROOKIE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    show = ["Class", "Rookies", "MPG_MAE_draft", "MPG_MAE_slot_only", "MPG_MAE_typical_rookie",
            "GAMES_MAE_draft", "GAMES_MAE_slot_only", "GAMES_MAE_current",
            "VALUE_Spearman_draft", "VALUE_Spearman_no_college", "VALUE_Spearman_pick_order"]
    print("\nRookie classes (draft prior vs baselines):")
    print(results[show].round(3).to_string(index=False))
    print("\nPer-36 rate error, rookies with 200+ minutes (draft prior vs replacement):")
    rate_table = pd.DataFrame(
        {
            "draft": [summary["metrics"][f"draft_{s}_MAE36"] for s in M.TARGET_STATS],
            "no_college": [summary["metrics"][f"no_college_{s}_MAE36"] for s in M.TARGET_STATS],
            "replacement": [summary["metrics"][f"replacement_{s}_MAE36"] for s in M.TARGET_STATS],
            "2nd_yr_draft": [summary["metrics"].get(f"SecondYear_draft_{s}_MAE36") for s in M.TARGET_STATS],
            "2nd_yr_replacement": [summary["metrics"].get(f"SecondYear_replacement_{s}_MAE36") for s in M.TARGET_STATS],
        },
        index=M.TARGET_STATS,
    )
    print(rate_table.round(3).to_string())
    print("\nBeats the baseline in every class:")
    for key, value in summary["beats_every_class"].items():
        print(f"  {'yes' if value else 'NO '}  {key}")
    print(f"\nWrote {ROOKIE_RESULTS_PATH.name}, {ROOKIE_PLAYERS_PATH.name} and {ROOKIE_SUMMARY_PATH.name}")


# --------------------------------------------------------------------------
# Allocation backtest: the full projection, team minutes budget included, on
# rosters rebuilt from each target season's end-of-season teams
# --------------------------------------------------------------------------
ALLOCATION_RESULTS_PATH = M.PROCESSED_DIR / "allocation_backtest_results.csv"
ALLOCATION_PLAYERS_PATH = M.PROCESSED_DIR / "allocation_backtest_players.csv"
TEAM_CROWD_SHRINK_GRID = (0.5, 1.0)
QUALITY_BETA_GRID = (0.5, 1.0, 1.5, 2.0)
ALLOCATION_VARIANTS = {
    "current": {"rookie_method": "draft", "position_method": "fitted", "team_crowd_method": "none"},
    "no_position": {"rookie_method": "draft", "position_method": "none", "team_crowd_method": "none"},
    "rookie_replacement": {"rookie_method": "replacement", "position_method": "fitted", "team_crowd_method": "none"},
    **{
        f"budget_{method}": {
            "rookie_method": "draft", "position_method": "fitted",
            "team_crowd_method": "none", "allocation_method": method,
        }
        for method in ("raw_priority", "proportional", "soft", "none")
    },
    **{
        f"team_crowd_{shrink:g}": {
            "rookie_method": "draft", "position_method": "fitted",
            "team_crowd_method": "fitted", "team_crowd_shrink": shrink,
        }
        for shrink in TEAM_CROWD_SHRINK_GRID
    },
    # Thin-roster fixes: each one switched off in turn, and the quality-aware budget.
    "no_minutes_shrink": {"rookie_method": "draft", "position_method": "fitted",
                          "team_crowd_method": "none", "minutes_shrink": "none"},
    "no_rookie_team_quality": {"rookie_method": "draft", "position_method": "fitted",
                               "team_crowd_method": "none", "rookie_team_quality": False},
    "no_rookie_playmaker": {"rookie_method": "draft", "position_method": "fitted",
                            "team_crowd_method": "none", "rookie_playmaker": False},
    "soft_budget": {"rookie_method": "draft", "position_method": "fitted",
                    "team_crowd_method": "none", "allocation_method": "soft"},
    "soft_budget_no_playmaker": {"rookie_method": "draft", "position_method": "fitted",
                                 "team_crowd_method": "none", "allocation_method": "soft",
                                 "rookie_playmaker": False},
    **{
        f"quality_{beta:g}": {
            "rookie_method": "draft", "position_method": "fitted", "team_crowd_method": "none",
            "allocation_method": "quality", "quality_beta": beta,
        }
        for beta in QUALITY_BETA_GRID
    },
}


def _season_totals(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """T_[stat] columns for _nine_cat_value, from projected totals or actual rates x minutes."""
    if source == "projected":
        return pd.DataFrame(
            {f"T_{s}": frame[f"TOTAL_{s}"].to_numpy() for s in M.TARGET_STATS}, index=frame.index
        )
    return pd.DataFrame(
        {f"T_{s}": (frame[f"{s}_RATE"] * frame["MIN"]).to_numpy() for s in M.TARGET_STATS},
        index=frame.index,
    )


def _allocation_metrics(
    pred: pd.DataFrame, actual: pd.DataFrame, mask: pd.Series, mpg_column: str, minutes_column: str
) -> dict:
    p, a = pred.loc[mask], actual.loc[mask]
    if len(p) < 10:
        return {}
    act_mpg = a["MIN"] / a["GP"]
    out = {
        "Players": int(len(p)),
        "MPG_MAE": float(np.mean(np.abs(p[mpg_column] - act_mpg))),
        "MPG_Bias": float(np.mean(p[mpg_column] - act_mpg)),
        "SEASON_MIN_MAE": float(np.mean(np.abs(p[minutes_column] - a["MIN"]))),
        "SEASON_MIN_Spearman": _spearman(p[minutes_column], a["MIN"]),
    }
    projected = _season_totals(p, "projected")
    if minutes_column != "PROJECTED_SEASON_MIN":
        # Same rates, different season minutes.
        projected = projected.mul(p[minutes_column] / p["PROJECTED_SEASON_MIN"], axis=0)
    out["VALUE_Spearman"] = _spearman(
        _nine_cat_value(projected), _nine_cat_value(_season_totals(a, "actual"))
    )
    return out


def _score_allocation_year(
    year: int, history: pd.DataFrame, pairs: pd.DataFrame, decisions: dict, draft: pd.DataFrame
) -> tuple[list[dict], pd.DataFrame]:
    print(f"\n=== Allocation backtest: {year} on end-of-season rosters, data through {year - 1} ===")
    past = history.loc[history["SEASON_YEAR"] < year]
    season = history.loc[history["SEASON_YEAR"] == year].set_index("PLAYER_ID")
    roster = pd.DataFrame(
        {
            "PLAYER_ID": season.index.to_numpy(),
            "TEAM_ABBREVIATION": season["TEAM_ABBREVIATION"].to_numpy(),
            "ROSTER_NAME": season["PLAYER_NAME"].to_numpy(),
        }
    )
    newcomers = ~season.index.isin(past["PLAYER_ID"])
    # Rookie-season age, as rookie_ages.csv gives for current rookies.
    current_ages = season.loc[newcomers, "AGE"]
    bundle = M.build_bundle(M._train_slice(pairs, year), decisions)
    latest = P.latest_season_rows(past)

    records, frames = [], []
    for name, options in ALLOCATION_VARIANTS.items():
        try:
            pred = P.project_rosters(
                bundle, roster, past, latest=latest, draft=draft, current_ages=current_ages, **options
            )
        except ValueError as error:
            print(f"{name}: allocation failed ({error})")
            records.append({"Target_Year": year, "Variant": name, "Group": "all", "Failed": str(error)})
            continue
        pred = pred.set_index("PLAYER_ID")
        actual = season.loc[pred.index]
        act_mpg = actual["MIN"] / actual["GP"]
        groups = {
            "200+ actual minutes": actual["MIN"] >= EXPANDED_MINUTES_THRESHOLD,
            "500+ actual minutes": actual["MIN"] >= MIN_ACTUAL_MINUTES,
            "starters_24_plus_actual": (act_mpg >= 24) & (actual["GP"] >= 20),
            "rookies": pd.Series(~pred.index.isin(past["PLAYER_ID"]), index=pred.index),
        }
        scored = [(name, "PROJECTED_MPG", "PROJECTED_SEASON_MIN")]
        for label, mpg_column, minutes_column in scored:
            for group, mask in groups.items():
                metrics = _allocation_metrics(pred, actual, mask, mpg_column, minutes_column)
                if metrics:
                    records.append({"Target_Year": year, "Variant": label, "Group": group, **metrics})
        frame = pred[
            ["PLAYER_NAME", "TEAM_ABBREVIATION", "RAW_MPG", "PROJECTED_MPG", "PROJECTED_GP", "PROJECTED_SEASON_MIN"]
        ].copy()
        frame["Variant"], frame["Target_Year"] = name, year
        frame["ACTUAL_MPG"], frame["ACTUAL_GP"], frame["ACTUAL_MIN"] = act_mpg, actual["GP"], actual["MIN"]
        frames.append(frame.reset_index())
    return records, pd.concat(frames, ignore_index=True)


def _recommend_team_crowd(results: pd.DataFrame, years: list[int]) -> dict:
    """Team crowding is recommended ("fitted", best shrink) only if, in at least two
    seasons, it lowers 500+ minute season-minutes error versus the current setup and
    keeps value rank within tolerance."""
    rows = results.loc[results["Group"] == "500+ actual minutes"].set_index(["Target_Year", "Variant"])
    best = {"TEAM_CROWD_ADJUSTMENT": "none", "TEAM_CROWD_SHRINK": P.TEAM_CROWD_SHRINK, "team_crowd_evidence": []}
    best_key = None
    for shrink in TEAM_CROWD_SHRINK_GRID:
        name = f"team_crowd_{shrink:g}"
        wins, gain = 0, 0.0
        for year in years:
            if (year, name) not in rows.index or (year, "current") not in rows.index:
                continue
            new, old = rows.loc[(year, name)], rows.loc[(year, "current")]
            passes = bool(
                new["SEASON_MIN_MAE"] < old["SEASON_MIN_MAE"]
                and new["VALUE_Spearman"] >= old["VALUE_Spearman"] - VALUE_TOLERANCE
            )
            wins += passes
            gain += float(old["SEASON_MIN_MAE"] - new["SEASON_MIN_MAE"])
            best["team_crowd_evidence"].append(
                {
                    "Target_Year": year, "Shrink": shrink, "passes": passes,
                    "SEASON_MIN_MAE_current": float(old["SEASON_MIN_MAE"]),
                    "SEASON_MIN_MAE_team_crowd": float(new["SEASON_MIN_MAE"]),
                    "VALUE_Spearman_current": float(old["VALUE_Spearman"]),
                    "VALUE_Spearman_team_crowd": float(new["VALUE_Spearman"]),
                }
            )
        if len(years) >= 2 and wins >= 2 and (best_key is None or (wins, gain) > best_key):
            best_key = (wins, gain)
            best["TEAM_CROWD_ADJUSTMENT"], best["TEAM_CROWD_SHRINK"] = "fitted", shrink
    return best


def run_allocation(years: list[int] | None = None) -> None:
    history = P.load_history()
    last_year = int(history["SEASON_YEAR"].max())
    years = years or list(range(last_year - 2, last_year + 1))
    decisions = json.loads(M.MODEL_METADATA_PATH.read_text())["policy_decisions"]
    pairs = M.load_training_pairs()
    draft = R.load_draft_history()
    records, frames = [], []
    for year in years:
        year_records, players = _score_allocation_year(year, history, pairs, decisions, draft)
        records.extend(year_records)
        frames.append(players)
    results = pd.DataFrame(records)
    results.to_csv(ALLOCATION_RESULTS_PATH, index=False)
    pd.concat(frames, ignore_index=True).to_csv(ALLOCATION_PLAYERS_PATH, index=False)

    shown = results.loc[results["Group"].isin(["500+ actual minutes", "starters_24_plus_actual", "rookies"])]
    columns = [
        c for c in ["MPG_MAE", "MPG_Bias", "SEASON_MIN_MAE", "SEASON_MIN_Spearman", "VALUE_Spearman"]
        if c in shown.columns
    ]
    for (year, group), table in shown.groupby(["Target_Year", "Group"], sort=True):
        print(f"\n{year} / {group}")
        print(table.set_index("Variant")[columns].round(3).to_string())
    recommendation = _recommend_team_crowd(results, years)
    if RECOMMENDATIONS_PATH.exists():
        merged = json.loads(RECOMMENDATIONS_PATH.read_text())
        merged.update(recommendation)
        RECOMMENDATIONS_PATH.write_text(json.dumps(merged, indent=2) + "\n")
    print(
        f'\nRecommended: TEAM_CROWD_ADJUSTMENT = "{recommendation["TEAM_CROWD_ADJUSTMENT"]}"'
        f' (shrink {recommendation["TEAM_CROWD_SHRINK"]})'
    )
    print(f"Wrote {ALLOCATION_RESULTS_PATH.name} and {ALLOCATION_PLAYERS_PATH.name}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rookies":
        run_rookies()
    elif len(sys.argv) > 1 and sys.argv[1] == "allocation":
        run_allocation()
    else:
        run(int(sys.argv[1]) if len(sys.argv) > 1 else None)
        run_rookies()
        run_allocation()
