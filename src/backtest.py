"""Backtest games-played blends and fantasy rank accuracy by target season.

Run from the project root with no year to evaluate the latest three seasons,
or pass one season-ending year (for example 2024).

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
from history import fit_age_mpg_curve, fit_games_model, fit_role_curve

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
            "last_season": last_mpg,
        },
    )
    players["BASELINE_MPG"] = table["BASELINE_ACTIVE_MPG"].to_numpy()
    players["PRED_MPG_AGE"] = mpg.to_numpy()
    players["PRED_MPG_ROLE"] = mpg_role.to_numpy()
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
    note = (
        "" if len(years) >= 2
        else "Only one season was run; recommendations need at least two "
        "(run with no year argument). Defaults kept."
    )
    return {
        "GAMES_METHOD": games_choice,
        "MPG_ADJUSTMENT": minutes_choice,
        "note": note,
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
        "default_blend", "fitted_games", "role_mpg", "fitted_games_role",
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
    if recommendations["note"]:
        print("  " + recommendations["note"])
    print(
        f"\nWrote {RESULTS_PATH.name}, {PLAYERS_PATH.name}, "
        f"{WEIGHT_SELECTION_PATH.name}, {MINUTES_TEST_PATH.name} and "
        f"{RECOMMENDATIONS_PATH.name} to {M.PROCESSED_DIR}"
    )


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else None)