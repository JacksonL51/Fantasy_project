"""Phase 4: rate regression engine.

Pipeline
--------
1. Build expanding-window validation folds from the most recent transitions.
2. Tune (formulation, alpha) for every stat on those folds.
3. Decide per stat whether the tuned model replaces a simple fallback
   (naive prior rate, prior shrinkage, or component-derived points).
4. Re-score the chosen policies fold by fold and run diagnostics.
5. Refit on all transitions and persist the bundle plus metadata for Phase 5.

Fold evaluation and production inference share `build_bundle` and
`predict_rates`, so the policy that gets validated is the policy that ships.

Selection rule (see `decide_policy`): the tuned model is accepted for a stat
only if its improvement over the naive prior is positive in EVERY validation
fold AND the 95% bootstrap interval of the pooled improvement (all fold rows
combined) excludes zero.

Known limitation: tuning, selection and scoring all use the same folds, so
the reported selected-policy numbers are optimistic. The first genuinely
out-of-sample test is the next season.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_STATS = [
    "FGM",
    "FGA",
    "FG3M",
    "FTM",
    "FTA",
    "REB",
    "AST",
    "STL",
    "BLK",
    "PTS",
    "TOV",
]
FEATURES_BY_STAT = {
    stat: [f"{stat}_RATE_prior", "AGE_prior", "AGE_SQ", "MIN_prior"]
    for stat in TARGET_STATS
}

# Tuning and validation
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0)
MODEL_FORMULATIONS = ("level", "delta")
N_VALIDATION_FOLDS = 2  # the most recent N target seasons become folds
# NOTE: training_pairs.csv already filters MIN_prior >= 200, so 200 removes
# nothing here. Raise it (e.g. 800) to rank only fantasy-relevant players.
RANKING_MINUTES_THRESHOLD = 200.0
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20261028

# Fallback policies
FT_PRIOR_SHRINKAGE = 0.05
SHRINKAGE_FALLBACK_STATS = frozenset({"FTM", "FTA"})
QUARTILE_CHECK_STATS = ("AST",)
WEIGHT_NORMALIZATION = "MIN_target divided by training-set mean"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAINING_PAIRS_PATH = PROCESSED_DIR / "training_pairs.csv"
VALIDATION_METRICS_PATH = PROCESSED_DIR / "validation_metrics.csv"
ALPHA_GRID_METRICS_PATH = PROCESSED_DIR / "alpha_grid_metrics.csv"
FOLD_METRICS_PATH = PROCESSED_DIR / "rolling_fold_metrics.csv"
BOOTSTRAP_METRICS_PATH = PROCESSED_DIR / "bootstrap_metrics.csv"
COEFFICIENTS_PATH = PROCESSED_DIR / "selected_coefficients.csv"
POLICY_DECISIONS_PATH = PROCESSED_DIR / "policy_decisions.csv"
PERCENTAGE_DIAGNOSTICS_PATH = PROCESSED_DIR / "percentage_diagnostics.csv"
MODEL_BUNDLE_PATH = PROCESSED_DIR / "rate_models.joblib"
MODEL_METADATA_PATH = PROCESSED_DIR / "rate_model_metadata.json"

REQUIRED_COLUMNS = [
    "AGE_prior",
    "MIN_prior",
    "MIN_target",
    "YEAR_CURRENT",
    "YEAR_NEXT",
    *[
        column
        for stat in TARGET_STATS
        for column in (f"{stat}_RATE_prior", f"{stat}_RATE_target")
    ],
]


# --------------------------------------------------------------------------
# Data loading and folds
# --------------------------------------------------------------------------
def load_training_pairs(
    csv_path: Path = TRAINING_PAIRS_PATH,
) -> pd.DataFrame:
    """Load transitions with complete, finite values in every modeling column."""
    pairs = pd.read_csv(csv_path)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in pairs.columns
    ]
    if missing_columns:
        raise ValueError(
            "Training pairs CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    required_values = pairs[REQUIRED_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    finite_rows = np.isfinite(required_values.to_numpy(dtype=float)).all(axis=1)
    pairs = pairs.loc[finite_rows].copy()
    pairs.loc[:, REQUIRED_COLUMNS] = required_values.loc[finite_rows]
    pairs["AGE_SQ"] = pairs["AGE_prior"] ** 2

    print(f"Valid transitions: {len(pairs)}")
    print("Transitions per YEAR_NEXT:")
    print(pairs["YEAR_NEXT"].value_counts().sort_index().to_string())
    return pairs


def rolling_origin_folds(
    pairs: pd.DataFrame,
    n_folds: int = N_VALIDATION_FOLDS,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Expanding-window folds over the most recent `n_folds` target seasons.

    Fold for season Y trains on every transition with YEAR_NEXT < Y and
    validates on the Y-1 -> Y transition. Adding seasons to the data
    automatically moves the folds forward.
    """
    years = sorted(int(year) for year in pairs["YEAR_NEXT"].unique())
    if len(years) <= n_folds:
        raise ValueError(
            f"Need more than {n_folds} target seasons to build {n_folds} "
            f"folds with training data; found {years}."
        )

    folds = []
    for year in years[-n_folds:]:
        train = pairs.loc[pairs["YEAR_NEXT"] < year].copy()
        validation = pairs.loc[
            (pairs["YEAR_CURRENT"] == year - 1) & (pairs["YEAR_NEXT"] == year)
        ].copy()
        if train.empty or validation.empty:
            raise ValueError(
                f"Cannot create fold for {year - 1} -> {year}: "
                f"train={len(train)}, validation={len(validation)}."
            )
        print(
            f"Fold {year}: train={len(train)} rows, "
            f"validation={len(validation)} rows"
        )
        if len(validation) < 100:
            print(f"Warning: fold {year} has fewer than 100 validation rows.")
        folds.append((year, train, validation))
    return folds


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def create_rate_model(alpha: float) -> Pipeline:
    """Scaler plus Ridge, so standardization is fit on training rows only."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def fit_stat_model(
    train: pd.DataFrame,
    stat: str,
    alpha: float,
    formulation: str,
) -> Pipeline:
    """Fit one rate model with mean-one minute weights.

    "level" predicts the target rate; "delta" predicts target minus prior,
    which shrinks toward "no change" instead of toward the league mean.
    """
    if formulation not in MODEL_FORMULATIONS:
        raise ValueError(f"Unknown model formulation: {formulation}")
    target = train[f"{stat}_RATE_target"]
    if formulation == "delta":
        target = target - train[f"{stat}_RATE_prior"]

    sample_weight = train["MIN_target"].to_numpy(dtype=float)
    sample_weight = sample_weight / sample_weight.mean()
    model = create_rate_model(alpha)
    model.fit(
        train[FEATURES_BY_STAT[stat]],
        target,
        ridge__sample_weight=sample_weight,
    )
    return model


def predict_stat_model(
    model: Pipeline,
    data: pd.DataFrame,
    stat: str,
    formulation: str,
) -> np.ndarray:
    """Predict a rate, adding the prior back for a delta model."""
    prediction = model.predict(data[FEATURES_BY_STAT[stat]])
    if formulation == "delta":
        prediction = data[f"{stat}_RATE_prior"].to_numpy(dtype=float) + prediction
    return np.asarray(prediction, dtype=float)


def _coefficient_record(
    model: Pipeline, stat: str, formulation: str, alpha: float, stage: str
) -> dict[str, float | str]:
    """Own-rate coefficient on the raw (unstandardized) level scale."""
    ridge = model.named_steps["ridge"]
    scaler = model.named_steps["scaler"]
    own_rate_coefficient = float((ridge.coef_ / scaler.scale_)[0])
    if formulation == "delta":
        own_rate_coefficient += 1.0  # implied level coefficient
    return {
        "Stage": stage,
        "Stat": stat,
        "Formulation": formulation,
        "Alpha": alpha,
        "Own_Rate_Raw_Coefficient": own_rate_coefficient,
    }


# --------------------------------------------------------------------------
# Policy bundle: one code path for validation and production
# --------------------------------------------------------------------------
def build_bundle(
    train: pd.DataFrame, decisions: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Fit everything the selected per-stat policies need, using `train` only."""
    models: dict[str, Pipeline] = {}
    settings: dict[str, dict[str, object]] = {}
    for stat in TARGET_STATS:
        decision = decisions[stat]
        policy = str(decision["selected_policy"])
        if policy in MODEL_FORMULATIONS:
            alpha = float(decision["tuned_alpha"])
            models[stat] = fit_stat_model(train, stat, alpha, policy)
            settings[stat] = {
                "formulation": policy,
                "alpha": alpha,
                "features": FEATURES_BY_STAT[stat],
            }
        elif policy == "naive_prior":
            settings[stat] = {"formulation": "prior", "alpha": None, "features": []}
        elif policy == "prior_shrinkage":
            settings[stat] = {
                "formulation": "prior_shrinkage",
                "alpha": None,
                "features": [],
                "shrinkage": FT_PRIOR_SHRINKAGE,
            }
        elif policy == "component_derived" and stat == "PTS":
            settings[stat] = {
                "formulation": "component_derived",
                "alpha": None,
                "features": [],
            }
        else:
            raise ValueError(f"Unsupported policy for {stat}: {policy}")

    return {
        "models": models,
        "target_stats": TARGET_STATS,
        "features_by_stat": FEATURES_BY_STAT,
        "settings_by_stat": settings,
        "prior_shrinkage_reference_by_stat": {
            stat: float(
                np.average(
                    train[f"{stat}_RATE_target"], weights=train["MIN_target"]
                )
            )
            for stat in TARGET_STATS
        },
        "weight_normalization": WEIGHT_NORMALIZATION,
        "fit_transition_years": sorted(
            int(year) for year in train["YEAR_NEXT"].unique()
        ),
    }


def predict_rates(bundle: dict[str, object], data: pd.DataFrame) -> pd.DataFrame:
    """Apply persisted model, prior, shrinkage, and component-derived policies."""
    models = bundle["models"]
    settings_by_stat = bundle["settings_by_stat"]
    references = bundle["prior_shrinkage_reference_by_stat"]
    predictions: dict[str, np.ndarray] = {}
    for stat in bundle["target_stats"]:
        settings = settings_by_stat[stat]
        formulation = settings["formulation"]
        prior = data[f"{stat}_RATE_prior"].to_numpy(dtype=float)
        if formulation in MODEL_FORMULATIONS:
            if stat not in models:
                raise ValueError(f"Persisted model missing for {stat}.")
            prediction = predict_stat_model(models[stat], data, stat, formulation)
        elif formulation == "prior":
            prediction = prior
        elif formulation == "prior_shrinkage":
            shrinkage = float(settings["shrinkage"])
            prediction = (1 - shrinkage) * prior + shrinkage * float(references[stat])
        elif formulation == "component_derived" and stat == "PTS":
            prediction = (
                2 * predictions["FGM"] + predictions["FG3M"] + predictions["FTM"]
            )
        else:
            raise ValueError(
                f"Unsupported persisted formulation for {stat}: {formulation}"
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"Prediction for {stat} is non-finite.")
        predictions[stat] = np.asarray(prediction, dtype=float)
    return pd.DataFrame(predictions, index=data.index)


def load_model_bundle(path: Path = MODEL_BUNDLE_PATH) -> dict[str, object]:
    """Load the persisted bundle for Phase 5 inference."""
    return joblib.load(path)


# --------------------------------------------------------------------------
# Scoring and bootstrap
# --------------------------------------------------------------------------
def _weighted_mae(
    actual: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> float:
    return float(mean_absolute_error(actual, prediction, sample_weight=weights))


def _debiased_mae(
    actual: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> float:
    bias = float(np.average(prediction - actual, weights=weights))
    return _weighted_mae(actual, prediction - bias, weights)


def _ranking_correlation(
    data: pd.DataFrame, actual: np.ndarray, prediction: np.ndarray
) -> float:
    eligible = data["MIN_prior"].to_numpy(dtype=float) >= RANKING_MINUTES_THRESHOLD
    if eligible.sum() < 2:
        return float("nan")
    return float(spearmanr(actual[eligible], prediction[eligible]).statistic)


def _score_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    stat: str,
    prediction: np.ndarray,
) -> dict[str, float]:
    actual = validation[f"{stat}_RATE_target"].to_numpy(dtype=float)
    weights = validation["MIN_target"].to_numpy(dtype=float)
    naive = validation[f"{stat}_RATE_prior"].to_numpy(dtype=float)
    bias = float(np.average(prediction - actual, weights=weights))
    naive_bias = float(np.average(naive - actual, weights=weights))
    naive_mae = _weighted_mae(actual, naive, weights)
    model_mae = _weighted_mae(actual, prediction, weights)
    league_mean = np.average(
        train[f"{stat}_RATE_target"], weights=train["MIN_target"]
    )
    league_mean_mae = _weighted_mae(
        actual, np.full(len(validation), league_mean), weights
    )
    return {
        "Model_MAE": model_mae,
        "Model_Debiased_MAE": _weighted_mae(actual, prediction - bias, weights),
        "Model_Unweighted_MAE": float(mean_absolute_error(actual, prediction)),
        "Model_Unweighted_RMSE": float(
            np.sqrt(mean_squared_error(actual, prediction))
        ),
        "Naive_MAE": naive_mae,
        "Naive_Debiased_MAE": _weighted_mae(actual, naive - naive_bias, weights),
        "League_Mean_MAE": league_mean_mae,
        "Model_Bias": bias,
        "Model_Unweighted_Bias": float(np.mean(prediction - actual)),
        "Naive_Bias": naive_bias,
        "Model_RMSE": float(
            np.sqrt(mean_squared_error(actual, prediction, sample_weight=weights))
        ),
        "Model_Spearman": _ranking_correlation(validation, actual, prediction),
        "Naive_Spearman": _ranking_correlation(validation, actual, naive),
        "Improvement_vs_Naive_pct": (
            100 * (naive_mae - model_mae) / naive_mae if naive_mae else np.nan
        ),
        "Improvement_vs_League_Mean_pct": (
            100 * (league_mean_mae - model_mae) / league_mean_mae
            if league_mean_mae
            else np.nan
        ),
        "Validation_Rows": len(validation),
        "Training_Rows": len(train),
    }


def _paired_bootstrap(
    actual: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Paired row bootstrap for MAE(first) - MAE(second).

    A positive difference means `second` is more accurate. Rows are resampled
    within the supplied data only, so the interval reflects player sampling
    noise but not season-to-season variation and is somewhat optimistic.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(actual)
    indices = rng.integers(0, n_rows, size=(BOOTSTRAP_ITERATIONS, n_rows))
    sampled_weights = weights[indices]
    first_error = np.abs(actual - first)[indices]
    second_error = np.abs(actual - second)[indices]
    differences = ((first_error - second_error) * sampled_weights).sum(
        axis=1
    ) / sampled_weights.sum(axis=1)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    point = _weighted_mae(actual, first, weights) - _weighted_mae(
        actual, second, weights
    )
    return float(point), {
        "CI_95_Lower": float(lower),
        "CI_95_Upper": float(upper),
        "Bootstrap_Iterations": BOOTSTRAP_ITERATIONS,
        "Bootstrap_Seed": seed,
    }


# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------
def tune_rate_models(
    folds: list[tuple[int, pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, dict[str, dict[str, float | str]]]:
    """Evaluate every formulation/alpha on each fold and pick one per stat.

    Selection minimizes mean debiased MAE across folds, then maximizes mean
    Spearman, then prefers "level" over "delta", then the smaller alpha.
    """
    records = []
    for validation_year, train, validation in folds:
        for stat in TARGET_STATS:
            for formulation in MODEL_FORMULATIONS:
                for alpha in ALPHA_GRID:
                    model = fit_stat_model(train, stat, alpha, formulation)
                    prediction = predict_stat_model(
                        model, validation, stat, formulation
                    )
                    records.append(
                        {
                            "Validation_Year": validation_year,
                            "Stat": stat,
                            "Formulation": formulation,
                            "Alpha": alpha,
                            **_score_predictions(train, validation, stat, prediction),
                        }
                    )

    grid = pd.DataFrame(records)
    keys = ["Stat", "Formulation", "Alpha"]
    mean_scores = grid.groupby(keys, as_index=False).agg(
        Mean_Debiased_MAE=("Model_Debiased_MAE", "mean"),
        Mean_Spearman=("Model_Spearman", "mean"),
        Mean_Raw_MAE=("Model_MAE", "mean"),
    )
    grid = grid.merge(mean_scores, on=keys, how="left", validate="many_to_one")

    ranked = mean_scores.assign(
        Formulation_Priority=mean_scores["Formulation"].map(
            {"level": 0, "delta": 1}
        )
    )
    selections: dict[str, dict[str, float | str]] = {}
    for stat in TARGET_STATS:
        best = (
            ranked.loc[ranked["Stat"] == stat]
            .sort_values(
                ["Mean_Debiased_MAE", "Mean_Spearman", "Formulation_Priority", "Alpha"],
                ascending=[True, False, True, True],
            )
            .iloc[0]
        )
        selections[stat] = {
            "formulation": str(best["Formulation"]),
            "alpha": float(best["Alpha"]),
        }
    return grid, selections


# --------------------------------------------------------------------------
# Policy decision
# --------------------------------------------------------------------------
def decide_policy(
    stat_index: int,
    stat: str,
    choice: dict[str, float | str],
    folds: list[tuple[int, pd.DataFrame, pd.DataFrame]],
) -> dict[str, object]:
    """Decide whether the tuned model for `stat` replaces its fallback.

    Accepted only if the improvement over naive is positive in every fold and
    the pooled bootstrap interval (all fold rows combined) excludes zero.
    Otherwise: FTM/FTA -> prior shrinkage, PTS -> component-derived,
    everything else -> naive prior.
    """
    formulation = str(choice["formulation"])
    alpha = float(choice["alpha"])
    bootstrap_records: list[dict[str, object]] = []
    metric_records: list[dict[str, object]] = []
    predictions: dict[int, np.ndarray] = {}
    fold_improvements: dict[int, float] = {}
    pooled: dict[str, list[np.ndarray]] = {
        "actual": [],
        "model": [],
        "naive": [],
        "weights": [],
    }

    for fold_index, (year, train, validation) in enumerate(folds):
        model = fit_stat_model(train, stat, alpha, formulation)
        prediction = predict_stat_model(model, validation, stat, formulation)
        predictions[year] = prediction
        actual = validation[f"{stat}_RATE_target"].to_numpy(dtype=float)
        naive = validation[f"{stat}_RATE_prior"].to_numpy(dtype=float)
        weights = validation["MIN_target"].to_numpy(dtype=float)

        improvement, interval = _paired_bootstrap(
            actual,
            naive,
            prediction,
            weights,
            BOOTSTRAP_SEED + stat_index * 100 + fold_index,
        )
        fold_improvements[year] = improvement
        bootstrap_records.append(
            {
                "Scope": "fold",
                "Validation_Year": year,
                "Stat": stat,
                "Formulation": formulation,
                "Alpha": alpha,
                "Compared_To": "naive_prior",
                "Improvement_MAE": improvement,
                **interval,
                "Decision": (
                    "model wins" if interval["CI_95_Lower"] > 0 else "tie_or_loss"
                ),
            }
        )
        metric_records.append(
            {
                "Evaluation_Set": "tuned_candidate",
                "Validation_Year": year,
                "Stat": stat,
                "Formulation": formulation,
                "Alpha": alpha,
                **_score_predictions(train, validation, stat, prediction),
            }
        )
        pooled["actual"].append(actual)
        pooled["model"].append(prediction)
        pooled["naive"].append(naive)
        pooled["weights"].append(weights)

    pooled_improvement, pooled_interval = _paired_bootstrap(
        np.concatenate(pooled["actual"]),
        np.concatenate(pooled["naive"]),
        np.concatenate(pooled["model"]),
        np.concatenate(pooled["weights"]),
        BOOTSTRAP_SEED + 5000 + stat_index,
    )
    bootstrap_records.append(
        {
            "Scope": "pooled",
            "Validation_Year": None,
            "Stat": stat,
            "Formulation": formulation,
            "Alpha": alpha,
            "Compared_To": "naive_prior",
            "Improvement_MAE": pooled_improvement,
            **pooled_interval,
            "Decision": (
                "model wins" if pooled_interval["CI_95_Lower"] > 0 else "tie_or_loss"
            ),
        }
    )

    positive_every_fold = all(value > 0 for value in fold_improvements.values())
    pooled_clear = pooled_interval["CI_95_Lower"] > 0
    accepted = positive_every_fold and pooled_clear
    if accepted:
        reason = "accepted"
    elif not positive_every_fold:
        reason = "not_positive_in_every_fold"
    else:
        reason = "pooled_ci_includes_zero"

    if accepted:
        selected_policy = formulation
    elif stat in SHRINKAGE_FALLBACK_STATS:
        selected_policy = "prior_shrinkage"
    elif stat == "PTS":
        selected_policy = "component_derived"
    else:
        selected_policy = "naive_prior"

    return {
        "decision": {
            "tuned_formulation": formulation,
            "tuned_alpha": alpha,
            "accepted": accepted,
            "reason": reason,
            "selected_policy": selected_policy,
            "fold_improvements": {
                str(year): value for year, value in fold_improvements.items()
            },
            "pooled_improvement": pooled_improvement,
            "pooled_ci_95": [
                pooled_interval["CI_95_Lower"],
                pooled_interval["CI_95_Upper"],
            ],
        },
        "bootstrap_records": bootstrap_records,
        "metric_records": metric_records,
        "predictions": predictions,
    }


def decisions_to_frame(decisions: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for stat in TARGET_STATS:
        decision = decisions[stat]
        row: dict[str, object] = {
            "Stat": stat,
            "Selected_Policy": decision["selected_policy"],
            "Tuned_Formulation": decision["tuned_formulation"],
            "Tuned_Alpha": decision["tuned_alpha"],
            "Accepted": decision["accepted"],
            "Reason": decision["reason"],
        }
        for year, value in decision["fold_improvements"].items():
            row[f"Improvement_MAE_{year}"] = value
        row["Pooled_Improvement_MAE"] = decision["pooled_improvement"]
        row["Pooled_CI_95_Lower"] = decision["pooled_ci_95"][0]
        row["Pooled_CI_95_Upper"] = decision["pooled_ci_95"][1]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def compare_points_policies(
    folds: list[tuple[int, pd.DataFrame, pd.DataFrame]],
    decisions: dict[str, dict[str, object]],
    candidate_predictions: dict[tuple[int, str], np.ndarray],
) -> list[dict[str, object]]:
    """Direct tuned PTS model vs PTS rebuilt from the selected components.

    Informational: the PTS policy itself comes from `decide_policy`.
    """
    component_decisions = {
        **decisions,
        "PTS": {**decisions["PTS"], "selected_policy": "component_derived"},
    }
    records = []
    for fold_index, (year, train, validation) in enumerate(folds):
        component = predict_rates(
            build_bundle(train, component_decisions), validation
        )["PTS"].to_numpy(dtype=float)
        direct = candidate_predictions[(year, "PTS")]
        actual = validation["PTS_RATE_target"].to_numpy(dtype=float)
        weights = validation["MIN_target"].to_numpy(dtype=float)
        difference, interval = _paired_bootstrap(
            actual,
            direct,
            component,
            weights,
            BOOTSTRAP_SEED + 10_000 + fold_index,
        )
        if interval["CI_95_Lower"] <= 0 <= interval["CI_95_Upper"]:
            label = "within_noise"
        elif difference < 0:
            label = "direct_better"
        else:
            label = "components_better"
        records.append(
            {
                "Scope": "fold",
                "Validation_Year": year,
                "Stat": "PTS",
                "Compared_To": "component_derived",
                "Direct_MAE": _weighted_mae(actual, direct, weights),
                "Direct_Debiased_MAE": _debiased_mae(actual, direct, weights),
                "Direct_Spearman": _ranking_correlation(validation, actual, direct),
                "Component_MAE": _weighted_mae(actual, component, weights),
                "Component_Debiased_MAE": _debiased_mae(actual, component, weights),
                "Component_Spearman": _ranking_correlation(
                    validation, actual, component
                ),
                "MAE_Difference_First_Minus_Second": difference,
                **interval,
                "Decision": label,
            }
        )
    return records


def _implied_percentage_records(
    year: int, validation: pd.DataFrame, predictions: pd.DataFrame
) -> list[dict[str, object]]:
    """Volume-weighted FG% and FT% implied by the selected policy.

    Compares policy-implied, naive-implied and actual percentages so you can
    see whether mixing level/delta/naive components biases the percentages.
    """
    weights = validation["MIN_target"].to_numpy(dtype=float)

    def implied(made: np.ndarray, attempted: np.ndarray) -> float:
        return float((weights * made).sum() / (weights * attempted).sum())

    records = []
    for label, made, attempted in (("FG%", "FGM", "FGA"), ("FT%", "FTM", "FTA")):
        actual = implied(
            validation[f"{made}_RATE_target"].to_numpy(dtype=float),
            validation[f"{attempted}_RATE_target"].to_numpy(dtype=float),
        )
        naive = implied(
            validation[f"{made}_RATE_prior"].to_numpy(dtype=float),
            validation[f"{attempted}_RATE_prior"].to_numpy(dtype=float),
        )
        policy = implied(
            predictions[made].to_numpy(dtype=float),
            predictions[attempted].to_numpy(dtype=float),
        )
        records.append(
            {
                "Validation_Year": year,
                "Metric": label,
                "Actual": actual,
                "Naive_Implied": naive,
                "Policy_Implied": policy,
                "Naive_Error_pp": 100 * (naive - actual),
                "Policy_Error_pp": 100 * (policy - actual),
            }
        )
    return records


def _points_consistency_record(
    year: int, validation: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, float]:
    """Gap between predicted PTS and 2*FGM + FG3M + FTM from predictions."""
    weights = validation["MIN_target"].to_numpy(dtype=float)
    derived = 2 * predictions["FGM"] + predictions["FG3M"] + predictions["FTM"]
    gap = (predictions["PTS"] - derived).to_numpy(dtype=float)
    return {
        "Validation_Year": year,
        "Weighted_Mean_Gap": float(np.average(gap, weights=weights)),
        "Weighted_MAE_Gap": float(np.average(np.abs(gap), weights=weights)),
    }


def _top_quartile_records(
    stat: str,
    folds: list[tuple[int, pd.DataFrame, pd.DataFrame]],
    candidate_predictions: dict[tuple[int, str], np.ndarray],
    policy_predictions: dict[int, pd.DataFrame],
) -> list[dict[str, object]]:
    """Bias and MAE in the top prior-rate quartile, candidate vs selected."""
    records = []
    for year, _, validation in folds:
        quartile = pd.qcut(
            validation[f"{stat}_RATE_prior"].rank(method="first"),
            4,
            labels=False,
        )
        mask = np.asarray(quartile == 3)
        actual = validation.loc[mask, f"{stat}_RATE_target"].to_numpy(dtype=float)
        weights = validation.loc[mask, "MIN_target"].to_numpy(dtype=float)
        naive = validation.loc[mask, f"{stat}_RATE_prior"].to_numpy(dtype=float)
        for label, prediction in (
            ("tuned_candidate", candidate_predictions[(year, stat)]),
            ("selected_policy", policy_predictions[year][stat].to_numpy(dtype=float)),
        ):
            records.append(
                {
                    "Evaluation_Set": label,
                    "Validation_Year": year,
                    "Stat": stat,
                    "Prior_Rate_Quartile": "Q4",
                    "Rows": int(mask.sum()),
                    "Weighted_Bias": float(
                        np.average(prediction[mask] - actual, weights=weights)
                    ),
                    "Weighted_MAE": _weighted_mae(actual, prediction[mask], weights),
                    "Naive_MAE": _weighted_mae(actual, naive, weights),
                }
            )
    return records


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def run_phase_four(pairs: pd.DataFrame) -> dict[str, object]:
    """Tune, decide, validate, and persist the Phase 4 rate models."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    folds = rolling_origin_folds(pairs)

    # 1. Tune formulation and alpha per stat.
    grid, selections = tune_rate_models(folds)
    grid.to_csv(ALPHA_GRID_METRICS_PATH, index=False)

    # 2. Decide the policy per stat.
    decisions: dict[str, dict[str, object]] = {}
    bootstrap_records: list[dict[str, object]] = []
    fold_metric_records: list[dict[str, object]] = []
    candidate_predictions: dict[tuple[int, str], np.ndarray] = {}
    for stat_index, stat in enumerate(TARGET_STATS):
        outcome = decide_policy(stat_index, stat, selections[stat], folds)
        decisions[stat] = outcome["decision"]
        bootstrap_records.extend(outcome["bootstrap_records"])
        fold_metric_records.extend(outcome["metric_records"])
        for year, prediction in outcome["predictions"].items():
            candidate_predictions[(year, stat)] = prediction
    bootstrap_records.extend(
        compare_points_policies(folds, decisions, candidate_predictions)
    )

    # 3. Re-score the selected policy fold by fold through the shared path.
    policy_predictions: dict[int, pd.DataFrame] = {}
    percentage_records: list[dict[str, object]] = []
    consistency_records: list[dict[str, float]] = []
    for year, train, validation in folds:
        predictions = predict_rates(build_bundle(train, decisions), validation)
        policy_predictions[year] = predictions
        for stat in TARGET_STATS:
            policy = str(decisions[stat]["selected_policy"])
            fold_metric_records.append(
                {
                    "Evaluation_Set": "selected_policy",
                    "Validation_Year": year,
                    "Stat": stat,
                    "Formulation": policy,
                    "Alpha": (
                        decisions[stat]["tuned_alpha"]
                        if policy in MODEL_FORMULATIONS
                        else np.nan
                    ),
                    **_score_predictions(
                        train,
                        validation,
                        stat,
                        predictions[stat].to_numpy(dtype=float),
                    ),
                }
            )
        percentage_records.extend(
            _implied_percentage_records(year, validation, predictions)
        )
        consistency_records.append(
            _points_consistency_record(year, validation, predictions)
        )

    quartile_records: list[dict[str, object]] = []
    for stat in QUARTILE_CHECK_STATS:
        quartile_records.extend(
            _top_quartile_records(stat, folds, candidate_predictions, policy_predictions)
        )

    # 4. Refit on all transitions and persist.
    coefficients = []
    candidate_models: dict[str, Pipeline] = {}
    candidate_settings: dict[str, dict[str, object]] = {}
    for stat in TARGET_STATS:
        decision = decisions[stat]
        formulation = str(decision["tuned_formulation"])
        alpha = float(decision["tuned_alpha"])
        candidate_models[stat] = fit_stat_model(pairs, stat, alpha, formulation)
        candidate_settings[stat] = {
            "formulation": formulation,
            "alpha": alpha,
            "features": FEATURES_BY_STAT[stat],
        }
        coefficients.append(
            _coefficient_record(
                candidate_models[stat], stat, formulation, alpha, "full_data_candidate"
            )
        )

    bundle = build_bundle(pairs, decisions)
    bundle["candidate_models"] = candidate_models
    bundle["candidate_settings_by_stat"] = candidate_settings
    bundle["fold_validation_years"] = [year for year, _, _ in folds]
    bundle["alpha_grid"] = ALPHA_GRID
    bundle["ranking_minutes_threshold"] = RANKING_MINUTES_THRESHOLD

    fold_metrics = pd.DataFrame(fold_metric_records)
    policy_decisions = decisions_to_frame(decisions)
    percentage_diagnostics = pd.DataFrame(percentage_records)
    bootstrap_metrics = pd.DataFrame(bootstrap_records)
    bootstrap_metrics["Validation_Year"] = bootstrap_metrics[
        "Validation_Year"
    ].astype("Int64")
    coefficient_table = pd.DataFrame(coefficients)

    last_year = folds[-1][0]
    fold_metrics.to_csv(FOLD_METRICS_PATH, index=False)
    fold_metrics.loc[
        (fold_metrics["Evaluation_Set"] == "selected_policy")
        & (fold_metrics["Validation_Year"] == last_year)
    ].to_csv(VALIDATION_METRICS_PATH, index=False)
    bootstrap_metrics.to_csv(BOOTSTRAP_METRICS_PATH, index=False)
    coefficient_table.to_csv(COEFFICIENTS_PATH, index=False)
    policy_decisions.to_csv(POLICY_DECISIONS_PATH, index=False)
    percentage_diagnostics.to_csv(PERCENTAGE_DIAGNOSTICS_PATH, index=False)
    joblib.dump(bundle, MODEL_BUNDLE_PATH)

    metadata = {
        "target_stats": TARGET_STATS,
        "features_by_stat": FEATURES_BY_STAT,
        "settings_by_stat": bundle["settings_by_stat"],
        "candidate_settings_by_stat": candidate_settings,
        "tuning_selection": selections,
        "policy_decisions": decisions,
        "selection_rule": (
            "accept the tuned model if improvement over naive is positive in "
            "every fold and the pooled 95% bootstrap CI excludes zero"
        ),
        "selection_note": (
            "tuning, selection and scoring share the same folds; reported "
            "selected-policy metrics are optimistic"
        ),
        "validation_folds": [
            {
                "validation_year": year,
                "training_rows": len(train),
                "validation_rows": len(validation),
            }
            for year, train, validation in folds
        ],
        "full_training_rows": len(pairs),
        "fit_transition_years": bundle["fit_transition_years"],
        "alpha_grid": ALPHA_GRID,
        "weight_normalization": WEIGHT_NORMALIZATION,
        "ranking_minutes_threshold": RANKING_MINUTES_THRESHOLD,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ft_prior_shrinkage": FT_PRIOR_SHRINKAGE,
        "prior_shrinkage_reference_by_stat": bundle[
            "prior_shrinkage_reference_by_stat"
        ],
        "implied_percentages": percentage_records,
        "points_consistency": consistency_records,
        "top_prior_rate_quartile": quartile_records,
    }
    MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, default=_json_default)
    )

    return {
        "grid": grid,
        "selections": selections,
        "decisions": decisions,
        "policy_decisions": policy_decisions,
        "fold_metrics": fold_metrics,
        "bootstrap_metrics": bootstrap_metrics,
        "percentage_diagnostics": percentage_diagnostics,
        "points_consistency": pd.DataFrame(consistency_records),
        "quartile_metrics": pd.DataFrame(quartile_records),
        "coefficients": coefficient_table,
        "bundle": bundle,
    }


def print_summary(results: dict[str, object]) -> None:
    print("\nPolicy decisions:")
    print(results["policy_decisions"].to_string(index=False))

    fold_metrics = results["fold_metrics"]
    for evaluation_set in ("tuned_candidate", "selected_policy"):
        subset = fold_metrics.loc[fold_metrics["Evaluation_Set"] == evaluation_set]
        pivot = subset.pivot(
            index="Stat", columns="Validation_Year", values="Improvement_vs_Naive_pct"
        ).reindex(TARGET_STATS)
        print(f"\nImprovement vs naive (%), {evaluation_set}:")
        print(pivot.round(2).to_string())

    print("\nImplied FG% / FT% (policy vs naive vs actual):")
    print(results["percentage_diagnostics"].round(4).to_string(index=False))
    print("\nPTS vs component identity (selected policy):")
    print(results["points_consistency"].round(5).to_string(index=False))
    print("\nDirect vs component-derived PTS:")
    points = results["bootstrap_metrics"]
    print(
        points.loc[points["Compared_To"] == "component_derived"]
        .dropna(axis=1, how="all")
        .round(5)
        .to_string(index=False)
    )
    print("\nTop prior-rate quartile diagnostics:")
    print(results["quartile_metrics"].round(5).to_string(index=False))
    print("\nOwn-rate level coefficients:")
    print(results["coefficients"].round(4).to_string(index=False))
    print(f"\nPersisted model bundle: {MODEL_BUNDLE_PATH}")
    print(f"Persisted metadata: {MODEL_METADATA_PATH}")


if __name__ == "__main__":
    print_summary(run_phase_four(load_training_pairs()))