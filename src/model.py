from pathlib import Path

import numpy as np
import pandas as pd


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
    stat: [f"{stat}_RATE_prior", "AGE_prior", "MIN_prior"]
    for stat in TARGET_STATS
}
RIDGE_ALPHA = 10.0
VALIDATION_CUTOFF_YEAR = 2025

TRAINING_PAIRS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "training_pairs.csv"
)
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

    print(f"Valid transitions: {len(pairs)}")
    print("Transitions per YEAR_NEXT:")
    print(pairs["YEAR_NEXT"].value_counts().sort_index().to_string())
    return pairs


def temporal_train_validation_split(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically, reserving the transition before the cutoff year."""
    train = pairs.loc[pairs["YEAR_NEXT"] < VALIDATION_CUTOFF_YEAR].copy()
    validation = pairs.loc[
        (pairs["YEAR_CURRENT"] == VALIDATION_CUTOFF_YEAR - 1)
        & (pairs["YEAR_NEXT"] == VALIDATION_CUTOFF_YEAR)
    ].copy()

    if train.empty:
        raise ValueError("Temporal training split is empty.")
    if validation.empty:
        raise ValueError(
            "Temporal validation split is empty for transition "
            f"{VALIDATION_CUTOFF_YEAR - 1} -> {VALIDATION_CUTOFF_YEAR}."
        )

    print(f"Training transitions: {len(train)}")
    print(f"Validation transitions: {len(validation)}")
    if len(validation) < 100:
        print(
            "Warning: validation has fewer than 100 rows; "
            "metrics may be noisy."
        )
    return train, validation


if __name__ == "__main__":
    training_pairs = load_training_pairs()
    temporal_train_validation_split(training_pairs)