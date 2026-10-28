from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_STATS_PATH = PROJECT_ROOT / "data" / "raw" / "historical_stats.csv"
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "training_data.csv"
TRAINING_PAIRS_PATH = PROJECT_ROOT / "data" / "processed" / "training_pairs.csv"

COUNTING_STATS = [
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
RATE_COLUMNS = {stat: f"{stat}_RATE" for stat in COUNTING_STATS}
MINIMUM_SEASON_MINUTES = 200


def prepare_training_data(
	input_path: Path = HISTORICAL_STATS_PATH,
	output_path: Path = TRAINING_DATA_PATH,
) -> pd.DataFrame:
	"""Add per-minute rate columns to historical stats and save them."""
	historical_stats = pd.read_csv(input_path)
	required_columns = {"MIN", "SEASON", *COUNTING_STATS}
	missing_columns = sorted(required_columns - set(historical_stats.columns))
	if missing_columns:
		raise ValueError(f"Historical stats are missing required columns: {missing_columns}")

	if (historical_stats["MIN"] <= 0).any():
		raise ValueError("Historical stats contain rows with non-positive MIN; per-minute rates are undefined.")

	season_start_year = historical_stats["SEASON"].astype("string").str.extract(r"^(\d{4})-\d{2}$", expand=False)
	if season_start_year.isna().any():
		raise ValueError("SEASON values must use the YYYY-YY format.")
	historical_stats["SEASON"] = season_start_year.astype(int) + 1

	rates = historical_stats[COUNTING_STATS].div(historical_stats["MIN"], axis=0)
	historical_stats[list(RATE_COLUMNS.values())] = rates.rename(columns=RATE_COLUMNS)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	historical_stats.to_csv(output_path, index=False)
	return historical_stats


def build_training_pairs(
	training_data: pd.DataFrame,
	output_path: Path = TRAINING_PAIRS_PATH,
) -> pd.DataFrame:
	"""Pair each player's season rates with their following season's rates."""
	required_columns = {"PLAYER_ID", "PLAYER_NAME", "SEASON", "AGE", "MIN", *RATE_COLUMNS.values()}
	missing_columns = sorted(required_columns - set(training_data.columns))
	if missing_columns:
		raise ValueError(f"Training data are missing required columns: {missing_columns}")

	prior_columns = {"SEASON": "YEAR_CURRENT", "AGE": "AGE_prior", "MIN": "MIN_prior"}
	prior_columns.update({rate: f"{stat}_RATE_prior" for stat, rate in RATE_COLUMNS.items()})
	prior = training_data[
		["PLAYER_ID", "PLAYER_NAME", "SEASON", "AGE", "MIN", *RATE_COLUMNS.values()]
	].rename(
		columns={
			**prior_columns,
		}
	)
	prior["YEAR_NEXT"] = prior["YEAR_CURRENT"] + 1

	target_columns = {"SEASON": "YEAR_NEXT", "MIN": "MIN_target"}
	target_columns.update({rate: f"{stat}_RATE_target" for stat, rate in RATE_COLUMNS.items()})
	target = training_data[["PLAYER_ID", "SEASON", "MIN", *RATE_COLUMNS.values()]].rename(columns=target_columns)

	training_pairs = prior.merge(
		target,
		on=["PLAYER_ID", "YEAR_NEXT"],
		how="inner",
		validate="one_to_one",
	)
	training_pairs = training_pairs[
		(training_pairs["MIN_prior"] >= MINIMUM_SEASON_MINUTES)
		& (training_pairs["MIN_target"] >= MINIMUM_SEASON_MINUTES)
	].copy()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	training_pairs.to_csv(output_path, index=False)
	return training_pairs


if __name__ == "__main__":
	training_data = prepare_training_data()
	print(f"Saved {len(training_data)} rows to {TRAINING_DATA_PATH}")
	training_pairs = build_training_pairs(training_data)
	print(f"Saved {len(training_pairs)} rows to {TRAINING_PAIRS_PATH}")
