from pathlib import Path

import pandas as pd
from nba_api.stats.library.parameters import Season


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_STATS_PATH = PROJECT_ROOT / "data" / "raw" / "historical_stats.csv"
CURRENT_ROSTERS_PATH = PROJECT_ROOT / "data" / "raw" / "current_rosters.csv"
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
	"""Resolve traded-player totals, keep completed seasons, and add rates."""
	historical_stats = pd.read_csv(input_path)
	required_columns = {
		"PLAYER_ID",
		"TEAM_ABBREVIATION",
		"MIN",
		"SEASON",
		*COUNTING_STATS,
	}
	missing_columns = sorted(required_columns - set(historical_stats.columns))
	if missing_columns:
		raise ValueError(f"Historical stats are missing required columns: {missing_columns}")

	historical_stats["TEAM_ABBREVIATION"] = (
		historical_stats["TEAM_ABBREVIATION"].astype("string").str.strip().str.upper()
	)
	season_parts = historical_stats["SEASON"].astype("string").str.extract(
		r"^(\d{4})-(\d{2})$"
	)
	if season_parts.isna().any().any():
		raise ValueError("SEASON values must use the YYYY-YY format.")
	season_start_year = season_parts[0].astype(int)
	season_end_year = (season_start_year // 100) * 100 + season_parts[1].astype(int)
	season_end_year += season_end_year.lt(season_start_year).astype(int) * 100
	last_completed_end_year = int(Season.previous_season[:4]) + 1
	historical_stats = historical_stats.loc[
		season_end_year <= last_completed_end_year
	].copy()
	historical_stats["SEASON_YEAR"] = season_end_year.loc[historical_stats.index]

	player_season = ["PLAYER_ID", "SEASON_YEAR"]
	tot_rows = historical_stats["TEAM_ABBREVIATION"].eq("TOT")
	tot_player_seasons = set(
		map(tuple, historical_stats.loc[tot_rows, player_season].to_numpy())
	)
	has_tot = pd.MultiIndex.from_frame(historical_stats[player_season]).isin(
		tot_player_seasons
	)
	historical_stats = historical_stats.loc[tot_rows | ~has_tot].copy()
	duplicate_player_seasons = historical_stats.duplicated(player_season, keep=False)
	if duplicate_player_seasons.any():
		examples = (
			historical_stats.loc[
				duplicate_player_seasons,
				player_season + ["TEAM_ABBREVIATION"],
			]
			.head(10)
			.to_dict("records")
		)
		raise ValueError(
			"Multiple non-TOT rows remain for a player-season after trade "
			f"resolution. Expected one team entry; examples: {examples}"
		)

	if CURRENT_ROSTERS_PATH.exists():
		current_rosters = pd.read_csv(CURRENT_ROSTERS_PATH)
		if "PLAYER_ID" not in current_rosters.columns:
			raise ValueError(
				f"{CURRENT_ROSTERS_PATH} is missing the PLAYER_ID column."
			)
		current_player_ids = pd.to_numeric(
			current_rosters["PLAYER_ID"], errors="coerce"
		).dropna()
		historical_stats["PLAYER_ID"] = pd.to_numeric(
			historical_stats["PLAYER_ID"], errors="coerce"
		)
		historical_stats = historical_stats.loc[
			historical_stats["PLAYER_ID"].isin(current_player_ids)
		].copy()

	if historical_stats.empty:
		raise ValueError(
			"No historical player stats remain through the last completed season."
		)
	if (historical_stats["MIN"] <= 0).any():
		raise ValueError("Historical stats contain rows with non-positive MIN; per-minute rates are undefined.")

	historical_stats = historical_stats.sort_values(
		["PLAYER_ID", "SEASON_YEAR"]
	).reset_index(drop=True)

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
	required_columns = {
		"PLAYER_ID",
		"PLAYER_NAME",
		"SEASON_YEAR",
		"AGE",
		"MIN",
		*RATE_COLUMNS.values(),
	}
	missing_columns = sorted(required_columns - set(training_data.columns))
	if missing_columns:
		raise ValueError(f"Training data are missing required columns: {missing_columns}")

	if training_data.duplicated(["PLAYER_ID", "SEASON_YEAR"]).any():
		raise ValueError("Training data contain duplicate PLAYER_ID/SEASON_YEAR rows.")

	prior_columns = {
		"SEASON_YEAR": "YEAR_CURRENT",
		"AGE": "AGE_prior",
		"MIN": "MIN_prior",
	}
	prior_columns.update({rate: f"{stat}_RATE_prior" for stat, rate in RATE_COLUMNS.items()})
	prior = training_data[
		[
			"PLAYER_ID",
			"PLAYER_NAME",
			"SEASON_YEAR",
			"AGE",
			"MIN",
			*RATE_COLUMNS.values(),
		]
	].rename(
		columns={
			**prior_columns,
		}
	)
	prior["JOIN_YEAR"] = prior["YEAR_CURRENT"] + 1

	target_columns = {"SEASON_YEAR": "JOIN_YEAR", "MIN": "MIN_target"}
	target_columns.update({rate: f"{stat}_RATE_target" for stat, rate in RATE_COLUMNS.items()})
	target = training_data[
		["PLAYER_ID", "SEASON_YEAR", "MIN", *RATE_COLUMNS.values()]
	].rename(columns=target_columns)

	training_pairs = prior.merge(
		target,
		on=["PLAYER_ID", "JOIN_YEAR"],
		how="inner",
		validate="one_to_one",
	)
	training_pairs["YEAR_NEXT"] = training_pairs["JOIN_YEAR"]
	training_pairs = training_pairs[
		(training_pairs["MIN_prior"] >= MINIMUM_SEASON_MINUTES)
		& (training_pairs["MIN_target"] >= MINIMUM_SEASON_MINUTES)
	].copy()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	training_pairs.to_csv(output_path, index=False)
	return training_pairs


if __name__ == "__main__":
	training_data = prepare_training_data()
	print(
		f"Saved {len(training_data)} player-seasons through "
		f"{training_data['SEASON'].max()} to {TRAINING_DATA_PATH}"
	)
	training_pairs = build_training_pairs(training_data)
	print(f"Saved {len(training_pairs)} rows to {TRAINING_PAIRS_PATH}")
