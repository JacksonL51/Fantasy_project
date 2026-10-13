from src.ingest import fetch_current_rosters, fetch_historical_stats


if __name__ == "__main__":
    print("Starting NBA fantasy data ingestion...")
    roster_df = fetch_current_rosters()
    fetch_historical_stats(roster_player_ids=roster_df["PLAYER_ID"].unique())
    print("Data ingestion complete.")
