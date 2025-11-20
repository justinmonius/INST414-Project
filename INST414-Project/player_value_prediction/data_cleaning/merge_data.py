import pandas as pd
import os


base_dir = r"C:\Users\justi\OneDrive\Documents\UMD\Senior\INST414\INST414-Project\INST414-Project\data\processed"
output_path = os.path.join(base_dir, "merged_players_dataset.csv")


profiles_path = os.path.join(base_dir, "cleaned_player_profiles.csv")
performances_path = os.path.join(base_dir, "cleaned_player_performances.csv")
market_path = os.path.join(base_dir, "cleaned_player_market_value.csv")
injuries_path = os.path.join(base_dir, "cleaned_player_injuries.csv")

print("📥 Loading cleaned datasets...")
profiles = pd.read_csv(profiles_path)
performances = pd.read_csv(performances_path)
market = pd.read_csv(market_path)
injuries = pd.read_csv(injuries_path)

##testing commit

print("🔗 Merging datasets...")
merged = (
    profiles
    .merge(performances, on="player_id", how="left")
    .merge(market, on="player_id", how="left")
    .merge(injuries, on="player_id", how="left")
)


merged["days_missed"] = merged["days_missed"].fillna(0)
merged["log_market_value"] = merged["log_market_value"].fillna(0)
merged["market_value_eur"] = merged["market_value_eur"].fillna(0)


before_rows = len(merged)
merged = merged[merged["market_value_eur"] > 0]
after_rows = len(merged)
print(f"🧹 Removed {before_rows - after_rows} rows with zero market value.")


if "goals" in merged.columns and "assists" in merged.columns:
    merged["goal_contrib"] = merged["goals"] + merged["assists"]


merged.columns = merged.columns.str.lower().str.replace(" ", "_")


merged.to_csv(output_path, index=False)

print(f"✅ Merged dataset saved successfully to:\n{output_path}")
print(f"Total rows: {len(merged)} | Total columns: {len(merged.columns)}")
