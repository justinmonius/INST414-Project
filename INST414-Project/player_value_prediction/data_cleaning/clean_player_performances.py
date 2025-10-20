import pandas as pd
import os

# --- Step 1: Define full paths ---
raw_path = r"C:\Users\justi\OneDrive\Documents\UMD\Senior\INST414\INST414-Project\INST414-Project\data\raw\player_performances.csv"
output_dir = r"C:\Users\justi\OneDrive\Documents\UMD\Senior\INST414\INST414-Project\INST414-Project\data\processed"
output_path = os.path.join(output_dir, "cleaned_player_performances.csv")

# --- Step 2: Load CSV ---
perform = pd.read_csv(raw_path)

# --- Step 3: Standardize column names ---
perform.columns = perform.columns.str.lower().str.strip()

# --- Step 4: Keep only desired columns ---
keep_cols = [
    "player_id",
    "nb_in_group",
    "nb_on_pitch",
    "goals",
    "assists",
    "minutes_played",
    "goals_conceded",
    "clean_sheets"
]
perform = perform[[c for c in keep_cols if c in perform.columns]]

# --- Step 5: Convert all numeric columns (except player_id) to numeric ---
numeric_cols = [c for c in perform.columns if c != "player_id"]
perform[numeric_cols] = perform[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# --- Step 6: Aggregate (sum) by player_id so each player has one row ---
perform = perform.groupby("player_id", as_index=False)[numeric_cols].sum()

# --- Step 7: Ensure output directory exists ---
os.makedirs(output_dir, exist_ok=True)

# --- Step 8: Save cleaned file ---
perform.to_csv(output_path, index=False)

print(f"✅ Cleaned and aggregated player_performances.csv saved to:\n{output_path}")
print(f"Rows kept (unique players): {len(perform)}")
