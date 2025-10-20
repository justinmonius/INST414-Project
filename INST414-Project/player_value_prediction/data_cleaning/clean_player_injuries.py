import pandas as pd
import os

#paths
raw_path = r"C:\Users\justi\OneDrive\Documents\UMD\Senior\INST414\INST414-Project\INST414-Project\data\raw\player_injuries.csv"
output_dir = r"C:\Users\justi\OneDrive\Documents\UMD\Senior\INST414\INST414-Project\INST414-Project\data\processed"
output_path = os.path.join(output_dir, "cleaned_player_injuries.csv")

# load csv
injuries = pd.read_csv(raw_path)


injuries.columns = injuries.columns.str.lower().str.strip()

#keep columns
keep_cols = ["player_id", "days_missed"]
injuries = injuries[[c for c in keep_cols if c in injuries.columns]]

#convert days_missed to numeric and filter
injuries["days_missed"] = pd.to_numeric(injuries["days_missed"], errors="coerce").fillna(0)
injuries = injuries[injuries["days_missed"] > 0]

# aggregate by player_id
injuries = injuries.groupby("player_id", as_index=False)["days_missed"].sum()

# ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# save cleaned file
injuries.to_csv(output_path, index=False)

print(f"✅ Cleaned and aggregated player_injuries.csv saved to:\n{output_path}")
print(f"Rows kept (unique players): {len(injuries)}")
