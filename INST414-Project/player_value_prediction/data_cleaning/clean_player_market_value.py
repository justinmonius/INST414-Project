import pandas as pd
import numpy as np
import os

# --- Step 1: Define full paths ---
raw_path = r"C:\Users\justi\OneDrive\Documents\UMD\Senior\INST414\INST414-Project\INST414-Project\data\raw\player_latest_market_value.csv"
output_dir = r"C:\Users\justi\OneDrive\Documents\UMD\Senior\INST414\INST414-Project\INST414-Project\data\processed"
output_path = os.path.join(output_dir, "cleaned_player_market_value.csv")

# --- Step 2: Load CSV ---
market = pd.read_csv(raw_path)

# --- Step 3: Standardize column names ---
market.columns = market.columns.str.lower().str.strip()

# --- Step 4: Rename columns for clarity ---
# Your file has: player_id | date_unix | value
market = market.rename(columns={"date_unix": "date", "value": "market_value_eur"})

# --- Step 5: Convert date and ensure numeric ---
market["date"] = pd.to_datetime(market["date"], errors="coerce")
market["market_value_eur"] = pd.to_numeric(market["market_value_eur"], errors="coerce")

# --- Step 6: Drop rows where value is 0 or missing ---
market = market[market["market_value_eur"].notna() & (market["market_value_eur"] > 0)]

# --- Step 7: Remove duplicates, keep latest per player ---
market = market.sort_values(["player_id", "date"], ascending=[True, False])
market = market.drop_duplicates(subset=["player_id"], keep="first")

# --- Step 8: Add log-transformed column for modeling ---
market["log_market_value"] = np.log1p(market["market_value_eur"])

# --- Step 9: Ensure output folder exists ---
os.makedirs(output_dir, exist_ok=True)

# --- Step 10: Save cleaned file ---
market.to_csv(output_path, index=False)

print(f"✅ Cleaned player_latest_market_value.csv saved to:\n{output_path}")
print(f"Rows kept: {len(market)}")
