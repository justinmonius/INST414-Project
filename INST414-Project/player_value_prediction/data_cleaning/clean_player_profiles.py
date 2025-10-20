import pandas as pd
import numpy as np
import os
import re

raw_path = r"C:\Users\justi\OneDrive\Documents\UMD\Senior\INST414\INST414-Project\INST414-Project\data\raw\player_profiles.csv"
output_dir = r"C:\Users\justi\OneDrive\Documents\UMD\Senior\INST414\INST414-Project\INST414-Project\data\processed"
output_path = os.path.join(output_dir, "cleaned_player_profiles.csv")


profiles = pd.read_csv(raw_path)


profiles.columns = profiles.columns.str.lower().str.strip()


keep_cols = [
    "player_id",
    "player_name",
    "date_of_birth",
    "country_of_birth",
    "height",
    "position",
    "current_club_name"
]
profiles = profiles[[c for c in keep_cols if c in profiles.columns]]


def clean_name(name):
    if pd.isna(name):
        return np.nan
   
    name = re.sub(r"\(.*?\)", "", name).strip()
    
    parts = name.split()
    return " ".join(parts[:2])

profiles["player_name"] = profiles["player_name"].apply(clean_name)


def simplify_position(pos):
    if pd.isna(pos):
        return "Unknown"
    pos = pos.lower()
    if "defend" in pos:
        return "Defender"
    elif "midfield" in pos:
        return "Midfield"
    elif any(word in pos for word in ["attack", "forward", "winger", "striker"]):
        return "Attack"
    elif "keeper" in pos or "goalkeeper" in pos:
        return "Goalkeeper"
    else:
        return "Other"

profiles["position"] = profiles["position"].apply(simplify_position)


if pd.api.types.is_numeric_dtype(profiles["height"]):
    profiles["height"] = profiles["height"].fillna(profiles["height"].median())

profiles = profiles.dropna(subset=["player_name", "position"])


profiles = profiles[~profiles["current_club_name"].str.strip().str.lower().eq("retired")]


profiles = profiles.drop_duplicates(subset=["player_id"], keep="first")


os.makedirs(output_dir, exist_ok=True)


profiles.to_csv(output_path, index=False)

print(f"✅ Cleaned player_profiles.csv saved to:\n{output_path}")
print(f"Rows kept: {len(profiles)}")
print("\nUnique positions after simplification:")
print(profiles['position'].value_counts())
