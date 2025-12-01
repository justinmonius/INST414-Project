"""
train.py

Train regression models to predict soccer player market value.

This version:
- Uses engineered numeric encodings for current_club_name and country_of_birth
  via mean market value (target-like encoding).
- Reduces categorical cardinality: only 'position' is one-hot encoded.
- Includes performance and usage features: nb_in_group, nb_on_pitch, goal_contrib, etc.
- Trains Baseline, Linear Regression, Ridge, Random Forest, and Gradient Boosting.
- Evaluates with R², RMSE, MAE (in euros).
- Saves models to models/ and metrics to results/metrics.json.
"""

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib


# -------- Paths -------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "merged_players_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# -------- Metrics helper -------- #
def compute_metrics(y_true_log, y_pred_log):
    """
    Compute R², RMSE, MAE in original euro scale.
    Inputs are log1p(market_value_eur).
    Returns plain Python floats so they can be JSON-serialized.
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {"r2": r2, "rmse": rmse, "mae": mae}


# -------- Data loading & preparation -------- #
def load_and_prepare_data():
    """
    Load merged dataset, engineer features, and return X, y_log, feature_cols.
    Applies:
    - Filtering to positive market values
    - Age calculation (if needed)
    - goal_contrib = goals + assists
    - target-like encodings for current_club_name and country_of_birth:
      mean market_value_eur by club and by country
    """
    df = pd.read_csv(DATA_PATH)

    # Filter to players with a positive market value
    df = df[df["market_value_eur"] > 0]

    # Standardize text columns
    for col in ["current_club_name", "position", "country_of_birth"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Compute age if needed
    if "age" not in df.columns and "date_of_birth" in df.columns:
        df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
        df["age"] = datetime.now().year - df["date_of_birth"].dt.year

    # Restrict to reasonable ages
    if "age" in df.columns:
        df = df[df["age"].between(16, 42)]

    # Compute goal contributions
    if "goals" in df.columns and "assists" in df.columns:
        df["goal_contrib"] = df["goals"] + df["assists"]

    # ---- Target-like encodings: mean market value by club and country ---- #
    if "current_club_name" in df.columns:
        club_means = (
            df.groupby("current_club_name")["market_value_eur"]
            .mean()
            .rename("club_mean_value")
        )
        df = df.merge(
            club_means,
            how="left",
            left_on="current_club_name",
            right_index=True,
        )
    else:
        df["club_mean_value"] = np.nan

    if "country_of_birth" in df.columns:
        country_means = (
            df.groupby("country_of_birth")["market_value_eur"]
            .mean()
            .rename("country_mean_value")
        )
        df = df.merge(
            country_means,
            how="left",
            left_on="country_of_birth",
            right_index=True,
        )
    else:
        df["country_mean_value"] = np.nan

    # If any encodings are missing, fill with global mean
    global_mean = df["market_value_eur"].mean()
    df["club_mean_value"] = df["club_mean_value"].fillna(global_mean)
    df["country_mean_value"] = df["country_mean_value"].fillna(global_mean)

    # -------- Feature set -------- #
    candidate_features = [
        "age",
        "position",          # categorical (low cardinality)
        "goals",
        "assists",
        "goal_contrib",
        "minutes_played",
        "nb_in_group",
        "nb_on_pitch",
        "clean_sheets",
        "days_missed",
        "height",
        "club_mean_value",      # numeric encoding
        "country_mean_value",   # numeric encoding
    ]

    feature_cols = [c for c in candidate_features if c in df.columns]
    target_col = "market_value_eur"

    # Drop rows with missing values in features or target
    df = df.dropna(subset=[target_col] + feature_cols)

    X = df[feature_cols].copy()
    y = df[target_col].astype(float)
    y_log = np.log1p(y)

    return X, y_log, feature_cols


# -------- Preprocessor -------- #
def build_preprocessor(feature_cols):
    """
    Build a ColumnTransformer:
    - numeric features -> StandardScaler
    - position (only) -> OneHotEncoder
    """
    categorical_features = [c for c in feature_cols if c == "position"]
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


# -------- Main training pipeline -------- #
def main():
    print(f"Loading data from: {DATA_PATH}")
    X, y_log, feature_cols = load_and_prepare_data()
    print(f"Dataset size: {X.shape[0]} rows, {X.shape[1]} features")

    preprocessor, numeric_features, categorical_features = build_preprocessor(feature_cols)
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # 70 / 15 / 15 split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_log, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1765, random_state=42
    )
    print(f"Splits -> train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ----- Baseline: predict mean of train ----- #
    baseline_mean = float(y_train.mean())

    baseline_train_log = np.full_like(y_train, baseline_mean)
    baseline_val_log = np.full_like(y_val, baseline_mean)
    baseline_test_log = np.full_like(y_test, baseline_mean)

    baseline_metrics_train = compute_metrics(y_train, baseline_train_log)
    baseline_metrics_val = compute_metrics(y_val, baseline_val_log)
    baseline_metrics_test = compute_metrics(y_test, baseline_test_log)

    print("\nBaseline performance:")
    print("Train:", baseline_metrics_train)
    print("Val  :", baseline_metrics_val)
    print("Test :", baseline_metrics_test)

    all_metrics = {
        "baseline": {
            "train": baseline_metrics_train,
            "val": baseline_metrics_val,
            "test": baseline_metrics_test,
        }
    }

    # ----- Models with tuned-ish hyperparameters ----- #
    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=800,
            max_depth=18,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
        ),
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        # Train metrics
        y_train_pred_log = pipeline.predict(X_train)
        metrics_train = compute_metrics(y_train, y_train_pred_log)

        # Validation metrics
        y_val_pred_log = pipeline.predict(X_val)
        metrics_val = compute_metrics(y_val, y_val_pred_log)

        # Test metrics
        y_test_pred_log = pipeline.predict(X_test)
        metrics_test = compute_metrics(y_test, y_test_pred_log)

        all_metrics[name] = {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        }

        print("Train:", metrics_train)
        print("Val  :", metrics_val)
        print("Test :", metrics_test)

        model_path = MODELS_DIR / f"{name}_log_target.joblib"
        joblib.dump(pipeline, model_path)
        print("Saved model to:", model_path)

    # Save all metrics
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("\nTraining complete.")
    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    main()
