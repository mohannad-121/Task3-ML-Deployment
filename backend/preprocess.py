from pathlib import Path
import sys
from typing import Tuple, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor



TARGET_COLUMN = "median_house_value"

NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

CATEGORICAL_FEATURES = ["ocean_proximity"]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the California housing CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X (features) and y (target)."""
    missing_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_model_pipeline() -> Pipeline:
    """Create a full preprocessing + model pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_and_save_model(
    csv_path: Path,
    model_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train the model on the dataset and save it to disk."""
    print(f"[INFO] Loading dataset from {csv_path}")
    df = load_dataset(csv_path)

    print("[INFO] Splitting features and target")
    X, y = split_features_target(df)

    print("[INFO] Train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("[INFO] Building pipeline")
    pipeline = build_model_pipeline()

    print("[INFO] Training model...")
    pipeline.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    r2 = r2_score(y_test, y_pred)

    metrics = {"rmse": float(rmse), "r2": float(r2)}
    print(f"[METRICS] RMSE = {rmse:.2f}, R2 = {r2:.3f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": pipeline,
        "feature_names": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "metrics": metrics,
    }

    print(f"[INFO] Saving trained model to {model_path}")
    joblib.dump(payload, model_path)

    return metrics


def main():
    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "datasets" / "housing.csv"
    default_model_path = project_root / "models" / "model.pkl"

  
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv

    metrics = train_and_save_model(csv_path, default_model_path)
    print("[DONE] Training complete.")
    print(metrics)


if __name__ == "__main__":
    main()
