from pathlib import Path
from typing import List, Tuple
import joblib
import pandas as pd
from .preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"


def load_trained_model_with_payload(path=DEFAULT_MODEL_PATH):
    payload = joblib.load(path)
    return payload['model'], payload


def prepare_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order the required columns for prediction."""
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    return df[required].copy()


def predict_from_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[float]]:
    """Run predictions on a dataframe that includes feature columns."""
    model, _ = load_trained_model()
    X = prepare_features_for_inference(df)
    preds = model.predict(X)
    preds = preds.tolist()

    result_df = df.copy()
    result_df["predicted_median_house_value"] = preds
    return result_df, preds



def predict_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    result_df, _ = predict_from_dataframe(df)
    return result_df


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python backend/predict.py <path_to_csv>")
        raise SystemExit(1)

    csv_path = Path(sys.argv[1])
    result_df = predict_from_csv(csv_path)
    print(result_df.head())


if __name__ == "__main__":
    main()
