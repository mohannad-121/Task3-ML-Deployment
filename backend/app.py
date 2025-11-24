# backend/app.py
from pydantic import BaseModel
from fastapi import Header

from pathlib import Path
import base64
import io

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .preprocess import TARGET_COLUMN
from .predict import predict_from_dataframe
from .visualize import create_prediction_plot
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHARTS_DIR = PROJECT_ROOT / "charts"

app = FastAPI(title="California Housing Prediction API")

# Allow frontend from any origin (simple for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "California Housing Prediction API is running."}


@app.post("/predict")

async def predict(file: UploadFile = File(...)):
    """
    Accept a CSV file, run preprocessing + prediction + visualization,
    return predictions and base64-encoded PNG chart.
    """
    
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    # If the CSV contains the target column, we can plot actual vs predicted
    if TARGET_COLUMN in df.columns:
        y_true = df[TARGET_COLUMN].values
        feature_df = df.drop(columns=[TARGET_COLUMN])
    else:
        y_true = None
        feature_df = df

    try:
        result_df, preds = predict_from_dataframe(feature_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    plot_b64 = None
    if y_true is not None:
        chart_path = CHARTS_DIR / "pred_vs_actual.png"
        create_prediction_plot(y_true, preds, chart_path)
        with open(chart_path, "rb") as f:
            img_bytes = f.read()
        plot_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # To keep response light, only send first 50 predictions
    preview = result_df.head(50).to_dict(orient="records")

    return JSONResponse(
        {
            "num_rows": int(len(result_df)),
            "preview": preview,
            "plot_image": plot_b64,
        }
        
    )
