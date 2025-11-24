# Task 3 – End-to-End California Housing ML Prediction System

This project is an end-to-end machine learning system built as part of the AI internship.  
It trains a regression model on the California housing dataset and exposes predictions through a FastAPI backend and a simple HTML/JavaScript frontend.

---

## Project Structure

```text
Task3-ML-Deployment/
  backend/
    app.py            # FastAPI app (API endpoints)
    preprocess.py     # Data loading, preprocessing, model training
    predict.py        # Model loading and prediction helpers
    visualize.py      # Creates Actual vs Predicted plot
  frontend/
    index.html        # Simple web UI for uploading CSV and viewing results
  models/
    model.pkl         # Trained model pipeline (created by preprocess.py)
  datasets/
    housing.csv       # California housing dataset
  .venv/              # Python virtual environment (local)
  README.md
