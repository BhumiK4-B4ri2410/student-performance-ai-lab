from io import BytesIO
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Student Performance ML Trainer")

model = None
feature_columns = []
problem_type = None
target_column = None
last_metrics = {}
last_dataset = None


class PredictionData(BaseModel):
    values: dict


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global last_dataset

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file.")

    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise HTTPException(400, f"Could not read CSV: {exc}")

    if df.empty or len(df.columns) < 2:
        raise HTTPException(400, "CSV needs data and at least two columns.")

    last_dataset = df
    preview = df.head(5).astype(object).where(pd.notna(df.head(5)), None)

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": preview.to_dict("records")
    }


@app.post("/train")
def train(
    target: str = Form(...),
    task: str = Form(...),
    features: str = Form(...),
    test_size: float = Form(0.2)
):
    global model, feature_columns, problem_type, target_column, last_metrics

    if last_dataset is None:
        raise HTTPException(400, "Upload a CSV file first.")
    if target not in last_dataset.columns:
        raise HTTPException(400, "Target column not found.")
    if task not in ("regression", "classification"):
        raise HTTPException(400, "Choose regression or classification.")
    if not 0.1 <= test_size <= 0.4:
        raise HTTPException(400, "Test size must be between 0.1 and 0.4.")

    selected = list(dict.fromkeys(
        col.strip() for col in features.split(",") if col.strip()
    ))
    if not selected:
        raise HTTPException(400, "Select at least one feature.")
    if target in selected:
        raise HTTPException(400, "Target cannot also be a feature.")
    if any(col not in last_dataset.columns for col in selected):
        raise HTTPException(400, "Invalid feature column.")

    data = last_dataset[selected + [target]].dropna(subset=[target])
    if len(data) < 5:
        raise HTTPException(400, "At least 5 valid rows are required.")

    X, y = data[selected], data[target]
    if task == "classification":
        if y.nunique() < 2 or y.value_counts().min() < 2:
            raise HTTPException(
                400, "Each class needs at least two rows for training."
            )

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in selected if c not in numeric_cols]

    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ])

    estimator = (
        RandomForestRegressor(n_estimators=100, random_state=42)
        if task == "regression"
        else RandomForestClassifier(n_estimators=100, random_state=42)
    )
    new_model = Pipeline([
        ("preprocessing", preprocessing),
        ("model", estimator)
    ])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if task == "classification" else None
        )
        new_model.fit(X_train, y_train)
        predictions = new_model.predict(X_test)
    except Exception as exc:
        raise HTTPException(400, f"Training failed: {exc}")

    if task == "regression":
        mse = mean_squared_error(y_test, predictions)
        metrics = {
            "MAE": round(float(mean_absolute_error(y_test, predictions)), 4),
            "RMSE": round(float(np.sqrt(mse)), 4),
            "R2": round(float(r2_score(y_test, predictions)), 4)
        }
    else:
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision_weighted": round(float(precision_score(
                y_test, predictions, average="weighted", zero_division=0
            )), 4),
            "recall_weighted": round(float(recall_score(
                y_test, predictions, average="weighted", zero_division=0
            )), 4),
            "f1_weighted": round(float(f1_score(
                y_test, predictions, average="weighted", zero_division=0
            )), 4)
        }

    model = new_model
    feature_columns = selected
    problem_type = task
    target_column = target
    last_metrics = {
        "task": task, "train_rows": len(X_train),
        "test_rows": len(X_test), **metrics
    }

    with open(BASE_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    return {
        "message": "Model trained successfully.",
        "target": target_column,
        "features": feature_columns,
        "metrics": last_metrics,
        "sample_prediction": str(predictions[0]),
        "sample_actual": str(y_test.iloc[0])
    }


@app.post("/predict")
def predict(data: PredictionData):
    if model is None:
        raise HTTPException(400, "Upload a CSV and train a model first.")

    missing = [c for c in feature_columns if c not in data.values]
    if missing:
        raise HTTPException(400, f"Missing feature values: {missing}")

    try:
        row = pd.DataFrame([{c: data.values[c] for c in feature_columns}])
        result = model.predict(row)[0]
        if isinstance(result, np.generic):
            result = result.item()
        return {
            "target": target_column,
            "prediction": result,
            "task": problem_type
        }
    except Exception as exc:
        raise HTTPException(400, f"Prediction failed: {exc}")
