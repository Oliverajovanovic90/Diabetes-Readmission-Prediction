# serve.py
"""
FastAPI web service for Diabetes Readmission Prediction.
Exposes endpoints for health check and model prediction.
"""

import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import uvicorn
import os


# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = "model_xgb.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Trained model file not found. Please run train.py first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

expected_features = model.feature_names_in_

# -----------------------------
# Define Input Schema
# -----------------------------
class Patient(BaseModel):
    race: Literal["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"]
    gender: Literal["Male", "Female", "Unknown"]
    age: Literal[
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ]
    admission_type_id: int = Field(..., ge=1)
    discharge_disposition_id: int = Field(..., ge=1)
    admission_source_id: int = Field(..., ge=1)
    time_in_hospital: int = Field(..., ge=1)
    num_lab_procedures: int = Field(..., ge=0)
    num_procedures: int = Field(..., ge=0)
    num_medications: int = Field(..., ge=0)
    number_outpatient: int = Field(..., ge=0)
    number_emergency: int = Field(..., ge=0)
    number_inpatient: int = Field(..., ge=0)
    number_diagnoses: int = Field(..., ge=1)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"]
    A1Cresult: Literal["None", "Norm", ">7", ">8"]
    metformin: Literal["Steady", "Up", "Down", "No"]
    insulin: Literal["Steady", "Up", "Down", "No"]
    change: Literal["Ch", "No"]
    diabetesMed: Literal["Yes", "No"]


# -----------------------------
# Define Response Schema
# -----------------------------
class PredictResponse(BaseModel):
    readmission_probability: float
    readmitted: bool


# -----------------------------
# Initialize FastAPI App
# -----------------------------
app = FastAPI(title="Diabetes Readmission Prediction API", version="1.0")


# -----------------------------
# Health Check Endpoint
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "message": "Model is ready to predict."}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(patient: Patient):
    """Return readmission prediction for one patient record."""

    input_dict = patient.model_dump()

    # Fill missing expected columns
    for col in expected_features:
        if col not in input_dict:
            input_dict[col] = "Unknown"

    # Convert to DataFrame
    X_new = pd.DataFrame([[input_dict[col] for col in expected_features]], columns=expected_features)

    # Predict
    y_pred = model.predict(X_new)
    y_prob = model.predict_proba(X_new)[0, 1]

    return PredictResponse(
        readmission_probability=round(float(y_prob), 3),
        readmitted=bool(y_prob >= 0.5)
    )


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)