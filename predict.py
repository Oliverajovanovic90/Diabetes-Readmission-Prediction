"""
Predict hospital readmission risk for diabetic patients
- Local mode: load model_xgb.pkl and predict
- API mode: send request to deployed FastAPI on AWS Elastic Beanstalk
"""

import pickle
import pandas as pd
import argparse
import json
import os
import requests

API_URL = "http://diabetes-readmission-env.eba-p33i43zn.us-east-2.elasticbeanstalk.com/predict"


# -----------------------------
# Load the Trained Model (Local)
# -----------------------------
def load_model(model_path: str = "model_xgb.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# -----------------------------
# Local Prediction
# -----------------------------
def predict_local(model, sample_input: dict):
    expected_features = model.feature_names_in_

    for col in expected_features:
        if col not in sample_input:
            sample_input[col] = "Unknown"

    X_new = pd.DataFrame([[sample_input[col] for col in expected_features]], columns=expected_features)

    y_pred = model.predict(X_new)
    y_prob = model.predict_proba(X_new)[0, 1]

    return {
        "predicted_class": int(y_pred[0]),
        "readmission_probability": round(float(y_prob), 3)
    }


# -----------------------------
# API Prediction
# -----------------------------
def predict_api(sample_input: dict):
    response = requests.post(API_URL, json=sample_input)
    if response.status_code != 200:
        raise RuntimeError(f"API Error: {response.status_code} - {response.text}")
    return response.json()


# -----------------------------
# Default Input
# -----------------------------
default_input = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 4,
    "num_lab_procedures": 41,
    "num_procedures": 0,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 8,
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "metformin": "Steady",
    "insulin": "Steady",
    "change": "Ch",
    "diabetesMed": "Yes"
}


# -----------------------------
# Main Entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict diabetes readmission probability.")
    parser.add_argument("--input", type=str, help="Path to JSON file with patient data.")
    parser.add_argument("--api", action="store_true", help="Send request to deployed API instead of local model")
    args = parser.parse_args()

    # Load input JSON if provided
    if args.input:
        with open(args.input, "r") as f:
            sample_input = json.load(f)
    else:
        sample_input = default_input

    # API mode
    if args.api:
        print("➡ Sending request to deployed API...")
        result = predict_api(sample_input)
        print(result)
        return

    # Local mode
    model = load_model()
    result = predict_local(model, sample_input)
    print("Predicted Class:", result["predicted_class"])
    print(f"Probability of Readmission: {result['readmission_probability']:.3f}")


if __name__ == "__main__":
    main()