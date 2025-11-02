# predict.py
"""
Predict hospital readmission risk for diabetic patients
using the trained XGBoost model (model_xgb.pkl).
"""

import pickle
import pandas as pd
import argparse
import json
import os


# -----------------------------
# Load the Trained Model
# -----------------------------
def load_model(model_path: str = "model_xgb.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# -----------------------------
# Make Prediction
# -----------------------------
def predict_single(model, sample_input: dict):
    """Make a prediction for one patient record."""

    # Get expected columns from model pipeline
    expected_features = model.feature_names_in_

    # Fill missing columns with 'Unknown'
    for col in expected_features:
        if col not in sample_input:
            sample_input[col] = "Unknown"

    # Convert to DataFrame with correct column order
    X_new = pd.DataFrame([[sample_input[col] for col in expected_features]], columns=expected_features)

    # Predict
    y_pred = model.predict(X_new)
    y_prob = model.predict_proba(X_new)[0, 1]

    return {
        "predicted_class": int(y_pred[0]),
        "readmission_probability": round(float(y_prob), 3)
    }


# -----------------------------
# Default Example Input
# -----------------------------
default_input = {
    'race': 'Caucasian',
    'gender': 'Female',
    'age': '[60-70)',
    'admission_type_id': 1,
    'discharge_disposition_id': 1,
    'admission_source_id': 7,
    'time_in_hospital': 4,
    'num_lab_procedures': 41,
    'num_procedures': 0,
    'num_medications': 15,
    'number_outpatient': 0,
    'number_emergency': 0,
    'number_inpatient': 0,
    'number_diagnoses': 8,
    'max_glu_serum': 'None',
    'A1Cresult': 'None',
    'metformin': 'Steady',
    'insulin': 'Steady',
    'change': 'Ch',
    'diabetesMed': 'Yes'
}


# -----------------------------
# Main Entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict diabetes readmission probability.")
    parser.add_argument("--input", type=str, help="Path to JSON file containing patient data.")
    args = parser.parse_args()

    # Load model
    model = load_model()

    # Load input data (if JSON provided)
    if args.input:
        with open(args.input, "r") as f:
            sample_input = json.load(f)
    else:
        sample_input = default_input

    # Run prediction
    result = predict_single(model, sample_input)

    # Output results
    print("Predicted Class:", result["predicted_class"])
    print(f"Probability of Readmission: {result['readmission_probability']:.3f}")


if __name__ == "__main__":
    main()