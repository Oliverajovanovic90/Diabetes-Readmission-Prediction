# Diabetes Readmission Prediction

Predict 30-day hospital readmission risk for diabetic patients using the **UCI / Kaggle diabetes dataset**.

---

## Project Overview
Hospital readmissions are costly and often preventable.  
This project builds a **machine-learning model** that predicts whether a diabetic patient will be readmitted within 30 days of discharge.  

The workflow demonstrates a complete **end-to-end ML pipeline** — from data cleaning and EDA to model development, evaluation, and deployment via a FastAPI web service containerized with Docker.

---

## Folder Structure
diabetes-readmission-prediction/
│
├── data/                     # raw + processed data (not committed)
├── plots/                    # saved charts & visualizations
│
├── EDA.ipynb                 # exploratory data analysis
├── Modeling.ipynb            # model training and evaluation
│
├── train.py                  # trains and saves the model artifact
├── predict.py                # local model inference test
├── serve.py                  # FastAPI web service (Docker entrypoint)
│
├── model_xgb.pkl             # trained XGBoost model
│
├── requirements.txt          # Python dependencies
├── Dockerfile                # container build recipe
├── pyproject.toml            # uv project configuration
├── uv.lock                   # environment lock file
└── README.md                 # project documentation

---

## Problem Statement
Predict whether a diabetic patient will be **readmitted within 30 days** after hospital discharge.  
This helps healthcare organizations identify **high-risk patients** and plan interventions (follow-ups, medication review, education).

---

## Dataset Summary
- Source: [UCI Machine Learning Repository / Kaggle – Diabetes 130-US hospitals (1999–2008)](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+readmission)
- Records: ~100 k encounters  
- Features: 50 (demographics, lab procedures, diagnoses, medications, etc.)
- Target: `readmitted` (`<30`, `>30`, or `NO`) — converted to **binary (1 = readmitted, 0 = no)**

---

## EDA Summary
- Significant class imbalance (`~55 k NO`, `~47 k readmitted`)
- Features such as `time_in_hospital`, `num_medications`, and `number_inpatient` correlated with readmission risk  
- Removed low-information or high-cardinality columns (`weight`, `payer_code`, `medical_specialty`, `diag_1–3`)

---

## Modeling Approach

### Base Model – Logistic Regression
- **Accuracy:** 0.6335  
- **F1-Score:** 0.5598  
- Served as interpretable baseline.

### Decision Tree
- **Best Depth:** 10  
- **F1:** 0.5568  

### Random Forest
- **Params:** `n_estimators=200`, `max_depth=20`  
- **F1:** 0.5655  

### XGBoost (Final Model)
- **Params:** `n_estimators=200`, `max_depth=10`, `learning_rate=0.1`  
- **Accuracy:** 0.636    **F1:** 0.5828    **ROC-AUC:** 0.688  

 **Selected Model:** XGBoost — best trade-off between precision & recall.

---

## Key Evaluation Plots
| Plot | Insight |
|------|----------|
| Confusion Matrix | Balanced identification of readmitted vs. non-readmitted patients |
| ROC Curve (AUC = 0.688) | Good separability between classes |
| Precision–Recall | High precision at lower recall — valuable for imbalanced healthcare data |

Top 5 important features:
`number_inpatient`, `discharge_disposition_id`, `number_emergency`, `time_in_hospital`, `num_medications`

---

## Deployment

##  Web Service Deployment (FastAPI)

After training and saving the model (`train.py`), we deployed it using a **FastAPI** web service (`serve.py`).  
The API loads the saved model, performs validation, and returns predictions in real time.

---

###  Endpoints Overview

| Method | Endpoint | Description |
|:--|:--|:--|
| `GET` | `/health` | Health check endpoint – confirms that the API is running |
| `POST` | `/predict` | Accepts patient features in JSON format and returns readmission probability |

---

##  API Endpoints and Responses

### Health Check

**Command:**
```bash
curl http://localhost:9696/health

Example Response:

{
  "status": "ok",
  "message": "Model is ready to predict."
}


Prediction Example
Command:

curl -X POST -H "Content-Type: application/json" \
-d '{
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
}' \
http://localhost:9696/predict

Example Response:
{
  "readmission_probability": 0.654,
  "readmitted": true
}

# Docker Setup

Build the image:

docker build -t diabetes-api .


Run the container:

docker run -it -p 9696:9696 diabetes-api


Then test with:

curl http://localhost:9696/health


and:

curl -X POST -H "Content-Type: application/json" \
-d '{"race":"Caucasian","gender":"Female", ... }' \
http://localhost:9696/predict

# Architecture Overview:

          +---------------------+
          |   Raw CSV Dataset   |
          +----------+----------+
                     |
                     v
          +---------------------+
          |  train.py (Model)   |
          +----------+----------+
                     |
                     v
          +---------------------+
          |  model_xgb.pkl      |
          +----------+----------+
                     |
                     v
          +---------------------+
          | serve.py (FastAPI)  |
          +----------+----------+
                     |
                     v
          +---------------------+
          |   User / Client     |
          |   (via cURL/API)    |
          +---------------------+


# Author
Olivera Jovanovic
Data Analyst | Machine Learning Zoomcamp 2025

