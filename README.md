# Diabetes-Readmission-Prediction
Predict 30-day readmission risk for diabetic patients (Kaggle dataset).
# Project Overview
This project predicts the 30-day readmission risk for diabetic patients using hospital data from Kaggle.  
The goal is to practice end-to-end machine learning — from data exploration to model building — on realistic healthcare data.

---

# Planned Folder Structure

```
diabetes-readmission-prediction/
├── data/                  # raw data (not committed) + processed
├── notebooks/             # Jupyter notebooks for EDA and modeling
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing_model.ipynb
│   └── 03_evaluation_explainability.ipynb
├── src/                   # Python scripts for main steps
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── serve.py
├── docker/                # optional later - for deployment
│   └── Dockerfile
├── requirements.txt       # list of required Python libraries
├── .gitignore             # ignore files you don’t want to commit
├── README.md              # project description (this file)
└── model_card.md          # short summary of final model details


# Author
Olivera Jovanovic
