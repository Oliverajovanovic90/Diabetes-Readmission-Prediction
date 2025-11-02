# train.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

#  Load dataset
file_path = "data/diabetes/diabetic_data.csv"
df = pd.read_csv(file_path)

# Prepare target
df['readmitted'] = df['readmitted'].replace({'>30': 1, '<30': 1, 'NO': 0})

# Drop unused columns
drop_cols = ['encounter_id', 'patient_nbr', 'weight', 'payer_code',
             'medical_specialty', 'diag_1', 'diag_2', 'diag_3']
df = df.drop(columns=drop_cols)

# Split numeric and categorical
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'readmitted' in numeric_cols:
    numeric_cols.remove('readmitted')
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Split train/test
X = df.drop('readmitted', axis=1)
y = df['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

#  Model
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        random_state=42
    ))
])

# Train model
xgb_model.fit(X_train, y_train)

# Save model
with open("model_xgb.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

print("Model trained and saved as model_xgb.pkl")