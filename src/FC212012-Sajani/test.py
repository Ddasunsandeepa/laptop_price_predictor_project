import os
import pandas as pd
import joblib

# Paths 
BASE_DIR = os.path.dirname(__file__)
SPLITS_DIR = os.path.join(BASE_DIR, "Splits")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
REPORTS_DIR = os.path.join(BASE_DIR, "Reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

# Load test data
test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))

X_test = test_df.drop(columns=["Price"])
y_test = test_df["Price"]

# Load scaler and model
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
model = joblib.load(os.path.join(MODELS_DIR, "svr_tuned.pkl"))

# Scale test features
X_test_scaled = scaler.transform(X_test)

# Predict 
predictions = model.predict(X_test_scaled)

# Save results 
results_df = X_test.copy()
results_df["True_Price"] = y_test
results_df["Predicted_Price"] = predictions

results_df.to_csv(os.path.join(REPORTS_DIR, "test_predictions.csv"), index=False)

print(f"Test predictions saved to {os.path.join(REPORTS_DIR, 'test_predictions.csv')}")
