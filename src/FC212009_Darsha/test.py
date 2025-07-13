import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
from math import sqrt


model_path = 'models/xgboost_model.pkl'
model = joblib.load(model_path)


test_df = pd.read_csv("splits/test.csv")
X_test = test_df.drop(columns=["Price"])
y_test = test_df["Price"].values


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== FINAL TEST METRICS ===")
print(f"MAE (Rupees)     : {mae:,.2f}")
print(f"R²               : {r2:.3f}")
print(f"RMSE (Rupees)    : {rmse:,.2f}")


Path("reports").mkdir(exist_ok=True)
pd.DataFrame({
    "ActualPrice": y_test,
    "PredictedPrice": y_pred
}).to_csv("reports/test_predictions.csv", index=False)
print("✓ Saved reports/test_predictions.csv")
