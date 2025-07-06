from re import L
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

model_path = 'models/rf_tuned.pkl'
test_df = pd.read_csv("splits/test.csv")

LOG_PRICE = True

X_test  = test_df.drop(columns=["Price"])
y_test  = test_df["Price"].values

model = joblib.load(model_path)
y_pred = model.predict(X_test)

if LOG_PRICE:
    y_pred_rupees = np.expm1(y_pred)
    y_true_rupees = y_test

    
    mae_rupees = mean_absolute_error(y_true_rupees, y_pred_rupees)
    r2_rupees  = r2_score(y_true_rupees, y_pred_rupees)

    
    r2_log = r2_score(np.log1p(y_true_rupees), y_pred)

    print("\n=== FINAL TEST METRICS ===")
    print(f"MAE (rupees)        : {mae_rupees:,.2f}")
    print(f"R²  (rupees)        : {r2_rupees:.3f}")
    print(f"R²  (log target)    : {r2_log:.3f}")

else:
    mae_rupees = mean_absolute_error(y_test, y_pred)
    r2_rupees  = r2_score(y_test, y_pred)

    print("\n===== FINAL TEST METRICS =====")
    print(f"MAE (rupees) : {mae_rupees:,.2f}")
    print(f"R²           : {r2_rupees:.3f}")


Path("reports").mkdir(exist_ok=True)
out = pd.DataFrame({
    "ActualPrice": y_test,
    "PredictedPrice": y_pred_rupees if LOG_PRICE else y_pred
})
out.to_csv("reports/test_predictions.csv", index=False)
print("✓ Saved reports/test_predictions.csv")