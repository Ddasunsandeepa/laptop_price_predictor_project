# This file is to hyper‑tune a Random Forest model on the provided dataset using RandomizedSearchCV.
# Dataset - splits/train.csv, splits/val.csv
# Model - RandomForestRegressor
# Libraries - pandas, scikit-learn, joblib
# Output - models/rf_tuned.pkl

import pandas as pd, numpy as np, sklearn, joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import randint, uniform
from pathlib import Path


train_df = pd.read_csv("splits/train.csv")
val_df   = pd.read_csv("splits/val.csv")

X_train, y_train = train_df.drop(columns=["Price"]), np.log1p(train_df["Price"])
X_val,   y_val   = val_df.drop(columns=["Price"]),   np.log1p(val_df["Price"])   


num_cols = X_train.select_dtypes("number").columns
cat_cols = X_train.select_dtypes("object").columns
s_kw = {"sparse_output": False} if sklearn.__version__ >= "1.2" else {"sparse": False}

preproc = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc" , StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh" , OneHotEncoder(handle_unknown="ignore", **s_kw))]), cat_cols)
])

pipe = Pipeline([
    ("prep", preproc),
    ("rf",   RandomForestRegressor(
                 random_state=42,
                 n_jobs=-1,
                 criterion="absolute_error"          
    ))
])


param_dist = {
    "rf__n_estimators":       randint(400, 1000),
    "rf__max_depth":          [None] + list(range(14, 30, 4)),
    "rf__max_features":       uniform(0.4, 0.5),     
    "rf__min_samples_leaf":   randint(1, 6),
    "rf__max_samples":        uniform(0.6, 0.4)       
}

search = RandomizedSearchCV(
    pipe,
    param_distributions = param_dist,
    n_iter = 40,
    cv = 5,
    scoring = "neg_mean_absolute_error",   
    n_jobs = -1,
    random_state = 42,
    verbose = 2
)


search.fit(X_train, y_train)

print("\n=== Best hyper‑parameters (CV) ===")
for k, v in search.best_params_.items():
    print(f"{k}: {v}")
print(f"Best CV (neg‑MAE, log‑space): {search.best_score_:.4f}")

best_model = search.best_estimator_


val_pred_log = best_model.predict(X_val)
val_pred     = np.expm1(val_pred_log)        
y_val_orig   = np.expm1(y_val)

print("\n=== Validation metrics ===")
print(f"R²  (log target): {r2_score(y_val, val_pred_log):.3f}")
print(f"MAE (LKR)    : {mean_absolute_error(y_val_orig, val_pred):.2f}")


Path("models").mkdir(exist_ok=True)
joblib.dump(best_model, "models/rf_tuned.pkl")
print("\n✓ Tuned model saved to models/rf_tuned.pkl")
