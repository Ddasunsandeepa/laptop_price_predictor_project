import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt


df = pd.read_csv("splits/train.csv")
X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


numeric = X.select_dtypes("number").columns
categorical = X.select_dtypes("object").columns

# Preprocessing pipeline
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler())
    ]), numeric),
    
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical)
])

# Full pipeline
pipeline = Pipeline([
    ("prep", preprocess),
    ("xgb", XGBRegressor(random_state=42, n_jobs=-1))
])

# Parameter grid
param_grid = {
    "xgb__n_estimators": [200, 400, 600],
    "xgb__max_depth": [3, 6, 9],
    "xgb__learning_rate": [0.01, 0.05, 0.1],
    "xgb__subsample": [0.6, 0.8, 1.0],
    "xgb__colsample_bytree": [0.4, 0.6, 0.8],
    "xgb__reg_alpha": [0, 0.1, 1],
    "xgb__reg_lambda": [1, 1.5, 2]
}

# Random search
search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=20,
                            cv=5, verbose=1, n_jobs=-1, scoring="neg_root_mean_squared_error")

search.fit(X_train, y_train)

best_model = search.best_estimator_

# Save model
pickle.dump(best_model, open("models/xgboost_model.pkl", "wb"))
print("✓ Best model saved as xgboost_best_model.pkl")
