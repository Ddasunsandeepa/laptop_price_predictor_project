import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


df = pd.read_csv("splits/val.csv")
X = df.drop(columns=["Price"])
y = df["Price"]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_features = X.select_dtypes(include="number").columns
categorical_features = X.select_dtypes(include="object").columns

# Preprocessing pipelines for numeric and categorical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# XGBoost pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgb", XGBRegressor(random_state=42, n_jobs=-1))
])

# Hyperparamter tuning setup
param_distributions = {
    "xgb__n_estimators": [100, 300, 500],
    "xgb__max_depth": [3, 5, 7],
    "xgb__learning_rate": [0.01, 0.05, 0.1],
    "xgb__subsample": [0.6, 0.8, 1.0],
    "xgb__colsample_bytree": [0.6, 0.8, 1.0],
    "xgb__reg_alpha": [0, 0.1, 1],
    "xgb__reg_lambda": [1, 1.5, 2]
}

# Tuning with randomized search
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    verbose=2,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42
)

print("Starting hyperparameter tuning...")
search.fit(X_train, y_train)
print("Tuning completed!")

# Evaluation metrics
best_model = search.best_estimator_
y_pred = best_model.predict(X_val)
r2 = r2_score(y_val, y_pred)
print(f"Validation R²: {r2:.3f}")
rmse = sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.2f}")
print(f"Best Parameters:\n{search.best_params_}")


with open("models/xgboost_tuned_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Tuned model saved to models/xgboost_best_model.pkl")
