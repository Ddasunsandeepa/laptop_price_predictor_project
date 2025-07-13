import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths 
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../notebooks/FC212012-Sajani/processed_data.csv"))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
SPLITS_DIR = os.path.join(BASE_DIR, "Splits")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

# Load dataset 
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Price"])

# Drop the original categorical columns (already one-hot encoded)
X = X.drop(columns=['Company', 'TypeName'])

y = df["Price"]

# Split: train -> 70%, val -> 15%, test -> 15% 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save raw splits
pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False)
pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)

# Feature scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save scaled train/val for quick access
joblib.dump((X_train_scaled, y_train), os.path.join(MODELS_DIR, "train_data.pkl"))
joblib.dump((X_val_scaled, y_val), os.path.join(MODELS_DIR, "val_data.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

# Train default SVR 
svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)
joblib.dump(svr_model, os.path.join(MODELS_DIR, "svr_model.pkl"))

# GridSearch for tuned SVR 
param_grid = {
    'kernel': ['rbf'],  # Best performance usually with RBF
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
    'epsilon': [0.1, 0.2, 0.5]
}

grid_search = GridSearchCV(SVR(), param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
joblib.dump(grid_search.best_estimator_, os.path.join(MODELS_DIR, "svr_tuned.pkl"))

# Evaluate and report 
def print_metrics(y_true, y_pred, label="Model"):
    print(f"\n{label} Evaluation:")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"R2: {r2_score(y_true, y_pred):.4f}")

print_metrics(y_val, svr_model.predict(X_val_scaled), label="Default SVR")
print_metrics(y_val, grid_search.predict(X_val_scaled), label="Tuned SVR")
