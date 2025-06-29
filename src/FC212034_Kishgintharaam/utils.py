from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def load_data(fname: str = "cleaned_data.csv") -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / fname)

# def save_model(model, fname: str = "random_forest_model.pkl") -> None:
#     joblib.dump(model, MODEL_DIR / fname)
#     print(f"✅ Model saved → {MODEL_DIR/fname}")

# def load_model(fname: str = "random_forest_model.pkl"):
#     return joblib.load(MODEL_DIR / fname)
