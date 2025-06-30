from pathlib import Path
import pandas as pd
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def load_data(fname: str = "cleaned_data.csv") -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / fname)


# Feature lists
features = [
    'Company', 'TypeName', 'Inches', 'Ram', 'Cpu_Speed', 
    'SSD', 'HDD', 'Gpu_Brand', 'Touchscreen', 'IPS',
    'Resolution_X', 'Resolution_Y', 'Weight', 'OpSys'
]

target = 'Price'

numeric_features = [
    'Inches', 'Ram', 'Cpu_Speed', 'SSD', 'HDD', 
    'Touchscreen', 'IPS', 'Resolution_X', 'Resolution_Y', 'Weight'
]

categorical_features = ['Company', 'TypeName', 'Gpu_Brand', 'OpSys']

def get_feature_names(pipeline):
    """Get feature names after preprocessing"""
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    return np.concatenate([numeric_features, cat_feature_names])