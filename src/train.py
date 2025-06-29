from pathlib import Path
import argparse, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the cleaned data
df = pd.read_csv('../../data/preprocesse    d_data_rf.csv')


# Selected features
features = [
    'Company', 'TypeName', 'Inches', 'Ram', 'Cpu_Speed', 
    'SSD', 'HDD', 'Gpu_Brand', 'Touchscreen', 'IPS',
    'Resolution_X', 'Resolution_Y', 'Weight', 'OpSys'
]

target = 'Price'