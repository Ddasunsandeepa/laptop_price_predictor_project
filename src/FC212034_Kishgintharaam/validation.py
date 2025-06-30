import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from data_preprocessing import features, target
import numpy as np
from utils import load_data

def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation and return scores"""
    scores = cross_val_score(
        model, X, y, 
        cv=cv, 
        scoring='neg_mean_squared_error'
    )
    return np.sqrt(-scores)  # Return RMSE scores

if __name__ == "__main__":
    # Load data
    df = load_data('cleaned_data.csv')
    
    if df.empty:
        raise ValueError("DataFrame is empty. Please check the data file.")
    # Ensure the DataFrame has the required columns
    if not all(col in df.columns for col in features + [target]):
        raise ValueError("DataFrame does not contain all required features and target column.")
    
    X = df[features]
    y = df[target]
    
    # Load trained model pipeline
    model = joblib.load('laptop_price_rf_model.pkl')
    
    # Cross-validate
    rmse_scores = cross_validate_model(model, X, y)
    
    print("\nCross-Validation Results:")
    print(f"RMSE Scores: {rmse_scores}")
    print(f"Mean RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")