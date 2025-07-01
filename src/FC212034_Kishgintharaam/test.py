import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from utils import features, target, get_feature_names, load_data

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    return metrics

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    importances = model.named_steps['regressor'].feature_importances_
    
    plt.figure(figsize=(12, 8))
    plt.barh(feature_names[:15], importances[:15])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Important Features')
    plt.gca().invert_yaxis()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    # Load test data
    df = load_data('cleaned_data.csv')

    if df.empty:
        raise ValueError("DataFrame is empty. Please check the data file.")
    
    # Ensure the DataFrame has the required columns
    if not all(col in df.columns for col in features + [target]):
        raise ValueError("DataFrame does not contain all required features and target column.")
    
    # Split into features and target
    X_test = df[features]
    y_test = df[target]
    
    # Load trained model
    model = joblib.load('laptop_price_rf_model.pkl')
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("\nModel Performance")
    print(f"MAE: {metrics['MAE']:.2f} LKR")
    print(f"RMSE: {metrics['RMSE']:.2f} LKR")
    print(f"R2 Score: {metrics['R2']:.4f}")
    
    # Plot feature importance
    feature_names = get_feature_names(model)  
    plot_feature_importance(model, feature_names)