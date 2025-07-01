import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import load_data, features, target, get_feature_names
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def plot_feature_importances(model, model_path=None, top_n=15):
    try:
        importances = model.named_steps['regressor'].feature_importances_
    except AttributeError:
        print("Model has no feature_importances_ attribute. Skipping plot.")
        return

    feature_names = get_feature_names(model)
    indices = np.argsort(importances)[::-1][:top_n]
    
    plot_dir = Path("src/FC212034_Kishgintharaam/test_results")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    if model_path:
        model_name = Path(model_path).stem
        fname = plot_dir / f"feature_importances_{model_name}.png"
    else:
        fname = plot_dir / "feature_importances.png"

    plt.figure(figsize=(10, 6))
    plt.title("Top Feature Importances")
    plt.bar(range(top_n), importances[indices], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Feature importances saved to {fname}")


def main(model_path: Path):
    # Load data
    df = load_data('cleaned_data.csv')
    if df.empty:
        raise ValueError("DataFrame is empty. Please check the data file.")
    
    X = df[features]
    y = df[target]
    
    # Load trained model
    model = joblib.load(model_path)
    print(f"\n Evaluating model from {model_path}") 

    # Evaluate model
    metrics = evaluate_model(model, X, y)
    
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    # Plot feature importances
    plot_feature_importances(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model and plot feature importances.")
    parser.add_argument('--model_path', type=Path, default='laptop_price_rf_model.pkl', help='Path to the trained model file')
    
    args = parser.parse_args()
    
    main(Path(args.model_path))