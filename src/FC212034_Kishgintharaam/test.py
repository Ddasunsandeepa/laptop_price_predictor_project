from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import load_data, features, target, get_feature_names
import matplotlib.pyplot as plt
import sys
from datetime import datetime


# Function to evaluate the model
# It calculates RMSE, MAE, R2, and standard deviation of predictions and actual
# Returns a dictionary with these metrics
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'Prediction_Std': np.std(y_pred),
            'Actual_Std': np.std(y_test)
        }
        return metrics
    except Exception as e:
        print(f"❌ Error during model evaluation: {str(e)}")
        return None



# FUnction to plot the feature importances as bar chart
def plot_feature_importances(model, model_path=None, top_n=15): 
    try:
        # Try to get feature importances
        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
            importances = model.named_steps['regressor'].feature_importances_
        else:
            print("⚠️ Model has no feature_importances_ attribute. Skipping plot.")
            return

        feature_names = get_feature_names(model)
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Create results directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        plot_dir = Path("test_results") / timestamp
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        model_name = Path(model_path).stem if model_path else "model"
        fname = plot_dir / f"feature_importances_{model_name}.png"

        # Create plot
        plt.figure(figsize=(12, 8))
        plt.title(f"Top {top_n} Feature Importances ({model_name})", fontsize=14)
        bars = plt.bar(range(top_n), importances[indices], align='center', color='skyblue')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel("Features", fontsize=12)
        plt.ylabel("Importance Score", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print("\n")
        print(f"✅ Feature importances saved to /test_results/{timestamp}")
        
    except Exception as e:
        print(f"❌ Error creating feature importance plot: {str(e)}")


# Main function to load data, model, evaluate and plot feature importances
# It handles errors gracefully and provides informative messages
# It also checks if the model file exists and if the data is valid
def main(model_path: Path):
    print(f"\n{'='*50}")
    print(f"{'MODEL EVALUATION':^50}")
    print(f"{'='*50}\n")
    
    try:
        # Load data with validation
        df = load_data('cleaned_data.csv')
        if df.empty:
            raise ValueError("Data file is empty or invalid")
            
        X = df[features]
        y = df[target]
        
        # Verify feature consistency
        if len(X.columns) != len(features):
            missing = set(features) - set(X.columns)
            raise ValueError(f"Missing features in data: {missing}")

        # Load model with error handling
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = joblib.load(model_path)
        print(f"🔍 Evaluating model: {model_path.name}\n")

        # Evaluate model
        metrics = evaluate_model(model, X, y)
        if metrics is None:
            raise RuntimeError("Model evaluation failed")
            
        print("\n📊 Model Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"• {metric:>15}: {value:.4f}")
        
        # Plot feature importances
        plot_feature_importances(model, model_path)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    print("\n✅ Evaluation completed successfully")


# Entry point for the script
# It allows the user to specify a model path via command line argument or use a default path
if __name__ == "__main__":
    # Configure default model path
    default_model = Path('model/random_forest_model.pkl')
    
    # Use command-line argument if provided, else use default
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
    else:
        model_path = default_model
    
    main(model_path)