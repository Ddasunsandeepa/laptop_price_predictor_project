import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import (cross_val_score, 
                                   GridSearchCV, 
                                   RandomizedSearchCV,
                                   KFold)
from sklearn.metrics import make_scorer, mean_squared_error
from utils import load_data, features, target, numeric_features, categorical_features
from train import create_pipeline
import joblib
import matplotlib.pyplot as plt

def custom_rmse(y_true, y_pred):
    """Custom RMSE calculation without squared parameter"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def run_cross_validation(model, X, y, cv=5):
    """Run k-fold cross validation with fixed scoring"""
    scorers = {
        'RMSE': make_scorer(custom_rmse),
        'R2': 'r2',
        'MAE': 'neg_mean_absolute_error'
    }
    
    results = {}
    for name, scorer in scorers.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            results[name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores.tolist()
            }
        except Exception as e:
            print(f"⚠️ Error calculating {name}: {str(e)}")
            results[name] = {
                'mean': np.nan,
                'std': np.nan,
                'scores': []
            }
    return results

def hyperparameter_tuning(X, y):
    """Perform hyperparameter optimization with fixed scoring"""
    pipeline = create_pipeline()
    
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
    }
    
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    scorer = make_scorer(custom_rmse)  # Using our custom RMSE
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=20,
        cv=kfold,
        scoring=scorer,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    
    print("\n⚙️ Starting hyperparameter tuning...")
    search.fit(X, y)
    return search

def save_tuning_results(search, filename="validation_results.csv"):
    """Save tuning results to CSV"""
    results = pd.DataFrame(search.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)
    
    output_dir = Path("validation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    results.to_csv(filepath, index=False)
    print(f"✅ Tuning results saved to {filepath}")
    return results

def main():
    print(f"\n{'='*50}")
    print(f"{'MODEL VALIDATION':^50}")
    print(f"{'='*50}\n")
    
    try:
        df = load_data('cleaned_data.csv')
        if df.empty:
            raise ValueError("❌ DataFrame is empty. Please check the data file.")
        X = df[features]
        y = df[target]
        
        # 1. Cross-validation with error handling
        print("\n🔍 Running cross-validation...")
        pipeline = create_pipeline()
        cv_results = run_cross_validation(pipeline, X, y)
        
        print("\n📊 Cross-Validation Results:")
        for metric, values in cv_results.items():
            print(f"{metric}: {values['mean']:.3f} ± {values['std']:.3f}")
        
        # 2. Hyperparameter tuning
        search = hyperparameter_tuning(X, y)
        
        print("\n🎯 Best Parameters:")
        for param, value in search.best_params_.items():
            print(f"{param:>25}: {value}")
        
        print(f"\n🏆 Best RMSE: {search.best_score_:.3f}")
        
        # 3. Save results
        results_df = save_tuning_results(search)
        
        # 4. Save best model
        best_model = search.best_estimator_
        model_path = Path("model/tuned_rf_model.pkl")
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(best_model, model_path)
        print(f"\n💾 Best model saved to: {model_path}")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()