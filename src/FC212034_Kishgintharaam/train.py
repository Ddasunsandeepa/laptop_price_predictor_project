import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from utils import features, target, numeric_features, categorical_features, load_data



def create_pipeline():
    """Create the preprocessing and modeling pipeline"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  
                ('scaler', 'passthrough')  
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf_model)
    ])

def train_model(X_train, y_train):
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline

if __name__ == "__main__":
    # Load data
    df = load_data('cleaned_data.csv')
    if df.empty:
        raise ValueError("DataFrame is empty. Please check the data file.")
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    print("Model trained successfully.")
