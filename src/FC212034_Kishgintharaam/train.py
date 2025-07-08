# This file is to train a Random Forest model on the provided dataset.
# Dataset - splits/train.csv
# Model - RandomForestRegressor
# Libraries - pandas, scikit-learn, pickle
# Output - models/train_data.pkl, models/random_forest_model.pkl

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv("splits/train.csv")
X = df.drop(columns=["Price"])
y = df["Price"]

numeric = X.select_dtypes("number").columns
categorical = X.select_dtypes("object").columns

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical)
])

model = Pipeline([
    ("prep", preprocess),
    ("rf",   RandomForestRegressor(n_estimators=600, 
                                   random_state=42, 
                                   max_samples=0.8,
                                   max_features=0.5,
                                   max_depth=None,
                                   n_jobs=-1))
])

model.fit(X, y)
pickle.dump(df, open("models/train_data.pkl", "wb"))
pickle.dump(model, open("models/random_forest_model.pkl", "wb"))
print("✓ model trained and saved")
