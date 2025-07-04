from sklearn.model_selection import train_test_split
import pandas as pd
import os

df = pd.read_csv("../../notebooks/FC212034_Kishgintharaam/cleaned_data.csv")
X, y = df.drop(columns=["Price"]), df["Price"]

# 15 % test first
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)

# 15 % validation from the remaining 85 %
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42)

os.makedirs("splits", exist_ok=True)
(X_train.assign(Price=y_train)
    .to_csv("splits/train.csv", index=False))
(X_val.assign(Price=y_val)
    .to_csv("splits/val.csv",   index=False))
(X_test.assign(Price=y_test)
    .to_csv("splits/test.csv",  index=False))
