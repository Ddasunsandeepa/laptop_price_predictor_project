import joblib
from sklearn.preprocessing import StandardScaler

def save_pickle(obj, path):
    """Save any Python object to a pickle file."""
    joblib.dump(obj, path)

def load_pickle(path):
    """Load a pickle file."""
    return joblib.load(path)

def scale_features(scaler: StandardScaler, data, fit=False):
    """Scale data using the provided scaler. Fit if needed."""
    if fit:
        return scaler.fit_transform(data)
    else:
        return scaler.transform(data)
