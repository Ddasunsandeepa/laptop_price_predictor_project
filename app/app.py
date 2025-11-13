from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../src/fc212030_Mudhitha/lapdata.pkl")
pipeline = joblib.load(MODEL_PATH)

EXCHANGE_API_URL = "https://api.exchangerate.host/convert?from=INR&to=LKR"

app = Flask(__name__)

def get_inr_to_lkr_rate():
    try:
        response = requests.get(EXCHANGE_API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        rate = data.get("result")
        return rate if rate else 4.0
    except Exception:
        return 4.0

@app.route("/")
def index():
    companies = ["Dell", "HP", "Lenovo", "Asus", "Apple", "Acer", "MSI"]
    types = ["Ultrabook", "Gaming", "Notebook", "2 in 1", "Workstation"]
    os_options = ["Windows", "MacOS", "Linux", "Chrome OS"]
    cpu_options = ["Intel i3", "Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"]
    gpu_options = ["Intel", "Nvidia GTX", "Nvidia RTX", "AMD Radeon"]

    return render_template(
        "index.html",
        companies=companies,
        types=types,
        os_options=os_options,
        cpu_options=cpu_options,
        gpu_options=gpu_options
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "Company": request.form["Company"],
            "TypeName": request.form["TypeName"],
            "Ram": int(request.form["Ram"]),
            "OpSys": request.form["OpSys"],
            "Weight": float(request.form["Weight"]),
            "CPU": request.form["CPU"],
            "SSD": float(request.form["SSD"]),
            "HDD": float(request.form["HDD"]),
            "GPU": request.form["GPU"],
            "TouchScreen": int(request.form.get("TouchScreen", 0)),
            "IPS": int(request.form.get("IPS", 0)),
            "PPI": float(request.form["PPI"])
        }

        input_df = pd.DataFrame([data])
        log_price_inr = pipeline.predict(input_df)[0]
        price_inr = np.exp(log_price_inr)

        exchange_rate = get_inr_to_lkr_rate()
        price_lkr = price_inr * exchange_rate

        prediction_msg = f"Estimated Laptop Price:\nLKR {price_lkr:,.2f}"

        # Pass dropdown options again so the form doesn't break
        return render_template(
            "index.html",
            prediction_text=prediction_msg,
            companies=["Dell", "HP", "Lenovo", "Asus", "Apple", "Acer", "MSI"],
            types=["Ultrabook", "Gaming", "Notebook", "2 in 1", "Workstation"],
            os_options=["Windows", "MacOS", "Linux", "Chrome OS"],
            cpu_options=["Intel i3", "Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"],
            gpu_options=["Intel", "Nvidia GTX", "Nvidia RTX", "AMD Radeon"]
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error predicting price: {str(e)}\nPlease check your input values.",
            companies=["Dell", "HP", "Lenovo", "Asus", "Apple", "Acer", "MSI"],
            types=["Ultrabook", "Gaming", "Notebook", "2 in 1", "Workstation"],
            os_options=["Windows", "MacOS", "Linux", "Chrome OS"],
            cpu_options=["Intel i3", "Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"],
            gpu_options=["Intel", "Nvidia GTX", "Nvidia RTX", "AMD Radeon"]
        )

if __name__ == "__main__":
    app.run(debug=True)
