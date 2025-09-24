from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../src/fc212030_Mudhitha/lapdata.pkl")
pipeline = joblib.load(MODEL_PATH)


# Initialize Flask app
app = Flask(__name__)

# Home route → form
@app.route("/")
def index():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Collect form data
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
                "TouchScreen": int(request.form["TouchScreen"]),
                "IPS": int(request.form["IPS"]),
                "PPI": float(request.form["PPI"])
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([data])

            # Predict log(price) → convert back
            log_price = pipeline.predict(input_df)[0]
            predicted_price = np.exp(log_price)

            return render_template(
                "index.html",
                prediction_text=f"Estimated Laptop Price: Rs. {predicted_price:,.2f}"
            )

        except Exception as e:
            return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
