import joblib
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load saved components
model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("standard_scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

# Numeric columns used during training
numerical_cols = ["age", "income", "loan_amount"]  # Replace with yours

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Create DataFrame
        input_df = pd.DataFrame([data])

        # Match training columns
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Scale numeric features
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Predict
        pred = model.predict(input_df)
        label = label_encoder.inverse_transform(pred)

        return jsonify({"prediction": label[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
