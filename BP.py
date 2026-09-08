import joblib
import json

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import numpy as np

app = Flask(__name__)

# Define filenames for saving cleaned components
model_filename = 'random_forest_model.joblib'
scaler_filename = 'standard_scaler.joblib'
label_encoder_filename = 'label_encoder.joblib'
feature_names_filename = 'feature_names.json'

# Save the cleaned model
joblib.dump(model_cleaned, model_filename)
print(f"Final model saved to {model_filename}")

# Save the cleaned scaler
joblib.dump(scaler_cleaned, scaler_filename)
print(f"Final scaler saved to {scaler_filename}")

# Save the label encoder
# Note: The original 'le' object is still valid as categories haven't changed
joblib.dump(le, label_encoder_filename)
print(f"Final label encoder saved to {label_encoder_filename}")

# Save the list of feature names used for training
feature_names = X_train_scaled_cleaned.columns.tolist()
with open(feature_names_filename, 'w') as f:
    json.dump(feature_names, f)
print(f"Feature names saved to {feature_names_filename}")


# Load the model, scaler, label encoder, and feature names
try:
    model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('standard_scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
    print("All components loaded successfully.")
except Exception as e:
    print(f"Error loading components: {e}")
    # In a real deployment, you might want to exit or log a critical error

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Convert incoming data to a pandas DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure all expected feature columns are present and in the correct order
        # Fill missing columns with 0 for one-hot encoded features, or mean/mode for others
        # based on your preprocessing strategy.
        # For simplicity here, we assume numerical features not provided are 0, 
        # and categorical features not provided mean their one-hot encoded column is 0.
        # A more robust solution would handle missing values based on training data statistics.
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Identify numerical columns for scaling within the current input_df
        numerical_cols_input = input_df.select_dtypes(include=[np.number]).columns

        # Apply scaler to the numerical columns
        # Ensure only columns that were originally numerical and scaled are transformed
        scaled_input_data = input_df.copy()
        scaled_input_data[numerical_cols_input] = scaler.transform(scaled_input_data[numerical_cols_input])

        # Make prediction
        prediction_encoded = model.predict(scaled_input_data)

        # Inverse transform the prediction to get original labels
        prediction_label = label_encoder.inverse_transform(prediction_encoded)

        return jsonify({'prediction': prediction_label[0].tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # In a production environment, debug=False and use a production-ready WSGI server
    app.run(debug=True, host='0.0.0.0', port=5000)
