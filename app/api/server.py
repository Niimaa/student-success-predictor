# File: app/api/server.py
from flask import Flask, request, jsonify
from app.model.predictor import StudentSuccessPredictor
import numpy as np
import os

# Paths to models and encoders
MODEL_PATH = "models/pass_fail_xgb_model.pkl"
ENCODER_PATH = "models/label_encoders.pkl"
MAPPING_PATH = "models/encoded_value_mapping.csv"

# Initialize predictor
predictor = StudentSuccessPredictor(MODEL_PATH, ENCODER_PATH, MAPPING_PATH)

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json.get("features")
        if not input_data:
            return jsonify({"error": "Missing 'features' in request body."}), 400

        encoded_input = np.array(input_data).reshape(1, -1)
        prediction, probability = predictor.predict(encoded_input)
        decoded_input = predictor.decode_input(encoded_input)

        # Sort decoded input to match feature order
        ordered_decoded_input = {key: decoded_input[key] for key in predictor.expected_features}

        result = {
            "predicted_class": "Pass" if prediction == 1 else "Fail",
            "probabilities": {
                "Fail": round(float(probability[0]), 4),
                "Pass": round(float(probability[1]), 4)
            },
            "decoded_input": ordered_decoded_input
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("✅ Flask server starting...")
    app.run(debug=True)

#in Ternminal type python -m app.api.server