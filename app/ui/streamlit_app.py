# File: app/ui/streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Load models and mappings
MODEL_PATH = "models/pass_fail_xgb_model.pkl"
ENCODER_PATH = "models/label_encoders.pkl"
MAPPING_PATH = "models/encoded_value_mapping.csv"

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)
mapping_df = pd.read_csv(MAPPING_PATH)

# Expected features in order
expected_features = [
    "ProgrammeName", "SecondarySchoolAward", "PreStudyActivity",
    "SecondarySchool", "Ethnicity", "FirstTertiaryYear", "Age_Group"
]

# Build dropdown options from encoders
def get_dropdown_options(col):
    return list(label_encoders[col].classes_)

# Encode input values
def encode_input(user_input):
    encoded = []
    for col, val in zip(expected_features, user_input):
        le = label_encoders[col]
        encoded_val = le.transform([val])[0]
        encoded.append(encoded_val)
    return np.array(encoded).reshape(1, -1)

# Decode input values
def decode_input(encoded_input):
    decoded = {}
    for col, val in zip(expected_features, encoded_input[0]):
        match = mapping_df[(mapping_df["Category"] == col) & (mapping_df["EncodedValue"] == val)]
        decoded[col] = match["DecodedValue"].values[0] if not match.empty else "Unknown"
    return decoded

# UI
st.title("🎓 Student Success Predictor")
st.write("Use the dropdowns below to predict whether a student is likely to complete their programme successfully.")

user_inputs = []
for feature in expected_features:
    options = get_dropdown_options(feature)
    selection = st.selectbox(f"Select {feature.replace('_', ' ')}:", options)
    user_inputs.append(selection)

if st.button("Predict Success"):
    encoded_input = encode_input(user_inputs)
    prediction = model.predict(encoded_input)[0]
    proba = model.predict_proba(encoded_input)[0]
    decoded = decode_input(encoded_input)

    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Prediction:** {'✅ Pass' if prediction == 1 else '❌ Fail'}")
    st.write({"Pass Probability": round(proba[1] * 100, 2), "Fail Probability": round(proba[0] * 100, 2)})

    st.subheader("🧾 Input Summary")
    st.json(decoded)
