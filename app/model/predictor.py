import joblib
import numpy as np
import pandas as pd

class StudentSuccessPredictor:
    def __init__(self, model_path, encoder_path, mapping_path):
        self.model = joblib.load(model_path)
        self.label_encoders = joblib.load(encoder_path)
        self.mapping_df = pd.read_csv(mapping_path)
        self.expected_features = [
            "ProgrammeName", "SecondarySchoolAward", "PreStudyActivity",
            "SecondarySchool", "Ethnicity", "FirstTertiaryYear", "Age_Group"
        ]

    def predict(self, encoded_input):
        prediction = self.model.predict(encoded_input)[0]
        probability = self.model.predict_proba(encoded_input)[0]
        return prediction, probability

    def decode_input(self, encoded_input):
        decoded = {}
        for col, encoded_value in zip(self.expected_features, encoded_input[0]):
            match = self.mapping_df[
                (self.mapping_df["Category"] == col) &
                (self.mapping_df["EncodedValue"] == encoded_value)
            ]
            decoded[col] = match["DecodedValue"].values[0] if not match.empty else "Not Found"
        return decoded
