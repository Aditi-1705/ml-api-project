from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 🔥 ADD THIS BLOCK EXACTLY
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained model
model = joblib.load("model.pkl")


# Input format
class InputData(BaseModel):
    features: list


@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API"}


@app.post("/predict")
def predict(data: InputData):
    features = data.features

    # Prediction
    prediction = model.predict([features])[0]

    # Probability (confidence)
    probability = model.predict_proba([features])[0][1]

    # Result label
    result = "Malignant" if prediction == 1 else "Benign"

    return {
        "prediction": int(prediction),
        "result": result,
        "confidence": round(probability * 100, 2)
    }
