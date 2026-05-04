from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

# Define request schema
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API"}

@app.post("/predict")
def predict(data: InputData):
    arr = np.array([data.features])
    prediction = model.predict(arr)[0]

    return {
        "prediction": int(prediction),
        "result": "Malignant" if prediction == 0 else "Benign"
    }
