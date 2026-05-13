from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

# ✅ CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

    prediction = model.predict([features])[0]

    probability = model.predict_proba([features])[0][1]

    result = "Malignant" if prediction == 1 else "Benign"

    return {
        "prediction": int(prediction),
        "result": result,
        "confidence": round(probability * 100, 2)
    }
