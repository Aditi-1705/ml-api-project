from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

# Enable CORS
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
    return {
        "message": "Breast Cancer Prediction API"
    }


@app.post("/predict")
def predict(data: InputData):

    features = data.features

    # Prediction
    prediction = model.predict([features])[0]

    # Confidence
    probability = model.predict_proba([features])[0]

    # IMPORTANT FIX
    # 0 = Malignant
    # 1 = Benign

    if prediction == 0:
        result = "Malignant"
        confidence = probability[0] * 100
    else:
        result = "Benign"
        confidence = probability[1] * 100

    return {
        "prediction": int(prediction),
        "result": result,
        "confidence": round(confidence, 2)
    }
