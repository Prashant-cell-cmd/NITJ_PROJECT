from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load trained model
model = joblib.load("fraud_detection_model.pkl")

# Initialize FastAPI app
app = FastAPI(title="Credit Card Fraud Detection API")

# -------------------------------
# Input schema (Transaction)
# -------------------------------
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# -------------------------------
# Home route
# -------------------------------
@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API is running"}


# -------------------------------
# PREDICT ROUTE (OPTION 1 FIX)
# -------------------------------
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convert input JSON to NumPy array
        data = np.array([list(transaction.dict().values())])

        # Make prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        return {
            "fraud": int(prediction),
            "risk_score": float(probability)
        }

    except Exception as e:
        return {"error": str(e)}
