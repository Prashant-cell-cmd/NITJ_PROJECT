from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf

app = FastAPI(title="Credit Card Fraud Detection API")

# Load model and scaler
model = tf.keras.models.load_model("cnn_fraud_model.h5")
scaler = joblib.load("scaler.pkl")

# Expected feature count (30)
EXPECTED_FEATURES = 30

class Transaction(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "CNN Fraud Detection API is running"}

@app.post("/predict")
def predict(data: Transaction):
    # STEP 1: Validate feature length
    if len(data.features) != EXPECTED_FEATURES:
        return {
            "error": f"Expected {EXPECTED_FEATURES} features, got {len(data.features)}"
        }

    # STEP 2: Convert to NumPy
    features = np.array(data.features).reshape(1, EXPECTED_FEATURES)

    # STEP 3: Scale
    features = scaler.transform(features)

    # STEP 4: Reshape for CNN (30, 1)
    features = features.reshape(1, EXPECTED_FEATURES, 1)

    # STEP 5: Predict
    prob = model.predict(features)[0][0]

    return {
        "fraud_probability": float(prob),
        "prediction": "Fraud" if prob > 0.5 else "Normal"
    }
