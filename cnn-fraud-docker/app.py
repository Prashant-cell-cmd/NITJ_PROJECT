from fastapi import FastAPI
import numpy as np
import joblib
import tensorflow as tf

app = FastAPI(title="Credit Card Fraud Detection API")

model = tf.keras.models.load_model("cnn_fraud_model.h5")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "CNN Fraud Detection API is running"}

@app.post("/predict")
def predict(features: list):
    data = np.array(features).reshape(1, -1)
    data = scaler.transform(data)
    data = data.reshape(1, data.shape[1], 1)

    prob = model.predict(data)[0][0]
    return {
        "fraud_probability": float(prob),
        "prediction": "Fraud" if prob > 0.5 else "Normal"
    }
