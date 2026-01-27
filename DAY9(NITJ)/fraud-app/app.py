import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")
st.caption("AI-powered fraud detection using XGBoost & CatBoost")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    cat_model = joblib.load("model_cat.pkl")
    xgb_model = joblib.load("model_xgb.pkl")
    scaler = joblib.load("scaler.pkl")
    return cat_model, xgb_model, scaler

cat_model, xgb_model, scaler = load_models()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["CatBoost", "XGBoost"]
)

threshold = st.sidebar.slider(
    "Fraud Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.4
)

# --------------------------------------------------
# DATA UPLOAD
# --------------------------------------------------
st.subheader("📂 Upload Transaction Data")
uploaded_file = st.file_uploader(
    "Upload CSV file (credit card transactions)",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.write("### 🔍 Sample Data")
    st.dataframe(data.head())

    # --------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------
    X = data.drop("Class", axis=1, errors="ignore")
    X_scaled = scaler.transform(X)

    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------
    model = cat_model if model_choice == "CatBoost" else xgb_model
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    data["Fraud_Probability"] = probabilities
    data["Prediction"] = predictions
    data["Risk_Level"] = data["Fraud_Probability"].apply(
        lambda x: "High" if x > 0.7 else "Medium" if x > 0.3 else "Low"
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------
    st.subheader("📊 Prediction Results")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", len(data))
    col2.metric("Fraudulent", data["Prediction"].sum())
    col3.metric("Fraud Rate (%)", round(100 * data["Prediction"].mean(), 2))

    st.dataframe(data.head(20))

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------
    st.subheader("📈 Fraud Probability Distribution")

    fig, ax = plt.subplots()
    ax.hist(probabilities, bins=40)
    ax.set_xlabel("Fraud Probability")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # --------------------------------------------------
    # EXPLAINABILITY (SHAP)
    # --------------------------------------------------
    st.subheader("🧠 Model Explainability (SHAP)")

    if st.button("Generate SHAP Explanation (Top 100)"):
        explainer = shap.Explainer(model)
        shap_values = explainer(X_scaled[:100])

        st.write("Feature importance for fraud detection:")
        fig2 = plt.figure()
        shap.summary_plot(shap_values, X.iloc[:100], show=False)
        st.pyplot(fig2)

    # --------------------------------------------------
    # DOWNLOAD RESULTS
    # --------------------------------------------------
    st.subheader("⬇️ Download Predictions")
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Result CSV",
        csv,
        "fraud_predictions.csv",
        "text/csv"
    )

else:
    st.info("Please upload a CSV file to start fraud detection.")
