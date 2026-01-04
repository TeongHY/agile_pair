# predictive_app.py
import streamlit as st
import pandas as pd
import pickle
import time
from datetime import datetime
import os

# Load models and scaler
with open("model_v1.pkl", "rb") as f:
    model_v1 = pickle.load(f)

with open("model_v2.pkl", "rb") as f:
    model_v2 = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Create CSV if it doesn't exist
if not os.path.exists("monitoring_logs.csv"):
    df_init = pd.DataFrame(columns=["timestamp", "Pregnancies", "Glucose", "Insulin", "BMI", "Age", "prediction_v1", "prediction_v2", "latency", "feedback_score", "feedback_comment"])

# Streamlit UI
st.title("Diabetes Prediction App (V1 vs V2)")

# Streamlit UI
st.title("Diabetes Prediction App (V1 vs V2)")

st.sidebar.header("Input your health data")
preg = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
gluc = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=100)
ins = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

input_df = pd.DataFrame([[preg, gluc, ins, bmi, age]], columns=["Pregnancies", "Glucose", "Insulin", "BMI", "Age"])

# Scale input for model_v2
input_scaled = scaler.transform(input_df)

# Predict
start_time = time.time()
pred_v1 = model_v1.predict(input_df)[0]
pred_v2 = model_v2.predict(input_scaled)[0]
latency = round((time.time() - start_time) * 1000, 2)  # in ms

st.write(f"**Prediction V1 (Logistic Regression):** {'Diabetic' if pred_v1==1 else 'Non-diabetic'}")
st.write(f"**Prediction V2 (Random Forest):** {'Diabetic' if pred_v2==1 else 'Non-diabetic'}")
st.write(f"**Prediction latency:** {latency} ms")

# User feedback
st.subheader("Feedback")
score = st.slider("Rate the prediction", 1, 5, 3)
comment = st.text_area("Comment")

if st.button("Submit Feedback"):
    new_log = pd.DataFrame([[datetime.now(), preg, gluc, ins, bmi, age, pred_v1, pred_v2, latency, score, comment]], columns=[
        "timestamp", "Pregnancies", "Glucose", "Insulin", "BMI", "Age",
        "prediction_v1", "prediction_v2", "latency", "feedback_score", "feedback_comment"])
    new_log.to_csv("monitoring_logs.csv", mode="a", index=False, header=False)
    st.success("Feedback submitted!")

