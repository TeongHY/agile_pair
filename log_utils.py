# log_utils.py
import pandas as pd
from datetime import datetime
import os

LOG_FILE = "monitoring_logs.csv"

def init_log():
# Create the CSV file if it doesn't exist, with proper columns
    if not os.path.exists(LOG_FILE):
        df_init = pd.DataFrame(columns=[
            "timestamp", "Pregnancies", "Glucose", "Insulin", "BMI", "Age",
            "prediction_v1", "prediction_v2", "latency", "feedback_score", "feedback_comment"
        ])
        df_init.to_csv(LOG_FILE, index=False)

def log_prediction(input_data, pred_v1, pred_v2, latency, feedback_score=None, feedback_comment=None):
    if not os.path.exists(LOG_FILE):init_log()
    new_log = pd.DataFrame([[
        datetime.now(),
        input_data["Pregnancies"].values[0],
        input_data["Glucose"].values[0],
        input_data["Insulin"].values[0],
        input_data["BMI"].values[0],
        input_data["Age"].values[0],
        pred_v1,
        pred_v2,
        latency,
        feedback_score,
        feedback_comment
    ]], columns=[
        "timestamp", "Pregnancies", "Glucose", "Insulin", "BMI", "Age",
        "prediction_v1", "prediction_v2", "latency", "feedback_score", "feedback_comment"
    ])

    new_log.to_csv(LOG_FILE, mode="a", index=False, header=False)