# monitor_dashboard.py
import streamlit as st
import pandas as pd

# Load logs
if not st.sidebar.checkbox("Hide logs", value=False):
    df = pd.read_csv("monitoring_logs.csv")
else:
    df = pd.DataFrame()

st.title("Model Monitoring Dashboard")

if df.empty:
    st.warning("No logs available. Submit some predictions first!")
else:
    # 1. Summary metrics
    st.subheader("Summary Metrics")
    avg_latency = df["latency"].mean()
    avg_feedback = df["feedback_score"].mean()
    st.metric("Average Latency (ms)", round(avg_latency,2))
    st.metric("Average Feedback Score", round(avg_feedback,2))

    # 2. Model comparison
    st.subheader("Predictions Comparison")
    v1_count = df["prediction_v1"].value_counts()
    v2_count = df["prediction_v2"].value_counts()
    comp_df = pd.DataFrame({"Model V1": v1_count, "Model V2": v2_count}).fillna(0).astype(int)
    st.bar_chart(comp_df)

     # 3. Recent comments
    st.subheader("Recent Feedback Comments")
    st.table(df[["timestamp", "feedback_score", "feedback_comment"]].tail(10))

     # 4. Raw logs
    st.subheader("Raw Monitoring Logs")
    st.dataframe(df)