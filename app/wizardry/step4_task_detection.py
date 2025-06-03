# app/wizardry/step4_task_detection.py
import streamlit as st
import pandas as pd

def run():
    st.header("🔍 Step 4: Task Detection")

    uploaded_data = st.session_state.get("clean_data")  # Assumes data from earlier step
    if uploaded_data is None:
        st.warning("Please complete data cleaning first.")
        return

    df = uploaded_data
    st.write("Preview of uploaded dataset:")
    st.dataframe(df.head())

    target_column = st.selectbox("Select target column (or None if unsupervised)", options=["None"] + list(df.columns))

    if target_column == "None":
        st.success("Task Type Detected: Clustering (Unsupervised)")
        st.session_state["task_type"] = "clustering"
        return

    target_series = df[target_column]

    # Task detection logic
    if pd.api.types.is_numeric_dtype(target_series):
        st.success("Task Type Detected: Regression")
        task = "regression"
    elif target_series.nunique() == 2:
        st.success("Task Type Detected: Binary Classification")
        task = "binary_classification"
    elif target_series.nunique() > 2 and target_series.apply(lambda x: isinstance(x, list)).any():
        st.success("Task Type Detected: Multi-Label Classification")
        task = "multi_label_classification"
    elif target_series.nunique() > 2:
        st.success("Task Type Detected: Multi-Class Classification")
        task = "multi_class_classification"
    elif any("date" in col.lower() for col in df.columns):
        st.success("Task Type Detected: Time Series Forecasting")
        task = "time_series_forecasting"
    else:
        st.error("Unable to confidently detect task type.")
        return

    st.session_state["task_type"] = task
    st.session_state["target_column"] = target_column
