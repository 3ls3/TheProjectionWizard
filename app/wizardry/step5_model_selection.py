# app/wizardry/step5_model_selection.py
import streamlit as st

def run():
    st.header("🤖 Step 5: Model Selection & AutoML")

    task = st.session_state.get("task_type")
    if task is None:
        st.warning("Please detect the task type first.")
        return

    st.write(f"Detected Task: `{task}`")

    tool = st.selectbox("Choose AutoML tool", ["PyCaret", "AutoGluon", "FLAML"])
    st.session_state["automl_tool"] = tool

    model_dict = {
        "binary_classification": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "MLP"],
        "multi_class_classification": ["Random Forest", "XGBoost", "CatBoost", "LightGBM"],
        "multi_label_classification": ["Binary Relevance + XGBoost", "MultiOutputClassifier"],
        "regression": ["Linear Regression", "ElasticNet", "XGBoost", "LightGBM", "MLP"],
        "clustering": ["KMeans", "DBSCAN", "Agglomerative"],
        "time_series_forecasting": ["Prophet", "ARIMA", "LSTM", "TFT", "AutoGluon-TS"]
    }

    candidates = model_dict.get(task, [])
    selected_models = st.multiselect("Select candidate models", candidates, default=candidates[:2])
    st.session_state["candidate_models"] = selected_models
