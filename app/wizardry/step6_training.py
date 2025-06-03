# app/wizardry/step6_training.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
import os
import importlib
import sys

def get_model_class(model_name):
    """Dynamically import and return the model class"""
    model_mapping = {
        "Logistic Regression": "sklearn.linear_model.LogisticRegression",
        "Random Forest": "sklearn.ensemble.RandomForestClassifier",
        "XGBoost": "xgboost.XGBClassifier",
        "LightGBM": "lightgbm.LGBMClassifier",
        "CatBoost": "catboost.CatBoostClassifier",
        "MLP": "sklearn.neural_network.MLPClassifier",
        "Linear Regression": "sklearn.linear_model.LinearRegression",
        "ElasticNet": "sklearn.linear_model.ElasticNet",
        "KMeans": "sklearn.cluster.KMeans",
        "DBSCAN": "sklearn.cluster.DBSCAN",
        "Agglomerative": "sklearn.cluster.AgglomerativeClustering"
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unknown model: {model_name}")
    
    module_path, class_name = model_mapping[model_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def preprocess_data(df, target_col=None, is_training=False):
    """Preprocess the data for training or prediction"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Remove target column if it exists
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])
    
    # Handle categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols)
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 0:
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler

def train_model(model_name, X, y, task_type):
    """Train a model and return it along with its performance metrics"""
    try:
        # Get the model class
        model_class = get_model_class(model_name)
        
        # Initialize the model with default parameters
        model = model_class()
        
        # Train the model
        model.fit(X, y)
        
        # Get cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        # Calculate performance metrics
        y_pred = model.predict(X)
        if task_type in ['regression', 'time_series_forecasting']:
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'r2': r2_score(y, y_pred),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        else:
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        
        return model, metrics
    
    except Exception as e:
        st.error(f"Error training {model_name}: {str(e)}")
        return None, None

def run():
    st.header("🏋️ Step 6: Model Training & Tuning")

    tool = st.session_state.get("automl_tool")
    models = st.session_state.get("candidate_models")
    task = st.session_state.get("task_type")
    df = st.session_state.get("clean_data")
    target_col = st.session_state.get("target_column")

    # Check if any required variables are missing
    required_vars = {
        "AutoML Tool": tool,
        "Candidate Models": models,
        "Task Type": task,
        "Dataset": df,
        "Target Column": target_col
    }
    
    missing_vars = [name for name, value in required_vars.items() if value is None]
    if missing_vars:
        st.warning(f"Missing configuration: {', '.join(missing_vars)}. Complete previous steps first.")
        return

    st.write(f"Training models for task: `{task}` using `{tool}`...")
    st.write("Selected candidate models:", models)

    # Preprocess the data
    processed_df, scaler = preprocess_data(df, target_col, is_training=True)
    
    # Store the scaler in session state for later use
    st.session_state["scaler"] = scaler
    
    # Prepare features and target
    X = processed_df  # No need to drop target_col as it's already removed in preprocessing
    y = df[target_col] if target_col in df.columns else None

    # Train models and collect results
    leaderboard = []
    trained_models = {}

    with st.spinner("Training models..."):
        for model_name in models:
            model, metrics = train_model(model_name, X, y, task)
            if model is not None and metrics is not None:
                trained_models[model_name] = model
                leaderboard.append({
                    "model": model_name,
                    "metrics": metrics
                })

    # Sort leaderboard by appropriate metric
    if task in ['regression', 'time_series_forecasting']:
        leaderboard.sort(key=lambda x: x['metrics']['r2'], reverse=True)
    else:
        leaderboard.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)

    # Store results in session state
    st.session_state["leaderboard"] = leaderboard
    st.session_state["trained_models"] = trained_models
    st.session_state["best_model"] = leaderboard[0]["model"] if leaderboard else None

<<<<<<< Updated upstream
    st.success(f"Best model: {leaderboard[0]['model']} (score: {leaderboard[0]['score']})")
