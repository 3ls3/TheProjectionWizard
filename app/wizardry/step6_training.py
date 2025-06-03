import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(df, target_col=None):
    """Preprocess the data for training"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Handle categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols)
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Separate features and target if target column is provided
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None
    
    return X, y

def run():
    st.header("🎯 Step 6: Model Training")

    # Get required data from session state
    task = st.session_state.get("task_type")
    candidate_models = st.session_state.get("candidate_models")
    df = st.session_state.get("clean_data")
    target_col = st.session_state.get("target_column")

    if not all([task, candidate_models, df]):
        st.warning("Please complete model selection first.")
        return

    # Preprocess the data
    X, y = preprocess_data(df, target_col)
    
    # Split the data
    test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
    random_state = st.number_input("Random state", value=42)
    
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        y_train, y_test = None, None

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store scaler in session state for later use
    st.session_state["scaler"] = scaler

    # Train models
    trained_models = {}
    model_metrics = {}
    
    with st.spinner("Training models..."):
        for model_name in candidate_models:
            try:
                # Import and initialize model
                if model_name == "Logistic Regression":
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(random_state=random_state)
                elif model_name == "Random Forest":
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    model = RandomForestClassifier(random_state=random_state) if task != "regression" else RandomForestRegressor(random_state=random_state)
                elif model_name == "XGBoost":
                    import xgboost as xgb
                    model = xgb.XGBClassifier(random_state=random_state) if task != "regression" else xgb.XGBRegressor(random_state=random_state)
                elif model_name == "LightGBM":
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(random_state=random_state) if task != "regression" else lgb.LGBMRegressor(random_state=random_state)
                elif model_name == "CatBoost":
                    from catboost import CatBoostClassifier, CatBoostRegressor
                    model = CatBoostClassifier(random_state=random_state) if task != "regression" else CatBoostRegressor(random_state=random_state)
                elif model_name == "MLP":
                    from sklearn.neural_network import MLPClassifier, MLPRegressor
                    model = MLPClassifier(random_state=random_state) if task != "regression" else MLPRegressor(random_state=random_state)
                elif model_name == "Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                elif model_name == "ElasticNet":
                    from sklearn.linear_model import ElasticNet
                    model = ElasticNet(random_state=random_state)
                elif model_name == "KMeans":
                    from sklearn.cluster import KMeans
                    model = KMeans(n_clusters=3, random_state=random_state)
                elif model_name == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    model = DBSCAN()
                elif model_name == "Agglomerative":
                    from sklearn.cluster import AgglomerativeClustering
                    model = AgglomerativeClustering(n_clusters=3)
                else:
                    st.error(f"Unknown model: {model_name}")
                    continue

                # Train the model
                if y_train is not None:
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train_scaled)

                # Store the trained model
                trained_models[model_name] = model

                # Calculate metrics
                if y_test is not None:
                    if task == "regression":
                        from sklearn.metrics import mean_squared_error, r2_score
                        y_pred = model.predict(X_test_scaled)
                        metrics = {
                            "MSE": mean_squared_error(y_test, y_pred),
                            "R2": r2_score(y_test, y_pred)
                        }
                    else:
                        from sklearn.metrics import accuracy_score, f1_score
                        y_pred = model.predict(X_test_scaled)
                        metrics = {
                            "Accuracy": accuracy_score(y_test, y_pred),
                            "F1 Score": f1_score(y_test, y_pred, average='weighted')
                        }
                    model_metrics[model_name] = metrics

            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
                continue

    # Store trained models in session state
    st.session_state["trained_models"] = trained_models

    # Display results
    st.subheader("Training Results")
    
    if model_metrics:
        # Convert metrics to DataFrame for better display
        metrics_df = pd.DataFrame(model_metrics).T
        st.write("Model Performance Metrics:")
        st.dataframe(metrics_df)
        
        # Find best model
        if task == "regression":
            best_model = min(model_metrics.items(), key=lambda x: x[1]["MSE"])[0]
        else:
            best_model = max(model_metrics.items(), key=lambda x: x[1]["Accuracy"])[0]
        
        st.success(f"Best performing model: {best_model}")
        st.session_state["best_model"] = best_model

    # Add navigation buttons at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅️ Back: Model Selection", use_container_width=True):
            st.session_state.current_step = "Step 5: Model Selection"
            st.rerun()
    
    with col3:
        if st.button("Next: Inference ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = "Step 7: Inference"
            st.rerun()
