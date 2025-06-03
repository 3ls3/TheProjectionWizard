# app/wizardry/step7_inference.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_input_data(df, scaler, target_col=None):
    """Preprocess input data using the same scaler from training"""
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
    
    # Scale numerical features using the same scaler from training
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 0:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

def run():
    st.header("🔮 Step 7: Inference")

    # Get required data from session state
    task = st.session_state.get("task_type")
    best_model_name = st.session_state.get("best_model")
    trained_models = st.session_state.get("trained_models")
    scaler = st.session_state.get("scaler")
    df = st.session_state.get("clean_data")
    target_col = st.session_state.get("target_column")

    # Check if any required variables are missing
    required_vars = {
        "Task Type": task,
        "Best Model": best_model_name,
        "Trained Models": trained_models,
        "Scaler": scaler,
        "Dataset": df
    }
    
    missing_vars = [name for name, value in required_vars.items() if value is None]
    if missing_vars:
        st.warning(f"Missing configuration: {', '.join(missing_vars)}. Complete previous steps first.")
        return

    st.write(f"Using best model: {best_model_name}")
    model = trained_models[best_model_name]

    # Display the data that will be used for inference
    st.subheader("Input Data")
    st.write("Using the dataset from Step 4:")
    st.dataframe(df.head())

    # Prediction section
    if st.button("Make Predictions", type="primary"):
        try:
            # Preprocess the input data
            processed_data = preprocess_input_data(df, scaler, target_col)
            
            # Make predictions
            predictions = model.predict(processed_data)
            
            # For classification tasks, also get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_data)
            
            # Display results
            st.subheader("Predictions")
            
            # Create a results dataframe
            results = pd.DataFrame()
            results['Prediction'] = predictions
            
            if hasattr(model, 'predict_proba'):
                # Add probability columns for each class
                if task == 'binary_classification':
                    results['Probability'] = probabilities[:, 1]
                else:
                    for i in range(probabilities.shape[1]):
                        results[f'Class_{i}_Probability'] = probabilities[:, i]
            
            # Add predictions to the original dataframe
            df_with_predictions = df.copy()
            df_with_predictions['Prediction'] = predictions
            if hasattr(model, 'predict_proba'):
                if task == 'binary_classification':
                    df_with_predictions['Probability'] = probabilities[:, 1]
                else:
                    for i in range(probabilities.shape[1]):
                        df_with_predictions[f'Class_{i}_Probability'] = probabilities[:, i]
            
            # Display results
            st.write("Predictions added to original data:")
            st.dataframe(df_with_predictions)
            
            # Add download button for results
            csv = df_with_predictions.to_csv(index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
            # Store predictions in session state for later use
            st.session_state["predictions"] = df_with_predictions
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            # Add more detailed error information
            if hasattr(model, 'feature_names_in_'):
                st.write("Model was trained with these features:")
                st.write(model.feature_names_in_)
                st.write("Current data has these features:")
                st.write(processed_data.columns.tolist())

    # Add navigation buttons at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅️ Back: Training", use_container_width=True):
            st.session_state.current_step = "Step 6: Training"
            st.rerun()
    
    with col3:
        if st.button("Next: Explainability ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = "Step 8: Explainability"
            st.rerun()
