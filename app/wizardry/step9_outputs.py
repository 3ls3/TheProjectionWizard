# app/wizardry/step9_outputs.py
import streamlit as st
import pandas as pd
import joblib
import os
import json
from datetime import datetime

def save_model_artifacts(model, scaler, task_type, model_name):
    """Save model artifacts to a temporary directory"""
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    
    # Save model
    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = os.path.join("artifacts", "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        "task_type": task_type,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "feature_names": model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else None
    }
    
    metadata_path = os.path.join("artifacts", "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_path, scaler_path, metadata_path

def run():
    st.header("📦 Step 9: Final Outputs")

    # Get required data from session state
    task = st.session_state.get("task_type")
    best_model_name = st.session_state.get("best_model")
    trained_models = st.session_state.get("trained_models")
    scaler = st.session_state.get("scaler")
    df = st.session_state.get("clean_data")
    target_col = st.session_state.get("target_column")
    explanations = st.session_state.get("explanations")

    if not all([task, best_model_name, trained_models, scaler]):
        st.warning("Please complete the training step first.")
        return

    st.write("🎁 Final Deliverables:")

    # Save and provide model artifacts
    try:
        model = trained_models[best_model_name]
        model_path, scaler_path, metadata_path = save_model_artifacts(
            model, scaler, task, best_model_name
        )

        # Create a zip file containing all artifacts
        import zipfile
        zip_path = os.path.join("artifacts", "model_artifacts.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(model_path, "model.pkl")
            zipf.write(scaler_path, "scaler.pkl")
            zipf.write(metadata_path, "metadata.json")

        # Download buttons for individual files
        st.subheader("📁 Model Files")
        with open(zip_path, 'rb') as f:
            st.download_button(
                "💾 Download All Model Artifacts (ZIP)",
                f,
                file_name="model_artifacts.zip",
                mime="application/zip"
            )

        # Data downloads
        st.subheader("📊 Data Files")
        if df is not None:
            st.download_button(
                "📄 Download Cleaned Dataset",
                df.to_csv(index=False),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

        # Explanation artifacts
        st.subheader("🧠 Model Explanations")
        if explanations:
            if explanations.get("shap_values") is not None:
                st.write("SHAP values are available in the model artifacts")
            if explanations.get("lime_explainer") is not None:
                st.write("LIME explainer is available in the model artifacts")

        # Model information
        st.subheader("ℹ️ Model Information")
        st.write(f"Task Type: {task}")
        st.write(f"Model: {best_model_name}")
        if hasattr(model, 'feature_names_in_'):
            st.write("Features used:")
            st.write(model.feature_names_in_.tolist())

    except Exception as e:
        st.error(f"Error saving model artifacts: {str(e)}")

    # Add back button at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("⬅️ Back: Explainability", use_container_width=True):
            st.session_state.current_step = "Step 8: Explainability"
            st.rerun()
