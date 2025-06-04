# app/wizardry/step9_outputs.py
import streamlit as st
<<<<<<< Updated upstream
=======
import pandas as pd
import joblib
import os
import json
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import tempfile

def get_google_sheets_client():
    """Initialize and return a Google Sheets client"""
    try:
        # Define the scope
        scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']
        
        # Check for credentials file
        creds_path = os.path.join('credentials', 'google_credentials.json')
        if not os.path.exists(creds_path):
            st.warning("""
            Google Sheets credentials not found. To enable Google Sheets export:
            1. Go to Google Cloud Console (https://console.cloud.google.com/)
            2. Create a new project or select an existing one
            3. Enable Google Sheets API and Google Drive API
            4. Create a service account and download the JSON key
            5. Save the key as 'google_credentials.json' in the 'credentials' directory
            """)
            return None
        
        # Create credentials
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            creds_path, scope)
        
        # Authorize the client
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"Error initializing Google Sheets client: {str(e)}")
        return None

def export_to_google_sheets(df, sheet_name):
    """Export DataFrame to a new Google Sheet"""
    try:
        client = get_google_sheets_client()
        if client is None:
            return None
        
        # Create a new spreadsheet
        spreadsheet = client.create(sheet_name)
        
        # Get the first worksheet
        worksheet = spreadsheet.get_worksheet(0)
        
        # Update the worksheet with the DataFrame
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())
        
        # Make the spreadsheet publicly accessible
        spreadsheet.share(None, perm_type='anyone', role='reader')
        
        return spreadsheet.url
    except Exception as e:
        st.error(f"Error exporting to Google Sheets: {str(e)}")
        return None

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
>>>>>>> Stashed changes

def run():
    st.header("📦 Step 9: Final Outputs to User")

    st.write("🎁 Final Deliverables:")
    
    if "predictions" in st.session_state:
        st.download_button("📄 Download Predictions CSV",
                           data=st.session_state["predictions"].to_csv(index=False),
                           file_name="final_predictions.csv")

    if "clean_data" in st.session_state:
        st.download_button("🧼 Download Cleaned Data CSV",
                           data=st.session_state["clean_data"].to_csv(index=False),
                           file_name="cleaned_data.csv")

    st.write("📁 Model File: (simulated)")
    st.download_button("💾 Download Model File", data="BinaryDataPlaceholder", file_name="model.pkl")

<<<<<<< Updated upstream
    st.write("📊 Explainability Artifacts:")
    st.text(st.session_state.get("explanations", "None available"))
=======
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
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📄 Download Cleaned Dataset",
                    df.to_csv(index=False),
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                if st.button("📊 Export to Google Sheets"):
                    with st.spinner("Exporting to Google Sheets..."):
                        sheet_name = f"ProjectionWizard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        sheet_url = export_to_google_sheets(df, sheet_name)
                        if sheet_url:
                            st.success("✅ Data exported successfully!")
                            st.markdown(f"[Open Google Sheet]({sheet_url})")
                        else:
                            st.error("❌ Failed to export to Google Sheets")

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
>>>>>>> Stashed changes

    # Add back button at the bottom
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("⬅️ Back: Explainability", use_container_width=True):
            st.session_state.current_step = "Step 8: Explainability"
            st.rerun()
