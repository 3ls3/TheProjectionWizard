"""
Streamlit page for data upload and ingestion (Step 1).
Implements the UI for CSV file upload with the exact specifications required.
"""

import streamlit as st
from pathlib import Path
from step_1_ingest import ingest_logic
from common import constants
from common.schemas import StageStatus
from common.storage import read_json


def show_upload_page():
    """Display the data upload page with exact specifications."""
    
    # Page Title
    st.title("Step 1: Upload Your Data")
    
    # File Uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    # Processing Logic
    if uploaded_file is not None:
        # Display a spinner
        with st.spinner("Processing uploaded file..."):
            # Construct base_runs_path
            base_runs_path = str(Path(constants.DATA_DIR_NAME) / constants.RUNS_DIR_NAME)
            
            # Call run_ingestion
            run_id = ingest_logic.run_ingestion(uploaded_file, base_runs_path)
            
            # Store run_id in session state
            st.session_state['run_id'] = run_id
            
            # Attempt to read the status.json for this run to confirm success
            status_data = read_json(run_id, constants.STATUS_FILE)
            
            if status_data and StageStatus(**status_data).status == 'completed':
                st.success(f"File uploaded successfully! Run ID: {run_id}")
                st.info(f"Data saved to data/runs/{run_id}/{constants.ORIGINAL_DATA_FILE}")
                
                # Display a button/link to navigate to the next step
                if st.button("Proceed to Target Confirmation"):
                    st.session_state['current_page'] = 'target_confirmation'
                    st.rerun()
            
            else:
                st.error(f"File upload or initial processing failed for Run ID: {run_id}. Check logs in data/runs/{run_id}/{constants.PIPELINE_LOG_FILE}.")
                
                # Display error message from status.json if available
                if status_data:
                    status_obj = StageStatus(**status_data)
                    if status_obj.message:
                        st.error(f"Error details: {status_obj.message}")
                    if status_obj.errors:
                        st.error("Specific errors:")
                        for error in status_obj.errors:
                            st.code(error)


def main():
    """Main function to run the upload page."""
    # Display Current Run ID (if exists)
    if 'run_id' in st.session_state:
        st.sidebar.info(f"Current Run ID: {st.session_state['run_id']}")
    
    # Show the upload page
    show_upload_page()


if __name__ == "__main__":
    # Configure Streamlit page
    st.set_page_config(
        page_title="Step 1: Upload Your Data - Projection Wizard",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    main() 