"""
Streamlit page for data upload and ingestion (Step 1).
Implements the UI for CSV file upload with the exact specifications required.
"""

import streamlit as st
from pathlib import Path
from pipeline.step_1_ingest import ingest_logic
from common import constants
from common.schemas import StageStatus
from common.storage import read_json


def show_upload_page():
    """Display the data upload page with exact specifications."""
    
    # Page Title
    st.title("Step 1: Upload Your Data")
    
    # Initialize processing state in session if not exists
    if 'processing_file' not in st.session_state:
        st.session_state['processing_file'] = False
    if 'last_processed_file' not in st.session_state:
        st.session_state['last_processed_file'] = None
    
    # File Uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    # Processing Logic
    if uploaded_file is not None:
        # Create a unique identifier for this file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}_{hash(uploaded_file.getvalue())}"
        
        # Check if this file has already been processed or is currently being processed
        if (not st.session_state['processing_file'] and 
            st.session_state['last_processed_file'] != file_id):
            
            # Mark as processing to prevent duplicate runs
            st.session_state['processing_file'] = True
            st.session_state['last_processed_file'] = file_id
            
            # Display a spinner
            with st.spinner("Processing uploaded file..."):
                try:
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
                        
                        # Auto-navigate to next step immediately
                        st.success("🚀 Proceeding to Target Confirmation...")
                        st.session_state['current_page'] = 'target_confirmation'
                        # Clear processing state before rerun
                        st.session_state['processing_file'] = False
                        st.rerun()
                    
                    else:
                        st.error(f"File upload or initial processing failed for Run ID: {run_id}. Check logs in data/runs/{run_id}/{constants.STAGE_LOG_FILENAMES[constants.INGEST_STAGE]}.")
                        
                        # Display error message from status.json if available
                        if status_data:
                            status_obj = StageStatus(**status_data)
                            if status_obj.message:
                                st.error(f"Error details: {status_obj.message}")
                            if status_obj.errors:
                                st.error("Specific errors:")
                                for error in status_obj.errors:
                                    st.code(error)
                        
                        # Clear processing state on error
                        st.session_state['processing_file'] = False
                        
                except Exception as e:
                    st.error(f"An unexpected error occurred during file processing: {str(e)}")
                    # Clear processing state on exception
                    st.session_state['processing_file'] = False
                    st.session_state['last_processed_file'] = None  # Allow retry on error
                    
        elif st.session_state['processing_file']:
            st.info("File is currently being processed. Please wait...")
        elif st.session_state['last_processed_file'] == file_id and 'run_id' in st.session_state:
            # File already processed successfully
            st.success(f"File already processed! Run ID: {st.session_state['run_id']}")
            st.info(f"Data saved to data/runs/{st.session_state['run_id']}/{constants.ORIGINAL_DATA_FILE}")
            
            # Show button to proceed to next step
            if st.button("🚀 Proceed to Target Confirmation"):
                st.session_state['current_page'] = 'target_confirmation'
                st.rerun()


def main():
    """Main function to run the upload page."""
    # Note: When run from main app.py, the sidebar is handled centrally
    # This main() function is primarily for standalone testing
    
    # Display Current Run ID (if exists) - only when running standalone
    if 'run_id' in st.session_state and not hasattr(st, '_is_running_from_main_app'):
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