"""
Streamlit page for data upload and ingestion (Step 1).
Implements the UI for CSV file upload with the exact specifications required.
"""

import streamlit as st
from pathlib import Path
from pipeline.step_1_ingest import ingest_logic
from common import constants, utils
from common.schemas import StageStatus
from common.storage import read_json


def show_upload_page():
    """Display the data upload page with exact specifications."""
    
    # Page Title
    st.title("Step 1: Upload Your Data")
    
    # Display current run ID if available
    if 'run_id' in st.session_state:
        st.info(f"**Current Run ID:** {st.session_state['run_id']}")
    
    # Introductory text
    st.write("Please upload your dataset in CSV format to begin the analysis pipeline.")
    
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
                        # Clear processing state immediately
                        st.session_state['processing_file'] = False
                        
                        # Display results section
                        st.success(f"File uploaded successfully! Run ID: {run_id}")
                        st.info(f"**Current Run ID:** {run_id}")
                        st.caption(f"Data saved to data/runs/{run_id}/{constants.ORIGINAL_DATA_FILE}")
                        
                        # Trigger rerun to refresh page state after processing
                        st.rerun()
                    
                    else:
                        # Construct comprehensive error message
                        error_message = f"File upload or initial processing failed for Run ID: {run_id}."
                        if status_data:
                            status_obj = StageStatus(**status_data)
                            if status_obj.message:
                                error_message += f" Details: {status_obj.message}"
                            if status_obj.errors:
                                error_message += f" Specific errors: {', '.join(status_obj.errors)}"
                        
                        # Use standardized error display
                        display_exception = Exception(error_message)
                        is_dev_mode = st.session_state.get("developer_mode_active", False)
                        utils.display_page_error(display_exception, run_id=run_id, stage_name=constants.INGEST_STAGE, dev_mode=is_dev_mode)
                        
                        # Clear processing state on error
                        st.session_state['processing_file'] = False
                        
                except Exception as e:
                    # Use standardized error display
                    is_dev_mode = st.session_state.get("developer_mode_active", False)
                    utils.display_page_error(e, run_id=st.session_state.get('run_id'), stage_name=constants.INGEST_STAGE, dev_mode=is_dev_mode)
                    
                    # Clear processing state on exception
                    st.session_state['processing_file'] = False
                    st.session_state['last_processed_file'] = None  # Allow retry on error
                    
        elif st.session_state['processing_file']:
            st.info("File is currently being processed. Please wait...")
        elif st.session_state['last_processed_file'] == file_id and 'run_id' in st.session_state:
            # File already processed successfully
            st.success(f"File already processed! Run ID: {st.session_state['run_id']}")
            st.info(f"**Current Run ID:** {st.session_state['run_id']}")
            st.caption(f"Data saved to data/runs/{st.session_state['run_id']}/{constants.ORIGINAL_DATA_FILE}")
    
    # Show "Continue to Next Step" button if file has been processed successfully
    if (uploaded_file is not None and 
        'run_id' in st.session_state and 
        st.session_state.get('last_processed_file') is not None and
        not st.session_state.get('processing_file', False)):
        
        # Check if this file matches the last processed file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}_{hash(uploaded_file.getvalue())}"
        if st.session_state['last_processed_file'] == file_id:
            if st.button("➡️ Continue to Target Confirmation", type="primary", use_container_width=True, key="continue_to_target"):
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