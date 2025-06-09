"""
General utility functions for The Projection Wizard.
"""

import uuid
from datetime import datetime
from typing import Optional
from pathlib import Path

import streamlit as st

from . import constants, storage


def generate_run_id() -> str:
    """
    Generate a unique run ID combining ISO 8601 timestamp (UTC) and short UUID.
    
    Format: YYYY-MM-DDTHH-MM-SSZ_shortUUID
    Example: 2025-06-07T103045Z_a1b2c3d4
    
    Returns:
        Unique run identifier string
    """
    # Get current UTC timestamp in ISO 8601 format with custom formatting
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    
    # Generate short UUID (first 8 characters)
    short_uuid = uuid.uuid4().hex[:8]
    
    return f"{timestamp}_{short_uuid}"


def display_page_error(
    exception_object: Exception, 
    run_id: Optional[str] = None, 
    stage_name: Optional[str] = None, 
    dev_mode: bool = False
) -> None:
    """
    Display error messages in a standardized way across all Streamlit pages.
    
    Provides user-friendly error messages for regular users and detailed technical
    information (stack traces, log file paths) when developer mode is enabled.
    
    Args:
        exception_object: The actual exception instance that was caught
        run_id: Current run ID, used for constructing log file paths (optional)
        stage_name: Pipeline stage where error occurred (e.g., constants.VALIDATION_STAGE) (optional)
        dev_mode: Whether developer mode is active, controls display of technical details
        
    Returns:
        None - displays error information directly in Streamlit UI
        
    Examples:
        Basic usage:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     display_page_error(e)
        
        With stage and run context:
        >>> try:
        ...     validate_data()
        ... except Exception as e:
        ...     display_page_error(e, run_id="2025-01-15T120000Z_abc123", 
        ...                       stage_name=constants.VALIDATION_STAGE, dev_mode=True)
    """
    # Construct user-friendly error message
    if stage_name:
        # Convert stage name to human-readable format (e.g., "step_1_ingest" -> "Step 1 Ingest")
        readable_stage = stage_name.replace('_', ' ').title()
        user_message = f"An error occurred during the {readable_stage} step: {str(exception_object)}"
    else:
        user_message = f"An unexpected error occurred: {str(exception_object)}"
    
    # Display the user-friendly error message
    st.error(user_message)
    
    # Show detailed technical information if developer mode is enabled
    if dev_mode:
        st.exception(exception_object)
    
    # Display log file paths for developers
    _display_log_file_info(run_id, stage_name)


def _display_log_file_info(run_id: Optional[str], stage_name: Optional[str]) -> None:
    """
    Display information about relevant log files for debugging.
    
    Args:
        run_id: Current run ID for constructing file paths
        stage_name: Pipeline stage name for stage-specific logs
        
    Returns:
        None - displays log file information directly in Streamlit UI
    """
    if not run_id:
        return
    
    try:
        # Try to display stage-specific log file path first
        if stage_name and stage_name in constants.STAGE_LOG_FILENAMES:
            stage_log_filename = constants.STAGE_LOG_FILENAMES[stage_name]
            stage_log_path = storage.get_run_dir(run_id) / stage_log_filename
            
            st.info(f"📝 **For developers:** Check the stage log file: `{stage_log_path}`")
            
        # Fallback to general pipeline log file
        else:
            pipeline_log_path = storage.get_run_dir(run_id) / constants.PIPELINE_LOG_FILENAME
            st.info(f"📝 **For developers:** Check the pipeline log file: `{pipeline_log_path}`")
            
    except Exception as log_error:
        # Don't let log path construction errors interfere with main error display
        # Just silently skip log file info if there's an issue
        pass 