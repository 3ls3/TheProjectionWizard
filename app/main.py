"""
Main Streamlit application for The Projection Wizard.
Entry point for the ML pipeline interface with multi-page navigation.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root and app directory to path
sys.path.append(str(Path(__file__).parent.parent))  # Project root
sys.path.append(str(Path(__file__).parent))  # App directory

import importlib

# Import common utilities
from common import status_utils, constants

# Import page modules
upload_page_module = importlib.import_module('pages.01_upload_page')
target_page_module = importlib.import_module('pages.02_target_page')
schema_page_module = importlib.import_module('pages.03_schema_page')
validation_page_module = importlib.import_module('pages.04_validation_page')
prep_page_module = importlib.import_module('pages.05_prep_page')
automl_page_module = importlib.import_module('pages.06_automl_page')
explain_page_module = importlib.import_module('pages.07_explain_page')
results_page_module = importlib.import_module('pages.08_results_page')


def get_page_config():
    """Get the Streamlit page configuration."""
    return {
        "page_title": "The Projection Wizard",
        "page_icon": "🔮",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }


def clear_run_session_state():
    """
    Clear all run-specific session state variables when starting a new run.
    
    This helper function ensures consistent cleanup of session state across
    different parts of the application (Upload Data confirmation, Start New Run button).
    """
    # Core run data
    st.session_state.pop('run_id', None)
    st.session_state.pop('current_page', None)
    
    # UI-specific overrides and confirmations
    st.session_state.pop('ui_feature_schemas_override', None)
    st.session_state.pop('target_confirmation_complete', None)
    st.session_state.pop('schema_confirmation_complete', None)
    
    # Processing flags and temporary states
    st.session_state.pop('validation_running', None)
    st.session_state.pop('prep_running', None)
    st.session_state.pop('automl_running', None)
    st.session_state.pop('explain_running', None)
    
    # Upload-related temporary data
    st.session_state.pop('uploaded_file_content', None)
    st.session_state.pop('processing_file', None)
    st.session_state.pop('last_processed_file', None)
    
    # Clear any confirmation states
    st.session_state.pop('confirm_new_upload', None)


def show_navigation_sidebar():
    """Display navigation options in the sidebar."""
    st.sidebar.title("🔮 Projection Wizard")
    
    # Define unified page configuration for navigation (available to both branches)
    current_page = st.session_state.get('current_page', 'upload')
    pages_config = [
        {'id': 'upload', 'name': 'Upload Data'},
        {'id': 'target_confirmation', 'name': 'Target Confirmation'},
        {'id': 'schema_confirmation', 'name': 'Schema Confirmation'}, 
        {'id': 'validation', 'name': 'Data Validation'},
        {'id': 'prep', 'name': 'Data Preparation'},
        {'id': 'automl', 'name': 'Model Training'},
        {'id': 'explain', 'name': 'Model Explanation'},
        {'id': 'results', 'name': 'Results'}
    ]
    
    # Show current run info if available
    if 'run_id' in st.session_state:
        run_id = st.session_state['run_id']
        st.sidebar.success(f"**Active Run:** {run_id}")
        
        # Get stage status summaries
        validation_summary = status_utils.get_stage_status_summary(run_id, constants.VALIDATION_STAGE)
        prep_summary = status_utils.get_stage_status_summary(run_id, constants.PREP_STAGE)
        automl_summary = status_utils.get_stage_status_summary(run_id, constants.AUTOML_STAGE)
        explain_summary = status_utils.get_stage_status_summary(run_id, constants.EXPLAIN_STAGE)
        
        # Show critical failure banner if validation failed
        if validation_summary.status == "failed_critically":
            st.sidebar.error("🚫 **Pipeline Stopped**")
            st.sidebar.error(validation_summary.message or "Validation failed - fix data issues")
        
        current_index = next((i for i, p in enumerate(pages_config) if p['id'] == current_page), 0)
        validation_failed = validation_summary.status == "failed_critically"
        validation_index = next((i for i, p in enumerate(pages_config) if p['id'] == 'validation'), 3)
    else:
        # No run_id - provide default values for status summaries
        validation_summary = status_utils.StageStatusSummary(status="pending", message="No run active", can_proceed=False)
        prep_summary = status_utils.StageStatusSummary(status="pending", message="No run active", can_proceed=False)
        automl_summary = status_utils.StageStatusSummary(status="pending", message="No run active", can_proceed=False)
        explain_summary = status_utils.StageStatusSummary(status="pending", message="No run active", can_proceed=False)
        
        current_index = 0
        validation_failed = False
        validation_index = next((i for i, p in enumerate(pages_config) if p['id'] == 'validation'), 3)
    
    st.sidebar.divider()
    
    # Unified Pipeline Steps (Navigation + Progress)
    st.sidebar.subheader("Pipeline Steps")
    
    # Create unified navigation buttons with progress emojis
    if 'run_id' in st.session_state:
        # Run exists - show all buttons with appropriate status
        for i, page_config in enumerate(pages_config):
            page_id = page_config['id']
            page_name = page_config['name']
            
            # Determine emoji based on progress logic
            if page_id == current_page:
                emoji = "👉"
            elif page_id == 'validation' and validation_failed:
                emoji = "🚫"
            elif validation_failed and i > validation_index:
                emoji = "🚫"  # Subsequent stages blocked if validation failed
            elif i < current_index:
                emoji = "✅"  # Completed steps
            else:
                emoji = "⏳"  # Pending steps
            
            button_label = f"{emoji} {i+1}. {page_name}"
            
            # Determine disabled state and help text for each page
            is_disabled = False
            help_text = None
            
            if page_id == 'upload':
                # Upload button - special confirmation handling
                if st.sidebar.button(button_label, key=f"nav_{page_id}", use_container_width=True):
                    if 'run_id' in st.session_state:
                        # Active run exists - request confirmation
                        st.session_state['confirm_new_upload'] = True
                        st.rerun()
                    else:
                        # No active run - proceed directly
                        st.session_state['current_page'] = 'upload'
                        st.rerun()
                        
            elif page_id in ['target_confirmation', 'schema_confirmation', 'validation']:
                # These are available after upload with run_id
                if st.sidebar.button(button_label, key=f"nav_{page_id}", use_container_width=True):
                    st.session_state['current_page'] = page_id
                    st.rerun()
                    
            elif page_id == 'prep':
                is_disabled = not validation_summary.can_proceed
                if validation_summary.status == "failed_critically":
                    help_text = validation_summary.message or "❌ Validation critically failed - fix data issues first"
                elif not validation_summary.can_proceed:
                    help_text = "Complete data validation first"
                    
            elif page_id == 'automl':
                is_disabled = not (validation_summary.can_proceed and prep_summary.can_proceed)
                if validation_summary.status == "failed_critically":
                    help_text = validation_summary.message or "❌ Validation critically failed - fix data issues first"
                elif not prep_summary.can_proceed:
                    help_text = "Complete data preparation first"
                elif not validation_summary.can_proceed:
                    help_text = "Complete data validation first"
                    
            elif page_id == 'explain':
                is_disabled = not (validation_summary.can_proceed and prep_summary.can_proceed and automl_summary.can_proceed)
                if validation_summary.status == "failed_critically":
                    help_text = validation_summary.message or "❌ Validation critically failed - fix data issues first"
                elif not automl_summary.can_proceed:
                    help_text = "Complete model training first"
                elif not prep_summary.can_proceed:
                    help_text = "Complete data preparation first"
                elif not validation_summary.can_proceed:
                    help_text = "Complete data validation first"
                    
            elif page_id == 'results':
                is_disabled = not (validation_summary.can_proceed and prep_summary.can_proceed and 
                                 automl_summary.can_proceed and explain_summary.can_proceed)
                if validation_summary.status == "failed_critically":
                    help_text = validation_summary.message or "❌ Validation critically failed - fix data issues first"
                elif not explain_summary.can_proceed:
                    help_text = "Complete model explanation first"
                elif not automl_summary.can_proceed:
                    help_text = "Complete model training first"
                elif not prep_summary.can_proceed:
                    help_text = "Complete data preparation first"
                elif not validation_summary.can_proceed:
                    help_text = "Complete data validation first"
            
            # Render button for prep, automl, explain, results with disabled/help logic
            if page_id in ['prep', 'automl', 'explain', 'results']:
                if is_disabled:
                    st.sidebar.button(button_label, key=f"nav_{page_id}", disabled=True, help=help_text)
                else:
                    if st.sidebar.button(button_label, key=f"nav_{page_id}", use_container_width=True):
                        st.session_state['current_page'] = page_id
                        st.rerun()
    else:
        # No run_id - show all buttons but only upload is enabled
        for i, page_config in enumerate(pages_config):
            page_id = page_config['id']
            page_name = page_config['name']
            emoji = "⏳"  # All pending when no run
            button_label = f"{emoji} {i+1}. {page_name}"
            
            if page_id == 'upload':
                if st.sidebar.button(button_label, key=f"nav_{page_id}", use_container_width=True):
                    st.session_state['current_page'] = 'upload'
                    st.rerun()
            else:
                st.sidebar.button(button_label, key=f"nav_{page_id}", disabled=True, help="Upload data first")
    
    # Show confirmation dialog if requested
    if st.session_state.get('confirm_new_upload', False):
        st.sidebar.warning(f"⚠️ **Confirm New Upload**")
        st.sidebar.warning(f"Starting a new upload will clear your current run (ID: `{st.session_state['run_id']}`). Do you want to proceed?")
        
        col1, col2 = st.sidebar.columns(2)
        if col1.button("✅ Yes, Start New", key="yes_new_upload", use_container_width=True):
            # Clear all run-specific session state
            clear_run_session_state()
            # Navigate to upload page
            st.session_state['current_page'] = 'upload'
            st.rerun()
        
        if col2.button("❌ No, Keep Run", key="no_new_upload", use_container_width=True):
            # Just clear the confirmation flag
            st.session_state.pop('confirm_new_upload', None)
            st.rerun()
    
    # Developer Mode Toggle (PRD R1.6)
    st.sidebar.divider()
    dev_mode_enabled = st.sidebar.toggle(
        "🔧 Developer Mode", 
        key="developer_mode_toggle", 
        value=st.session_state.get("developer_mode_active", False),
        help="Show detailed debug information and stack traces on error pages."
    )
    # Store the state in session for other parts of the application to access
    st.session_state["developer_mode_active"] = dev_mode_enabled


def route_to_page():
    """Route to the appropriate page based on session state."""
    # Initialize current_page if not set
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'upload'
    
    current_page = st.session_state['current_page']
    
    # Route to appropriate page
    if current_page == 'upload':
        upload_page_module.show_upload_page()
    elif current_page == 'target_confirmation':
        target_page_module.show_target_page()
    elif current_page == 'schema_confirmation':
        schema_page_module.show_schema_page()
    elif current_page == 'validation':
        validation_page_module.show_validation_page()
    elif current_page == 'prep':
        prep_page_module.show_prep_page()
    elif current_page == 'automl':
        automl_page_module.show_automl_page()
    elif current_page == 'explain':
        explain_page_module.show_explain_page()
    elif current_page == 'results':
        results_page_module.show_results_page()
    else:
        st.error(f"Page '{current_page}' is not yet implemented.")
        st.info("Please use the navigation sidebar to go to an available page.")


def main():
    """Main application entry point."""
    # Configure Streamlit page
    st.set_page_config(**get_page_config())
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
    }
    .stSuccess {
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show navigation sidebar
    show_navigation_sidebar()
    
    # Route to the appropriate page
    route_to_page()
    
    # Show footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
        🔮 The Projection Wizard - ML Pipeline for Tabular Data
        </div>
        """, 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main() 