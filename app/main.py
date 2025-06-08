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


def show_navigation_sidebar():
    """Display navigation options in the sidebar."""
    st.sidebar.title("🔮 Projection Wizard")
    
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
        
        # Show progress indicator
        current_page = st.session_state.get('current_page', 'upload')
        pages = ['upload', 'target_confirmation', 'schema_confirmation', 'validation', 'prep', 'automl', 'explain', 'results']
        page_names = ['Upload Data', 'Target Confirmation', 'Schema Confirmation', 'Data Validation', 'Data Preparation', 'Model Training', 'Model Explanation', 'Results']
        
        st.sidebar.subheader("Pipeline Progress")
        current_index = pages.index(current_page) if current_page in pages else 0
        
        # Enhanced emoji logic considering stage status
        validation_failed = validation_summary.status == "failed_critically"
        
        for i, (page, name) in enumerate(zip(pages, page_names)):
            if page == current_page:
                st.sidebar.write(f"👉 **{i+1}. {name}**")
            elif page == 'validation' and validation_failed:
                st.sidebar.write(f"🚫 {i+1}. {name}")
            elif validation_failed and i > pages.index('validation'):
                # Subsequent stages are blocked if validation failed
                st.sidebar.write(f"🚫 {i+1}. {name}")
            elif i < current_index:
                st.sidebar.write(f"✅ {i+1}. {name}")
            else:
                st.sidebar.write(f"⏳ {i+1}. {name}")
    
    st.sidebar.divider()
    
    # Navigation buttons
    st.sidebar.subheader("Navigation")
    
    # Always available: Upload page
    if st.sidebar.button("📁 Upload Data", use_container_width=True):
        st.session_state['current_page'] = 'upload'
        st.rerun()
    
    # Available after upload: Target confirmation, Schema confirmation, Data validation
    if 'run_id' in st.session_state:
        if st.sidebar.button("🎯 Target Confirmation", use_container_width=True):
            st.session_state['current_page'] = 'target_confirmation'
            st.rerun()
            
        if st.sidebar.button("📋 Schema Confirmation", use_container_width=True):
            st.session_state['current_page'] = 'schema_confirmation'
            st.rerun()
            
        if st.sidebar.button("🔍 Data Validation", use_container_width=True):
            st.session_state['current_page'] = 'validation'
            st.rerun()
        
        # Data Preparation button
        prep_disabled = not validation_summary.can_proceed
        prep_help = None
        if validation_summary.status == "failed_critically":
            prep_help = validation_summary.message or "❌ Validation critically failed - fix data issues first"
        elif not validation_summary.can_proceed:
            prep_help = "Complete data validation first"
        
        if prep_disabled:
            st.sidebar.button("🔧 Data Preparation", disabled=True, help=prep_help)
        else:
            if st.sidebar.button("🔧 Data Preparation", use_container_width=True):
                st.session_state['current_page'] = 'prep'
                st.rerun()
        
        # Model Training button
        training_disabled = not (validation_summary.can_proceed and prep_summary.can_proceed)
        training_help = None
        if validation_summary.status == "failed_critically":
            training_help = validation_summary.message or "❌ Validation critically failed - fix data issues first"
        elif not prep_summary.can_proceed:
            training_help = "Complete data preparation first"
        elif not validation_summary.can_proceed:
            training_help = "Complete data validation first"
        
        if training_disabled:
            st.sidebar.button("🤖 Model Training", disabled=True, help=training_help)
        else:
            if st.sidebar.button("🤖 Model Training", use_container_width=True):
                st.session_state['current_page'] = 'automl'
                st.rerun()
        
        # Model Explanation button
        explain_disabled = not (validation_summary.can_proceed and prep_summary.can_proceed and automl_summary.can_proceed)
        explain_help = None
        if validation_summary.status == "failed_critically":
            explain_help = validation_summary.message or "❌ Validation critically failed - fix data issues first"
        elif not automl_summary.can_proceed:
            explain_help = "Complete model training first"
        elif not prep_summary.can_proceed:
            explain_help = "Complete data preparation first"
        elif not validation_summary.can_proceed:
            explain_help = "Complete data validation first"
        
        if explain_disabled:
            st.sidebar.button("📊 Model Explanation", disabled=True, help=explain_help)
        else:
            if st.sidebar.button("📊 Model Explanation", use_container_width=True):
                st.session_state['current_page'] = 'explain'
                st.rerun()
        
        # Results button
        results_disabled = not (validation_summary.can_proceed and prep_summary.can_proceed and 
                               automl_summary.can_proceed and explain_summary.can_proceed)
        results_help = None
        if validation_summary.status == "failed_critically":
            results_help = validation_summary.message or "❌ Validation critically failed - fix data issues first"
        elif not explain_summary.can_proceed:
            results_help = "Complete model explanation first"
        elif not automl_summary.can_proceed:
            results_help = "Complete model training first"
        elif not prep_summary.can_proceed:
            results_help = "Complete data preparation first"
        elif not validation_summary.can_proceed:
            results_help = "Complete data validation first"
        
        if results_disabled:
            st.sidebar.button("📈 Results", disabled=True, help=results_help)
        else:
            if st.sidebar.button("📈 Results", use_container_width=True):
                st.session_state['current_page'] = 'results'
                st.rerun()
    else:
        # No run_id - disable all buttons except Upload Data
        st.sidebar.button("🎯 Target Confirmation", disabled=True, help="Upload data first")
        st.sidebar.button("📋 Schema Confirmation", disabled=True, help="Upload data first")
        st.sidebar.button("🔍 Data Validation", disabled=True, help="Upload data first")
        st.sidebar.button("🔧 Data Preparation", disabled=True, help="Upload data first")
        st.sidebar.button("🤖 Model Training", disabled=True, help="Upload data first")
        st.sidebar.button("📊 Model Explanation", disabled=True, help="Upload data first")
        st.sidebar.button("📈 Results", disabled=True, help="Upload data first")


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