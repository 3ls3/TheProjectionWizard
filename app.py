"""
Main Streamlit application for The Projection Wizard.
Entry point for the ML pipeline interface with multi-page navigation.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import importlib

# Import page modules
upload_page_module = importlib.import_module('ui.01_upload_page')
target_page_module = importlib.import_module('ui.02_target_page')
schema_page_module = importlib.import_module('ui.03_schema_page')
validation_page_module = importlib.import_module('ui.04_validation_page')
prep_page_module = importlib.import_module('ui.05_prep_page')
automl_page_module = importlib.import_module('ui.06_automl_page')


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
        st.sidebar.success(f"**Active Run:** {st.session_state['run_id']}")
        
        # Show progress indicator
        current_page = st.session_state.get('current_page', 'upload')
        pages = ['upload', 'target_confirmation', 'schema_confirmation', 'validation', 'prep', 'automl', 'explain', 'results']
        page_names = ['Upload Data', 'Target Confirmation', 'Schema Confirmation', 'Data Validation', 'Data Preparation', 'Model Training', 'Model Explanation', 'Results']
        
        st.sidebar.subheader("Pipeline Progress")
        current_index = pages.index(current_page) if current_page in pages else 0
        
        for i, (page, name) in enumerate(zip(pages, page_names)):
            if page == current_page:
                st.sidebar.write(f"👉 **{i+1}. {name}**")
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
    
    # Available after upload: Target confirmation
    if 'run_id' in st.session_state:
        if st.sidebar.button("🎯 Target Confirmation", use_container_width=True):
            st.session_state['current_page'] = 'target_confirmation'
            st.rerun()
            
        # Available after target confirmation: Schema confirmation
        if st.sidebar.button("📋 Schema Confirmation", use_container_width=True):
            st.session_state['current_page'] = 'schema_confirmation'
            st.rerun()
            
        # Available after schema confirmation: Data validation
        if st.sidebar.button("🔍 Data Validation", use_container_width=True):
            st.session_state['current_page'] = 'validation'
            st.rerun()
    
    # Future pages (conditionally enabled)
    # Check if validation is complete and successful enough
    validation_complete = False
    if 'run_id' in st.session_state:
        try:
            from common import storage, constants, schemas
            validation_data = storage.read_json(st.session_state['run_id'], constants.VALIDATION_FILENAME)
            if validation_data:
                validation_summary = schemas.ValidationReportSummary(**validation_data)
                success_rate = (validation_summary.successful_expectations / validation_summary.total_expectations) if validation_summary.total_expectations > 0 else 0
                validation_complete = validation_summary.overall_success or success_rate >= 0.95
        except:
            validation_complete = False
    
    if validation_complete:
        if st.sidebar.button("🔧 Data Preparation", use_container_width=True):
            st.session_state['current_page'] = 'prep'
            st.rerun()
    else:
        st.sidebar.button("🔧 Data Preparation", disabled=True, help="Complete data validation first")
    
    # Check if data preparation is complete
    prep_complete = False
    if 'run_id' in st.session_state:
        try:
            from common import storage, constants
            status_data = storage.read_json(st.session_state['run_id'], constants.STATUS_FILENAME)
            prep_complete = (status_data and 
                           status_data.get('stage') == constants.PREP_STAGE and 
                           status_data.get('status') == 'completed')
        except:
            prep_complete = False
    
    if prep_complete:
        if st.sidebar.button("🤖 Model Training", use_container_width=True):
            st.session_state['current_page'] = 'automl'
            st.rerun()
    else:
        st.sidebar.button("🤖 Model Training", disabled=True, help="Complete data preparation first")
    
    # Check if AutoML training is complete
    automl_complete = False
    if 'run_id' in st.session_state:
        try:
            from common import storage, constants
            status_data = storage.read_json(st.session_state['run_id'], constants.STATUS_FILENAME)
            automl_complete = (status_data and 
                             status_data.get('stage') == constants.AUTOML_STAGE and 
                             status_data.get('status') == 'completed')
        except:
            automl_complete = False
    
    if automl_complete:
        if st.sidebar.button("📊 Model Explanation", use_container_width=True):
            st.session_state['current_page'] = 'explain'
            st.rerun()
    else:
        st.sidebar.button("📊 Model Explanation", disabled=True, help="Complete model training first")
    
    st.sidebar.button("📈 Results", disabled=True, help="Complete model explanation first")


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