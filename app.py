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
        for i, (page, name) in enumerate(zip(pages, page_names)):
            if page == current_page:
                st.sidebar.write(f"👉 **{i+1}. {name}**")
            elif page == 'upload':
                st.sidebar.write(f"✅ {i+1}. {name}")
            elif page == 'target_confirmation' and current_page in pages[1:]:
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
    
    # Future pages (disabled for now)
    st.sidebar.button("🔍 Data Validation", disabled=True, help="Complete schema confirmation first")
    st.sidebar.button("🔧 Data Preparation", disabled=True, help="Complete data validation first")
    st.sidebar.button("🤖 Model Training", disabled=True, help="Complete data preparation first")
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