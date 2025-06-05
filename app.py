"""
Main Streamlit application for The Projection Wizard.
Entry point for the ML pipeline interface.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import importlib
upload_page_module = importlib.import_module('ui.01_upload_page')
show_upload_page = upload_page_module.show_upload_page


def main():
    """Main application entry point."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="The Projection Wizard",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Show the upload page (for now, this is our only implemented step)
    show_upload_page()


if __name__ == "__main__":
    main() 