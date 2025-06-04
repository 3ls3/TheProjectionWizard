"""
Main Streamlit app for Team A's data pipeline.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import sections
from app.sections.upload import upload_pipeline_section
from app.sections.type_override import (
    type_override_pipeline_section,
    type_override_section,
    continuous_type_override_section,
    apply_type_conversions
)
from app.sections.validation import (
    validation_pipeline_section,
    validation_section,
    run_data_validation,
    display_validation_results,
    validation_debug_section
)
from app.sections.cleaning import (
    cleaning_pipeline_section,
    show_cleaning_section,
    cleaning_debug_section
)
from app.sections.final import show_final_results_section
from app.sections.utils import make_json_serializable

def render_one_stage():
    """Render the current stage of the pipeline."""
    current_stage = st.session_state.get("stage", "upload")

    if current_stage == "upload":
        upload_pipeline_section(is_current=True)
    elif current_stage == "type_override":
        type_override_pipeline_section(is_current=True)
    elif current_stage == "validation":
        validation_pipeline_section(is_current=True)
    elif current_stage == "cleaning":
        cleaning_pipeline_section(is_current=True)
    elif current_stage == "final":
        show_final_results_section(is_current=True)
    else:
        st.error(f"Unknown stage: {current_stage}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Data Pipeline - Team A",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if "stage" not in st.session_state:
        st.session_state["stage"] = "upload"

    # Show pipeline stages in sidebar
    st.sidebar.title("Pipeline Stages")
    stages = ["upload", "type_override", "validation", "cleaning", "final"]
    current_stage = st.session_state.get("stage", "upload")

    for stage in stages:
        if stage == current_stage:
            st.sidebar.markdown(f"**{stage.replace('_', ' ').title()}**")
        else:
            if st.sidebar.button(stage.replace("_", " ").title()):
                st.session_state["stage"] = stage
                st.rerun()

    # Show debug section in sidebar
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.markdown("### Debug Info")
        st.sidebar.json(st.session_state)

    # Render current stage
    render_one_stage()

if __name__ == "__main__":
    main()
