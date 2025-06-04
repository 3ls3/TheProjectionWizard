"""
Main entry point for Team A's Part 1 (EDA, Cleaning, Validation) pipeline.

This module serves as the main entry point for the data preprocessing pipeline,
handling the initial stages of data processing including:
- Data upload and validation
- Type inference and override
- Data validation
- Data cleaning
- Final results and export

Usage:
    streamlit run app/team_a/main-part1.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import sections
from app.team_a.sections.upload import upload_pipeline_section
from app.team_a.sections.type_override import (
    type_override_pipeline_section,

)
from app.team_a.sections.validation import (
    validation_pipeline_section

)
from app.team_a.sections.cleaning import (
    cleaning_pipeline_section
)
from app.team_a.sections.final import show_final_results_section
from app.team_a.sections.utils import make_json_serializable

# Pipeline stages
STAGES = ["upload", "type_override", "validation", "cleaning", "final"]

def render_sidebar(current_stage: str):
    """Draw left‑hand checklist based on current stage."""
    stage_icons = {
        "upload": "📁",
        "type_override": "🎯",
        "validation": "🔍",
        "cleaning": "🧹",
        "final": "📋"
    }
    for stage in STAGES:
        icon = stage_icons[stage]
        if STAGES.index(stage) < STAGES.index(current_stage):
            st.sidebar.markdown(f"✅ {icon} {stage.replace('_',' ').title()}")
        elif stage == current_stage:
            st.sidebar.markdown(f"🔄 {icon} {stage.replace('_',' ').title()}")
        else:
            st.sidebar.markdown(f"⏳ {icon} {stage.replace('_',' ').title()}")
    st.sidebar.markdown("---")

def render_stage_sequence(current_stage: str):
    """Render all stages up to and including the current one in sequence."""
    for stage in STAGES:
        if STAGES.index(stage) > STAGES.index(current_stage):
            break
        render_one_stage(stage, is_current=(stage == current_stage))

def render_one_stage(stage: str, is_current: bool = True):
    """Call the section UI matching the stage with current/summary mode."""
    if stage == "upload":
        upload_pipeline_section(is_current)
    elif stage == "type_override":
        type_override_pipeline_section(is_current)
    elif stage == "validation":
        validation_pipeline_section(is_current)
    elif stage == "cleaning":
        cleaning_pipeline_section(is_current)
    elif stage == "final":
        show_final_results_section(is_current)

def main():
    """Main Streamlit app for Team A's Part 1 pipeline."""
    st.set_page_config(
        page_title="Data Pipeline - Team A (Part 1)",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔍 EDA & Validation Pipeline (Team A - Part 1)")
    st.markdown("---")

    # Initialize session state for pipeline stages
    if 'stage' not in st.session_state:
        st.session_state.stage = "upload"

    # Initialize debug mode
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    # Sidebar with debug toggle
    st.sidebar.title("⚙️ Pipeline Control")

    # Debug toggle with gear icon
    if st.sidebar.button("⚙️ Debug Toggle"):
        st.session_state.debug_mode = not st.session_state.debug_mode

    if st.session_state.debug_mode:
        st.sidebar.write("Debug mode ON")

    # Render sidebar and main stage
    current_stage = st.session_state.get("stage", "upload")
    render_sidebar(current_stage)
    render_stage_sequence(current_stage)

if __name__ == "__main__":
    main()
