import streamlit as st
from app.wizardry import model_dict
from app.wizardry import (
    step4_task_detection, step5_model_selection, step6_training,
    step7_inference, step8_explainability, step9_outputs
)

st.title("Smart Prediction Wizard")

# Initialize navigation state
if "current_step" not in st.session_state:
    st.session_state.current_step = "Step 4: Task Detection"

# Upload dataset once and store in session state
if "raw_data" not in st.session_state:
    uploaded_file = st.file_uploader("Upload your CSV or text file")
    if uploaded_file:
        import pandas as pd
        st.session_state.raw_data = pd.read_csv(uploaded_file)
        st.success(f"Loaded dataset with {st.session_state.raw_data.shape[0]} rows and {st.session_state.raw_data.shape[1]} columns.")

# Pass the uploaded raw_data down the pipeline
if "raw_data" in st.session_state:
    # Store or update in session_state for cleaning/processing steps
    if "clean_data" not in st.session_state:
        st.session_state.clean_data = st.session_state.raw_data.copy()

# Define page order for navigation
PAGE_ORDER = [
    "Step 4: Task Detection",
    "Step 5: Model Selection",
    "Step 6: Training",
    "Step 7: Inference",
    "Step 8: Explainability",
    "Step 9: Outputs"
]

# Sidebar navigation for wizard steps 4-9
PAGES = {
    "Step 4: Task Detection": step4_task_detection,
    "Step 5: Model Selection": step5_model_selection,
    "Step 6: Training": step6_training,
    "Step 7: Inference": step7_inference,
    "Step 8: Explainability": step8_explainability,
    "Step 9: Outputs": step9_outputs,
}

# Navigation functions
def go_to_next_step():
    current_index = PAGE_ORDER.index(st.session_state.current_step)
    if current_index < len(PAGE_ORDER) - 1:
        st.session_state.current_step = PAGE_ORDER[current_index + 1]

def go_to_previous_step():
    current_index = PAGE_ORDER.index(st.session_state.current_step)
    if current_index > 0:
        st.session_state.current_step = PAGE_ORDER[current_index - 1]

# Sidebar navigation
st.sidebar.title("🧙 The Projection Wizard")
selection = st.sidebar.radio("Go to", PAGE_ORDER, index=PAGE_ORDER.index(st.session_state.current_step))

# Update current step if changed in sidebar
if selection != st.session_state.current_step:
    st.session_state.current_step = selection

# Run the selected step
PAGES[st.session_state.current_step].run()
