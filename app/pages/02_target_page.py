"""
Streamlit page for target variable confirmation (Step 2).
Implements the UI for target column selection and task type confirmation.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from pipeline.step_2_schema import target_definition_logic
from common import constants, storage, utils


def get_ml_type_options(selected_task: str, target_series: pd.Series) -> list:
    """
    Get appropriate ML type options based on selected task and target data.
    
    Args:
        selected_task: Selected task type ('classification' or 'regression')
        target_series: Pandas series of the target column
        
    Returns:
        List of appropriate ML type options
    """
    if selected_task == "classification":
        # Determine options based on target data characteristics
        unique_vals = target_series.nunique()
        dtype = target_series.dtype
        
        options = []
        
        if pd.api.types.is_bool_dtype(dtype):
            options.append("binary_boolean")
        elif pd.api.types.is_numeric_dtype(dtype):
            if unique_vals == 2:
                # Check if it's 0/1 binary
                unique_sorted = sorted(target_series.dropna().unique())
                if len(unique_sorted) == 2 and unique_sorted[0] == 0 and unique_sorted[1] == 1:
                    options.append("binary_01")
                else:
                    options.append("binary_numeric")
            else:
                options.append("multiclass_int_labels")
        else:
            # Text/categorical
            if unique_vals == 2:
                options.append("binary_text_labels")
            else:
                options.append("multiclass_text_labels")
                if unique_vals > constants.SCHEMA_CONFIG["max_categorical_cardinality"]:
                    options.append("high_cardinality_text")
        
        # Always include the general options as fallbacks
        all_classification_options = [
            "binary_01", "binary_numeric", "binary_text_labels", "binary_boolean",
            "multiclass_int_labels", "multiclass_text_labels", "high_cardinality_text"
        ]
        
        # Add any missing options that weren't already included
        for opt in all_classification_options:
            if opt not in options:
                options.append(opt)
                
        return options
        
    else:  # regression
        return ["numeric_continuous"]


def get_ml_type_description(ml_type: str) -> str:
    """Get description for each ML type option."""
    descriptions = {
        "binary_01": "Binary classification with 0/1 numeric labels",
        "binary_numeric": "Binary classification with numeric labels (not 0/1)",
        "binary_text_labels": "Binary classification with text labels (e.g., 'yes'/'no')",
        "binary_boolean": "Binary classification with True/False boolean values",
        "multiclass_int_labels": "Multi-class classification with integer labels",
        "multiclass_text_labels": "Multi-class classification with text labels",
        "high_cardinality_text": "Classification with many unique text categories (may need preprocessing)",
        "numeric_continuous": "Regression with continuous numeric values",
        "unknown_type": "Unknown or unrecognized data type"
    }
    return descriptions.get(ml_type, "No description available")


def show_target_page():
    """Display the target confirmation page."""
    
    # Page Title
    st.title("Step 2: Confirm Target Variable & Task Type")
    
    # Check if run_id exists in session state
    if 'run_id' not in st.session_state:
        st.error("No active run found. Please upload a file first.")
        if st.button("Go to Upload Page"):
            st.session_state['current_page'] = 'upload'
            st.rerun()
        return
    
    run_id = st.session_state['run_id']
    
    # Display current run ID
    st.info(f"**Current Run ID:** {run_id}")
    
    # Introductory text
    st.write("Please select the column you want to predict (target variable) and specify the type of machine learning task.")
    
    try:
        # Load Data and Suggestions
        run_dir_path = storage.get_run_dir(run_id)
        original_data_path = run_dir_path / constants.ORIGINAL_DATA_FILENAME
        
        if not original_data_path.exists():
            st.error(f"Original data file not found: {original_data_path}")
            return
        
        # Load the CSV data
        with st.spinner("Loading data..."):
            df = pd.read_csv(original_data_path)
        
        # Get AI suggestions
        suggested_target, suggested_task, suggested_ml_type = target_definition_logic.suggest_target_and_task(df)
        
        # Attempt to load existing target info from metadata.json
        metadata = storage.read_metadata(run_id)
        target_info_exists = bool(metadata and 'target_info' in metadata and metadata['target_info'])
        
        if target_info_exists:
            target_info = metadata['target_info']
            current_target = target_info.get('name', suggested_target)
            current_task = target_info.get('task_type', suggested_task)
            current_ml_type = target_info.get('ml_type', suggested_ml_type)
        else:
            current_target = suggested_target
            current_task = suggested_task
            current_ml_type = suggested_ml_type
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
        
        # UI Elements
        
        # Target Column Selection
        st.subheader("Target Column")
        column_options = list(df.columns)
        
        # Find index for current target, default to 0 if not found
        try:
            current_target_index = column_options.index(current_target) if current_target in column_options else 0
        except (ValueError, TypeError):
            current_target_index = 0
        
        selected_target = st.selectbox(
            "Select your target column:",
            options=column_options,
            index=current_target_index,
            disabled=target_info_exists,
            help="This is the column you want to predict (your dependent variable)"
        )
        
        # Show target column statistics
        if selected_target:
            target_series = df[selected_target]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Values", target_series.nunique())
            with col2:
                st.metric("Missing Values", target_series.isna().sum())
            with col3:
                st.metric("Data Type", str(target_series.dtype))
        
        # Task Type Selection
        st.subheader("Task Type")
        task_options = constants.TASK_TYPES  # ["classification", "regression"]
        
        # Find index for current task
        try:
            current_task_index = task_options.index(current_task) if current_task in task_options else 0
        except (ValueError, TypeError):
            current_task_index = 0
        
        selected_task = st.radio(
            "What type of ML task is this?",
            options=task_options,
            index=current_task_index,
            disabled=target_info_exists,
            help="Classification: Predicting categories/classes. Regression: Predicting continuous numbers."
        )
        
        # ML Type Selection
        st.subheader("Target Variable ML Format")
        
        if selected_target:
            target_series = df[selected_target]
            ml_type_options = get_ml_type_options(selected_task, target_series)
            
            # Find index for current ML type
            try:
                current_ml_type_index = ml_type_options.index(current_ml_type) if current_ml_type in ml_type_options else 0
            except (ValueError, TypeError):
                current_ml_type_index = 0
            
            selected_ml_type = st.selectbox(
                "How should the target variable be treated for the model?",
                options=ml_type_options,
                index=current_ml_type_index,
                disabled=target_info_exists,
                help="This determines how the target will be encoded for machine learning"
            )
            
            # Show description for selected ML type
            st.caption(f"💡 {get_ml_type_description(selected_ml_type)}")
            
            # Show AI suggestions for reference (only if not already completed)
            if (not target_info_exists and 
                (selected_target != suggested_target or 
                 selected_task != suggested_task or 
                 selected_ml_type != suggested_ml_type)):
                
                with st.expander("🤖 AI Suggestions", expanded=False):
                    st.write("**AI suggested the following:**")
                    st.write(f"• Target Column: `{suggested_target}`")
                    st.write(f"• Task Type: `{suggested_task}`")
                    st.write(f"• ML Type: `{suggested_ml_type}`")
        
        # Primary Action Button
        st.divider()
        
        if st.button("✅ Confirm & Proceed", type="primary", use_container_width=True):
            if target_info_exists:
                # Step already completed - navigate directly
                st.session_state['current_page'] = 'schema_confirmation'
                st.rerun()
            else:
                # First time through - save the target definition then navigate
                with st.spinner("Saving target definition..."):
                    success = target_definition_logic.confirm_target_definition(
                        run_id, selected_target, selected_task, selected_ml_type
                    )
                    
                    if success:
                        st.success("✅ Target definition saved successfully!")
                        st.balloons()
                        # Navigate directly to next page
                        st.session_state['current_page'] = 'schema_confirmation'
                        st.rerun()
                    else:
                        # Use standardized error display
                        error_message = "Failed to save target definition."
                        save_exception = Exception(error_message)
                        is_dev_mode = st.session_state.get("developer_mode_active", False)
                        utils.display_page_error(save_exception, run_id=run_id, stage_name=constants.SCHEMA_STAGE, dev_mode=is_dev_mode)
    
    except Exception as e:
        is_dev_mode = st.session_state.get("developer_mode_active", False)
        utils.display_page_error(e, run_id=st.session_state.get('run_id'), stage_name="Target Confirmation", dev_mode=is_dev_mode)


def main():
    """Main function to run the target confirmation page."""
    # Note: When run from main app.py, the sidebar is handled centrally
    # This main() function is primarily for standalone testing
    
    # Display Current Run ID (if exists) - only when running standalone
    if 'run_id' in st.session_state and not hasattr(st, '_is_running_from_main_app'):
        st.sidebar.info(f"Current Run ID: {st.session_state['run_id']}")
        
        # Show run metadata in sidebar
        try:
            metadata = storage.read_metadata(st.session_state['run_id'])
            if metadata:
                st.sidebar.subheader("Run Information")
                st.sidebar.text(f"File: {metadata.get('original_filename', 'Unknown')}")
                if 'initial_rows' in metadata:
                    st.sidebar.text(f"Rows: {metadata['initial_rows']}")
                if 'initial_cols' in metadata:
                    st.sidebar.text(f"Columns: {metadata['initial_cols']}")
        except Exception:
            pass  # Silently handle metadata loading errors
    
    # Show the target confirmation page
    show_target_page()


if __name__ == "__main__":
    # Configure Streamlit page
    st.set_page_config(
        page_title="Step 2: Target Confirmation - Projection Wizard",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    main() 