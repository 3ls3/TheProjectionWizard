"""
Streamlit page for feature schema confirmation (Step 3).
Implements the UI for Key Feature Schema Assist & Confirmation.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from step_2_schema import feature_definition_logic, target_definition_logic
from common import constants, storage, utils


# UI Configuration Constants
DTYPE_OPTIONS = {
    "object": "Text/String",
    "int64": "Integer Numbers",
    "float64": "Decimal Numbers", 
    "bool": "True/False (Boolean)",
    "datetime64[ns]": "Date/Time",
    "category": "Category"
}

DTYPE_REVERSE_MAP = {v: k for k, v in DTYPE_OPTIONS.items()}

ENCODING_ROLE_OPTIONS = {
    "numeric-continuous": "Numeric (Continuous) - e.g., price, temperature",
    "numeric-discrete": "Numeric (Discrete) - e.g., count, age",
    "categorical-nominal": "Categorical (No Order) - e.g., color, brand",
    "categorical-ordinal": "Categorical (Ordered) - e.g., small/medium/large",
    "text": "Text (for NLP or hashing) - e.g., descriptions, comments",
    "datetime": "Date/Time features - e.g., timestamp, date",
    "boolean": "Boolean (True/False) - e.g., is_active, has_discount",
    "target": "Target variable (prediction goal)"
}

ENCODING_ROLE_REVERSE_MAP = {v.split(" - ")[0]: k for k, v in ENCODING_ROLE_OPTIONS.items()}


def get_user_friendly_dtype(dtype_str: str) -> str:
    """Convert pandas dtype to user-friendly name."""
    return DTYPE_OPTIONS.get(dtype_str, dtype_str)


def get_user_friendly_encoding_role(role: str) -> str:
    """Convert encoding role to user-friendly description."""
    return ENCODING_ROLE_OPTIONS.get(role, role)


def get_column_stats(series: pd.Series) -> dict:
    """Get basic statistics for a column."""
    return {
        "unique_values": series.nunique(),
        "missing_values": series.isna().sum(),
        "missing_percentage": round((series.isna().sum() / len(series)) * 100, 1),
        "data_type": str(series.dtype)
    }


def show_schema_page():
    """Display the feature schema confirmation page."""
    
    # Page Title
    st.title("Step 3: Review Key Feature Types & Encoding Roles")
    
    # Check if run_id exists in session state
    if 'run_id' not in st.session_state:
        st.error("No active run found. Please upload a file first.")
        if st.button("Go to Upload Page"):
            st.session_state['current_page'] = 'upload'
            st.rerun()
        return
    
    run_id = st.session_state['run_id']
    
    try:
        # Load Data & Metadata
        run_dir_path = storage.get_run_dir(run_id)
        original_data_path = run_dir_path / constants.ORIGINAL_DATA_FILENAME
        
        if not original_data_path.exists():
            st.error(f"Original data file not found: {original_data_path}")
            return
        
        # Load the CSV data
        with st.spinner("Loading data..."):
            df = pd.read_csv(original_data_path)
        
        # Load metadata
        metadata = storage.read_metadata(run_id)
        if not metadata or 'target_info' not in metadata or not metadata['target_info']:
            st.error("Target information not found. Please complete Step 2: Target Confirmation first.")
            if st.button("Go to Target Confirmation"):
                st.session_state['current_page'] = 'target_confirmation'
                st.rerun()
            return
        
        target_info = metadata['target_info']
        
        # Generate suggestions and identify key features
        with st.spinner("Analyzing features and generating suggestions..."):
            # Get all initial schema suggestions
            all_initial_schemas = feature_definition_logic.suggest_initial_feature_schemas(df)
            
            # Identify key features
            num_features_to_surface = min(7, len(df.columns) - 1)  # Exclude target from count
            key_feature_names = feature_definition_logic.identify_key_features(
                df, target_info, num_features_to_surface=num_features_to_surface
            )
        
        # Load existing feature schemas if user revisits page
        existing_feature_schemas = {}
        if 'feature_schemas' in metadata and metadata['feature_schemas']:
            existing_feature_schemas = metadata['feature_schemas']
        
        # Initialize session state for UI choices
        if 'ui_feature_schemas_override' not in st.session_state:
            st.session_state.ui_feature_schemas_override = {}
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
        
        # Show target information
        st.info(f"🎯 **Target Column:** {target_info['name']} | **Task:** {target_info['task_type']}")
        
        # Key Features Section
        st.subheader("Key Features for Review")
        st.caption("These features were identified as potentially important for your prediction task.")
        
        if not key_feature_names:
            st.warning("No key features were identified. You can review all columns below.")
        else:
            for i, feature_name in enumerate(key_feature_names):
                if feature_name == target_info['name']:
                    continue  # Skip target column
                    
                with st.container():
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    feature_series = df[feature_name]
                    stats = get_column_stats(feature_series)
                    
                    with col1:
                        st.write(f"**{feature_name}**")
                        st.caption(f"Unique: {stats['unique_values']} | Missing: {stats['missing_values']} ({stats['missing_percentage']}%)")
                        # Show sample values
                        sample_values = feature_series.dropna().head(3).tolist()
                        st.caption(f"Sample: {sample_values}")
                    
                    with col2:
                        # Data Type Selection
                        initial_dtype = all_initial_schemas[feature_name]['initial_dtype']
                        existing_dtype = existing_feature_schemas.get(feature_name, {}).get('dtype', initial_dtype)
                        
                        # Get current selection or use existing/initial
                        current_dtype_key = f"dtype_{feature_name}"
                        if current_dtype_key not in st.session_state:
                            st.session_state[current_dtype_key] = existing_dtype
                        
                        user_friendly_dtype = get_user_friendly_dtype(st.session_state[current_dtype_key])
                        dtype_options_list = list(DTYPE_OPTIONS.values())
                        
                        try:
                            dtype_index = dtype_options_list.index(user_friendly_dtype)
                        except ValueError:
                            dtype_index = 0
                        
                        selected_dtype_friendly = st.selectbox(
                            "Data Type:",
                            options=dtype_options_list,
                            index=dtype_index,
                            key=f"selectbox_dtype_{feature_name}"
                        )
                        
                        # Convert back to pandas dtype
                        selected_dtype = DTYPE_REVERSE_MAP[selected_dtype_friendly]
                        st.session_state[current_dtype_key] = selected_dtype
                    
                    with col3:
                        # Encoding Role Selection
                        initial_role = all_initial_schemas[feature_name]['suggested_encoding_role']
                        existing_role = existing_feature_schemas.get(feature_name, {}).get('encoding_role', initial_role)
                        
                        # Get current selection or use existing/initial
                        current_role_key = f"role_{feature_name}"
                        if current_role_key not in st.session_state:
                            st.session_state[current_role_key] = existing_role
                        
                        user_friendly_role = get_user_friendly_encoding_role(st.session_state[current_role_key])
                        role_options_list = list(ENCODING_ROLE_OPTIONS.values())
                        
                        try:
                            role_index = role_options_list.index(user_friendly_role)
                        except ValueError:
                            role_index = 0
                        
                        selected_role_friendly = st.selectbox(
                            "Encoding Role:",
                            options=role_options_list,
                            index=role_index,
                            key=f"selectbox_role_{feature_name}",
                            help="How this feature should be processed for machine learning"
                        )
                        
                        # Convert back to actual role
                        selected_role = ENCODING_ROLE_REVERSE_MAP.get(
                            selected_role_friendly.split(" - ")[0], 
                            st.session_state[current_role_key]
                        )
                        st.session_state[current_role_key] = selected_role
                    
                    # Store the user's choice
                    st.session_state.ui_feature_schemas_override[feature_name] = {
                        'final_dtype': selected_dtype,
                        'final_encoding_role': selected_role
                    }
                    
                    st.divider()
        
        # Advanced Section - All Columns
        with st.expander("Review/Override All Columns (Advanced)", expanded=False):
            st.caption("Review and modify schema settings for all columns in your dataset.")
            
            # Filter out target column
            all_columns = [col for col in df.columns if col != target_info['name']]
            
            for feature_name in all_columns:
                with st.container():
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    feature_series = df[feature_name]
                    stats = get_column_stats(feature_series)
                    
                    with col1:
                        st.write(f"**{feature_name}**")
                        if feature_name in key_feature_names:
                            st.caption("🌟 Key Feature")
                        st.caption(f"Unique: {stats['unique_values']} | Missing: {stats['missing_values']} ({stats['missing_percentage']}%)")
                    
                    with col2:
                        # Data Type Selection
                        initial_dtype = all_initial_schemas[feature_name]['initial_dtype']
                        existing_dtype = existing_feature_schemas.get(feature_name, {}).get('dtype', initial_dtype)
                        
                        # Use existing session state value if available
                        session_dtype = st.session_state.ui_feature_schemas_override.get(feature_name, {}).get('final_dtype', existing_dtype)
                        
                        user_friendly_dtype = get_user_friendly_dtype(session_dtype)
                        dtype_options_list = list(DTYPE_OPTIONS.values())
                        
                        try:
                            dtype_index = dtype_options_list.index(user_friendly_dtype)
                        except ValueError:
                            dtype_index = 0
                        
                        selected_dtype_friendly = st.selectbox(
                            "Data Type:",
                            options=dtype_options_list,
                            index=dtype_index,
                            key=f"advanced_dtype_{feature_name}"
                        )
                        
                        selected_dtype = DTYPE_REVERSE_MAP[selected_dtype_friendly]
                    
                    with col3:
                        # Encoding Role Selection
                        initial_role = all_initial_schemas[feature_name]['suggested_encoding_role']
                        existing_role = existing_feature_schemas.get(feature_name, {}).get('encoding_role', initial_role)
                        
                        # Use existing session state value if available
                        session_role = st.session_state.ui_feature_schemas_override.get(feature_name, {}).get('final_encoding_role', existing_role)
                        
                        user_friendly_role = get_user_friendly_encoding_role(session_role)
                        role_options_list = list(ENCODING_ROLE_OPTIONS.values())
                        
                        try:
                            role_index = role_options_list.index(user_friendly_role)
                        except ValueError:
                            role_index = 0
                        
                        selected_role_friendly = st.selectbox(
                            "Encoding Role:",
                            options=role_options_list,
                            index=role_index,
                            key=f"advanced_role_{feature_name}"
                        )
                        
                        selected_role = ENCODING_ROLE_REVERSE_MAP.get(
                            selected_role_friendly.split(" - ")[0], 
                            session_role
                        )
                    
                    # Store the user's choice for all columns
                    if feature_name not in st.session_state.ui_feature_schemas_override:
                        st.session_state.ui_feature_schemas_override[feature_name] = {}
                    
                    st.session_state.ui_feature_schemas_override[feature_name].update({
                        'final_dtype': selected_dtype,
                        'final_encoding_role': selected_role
                    })
                    
                    st.divider()
        
        # Show Summary
        st.subheader("Summary of Changes")
        changes_made = len(st.session_state.ui_feature_schemas_override)
        if changes_made > 0:
            st.write(f"You have reviewed {changes_made} feature(s).")
            
            # Show key changes
            key_changes = {k: v for k, v in st.session_state.ui_feature_schemas_override.items() 
                          if k in key_feature_names}
            if key_changes:
                st.write("**Key Features Reviewed:**")
                for feature, schema in key_changes.items():
                    st.write(f"• {feature}: {get_user_friendly_dtype(schema['final_dtype'])} → {get_user_friendly_encoding_role(schema['final_encoding_role'])}")
        else:
            st.info("Using AI suggestions for all features. You can review individual features above.")
        
        # Confirmation Section
        st.divider()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("✅ Confirm Feature Schemas", type="primary", use_container_width=True):
                with st.spinner("Saving feature schemas..."):
                    # Get user overrides from session state
                    user_overrides = st.session_state.get('ui_feature_schemas_override', {})
                    
                    # Call the confirmation function
                    success = feature_definition_logic.confirm_feature_schemas(
                        run_id, user_overrides, all_initial_schemas
                    )
                    
                    if success:
                        st.success("✅ Feature schemas saved successfully!")
                        st.balloons()
                        
                        # Clear session state
                        st.session_state.ui_feature_schemas_override = {}
                        
                        # Auto-navigate to next step immediately
                        st.session_state['current_page'] = 'validation'
                        st.rerun()
                    else:
                        st.error("❌ Failed to save feature schemas. Check logs for details.")
                        
                        # Show log file location
                        log_path = run_dir_path / constants.PIPELINE_LOG_FILENAME
                        st.error(f"Check log file: `{log_path}`")
        
        with col2:
            if st.button("← Back", use_container_width=True):
                st.session_state['current_page'] = 'target_confirmation'
                st.rerun()
    
    except Exception as e:
        st.error(f"An error occurred while loading the schema page: {str(e)}")
        st.exception(e)


def main():
    """Main function to run the schema confirmation page."""
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
                if 'target_info' in metadata and metadata['target_info']:
                    st.sidebar.text(f"Target: {metadata['target_info']['name']}")
                    st.sidebar.text(f"Task: {metadata['target_info']['task_type']}")
        except Exception:
            pass  # Silently handle metadata loading errors
    
    # Show the schema confirmation page
    show_schema_page()


if __name__ == "__main__":
    # Configure Streamlit page
    st.set_page_config(
        page_title="Step 3: Feature Schema - Projection Wizard",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    main() 