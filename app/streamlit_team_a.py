"""
Team A Temporary Streamlit Frontend
==================================

This is a temporary frontend for testing and visualizing the EDA and validation stages.
Team B will handle the main app.py integration.

Usage:
    streamlit run app/streamlit_team_a.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import numpy as np

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from eda_validation import ydata_profile, cleaning, utils
from eda_validation.validation import setup_expectations, run_validation


def main():
    """Main Streamlit app for Team A's EDA and validation pipeline."""
    st.title("🔍 EDA & Validation Pipeline (Team A)")
    st.markdown("---")

    # Initialize session state for pipeline stages
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = "upload"
    
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
    
    # Always show the main pipeline flow
    main_pipeline_flow()


def main_pipeline_flow():
    """Main continuous pipeline flow on the main page."""
    # Define pipeline stages
    stages = [
        ("upload", "📁 Upload"),
        ("type_override", "🎯 Type Override"),
        ("validation", "🔍 Validation"),
        ("cleaning", "🧹 Cleaning"),
        ("final_eda", "📋 Final EDA")
    ]
    
    current_stage = st.session_state.current_stage
    
    # Sidebar checklist showing pipeline progress - fixed logic
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Pipeline Progress**")
    
    for i, (stage, display_name) in enumerate(stages):
        # Determine the status based on actual progress
        if stage == "upload":
            if st.session_state.get('uploaded_df') is not None:
                st.sidebar.markdown(f"✅ {display_name}")
            else:
                st.sidebar.markdown(f"⏳ {display_name}")
        elif stage == "type_override":
            if st.session_state.get('types_confirmed', False):
                st.sidebar.markdown(f"✅ {display_name}")
            elif st.session_state.get('uploaded_df') is not None:
                st.sidebar.markdown(f"🔄 {display_name}")
            else:
                st.sidebar.markdown(f"⏳ {display_name}")
        elif stage == "validation":
            if st.session_state.get('validation_results') is not None:
                if st.session_state.get('validation_success', False):
                    st.sidebar.markdown(f"✅ {display_name}")
                else:
                    st.sidebar.markdown(f"⚠️ {display_name} (Override)")
            elif st.session_state.get('types_confirmed', False):
                st.sidebar.markdown(f"🔄 {display_name}")
            else:
                st.sidebar.markdown(f"⏳ {display_name}")
        elif stage == "cleaning":
            if st.session_state.get('cleaning_done', False):
                st.sidebar.markdown(f"✅ {display_name}")
            elif st.session_state.get('validation_results') is not None or st.session_state.get('validation_override', False):
                st.sidebar.markdown(f"🔄 {display_name}")
            else:
                st.sidebar.markdown(f"⏳ {display_name}")
        elif stage == "final_eda":
            if st.session_state.get('cleaning_done', False):
                st.sidebar.markdown(f"✅ {display_name}")
            else:
                st.sidebar.markdown(f"⏳ {display_name}")
        else:
            st.sidebar.markdown(f"⏳ {display_name}")

    # Always render the continuous pipeline (no stage-based switching)
    upload_pipeline_section()


def validation_section(df=None):
    """Handle data validation with Great Expectations directly."""
    if df is None:
        st.header("✅ Data Validation")
        st.write("Validate data quality using Great Expectations.")

        # Check if types have been confirmed
        if not st.session_state.get('types_confirmed', False):
            st.warning("⚠️ Please upload data and confirm types & target selection first in the 'Upload Data' section.")
            return

        # Check if we have processed data
        if 'processed_df' not in st.session_state:
            st.error("❌ No processed data found. Please go back to 'Upload Data' and confirm your type selections.")
            return

        df = st.session_state['processed_df']
        target_column = st.session_state.get('target_column')

        st.success(f"✅ Using processed data with {len(df)} rows, {len(df.columns)} columns")
        st.info(f"🎯 Target column: `{target_column}`")

        # TODO: Implement validation logic with Great Expectations using the processed dataframe
        st.info("🚧 Data validation functionality to be implemented")
    else:
        # Inline validation after type confirmation
        st.markdown("---")
        st.subheader("🔍 Data Validation")
        st.write("Running automatic validation with Great Expectations...")

        # Initialize validation state
        if 'validation_results' not in st.session_state:
            st.session_state['validation_results'] = None
        if 'validation_override' not in st.session_state:
            st.session_state['validation_override'] = False

        # Run validation button
        if st.button("🚀 Run Data Validation", type="primary"):
            with st.spinner("Running validation..."):
                try:
                    # Run the validation process
                    success, results = run_data_validation(df)
                    st.session_state['validation_results'] = results

                    if success:
                        st.success("✅ **Validation PASSED!** All data quality checks passed.")
                        st.balloons()
                    else:
                        st.error("❌ **Validation FAILED!** Some data quality issues were found.")

                except Exception as e:
                    st.error(f"❌ Error during validation: {str(e)}")
                    st.session_state['validation_results'] = {"error": str(e)}

        # Display validation results
        if st.session_state['validation_results']:
            display_validation_results(st.session_state['validation_results'])


def run_data_validation(df):
    """Run Great Expectations validation on the DataFrame."""
    try:
        # Create expectation suite based on user-confirmed types in the DataFrame
        expectations = setup_expectations.create_typed_expectation_suite(df, "streamlit_typed_validation_suite")

        # Convert to proper format for validation
        expectation_suite = {
            "expectation_suite_name": "streamlit_typed_validation_suite",
            "expectations": expectations
        }

        # Run validation
        success, results = run_validation.validate_dataframe_with_suite(df, expectation_suite)

        return success, results

    except Exception as e:
        return False, {"error": str(e)}


def display_validation_results(results):
    """Display validation results in a user-friendly format."""
    if "error" in results:
        st.error(f"Validation error: {results['error']}")
        return

    # Overall status
    overall_success = results.get('overall_success', False)
    total_expectations = results.get('total_expectations', 0)
    successful_expectations = results.get('successful_expectations', 0)
    failed_expectations = results.get('failed_expectations', 0)

    # Status badge
    if overall_success:
        st.success(f"🎉 **Validation Status: PASSED** ({successful_expectations}/{total_expectations} checks passed)")
    else:
        st.error(f"⚠️ **Validation Status: FAILED** ({successful_expectations}/{total_expectations} checks passed)")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Checks", total_expectations)
    with col2:
        st.metric("Passed", successful_expectations, delta=None if overall_success else f"-{failed_expectations}")
    with col3:
        success_rate = successful_expectations / total_expectations if total_expectations > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1%}")

    # Detailed results
    if not overall_success and 'expectation_results' in results:
        st.subheader("🔍 Failed Validation Details")

        failed_results = [r for r in results['expectation_results'] if not r.get('success', True)]

        for i, result in enumerate(failed_results):
            with st.expander(f"❌ Failed Check {i+1}: {result.get('expectation_type', 'Unknown')}", expanded=False):
                st.write(f"**Details:** {result.get('details', 'No details available')}")
                if 'kwargs' in result:
                    st.write(f"**Parameters:** {result['kwargs']}")

        # Override option
        st.markdown("---")
        st.subheader("⚠️ Validation Override")
        st.write("The data validation has failed, but you can choose to proceed anyway.")

        if st.checkbox("🚨 I acknowledge the validation issues and want to proceed anyway"):
            st.session_state['validation_override'] = True
            st.warning("✅ Validation override enabled. You can proceed to EDA and cleaning steps.")
            st.success("🚀 **Ready for next steps (EDA/Cleaning)**")
        else:
            st.session_state['validation_override'] = False
    else:
        # All validations passed
        st.success("🚀 **Ready for next steps (EDA/Cleaning)**")

    # Save results to file
    try:
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = processed_dir / f"validation_report_{timestamp}.json"

        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        st.info(f"📄 Validation report saved to: `{results_path}`")

    except Exception as e:
        st.warning(f"⚠️ Could not save validation report: {str(e)}")


def type_override_section(df):
    """Handle type override and target selection UI."""
    st.subheader("🎯 Type Override & Target Selection")
    st.write("Review and adjust data types, then select your target variable for ML.")

    # Initialize session state for type overrides if not exists
    if 'type_overrides' not in st.session_state:
        st.session_state['type_overrides'] = {}
    if 'target_column' not in st.session_state:
        st.session_state['target_column'] = None
    if 'types_confirmed' not in st.session_state:
        st.session_state['types_confirmed'] = False

    # Define available data types for dropdown
    available_types = [
        'string/object',
        'integer',
        'float',
        'boolean',
        'category',
        'datetime'
    ]

    # Map pandas dtypes to our simplified types
    def pandas_to_simple_type(dtype_str):
        dtype_str = str(dtype_str).lower()
        if 'int' in dtype_str:
            return 'integer'
        elif 'float' in dtype_str:
            return 'float'
        elif 'bool' in dtype_str:
            return 'boolean'
        elif 'datetime' in dtype_str:
            return 'datetime'
        elif 'category' in dtype_str:
            return 'category'
        else:
            return 'string/object'

    # Display the type override table
    st.write("**Column Type Configuration:**")

    # Create columns for the table header
    header_cols = st.columns([3, 2, 3, 2])
    with header_cols[0]:
        st.write("**Column Name**")
    with header_cols[1]:
        st.write("**Inferred Type**")
    with header_cols[2]:
        st.write("**New Type**")
    with header_cols[3]:
        st.write("**Is Target?**")

    st.markdown("---")

    # Create rows for each column
    for idx, column in enumerate(df.columns):
        col_row = st.columns([3, 2, 3, 2])

        with col_row[0]:
            st.write(f"`{column}`")

        with col_row[1]:
            inferred_type = pandas_to_simple_type(df[column].dtype)
            st.write(inferred_type)

        with col_row[2]:
            # Get current override type or default to inferred type
            current_type = st.session_state['type_overrides'].get(column, inferred_type)
            new_type = st.selectbox(
                "Select type",
                available_types,
                index=available_types.index(current_type) if current_type in available_types else 0,
                key=f"type_{column}_{idx}",
                label_visibility="collapsed"
            )
            st.session_state['type_overrides'][column] = new_type

        with col_row[3]:
            is_target = st.radio(
                "Target",
                ["No", "Yes"],
                index=1 if st.session_state['target_column'] == column else 0,
                key=f"target_{column}_{idx}",
                horizontal=True,
                label_visibility="collapsed"
            )

            # Update target column in session state
            if is_target == "Yes":
                st.session_state['target_column'] = column
            elif st.session_state['target_column'] == column and is_target == "No":
                st.session_state['target_column'] = None

    st.markdown("---")

    # Display current selections summary
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state['target_column']:
            st.success(f"🎯 **Target Column:** `{st.session_state['target_column']}`")
        else:
            st.warning("⚠️ **No target column selected**")

    with col2:
        type_changes = sum(1 for col in df.columns
                          if st.session_state['type_overrides'].get(col) != pandas_to_simple_type(df[col].dtype))
        if type_changes > 0:
            st.info(f"📝 **Type changes:** {type_changes} columns")
        else:
            st.info("📝 **No type changes**")

    # Confirmation button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("✅ Confirm Types & Target", type="primary"):
            if st.session_state['target_column'] is None:
                st.error("❌ Please select a target column before confirming.")
            else:
                # Apply type conversions to the dataframe
                success, updated_df = apply_type_conversions(df, st.session_state['type_overrides'])

                if success:
                    st.session_state['types_confirmed'] = True
                    st.session_state['processed_df'] = updated_df
                    st.success("✅ **Data types and target column confirmed!**")

                    # Display summary of changes
                    with st.expander("📋 View Applied Changes", expanded=False):
                        changes_df = pd.DataFrame({
                            'Column': df.columns,
                            'Original Type': [pandas_to_simple_type(df[col].dtype) for col in df.columns],
                            'New Type': [st.session_state['type_overrides'].get(col, pandas_to_simple_type(df[col].dtype)) for col in df.columns],
                            'Is Target': ['✅' if col == st.session_state['target_column'] else '' for col in df.columns]
                        })
                        st.dataframe(changes_df, use_container_width=True)

                    # Automatically run validation
                    st.markdown("---")
                    st.subheader("🔍 Running Data Validation...")
                    with st.spinner("Validation running..."):
                        validation_success, validation_results = run_data_validation(updated_df)
                        st.session_state['validation_results'] = validation_results
                        st.session_state['validation_success'] = validation_success

                    # Handle validation results
                    if validation_success:
                        st.success("🎉 **Validation PASSED!** All data quality checks passed.")
                        # Auto-trigger cleaning
                        with st.spinner("Cleaning data..."):
                            # Run auto-cleaning with default parameters
                            df_clean, report = cleaning.clean_dataframe(
                                updated_df,
                                missing_strategy="drop",
                                missing_threshold=0.5,
                                standardize_columns=True,
                                remove_dups=True,
                                convert_dtypes=True,
                                target_column=st.session_state['target_column']
                            )
                            st.session_state['cleaned_df'] = df_clean
                            st.session_state['cleaning_report'] = report
                            st.session_state['cleaning_done'] = True
                        
                        st.success("✅ **Auto-cleaning completed!** Proceeding to final EDA...")
                        st.session_state.current_stage = "final_eda"
                        st.rerun()
                    else:
                        # Show validation failure with options
                        st.error("❌ **Validation FAILED!** Some data quality issues were found.")
                        display_validation_results(validation_results)
                        
                        st.markdown("---")
                        st.subheader("⚠️ Choose how to proceed:")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("🔙 **Abort** - Start Over", type="secondary", key="abort_validation"):
                                # Reset pipeline to upload stage
                                keys_to_clear = ['uploaded_df', 'processed_df', 'types_confirmed', 'type_overrides', 
                                               'target_column', 'validation_results', 'validation_success', 
                                               'cleaned_df', 'cleaning_report', 'cleaning_done']
                                for key in keys_to_clear:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.session_state.current_stage = "upload"
                                st.rerun()
                        
                        with col2:
                            if st.button("⚠️ **Proceed Anyway**", type="primary", key="proceed_validation"):
                                st.session_state['validation_override'] = True

    # Show confirmation status
    if st.session_state.get('types_confirmed', False):
        st.success("✅ Types and target confirmed! You can now proceed to EDA Profiling in the sidebar.")


def apply_type_conversions(df, type_overrides):
    """Apply type conversions to the dataframe with error handling."""
    try:
        updated_df = df.copy()

        for column, new_type in type_overrides.items():
            if column not in df.columns:
                continue

            try:
                if new_type == 'integer':
                    # Handle potential NaN values for integer conversion
                    if updated_df[column].isnull().any():
                        # Convert to nullable integer type
                        updated_df[column] = pd.to_numeric(updated_df[column], errors='coerce').astype('Int64')
                    else:
                        updated_df[column] = pd.to_numeric(updated_df[column], errors='coerce').astype('int64')

                elif new_type == 'float':
                    updated_df[column] = pd.to_numeric(updated_df[column], errors='coerce').astype('float64')

                elif new_type == 'boolean':
                    # Convert to boolean, handling common string representations
                    updated_df[column] = updated_df[column].astype(str).str.lower()
                    updated_df[column] = updated_df[column].replace({
                        'true': True, 'false': False, '1': True, '0': False,
                        'yes': True, 'no': False, 't': True, 'f': False
                    }).astype('boolean')

                elif new_type == 'category':
                    updated_df[column] = updated_df[column].astype('category')

                elif new_type == 'datetime':
                    updated_df[column] = pd.to_datetime(updated_df[column], errors='coerce')

                elif new_type == 'string/object':
                    updated_df[column] = updated_df[column].astype('object')

            except Exception as e:
                st.warning(f"⚠️ Could not convert column '{column}' to {new_type}: {str(e)}")
                # Keep original type if conversion fails
                continue

        return True, updated_df

    except Exception as e:
        st.error(f"❌ Error during type conversion: {str(e)}")
        return False, df


def upload_pipeline_section():
    """Upload section for the main pipeline flow."""
    st.header("📁 Upload File")
    st.write("Upload your CSV file to start the pipeline.")

    # Create data/raw directory if it doesn't exist
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file (max 200MB)"
    )

    if uploaded_file is not None:
        try:
            # Validate file size (200MB limit)
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 200:
                st.error(f"❌ File too large: {file_size_mb:.1f}MB. Maximum allowed: 200MB")
                return

            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {file_size_mb:.2f}MB")

            # Try to read the CSV file
            try:
                df = pd.read_csv(uploaded_file)

                # Display basic info about the dataset
                st.subheader("📋 Dataset Overview")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f}KB")

                # Show column info
                st.subheader("📝 Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Null %': (df.isnull().sum() / len(df) * 100).round(2)
                })
                st.dataframe(col_info, use_container_width=True)

                # Display preview of the data
                st.subheader("👀 Data Preview")
                preview_rows = st.slider("Number of rows to preview", 5, min(50, len(df)), 10)
                st.dataframe(df.head(preview_rows), use_container_width=True)

                # Show success message and automatically continue with type override
                st.success("✅ **Data uploaded successfully!**")
                
                # Store the uploaded data
                st.session_state['uploaded_df'] = df
                st.session_state['filename'] = uploaded_file.name
                
                # Show basic EDA summary
                st.markdown("---")
                st.subheader("📊 Data Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                    st.metric("Missing Values", f"{missing_pct:.1f}%")
                
                # Automatically show type override section
                continuous_type_override_section(df)

            except pd.errors.EmptyDataError:
                st.error("❌ The uploaded file is empty or contains no data.")
            except pd.errors.ParserError as e:
                st.error(f"❌ Error parsing CSV file: {str(e)}")
                st.info("💡 Please ensure the file is a valid CSV format.")
            except UnicodeDecodeError:
                st.error("❌ Error reading file encoding. Please ensure the file uses UTF-8 encoding.")
            except Exception as e:
                st.error(f"❌ Unexpected error reading file: {str(e)}")

        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {str(e)}")


def basic_eda_pipeline_section():
    """Basic EDA section for the main pipeline flow."""
    st.header("📊 Step 2: Basic Data Analysis")

    if 'uploaded_df' not in st.session_state:
        st.error("❌ No data found. Please go back to upload a CSV file.")
        if st.button("🔙 Back to Upload"):
            st.session_state.current_stage = "upload"
            st.rerun()
        return

    df = st.session_state['uploaded_df']

    st.write("Quick statistical overview of your data:")

    # Basic statistics
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(include='all'), use_container_width=True)

    # Missing values analysis
    st.subheader("❓ Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

    if len(missing_data) > 0:
        st.write("Columns with missing values:")
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': (missing_data.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ No missing values found!")

    # Data types overview
    st.subheader("🔢 Data Types Overview")
    dtype_counts = df.dtypes.value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Type Distribution:**")
        for dtype, count in dtype_counts.items():
            st.write(f"- {dtype}: {count} columns")

    with col2:
        # Potential issues detection
        st.write("**Potential Issues:**")
        issues = []

        # Check for high cardinality object columns
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                issues.append(f"High cardinality in '{col}' ({unique_ratio:.1%} unique)")

        # Check for very sparse columns
        for col in df.columns:
            null_ratio = df[col].isnull().sum() / len(df)
            if null_ratio > 0.5:
                issues.append(f"Sparse column '{col}' ({null_ratio:.1%} missing)")

        if issues:
            for issue in issues:
                st.warning(f"⚠️ {issue}")
        else:
            st.success("✅ No obvious data quality issues detected")

    # Continue button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Continue to Type Override", type="primary"):
            st.session_state.current_stage = "type_override"
            st.rerun()


def type_override_pipeline_section():
    """Type override section for the main pipeline flow with continuous validation and cleaning."""
    st.header("🎯 Type Override & Target Selection")

    if 'uploaded_df' not in st.session_state:
        st.error("❌ No data found. Please start from the beginning.")
        if st.button("🔙 Back to Upload"):
            st.session_state.current_stage = "upload"
            st.rerun()
        return

    df = st.session_state['uploaded_df']

    # Show type override section
    continuous_type_override_section(df)


def continuous_type_override_section(df):
    """Continuous type override section with validation, cleaning, and final results on same page."""
    # Initialize session state for type overrides if not exists
    if 'type_overrides' not in st.session_state:
        st.session_state['type_overrides'] = {}
    if 'target_column' not in st.session_state:
        st.session_state['target_column'] = None
    if 'types_confirmed' not in st.session_state:
        st.session_state['types_confirmed'] = False

    # Define available data types for dropdown
    available_types = [
        'string/object',
        'integer',
        'float',
        'boolean',
        'category',
        'datetime'
    ]

    # Map pandas dtypes to our simplified types
    def pandas_to_simple_type(dtype_str):
        dtype_str = str(dtype_str).lower()
        if 'int' in dtype_str:
            return 'integer'
        elif 'float' in dtype_str:
            return 'float'
        elif 'bool' in dtype_str:
            return 'boolean'
        elif 'datetime' in dtype_str:
            return 'datetime'
        elif 'category' in dtype_str:
            return 'category'
        else:
            return 'string/object'

    # Always show the type configuration section (even if completed)
    st.markdown("---")
    st.subheader("🎯 Configure Data Types & Target")
    
    # Show completion status if types are confirmed
    if st.session_state.get('types_confirmed', False):
        st.success("✅ **Types and target already confirmed!**")
        
        # Show summary of confirmed settings
        target_col = st.session_state.get('target_column')
        st.info(f"🎯 **Target Variable:** `{target_col}`")
        
        # Show any type changes made
        type_changes = sum(1 for col in df.columns
                          if st.session_state['type_overrides'].get(col) != pandas_to_simple_type(df[col].dtype))
        if type_changes > 0:
            st.info(f"📝 **Type changes applied:** {type_changes} columns")
            
            # Show the changes in an expander
            with st.expander("📋 View Applied Type Changes", expanded=False):
                changes_df = pd.DataFrame({
                    'Column': df.columns,
                    'Original Type': [pandas_to_simple_type(df[col].dtype) for col in df.columns],
                    'Applied Type': [st.session_state['type_overrides'].get(col, pandas_to_simple_type(df[col].dtype)) for col in df.columns],
                    'Is Target': ['✅' if col == target_col else '' for col in df.columns]
                })
                st.dataframe(changes_df, use_container_width=True)
        else:
            st.info("📝 **No type changes** were needed")
    
    else:
        # Show type override UI if not confirmed yet
        st.write("Select your target variable and review data types for optimal ML performance.")

        # Suggest target variable if not already set
        if st.session_state['target_column'] is None:
            # Smart target suggestion: look for likely target columns
            suggested_target = None
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['target', 'label', 'class', 'outcome', 'result', 'prediction', 'survived', 'price', 'salary', 'score']):
                    suggested_target = col
                    break
            
            # If no keyword match, suggest the last column
            if suggested_target is None:
                suggested_target = df.columns[-1]
            
            st.session_state['target_column'] = suggested_target

        # Target variable selection at the top
        st.write("**🎯 Target Variable Selection:**")
        target_options = list(df.columns)
        current_target_idx = target_options.index(st.session_state['target_column']) if st.session_state['target_column'] in target_options else 0
        
        selected_target = st.selectbox(
            "Choose the column you want to predict (target variable):",
            target_options,
            index=current_target_idx,
            help="This is the variable your ML model will learn to predict"
        )
        st.session_state['target_column'] = selected_target
        
        # Show target column info
        target_col = df[selected_target]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Type", pandas_to_simple_type(target_col.dtype))
        with col2:
            st.metric("Unique Values", target_col.nunique())
        with col3:
            st.metric("Missing Values", target_col.isnull().sum())

        st.markdown("---")

        # Data types configuration table (simplified)
        st.write("**📝 Data Types Configuration:**")
        st.write("Review and adjust data types if needed. Most columns should be correctly detected.")

        # Show only columns that might need type changes or are important
        columns_to_show = []
        for col in df.columns:
            if col == selected_target:
                continue  # Skip target column in type override
                
            inferred_type = pandas_to_simple_type(df[col].dtype)
            # Show columns that might need attention
            if (inferred_type == 'string/object' and df[col].nunique() < 20) or \
               (inferred_type == 'float' and df[col].nunique() < 10):
                columns_to_show.append(col)
            elif df[col].isnull().sum() > 0:  # Show columns with missing values
                columns_to_show.append(col)
        
        # If no special columns, show first few columns
        if not columns_to_show:
            columns_to_show = [col for col in df.columns[:5] if col != selected_target]

        # Display type override table
        if columns_to_show:
            # Create columns for the table header
            header_cols = st.columns([3, 2, 3])
            with header_cols[0]:
                st.write("**Column Name**")
            with header_cols[1]:
                st.write("**Detected Type**")
            with header_cols[2]:
                st.write("**Override Type**")

            st.markdown("---")

            # Create rows for important columns
            for idx, column in enumerate(columns_to_show):
                col_row = st.columns([3, 2, 3])

                with col_row[0]:
                    st.write(f"`{column}`")
                    # Show some sample values
                    sample_vals = df[column].dropna().iloc[:3].astype(str).tolist()
                    if sample_vals:
                        st.caption(f"Sample: {', '.join(sample_vals[:2])}...")

                with col_row[1]:
                    inferred_type = pandas_to_simple_type(df[column].dtype)
                    st.write(inferred_type)

                with col_row[2]:
                    # Get current override type or default to inferred type
                    current_type = st.session_state['type_overrides'].get(column, inferred_type)
                    new_type = st.selectbox(
                        "Select type",
                        available_types,
                        index=available_types.index(current_type) if current_type in available_types else 0,
                        key=f"type_{column}_{idx}",
                        label_visibility="collapsed"
                    )
                    st.session_state['type_overrides'][column] = new_type

            # Show expandable section for all other columns
            with st.expander("⚙️ Advanced: Override types for other columns", expanded=False):
                other_columns = [col for col in df.columns if col not in columns_to_show and col != selected_target]
                if other_columns:
                    st.write("**All other columns (usually don't need changes):**")
                    for idx, column in enumerate(other_columns):
                        col_row = st.columns([3, 2, 3])
                        with col_row[0]:
                            st.write(f"`{column}`")
                        with col_row[1]:
                            inferred_type = pandas_to_simple_type(df[column].dtype)
                            st.write(inferred_type)
                        with col_row[2]:
                            current_type = st.session_state['type_overrides'].get(column, inferred_type)
                            new_type = st.selectbox(
                                "Select type",
                                available_types,
                                index=available_types.index(current_type) if current_type in available_types else 0,
                                key=f"type_other_{column}_{idx}",
                                label_visibility="collapsed"
                            )
                            st.session_state['type_overrides'][column] = new_type

        st.markdown("---")

        # Display current selections summary
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🎯 **Target Column:** `{st.session_state['target_column']}`")

        with col2:
            type_changes = sum(1 for col in df.columns
                              if st.session_state['type_overrides'].get(col) != pandas_to_simple_type(df[col].dtype))
            if type_changes > 0:
                st.info(f"📝 **Type changes:** {type_changes} columns")
            else:
                st.info("📝 **No type changes**")

        # Confirmation button
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            confirm_clicked = st.button("✅ Confirm Types & Target", type="primary")
        
        # Handle confirmation outside of column layout to avoid width constraints
        if confirm_clicked and not st.session_state.get('types_confirmed', False):
            if st.session_state['target_column'] is None:
                st.error("❌ Please select a target column before confirming.")
            else:
                # Apply type conversions to the dataframe
                success, updated_df = apply_type_conversions(df, st.session_state['type_overrides'])

                if success:
                    st.session_state['types_confirmed'] = True
                    st.session_state['processed_df'] = updated_df
                    
                    # Immediately show validation without page refresh - outside column layout
                    st.markdown("---")
                    st.subheader("🔍 Running Data Validation...")
                    with st.spinner("Validating data quality..."):
                        validation_success, validation_results = run_data_validation(updated_df)
                        st.session_state['validation_results'] = validation_results
                        st.session_state['validation_success'] = validation_success
                    
                    # Show validation results and cleaning options immediately
                    show_validation_and_cleaning_results(updated_df)
                else:
                    st.error("❌ Error applying type conversions. Please check your type selections.")

    # Show validation and cleaning results if types are confirmed (for subsequent interactions)
    if st.session_state.get('types_confirmed', False):
        updated_df = st.session_state['processed_df']
        
        # If validation results exist, show them
        if 'validation_results' in st.session_state:
            show_validation_and_cleaning_results(updated_df)
        else:
            # This shouldn't happen in normal flow, but handle it gracefully
            st.markdown("---")
            st.subheader("🔍 Running Data Validation...")
            with st.spinner("Validating data quality..."):
                validation_success, validation_results = run_data_validation(updated_df)
                st.session_state['validation_results'] = validation_results
                st.session_state['validation_success'] = validation_success
            show_validation_and_cleaning_results(updated_df)


def show_validation_and_cleaning_results(df):
    """Show validation results and cleaning options on the same page."""
    validation_success = st.session_state.get('validation_success', False)
    validation_results = st.session_state.get('validation_results', {})

    # Show validation results
    st.subheader("🔍 Data Validation Results")
    
    if validation_success:
        st.success("🎉 **Validation PASSED!** All data quality checks passed.")
        total = validation_results.get('total_expectations', 0)
        passed = validation_results.get('successful_expectations', 0)
        st.info(f"✅ All {passed}/{total} validation checks passed")
    else:
        st.error("❌ **Validation FAILED!** Some data quality issues were found.")
        display_validation_results(validation_results)
        
        # Show abort option for validation failure
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔙 **Abort** - Start Over", type="secondary", key="abort_validation"):
                # Reset pipeline to upload stage
                keys_to_clear = ['uploaded_df', 'processed_df', 'types_confirmed', 'type_overrides', 
                               'target_column', 'validation_results', 'validation_success', 
                               'cleaned_df', 'cleaning_report', 'cleaning_done']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.current_stage = "upload"
                st.rerun()
        with col2:
            if st.button("⚠️ **Proceed Anyway**", type="primary", key="proceed_validation"):
                st.session_state['validation_override'] = True

    # Show cleaning options if validation passed or user chose to proceed
    if validation_success or st.session_state.get('validation_override', False):
        show_cleaning_section(df)

    # Show final results if cleaning is done
    if st.session_state.get('cleaning_done', False):
        show_final_results_section()


def show_cleaning_section(df):
    """Show cleaning options with auto/manual toggle."""
    st.markdown("---")
    st.subheader("🧹 Data Cleaning")
    
    # Auto vs Manual toggle
    use_auto_cleaning = st.toggle("🤖 Use Automatic Cleaning", value=True, 
                                  help="Enable to use default cleaning parameters, disable to customize")
    
    if use_auto_cleaning:
        st.info("✨ **Automatic cleaning** will use default parameters: drop missing values (50% threshold), standardize column names, remove duplicates, auto-convert data types.")
        
        if st.button("🧹 **Run Automatic Cleaning**", type="primary", key="run_auto_cleaning"):
            with st.spinner("Cleaning data with automatic parameters..."):
                df_clean, report = cleaning.clean_dataframe(
                    df,
                    missing_strategy="drop",
                    missing_threshold=0.5,
                    standardize_columns=True,
                    remove_dups=True,
                    convert_dtypes=True,
                    target_column=st.session_state['target_column']
                )
                st.session_state['cleaned_df'] = df_clean
                st.session_state['cleaning_report'] = report
                st.session_state['cleaning_done'] = True
            
            # Show final results immediately
            show_final_results_section()
    
    else:
        st.info("⚙️ **Manual cleaning** - customize the parameters below:")
        
        # Create columns for options
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Missing Values Handling**")
            missing_strategy = st.selectbox(
                "Strategy",
                ["drop", "fill_mean", "fill_median", "fill_mode", "forward_fill"],
                help="How to handle missing values in the dataset"
            )

            missing_threshold = st.slider(
                "Missing Value Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Drop columns with more than this fraction of missing values"
            )

        with col2:
            st.write("**Additional Cleaning Steps**")
            standardize_columns = st.checkbox(
                "Standardize Column Names",
                value=True,
                help="Convert column names to snake_case"
            )

            remove_dups = st.checkbox(
                "Remove Duplicates",
                value=True,
                help="Remove duplicate rows"
            )

            convert_dtypes = st.checkbox(
                "Auto-convert Data Types",
                value=True,
                help="Automatically detect and convert data types"
            )

        if st.button("🧹 **Run Custom Cleaning**", type="primary", key="run_custom_cleaning"):
            with st.spinner("Cleaning data with custom parameters..."):
                df_clean, report = cleaning.clean_dataframe(
                    df,
                    missing_strategy=missing_strategy,
                    missing_threshold=missing_threshold,
                    standardize_columns=standardize_columns,
                    remove_dups=remove_dups,
                    convert_dtypes=convert_dtypes,
                    target_column=st.session_state['target_column']
                )
                st.session_state['cleaned_df'] = df_clean
                st.session_state['cleaning_report'] = report
                st.session_state['cleaning_done'] = True
            
            # Show final results immediately
            show_final_results_section()


def show_final_results_section():
    """Show final results and download options."""
    st.markdown("---")
    st.subheader("📋 Final Data Summary & Export")
    
    df = st.session_state['cleaned_df']
    cleaning_report = st.session_state.get('cleaning_report', {})
    target_column = cleaning_report.get('target_column_renamed') or st.session_state.get('target_column')

    st.success("🎉 **Data pipeline completed successfully!**")

    # Final summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Rows", len(df))
    with col2:
        st.metric("Final Columns", len(df.columns))
    with col3:
        st.metric("Target Column", target_column or "None")
    with col4:
        validation_status = "✅ Passed" if st.session_state.get('validation_success', False) else "⚠️ Override"
        st.metric("Validation", validation_status)

    # Data preview
    st.subheader("👀 Final Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Cleaning report
    with st.expander("📋 View Cleaning Report", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Rows",
                cleaning_report['final_shape'][0],
                delta=cleaning_report['rows_removed']
            )
        with col2:
            st.metric(
                "Columns", 
                cleaning_report['final_shape'][1],
                delta=cleaning_report['columns_removed']
            )
        if target_column:
            target_preserved = cleaning_report.get('target_column_preserved', False)
            if target_preserved:
                st.success(f"✅ Target column '{target_column}' was preserved")
            else:
                st.error(f"❌ Target column '{target_column}' was not preserved!")
        st.write("**Steps Performed:**")
        for step in cleaning_report['steps_performed']:
            st.write(f"- {step.replace('_', ' ').title()}")

    # Download section
    st.subheader("💾 Download Cleaned Data & Report")
    st.write("This cleaned data and report are ready for Team B's modeling pipeline.")

    # Save files
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = processed_dir / "final_cleaned_data.csv"
    json_path = processed_dir / "final_cleaned_report.json"
    df.to_csv(csv_path, index=False)

    # Generate schema and final report
    schema = cleaning.generate_schema(df, target_column=target_column)
    final_report = {
        "schema": schema,
        "cleaning_report": cleaning_report,
        "target_column": target_column,
        "original_filename": st.session_state.get('filename', 'unknown'),
        "final_rows": len(df),
        "final_columns": len(df.columns),
        "validation_passed": st.session_state.get('validation_success', False),
        "pipeline_completed": True,
        "timestamp": datetime.now().isoformat()
    }
    
    import json
    with open(json_path, 'w') as f:
        json.dump(make_json_serializable(final_report), f, indent=2)

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        with open(csv_path, 'rb') as f:
            st.download_button(
                label="📥 Download final_cleaned_data.csv",
                data=f,
                file_name="final_cleaned_data.csv",
                mime="text/csv",
                type="primary",
                key="download_csv_final"
            )
    with col2:
        with open(json_path, 'rb') as f:
            st.download_button(
                label="📄 Download final_cleaned_report.json",
                data=f,
                file_name="final_cleaned_report.json",
                mime="application/json",
                key="download_json_final"
            )

    # Option to start over
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Start Over", type="secondary", key="start_over_final"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.current_stage = "upload"
            st.rerun()


def validation_pipeline_section():
    """Validation section for the main pipeline flow."""
    st.header("✅ Step 3: Data Validation")
    
    # This step is now handled automatically in type override
    st.info("🔄 **Validation is handled automatically** after type confirmation in the previous step.")
    
    if 'validation_results' in st.session_state:
        results = st.session_state['validation_results']
        success = st.session_state.get('validation_success', False)
        
        if success:
            st.success("🎉 **Validation completed successfully!**")
            total = results.get('total_expectations', 0)
            passed = results.get('successful_expectations', 0)
            st.info(f"✅ All {passed}/{total} validation checks passed")
        else:
            st.warning("⚠️ **Validation had issues** but pipeline continued with override.")
            display_validation_results(results)
    else:
        st.warning("⚠️ No validation results found. Please go back to type override.")
        
    # Redirect to appropriate stage
    if st.button("🔙 Back to Type Override"):
        st.session_state.current_stage = "type_override"
        st.rerun()


def cleaning_pipeline_section():
    """Data cleaning section for the main pipeline flow."""
    st.header("🧹 Step 4: Data Cleaning")
    
    # This step is now handled automatically after validation
    st.info("🔄 **Cleaning is handled automatically** after validation in the type override step.")
    
    # If cleaning is done, show results and redirect to Final EDA
    if st.session_state.get('cleaning_done', False):
        st.success("✅ **Data cleaning completed automatically!**")
        
        if 'cleaning_report' in st.session_state:
            report = st.session_state['cleaning_report']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Rows", report['final_shape'][0])
            with col2:
                st.metric("Final Columns", report['final_shape'][1])
        
        if st.button("📋 Continue to Final EDA", type="primary"):
            st.session_state.current_stage = 'final_eda'
            st.rerun()
        return
    
    # If not done, redirect back
    st.warning("⚠️ Cleaning not completed yet. Please go back to type override.")
    if st.button("🔙 Back to Type Override"):
        st.session_state.current_stage = "type_override"
        st.rerun()
        return


def make_json_serializable(obj):
    """Recursively convert pandas/numpy types and pandas extension dtypes to native Python types for JSON serialization."""
    import pandas as pd
    import numpy as np
    from pandas.api.types import is_extension_array_dtype

    # Handle pandas extension dtypes and numpy dtypes
    if isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif hasattr(obj, 'item') and callable(obj.item):
        return obj.item()
    # Handle pandas and numpy dtype objects
    elif isinstance(obj, (np.dtype, pd.api.extensions.ExtensionDtype)):
        return str(obj)
    # Handle pandas NA (pd.NA)
    elif obj is pd.NA:
        return None
    # Handle pandas extension scalar types (like pd.Int64Dtype type values)
    elif is_extension_array_dtype(type(obj)):
        return obj if obj is not pd.NA else None
    # Handle pandas DataFrame
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    # Handle any object whose class name ends with 'Dtype'
    elif obj is not None and hasattr(obj, '__class__') and obj.__class__.__name__.endswith('Dtype'):
        return str(obj)
    # As a last resort, convert any non-JSON-native type to string
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj


def upload_data_section():
    """Handle data upload and initial inspection."""
    st.header("📁 Data Upload")
    st.write("Upload your CSV file to start the pipeline.")

    # Create data/raw directory if it doesn't exist
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file (max 200MB)"
    )

    if uploaded_file is not None:
        try:
            # Validate file size (200MB limit)
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 200:
                st.error(f"❌ File too large: {file_size_mb:.1f}MB. Maximum allowed: 200MB")
                return

            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {file_size_mb:.2f}MB")

            # Try to read the CSV file
            try:
                df = pd.read_csv(uploaded_file)

                # Display basic info about the dataset
                st.subheader("📋 Dataset Overview")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f}KB")

                # Show column info
                st.subheader("📝 Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Null %': (df.isnull().sum() / len(df) * 100).round(2)
                })
                st.dataframe(col_info, use_container_width=True)

                # Display preview of the data
                st.subheader("👀 Data Preview")
                preview_rows = st.slider("Number of rows to preview", 5, min(50, len(df)), 10)
                st.dataframe(df.head(preview_rows), use_container_width=True)

                # Type Override UI - NEW SECTION
                type_override_section(df)

                # Save file option
                st.subheader("💾 Save to Raw Data Directory")

                # Generate filename with timestamp to avoid conflicts
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                original_name = Path(uploaded_file.name).stem
                extension = Path(uploaded_file.name).suffix
                suggested_filename = f"{original_name}_{timestamp}{extension}"

                filename = st.text_input(
                    "Filename for saved data:",
                    value=suggested_filename,
                    help="File will be saved to data/raw/ directory"
                )

                if st.button("💾 Save File", type="primary"):
                    try:
                        # Ensure filename ends with .csv
                        if not filename.endswith('.csv'):
                            filename += '.csv'

                        save_path = raw_data_dir / filename

                        # Use processed df if types have been confirmed, otherwise use original df
                        df_to_save = st.session_state.get('processed_df', df)

                        # Check if file already exists
                        if save_path.exists():
                            if st.checkbox("⚠️ File exists. Overwrite?"):
                                df_to_save.to_csv(save_path, index=False)
                                st.success(f"✅ File saved successfully to: {save_path}")

                                # Store filename in session state for other sections
                                st.session_state['current_dataset'] = str(save_path)
                                st.session_state['current_df'] = df_to_save
                        else:
                            df_to_save.to_csv(save_path, index=False)
                            st.success(f"✅ File saved successfully to: {save_path}")

                            # Store filename in session state for other sections
                            st.session_state['current_dataset'] = str(save_path)
                            st.session_state['current_df'] = df_to_save

                    except Exception as e:
                        st.error(f"❌ Error saving file: {str(e)}")

            except pd.errors.EmptyDataError:
                st.error("❌ The uploaded file is empty or contains no data.")
            except pd.errors.ParserError as e:
                st.error(f"❌ Error parsing CSV file: {str(e)}")
                st.info("💡 Please ensure the file is a valid CSV format.")
            except UnicodeDecodeError:
                st.error("❌ Error reading file encoding. Please ensure the file uses UTF-8 encoding.")
            except Exception as e:
                st.error(f"❌ Unexpected error reading file: {str(e)}")

        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {str(e)}")


def eda_profiling_section():
    """Handle EDA profiling with ydata-profiling."""
    st.header("📊 Exploratory Data Analysis")
    st.write("Generate comprehensive data profiling report.")

    # Check if types have been confirmed
    if not st.session_state.get('types_confirmed', False):
        st.warning("⚠️ Please upload data and confirm types & target selection first in the 'Upload Data' section.")
        return

    # Check if we have processed data
    if 'processed_df' not in st.session_state:
        st.error("❌ No processed data found. Please go back to 'Upload Data' and confirm your type selections.")
        return

    df = st.session_state['processed_df']
    target_column = st.session_state.get('target_column')

    st.success(f"✅ Using processed data with {len(df)} rows, {len(df.columns)} columns")
    st.info(f"🎯 Target column: `{target_column}`")

    # TODO: Implement EDA profiling with ydata-profiling using the processed dataframe
    st.info("🚧 EDA profiling functionality to be implemented")


def cleaning_section():
    """Handle data cleaning operations."""
    st.header("🧹 Data Cleaning")
    st.write("Clean and preprocess the data.")

    # Check if types have been confirmed
    if not st.session_state.get('types_confirmed', False):
        st.warning("⚠️ Please upload data and confirm types & target selection first in the 'Upload Data' section.")
        return

    # Check if we have processed data
    if 'processed_df' not in st.session_state:
        st.error("❌ No processed data found. Please go back to 'Upload Data' and confirm your type selections.")
        return

    df = st.session_state['processed_df']
    target_column = st.session_state.get('target_column')

    st.success(f"✅ Using processed data with {len(df)} rows, {len(df.columns)} columns")
    st.info(f"🎯 Target column: `{target_column}`")

    # TODO: Implement cleaning logic using the processed dataframe
    st.info("🚧 Data cleaning functionality to be implemented")


def export_section(df: pd.DataFrame):
    """Handle data export functionality."""
    st.subheader("📤 Export Cleaned Data")
    st.write("Export your cleaned data and schema to the processed directory.")

    if df is None:
        st.warning("⚠️ Please upload and process data first.")
        return

    # Get target column from session state
    target_column = st.session_state.get('target_column')

    # Export options
    col1, col2 = st.columns(2)
    with col1:
        export_filename = st.text_input(
            "Export Filename",
            value="cleaned_data",
            help="Base filename for the exported files (without extension)"
        )
    with col2:
        include_schema = st.checkbox(
            "Include Schema",
            value=True,
            help="Generate and export schema.json with the data"
        )

    if st.button("📤 Export Data", type="primary"):
        try:
            with st.spinner("Exporting data..."):
                # Create processed directory if it doesn't exist
                processed_dir = Path("data/processed")
                processed_dir.mkdir(parents=True, exist_ok=True)

                # Export the data
                csv_path, schema_path = cleaning.export_cleaned_data(
                    df=df,
                    output_dir=str(processed_dir),
                    filename=export_filename,
                    target_column=target_column,
                    include_schema=include_schema
                )

                # Display success message with file paths
                st.success("✅ Data exported successfully!")
                st.info(f"📄 CSV file: `{csv_path}`")
                if schema_path:
                    st.info(f"📄 Schema file: `{schema_path}`")

                # Add download buttons
                with open(csv_path, 'rb') as f:
                    st.download_button(
                        "⬇️ Download CSV",
                        f,
                        file_name=csv_path.name,
                        mime="text/csv"
                    )

                if schema_path:
                    with open(schema_path, 'rb') as f:
                        st.download_button(
                            "⬇️ Download Schema",
                            f,
                            file_name=schema_path.name,
                            mime="application/json"
                        )

        except Exception as e:
            st.error(f"❌ Error exporting data: {str(e)}")


def upload_data_debug_section():
    """Detailed debugging information for upload step."""
    st.header("🔍 Upload Data - Debug Information")

    st.subheader("📊 Session State")
    upload_state = {
        "uploaded_df": "uploaded_df" in st.session_state,
        "filename": st.session_state.get('filename', 'Not set'),
        "current_stage": st.session_state.get('current_stage', 'Not set')
    }
    st.json(upload_state)

    if 'uploaded_df' in st.session_state:
        df = st.session_state['uploaded_df']
        st.subheader("📋 DataFrame Information")
        st.write(f"Shape: {df.shape}")
        st.write(f"Memory usage: {df.memory_usage(deep=True).sum()} bytes")
        st.write(f"Data types: {df.dtypes.to_dict()}")

        st.subheader("🔢 Column Analysis")
        for col in df.columns:
            with st.expander(f"Column: {col}"):
                st.write(f"Type: {df[col].dtype}")
                st.write(f"Unique values: {df[col].nunique()}")
                st.write(f"Null count: {df[col].isnull().sum()}")
                if df[col].dtype in ['object']:
                    st.write(f"Sample values: {list(df[col].dropna().unique()[:5])}")


def type_override_debug_section():
    """Detailed debugging information for type override step."""
    st.header("🎯 Type Override - Debug Information")

    st.subheader("📊 Session State")
    override_state = {
        "types_confirmed": st.session_state.get('types_confirmed', False),
        "type_overrides": st.session_state.get('type_overrides', {}),
        "target_column": st.session_state.get('target_column', 'Not set'),
        "processed_df": "processed_df" in st.session_state
    }
    st.json(override_state)

    if 'type_overrides' in st.session_state and st.session_state['type_overrides']:
        st.subheader("🔄 Type Changes Made")
        if 'uploaded_df' in st.session_state:
            df = st.session_state['uploaded_df']
            changes = []
            for col in df.columns:
                original_type = str(df[col].dtype)
                new_type = st.session_state['type_overrides'].get(col, 'No change')
                changes.append({
                    'Column': col,
                    'Original Type': original_type,
                    'New Type': new_type,
                    'Changed': new_type != 'No change'
                })
            st.dataframe(pd.DataFrame(changes))

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        st.subheader("📋 Processed DataFrame")
        st.write(f"Shape: {df.shape}")
        st.write(f"Data types after conversion: {df.dtypes.to_dict()}")


def validation_debug_section():
    """Detailed debugging information for validation step."""
    st.header("✅ Data Validation - Debug Information")

    st.subheader("📊 Session State")
    validation_state = {
        "validation_results": "validation_results" in st.session_state,
        "validation_success": st.session_state.get('validation_success', 'Not set'),
        "validation_override": st.session_state.get('validation_override', False)
    }
    st.json(validation_state)

    if 'validation_results' in st.session_state:
        st.subheader("🔍 Validation Results Details")
        results = st.session_state['validation_results']
        st.json(results)

        if 'expectation_results' in results:
            st.subheader("📋 Individual Expectation Results")
            for i, result in enumerate(results['expectation_results']):
                success_icon = "✅" if result.get('success', False) else "❌"
                with st.expander(f"{success_icon} Expectation {i+1}: {result.get('expectation_type', 'Unknown')}"):
                    st.json(result)


def cleaning_debug_section():
    """Detailed debugging information for cleaning step."""
    st.header("🧹 Data Cleaning - Debug Information")

    st.subheader("📊 Session State")
    cleaning_state = {
        "cleaned_df": "cleaned_df" in st.session_state,
        "cleaning_report": "cleaning_report" in st.session_state,
        "types_confirmed": st.session_state.get('types_confirmed', False)
    }
    st.json(cleaning_state)

    if 'cleaning_report' in st.session_state:
        st.subheader("📋 Cleaning Report Details")
        report = st.session_state['cleaning_report']

        # Show original vs final state
        st.write("**Data Shape Changes:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Shape:", report['original_shape'])
        with col2:
            st.write("Final Shape:", report['final_shape'])

        # Show column changes
        st.write("**Column Changes:**")
        original_cols = set(report['original_columns'])
        final_cols = set(report['final_columns'])
        removed_cols = original_cols - final_cols
        added_cols = final_cols - original_cols

        if removed_cols:
            st.warning(f"Removed columns: {', '.join(removed_cols)}")
        if added_cols:
            st.info(f"Added columns: {', '.join(added_cols)}")

        # Show data type changes
        st.write("**Data Type Changes:**")
        dtype_changes = []
        for col in set(report['original_columns']) & set(report['final_columns']):
            orig_type = report['original_dtypes'].get(col)
            final_type = report['final_dtypes'].get(col)
            if orig_type != final_type:
                dtype_changes.append({
                    'Column': col,
                    'Original Type': orig_type,
                    'Final Type': final_type
                })

        if dtype_changes:
            st.dataframe(pd.DataFrame(dtype_changes))
        else:
            st.info("No data type changes")

        # Show missing value changes
        st.write("**Missing Value Changes:**")
        missing_changes = []
        for col in set(report['original_columns']) & set(report['final_columns']):
            orig_missing = report['missing_values_original'].get(col, 0)
            final_missing = report['missing_values_final'].get(col, 0)
            if orig_missing != final_missing:
                missing_changes.append({
                    'Column': col,
                    'Original Missing': orig_missing,
                    'Final Missing': final_missing,
                    'Change': final_missing - orig_missing
                })

        if missing_changes:
            st.dataframe(pd.DataFrame(missing_changes))
        else:
            st.info("No missing value changes")

        # Show steps performed
        st.write("**Cleaning Steps Performed:**")
        for step in report['steps_performed']:
            st.write(f"- {step.replace('_', ' ').title()}")

    if 'cleaned_df' in st.session_state:
        st.subheader("🔍 Cleaned DataFrame Info")
        df = st.session_state['cleaned_df']

        # Show basic info
        st.write(f"Shape: {df.shape}")
        st.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f}KB")

        # Show data types
        st.write("**Data Types:**")
        st.write(df.dtypes)

        # Show sample data
        st.write("**Sample Data (First 5 rows):**")
        st.dataframe(df.head())


def eda_debug_section():
    """Detailed debugging information for EDA step."""
    st.header("📊 EDA Profiling - Debug Information")

    st.subheader("📊 Session State")
    eda_state = {
        "cleaned_df": "cleaned_df" in st.session_state,
        "filename": st.session_state.get('filename', 'Not set')
    }
    st.json(eda_state)

    st.info("🚧 EDA profiling functionality not yet implemented.")
    st.write("This section will show detailed EDA reports and statistics.")


if __name__ == "__main__":
    main()
