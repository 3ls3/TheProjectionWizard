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

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from eda_validation import ydata_profile, cleaning, utils
from eda_validation.validation import setup_expectations, run_validation


def main():
    """Main Streamlit app for Team A's EDA and validation pipeline."""
    st.title("🔍 EDA & Validation Pipeline (Team A)")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Pipeline Steps")
    step = st.sidebar.selectbox(
        "Choose step:",
        ["Upload Data", "EDA Profiling", "Data Validation", "Data Cleaning", "Export Results"]
    )
    
    if step == "Upload Data":
        upload_data_section()
    elif step == "EDA Profiling":
        eda_profiling_section()
    elif step == "Data Validation":
        validation_section()
    elif step == "Data Cleaning":
        cleaning_section()
    elif step == "Export Results":
        export_section()


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
                    
                    # Add validation section after type confirmation
                    validation_section(updated_df)
                else:
                    st.error("❌ Error applying type conversions. Please check your type selections.")

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
    
    # Show existing files in raw data directory
    st.subheader("📂 Existing Files in Raw Data Directory")
    if raw_data_dir.exists():
        csv_files = list(raw_data_dir.glob("*.csv"))
        if csv_files:
            st.write("Available CSV files:")
            for file_path in sorted(csv_files):
                file_size = file_path.stat().st_size / 1024  # KB
                modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    if st.button(f"📄 {file_path.name}", key=f"load_{file_path.name}"):
                        try:
                            df = pd.read_csv(file_path)
                            st.session_state['current_dataset'] = str(file_path)
                            st.session_state['current_df'] = df
                            st.success(f"✅ Loaded {file_path.name}")
                        except Exception as e:
                            st.error(f"❌ Error loading {file_path.name}: {str(e)}")
                
                with col2:
                    st.text(f"{file_size:.1f}KB")
                with col3:
                    st.text(modified_time.strftime("%Y-%m-%d %H:%M"))
        else:
            st.info("No CSV files found in data/raw/ directory.")
    else:
        st.info("Raw data directory doesn't exist yet.")


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


def export_section():
    """Handle exporting cleaned data for Team B."""
    st.header("💾 Export Results")
    st.write("Export cleaned and validated data for modeling pipeline.")
    
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
    
    # TODO: Implement export logic - export cleaned_data.csv and schema.json
    st.info("🚧 Export functionality to be implemented")


if __name__ == "__main__":
    main() 