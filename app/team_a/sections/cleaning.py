"""
Data cleaning section for the main pipeline flow.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from eda_validation import cleaning

def cleaning_pipeline_section(is_current: bool = True):
    """Data cleaning section for the main pipeline flow."""
    st.header("🧹 Data Cleaning")

    # If not current stage, show summary
    if not is_current and st.session_state.get('cleaning_done', False):
        cleaning_report = st.session_state.get('cleaning_report', {})
        if cleaning_report:
            final_rows = cleaning_report.get('final_shape', [0, 0])[0]
            final_cols = cleaning_report.get('final_shape', [0, 0])[1]
            rows_removed = cleaning_report.get('rows_removed', 0)
            cols_removed = cleaning_report.get('columns_removed', 0)

            with st.expander(f"✅ **Data Cleaned** | {final_rows} rows, {final_cols} columns", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Rows", final_rows, delta=-rows_removed if rows_removed > 0 else None)
                with col2:
                    st.metric("Final Columns", final_cols, delta=-cols_removed if cols_removed > 0 else None)

                steps = cleaning_report.get('steps_performed', [])
                if steps:
                    st.write("**Steps performed:**", ", ".join([step.replace('_', ' ').title() for step in steps]))
        else:
            st.success("✅ **Data cleaning completed**")
        return

    # If current stage or cleaning not done, show full UI
    if 'processed_df' not in st.session_state:
        st.error("❌ No processed data found. Please go back to validation.")
        if is_current and st.button("🔙 Back to Validation"):
            st.session_state.stage = "validation"
            st.rerun()
        return

    df = st.session_state['processed_df']

    # 4️⃣ Guard: if cleaning already done, skip to final
    if st.session_state.get("cleaning_done", False):
        st.session_state["stage"] = "final"
        st.rerun()
        return

    # Show cleaning section
    show_cleaning_section(df, is_current)

def show_cleaning_section(df, is_current: bool = True):
    """Show cleaning options with auto/manual toggle."""
    # 4️⃣ Guard: if already done, return early to avoid duplicates
    if st.session_state.get("cleaning_done", False):
        return

    st.markdown("---")
    st.subheader("🧹 Data Cleaning")

    # Auto vs Manual toggle
    use_auto_cleaning = st.toggle("🤖 Use Automatic Cleaning", value=True,
                                  help="Enable to use default cleaning parameters, disable to customize")

    if use_auto_cleaning:
        st.info("✨ **Automatic cleaning** will use default parameters: drop missing values (50% threshold), standardize column names, remove duplicates, auto-convert data types.")

        if is_current and st.button("🧹 **Run Automatic Cleaning**", type="primary", key="run_auto_cleaning"):
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
                # 4️⃣ Transition to final stage
                st.session_state["stage"] = "final"

            st.rerun()

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

        if is_current and st.button("🧹 **Run Custom Cleaning**", type="primary", key="run_custom_cleaning"):
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
                # 4️⃣ Transition to final stage
                st.session_state["stage"] = "final"

            st.rerun()

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
