"""
Final results section for the main pipeline flow.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from io import BytesIO

def show_final_results_section(is_current: bool = True):
    """Show final results section with download options."""
    st.header("🎉 Final Results")

    if 'cleaned_df' not in st.session_state:
        st.error("❌ No cleaned data found. Please go back to cleaning.")
        if is_current and st.button("🔙 Back to Cleaning"):
            st.session_state.stage = "cleaning"
            st.rerun()
        return

    df = st.session_state['cleaned_df']

    # Show final dataset info
    st.subheader("📊 Final Dataset")

    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f}KB")

    # Show data preview
    st.write("**Data Preview:**")
    st.dataframe(df.head())

    # Show data types
    st.write("**Data Types:**")
    st.write(df.dtypes)

    # Show missing values
    st.write("**Missing Values:**")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning(f"Found {missing.sum()} missing values")
        st.write(missing[missing > 0])
    else:
        st.success("No missing values found")

    # Show download options
    st.subheader("💾 Download Options")

    # Create download buttons
    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            "cleaned_data.csv",
            "text/csv",
            key='download-csv'
        )

    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        excel_data = output.getvalue()
        st.download_button(
            "📥 Download Excel",
            excel_data,
            "cleaned_data.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key='download-excel'
        )

    # Show restart option
    st.markdown("---")
    if st.button("🔄 Start New Pipeline"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
