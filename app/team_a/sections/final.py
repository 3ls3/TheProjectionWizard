"""
Final results section for the main pipeline flow.
"""

import streamlit as st
import pandas as pd
import json
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
        # Create JSON report
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "pipeline_version": "Team A - Part 1",
                "total_rows": int(df.shape[0]),
                "total_columns": int(df.shape[1])
            },
            "columns": {},
            "data_quality": {
                "missing_values": missing.to_dict(),
                "total_missing": int(missing.sum()),
                "data_types": df.dtypes.astype(str).to_dict()
            }
        }
        
        # Add column statistics
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum())
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "std": float(df[col].std()) if not pd.isna(df[col].std()) else None
                })
            
            report["columns"][col] = col_info
        
        json_str = json.dumps(report, indent=2)
        st.download_button(
            "📥 Download JSON Report",
            json_str,
            "data_report.json",
            "application/json",
            key='download-json'
        )

    # Show restart option
    st.markdown("---")
    if st.button("🔄 Start New Pipeline"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
