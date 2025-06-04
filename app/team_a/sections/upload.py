"""
Upload section for the main pipeline flow.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

def upload_pipeline_section(is_current: bool = True):
    """Upload section for the main pipeline flow."""
    st.header("📁 Upload File")

    # If not current stage, show summary
    if not is_current and 'uploaded_df' in st.session_state:
        df = st.session_state['uploaded_df']
        filename = st.session_state.get('filename', 'Unknown file')

        with st.expander(f"✅ **File Uploaded:** {filename}", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                st.metric("Missing Values", f"{missing_pct:.1f}%")
        return

    # If current stage or no data, show full UI
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

                # Show success message and store data
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

                # 1️⃣ Transition to type_override stage
                st.session_state["stage"] = "type_override"
                st.rerun()

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
