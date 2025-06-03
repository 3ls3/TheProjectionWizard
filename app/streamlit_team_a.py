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
                        
                        # Check if file already exists
                        if save_path.exists():
                            if st.checkbox("⚠️ File exists. Overwrite?"):
                                df.to_csv(save_path, index=False)
                                st.success(f"✅ File saved successfully to: {save_path}")
                                
                                # Store filename in session state for other sections
                                st.session_state['current_dataset'] = str(save_path)
                                st.session_state['current_df'] = df
                        else:
                            df.to_csv(save_path, index=False)
                            st.success(f"✅ File saved successfully to: {save_path}")
                            
                            # Store filename in session state for other sections
                            st.session_state['current_dataset'] = str(save_path)
                            st.session_state['current_df'] = df
                            
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
    
    # TODO: Implement EDA profiling
    st.info("🚧 EDA profiling functionality to be implemented")


def validation_section():
    """Handle data validation with Great Expectations."""
    st.header("✅ Data Validation")
    st.write("Validate data quality using Great Expectations.")
    
    # TODO: Implement validation logic
    st.info("🚧 Data validation functionality to be implemented")


def cleaning_section():
    """Handle data cleaning operations."""
    st.header("🧹 Data Cleaning")
    st.write("Clean and preprocess the data.")
    
    # TODO: Implement cleaning logic
    st.info("🚧 Data cleaning functionality to be implemented")


def export_section():
    """Handle exporting cleaned data for Team B."""
    st.header("💾 Export Results")
    st.write("Export cleaned and validated data for modeling pipeline.")
    
    # TODO: Implement export logic
    st.info("🚧 Export functionality to be implemented")


if __name__ == "__main__":
    main() 