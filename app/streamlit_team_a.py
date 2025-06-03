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
from eda_validation.ydata_profile import LARGE_DATASET_THRESHOLD, DEFAULT_SAMPLE_SIZE, REPORTS_DIR


def main():
    """Main Streamlit app for Team A's EDA and validation pipeline."""
    st.title("🔍 EDA & Validation Pipeline (Team A)")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Pipeline Steps")
    step = st.sidebar.selectbox(
        "Choose step:",
        ["Upload & Analyze Data", "View EDA Reports", "Data Validation", "Data Cleaning", "Export Results"]
    )
    
    if step == "Upload & Analyze Data":
        upload_and_analyze_section()
    elif step == "View EDA Reports":
        view_eda_reports_section()
    elif step == "Data Validation":
        validation_section()
    elif step == "Data Cleaning":
        cleaning_section()
    elif step == "Export Results":
        export_section()


def upload_and_analyze_section():
    """Handle data upload and automatic EDA analysis."""
    st.header("📁 Data Upload & Analysis")
    st.write("Upload your CSV file and automatically generate EDA reports.")
    
    # Create data/raw directory if it doesn't exist
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # EDA Configuration (show before upload)
    with st.expander("⚙️ EDA Configuration (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Large Dataset Sampling")
            enable_sampling = st.checkbox(
                "Enable sampling for large datasets",
                value=True,
                help=f"Automatically sample datasets with more than {LARGE_DATASET_THRESHOLD:,} rows"
            )
            
            sample_size = st.number_input(
                "Sample size:",
                min_value=1000,
                max_value=100000,
                value=DEFAULT_SAMPLE_SIZE,
                step=1000,
                help="Number of rows to sample from large datasets"
            )
        
        with col2:
            st.subheader("🔧 Advanced Options")
            disable_correlations = st.checkbox(
                "Disable advanced correlations",
                value=False,
                help="Disable Spearman, Kendall, and other advanced correlations for faster processing"
            )
            
            disable_interactions = st.checkbox(
                "Disable interactions analysis",
                value=True,
                help="Disable variable interactions analysis for faster processing"
            )
            
            output_format = st.selectbox(
                "Report format:",
                ["both", "html", "json"],
                index=0,
                help="Choose output format(s) for the report"
            )
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file (max 200MB). EDA report will be generated automatically."
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
                
                # Save file section
                st.subheader("💾 Save & Analyze")
                
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
                
                if st.button("💾 Save File & Generate EDA Report", type="primary"):
                    try:
                        # Ensure filename ends with .csv
                        if not filename.endswith('.csv'):
                            filename += '.csv'
                        
                        save_path = raw_data_dir / filename
                        
                        # Check if file already exists
                        if save_path.exists():
                            if st.checkbox("⚠️ File exists. Overwrite?"):
                                # Save file
                                df.to_csv(save_path, index=False)
                                st.success(f"✅ File saved successfully to: {save_path}")
                                
                                # Store in session state
                                st.session_state['current_dataset'] = str(save_path)
                                st.session_state['current_df'] = df
                                
                                # Automatically generate EDA report
                                generate_automatic_eda_report(df, save_path, enable_sampling, sample_size, 
                                                            disable_correlations, disable_interactions, output_format)
                        else:
                            # Save file
                            df.to_csv(save_path, index=False)
                            st.success(f"✅ File saved successfully to: {save_path}")
                            
                            # Store in session state
                            st.session_state['current_dataset'] = str(save_path)
                            st.session_state['current_df'] = df
                            
                            # Automatically generate EDA report
                            generate_automatic_eda_report(df, save_path, enable_sampling, sample_size, 
                                                        disable_correlations, disable_interactions, output_format)
                            
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


def generate_automatic_eda_report(df, dataset_path, enable_sampling, sample_size, 
                                disable_correlations, disable_interactions, output_format):
    """Generate EDA report automatically after file upload."""
    
    # Validate ydata-profiling availability
    if not ydata_profile.YDATA_AVAILABLE:
        st.error("❌ ydata-profiling not available. Please install it:")
        st.code("pip install ydata-profiling>=4.8.3")
        return
    
    st.subheader("🚀 Generating EDA Report")
    st.info("📊 Automatically generating comprehensive EDA report...")
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Prepare configuration
        status_text.text("⚙️ Preparing configuration...")
        progress_bar.progress(10)
        
        report_title = f"EDA Report - {Path(dataset_path).stem}"
        
        config = {
            "title": report_title,
            "correlations": {
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": not disable_correlations},
                "kendall": {"calculate": not disable_correlations},
                "phi_k": {"calculate": not disable_correlations},
                "cramers": {"calculate": not disable_correlations},
            },
            "interactions": {
                "continuous": not disable_interactions,
                "targets": []
            }
        }
        
        # Step 2: Generate profile
        status_text.text("📊 Generating profile report...")
        progress_bar.progress(30)
        
        # Adjust sample size if needed
        if enable_sampling and df.shape[0] > LARGE_DATASET_THRESHOLD:
            st.warning(f"⚠️ Large dataset detected ({df.shape[0]:,} rows). Using sampling with {sample_size:,} rows.")
        
        result = ydata_profile.generate_profile(
            df=df,
            title=report_title,
            config=config,
            enable_sampling=enable_sampling,
            sample_size=sample_size
        )
        
        if result is None:
            st.error("❌ Failed to generate profile report.")
            return
        
        profile, profile_info = result
        
        # Step 3: Save report
        status_text.text("💾 Saving report files...")
        progress_bar.progress(70)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = Path(dataset_path).stem
        output_filename = f"{dataset_name}_eda_{timestamp}"
        
        success, output_files = ydata_profile.save_profile_report(
            profile=profile,
            output_path=output_filename,
            format=output_format,
            profile_info=profile_info
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Report generation completed!")
        
        if success:
            st.success("🎉 EDA report generated successfully!")
            
            # Display report information
            st.subheader("📋 Generated Reports")
            
            for file_type, file_path in output_files.items():
                file_path_obj = Path(file_path)
                file_size_kb = file_path_obj.stat().st_size / 1024
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.text(f"📄 {file_type.upper()}: {file_path_obj.name}")
                with col2:
                    st.text(f"{file_size_kb:.1f} KB")
                with col3:
                    if file_type == 'html':
                        # Create download button for HTML report
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download",
                                data=f.read(),
                                file_name=file_path_obj.name,
                                mime="text/html",
                                key=f"auto_{file_type}"
                            )
                    elif file_type == 'json':
                        # Create download button for JSON report
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download",
                                data=f.read(),
                                file_name=file_path_obj.name,
                                mime="application/json",
                                key=f"auto_{file_type}"
                            )
            
            # Display sampling information if applicable
            if profile_info and profile_info.get('sampling', {}).get('sampled', False):
                sampling_info = profile_info['sampling']
                st.info(f"ℹ️ Report generated from a sample of {sampling_info['sample_rows']:,} rows "
                       f"({sampling_info['sampling_ratio']:.1%} of original dataset)")
            
            st.success("🔗 You can view all reports in the 'View EDA Reports' section.")
            
        else:
            st.error("❌ Failed to save report files.")
            
    except Exception as e:
        st.error(f"❌ Error generating EDA report: {str(e)}")
    
    finally:
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()


def view_eda_reports_section():
    """Handle viewing and managing existing EDA reports."""
    st.header("📊 View EDA Reports")
    st.write("Browse and download previously generated EDA reports.")
    
    try:
        available_reports = ydata_profile.get_available_reports()
        
        if available_reports:
            st.write(f"Found {len(available_reports)} report(s):")
            
            for report in available_reports:
                with st.expander(f"📊 {report['filename']}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.text(f"📁 Path: {report['path']}")
                        st.text(f"📊 Size: {report['size_kb']:.1f} KB")
                        st.text(f"📅 Modified: {report['modified'][:19]}")
                        
                        # Show available formats
                        formats = ["HTML"]
                        if report['has_json']:
                            formats.append("JSON")
                        if report['has_metadata']:
                            formats.append("Metadata")
                        st.text(f"📄 Formats: {', '.join(formats)}")
                    
                    with col2:
                        # Download buttons
                        report_path = Path(report['path'])
                        
                        # HTML download
                        if report_path.exists():
                            with open(report_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ HTML",
                                    data=f.read(),
                                    file_name=report_path.name,
                                    mime="text/html",
                                    key=f"view_html_{report['filename']}"
                                )
                        
                        # JSON download if available
                        json_path = report_path.with_suffix('.json')
                        if json_path.exists():
                            with open(json_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ JSON",
                                    data=f.read(),
                                    file_name=json_path.name,
                                    mime="application/json",
                                    key=f"view_json_{report['filename']}"
                                )
                    
                    # Preview option for smaller HTML reports
                    if report['size_kb'] < 10240:  # Less than 10MB
                        if st.button(f"👀 Preview Report", key=f"preview_{report['filename']}"):
                            try:
                                with open(report['path'], 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                
                                st.subheader("📖 Report Preview")
                                st.info("🔗 For best viewing experience, download the HTML report.")
                                st.components.v1.html(html_content, height=600, scrolling=True)
                            except Exception as e:
                                st.error(f"Could not preview report: {e}")
                    else:
                        st.info("📄 Report too large for preview. Please download to view.")
        else:
            st.info("No reports found. Upload data in 'Upload & Analyze Data' to generate reports!")
            
    except Exception as e:
        st.warning(f"⚠️ Could not load existing reports: {e}")


def eda_profiling_section():
    """Redirect to view reports section."""
    st.info("🔄 This section has been moved. Please use:")
    st.markdown("- **Upload & Analyze Data**: To upload CSV and automatically generate EDA reports")
    st.markdown("- **View EDA Reports**: To browse and download existing reports")


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