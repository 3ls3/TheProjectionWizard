"""
Results & Downloads page for The Projection Wizard.
Provides comprehensive summary of pipeline run and download access to all artifacts.
"""

import streamlit as st
import pandas as pd
import json
import zipfile
import os
import io
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from common import constants, storage, schemas


def create_download_button(label: str, data_bytes: bytes, file_name: str, mime_type: str, key_suffix: str, run_id: str):
    """Helper function to create consistent download buttons."""
    return st.download_button(
        label=label,
        data=data_bytes,
        file_name=file_name,
        mime=mime_type,
        key=f"download_{key_suffix}_{run_id}",
        use_container_width=True
    )


def show_results_page():
    """Display the Results & Downloads page."""
    
    # Page Title
    st.title("🏁 Step 8: Pipeline Results & Downloads")
    
    # Check if run_id exists in session state
    if 'run_id' not in st.session_state:
        st.error("❌ No active run found. Please upload a file first.")
        if st.button("🏠 Go to Upload Page", type="primary"):
            st.session_state['current_page'] = 'upload'
            st.rerun()
        return
    
    run_id = st.session_state['run_id']
    
    # Load Run Information and Artifacts
    try:
        run_dir_path = storage.get_run_dir(run_id)
        
        # Load key files with graceful error handling
        metadata_dict = None
        status_dict = None
        validation_summary_dict = None
        
        try:
            metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
        except Exception as e:
            st.warning(f"Could not load metadata: {e}")
        
        try:
            status_dict = storage.read_json(run_id, constants.STATUS_FILENAME)
        except Exception as e:
            st.warning(f"Could not load status: {e}")
        
        try:
            validation_summary_dict = storage.read_json(run_id, constants.VALIDATION_FILENAME)
        except Exception as e:
            st.warning(f"Could not load validation report: {e}")
        
        # Convert to Pydantic Models for type safety and easy access
        metadata = None
        status = None
        validation_summary = None
        
        if metadata_dict:
            try:
                # Use the most comprehensive metadata model available
                metadata = schemas.MetadataWithExplain(**metadata_dict)
            except Exception as e:
                st.warning(f"Could not parse metadata structure: {e}")
                # Try to access as plain dict if Pydantic fails
                metadata = type('obj', (object,), metadata_dict)
        
        if status_dict:
            try:
                status = schemas.StageStatus(**status_dict)
            except Exception as e:
                st.warning(f"Could not parse status structure: {e}")
                # Use as plain dict
                status = type('obj', (object,), status_dict)
        
        if validation_summary_dict:
            try:
                validation_summary = schemas.ValidationReportSummary(**validation_summary_dict)
            except Exception as e:
                st.warning(f"Could not parse validation summary: {e}")
                # Use as plain dict
                validation_summary = type('obj', (object,), validation_summary_dict)

        # Display Run Summary Section
        st.header("📊 Run Summary")
        
        if metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Run ID:** `{metadata.run_id if hasattr(metadata, 'run_id') else run_id}`")
                
                # Handle timestamp display
                timestamp_str = "N/A"
                if hasattr(metadata, 'timestamp') and metadata.timestamp:
                    try:
                        if isinstance(metadata.timestamp, str):
                            timestamp = datetime.fromisoformat(metadata.timestamp.replace('Z', '+00:00'))
                        else:
                            timestamp = metadata.timestamp
                        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
                    except:
                        timestamp_str = str(metadata.timestamp)
                st.write(f"**Timestamp:** {timestamp_str}")
                
                st.write(f"**Original File:** {getattr(metadata, 'original_filename', 'N/A')}")
                
            with col2:
                initial_rows = getattr(metadata, 'initial_rows', None)
                initial_cols = getattr(metadata, 'initial_cols', None)
                if initial_rows and initial_cols:
                    st.write(f"**Initial Shape:** {initial_rows:,} rows, {initial_cols:,} columns")
                
                # Target information
                if hasattr(metadata, 'target_info') and metadata.target_info:
                    target_info = metadata.target_info
                    st.write(f"**Target Column:** {getattr(target_info, 'name', 'N/A')}")
                    st.write(f"**Task Type:** {getattr(target_info, 'task_type', 'N/A').title()}")
        
        else:
            st.warning("Run metadata not available.")
        
        # Pipeline status
        if status:
            status_color = "🟢" if getattr(status, 'status', None) == 'completed' else "🟡"
            stage_name = constants.STAGE_DISPLAY_NAMES.get(getattr(status, 'stage', ''), getattr(status, 'stage', 'Unknown'))
            st.write(f"**Final Pipeline Status:** {status_color} {getattr(status, 'status', 'Unknown').title()} (at stage: {stage_name})")
            
            if hasattr(status, 'message') and status.message:
                st.info(f"💬 {status.message}")
            
            if hasattr(status, 'errors') and status.errors:
                with st.expander("⚠️ View Errors"):
                    for error in status.errors:
                        st.error(error)

        # Display Validation Summary Section
        st.header("✅ Validation Summary")
        
        if validation_summary:
            overall_success = getattr(validation_summary, 'overall_success', False)
            success_icon = "✅" if overall_success else "❌"
            st.metric("Overall Validation", f"{success_icon} {'Passed' if overall_success else 'Failed'}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Expectations", getattr(validation_summary, 'total_expectations', 0))
            with col2:
                st.metric("Successful", getattr(validation_summary, 'successful_expectations', 0))
            with col3:
                st.metric("Failed", getattr(validation_summary, 'failed_expectations', 0))
        else:
            st.write("Validation report not available or stage not run.")

        # Display Data Preparation Summary Section
        st.header("🧹 Data Preparation Summary")
        
        if metadata and hasattr(metadata, '__dict__') and 'prep_info' in metadata.__dict__:
            prep_info = metadata.__dict__['prep_info']
        elif metadata_dict and 'prep_info' in metadata_dict:
            prep_info = metadata_dict['prep_info']
        else:
            prep_info = None
        
        if prep_info:
            final_shape = prep_info.get('final_shape_after_prep', [0, 0])
            if len(final_shape) == 2:
                st.write(f"**Final Shape After Prep:** {final_shape[0]:,} rows, {final_shape[1]:,} columns")
            
            cleaning_steps = prep_info.get('cleaning_steps_performed', [])
            if cleaning_steps:
                with st.expander("🔍 Cleaning Steps Performed"):
                    for step in cleaning_steps:
                        st.markdown(f"- {step}")
            else:
                st.write("No specific cleaning steps recorded.")
        else:
            st.write("Preparation info not available or stage not run.")

        # Display AutoML Model Summary Section
        st.header("🤖 AutoML Model Summary")
        
        if metadata and hasattr(metadata, 'automl_info') and metadata.automl_info:
            automl_info = metadata.automl_info
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Tool Used:** {getattr(automl_info, 'tool_used', 'N/A')}")
                st.write(f"**Best Model Algorithm:** {getattr(automl_info, 'best_model_name', 'N/A')}")
            
            with col2:
                target_column = getattr(automl_info, 'target_column', 'N/A')
                task_type = getattr(automl_info, 'task_type', 'N/A')
                st.write(f"**Target Column:** {target_column}")
                st.write(f"**Task Type:** {task_type.title() if task_type != 'N/A' else 'N/A'}")
            
            # Performance Metrics
            if hasattr(automl_info, 'performance_metrics') and automl_info.performance_metrics:
                st.subheader("📈 Performance Metrics")
                
                metrics_dict = automl_info.performance_metrics
                if isinstance(metrics_dict, dict):
                    # Create a nice table display
                    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['Value'])
                    metrics_df.index.name = 'Metric'
                    
                    # Format the values nicely
                    formatted_metrics = {}
                    for metric_name, value in metrics_dict.items():
                        if isinstance(value, float):
                            if metric_name.lower() in ['auc', 'accuracy', 'f1', 'recall', 'precision']:
                                formatted_metrics[metric_name] = f"{value:.1%}"
                            else:
                                formatted_metrics[metric_name] = f"{value:.4f}"
                        else:
                            formatted_metrics[metric_name] = str(value)
                    
                    # Display in columns for better layout
                    metric_cols = st.columns(min(len(formatted_metrics), 4))
                    for i, (metric_name, metric_value) in enumerate(formatted_metrics.items()):
                        with metric_cols[i % len(metric_cols)]:
                            st.metric(metric_name, metric_value)
                else:
                    st.write("Performance metrics available but format not recognized.")
            else:
                st.write("No performance metrics available.")
        else:
            st.write("AutoML info not available or stage not run.")

        # Display Explainability Section (SHAP Plot)
        st.header("🧠 Model Explainability")
        
        if metadata and hasattr(metadata, 'explain_info') and metadata.explain_info:
            explain_info = metadata.explain_info
            
            # Show explainability metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tool Used", getattr(explain_info, 'tool_used', 'N/A'))
            with col2:
                st.metric("Features Explained", f"{getattr(explain_info, 'features_explained', 0):,}")
            with col3:
                st.metric("Samples Used", f"{getattr(explain_info, 'samples_used_for_explanation', 0):,}")
            
            # Display SHAP plot if available
            shap_plot_path = getattr(explain_info, 'shap_summary_plot_path', None)
            if shap_plot_path:
                shap_plot_full_path = run_dir_path / shap_plot_path
                if shap_plot_full_path.exists():
                    st.subheader("📊 SHAP Feature Importance Plot")
                    st.image(
                        str(shap_plot_full_path), 
                        caption="SHAP Global Feature Importance Summary",
                        use_column_width=True
                    )
                    
                    # Show plot file info
                    file_size_kb = shap_plot_full_path.stat().st_size / 1024
                    st.caption(f"📈 Plot file size: {file_size_kb:.1f} KB")
                else:
                    st.warning("SHAP summary plot file not found.")
            else:
                st.write("No SHAP plot path available.")
        else:
            st.write("Explainability results not available or stage not run.")

        # Implement Artifact Downloads Section
        st.header("💾 Download Artifacts")
        st.write("Download all the files and results generated during your pipeline run:")

        # Create download sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Data Files")
            
            # Original Data CSV
            original_data_path = run_dir_path / constants.ORIGINAL_DATA_FILENAME
            if original_data_path.exists():
                try:
                    with open(original_data_path, "rb") as fp:
                        create_download_button(
                            "📥 Original Data (CSV)",
                            fp.read(),
                            constants.ORIGINAL_DATA_FILENAME,
                            "text/csv",
                            "orig_csv",
                            run_id
                        )
                    # Show file info
                    file_size_mb = original_data_path.stat().st_size / (1024 * 1024)
                    st.caption(f"Size: {file_size_mb:.2f} MB")
                except Exception as e:
                    st.error(f"Error reading original data: {e}")
            else:
                st.caption("❌ Original data file not found")
            
            # Cleaned Data CSV
            cleaned_data_path = run_dir_path / constants.CLEANED_DATA_FILE
            if cleaned_data_path.exists():
                try:
                    with open(cleaned_data_path, "rb") as fp:
                        create_download_button(
                            "📥 Cleaned & Encoded Data (CSV)",
                            fp.read(),
                            constants.CLEANED_DATA_FILE,
                            "text/csv",
                            "clean_csv",
                            run_id
                        )
                    # Show file info  
                    file_size_mb = cleaned_data_path.stat().st_size / (1024 * 1024)
                    st.caption(f"Size: {file_size_mb:.2f} MB")
                except Exception as e:
                    st.error(f"Error reading cleaned data: {e}")
            else:
                st.caption("❌ Cleaned data file not found")

        with col2:
            st.subheader("📊 Report Files")
            
            # Metadata JSON
            metadata_path = run_dir_path / constants.METADATA_FILENAME
            if metadata_path.exists():
                try:
                    with open(metadata_path, "rb") as fp:
                        create_download_button(
                            "📥 Metadata (JSON)",
                            fp.read(),
                            constants.METADATA_FILENAME,
                            "application/json",
                            "meta_json",
                            run_id
                        )
                    # Show file info
                    file_size_kb = metadata_path.stat().st_size / 1024
                    st.caption(f"Size: {file_size_kb:.1f} KB")
                except Exception as e:
                    st.error(f"Error reading metadata: {e}")
            else:
                st.caption("❌ Metadata file not found")
            
            # Validation Report JSON
            validation_path = run_dir_path / constants.VALIDATION_FILENAME
            if validation_path.exists():
                try:
                    with open(validation_path, "rb") as fp:
                        create_download_button(
                            "📥 Validation Report (JSON)",
                            fp.read(),
                            constants.VALIDATION_FILENAME,
                            "application/json",
                            "val_json",
                            run_id
                        )
                    # Show file info
                    file_size_kb = validation_path.stat().st_size / 1024
                    st.caption(f"Size: {file_size_kb:.1f} KB")
                except Exception as e:
                    st.error(f"Error reading validation report: {e}")
            else:
                st.caption("❌ Validation report not found")

        # Data Profile Report (HTML)
        st.subheader("📈 Additional Reports")
        
        # Check for profiling report in prep_info
        profile_report_path = None
        if prep_info and 'profiling_report_path' in prep_info:
            profile_report_path = run_dir_path / prep_info['profiling_report_path']
        
        if profile_report_path and profile_report_path.exists():
            try:
                with open(profile_report_path, "rb") as fp:
                    create_download_button(
                        "📥 Data Profile Report (HTML)",
                        fp.read(),
                        profile_report_path.name,
                        "text/html",
                        "profile_html",
                        run_id
                    )
                # Show file info
                file_size_kb = profile_report_path.stat().st_size / 1024
                st.caption(f"Size: {file_size_kb:.1f} KB")
            except Exception as e:
                st.error(f"Error reading profile report: {e}")
        else:
            st.caption("❌ Data profile report not found")

        # Model Artifacts (Zipped)
        st.subheader("🤖 Model Artifacts")
        
        model_dir = run_dir_path / constants.MODEL_DIR
        if model_dir.exists() and any(model_dir.iterdir()):
            try:
                # Create a zip file in memory containing all files from model_dir
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for root, dirs, files in os.walk(model_dir):
                        for file in files:
                            file_path = Path(root) / file
                            # Arcname should be relative to model_dir
                            arcname = file_path.relative_to(model_dir)
                            zip_file.write(file_path, arcname=arcname)
                
                zip_buffer.seek(0)
                
                create_download_button(
                    "📥 Model Artifacts (ZIP)",
                    zip_buffer.getvalue(),
                    f"{run_id}_model_artifacts.zip",
                    "application/zip",
                    "model_zip",
                    run_id
                )
                
                # Show info about what's in the zip
                model_files = list(model_dir.rglob("*"))
                model_files = [f for f in model_files if f.is_file()]
                st.caption(f"Contains {len(model_files)} files")
                
                with st.expander("📁 View model files"):
                    for file_path in sorted(model_files):
                        rel_path = file_path.relative_to(model_dir)
                        file_size_kb = file_path.stat().st_size / 1024
                        st.write(f"- {rel_path} ({file_size_kb:.1f} KB)")
                        
            except Exception as e:
                st.error(f"Error creating model artifacts zip: {e}")
        else:
            st.caption("❌ No model artifacts found")

        # SHAP Plot Download
        if metadata and hasattr(metadata, 'explain_info') and metadata.explain_info:
            explain_info = metadata.explain_info
            shap_plot_path = getattr(explain_info, 'shap_summary_plot_path', None)
            
            if shap_plot_path:
                shap_plot_full_path = run_dir_path / shap_plot_path
                if shap_plot_full_path.exists():
                    st.subheader("🧠 Explainability Artifacts")
                    try:
                        with open(shap_plot_full_path, "rb") as fp:
                            create_download_button(
                                "📥 SHAP Feature Importance Plot (PNG)",
                                fp.read(),
                                f"{run_id}_shap_summary.png",
                                "image/png",
                                "shap_png",
                                run_id
                            )
                        # Show file info
                        file_size_kb = shap_plot_full_path.stat().st_size / 1024
                        st.caption(f"Size: {file_size_kb:.1f} KB")
                    except Exception as e:
                        st.error(f"Error reading SHAP plot: {e}")

        # Log File
        st.subheader("📄 Log Files")
        
        log_path = run_dir_path / constants.PIPELINE_LOG_FILENAME
        if log_path.exists():
            try:
                with open(log_path, "rb") as fp:
                    create_download_button(
                        "📥 Pipeline Log File",
                        fp.read(),
                        constants.PIPELINE_LOG_FILENAME,
                        "text/plain",
                        "log_txt",
                        run_id
                    )
                # Show file info
                file_size_kb = log_path.stat().st_size / 1024
                st.caption(f"Size: {file_size_kb:.1f} KB")
            except Exception as e:
                st.error(f"Error reading log file: {e}")
        else:
            st.caption("❌ Pipeline log file not found")

        # Start New Run Section
        st.divider()
        st.header("🔄 Start New Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Ready to analyze a new dataset? Start a fresh pipeline run:")
            
        with col2:
            if st.button("🏠 Start New Run / Upload New Data", type="primary", use_container_width=True):
                # Clear relevant session state
                keys_to_clear = ['run_id', 'current_page']
                for key in keys_to_clear:
                    st.session_state.pop(key, None)
                
                # Navigate to upload page
                st.session_state['current_page'] = 'upload'
                st.rerun()

        # Display Current Run ID (sidebar or footer)
        st.divider()
        st.info(f"📋 **Current Run ID:** `{run_id}` - Keep this ID to reference this analysis later.")

    except Exception as e:
        st.error(f"❌ Error loading run information: {str(e)}")
        st.exception(e)
        
        # Still provide option to start new run
        st.divider()
        if st.button("🏠 Start New Run", type="primary"):
            keys_to_clear = ['run_id', 'current_page']
            for key in keys_to_clear:
                st.session_state.pop(key, None)
            st.session_state['current_page'] = 'upload'
            st.rerun()


if __name__ == "__main__":
    show_results_page() 