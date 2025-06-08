"""
Data preparation page for The Projection Wizard.
Provides UI for running data preparation (cleaning, encoding, profiling) and viewing results.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.step_4_prep import prep_runner
from common import constants, storage, schemas, utils


def show_prep_page():
    """Display the data preparation page."""
    
    # Page Title
    st.title("Step 5: Data Preparation")
    
    # Check if run_id exists in session state
    if 'run_id' not in st.session_state:
        st.error("No active run found. Please upload a file first.")
        if st.button("Go to Upload Page"):
            st.session_state['current_page'] = 'upload'
            st.rerun()
        return
    
    run_id = st.session_state['run_id']
    
    # Display Current Run ID
    st.info(f"**Current Run ID:** {run_id}")
    
    # Introductory text
    st.write("This step cleans your data, encodes features for machine learning, and can generate a detailed data profile report. Review the outcomes before model training.")
    

    
    # Display Existing Results (if page is revisited)
    st.subheader("Data Preparation Status")
    
    try:
        # Check status.json to see if prep stage completed
        status_data = storage.read_json(run_id, constants.STATUS_FILENAME)
        prep_completed = False
        
        if status_data:
            prep_completed = (status_data.get('stage') == constants.PREP_STAGE and 
                             status_data.get('status') == 'completed')
        
        if prep_completed:
            st.success("✅ Data preparation has already been completed for this run.")
            
            try:
                # Load metadata to show prep results
                metadata = storage.read_json(run_id, constants.METADATA_FILENAME)
                prep_info = metadata.get('prep_info', {}) if metadata else {}
                
                if prep_info:
                    # Display summary
                    st.subheader("Preparation Results Summary")
                    
                    final_shape = prep_info.get('final_shape_after_prep', [0, 0])
                    cleaning_steps = prep_info.get('cleaning_steps_performed', [])
                    encoders_info = prep_info.get('encoders_scalers_info', {})
                    profile_report_path = prep_info.get('profiling_report_path')
                    
                    # Main metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Final Data Shape", f"{final_shape[0]:,} rows × {final_shape[1]:,} columns")
                    
                    with col2:
                        st.metric("Cleaning Steps", len(cleaning_steps))
                    
                    with col3:
                        st.metric("Encoders/Scalers", len(encoders_info))
                    
                    # Show completion timestamp if available
                    if status_data.get('timestamp'):
                        try:
                            ts = datetime.fromisoformat(status_data['timestamp'].replace('Z', '+00:00'))
                            formatted_ts = ts.strftime("%Y-%m-%d %H:%M:%S")
                            st.caption(f"🕒 Preparation completed at: {formatted_ts}")
                        except:
                            st.caption(f"🕒 Completion timestamp: {status_data['timestamp']}")
                    
                    # File Downloads Section
                    st.subheader("📥 Download Results")
                    
                    col_download1, col_download2 = st.columns(2)
                    
                    with col_download1:
                        # Download cleaned data
                        try:
                            run_dir = storage.get_run_dir(run_id)
                            cleaned_data_path = run_dir / constants.CLEANED_DATA_FILE
                            
                            if cleaned_data_path.exists():
                                with open(cleaned_data_path, 'rb') as f:
                                    st.download_button(
                                        label="📊 Download Cleaned Data (CSV)",
                                        data=f.read(),
                                        file_name=f"{run_id}_cleaned_data.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                
                                # Show file info
                                file_size_mb = cleaned_data_path.stat().st_size / (1024 * 1024)
                                st.caption(f"File size: {file_size_mb:.1f} MB")
                            else:
                                st.error("Cleaned data file not found!")
                        except Exception as e:
                            st.error(f"Error loading cleaned data: {e}")
                    
                    with col_download2:
                        # Profile report link/download
                        if profile_report_path:
                            try:
                                profile_full_path = storage.get_run_dir(run_id) / profile_report_path
                                
                                if profile_full_path.exists():
                                    # Show profile report options
                                    st.write("**📈 Data Profile Report:**")
                                    
                                    # Option 1: Download button
                                    with open(profile_full_path, 'rb') as f:
                                        st.download_button(
                                            label="📈 Download Profile Report (HTML)",
                                            data=f.read(),
                                            file_name=f"{run_id}_profile_report.html",
                                            mime="text/html",
                                            use_container_width=True
                                        )
                                    
                                    # Show file info
                                    file_size_mb = profile_full_path.stat().st_size / (1024 * 1024)
                                    st.caption(f"File size: {file_size_mb:.1f} MB")
                                    
                                    # Option 2: Show in iframe (expandable)
                                    with st.expander("🔍 View Profile Report", expanded=False):
                                        st.warning("⚠️ Large reports may take time to load. Consider downloading instead.")
                                        try:
                                            # Read and display HTML content
                                            with open(profile_full_path, 'r', encoding='utf-8') as f:
                                                html_content = f.read()
                                            st.components.v1.html(html_content, height=600, scrolling=True)
                                        except Exception as e:
                                            st.error(f"Could not display report inline: {e}")
                                            st.info("Please use the download button above to view the report.")
                                else:
                                    st.warning("Profile report file not found!")
                            except Exception as e:
                                st.error(f"Error with profile report: {e}")
                        else:
                            st.info("No profile report was generated.")
                    
                    # Detailed Information Section
                    st.subheader("📋 Detailed Information")
                    
                    # Cleaning Steps
                    with st.expander("🧹 Cleaning Steps Performed", expanded=False):
                        if cleaning_steps:
                            st.write("The following cleaning operations were performed:")
                            for i, step in enumerate(cleaning_steps, 1):
                                st.write(f"{i}. {step}")
                        else:
                            st.info("No cleaning steps recorded.")
                    
                    # Encoding Information
                    with st.expander("🔄 Encoding & Transformation Details", expanded=False):
                        if encoders_info:
                            st.write("The following encoders and scalers were applied:")
                            
                            # Group by encoder type
                            encoder_types = {}
                            for encoder_name, encoder_details in encoders_info.items():
                                encoder_type = encoder_details.get('type', 'Unknown')
                                if encoder_type not in encoder_types:
                                    encoder_types[encoder_type] = []
                                encoder_types[encoder_type].append((encoder_name, encoder_details))
                            
                            for encoder_type, encoders in encoder_types.items():
                                st.write(f"**{encoder_type}:**")
                                for encoder_name, details in encoders:
                                    columns = details.get('columns_affected', ['Unknown'])
                                    column_text = ', '.join(columns) if isinstance(columns, list) else str(columns)
                                    st.write(f"  - {encoder_name}: {column_text}")
                                st.write("")
                        else:
                            st.info("No encoding information recorded.")
                    
                    # Model Directory Files
                    with st.expander("📁 Model Artifacts", expanded=False):
                        try:
                            model_dir = storage.get_run_dir(run_id) / "model"
                            if model_dir.exists():
                                model_files = list(model_dir.glob("*"))
                                if model_files:
                                    st.write(f"**{len(model_files)} encoder/scaler files saved:**")
                                    for file_path in sorted(model_files):
                                        file_size_kb = file_path.stat().st_size / 1024
                                        st.write(f"  - `{file_path.name}` ({file_size_kb:.1f} KB)")
                                else:
                                    st.info("No model files found.")
                            else:
                                st.info("Model directory not found.")
                        except Exception as e:
                            st.error(f"Error checking model files: {e}")
                    
                    # Continue to next step
                    if st.button("➡️ Continue to Model Training", type="primary", use_container_width=True, key="continue_to_automl"):
                        st.session_state.pop('prep_run_complete', None)
                        st.session_state['current_page'] = 'automl'
                        st.rerun()
                
                else:
                    st.warning("Preparation completed but detailed results not found in metadata.")
                    
            except Exception as e:
                is_dev_mode = st.session_state.get("developer_mode_active", False)
                utils.display_page_error(e, run_id=run_id, stage_name=constants.PREP_STAGE, dev_mode=is_dev_mode)
        
        else:
            # Check if prep stage failed
            if status_data and status_data.get('stage') == constants.PREP_STAGE and status_data.get('status') == 'failed':
                st.error("❌ Previous data preparation attempt failed.")
                st.error(f"Error: {status_data.get('message', 'Unknown error')}")
                st.info("You can try running data preparation again.")
            else:
                st.info("⏳ Data preparation has not been run yet for this run.")
    
    except Exception as e:
        is_dev_mode = st.session_state.get("developer_mode_active", False)
        utils.display_page_error(e, run_id=run_id, stage_name=constants.PREP_STAGE, dev_mode=is_dev_mode)
        prep_completed = False
    
    # Run Data Preparation Button (show if no existing results)
    if not prep_completed:
        st.subheader("Run Data Preparation")
        st.write("This step will clean your data, encode features for ML, and generate a comprehensive data profile report.")
        
        # Show what will happen
        with st.expander("ℹ️ What happens during data preparation?", expanded=False):
            st.write("""
            **Data Cleaning:**
            - Handle missing values based on feature types
            - Remove duplicate rows
            - Apply data quality fixes
            
            **Feature Encoding:**
            - Convert categorical variables to numeric (one-hot, ordinal)
            - Scale numeric features (StandardScaler)
            - Extract datetime features (year, month, day, etc.)
            - Apply text vectorization (TF-IDF)
            - Save all encoders for later use
            
            **Data Profiling:**
            - Generate comprehensive HTML report with ydata-profiling
            - Show data distributions, correlations, and quality metrics
            - Provide insights into the prepared dataset
            """)
        
        # Check if preparation is already running to prevent multiple runs
        prep_running = st.session_state.get('prep_running', False)
        
        if prep_running:
            st.warning("⏳ Data preparation is currently running. Please wait...")
            st.button("🚀 Start Data Preparation", type="primary", use_container_width=True, disabled=True)
        elif st.button("🚀 Start Data Preparation", type="primary", use_container_width=True):
            # Set running flag
            st.session_state['prep_running'] = True
            
            with st.spinner("Running data preparation... This may take several minutes."):
                try:
                    # Call the prep runner
                    stage_success = prep_runner.run_preparation_stage(run_id)
                    
                    if stage_success:
                        # The runner itself completed successfully
                        st.success("✅ Data preparation completed successfully!")
                        
                        # Load prep results for immediate display
                        try:
                            metadata = storage.read_json(run_id, constants.METADATA_FILENAME)
                            prep_info = metadata.get('prep_info', {}) if metadata else {}
                            
                            if prep_info:
                                final_shape = prep_info.get('final_shape_after_prep', [0, 0])
                                cleaning_steps = prep_info.get('cleaning_steps_performed', [])
                                encoders_info = prep_info.get('encoders_scalers_info', {})
                                
                                st.subheader("🎉 Preparation Results")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Final Shape", f"{final_shape[0]:,} × {final_shape[1]:,}")
                                with col2:
                                    st.metric("Cleaning Steps", len(cleaning_steps))
                                with col3:
                                    st.metric("Encoders Saved", len(encoders_info))
                                
                                # Show key cleaning steps
                                if cleaning_steps:
                                    st.write("**Key cleaning operations:**")
                                    for step in cleaning_steps[:5]:  # Show first 5
                                        st.write(f"• {step}")
                                    if len(cleaning_steps) > 5:
                                        st.write(f"• ... and {len(cleaning_steps) - 5} more steps")
                                
                                # Set completion flag and refresh page to show full results
                                st.balloons()
                                st.session_state['prep_run_complete'] = True
                                st.rerun()
                            
                            else:
                                st.warning("Preparation completed but results not found in metadata.")
                        
                        except Exception as e:
                            is_dev_mode = st.session_state.get("developer_mode_active", False)
                            utils.display_page_error(e, run_id=run_id, stage_name=constants.PREP_STAGE, dev_mode=is_dev_mode)
                    
                    else:
                        # The runner function failed
                        try:
                            status_data = storage.read_json(run_id, constants.STATUS_FILENAME)
                            if status_data and status_data.get('message'):
                                prep_error = Exception(f"Data preparation failed: {status_data['message']}")
                            else:
                                prep_error = Exception("Data preparation failed - check logs for details.")
                            
                            is_dev_mode = st.session_state.get("developer_mode_active", False)
                            utils.display_page_error(prep_error, run_id=run_id, stage_name=constants.PREP_STAGE, dev_mode=is_dev_mode)
                        except:
                            # Fallback error handling
                            prep_error = Exception("Data preparation failed - check logs for details.")
                            is_dev_mode = st.session_state.get("developer_mode_active", False)
                            utils.display_page_error(prep_error, run_id=run_id, stage_name=constants.PREP_STAGE, dev_mode=is_dev_mode)
                
                except Exception as e:
                    is_dev_mode = st.session_state.get("developer_mode_active", False)
                    utils.display_page_error(e, run_id=run_id, stage_name=constants.PREP_STAGE, dev_mode=is_dev_mode)
                finally:
                    # Always clear the running flag
                    if 'prep_running' in st.session_state:
                        del st.session_state['prep_running']
    



if __name__ == "__main__":
    show_prep_page() 