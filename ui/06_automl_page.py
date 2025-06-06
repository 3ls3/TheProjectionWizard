"""
AutoML page for The Projection Wizard.
Provides UI for running AutoML (PyCaret) training and viewing model results.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from step_5_automl import automl_runner
from common import constants, storage, schemas


def show_automl_page():
    """Display the AutoML page."""
    
    # Page Title
    st.title("Step 6: AutoML Model Training")
    
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
    
    # Debug info (can be removed later)
    if st.checkbox("🔧 Show Debug Info", value=False):
        st.write("**Session State:**")
        st.write(f"- force_automl_rerun: {st.session_state.get('force_automl_rerun', 'Not set')}")
        st.write(f"- current_page: {st.session_state.get('current_page', 'Not set')}")
        
        # Check model files
        try:
            run_dir = storage.get_run_dir(run_id)
            model_file_path = run_dir / constants.MODEL_DIR / 'pycaret_pipeline.pkl'
            if model_file_path.exists():
                import os
                mtime = os.path.getmtime(model_file_path)
                last_modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                st.write(f"- pycaret_pipeline.pkl last modified: {last_modified}")
                st.write(f"- pycaret_pipeline.pkl size: {model_file_path.stat().st_size / 1024:.1f} KB")
            else:
                st.write("- pycaret_pipeline.pkl: Not found")
        except Exception as e:
            st.write(f"- pycaret_pipeline.pkl: Error checking ({e})")
    
    # Display Existing Results (if page is revisited)
    st.subheader("AutoML Training Status")
    
    # Check if user requested a re-run
    force_rerun = st.session_state.get('force_automl_rerun', False)
    
    try:
        # Check status.json to see if automl stage completed
        status_data = storage.read_json(run_id, constants.STATUS_FILENAME)
        automl_completed = False
        
        if status_data and not force_rerun:
            automl_completed = (status_data.get('stage') == constants.AUTOML_STAGE and 
                              status_data.get('status') == 'completed')
        
        if automl_completed:
            st.success("✅ AutoML training has already been completed for this run.")
            
            try:
                # Load metadata to show automl results
                metadata = storage.read_json(run_id, constants.METADATA_FILENAME)
                automl_info = metadata.get('automl_info', {}) if metadata else {}
                
                if automl_info:
                    # Display main results summary
                    st.subheader("🤖 AutoML Results Summary")
                    
                    tool_used = automl_info.get('tool_used', 'Unknown')
                    best_model_name = automl_info.get('best_model_name', 'Unknown')
                    task_type = automl_info.get('task_type', 'Unknown')
                    target_column = automl_info.get('target_column', 'Unknown')
                    dataset_shape = automl_info.get('dataset_shape_for_training', [0, 0])
                    
                    # Main info cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("AutoML Tool", tool_used)
                        st.metric("Task Type", task_type.title())
                    
                    with col2:
                        st.metric("Best Model", best_model_name)
                        st.metric("Target Column", target_column)
                    
                    with col3:
                        if dataset_shape:
                            st.metric("Training Data", f"{dataset_shape[0]:,} × {dataset_shape[1]:,}")
                        
                        # Check if model file exists
                        try:
                            model_file_path = storage.get_run_dir(run_id) / constants.MODEL_DIR / 'pycaret_pipeline.pkl'
                            if model_file_path.exists():
                                file_size_kb = model_file_path.stat().st_size / 1024
                                st.metric("Model File", f"{file_size_kb:.1f} KB")
                            else:
                                st.metric("Model File", "❌ Not Found")
                        except:
                            st.metric("Model File", "❌ Error")
                    
                    # Performance Metrics Section
                    performance_metrics = automl_info.get('performance_metrics', {})
                    if performance_metrics:
                        st.subheader("📊 Model Performance")
                        
                        # Display metrics in a nice format
                        metric_cols = st.columns(min(len(performance_metrics), 4))
                        
                        for i, (metric_name, metric_value) in enumerate(performance_metrics.items()):
                            with metric_cols[i % len(metric_cols)]:
                                if isinstance(metric_value, float):
                                    if metric_name in ['AUC', 'Accuracy', 'F1', 'Recall', 'Precision']:
                                        # Show as percentage for these metrics
                                        st.metric(metric_name, f"{metric_value:.1%}")
                                    else:
                                        # Show raw value for others (like RMSE)
                                        st.metric(metric_name, f"{metric_value:.4f}")
                                else:
                                    st.metric(metric_name, str(metric_value))
                        
                        # Also show in table format
                        with st.expander("📋 Detailed Performance Metrics", expanded=False):
                            metrics_data = []
                            for metric_name, metric_value in performance_metrics.items():
                                if isinstance(metric_value, (int, float)):
                                    metrics_data.append({
                                        "Metric": metric_name,
                                        "Value": f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                                    })
                            
                            if metrics_data:
                                df_metrics = pd.DataFrame(metrics_data)
                                st.table(df_metrics)
                    
                    # Show completion timestamp if available
                    completion_time = automl_info.get('automl_completed_at')
                    if completion_time:
                        try:
                            ts = datetime.fromisoformat(completion_time.replace('Z', '+00:00'))
                            formatted_ts = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
                            st.caption(f"🕒 Training completed at: {formatted_ts}")
                        except:
                            st.caption(f"🕒 Completion timestamp: {completion_time}")
                    
                    # Model Artifacts Section
                    st.subheader("💾 Model Artifacts")
                    
                    col_model1, col_model2 = st.columns(2)
                    
                    with col_model1:
                        # PyCaret Pipeline Download
                        try:
                            model_file_path = storage.get_run_dir(run_id) / constants.MODEL_DIR / 'pycaret_pipeline.pkl'
                            
                            if model_file_path.exists():
                                with open(model_file_path, 'rb') as f:
                                    st.download_button(
                                        label="🤖 Download PyCaret Pipeline",
                                        data=f.read(),
                                        file_name=f"{run_id}_pycaret_pipeline.pkl",
                                        mime="application/octet-stream",
                                        use_container_width=True
                                    )
                                
                                # Show file info
                                file_size_kb = model_file_path.stat().st_size / 1024
                                st.caption(f"File size: {file_size_kb:.1f} KB")
                            else:
                                st.error("❌ PyCaret pipeline file not found!")
                        except Exception as e:
                            st.error(f"Error accessing model file: {e}")
                    
                    with col_model2:
                        # Show information about the model
                        st.write("**🔍 Model Information:**")
                        st.write(f"• **Algorithm:** {best_model_name}")
                        st.write(f"• **Tool:** {tool_used}")
                        st.write(f"• **Task:** {task_type.title()}")
                        st.write(f"• **Target:** {target_column}")
                        
                        # Show encoder files count
                        try:
                            model_dir = storage.get_run_dir(run_id) / constants.MODEL_DIR
                            if model_dir.exists():
                                encoder_files = [f for f in model_dir.glob("*.joblib")]
                                st.write(f"• **Encoders:** {len(encoder_files)} preprocessing files")
                        except:
                            pass
                    
                    # Navigation to next step (only show if automl is already complete)
                    st.subheader("Next Steps")
                    col_nav1, col_nav2 = st.columns([3, 1])
                    
                    with col_nav1:
                        if st.button("🔍 Proceed to Model Explanation", type="primary", use_container_width=True):
                            st.session_state['current_page'] = 'explain'
                            st.rerun()
                    
                    with col_nav2:
                        if st.button("🔄 Re-train", use_container_width=True):
                            # Set session state to force re-run and refresh page
                            st.session_state['force_automl_rerun'] = True
                            st.rerun()
                
                else:
                    st.warning("AutoML completed but detailed results not found in metadata.")
                    
            except Exception as e:
                st.error(f"Error loading AutoML results: {str(e)}")
        
        else:
            # Check if automl stage failed
            if status_data and status_data.get('stage') == constants.AUTOML_STAGE and status_data.get('status') == 'failed':
                st.error("❌ Previous AutoML training attempt failed.")
                st.error(f"Error: {status_data.get('message', 'Unknown error')}")
                st.info("You can try running AutoML training again.")
            else:
                st.info("⏳ AutoML training has not been run yet for this run.")
    
    except Exception as e:
        st.error(f"Error checking AutoML status: {str(e)}")
        automl_completed = False
    
    # Run AutoML Button (show if no existing results or user wants to re-run)
    if not automl_completed:
        # Check if previous stages are completed
        try:
            # Validate that prep stage is completed
            validation_success = automl_runner.validate_automl_stage_inputs(run_id)
            
            if not validation_success:
                st.error("❌ **Prerequisites not met for AutoML training.**")
                st.error("Please ensure the following stages are completed:")
                st.write("• Data upload and ingestion")
                st.write("• Target and schema confirmation")
                st.write("• Data validation")
                st.write("• **Data preparation (most important)**")
                
                if st.button("← Go to Data Preparation", use_container_width=True):
                    st.session_state['current_page'] = 'prep'
                    st.rerun()
                return
        
        except Exception as e:
            st.warning(f"Could not validate prerequisites: {e}")
        
        if force_rerun:
            st.subheader("Re-run AutoML Training")
            st.info("⚠️ **Re-running AutoML training** - This will overwrite the previous model.")
        else:
            st.subheader("Run AutoML Training")
            st.write("This step will automatically train and compare multiple machine learning models using PyCaret, then select the best performing one.")
        
        # Show what will happen
        with st.expander("ℹ️ What happens during AutoML training?", expanded=False):
            st.write("""
            **Model Training Process:**
            - Load your cleaned and encoded data from the preparation stage
            - Set up PyCaret environment for your task type (classification/regression)
            - Compare multiple machine learning algorithms using cross-validation
            - Select the best performing model based on appropriate metrics
            - Train the final model on the complete dataset
            - Save the trained pipeline for predictions
            
            **Models Compared:**
            - Logistic Regression / Linear Regression
            - Random Forest
            - Extra Trees
            - Ridge Classifier / Ridge Regression
            - Support Vector Machine
            - Gradient Boosting (LightGBM, XGBoost if available)
            - And more...
            
            **Performance Metrics:**
            - **Classification:** AUC, Accuracy, F1-Score, Precision, Recall
            - **Regression:** R², RMSE, MAE, MAPE
            
            **Output:**
            - Trained PyCaret pipeline (model + preprocessing)
            - Performance metrics and model comparison results
            - Model ready for explanations and predictions
            """)
        
        # Check if training is already running to prevent multiple runs
        automl_running = st.session_state.get('automl_running', False)
        
        if automl_running:
            st.warning("⏳ AutoML training is currently running. Please wait...")
            st.button("🚀 Run AutoML Training", type="primary", use_container_width=True, disabled=True)
        elif st.button("🚀 Run AutoML Training", type="primary", use_container_width=True):
            # Set running flag
            st.session_state['automl_running'] = True
            
            with st.spinner("Training machine learning models... This may take several minutes."):
                try:
                    # Clear the force rerun flag before running
                    if 'force_automl_rerun' in st.session_state:
                        del st.session_state['force_automl_rerun']
                    
                    # Call the automl runner
                    stage_success = automl_runner.run_automl_stage(run_id)
                    
                    if stage_success:
                        # The runner itself completed successfully
                        st.success("✅ AutoML training completed successfully!")
                        
                        # Load results for immediate display
                        try:
                            summary = automl_runner.get_automl_stage_summary(run_id)
                            
                            if summary:
                                st.subheader("🎉 Training Results")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Best Model", summary.get('best_model_name', 'Unknown'))
                                with col2:
                                    st.metric("Task Type", summary.get('task_type', 'Unknown').title())
                                with col3:
                                    st.metric("Target", summary.get('target_column', 'Unknown'))
                                
                                # Show key performance metrics
                                performance_metrics = summary.get('performance_metrics', {})
                                if performance_metrics:
                                    st.write("**🏆 Performance Metrics:**")
                                    metric_text = []
                                    for metric_name, metric_value in performance_metrics.items():
                                        if isinstance(metric_value, float):
                                            if metric_name in ['AUC', 'Accuracy', 'F1', 'Recall', 'Precision']:
                                                metric_text.append(f"**{metric_name}:** {metric_value:.1%}")
                                            else:
                                                metric_text.append(f"**{metric_name}:** {metric_value:.4f}")
                                    
                                    if metric_text:
                                        st.write(" | ".join(metric_text))
                                
                                # Auto-navigate to next step immediately
                                st.balloons()
                                st.success("🔍 Proceeding to Model Explanation...")
                                st.session_state['current_page'] = 'explain'
                                st.rerun()
                                
                                # Clear running flag on successful completion
                                if 'automl_running' in st.session_state:
                                    del st.session_state['automl_running']
                            
                            else:
                                st.warning("Training completed but results summary not available.")
                        
                        except Exception as e:
                            st.error(f"Training completed but error loading results: {str(e)}")
                    
                    else:
                        # The runner function failed
                        st.error("❌ AutoML training failed. Check logs for details.")
                        
                        # Try to get more details from status.json
                        status_data = storage.read_json(run_id, constants.STATUS_FILENAME)
                        if status_data and status_data.get('message'):
                            st.error(f"**Error details:** {status_data['message']}")
                        
                        # Show log file location
                        run_dir_path = storage.get_run_dir(run_id)
                        log_path = run_dir_path / constants.PIPELINE_LOG_FILENAME
                        st.info(f"Check log file for more details: `{log_path}`")
                
                except Exception as e:
                    st.error(f"❌ An error occurred during AutoML training: {str(e)}")
                    st.exception(e)
                finally:
                    # Always clear the running flag
                    if 'automl_running' in st.session_state:
                        del st.session_state['automl_running']
    
    # Navigation section (only show if automl not yet run)
    if not automl_completed:
        st.divider()
        
        if st.button("← Back to Data Preparation", use_container_width=True):
            st.session_state['current_page'] = 'prep'
            st.rerun()


if __name__ == "__main__":
    show_automl_page() 