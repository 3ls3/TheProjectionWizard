"""
Model Explainability page for The Projection Wizard.
Provides UI for running SHAP explainability analysis and viewing explanation results.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.step_6_explain import explain_runner
from common import constants, storage, schemas, utils


def show_explain_page():
    """Display the model explainability page."""
    
    # Page Title
    st.title("Step 7: Model Explainability")
    
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
    st.write("This step generates SHAP (SHapley Additive exPlanations) values to help understand your model's predictions and identify key feature importances. Review the analysis before viewing the final results.")
    

    
    # Display Existing Results (if page is revisited)
    st.subheader("Model Explainability Status")
    
    try:
        # Check status.json to see if explainability stage completed
        status_data = storage.read_json(run_id, constants.STATUS_FILENAME)
        explain_completed = False
        
        if status_data:
            explain_completed = (status_data.get('stage') == constants.EXPLAIN_STAGE and 
                               status_data.get('status') == 'completed')
        
        if explain_completed:
            st.success("✅ Model explainability analysis has already been completed for this run.")
            
            try:
                # Load metadata to show explainability results
                metadata = storage.read_json(run_id, constants.METADATA_FILENAME)
                explain_info = metadata.get('explain_info', {}) if metadata else {}
                automl_info = metadata.get('automl_info', {}) if metadata else {}
                
                if explain_info:
                    # Display main results summary
                    st.subheader("🔍 Explainability Analysis Results")
                    
                    tool_used = explain_info.get('tool_used', 'Unknown')
                    explanation_type = explain_info.get('explanation_type', 'Unknown')
                    target_column = explain_info.get('target_column', 'Unknown')
                    task_type = explain_info.get('task_type', 'Unknown')
                    features_explained = explain_info.get('features_explained', 0)
                    samples_used = explain_info.get('samples_used_for_explanation', 0)
                    best_model = automl_info.get('best_model_name', 'Unknown')
                    
                    # Main info cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Explanation Tool", tool_used)
                        st.metric("Analysis Type", explanation_type.replace('_', ' ').title())
                    
                    with col2:
                        st.metric("Model Explained", best_model)
                        st.metric("Task Type", task_type.title())
                    
                    with col3:
                        st.metric("Features Analyzed", f"{features_explained:,}")
                        st.metric("Samples Used", f"{samples_used:,}")
                    
                    # SHAP Plot Display Section
                    st.subheader("📊 SHAP Feature Importance Plot")
                    
                    try:
                        # Check if SHAP plot exists
                        plot_path = storage.get_run_dir(run_id) / constants.PLOTS_DIR / constants.SHAP_SUMMARY_PLOT
                        
                        if plot_path.exists():
                            # Display the SHAP plot
                            st.image(
                                str(plot_path), 
                                caption=f"SHAP Feature Importance for {best_model} ({task_type.title()})",
                                use_container_width=True
                            )
                            
                            # Plot information
                            file_size_kb = plot_path.stat().st_size / 1024
                            st.caption(f"📈 Plot file size: {file_size_kb:.1f} KB")
                            
                            # Download button for the plot
                            with open(plot_path, 'rb') as f:
                                st.download_button(
                                    label="📥 Download SHAP Plot",
                                    data=f.read(),
                                    file_name=f"{run_id}_shap_summary.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                            
                        else:
                            st.error("❌ SHAP plot file not found!")
                            st.write(f"Expected location: `{plot_path}`")
                    
                    except Exception as e:
                        st.error(f"Error displaying SHAP plot: {e}")
                    
                    # Model Performance Context (from AutoML stage)
                    performance_metrics = automl_info.get('performance_metrics', {})
                    if performance_metrics:
                        with st.expander("📈 Model Performance (for context)", expanded=False):
                            st.write("**Performance of the explained model:**")
                            
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
                    
                    # Explanation Information
                    with st.expander("ℹ️ Understanding SHAP Feature Importance", expanded=False):
                        st.write("""
                        **What this plot shows:**
                        - **Feature Importance:** Features are ranked by their average impact on model predictions
                        - **Mean |SHAP Value|:** The average absolute contribution of each feature to the model's decisions
                        - **Top Features:** The most influential features appear at the top of the plot
                        
                        **How to interpret:**
                        - **Higher bars** = More important features for the model
                        - **Feature names** show the original column names from your data
                        - **SHAP values** represent how much each feature pushes the prediction above or below the average
                        
                        **SHAP (SHapley Additive exPlanations):**
                        - Provides consistent and accurate feature importance scores
                        - Based on game theory principles
                        - Shows both the magnitude and direction of feature impacts
                        - Allows for both global (dataset-wide) and local (individual prediction) explanations
                        """)
                    
                    # Show completion timestamp if available
                    completion_time = explain_info.get('explain_completed_at')
                    if completion_time:
                        try:
                            ts = datetime.fromisoformat(completion_time.replace('Z', '+00:00'))
                            formatted_ts = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
                            st.caption(f"🕒 Analysis completed at: {formatted_ts}")
                        except:
                            st.caption(f"🕒 Completion timestamp: {completion_time}")
                    
                    # Continue to next step
                    if st.button("➡️ Continue to Results", type="primary", use_container_width=True, key="continue_to_results_page"):
                        st.session_state.pop('explain_run_complete', None)
                        st.session_state['current_page'] = 'results'
                        st.rerun()
                
                else:
                    st.warning("Explainability analysis completed but detailed results not found in metadata.")
                    
            except Exception as e:
                is_dev_mode = st.session_state.get("developer_mode_active", False)
                utils.display_page_error(e, run_id=run_id, stage_name=constants.EXPLAIN_STAGE, dev_mode=is_dev_mode)
        
        else:
            # Check if explainability stage failed
            if status_data and status_data.get('stage') == constants.EXPLAIN_STAGE and status_data.get('status') == 'failed':
                st.error("❌ Previous explainability analysis attempt failed.")
                st.error(f"Error: {status_data.get('message', 'Unknown error')}")
                st.info("You can try running the analysis again.")
            else:
                st.info("⏳ Model explainability analysis has not been run yet for this run.")
    
    except Exception as e:
        is_dev_mode = st.session_state.get("developer_mode_active", False)
        utils.display_page_error(e, run_id=run_id, stage_name=constants.EXPLAIN_STAGE, dev_mode=is_dev_mode)
        explain_completed = False
    
    # Run Explainability Button (show if no existing results)
    if not explain_completed:
        st.subheader("Run Model Explainability Analysis")
        st.write("This step will generate global explanations for your trained model using SHAP (SHapley Additive exPlanations).")
        
        # Show what will happen
        with st.expander("ℹ️ What happens during explainability analysis?", expanded=False):
            st.write("""
            **SHAP Analysis Process:**
            - Load your trained PyCaret model pipeline from the AutoML stage
            - Load the cleaned feature data (without target column)
            - Create SHAP explainer appropriate for your model type
            - Calculate SHAP values to understand feature importance
            - Generate a global summary plot showing feature importance rankings
            - Save the plot for download and future reference
            
            **What you'll get:**
            - **Global Feature Importance:** Which features matter most to your model
            - **Visual Summary:** Clear bar chart showing feature impact rankings
            - **Quantitative Scores:** Numerical SHAP values indicating feature influence
            - **Downloadable Plot:** High-quality PNG image for reports and presentations
            
            **SHAP Benefits:**
            - **Model-Agnostic:** Works with any machine learning model
            - **Theoretically Sound:** Based on game theory (Shapley values)
            - **Consistent:** Provides reliable and consistent explanations
            - **Actionable:** Helps identify which features to focus on
            
            **Analysis Speed:**
            - Small datasets (< 1000 rows): Usually under 30 seconds
            - Medium datasets (1000-10000 rows): 1-3 minutes
            - Large datasets: May sample data for performance
            """)
        
        # Check if analysis is already running to prevent multiple runs
        explain_running = st.session_state.get('explain_running', False)
        
        if explain_running:
            st.warning("⏳ Explainability analysis is currently running. Please wait...")
            st.button("🚀 Start Explainability Analysis", type="primary", use_container_width=True, disabled=True)
        elif st.button("🚀 Start Explainability Analysis", type="primary", use_container_width=True):
            # Set running flag
            st.session_state['explain_running'] = True
            
            with st.spinner("Analyzing model... This may take a few minutes depending on data size."):
                try:
                    # Call the explainability runner
                    stage_success = explain_runner.run_explainability_stage(run_id)
                    
                    if stage_success:
                        # The runner itself completed successfully
                        st.success("✅ Model explainability analysis completed successfully!")
                        
                        # Load results for immediate display
                        try:
                            summary = explain_runner.get_explainability_stage_summary(run_id)
                            
                            if summary:
                                st.subheader("🎉 Analysis Results")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Tool Used", summary.get('tool_used', 'Unknown'))
                                with col2:
                                    st.metric("Features Analyzed", f"{summary.get('features_explained', 0):,}")
                                with col3:
                                    st.metric("Samples Used", f"{summary.get('samples_used', 0):,}")
                                
                                # Show SHAP plot immediately if available
                                if summary.get('plot_file_exists', False):
                                    st.write("**🎯 SHAP Feature Importance:**")
                                    plot_path = summary.get('plot_path')
                                    if plot_path:
                                        st.image(
                                            plot_path,
                                            caption="SHAP Feature Importance Summary",
                                            use_container_width=True
                                        )
                                
                                # Set completion flag and refresh page to show full results
                                st.balloons()
                                st.session_state['explain_run_complete'] = True
                                st.rerun()
                            
                            else:
                                st.warning("Analysis completed but results summary not available.")
                        
                        except Exception as e:
                            is_dev_mode = st.session_state.get("developer_mode_active", False)
                            utils.display_page_error(e, run_id=run_id, stage_name=constants.EXPLAIN_STAGE, dev_mode=is_dev_mode)
                    
                    else:
                        # The runner function failed
                        try:
                            status_data = storage.read_json(run_id, constants.STATUS_FILENAME)
                            if status_data and status_data.get('message'):
                                explain_error = Exception(f"Model explainability analysis failed: {status_data['message']}")
                            else:
                                explain_error = Exception("Model explainability analysis failed - check logs for details.")
                            
                            is_dev_mode = st.session_state.get("developer_mode_active", False)
                            utils.display_page_error(explain_error, run_id=run_id, stage_name=constants.EXPLAIN_STAGE, dev_mode=is_dev_mode)
                        except:
                            # Fallback error handling
                            explain_error = Exception("Model explainability analysis failed - check logs for details.")
                            is_dev_mode = st.session_state.get("developer_mode_active", False)
                            utils.display_page_error(explain_error, run_id=run_id, stage_name=constants.EXPLAIN_STAGE, dev_mode=is_dev_mode)
                
                except Exception as e:
                    is_dev_mode = st.session_state.get("developer_mode_active", False)
                    utils.display_page_error(e, run_id=run_id, stage_name=constants.EXPLAIN_STAGE, dev_mode=is_dev_mode)
                finally:
                    # Always clear the running flag
                    if 'explain_running' in st.session_state:
                        del st.session_state['explain_running']
    



if __name__ == "__main__":
    show_explain_page() 