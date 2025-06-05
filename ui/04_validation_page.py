"""
Data validation page for The Projection Wizard.
Provides UI for running and viewing data validation results.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from step_3_validation import validation_runner
from common import constants, storage, schemas


def show_validation_page():
    """Display the data validation page."""
    
    # Page Title
    st.title("Step 4: Data Validation")
    
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
    
    # Display Existing Results (if page is revisited)
    st.subheader("Validation Status")
    
    try:
        # Try to read existing validation.json
        validation_report_data = storage.read_json(run_id, constants.VALIDATION_FILENAME)
        
        if validation_report_data is not None:
            st.success("✅ Data validation has already been completed for this run.")
            
            try:
                # Parse with ValidationReportSummary (only the top summary part)
                validation_summary = schemas.ValidationReportSummary(**validation_report_data)
                
                # Display summary
                st.subheader("Validation Results Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if validation_summary.overall_success:
                        st.metric("Overall Status", "✅ PASSED", delta="Success")
                    else:
                        st.metric("Overall Status", "❌ FAILED", delta="Issues Found")
                
                with col2:
                    st.metric("Total Expectations", validation_summary.total_expectations)
                
                with col3:
                    success_rate = (validation_summary.successful_expectations / validation_summary.total_expectations * 100) if validation_summary.total_expectations > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Additional metrics
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    st.metric("Successful", validation_summary.successful_expectations, delta="Passed")
                
                with col5:
                    st.metric("Failed", validation_summary.failed_expectations, delta="Failed" if validation_summary.failed_expectations > 0 else None)
                
                with col6:
                    if validation_summary.ge_version:
                        st.metric("GE Version", validation_summary.ge_version)
                
                # If failed, provide an expander to show more details
                if not validation_summary.overall_success and validation_summary.failed_expectations > 0:
                    with st.expander("🔍 View Failed Expectations Details", expanded=False):
                        st.subheader("Failed Expectations")
                        
                        # Show details from results_ge_native
                        ge_results = validation_summary.results_ge_native.get('results', [])
                        failed_results = [result for result in ge_results if not result.get('success', True)]
                        
                        if failed_results:
                            for i, failed_result in enumerate(failed_results[:10]):  # Limit to first 10
                                expectation_config = failed_result.get('expectation_config', {})
                                expectation_type = expectation_config.get('expectation_type', 'Unknown')
                                column = expectation_config.get('kwargs', {}).get('column', 'Table-level')
                                
                                st.write(f"**{i+1}. {expectation_type}**")
                                st.write(f"   - Column: `{column}`")
                                
                                # Show result details if available
                                result_detail = failed_result.get('result', {})
                                if 'observed_value' in result_detail:
                                    st.write(f"   - Observed: {result_detail['observed_value']}")
                                if 'details' in result_detail:
                                    st.write(f"   - Details: {result_detail['details']}")
                                
                                st.divider()
                            
                            if len(failed_results) > 10:
                                st.info(f"Showing first 10 of {len(failed_results)} failed expectations.")
                        else:
                            st.info("No detailed failure information available.")
                
                # Navigation to next step
                st.subheader("Next Steps")
                col_nav1, col_nav2 = st.columns(2)
                
                with col_nav1:
                    if st.button("🚀 Proceed to Data Preparation", type="primary", use_container_width=True):
                        st.session_state['current_page'] = 'prep'
                        st.rerun()
                
                with col_nav2:
                    if st.button("🔄 Re-run Validation", use_container_width=True):
                        # Clear existing results and allow re-run
                        st.info("Click 'Run Data Validation Checks' below to re-run validation.")
                        validation_report_data = None  # This will trigger the validation section below
                
            except Exception as e:
                st.error(f"Error parsing validation results: {str(e)}")
                st.json(validation_report_data)  # Show raw data for debugging
        
        else:
            st.info("⏳ Data validation has not been run yet for this run.")
    
    except Exception as e:
        st.error(f"Error checking validation status: {str(e)}")
        validation_report_data = None
    
    # Run Validation Button (show if no existing results or user wants to re-run)
    if validation_report_data is None:
        st.subheader("Run Data Validation")
        st.write("This step will validate your data against the confirmed schema using Great Expectations.")
        
        if st.button("Run Data Validation Checks", type="primary", use_container_width=True):
            with st.spinner("Running validation..."):
                try:
                    # Call the validation runner
                    stage_success = validation_runner.run_validation_stage(run_id)
                    
                    if stage_success:
                        # The runner itself completed successfully
                        st.success("✅ Data validation process completed.")
                        
                        # Load validation.json
                        validation_report_data = storage.read_json(run_id, constants.VALIDATION_FILENAME)
                        
                        if validation_report_data:
                            try:
                                # Parse with ValidationReportSummary
                                validation_summary = schemas.ValidationReportSummary(**validation_report_data)
                                
                                # Display summary
                                st.subheader("Validation Results")
                                
                                if validation_summary.overall_success:
                                    st.success(f"🎉 **Validation PASSED!** {validation_summary.successful_expectations}/{validation_summary.total_expectations} expectations met.")
                                else:
                                    st.warning(f"⚠️ **Validation completed with issues.** {validation_summary.successful_expectations}/{validation_summary.total_expectations} expectations met.")
                                
                                # Show basic metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Expectations", validation_summary.total_expectations)
                                with col2:
                                    st.metric("Successful", validation_summary.successful_expectations)
                                with col3:
                                    st.metric("Failed", validation_summary.failed_expectations)
                                
                                # Provide navigation to next step
                                st.balloons()
                                st.info("🚀 You can now proceed to Data Preparation.")
                                
                                if st.button("Continue to Data Preparation", use_container_width=True):
                                    st.session_state['current_page'] = 'prep'
                                    st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error parsing validation results: {str(e)}")
                                st.json(validation_report_data)
                        else:
                            st.error("Validation completed but results could not be loaded.")
                    
                    else:
                        # The runner function failed
                        st.error("❌ Validation stage execution failed. Check logs.")
                        
                        # Try to get more details from status.json
                        status_data = storage.read_status(run_id)
                        if status_data and 'errors' in status_data:
                            st.error("Error details:")
                            for error in status_data['errors']:
                                st.code(error)
                        
                        # Show log file location
                        run_dir_path = storage.get_run_dir(run_id)
                        log_path = run_dir_path / constants.PIPELINE_LOG_FILENAME
                        st.info(f"Check log file for more details: `{log_path}`")
                
                except Exception as e:
                    st.error(f"❌ An error occurred during validation: {str(e)}")
                    st.exception(e)
    
    # Navigation section
    st.divider()
    st.subheader("Navigation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Schema Confirmation", use_container_width=True):
            st.session_state['current_page'] = 'schema_confirmation'
            st.rerun()
    
    with col2:
        # Only enable next step if validation has passed
        try:
            validation_report_data = storage.read_json(run_id, constants.VALIDATION_FILENAME)
            can_proceed = False
            if validation_report_data:
                validation_summary = schemas.ValidationReportSummary(**validation_report_data)
                can_proceed = validation_summary.overall_success
        except:
            can_proceed = False
        
        if can_proceed:
            if st.button("Next: Data Preparation →", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'prep'
                st.rerun()
        else:
            st.button("Next: Data Preparation →", disabled=True, help="Complete validation successfully first")


if __name__ == "__main__":
    show_validation_page() 