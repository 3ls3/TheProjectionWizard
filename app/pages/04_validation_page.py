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

from pipeline.step_3_validation import validation_runner
from common import constants, storage, schemas, utils


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
    
    # Introductory text
    st.write("This step validates your data against the confirmed schema and data quality expectations using Great Expectations. Review the results before proceeding.")
    

    
    # Display Existing Results (if page is revisited)
    st.subheader("Validation Status")
    
    try:
        # Check validation status from status.json first
        status_data = storage.read_json(run_id, constants.STATUS_FILENAME)
        validation_failed = (status_data and 
                           status_data.get('stage') == constants.VALIDATION_STAGE and 
                           status_data.get('status') == 'failed')
        
        if validation_failed:
            # Show validation failure banner
            st.error("🚫 **Pipeline Stopped: Data Validation Failed**")
            st.error("Critical data quality issues were found that prevent safe model training.")
            
            # Show failure details
            validation_errors = status_data.get('errors', [])
            if validation_errors:
                st.subheader("Critical Issues Found:")
                for i, error in enumerate(validation_errors, 1):
                    st.error(f"**{i}.** {error}")
            
            # Show timestamp
            if 'timestamp' in status_data:
                try:
                    from datetime import datetime
                    ts = datetime.fromisoformat(status_data['timestamp'].replace('Z', '+00:00'))
                    formatted_ts = ts.strftime("%Y-%m-%d %H:%M:%S")
                    st.caption(f"🕒 Validation failed at: {formatted_ts}")
                except:
                    st.caption(f"🕒 Validation failed at: {status_data['timestamp']}")
            
            # Guidance section
            st.subheader("What This Means:")
            st.info("Your data has critical quality issues that could lead to unreliable models. "
                   "The pipeline has been stopped to prevent wasted computation and ensure data quality.")
            
            st.subheader("Recommended Actions:")
            st.info("1. **Fix your source data** - Address the critical issues listed above")
            st.info("2. **Re-upload corrected data** - Start a new run with the fixed dataset")
            st.info("3. **Review data requirements** - Ensure your data meets ML modeling standards")
            
            # Action button
            if st.button("🔄 Start Over with New Data", type="primary", use_container_width=True):
                # Clear session and go back to upload
                if 'run_id' in st.session_state:
                    del st.session_state['run_id']
                st.session_state['current_page'] = 'upload'
                st.rerun()
            
            return  # Exit early, don't show normal validation content
        
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
                    success_rate = (validation_summary.successful_expectations / validation_summary.total_expectations * 100) if validation_summary.total_expectations > 0 else 0
                    if validation_summary.overall_success:
                        st.metric("Overall Status", "✅ PASSED", delta="Success")
                    elif success_rate >= 95.0:
                        st.metric("Overall Status", "⚠️ PASSED*", delta="Minor Issues")
                    else:
                        st.metric("Overall Status", "❌ FAILED", delta="Issues Found")
                
                with col2:
                    st.metric("Total Expectations", validation_summary.total_expectations)
                
                with col3:
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
                
                # Show validation timestamp if available
                ge_meta = validation_summary.results_ge_native.get('meta', {})
                if 'validation_timestamp' in ge_meta:
                    from datetime import datetime
                    try:
                        # Parse and format timestamp
                        ts = datetime.fromisoformat(ge_meta['validation_timestamp'].replace('Z', '+00:00'))
                        formatted_ts = ts.strftime("%Y-%m-%d %H:%M:%S")
                        st.caption(f"🕒 Validation run at: {formatted_ts}")
                    except:
                        st.caption(f"🕒 Validation timestamp: {ge_meta['validation_timestamp']}")
                
                # Add interpretation note
                if not validation_summary.overall_success and success_rate >= 95.0:
                    st.info("ℹ️ **Note:** Validation passed with minor issues. A 98%+ success rate is typically acceptable for ML pipelines. You can proceed to data preparation.")
                elif not validation_summary.overall_success:
                    st.warning("⚠️ **Note:** Validation found significant data quality issues. Review the failures below before proceeding.")
                
                # If failed, provide an expander to show more details
                if not validation_summary.overall_success and validation_summary.failed_expectations > 0:
                    with st.expander("🔍 View Failed Expectations Details", expanded=False):
                        st.subheader("Failed Expectations")
                        
                        # Show details from results_ge_native
                        ge_results = validation_summary.results_ge_native.get('results', [])
                        # Handle both boolean and string success values
                        failed_results = [result for result in ge_results 
                                        if result.get('success') not in [True, 'True', "True"]]
                        
                        if failed_results:
                            for i, failed_result in enumerate(failed_results[:10]):  # Limit to first 10
                                expectation_config = failed_result.get('expectation_config', {})
                                expectation_type = expectation_config.get('expectation_type', 'Unknown')
                                column = expectation_config.get('kwargs', {}).get('column', 'Table-level')
                                
                                st.write(f"**{i+1}. {expectation_type}**")
                                st.write(f"   - Column: `{column}`")
                                st.write(f"   - Success: {failed_result.get('success', 'Unknown')}")
                                
                                # Show expectation details
                                kwargs = expectation_config.get('kwargs', {})
                                if 'mostly' in kwargs:
                                    st.write(f"   - Required threshold: {kwargs['mostly']*100:.1f}%")
                                if 'min_value' in kwargs or 'max_value' in kwargs:
                                    min_val = kwargs.get('min_value', 'None')
                                    max_val = kwargs.get('max_value', 'None')
                                    st.write(f"   - Expected range: {min_val} to {max_val}")
                                if 'value_set' in kwargs:
                                    st.write(f"   - Expected values: {kwargs['value_set']}")
                                if 'type_' in kwargs:
                                    st.write(f"   - Expected type: {kwargs['type_']}")
                                
                                # Show result details if available (from our manual validation)
                                result_detail = failed_result.get('result', {})
                                if result_detail:
                                    if 'observed_value' in result_detail:
                                        st.write(f"   - Observed: {result_detail['observed_value']}")
                                    if 'details' in result_detail:
                                        st.write(f"   - Details: {result_detail['details']}")
                                else:
                                    # Provide helpful context for manual validation results
                                    if expectation_type == "expect_column_values_to_not_be_null":
                                        mostly = kwargs.get('mostly', 1.0)
                                        st.write(f"   - Issue: Column '{column}' has too many null values (required: ≥{mostly*100:.1f}% non-null)")
                                    elif expectation_type == "expect_column_values_to_be_of_type":
                                        expected_type = kwargs.get('type_', 'unknown')
                                        st.write(f"   - Issue: Column '{column}' has incorrect data type (expected: {expected_type})")
                                    elif expectation_type == "expect_column_to_exist":
                                        st.write(f"   - Issue: Column '{column}' is missing from dataset")
                                    elif expectation_type == "expect_column_values_to_be_in_set":
                                        expected_values = kwargs.get('value_set', [])
                                        st.write(f"   - Issue: Column '{column}' contains values outside expected set: {expected_values}")
                                    elif expectation_type == "expect_column_unique_value_count_to_be_between":
                                        min_val = kwargs.get('min_value', 'None')
                                        max_val = kwargs.get('max_value', 'None')
                                        st.write(f"   - Issue: Column '{column}' unique value count outside expected range: {min_val} to {max_val}")
                                    else:
                                        st.write(f"   - Issue: Expectation failed for column '{column}'")
                                
                                st.divider()
                            
                            if len(failed_results) > 10:
                                st.info(f"Showing first 10 of {len(failed_results)} failed expectations.")
                        else:
                            st.info("No detailed failure information available.")
                
                # Continue button (only show if validation allows proceeding and validation not just completed)
                success_rate = (validation_summary.successful_expectations / validation_summary.total_expectations * 100) if validation_summary.total_expectations > 0 else 0
                can_proceed = validation_summary.overall_success or success_rate >= 95.0
                validation_just_completed = st.session_state.get('validation_run_complete', False)
                
                if not validation_just_completed:
                    if can_proceed:
                        if st.button("➡️ Continue to Data Preparation", type="primary", use_container_width=True, key="continue_to_prep"):
                            st.session_state['current_page'] = 'prep'
                            st.rerun()
                    else:
                        # Show disabled button with explanation when validation failed
                        st.button("➡️ Continue to Data Preparation", type="primary", use_container_width=True, disabled=True, 
                                help="Cannot proceed - validation failed with critical issues. Fix your data and start a new run.")
                        st.warning("⚠️ **Cannot proceed to data preparation.** Validation found critical issues that must be resolved first.")
                
            except Exception as e:
                is_dev_mode = st.session_state.get("developer_mode_active", False)
                utils.display_page_error(e, run_id=run_id, stage_name=constants.VALIDATION_STAGE, dev_mode=is_dev_mode)
                if is_dev_mode:
                    st.json(validation_report_data)  # Show raw data for debugging in dev mode
        
        else:
            st.info("⏳ Data validation has not been run yet for this run.")
    
    except Exception as e:
        is_dev_mode = st.session_state.get("developer_mode_active", False)
        utils.display_page_error(e, run_id=run_id, stage_name=constants.VALIDATION_STAGE, dev_mode=is_dev_mode)
        validation_report_data = None
    
    # Show continue button if validation was just completed
    if st.session_state.get('validation_run_complete', False):
        # Check if the just-completed validation allows proceeding
        try:
            validation_report_data = storage.read_json(run_id, constants.VALIDATION_FILENAME)
            if validation_report_data:
                validation_summary = schemas.ValidationReportSummary(**validation_report_data)
                success_rate = (validation_summary.successful_expectations / validation_summary.total_expectations * 100) if validation_summary.total_expectations > 0 else 0
                can_proceed = validation_summary.overall_success or success_rate >= 95.0
                
                if can_proceed:
                    if st.button("➡️ Continue to Data Preparation", type="primary", use_container_width=True, key="continue_after_run"):
                        st.session_state['current_page'] = 'prep'
                        st.session_state.pop('validation_run_complete', None)
                        st.rerun()
                else:
                    # Show disabled button when validation failed
                    st.button("➡️ Continue to Data Preparation", type="primary", use_container_width=True, disabled=True, 
                            help="Cannot proceed - validation failed with critical issues. Fix your data and start a new run.")
                    st.warning("⚠️ **Cannot proceed to data preparation.** Validation found critical issues that must be resolved first.")
                    
                    # Provide option to start over
                    if st.button("🔄 Start Over with New Data", type="secondary", use_container_width=True):
                        # Clear session and go back to upload
                        if 'run_id' in st.session_state:
                            del st.session_state['run_id']
                        st.session_state.pop('validation_run_complete', None)
                        st.session_state['current_page'] = 'upload'
                        st.rerun()
        except Exception:
            # If we can't read validation results, don't show continue button
            st.error("Error reading validation results. Cannot determine if proceeding is safe.")
    
    # Run Validation Button (show if no existing results)
    if validation_report_data is None:
        st.subheader("Run Data Validation")
        st.write("This step will validate your data against the confirmed schema using Great Expectations.")
        
        # Check if validation is already running to prevent multiple runs
        validation_running = st.session_state.get('validation_running', False)
        
        if validation_running:
            st.warning("⏳ Validation is currently running. Please wait...")
            st.button("🚀 Start Data Validation", type="primary", use_container_width=True, disabled=True)
        elif st.button("🚀 Start Data Validation", type="primary", use_container_width=True):
            # Set running flag
            st.session_state['validation_running'] = True
            
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
                                
                                success_rate = (validation_summary.successful_expectations / validation_summary.total_expectations * 100) if validation_summary.total_expectations > 0 else 0
                                
                                if validation_summary.overall_success:
                                    st.success(f"🎉 **Validation PASSED!** {validation_summary.successful_expectations}/{validation_summary.total_expectations} expectations met ({success_rate:.1f}%).")
                                elif success_rate >= 95.0:
                                    st.warning(f"⚠️ **Validation PASSED with minor issues.** {validation_summary.successful_expectations}/{validation_summary.total_expectations} expectations met ({success_rate:.1f}%).")
                                else:
                                    st.error(f"❌ **Validation FAILED.** {validation_summary.successful_expectations}/{validation_summary.total_expectations} expectations met ({success_rate:.1f}%).")
                                
                                # Show basic metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Expectations", validation_summary.total_expectations)
                                with col2:
                                    st.metric("Successful", validation_summary.successful_expectations)
                                with col3:
                                    st.metric("Failed", validation_summary.failed_expectations)
                                
                                # Set completion flag and refresh page to show continue button
                                st.balloons()
                                st.session_state['validation_run_complete'] = True
                                st.rerun()
                                
                            except Exception as e:
                                is_dev_mode = st.session_state.get("developer_mode_active", False)
                                utils.display_page_error(e, run_id=run_id, stage_name=constants.VALIDATION_STAGE, dev_mode=is_dev_mode)
                                if is_dev_mode:
                                    st.json(validation_report_data)
                        else:
                            st.error("Validation completed but results could not be loaded.")
                    
                    else:
                        # The runner function failed - check if it's a validation failure or execution error
                        try:
                            status_data = storage.read_status(run_id)
                            if status_data and status_data.get('status') == 'failed':
                                # Pipeline stopped due to validation failure
                                st.error("🚫 **Pipeline Stopped: Validation Failed**")
                                st.error("The data validation found critical issues that prevent safe model training.")
                                
                                # Show failure reasons
                                validation_errors = status_data.get('errors', [])
                                if validation_errors:
                                    st.subheader("Critical Issues Found:")
                                    for i, error in enumerate(validation_errors, 1):
                                        st.error(f"{i}. {error}")
                                
                                # Provide guidance
                                st.info("**Next Steps:**")
                                st.info("1. Review your data quality and fix critical issues")
                                st.info("2. Re-upload corrected data and run validation again")
                                st.info("3. Or adjust validation thresholds if the failures are acceptable")
                                
                                # Option to restart with new data
                                if st.button("🔄 Start Over with New Data", type="secondary"):
                                    # Clear session and go back to upload
                                    if 'run_id' in st.session_state:
                                        del st.session_state['run_id']
                                    st.session_state['current_page'] = 'upload'
                                    st.rerun()
                            else:
                                # Regular execution error
                                st.error("❌ Validation stage execution failed. Check logs.")
                                
                                # Try to get more details from status.json
                                if status_data and 'errors' in status_data:
                                    st.error("Error details:")
                                    for error in status_data['errors']:
                                        st.code(error)
                        except:
                            # Regular execution error - use standardized error handling
                            execution_error = Exception("Validation stage execution failed.")
                            is_dev_mode = st.session_state.get("developer_mode_active", False)
                            utils.display_page_error(execution_error, run_id=run_id, stage_name=constants.VALIDATION_STAGE, dev_mode=is_dev_mode)
                
                except Exception as e:
                    is_dev_mode = st.session_state.get("developer_mode_active", False)
                    utils.display_page_error(e, run_id=run_id, stage_name=constants.VALIDATION_STAGE, dev_mode=is_dev_mode)
                finally:
                    # Always clear the running flag
                    if 'validation_running' in st.session_state:
                        del st.session_state['validation_running']
    



if __name__ == "__main__":
    show_validation_page() 