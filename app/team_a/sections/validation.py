"""
Validation section for the main pipeline flow.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

from eda_validation.validation import setup_expectations, run_validation

def validation_pipeline_section(is_current: bool = True):
    """Validation section for the main pipeline flow."""
    st.header("✅ Data Validation")

    # If not current stage, show summary
    if not is_current and 'validation_results' in st.session_state:
        validation_success = st.session_state.get('validation_success', False)
        validation_results = st.session_state.get('validation_results', {})

        if validation_success:
            total = validation_results.get('total_expectations', 0)
            passed = validation_results.get('successful_expectations', 0)
            with st.expander(f"✅ **Validation Passed** | {passed}/{total} checks passed", expanded=False):
                st.success("🎉 All data quality checks passed successfully!")
        else:
            total = validation_results.get('total_expectations', 0)
            passed = validation_results.get('successful_expectations', 0)
            failed = validation_results.get('failed_expectations', 0)
            override_used = st.session_state.get('validation_override', False)
            if override_used:
                with st.expander(f"⚠️ **Validation Override** | {passed}/{total} passed, proceeded anyway", expanded=False):
                    st.warning("❗ Some validation checks failed, but user chose to proceed")
            else:
                with st.expander(f"❌ **Validation Failed** | {passed}/{total} checks passed", expanded=False):
                    st.error("❌ Data quality issues found")
        return

    # If current stage or no validation results, show full UI
    if 'processed_df' not in st.session_state:
        st.error("❌ No processed data found. Please go back to type override.")
        if is_current and st.button("🔙 Back to Type Override"):
            st.session_state.stage = "type_override"
            st.rerun()
        return

    df = st.session_state['processed_df']

    # 3️⃣ Set stage at the top
    st.session_state["stage"] = "validation"

    # Run validation if not already done
    if 'validation_results' not in st.session_state:
        st.subheader("🔍 Running Data Validation...")
        with st.spinner("Validating data quality..."):
            validation_success, validation_results = run_data_validation(df)
            st.session_state['validation_results'] = validation_results
            st.session_state['validation_success'] = validation_success
        st.rerun()

    # Show validation results
    show_validation_and_cleaning_results(df)

def validation_section(df=None):
    """Handle data validation with Great Expectations directly."""
    if df is None:
        st.header("✅ Data Validation")
        st.write("Validate data quality using Great Expectations.")

        # Check if types have been confirmed
        if not st.session_state.get('types_confirmed', False):
            st.warning("⚠️ Please upload data and confirm types & target selection first in the 'Upload Data' section.")
            return

        # Check if we have processed data
        if 'processed_df' not in st.session_state:
            st.error("❌ No processed data found. Please go back to 'Upload Data' and confirm your type selections.")
            return

        df = st.session_state['processed_df']
        target_column = st.session_state.get('target_column')

        st.success(f"✅ Using processed data with {len(df)} rows, {len(df.columns)} columns")
        st.info(f"🎯 Target column: `{target_column}`")

        # TODO: Implement validation logic with Great Expectations using the processed dataframe
        st.info("🚧 Data validation functionality to be implemented")
    else:
        # Inline validation after type confirmation
        st.markdown("---")
        st.subheader("🔍 Data Validation")
        st.write("Running automatic validation with Great Expectations...")

        # Initialize validation state
        if 'validation_results' not in st.session_state:
            st.session_state['validation_results'] = None
        if 'validation_override' not in st.session_state:
            st.session_state['validation_override'] = False

        # Run validation button
        if st.button("🚀 Run Data Validation", type="primary"):
            with st.spinner("Running validation..."):
                try:
                    # Run the validation process
                    success, results = run_data_validation(df)
                    st.session_state['validation_results'] = results

                    if success:
                        st.success("✅ **Validation PASSED!** All data quality checks passed.")
                        st.balloons()
                    else:
                        st.error("❌ **Validation FAILED!** Some data quality issues were found.")

                except Exception as e:
                    st.error(f"❌ Error during validation: {str(e)}")
                    st.session_state['validation_results'] = {"error": str(e)}

        # Display validation results
        if st.session_state['validation_results']:
            display_validation_results(st.session_state['validation_results'])

def run_data_validation(df):
    """Run Great Expectations validation on the DataFrame."""
    try:
        # Create expectation suite based on user-confirmed types in the DataFrame
        expectations = setup_expectations.create_typed_expectation_suite(df, "streamlit_typed_validation_suite")

        # Convert to proper format for validation
        expectation_suite = {
            "expectation_suite_name": "streamlit_typed_validation_suite",
            "expectations": expectations
        }

        # Run validation
        success, results = run_validation.validate_dataframe_with_suite(df, expectation_suite)

        return success, results

    except Exception as e:
        return False, {"error": str(e)}

def display_validation_results(results):
    """Display validation results in a user-friendly format."""
    if "error" in results:
        st.error(f"Validation error: {results['error']}")
        return

    # Overall status
    overall_success = results.get('overall_success', False)
    total_expectations = results.get('total_expectations', 0)
    successful_expectations = results.get('successful_expectations', 0)
    failed_expectations = results.get('failed_expectations', 0)

    # Status badge
    if overall_success:
        st.success(f"🎉 **Validation Status: PASSED** ({successful_expectations}/{total_expectations} checks passed)")
    else:
        st.error(f"⚠️ **Validation Status: FAILED** ({successful_expectations}/{total_expectations} checks passed)")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Checks", total_expectations)
    with col2:
        st.metric("Passed", successful_expectations, delta=None if overall_success else f"-{failed_expectations}")
    with col3:
        success_rate = successful_expectations / total_expectations if total_expectations > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1%}")

    # Detailed results
    if not overall_success and 'expectation_results' in results:
        st.subheader("🔍 Failed Validation Details")

        failed_results = [r for r in results['expectation_results'] if not r.get('success', True)]

        for i, result in enumerate(failed_results):
            with st.expander(f"❌ Failed Check {i+1}: {result.get('expectation_type', 'Unknown')}", expanded=False):
                st.write(f"**Details:** {result.get('details', 'No details available')}")
                if 'kwargs' in result:
                    st.write(f"**Parameters:** {result['kwargs']}")

        # Override option
        st.markdown("---")
        st.subheader("⚠️ Validation Override")
        st.write("The data validation has failed, but you can choose to proceed anyway.")

        if st.checkbox("🚨 I acknowledge the validation issues and want to proceed anyway"):
            st.session_state['validation_override'] = True
            st.warning("✅ Validation override enabled. You can proceed to EDA and cleaning steps.")
            st.success("🚀 **Ready for next steps (EDA/Cleaning)**")
        else:
            st.session_state['validation_override'] = False
    else:
        # All validations passed
        st.success("🚀 **Ready for next steps (EDA/Cleaning)**")

    # Save results to file
    try:
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = processed_dir / f"validation_report_{timestamp}.json"

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        st.info(f"📄 Validation report saved to: `{results_path}`")

    except Exception as e:
        st.warning(f"⚠️ Could not save validation report: {str(e)}")

def show_validation_and_cleaning_results(df):
    """Show validation results and cleaning options on the same page."""
    validation_success = st.session_state.get('validation_success', False)
    validation_results = st.session_state.get('validation_results', {})

    # Show validation results
    st.subheader("🔍 Data Validation Results")

    if validation_success:
        st.success("🎉 **Validation PASSED!** All data quality checks passed.")
        total = validation_results.get('total_expectations', 0)
        passed = validation_results.get('successful_expectations', 0)
        st.info(f"✅ All {passed}/{total} validation checks passed")

        # On validation pass → transition to cleaning
        st.session_state["stage"] = "cleaning"
        st.rerun()

    else:
        st.error("❌ **Validation FAILED!** Some data quality issues were found.")
        display_validation_results(validation_results)

        # Show abort option for validation failure
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔙 **Abort** - Start Over", type="secondary", key="abort_validation"):
                # Reset pipeline to upload stage
                keys_to_clear = ['uploaded_df', 'processed_df', 'types_confirmed', 'type_overrides',
                               'target_column', 'validation_results', 'validation_success',
                               'cleaned_df', 'cleaning_report', 'cleaning_done']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.stage = "upload"
                st.rerun()
        with col2:
            if st.button("⚠️ **Proceed Anyway**", type="primary", key="proceed_validation"):
                st.session_state['validation_override'] = True
                # On validation fail + proceed → transition to cleaning
                st.session_state["stage"] = "cleaning"
                st.rerun()

def validation_debug_section():
    """Detailed debugging information for validation step."""
    st.header("✅ Data Validation - Debug Information")

    st.subheader("📊 Session State")
    validation_state = {
        "validation_results": "validation_results" in st.session_state,
        "validation_success": st.session_state.get('validation_success', 'Not set'),
        "validation_override": st.session_state.get('validation_override', False)
    }
    st.json(validation_state)

    if 'validation_results' in st.session_state:
        st.subheader("🔍 Validation Results Details")
        results = st.session_state['validation_results']
        st.json(results)

        if 'expectation_results' in results:
            st.subheader("📋 Individual Expectation Results")
            for i, result in enumerate(results['expectation_results']):
                success_icon = "✅" if result.get('success', False) else "❌"
                with st.expander(f"{success_icon} Expectation {i+1}: {result.get('expectation_type', 'Unknown')}"):
                    st.json(result)
