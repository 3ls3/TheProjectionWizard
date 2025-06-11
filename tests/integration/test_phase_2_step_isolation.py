"""
Phase 2 Step-by-Step Isolation Testing
Implements isolated testing of individual pipeline steps.

This module focuses on Step 2: Schema Definition testing to verify
correct feature type classification, especially ensuring numeric features
are properly identified and not misclassified as categorical.

Part of the 5-phase testing plan for fixing prediction pipeline bugs.
"""

import json
import time
import tempfile
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from io import BytesIO

# System path setup for imports
import sys
sys.path.append('.')

# Core pipeline imports
from pipeline.step_2_schema.feature_definition_logic import (
    suggest_initial_feature_schemas_from_gcs,
    suggest_initial_feature_schemas
)

# GCS utilities
from api.utils.gcs_utils import (
    PROJECT_BUCKET_NAME, upload_to_gcs, download_from_gcs, 
    check_gcs_file_exists, upload_run_file, download_run_file
)

# Common utilities
from common import constants
from common.logger import get_logger, get_stage_logger

# Initialize logger function (will be used with test run id)
def get_test_logger(run_id: str):
    return get_stage_logger(run_id, "test_step2")


class TestResult:
    """Enhanced TestResult class for structured test reporting."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.status = "running"
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        
        # Data storage
        self.measurements = {}
        self.info = {}
        self.assertions = []
        self.errors = []
        self.messages = []
        
    def add_measurement(self, key: str, value: float, unit: str = ""):
        """Add a performance measurement."""
        self.measurements[key] = {
            "value": value,
            "unit": unit,
            "timestamp": time.time()
        }
        
    def add_info(self, key: str, value: Any):
        """Add contextual information."""
        self.info[key] = value
        
    def assert_equals(self, expected: Any, actual: Any, message: str = ""):
        """Assert two values are equal."""
        assertion = {
            "type": "equals",
            "expected": expected,
            "actual": actual,
            "message": message,
            "passed": expected == actual,
            "timestamp": time.time()
        }
        self.assertions.append(assertion)
        
        if not assertion["passed"]:
            error_msg = f"Assertion failed: {message}. Expected: {expected}, Got: {actual}"
            self.errors.append(error_msg)
            
    def assert_in_range(self, value: Any, min_val: Any, max_val: Any, message: str = ""):
        """Assert value is within specified range."""
        in_range = min_val <= value <= max_val
        assertion = {
            "type": "in_range",
            "value": value,
            "min": min_val,
            "max": max_val,
            "message": message,
            "passed": in_range,
            "timestamp": time.time()
        }
        self.assertions.append(assertion)
        
        if not assertion["passed"]:
            error_msg = f"Range assertion failed: {message}. Value {value} not in range [{min_val}, {max_val}]"
            self.errors.append(error_msg)
            
    def assert_true(self, condition: bool, message: str = ""):
        """Assert condition is true."""
        assertion = {
            "type": "true",
            "condition": condition,
            "message": message,
            "passed": condition,
            "timestamp": time.time()
        }
        self.assertions.append(assertion)
        
        if not assertion["passed"]:
            error_msg = f"Truth assertion failed: {message}"
            self.errors.append(error_msg)
            
    def assert_feature_type(self, feature_name: str, expected_type: str, actual_type: str, message: str = ""):
        """Assert feature type classification is correct."""
        assertion = {
            "type": "feature_type",
            "feature_name": feature_name,
            "expected_type": expected_type,
            "actual_type": actual_type,
            "message": message,
            "passed": expected_type == actual_type,
            "timestamp": time.time()
        }
        self.assertions.append(assertion)
        
        if not assertion["passed"]:
            error_msg = f"Feature type assertion failed: {message}. Feature '{feature_name}' expected type '{expected_type}', got '{actual_type}'"
            self.errors.append(error_msg)
            
    def success(self, message: str):
        """Mark test as successful with message."""
        self.status = "success"
        self.messages.append(f"SUCCESS: {message}")
        self._finalize()
        
    def error(self, message: str):
        """Mark test as failed with error message."""
        self.status = "error"
        self.errors.append(message)
        self.messages.append(f"ERROR: {message}")
        self._finalize()
        
    def _finalize(self):
        """Finalize test timing."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        if self.end_time is None:
            self._finalize()
            
        return {
            "test_name": self.test_name,
            "status": self.status,
            "duration": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "measurements": self.measurements,
            "info": self.info,
            "assertions": self.assertions,
            "errors": self.errors,
            "messages": self.messages,
            "summary": {
                "total_assertions": len(self.assertions),
                "passed_assertions": len([a for a in self.assertions if a["passed"]]),
                "failed_assertions": len([a for a in self.assertions if not a["passed"]]),
                "total_errors": len(self.errors),
                "has_measurements": len(self.measurements) > 0
            }
        }


class Step2SchemaDefinitionTest:
    """
    Step 2 Schema Definition Isolation Tests.
    
    Tests feature type classification to ensure:
    1. Numeric features (bedrooms, garage_spaces, etc.) are NOT classified as categorical
    2. Categorical features are properly identified
    3. Schema generation produces valid output
    """
    
    def __init__(self):
        self.cleanup_files = []  # Track files for cleanup
        
    def test_step_2_isolated_regression(self) -> TestResult:
        """
        Test Step 2 Schema Definition with regression dataset (house prices).
        
        Critical focus: Ensure numeric features like bedrooms, garage_spaces, 
        neighborhood_quality_score are classified as numeric, not categorical.
        """
        result = TestResult("step_2_isolated_regression")
        
        try:
            # Create isolated test environment first to get test_run_id for logger
            test_run_id = self._setup_isolated_test_environment("regression", result)
            test_logger = get_test_logger(test_run_id)
            
            test_logger.info("🧪 Starting Phase 2 Step 2 Regression Test")
            result.add_info("dataset_type", "regression")
            result.add_info("test_timestamp", datetime.now().isoformat())
            
            # Generate house prices test data and upload to GCS
            test_data = self._generate_regression_test_data()
            upload_success = self._upload_test_data_to_gcs(test_run_id, test_data, result, test_logger)
            
            if not upload_success:
                result.error("Failed to upload test data to GCS")
                return result
                
            # Execute Step 2 logic
            schema_suggestions = self._execute_step_2_logic(test_run_id, result, test_logger)
            
            if not schema_suggestions:
                result.error("Step 2 logic execution failed - no schema suggestions returned")
                return result
                
            # Validate critical numeric features are classified correctly
            success = self._validate_regression_feature_classification(schema_suggestions, result, test_logger)
            
            if success:
                result.success("Step 2 regression test passed - all numeric features correctly classified")
                test_logger.info("✅ Phase 2 Step 2 Regression Test: SUCCESS")
            else:
                result.error("Step 2 regression test failed - incorrect feature classification detected")
                test_logger.error("❌ Phase 2 Step 2 Regression Test: FAILED")
                
        except Exception as e:
            result.error(f"Test execution failed: {str(e)}")
            result.add_info("exception_traceback", traceback.format_exc())
            test_logger.error(f"❌ Phase 2 Step 2 Regression Test: EXCEPTION - {str(e)}")
            
        finally:
            # Cleanup test artifacts
            self._cleanup_test_files()
            
        return result
    
    def test_step_2_isolated_classification(self) -> TestResult:
        """
        Test Step 2 Schema Definition with classification dataset (loan approval).
        
        Critical focus: Ensure numeric features like applicant_age, credit_score, 
        annual_income are classified as numeric, not categorical.
        """
        result = TestResult("step_2_isolated_classification")
        
        try:
            # Create isolated test environment first to get test_run_id for logger
            test_run_id = self._setup_isolated_test_environment("classification", result)
            test_logger = get_test_logger(test_run_id)
            
            test_logger.info("🧪 Starting Phase 2 Step 2 Classification Test")
            result.add_info("dataset_type", "classification")
            result.add_info("test_timestamp", datetime.now().isoformat())
            
            # Generate loan approval test data and upload to GCS
            test_data = self._generate_classification_test_data()
            upload_success = self._upload_test_data_to_gcs(test_run_id, test_data, result, test_logger)
            
            if not upload_success:
                result.error("Failed to upload test data to GCS")
                return result
                
            # Execute Step 2 logic
            schema_suggestions = self._execute_step_2_logic(test_run_id, result, test_logger)
            
            if not schema_suggestions:
                result.error("Step 2 logic execution failed - no schema suggestions returned")
                return result
                
            # Validate critical numeric features are classified correctly
            success = self._validate_classification_feature_classification(schema_suggestions, result, test_logger)
            
            if success:
                result.success("Step 2 classification test passed - all numeric features correctly classified")
                test_logger.info("✅ Phase 2 Step 2 Classification Test: SUCCESS")
            else:
                result.error("Step 2 classification test failed - incorrect feature classification detected")
                test_logger.error("❌ Phase 2 Step 2 Classification Test: FAILED")
                
        except Exception as e:
            result.error(f"Test execution failed: {str(e)}")
            result.add_info("exception_traceback", traceback.format_exc())
            test_logger.error(f"❌ Phase 2 Step 2 Classification Test: EXCEPTION - {str(e)}")
            
        finally:
            # Cleanup test artifacts
            self._cleanup_test_files()
            
        return result
    
    def _setup_isolated_test_environment(self, task_type: str, result: TestResult) -> str:
        """Create isolated test environment for Step 2 testing."""
        test_run_id = f"phase2_step2_{task_type}_{int(time.time())}"
        result.add_info("test_run_id", test_run_id)
        
        # Add to cleanup list
        self.cleanup_files.append(test_run_id)
        
        return test_run_id
    
    def _generate_regression_test_data(self) -> pd.DataFrame:
        """Generate controlled regression test data (house prices structure)."""
        np.random.seed(42)  # Reproducible results
        n_samples = 100
        
        # Generate test data matching house_prices.csv structure
        data = {
            'square_feet': np.random.normal(2000, 600, n_samples).astype(int),
            'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
            'bathrooms': np.random.uniform(1.0, 4.5, n_samples).round(1),
            'garage_spaces': np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.5, 0.1]),
            'property_type': np.random.choice(['Single Family', 'Condo', 'Townhouse', 'Ranch'], 
                                            n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            'neighborhood_quality_score': np.random.choice(range(1, 11), n_samples),
            'price': np.random.normal(425000, 150000, n_samples).astype(int)
        }
        
        # Ensure reasonable bounds
        data['square_feet'] = np.clip(data['square_feet'], 800, 4000)
        data['price'] = np.clip(data['price'], 175000, 787000)
        
        return pd.DataFrame(data)
    
    def _generate_classification_test_data(self) -> pd.DataFrame:
        """Generate controlled classification test data (loan approval structure)."""
        np.random.seed(42)  # Reproducible results
        n_samples = 100
        
        # Generate test data matching loan_approval.csv structure
        data = {
            'applicant_age': np.random.randint(18, 65, n_samples),
            'annual_income': np.random.normal(75000, 35000, n_samples).astype(int),
            'credit_score': np.random.normal(650, 120, n_samples).astype(int),
            'employment_years': np.random.exponential(5, n_samples).round(1),
            'loan_amount': np.random.normal(250000, 150000, n_samples).astype(int),
            'debt_to_income_ratio': np.random.uniform(0.05, 0.4, n_samples).round(3),
            'education_level': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], 
                                              n_samples, p=[0.3, 0.4, 0.25, 0.05]),
            'property_type': np.random.choice(['Single Family', 'Condo', 'Townhouse', 'Multi-family'], 
                                            n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            'approved': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        }
        
        # Ensure reasonable bounds
        data['annual_income'] = np.clip(data['annual_income'], 25000, 285000)
        data['credit_score'] = np.clip(data['credit_score'], 391, 850)
        data['employment_years'] = np.clip(data['employment_years'], 0.0, 21.9)
        data['loan_amount'] = np.clip(data['loan_amount'], 50000, 834000)
        
        return pd.DataFrame(data)
    
    def _upload_test_data_to_gcs(self, test_run_id: str, test_data: pd.DataFrame, result: TestResult, test_logger) -> bool:
        """Upload test data to GCS as original_data.csv."""
        try:
            start_time = time.time()
            
            # Convert DataFrame to CSV bytes
            csv_buffer = BytesIO()
            test_data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            # Upload to GCS
            success = upload_run_file(test_run_id, constants.ORIGINAL_DATA_FILENAME, csv_buffer)
            
            upload_time = time.time() - start_time
            result.add_measurement("data_upload_time", upload_time, "seconds")
            result.add_info("test_data_rows", len(test_data))
            result.add_info("test_data_columns", list(test_data.columns))
            
            if success:
                test_logger.info(f"✅ Test data uploaded to GCS: {test_run_id}/{constants.ORIGINAL_DATA_FILENAME}")
            else:
                test_logger.error(f"❌ Failed to upload test data to GCS")
                
            return success
            
        except Exception as e:
            test_logger.error(f"❌ Exception during test data upload: {str(e)}")
            result.add_info("upload_exception", str(e))
            return False
    
    def _execute_step_2_logic(self, test_run_id: str, result: TestResult, test_logger) -> Optional[Dict[str, Dict[str, str]]]:
        """Execute Step 2 schema definition logic."""
        try:
            start_time = time.time()
            
            test_logger.info("🔍 Executing Step 2 schema definition logic")
            
            # Call the main Step 2 function
            schema_suggestions = suggest_initial_feature_schemas_from_gcs(test_run_id, PROJECT_BUCKET_NAME)
            
            execution_time = time.time() - start_time
            result.add_measurement("step2_execution_time", execution_time, "seconds")
            result.add_info("schema_suggestions_count", len(schema_suggestions) if schema_suggestions else 0)
            
            if schema_suggestions:
                test_logger.info(f"✅ Step 2 executed successfully, generated {len(schema_suggestions)} schema suggestions")
                result.add_info("schema_suggestions", schema_suggestions)
            else:
                test_logger.error("❌ Step 2 execution returned empty schema suggestions")
                
            return schema_suggestions
            
        except Exception as e:
            test_logger.error(f"❌ Step 2 execution failed: {str(e)}")
            result.add_info("execution_exception", str(e))
            result.add_info("execution_traceback", traceback.format_exc())
            return None
    
    def _validate_regression_feature_classification(self, schema_suggestions: Dict[str, Dict[str, str]], result: TestResult, test_logger) -> bool:
        """Validate regression dataset feature classification."""
        test_logger.info("🔍 Validating regression feature classification")
        
        # Define critical numeric features that should NOT be classified as categorical
        critical_numeric_features = {
            'square_feet': 'numeric-continuous',
            'bedrooms': 'numeric-discrete',
            'bathrooms': 'numeric-continuous',
            'garage_spaces': 'numeric-discrete',
            'neighborhood_quality_score': 'numeric-discrete'
        }
        
        # Define expected categorical features
        expected_categorical_features = {
            'property_type': 'categorical-nominal'
        }
        
        # Define expected numeric target
        expected_target = {
            'price': ['numeric-continuous', 'numeric-discrete']  # Either is acceptable for price
        }
        
        all_correct = True
        
        # Check critical numeric features
        for feature, expected_type in critical_numeric_features.items():
            if feature in schema_suggestions:
                actual_type = schema_suggestions[feature]['suggested_encoding_role']
                
                # For bedrooms and garage_spaces, accept either numeric-discrete or numeric-continuous
                if feature in ['bedrooms', 'garage_spaces']:
                    is_correct = actual_type in ['numeric-discrete', 'numeric-continuous']
                    expected_display = 'numeric-discrete OR numeric-continuous'
                else:
                    is_correct = actual_type == expected_type
                    expected_display = expected_type
                
                result.assert_true(is_correct, 
                                 f"Feature '{feature}' should be {expected_display}, got '{actual_type}'")
                
                if is_correct:
                    test_logger.info(f"✅ {feature}: {actual_type} (CORRECT)")
                else:
                    test_logger.error(f"❌ {feature}: {actual_type} (INCORRECT, expected {expected_display})")
                    all_correct = False
            else:
                result.assert_true(False, f"Critical feature '{feature}' missing from schema suggestions")
                test_logger.error(f"❌ {feature}: MISSING from schema suggestions")
                all_correct = False
        
        # Check categorical features
        for feature, expected_type in expected_categorical_features.items():
            if feature in schema_suggestions:
                actual_type = schema_suggestions[feature]['suggested_encoding_role']
                is_correct = actual_type == expected_type
                
                result.assert_equals(expected_type, actual_type, 
                                   f"Feature '{feature}' should be {expected_type}")
                
                if is_correct:
                    test_logger.info(f"✅ {feature}: {actual_type} (CORRECT)")
                else:
                    test_logger.error(f"❌ {feature}: {actual_type} (INCORRECT, expected {expected_type})")
                    all_correct = False
            else:
                result.assert_true(False, f"Expected categorical feature '{feature}' missing from schema suggestions")
                all_correct = False
        
        # Check target feature
        for feature, expected_types in expected_target.items():
            if feature in schema_suggestions:
                actual_type = schema_suggestions[feature]['suggested_encoding_role']
                is_correct = actual_type in expected_types
                
                result.assert_true(is_correct, 
                                 f"Target feature '{feature}' should be one of {expected_types}, got '{actual_type}'")
                
                if is_correct:
                    test_logger.info(f"✅ {feature}: {actual_type} (CORRECT)")
                else:
                    test_logger.error(f"❌ {feature}: {actual_type} (INCORRECT, expected one of {expected_types})")
                    all_correct = False
            else:
                result.assert_true(False, f"Target feature '{feature}' missing from schema suggestions")
                all_correct = False
        
        # Add summary information
        result.add_info("critical_numeric_features_tested", list(critical_numeric_features.keys()))
        result.add_info("categorical_features_tested", list(expected_categorical_features.keys()))
        result.add_info("target_features_tested", list(expected_target.keys()))
        
        return all_correct
    
    def _validate_classification_feature_classification(self, schema_suggestions: Dict[str, Dict[str, str]], result: TestResult, test_logger) -> bool:
        """Validate classification dataset feature classification."""
        test_logger.info("🔍 Validating classification feature classification")
        
        # Define critical numeric features that should NOT be classified as categorical
        critical_numeric_features = {
            'applicant_age': 'numeric-discrete',
            'annual_income': 'numeric-continuous',
            'credit_score': 'numeric-discrete',
            'employment_years': 'numeric-continuous',
            'loan_amount': 'numeric-continuous',
            'debt_to_income_ratio': 'numeric-continuous'
        }
        
        # Define expected categorical features
        expected_categorical_features = {
            'education_level': 'categorical-nominal',
            'property_type': 'categorical-nominal'
        }
        
        # Define expected target (binary classification)
        expected_target = {
            'approved': ['numeric-discrete', 'boolean', 'categorical-nominal']  # Multiple acceptable types for binary
        }
        
        all_correct = True
        
        # Check critical numeric features
        for feature, expected_type in critical_numeric_features.items():
            if feature in schema_suggestions:
                actual_type = schema_suggestions[feature]['suggested_encoding_role']
                
                # For age and credit_score, accept either numeric-discrete or numeric-continuous
                if feature in ['applicant_age', 'credit_score']:
                    is_correct = actual_type in ['numeric-discrete', 'numeric-continuous']
                    expected_display = 'numeric-discrete OR numeric-continuous'
                else:
                    is_correct = actual_type == expected_type
                    expected_display = expected_type
                
                result.assert_true(is_correct, 
                                 f"Feature '{feature}' should be {expected_display}, got '{actual_type}'")
                
                if is_correct:
                    test_logger.info(f"✅ {feature}: {actual_type} (CORRECT)")
                else:
                    test_logger.error(f"❌ {feature}: {actual_type} (INCORRECT, expected {expected_display})")
                    all_correct = False
            else:
                result.assert_true(False, f"Critical feature '{feature}' missing from schema suggestions")
                test_logger.error(f"❌ {feature}: MISSING from schema suggestions")
                all_correct = False
        
        # Check categorical features
        for feature, expected_type in expected_categorical_features.items():
            if feature in schema_suggestions:
                actual_type = schema_suggestions[feature]['suggested_encoding_role']
                is_correct = actual_type == expected_type
                
                result.assert_equals(expected_type, actual_type, 
                                   f"Feature '{feature}' should be {expected_type}")
                
                if is_correct:
                    test_logger.info(f"✅ {feature}: {actual_type} (CORRECT)")
                else:
                    test_logger.error(f"❌ {feature}: {actual_type} (INCORRECT, expected {expected_type})")
                    all_correct = False
            else:
                result.assert_true(False, f"Expected categorical feature '{feature}' missing from schema suggestions")
                all_correct = False
        
        # Check target feature
        for feature, expected_types in expected_target.items():
            if feature in schema_suggestions:
                actual_type = schema_suggestions[feature]['suggested_encoding_role']
                is_correct = actual_type in expected_types
                
                result.assert_true(is_correct, 
                                 f"Target feature '{feature}' should be one of {expected_types}, got '{actual_type}'")
                
                if is_correct:
                    test_logger.info(f"✅ {feature}: {actual_type} (CORRECT)")
                else:
                    test_logger.error(f"❌ {feature}: {actual_type} (INCORRECT, expected one of {expected_types})")
                    all_correct = False
            else:
                result.assert_true(False, f"Target feature '{feature}' missing from schema suggestions")
                all_correct = False
        
        # Add summary information
        result.add_info("critical_numeric_features_tested", list(critical_numeric_features.keys()))
        result.add_info("categorical_features_tested", list(expected_categorical_features.keys()))
        result.add_info("target_features_tested", list(expected_target.keys()))
        
        return all_correct
    
    def _cleanup_test_files(self):
        """Clean up test artifacts from GCS."""
        for test_run_id in self.cleanup_files:
            try:
                # Note: In a real implementation, you would delete GCS files here
                # For now, we'll just log the cleanup intent
                print(f"🧹 Cleaning up test artifacts for run: {test_run_id}")
            except Exception as e:
                print(f"⚠️  Failed to cleanup test run {test_run_id}: {str(e)}")


def run_phase_2_step_2_tests() -> Dict[str, TestResult]:
    """
    Run all Phase 2 Step 2 tests and return results.
    
    Returns:
        Dictionary mapping test names to TestResult objects
    """
    test_suite = Step2SchemaDefinitionTest()
    
    results = {}
    
    # Run regression test
    print("🚀 Running Phase 2 Step 2 Regression Test...")
    results["regression"] = test_suite.test_step_2_isolated_regression()
    
    # Run classification test
    print("🚀 Running Phase 2 Step 2 Classification Test...")
    results["classification"] = test_suite.test_step_2_isolated_classification()
    
    return results


def generate_test_report(test_results: Dict[str, TestResult]) -> Dict[str, Any]:
    """Generate comprehensive test report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        "test_suite": "Phase 2 Step 2 Schema Definition Tests",
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(test_results),
        "passed_tests": len([r for r in test_results.values() if r.status == "success"]),
        "failed_tests": len([r for r in test_results.values() if r.status == "error"]),
        "test_results": {name: result.to_dict() for name, result in test_results.items()},
        "summary": {
            "success_rate": len([r for r in test_results.values() if r.status == "success"]) / len(test_results),
            "total_assertions": sum(len(r.assertions) for r in test_results.values()),
            "passed_assertions": sum(len([a for a in r.assertions if a["passed"]]) for r in test_results.values()),
            "total_duration": sum(r.duration for r in test_results.values() if r.duration),
            "critical_failures": [
                name for name, result in test_results.items() 
                if result.status == "error"
            ]
        }
    }
    
    # Save report
    report_dir = Path("tests/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"phase_2_step_2_test_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📊 Test report saved: {report_path}")
    
    return report


def main():
    """Main test execution function."""
    print("=" * 80)
    print("Phase 2 Step-by-Step Isolation Testing")
    print("Step 2: Schema Definition Tests")
    print("=" * 80)
    print()
    
    # Run all tests
    test_results = run_phase_2_step_2_tests()
    
    # Generate report
    report = generate_test_report(test_results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {report['total_tests']}")
    print(f"Passed: {report['passed_tests']}")
    print(f"Failed: {report['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"Total Duration: {report['summary']['total_duration']:.2f} seconds")
    
    if report['summary']['critical_failures']:
        print(f"Critical Failures: {', '.join(report['summary']['critical_failures'])}")
    
    print("\nDetailed Results:")
    for test_name, result in test_results.items():
        status_emoji = "✅" if result.status == "success" else "❌"
        print(f"{status_emoji} {test_name}: {result.status.upper()} ({result.duration:.2f}s)")
        if result.errors:
            for error in result.errors:
                print(f"   - {error}")
    
    print("\n" + "=" * 80)
    
    # Exit with appropriate code
    exit_code = 0 if report['failed_tests'] == 0 else 1
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 