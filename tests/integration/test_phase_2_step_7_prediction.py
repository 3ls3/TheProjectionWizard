#!/usr/bin/env python3

"""
Phase 2 Step 7: Prediction System Testing

CRITICAL PURPOSE: Identify root cause of prediction magnitude bugs
- Regression: Getting ~$150M instead of ~$425k
- Classification: Need valid probabilities (0.0-1.0)

This test isolates Step 7 prediction functionality to determine if the bug is in:
1. column_mapper.encode_user_input_gcs() function
2. API endpoint routing differences
3. Model artifact loading
4. Scaler application during prediction

SUCCESS CRITERIA:
- Regression predictions: $200k-$800k range (mean ~$425k)
- Classification predictions: 0.0-1.0 probabilities
- Consistent results between direct function calls and API endpoints
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
from datetime import datetime
import requests
import traceback

# Add project root to path
sys.path.append('.')

# Core imports
from common import constants
from common import logger
from api.utils.gcs_utils import upload_run_file, download_run_file, PROJECT_BUCKET_NAME
# Base test functionality implemented directly
from tests.fixtures.fixture_generator import TestFixtureGenerator

# Pipeline imports
from pipeline.step_7_predict import column_mapper, predict_runner
from pipeline.step_4_prep import prep_runner
from pipeline.step_5_automl import automl_runner


class TestResult:
    """
    Enhanced TestResult class matching workflow documentation requirements.
    Supports measurements, assertions, and structured reporting.
    """
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.status = "pending"
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
        
    def add_info(self, key: str, value):
        """Add contextual information."""
        self.info[key] = value
        
    def assert_equals(self, expected, actual, message: str = ""):
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
            
    def assert_in_range(self, value, min_val, max_val, message: str = ""):
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
            
    def success(self, message: str):
        """Mark test as successful with message."""
        self.status = "success"
        self.message = message
        self.messages.append(f"SUCCESS: {message}")
        self._finalize()
        
    def error(self, message: str):
        """Mark test as failed with error message."""
        self.status = "error"
        self.message = message
        self.errors.append(message)
        self.messages.append(f"ERROR: {message}")
        self._finalize()
        
    def _finalize(self):
        """Finalize test timing."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
    def to_dict(self):
        """Convert to dictionary for reporting."""
        if self.end_time is None:
            self._finalize()
            
        return {
            "test_name": self.test_name,
            "status": self.status,
            "duration": self.duration,
            "measurements": self.measurements,
            "info": self.info,
            "assertions": self.assertions,
            "errors": self.errors,
            "messages": self.messages
        }

class PhaseSevenPredictionTest:
    """Test Step 7 prediction system in isolation"""
    
    def __init__(self):
        # Initialize without BaseStageTest since we don't have test_run_id yet
        self.logger = logger.get_logger("test_step7_prediction", "test")
        self.test_results = []
        self.fixture_generator = TestFixtureGenerator()
        
        # Test constants
        self.REGRESSION_PREDICTION_RANGE = (200000, 800000)  # $200k-$800k
        self.CLASSIFICATION_PREDICTION_RANGE = (0.0, 1.0)    # Valid probabilities
        self.API_BASE_URL = "http://localhost:8000"
        
    def create_complete_test_run(self, test_run_id: str, test_data: pd.DataFrame, task_type: str) -> bool:
        """
        Create complete test run matching production API behavior.
        CRITICAL: Must match exactly what /api/upload does.
        """
        try:
            self.logger.info(f"🔧 Creating complete test run: {test_run_id}")
            
            # Step 1: Upload CSV data
            csv_buffer = BytesIO()
            test_data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            success = upload_run_file(test_run_id, constants.ORIGINAL_DATA_FILENAME, csv_buffer)
            if not success:
                self.logger.error("Failed to upload CSV data")
                return False
            
            # Step 2: Create status.json (REQUIRED - pipeline steps expect this)
            upload_ts = datetime.now().isoformat()
            status_data = {
                "run_id": test_run_id,
                "stage": "upload",
                "status": "completed",
                "message": "File uploaded successfully to GCS",
                "progress_pct": 5,
                "last_updated": upload_ts,
                "stages": {
                    "upload": {"status": "completed", "message": "CSV file uploaded to GCS"},
                    "target_suggestion": {"status": "pending"},
                    "feature_suggestion": {"status": "pending"},
                    "pipeline_execution": {"status": "pending"}
                }
            }
            status_json = json.dumps(status_data, indent=2).encode('utf-8')
            status_io = BytesIO(status_json)
            success = upload_run_file(test_run_id, constants.STATUS_FILENAME, status_io)
            if not success:
                self.logger.error("Failed to upload status.json")
                return False
            
            # Step 3: Create metadata.json (REQUIRED - contains run configuration)
            metadata = {
                "run_id": test_run_id,
                "timestamp": upload_ts,
                "original_filename": f"test_{task_type}_data.csv",
                "initial_rows": len(test_data),
                "initial_cols": len(test_data.columns),
                "initial_dtypes": {col: str(dtype) for col, dtype in test_data.dtypes.items()},
                "storage": {
                    "type": "gcs",
                    "bucket": PROJECT_BUCKET_NAME,
                    "csv_path": f"runs/{test_run_id}/original_data.csv",
                    "metadata_path": f"runs/{test_run_id}/metadata.json",
                    "status_path": f"runs/{test_run_id}/status.json"
                }
            }
            metadata_json = json.dumps(metadata, indent=2).encode('utf-8')
            metadata_io = BytesIO(metadata_json)
            success = upload_run_file(test_run_id, constants.METADATA_FILENAME, metadata_io)
            if not success:
                self.logger.error("Failed to upload metadata.json")
                return False
            
            self.logger.info(f"✅ Complete test run initialized: {test_run_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create complete test run: {str(e)}")
            return False

    def simulate_target_confirmation(self, test_run_id: str, target_column: str, task_type: str, ml_type: str) -> bool:
        """Simulate target column confirmation"""
        try:
            metadata_bytes = download_run_file(test_run_id, constants.METADATA_FILENAME)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            # Add target info to metadata (what /api/confirm-target does)
            metadata['target_info'] = {
                'name': target_column,
                'task_type': task_type,
                'ml_type': ml_type,
                'confirmed_at': datetime.now().isoformat()
            }
            
            # Upload updated metadata
            metadata_json = json.dumps(metadata, indent=2).encode('utf-8')
            metadata_io = BytesIO(metadata_json)
            return upload_run_file(test_run_id, constants.METADATA_FILENAME, metadata_io)
            
        except Exception as e:
            self.logger.error(f"Target confirmation failed: {str(e)}")
            return False

    def simulate_feature_confirmation(self, test_run_id: str, test_data: pd.DataFrame) -> bool:
        """Simulate feature schema confirmation with correct feature types"""
        try:
            metadata_bytes = download_run_file(test_run_id, constants.METADATA_FILENAME)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            # Create feature schemas (based on what Step 4 test confirmed works)
            target_column = metadata['target_info']['name']
            
            # Use the EXACT same feature schemas that worked in Step 4 test
            if target_column == 'sale_price':  # Regression (house_prices.csv uses 'sale_price')
                feature_schemas = {
                    'square_feet': {'dtype': 'int64', 'encoding_role': 'numeric-continuous', 'source': 'user_confirmed'},
                    'bedrooms': {'dtype': 'int64', 'encoding_role': 'numeric-discrete', 'source': 'system_defaulted'},
                    'bathrooms': {'dtype': 'float64', 'encoding_role': 'numeric-discrete', 'source': 'system_defaulted'},
                    'house_age_years': {'dtype': 'int64', 'encoding_role': 'numeric-discrete', 'source': 'system_defaulted'},
                    'garage_spaces': {'dtype': 'int64', 'encoding_role': 'numeric-discrete', 'source': 'system_defaulted'},
                    'neighborhood_quality_score': {'dtype': 'int64', 'encoding_role': 'numeric-discrete', 'source': 'system_defaulted'},
                    'school_district_rating': {'dtype': 'float64', 'encoding_role': 'numeric-continuous', 'source': 'system_defaulted'},
                    'distance_to_city_miles': {'dtype': 'float64', 'encoding_role': 'numeric-continuous', 'source': 'system_defaulted'},
                    'property_type': {'dtype': 'object', 'encoding_role': 'categorical-nominal', 'source': 'system_defaulted'},
                    'sale_price': {'dtype': 'float64', 'encoding_role': 'target', 'source': 'system_defaulted'}
                }
            else:  # Classification
                feature_schemas = {
                    'applicant_age': {'dtype': 'int64', 'encoding_role': 'numeric-discrete', 'source': 'system_defaulted'},
                    'annual_income': {'dtype': 'float64', 'encoding_role': 'numeric-continuous', 'source': 'user_confirmed'},
                    'credit_score': {'dtype': 'int64', 'encoding_role': 'numeric-discrete', 'source': 'system_defaulted'},
                    'employment_years': {'dtype': 'float64', 'encoding_role': 'numeric-continuous', 'source': 'user_confirmed'},
                    'loan_amount': {'dtype': 'float64', 'encoding_role': 'numeric-continuous', 'source': 'user_confirmed'},
                    'debt_to_income_ratio': {'dtype': 'float64', 'encoding_role': 'numeric-continuous', 'source': 'system_defaulted'},
                    'education_level': {'dtype': 'object', 'encoding_role': 'categorical-nominal', 'source': 'system_defaulted'},
                    'property_type': {'dtype': 'object', 'encoding_role': 'categorical-nominal', 'source': 'system_defaulted'},
                    'approved': {'dtype': 'int64', 'encoding_role': 'target', 'source': 'system_defaulted'}
                }
            
            # Add feature schemas to metadata
            metadata['feature_schemas'] = feature_schemas
            metadata['feature_schemas_confirmed_at'] = datetime.now().isoformat()
            
            # Upload updated metadata
            metadata_json = json.dumps(metadata, indent=2).encode('utf-8')
            metadata_io = BytesIO(metadata_json)
            return upload_run_file(test_run_id, constants.METADATA_FILENAME, metadata_io)
            
        except Exception as e:
            self.logger.error(f"Feature confirmation failed: {str(e)}")
            return False

    def run_complete_pipeline_to_step_5(self, test_run_id: str) -> bool:
        """Run pipeline through Step 5 to create model artifacts for Step 7 testing"""
        try:
            self.logger.info(f"🔄 Running pipeline Steps 1-5 for: {test_run_id}")
            
            # Step 4: Data Preparation
            prep_result = prep_runner.run_preparation_stage_gcs(test_run_id)
            if not prep_result:
                self.logger.error("Step 4 data preparation failed")
                return False
            self.logger.info("✅ Step 4 data preparation completed")
            
            # Step 5: AutoML Training
            automl_result = automl_runner.run_automl_stage_gcs(test_run_id)
            if not automl_result:
                self.logger.error("Step 5 AutoML training failed")
                return False
            self.logger.info("✅ Step 5 AutoML training completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return False

    def test_step_7_regression_prediction(self) -> TestResult:
        """Test Step 7 regression prediction system"""
        result = TestResult("step7_regression_prediction")
        test_run_id = f"step7_pred_regression_{int(time.time())}"
        
        try:
            self.logger.info("🧪 Starting Step 7 Regression Prediction Test")
            result.add_info("test_run_id", test_run_id)
            
            # Load regression test data
            regression_data = pd.read_csv("data/fixtures/house_prices.csv")
            result.add_info("test_data_shape", regression_data.shape)
            
            # Setup complete test run
            if not self.create_complete_test_run(test_run_id, regression_data, "regression"):
                result.error("Failed to create complete test run")
                return result
            
            # Simulate confirmations
            if not self.simulate_target_confirmation(test_run_id, "sale_price", "regression", "numeric_continuous"):
                result.error("Failed to simulate target confirmation")
                return result
            
            if not self.simulate_feature_confirmation(test_run_id, regression_data):
                result.error("Failed to simulate feature confirmation")
                return result
            
            # Run pipeline to create model artifacts
            if not self.run_complete_pipeline_to_step_5(test_run_id):
                result.error("Failed to run pipeline to Step 5")
                return result
            
            # Now test Step 7 prediction functionality
            # COMPLETE input with ALL features the model expects
            test_input = {
                # Basic numeric features
                "square_feet": 2500,
                "bedrooms": 3,
                "bathrooms": 2.5,
                "house_age_years": 15,
                "garage_spaces": 2,
                "neighborhood_quality_score": 8,
                "school_district_rating": 7,
                "distance_to_city_miles": 5,
                # One-hot encoded property_type (only one should be 1)
                "property_type_Condo": 0,
                "property_type_Ranch": 0,
                "property_type_Single Family": 1,
                "property_type_Townhouse": 0
            }
            result.add_info("test_input", test_input)
            
            # Test 1: Direct column_mapper function call
            self.logger.info("🔍 Testing column_mapper.encode_user_input_gcs() directly")
            
            # Get target column from metadata
            metadata_bytes = download_run_file(test_run_id, constants.METADATA_FILENAME)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            target_column = metadata['target_info']['name']
            
            encoded_input, issues = column_mapper.encode_user_input_gcs(
                test_input, test_run_id, regression_data, target_column
            )
            if encoded_input is None:
                result.error(f"column_mapper.encode_user_input_gcs() returned None. Issues: {issues}")
                return result
            
            result.add_info("encoded_input_shape", encoded_input.shape)
            result.add_info("encoded_input_columns", list(encoded_input.columns))
            self.logger.info(f"✅ Input encoded successfully: {encoded_input.shape}")
            
            # Test 2: Direct prediction runner call
            self.logger.info("🔍 Testing predict_runner.make_single_prediction_gcs() directly")
            
            prediction_result = predict_runner.make_single_prediction_gcs(test_run_id, test_input, return_details=True)
            if prediction_result is None:
                result.error("predict_runner.make_single_prediction_gcs() returned None")
                return result
            
            prediction_value = prediction_result.get('prediction')
            if prediction_value is None:
                result.error("No prediction value in result")
                return result
            
            result.add_info("direct_prediction", prediction_value)
            result.add_measurement("direct_prediction_value", prediction_value)
            
            # Validate prediction magnitude
            if self.REGRESSION_PREDICTION_RANGE[0] <= prediction_value <= self.REGRESSION_PREDICTION_RANGE[1]:
                self.logger.info(f"✅ Direct prediction in valid range: ${prediction_value:,.0f}")
                result.add_info("direct_prediction_valid", True)
            else:
                self.logger.error(f"❌ Direct prediction out of range: ${prediction_value:,.0f}")
                result.add_info("direct_prediction_valid", False)
                result.error(f"Direct prediction ${prediction_value:,.0f} not in range ${self.REGRESSION_PREDICTION_RANGE[0]:,}-${self.REGRESSION_PREDICTION_RANGE[1]:,}")
            
            # Test 3: API endpoint comparison (if server is running)
            api_results = self._test_api_endpoints_regression(test_input, result)
            
            # Final assessment
            if result.add_info("direct_prediction_valid", True):
                result.success(f"Step 7 regression prediction test passed - ${prediction_value:,.0f}")
            else:
                result.error("Step 7 regression prediction failed validation")
            
        except Exception as e:
            result.error(f"Step 7 regression test failed: {str(e)}")
            result.add_info("error_traceback", traceback.format_exc())
        
        finally:
            self._cleanup_test_artifacts(test_run_id)
        
        return result

    def test_step_7_classification_prediction(self) -> TestResult:
        """Test Step 7 classification prediction system"""
        result = TestResult("step7_classification_prediction")
        test_run_id = f"step7_pred_classification_{int(time.time())}"
        
        try:
            self.logger.info("🧪 Starting Step 7 Classification Prediction Test")
            result.add_info("test_run_id", test_run_id)
            
            # Load classification test data
            classification_data = pd.read_csv("data/fixtures/loan_approval.csv")
            result.add_info("test_data_shape", classification_data.shape)
            
            # Setup complete test run
            if not self.create_complete_test_run(test_run_id, classification_data, "classification"):
                result.error("Failed to create complete test run")
                return result
            
            # Simulate confirmations
            if not self.simulate_target_confirmation(test_run_id, "approved", "classification", "binary_01"):
                result.error("Failed to simulate target confirmation")
                return result
            
            if not self.simulate_feature_confirmation(test_run_id, classification_data):
                result.error("Failed to simulate feature confirmation")
                return result
            
            # Run pipeline to create model artifacts
            if not self.run_complete_pipeline_to_step_5(test_run_id):
                result.error("Failed to run pipeline to Step 5")
                return result
            
            # Now test Step 7 prediction functionality
            # COMPLETE input with ALL features the model expects
            test_input = {
                # Basic numeric features
                "applicant_age": 35,
                "annual_income": 75000,
                "credit_score": 720,
                "employment_years": 5.5,
                "loan_amount": 250000,
                "debt_to_income_ratio": 0.25,
                # One-hot encoded education_level (only one should be 1)
                "education_level_Bachelors": 1,
                "education_level_High School": 0,
                "education_level_Masters": 0,
                "education_level_PhD": 0,
                # One-hot encoded property_type (only one should be 1)
                "property_type_Condo": 0,
                "property_type_Multi-family": 0,
                "property_type_Single Family": 1,
                "property_type_Townhouse": 0
            }
            result.add_info("test_input", test_input)
            
            # Test 1: Direct column_mapper function call
            self.logger.info("🔍 Testing column_mapper.encode_user_input_gcs() for classification")
            
            # Get target column from metadata
            metadata_bytes = download_run_file(test_run_id, constants.METADATA_FILENAME)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            target_column = metadata['target_info']['name']
            
            encoded_input, issues = column_mapper.encode_user_input_gcs(
                test_input, test_run_id, classification_data, target_column
            )
            if encoded_input is None:
                result.error(f"column_mapper.encode_user_input_gcs() returned None. Issues: {issues}")
                return result
            
            result.add_info("encoded_input_shape", encoded_input.shape)
            result.add_info("encoded_input_columns", list(encoded_input.columns))
            self.logger.info(f"✅ Input encoded successfully: {encoded_input.shape}")
            
            # Test 2: Direct prediction runner call
            self.logger.info("🔍 Testing predict_runner.make_single_prediction_gcs() for classification")
            
            prediction_result = predict_runner.make_single_prediction_gcs(test_run_id, test_input, return_details=True)
            if prediction_result is None:
                result.error("predict_runner.make_single_prediction_gcs() returned None")
                return result
            
            prediction_value = prediction_result.get('prediction')
            if prediction_value is None:
                result.error("No prediction value in result")
                return result
            
            result.add_info("direct_prediction", prediction_value)
            result.add_measurement("direct_prediction_probability", prediction_value)
            
            # Validate prediction is valid probability
            if self.CLASSIFICATION_PREDICTION_RANGE[0] <= prediction_value <= self.CLASSIFICATION_PREDICTION_RANGE[1]:
                self.logger.info(f"✅ Direct prediction is valid probability: {prediction_value:.3f}")
                result.add_info("direct_prediction_valid", True)
                
                # Check predicted class
                predicted_class = 1 if prediction_value >= 0.5 else 0
                result.add_info("predicted_class", predicted_class)
                result.add_info("prediction_confidence", abs(prediction_value - 0.5))
                
            else:
                self.logger.error(f"❌ Direct prediction not valid probability: {prediction_value}")
                result.add_info("direct_prediction_valid", False)
                result.error(f"Direct prediction {prediction_value} not in range [0.0, 1.0]")
            
            # Test 3: API endpoint comparison (if server is running)
            api_results = self._test_api_endpoints_classification(test_input, result)
            
            # Final assessment
            if result.add_info("direct_prediction_valid", True):
                result.success(f"Step 7 classification prediction test passed - {prediction_value:.3f}")
            else:
                result.error("Step 7 classification prediction failed validation")
            
        except Exception as e:
            result.error(f"Step 7 classification test failed: {str(e)}")
            result.add_info("error_traceback", traceback.format_exc())
        
        finally:
            self._cleanup_test_artifacts(test_run_id)
        
        return result

    def _test_api_endpoints_regression(self, test_input: dict, result: TestResult) -> dict:
        """Test API endpoints for regression predictions"""
        api_results = {}
        
        try:
            self.logger.info("🌐 Testing API endpoints for regression")
            
            endpoints = [
                "/api/predict",
                "/api/predict/single"
            ]
            
            for endpoint in endpoints:
                try:
                    self.logger.info(f"📡 Testing {endpoint}")
                    response = requests.post(
                        f"{self.API_BASE_URL}{endpoint}",
                        json=test_input,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        prediction = response_data.get('prediction')
                        
                        api_results[endpoint] = {
                            'status_code': response.status_code,
                            'prediction': prediction,
                            'response_time': response.elapsed.total_seconds(),
                            'valid_range': self.REGRESSION_PREDICTION_RANGE[0] <= prediction <= self.REGRESSION_PREDICTION_RANGE[1] if prediction else False
                        }
                        
                        self.logger.info(f"✅ {endpoint}: ${prediction:,.0f} ({'VALID' if api_results[endpoint]['valid_range'] else 'INVALID'})")
                        
                    else:
                        api_results[endpoint] = {
                            'status_code': response.status_code,
                            'error': response.text,
                            'prediction': None,
                            'valid_range': False
                        }
                        self.logger.error(f"❌ {endpoint}: HTTP {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    api_results[endpoint] = {
                        'error': str(e),
                        'prediction': None,
                        'valid_range': False
                    }
                    self.logger.warning(f"⚠️ {endpoint}: Connection failed - {str(e)}")
            
            result.add_info("api_endpoint_results", api_results)
            
        except Exception as e:
            self.logger.warning(f"API endpoint testing failed: {str(e)}")
            result.add_info("api_test_error", str(e))
        
        return api_results

    def _test_api_endpoints_classification(self, test_input: dict, result: TestResult) -> dict:
        """Test API endpoints for classification predictions"""
        api_results = {}
        
        try:
            self.logger.info("🌐 Testing API endpoints for classification")
            
            endpoints = [
                "/api/predict",
                "/api/predict/single"
            ]
            
            for endpoint in endpoints:
                try:
                    self.logger.info(f"📡 Testing {endpoint}")
                    response = requests.post(
                        f"{self.API_BASE_URL}{endpoint}",
                        json=test_input,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        prediction = response_data.get('prediction')
                        
                        api_results[endpoint] = {
                            'status_code': response.status_code,
                            'prediction': prediction,
                            'response_time': response.elapsed.total_seconds(),
                            'valid_probability': 0.0 <= prediction <= 1.0 if prediction is not None else False
                        }
                        
                        self.logger.info(f"✅ {endpoint}: {prediction:.3f} ({'VALID' if api_results[endpoint]['valid_probability'] else 'INVALID'})")
                        
                    else:
                        api_results[endpoint] = {
                            'status_code': response.status_code,
                            'error': response.text,
                            'prediction': None,
                            'valid_probability': False
                        }
                        self.logger.error(f"❌ {endpoint}: HTTP {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    api_results[endpoint] = {
                        'error': str(e),
                        'prediction': None,
                        'valid_probability': False
                    }
                    self.logger.warning(f"⚠️ {endpoint}: Connection failed - {str(e)}")
            
            result.add_info("api_endpoint_results", api_results)
            
        except Exception as e:
            self.logger.warning(f"API endpoint testing failed: {str(e)}")
            result.add_info("api_test_error", str(e))
        
        return api_results

    def _cleanup_test_artifacts(self, test_run_id: str):
        """Clean up test artifacts"""
        try:
            self.logger.info(f"🧹 Cleaning up test artifacts for run: {test_run_id}")
            # Note: Could implement GCS cleanup here if needed
            # For now, test artifacts remain for debugging
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {str(e)}")

    def run_all_tests(self):
        """Run all Step 7 prediction tests"""
        self.logger.info("================================================================================")
        self.logger.info("Phase 2 Step 7 Prediction System Testing")
        self.logger.info("Testing prediction functionality and magnitude validation")
        self.logger.info("================================================================================")
        
        # Run regression test
        regression_result = self.test_step_7_regression_prediction()
        self.test_results.append(regression_result)
        
        # Run classification test
        classification_result = self.test_step_7_classification_prediction()
        self.test_results.append(classification_result)
        
        # Generate summary
        self._generate_test_summary()

    def _generate_test_summary(self):
        """Generate comprehensive test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.status == 'success')
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        self.logger.info("================================================================================")
        self.logger.info("STEP 7 PREDICTION SYSTEM TEST SUMMARY")
        self.logger.info("================================================================================")
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {failed_tests}")
        self.logger.info(f"Success Rate: {success_rate:.1f}%")
        
        for result in self.test_results:
            status_icon = "✅ SUCCESS" if result.status == 'success' else "❌ FAILED"
            self.logger.info(f"{status_icon} {result.test_name}: {result.message}")
        
        self.logger.info("")


def main():
    """Main test execution"""
    tester = PhaseSevenPredictionTest()
    tester.run_all_tests()


if __name__ == "__main__":
    main() 