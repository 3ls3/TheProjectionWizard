# Testing Framework & Standards

## Testing Infrastructure Overview

### Test Directory Structure
```
tests/
├── fixtures/                    # Test data generation
│   ├── fixture_generator.py     # Comprehensive test data creation
│   └── test_data_*.csv          # Generated test datasets
├── integration/                 # End-to-end integration tests
│   ├── test_orchestrator.py     # Pipeline integration tests
│   └── test_api_integration.py  # API integration tests
├── unit/                        # Component unit tests
│   └── stage_tests/             # Per-pipeline-step tests
│       └── base_stage_test.py   # Base testing framework
├── reports/                     # Test execution reports
│   └── test_report_*.json       # Structured test results
└── data/                        # Test-specific data
    └── test_runs/               # Isolated test pipeline runs
```

## ⚠️ CRITICAL: Test Environment Parity Requirements (Added 2025-06-11)

### Mandatory Test Setup Standards
**LESSON LEARNED FROM BUG FIX**: Tests must EXACTLY match production environment to avoid false failures.

#### The Complete Test Run Initialization Pattern
```python
class PipelineTestBase(BaseStageTest):
    def create_complete_test_run(self, test_run_id: str, test_data: pd.DataFrame, task_type: str) -> bool:
        """
        Creates a complete test run matching production API behavior.
        
        CRITICAL: This must match exactly what /api/upload does.
        Missing files will cause cryptic "processing failed" errors.
        
        Returns:
            bool: True if all files uploaded successfully, False otherwise
        """
        try:
            # Step 1: Upload CSV data (what most tests already do)
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
```

#### Why This Matters - Real Bug Example
```python
# ❌ WRONG: Incomplete test setup (what we were doing before)
def broken_test_setup(test_run_id, test_data):
    # Only uploads CSV - WILL CAUSE DOWNSTREAM FAILURES
    csv_buffer = BytesIO()
    test_data.to_csv(csv_buffer, index=False)
    upload_run_file(test_run_id, "original_data.csv", csv_buffer)
    
    # Missing status.json and metadata.json
    # Result: "Schema confirmation failed" error because:
    # - confirm_feature_schemas_gcs() tries to download status.json
    # - File doesn't exist -> download fails -> function returns False
    # - API returns "Schema confirmation failed" (misleading error message)

# ✅ CORRECT: Complete test setup (what we do now)
def proper_test_setup(test_run_id, test_data, task_type):
    return self.create_complete_test_run(test_run_id, test_data, task_type)
    # Result: All pipeline steps work correctly because all required files exist
```

#### Verification Checklist for Every Pipeline Test
Before testing any pipeline functionality, verify:
- [ ] `runs/{test_run_id}/original_data.csv` exists in GCS
- [ ] `runs/{test_run_id}/status.json` exists in GCS (CRITICAL)
- [ ] `runs/{test_run_id}/metadata.json` exists in GCS (CRITICAL)
- [ ] Test setup matches `/api/upload` endpoint behavior exactly

#### Common Error Patterns and Solutions
1. **"Schema confirmation failed"** → Missing `status.json` file
2. **"Metadata not found"** → Missing `metadata.json` file  
3. **"Run not found"** → Incomplete file structure in GCS
4. **"Processing failed"** → Check all 3 required files exist

#### Test Environment Anti-Patterns to Avoid
```python
# ❌ DON'T: Mock GCS operations in integration tests
with patch('api.utils.gcs_utils.download_run_file'):
    # This hides real integration issues

# ❌ DON'T: Skip file structure setup 
test_data.to_csv("test.csv")  # Local file only
# Pipeline expects files in GCS with specific structure

# ❌ DON'T: Assume error messages point to real bug
if "Schema confirmation failed":
    # Check test setup first, not backend logic
```

## Base Testing Framework

### BaseStageTest Class
```python
from tests.unit.stage_tests.base_stage_test import BaseStageTest, TestResult

class MyTest(BaseStageTest):
    def setUp(self):
        """Initialize test environment"""
        super().setUp()
        self.test_run_id = f"test_{int(time.time())}"
        self.test_data_dir = Path("tests/data/test_runs") / self.test_run_id
        
    def test_specific_functionality(self):
        """Test specific functionality with structured reporting"""
        result = TestResult("test_name")
        try:
            # Test implementation
            expected = "expected_value"
            actual = function_under_test()
            result.assert_equals(expected, actual, "Description of test")
            result.success("Test passed successfully")
        except Exception as e:
            result.error(f"Test failed: {str(e)}")
        
        self.report_test_result(result)
        return result
```

### TestResult Class Usage
```python
# Initialize test result
result = TestResult("test_prediction_accuracy")

# Add context information
result.add_info("model_path", str(model_path))
result.add_info("test_data_size", len(test_data))

# Perform assertions
result.assert_equals(expected_value, actual_value, "Prediction value check")
result.assert_in_range(actual_value, min_val, max_val, "Value range validation")

# Add detailed measurements
result.add_measurement("prediction_time", 0.045)
result.add_measurement("accuracy_score", 0.892)

# Report success or failure
if all_tests_passed:
    result.success("All prediction tests passed")
else:
    result.error("Prediction accuracy below threshold")
```

## Test Data Generation Framework

### TestFixtureGenerator Usage
```python
from tests.fixtures.fixture_generator import TestFixtureGenerator

# Create test data generator
generator = TestFixtureGenerator()

# Generate controlled test datasets
# For regression testing
regression_data = generator.generate_house_data(
    num_samples=100,
    property_type_distribution={'Single Family': 0.6, 'Condo': 0.4},
    price_range=(200000, 500000),
    seed=42  # For reproducible tests
)

# For classification testing  
classification_data = generator.generate_loan_approval_data(
    num_samples=100,
    approval_rate=0.7,
    income_range=(25000, 285000),
    seed=42  # For reproducible tests
)

# Generate edge cases
edge_cases = generator.generate_edge_cases([
    'minimum_values',
    'maximum_values', 
    'boundary_conditions',
    'invalid_data'
])

# Save test datasets
regression_data_path = Path("tests/data") / "test_house_data.csv"
regression_data.to_csv(regression_data_path, index=False)

classification_data_path = Path("tests/data") / "test_loan_approval_data.csv"  
classification_data.to_csv(classification_data_path, index=False)
```

## Testing Patterns & Standards

### 1. Atomic Testing Pattern
```python
def test_single_component_atomic():
    """Test one specific component in isolation"""
    # Arrange - Set up test conditions
    test_input = create_test_input()
    expected_output = define_expected_output()
    
    # Act - Execute the component
    actual_output = component_under_test(test_input)
    
    # Assert - Verify results
    assert actual_output == expected_output
    
    # Report - Structured logging
    self.log_test_result("component_test", True, {
        'input': test_input,
        'expected': expected_output,
        'actual': actual_output
    })
```

### 2. Integration Testing Pattern
```python
def test_end_to_end_pipeline():
    """Test complete pipeline flow"""
    # Setup isolated test environment
    test_run_id = f"integration_test_{int(time.time())}"
    
    # Execute pipeline steps sequentially
    step_results = {}
    for step_num in range(1, 8):
        step_result = self.execute_pipeline_step(step_num, test_run_id)
        step_results[f"step_{step_num}"] = step_result
        
        # Validate step output before proceeding
        self.validate_step_output(step_num, step_result)
    
    # Validate end-to-end results
    final_result = self.validate_complete_pipeline(test_run_id)
    
    # Cleanup test artifacts
    self.cleanup_test_run(test_run_id)
    
    return final_result
```

### 3. API Testing Pattern
```python
def test_api_endpoint_consistency():
    """Test API endpoint behavior and consistency"""
    # Prepare test request
    test_payload = {
        "square_feet": 2500,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "property_type": "Single Family"
    }
    
    # Test multiple endpoints with same input
    endpoints = ["/api/predict", "/api/predict/single"]
    results = {}
    
    for endpoint in endpoints:
        response = self.call_api_endpoint(endpoint, test_payload)
        results[endpoint] = {
            'status_code': response.status_code,
            'prediction': response.json().get('prediction'),
            'response_time': response.elapsed.total_seconds()
        }
    
    # Validate consistency between endpoints
    self.assert_prediction_consistency(results)
    
    return results
```

## Test Environment Management

### Test Run Isolation
```python
def create_isolated_test_environment(test_name):
    """Create isolated test environment"""
    test_run_id = f"{test_name}_{int(time.time())}"
    test_dir = Path("tests/data/test_runs") / test_run_id
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy required test data
    shutil.copy("tests/fixtures/test_house_data.csv", 
                test_dir / "original_data.csv")
    
    # Set up test-specific GCS prefix
    test_gcs_prefix = f"test_runs/{test_run_id}/"
    
    return {
        'run_id': test_run_id,
        'test_dir': test_dir,
        'gcs_prefix': test_gcs_prefix
    }

def cleanup_test_environment(test_env):
    """Clean up test environment"""
    # Remove local test files
    if test_env['test_dir'].exists():
        shutil.rmtree(test_env['test_dir'])
    
    # Clean up GCS test artifacts (if needed)
    # gcs_cleanup(test_env['gcs_prefix'])
```

### Test Data Validation
```python
def validate_test_data_quality(data_path):
    """Validate test data meets quality requirements"""
    df = pd.read_csv(data_path)
    
    quality_checks = {
        'no_missing_values': df.isnull().sum().sum() == 0,
        'correct_columns': set(df.columns) == set(EXPECTED_COLUMNS),
        'valid_data_types': all(df.dtypes == EXPECTED_DTYPES),
        'value_ranges': all(df[col].between(min_val, max_val).all() 
                          for col, (min_val, max_val) in VALUE_RANGES.items())
    }
    
    return all(quality_checks.values()), quality_checks
```

## Assertion Utilities

### Custom Assertions for ML Testing
```python
def assert_prediction_in_range(self, prediction, min_val, max_val, message=""):
    """Assert prediction value is within expected range"""
    assert min_val <= prediction <= max_val, \
        f"{message}: Prediction {prediction} not in range [{min_val}, {max_val}]"

def assert_regression_prediction(self, prediction, message=""):
    """Assert regression prediction is in reasonable range"""
    self.assert_prediction_in_range(
        prediction, 
        REGRESSION_PREDICTION_RANGE[0], 
        REGRESSION_PREDICTION_RANGE[1], 
        f"Regression {message}"
    )

def assert_classification_prediction(self, prediction, message=""):
    """Assert classification prediction is valid probability"""
    self.assert_prediction_in_range(
        prediction, 
        CLASSIFICATION_PREDICTION_RANGE[0], 
        CLASSIFICATION_PREDICTION_RANGE[1], 
        f"Classification {message}"
    )

def assert_classification_decision(self, prediction, expected_class=None, message=""):
    """Assert classification decision is reasonable"""
    predicted_class = 1 if prediction >= CLASSIFICATION_THRESHOLD else 0
    
    if expected_class is not None:
        assert predicted_class == expected_class, \
            f"{message}: Predicted class {predicted_class}, expected {expected_class}"
    
    # Ensure prediction is not too close to threshold (indicates model confidence)
    distance_from_threshold = abs(prediction - CLASSIFICATION_THRESHOLD)
    assert distance_from_threshold >= 0.1, \
        f"{message}: Prediction {prediction} too close to threshold {CLASSIFICATION_THRESHOLD}"

def assert_model_predictions_consistent(self, predictions, tolerance=None, task_type="regression"):
    """Assert model predictions are consistent across runs"""
    if tolerance is None:
        tolerance = REGRESSION_PREDICTION_TOLERANCE if task_type == "regression" else CLASSIFICATION_PREDICTION_TOLERANCE
    
    if task_type == "regression":
        mean_pred = np.mean(predictions)
        max_deviation = max(abs(p - mean_pred) for p in predictions)
        relative_deviation = max_deviation / mean_pred if mean_pred != 0 else max_deviation
        
        assert relative_deviation <= tolerance, \
            f"Regression predictions inconsistent: {relative_deviation:.3%} deviation > {tolerance:.1%}"
    else:
        # For classification, check consistency of predicted classes
        predicted_classes = [1 if p >= CLASSIFICATION_THRESHOLD else 0 for p in predictions]
        class_consistency = len(set(predicted_classes)) / len(predicted_classes)
        
        # Allow some variance but not complete inconsistency
        assert class_consistency <= 0.5 or len(set(predicted_classes)) == 1, \
            f"Classification predictions too inconsistent: {class_consistency:.1%} class variance"

def assert_api_response_format(self, response, expected_schema):
    """Assert API response matches expected schema"""
    response_data = response.json()
    
    for field, field_type in expected_schema.items():
        assert field in response_data, f"Missing field: {field}"
        assert isinstance(response_data[field], field_type), \
            f"Field {field} has wrong type: {type(response_data[field])}"

def assert_task_type_detection(self, detected_task_type, dataset_type):
    """Assert correct task type detection for dataset"""
    expected_task_types = {
        "regression": "regression",
        "classification": "classification"
    }
    
    expected = expected_task_types.get(dataset_type)
    assert detected_task_type == expected, \
        f"Task type detection failed: detected '{detected_task_type}', expected '{expected}' for {dataset_type}"
```

## Test Execution Patterns

### Phase-Based Testing
```python
def execute_phase_1_infrastructure_tests():
    """Phase 1: Test basic infrastructure"""
    tests = [
        'test_gcs_connectivity',
        'test_api_server_startup', 
        'test_pipeline_orchestrator_init'
    ]
    return self.run_test_suite(tests)

def execute_phase_2_component_tests():
    """Phase 2: Test individual components"""
    tests = [
        'test_step_1_ingest',
        'test_step_2_schema',
        'test_step_4_encoding',
        'test_step_7_prediction'
    ]
    return self.run_test_suite(tests)
```

### Error Handling Patterns
```python
def test_with_comprehensive_error_handling():
    """Test with structured error handling and reporting"""
    result = TestResult("comprehensive_test")
    
    try:
        # Test execution
        output = execute_test_logic()
        
        # Validate output
        if self.validate_output(output):
            result.success("Test completed successfully")
        else:
            result.error("Output validation failed")
            
    except ValidationError as e:
        result.error(f"Validation error: {str(e)}")
        result.add_info("validation_details", e.details)
        
    except Exception as e:
        result.error(f"Unexpected error: {str(e)}")
        result.add_info("error_type", type(e).__name__)
        result.add_info("error_traceback", traceback.format_exc())
    
    finally:
        # Always cleanup, regardless of test outcome
        self.cleanup_test_resources()
    
    return result
```

## Test Reporting & Analysis

### Structured Test Results
```python
def generate_test_report(test_results):
    """Generate comprehensive test report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': len(test_results),
        'passed': sum(1 for r in test_results if r.status == 'success'),
        'failed': sum(1 for r in test_results if r.status == 'error'),
        'test_details': [r.to_dict() for r in test_results],
        'summary': {
            'success_rate': sum(1 for r in test_results if r.status == 'success') / len(test_results),
            'critical_failures': [r for r in test_results if r.status == 'error' and r.critical],
            'performance_metrics': extract_performance_metrics(test_results)
        }
    }
    
    # Save report
    report_path = Path("tests/reports") / f"test_report_{int(time.time())}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report
```

## Configuration & Constants

### Test Configuration
```python
# Test constants
TEST_DATA_SIZE = 100
TEST_TIMEOUT = 300  # 5 minutes

# Regression test configuration
REGRESSION_PREDICTION_RANGE = (200000, 800000)  # $200k - $800k
REGRESSION_PREDICTION_TOLERANCE = 0.05  # 5% tolerance

# Classification test configuration  
CLASSIFICATION_PREDICTION_RANGE = (0, 1)  # Binary classification
CLASSIFICATION_PREDICTION_TOLERANCE = 0.1  # 10% tolerance
CLASSIFICATION_THRESHOLD = 0.5  # Classification threshold

# Test datasets
REGRESSION_DATASET = "data/fixtures/house_prices.csv"
CLASSIFICATION_DATASET = "data/fixtures/loan_approval.csv"

# Test data paths
TEST_DATA_DIR = Path("tests/data")
TEST_FIXTURES_DIR = Path("tests/fixtures")
TEST_REPORTS_DIR = Path("tests/reports")

# API test configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30  # seconds
```

This framework provides comprehensive testing capabilities with structured reporting, error handling, and consistent patterns for all test implementations. 