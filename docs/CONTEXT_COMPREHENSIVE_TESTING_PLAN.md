# Comprehensive Pipeline Testing Plan

## Testing Plan Overview

This document outlines a systematic 5-phase testing approach to identify and fix the remaining prediction bugs in the frontend-backend integration. Each phase builds upon the previous, providing atomic bug isolation capabilities.

## ⚠️ CRITICAL: Test Environment Parity Requirements

### Mandatory Test Setup Guidelines (Added 2025-06-11)
**LESSON LEARNED**: Tests must EXACTLY match production environment to avoid false failures.

#### Required File Structure for All Pipeline Tests
Every test run MUST create the complete file structure that production creates:

```python
# ✅ CORRECT: Complete run initialization (matches production API)
def setup_test_run(test_run_id, test_data, task_type):
    # Step 1: Upload CSV data
    csv_buffer = BytesIO()
    test_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    upload_run_file(test_run_id, constants.ORIGINAL_DATA_FILENAME, csv_buffer)
    
    # Step 2: Create status.json (REQUIRED - pipeline steps fail without this)
    status_data = {
        "run_id": test_run_id,
        "stage": "upload",
        "status": "completed", 
        "message": "File uploaded successfully to GCS",
        "progress_pct": 5,
        "last_updated": datetime.now().isoformat(),
        "stages": {
            "upload": {"status": "completed", "message": "CSV file uploaded to GCS"},
            "target_suggestion": {"status": "pending"},
            "feature_suggestion": {"status": "pending"},
            "pipeline_execution": {"status": "pending"}
        }
    }
    upload_run_file(test_run_id, constants.STATUS_FILENAME, status_io)
    
    # Step 3: Create metadata.json (REQUIRED - contains run configuration)
    metadata = {
        "run_id": test_run_id,
        "timestamp": datetime.now().isoformat(),
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
    upload_run_file(test_run_id, constants.METADATA_FILENAME, metadata_io)
```

#### Common Testing Mistake to Avoid
```python
# ❌ WRONG: Only uploading CSV data (will cause cryptic failures)
upload_run_file(test_run_id, "original_data.csv", csv_buffer)
# This causes "Schema confirmation failed" and other mysterious errors
```

#### Why This Matters
- **File Dependencies**: Pipeline steps expect specific files to exist (especially `status.json`)
- **Error Propagation**: Missing files cause confusing downstream errors 
- **False Debugging**: Tests that don't match production lead to incorrect root cause analysis
- **Production Confidence**: Only tests that mirror production give reliable results

#### Verification Checklist
Before any pipeline test, verify these files exist in GCS:
- [ ] `runs/{test_run_id}/original_data.csv`
- [ ] `runs/{test_run_id}/status.json` 
- [ ] `runs/{test_run_id}/metadata.json`

## Phase 1: Infrastructure Validation 🔧

**Purpose**: Verify basic system components are operational
**Dependencies**: None

### Phase 1 Tests
1. **GCS Connectivity Test**
   - Verify GCS bucket access and permissions
   - Test artifact upload/download operations
   - Validate run directory structure

2. **API Server Startup Test**
   - Confirm FastAPI server starts correctly
   - Test basic health endpoints
   - Verify all route registrations

3. **Pipeline Orchestrator Initialization**
   - Test orchestrator can load and initialize
   - Verify step registration and discovery
   - Check configuration loading

4. **Model Loading Verification**
   - Confirm trained model can be loaded from GCS
   - Test model artifact integrity
   - Verify metadata consistency

### Expected Outcomes
- ✅ All infrastructure components operational
- ✅ GCS artifacts accessible and valid
- ✅ API server responds to basic requests
- ✅ Model loading successful

### Critical Success Criteria
- GCS connectivity: 100% success rate
- API response time: < 2 seconds
- Model loading time: < 5 seconds

## Phase 2: Step-by-Step Isolation Testing 🧪

**Purpose**: Test each pipeline step in complete isolation
**Dependencies**: Phase 1 success

### Phase 2 Tests

#### Step 1 Test: Data Ingest
```python
def test_step_1_isolated():
    # Input: Raw CSV file
    # Expected: Valid original_data.csv in GCS
    # Validation: File format, column presence, data integrity
```

#### Step 2 Test: Schema Definition
```python
def test_step_2_isolated_regression():
    # Input: House prices dataset (regression)
    # Expected: Reasonable feature suggestions (not perfect)
    # Validation: Critical features not misclassified as categorical
    # NOTE: Some continuous features may be suggested as discrete - this is acceptable
    # CRITICAL: Test that user can override suggestions via /api/confirm-features

def test_step_2_isolated_classification():
    # Input: Loan approval dataset (classification)  
    # Expected: Reasonable feature suggestions (not perfect)
    # Validation: Critical features not misclassified as categorical
    # NOTE: Some continuous features may be suggested as discrete - this is acceptable
    # CRITICAL: Test that user can override suggestions via /api/confirm-features

def test_step_2_user_override():
    # Input: Automatic suggestions + corrected user input
    # Expected: User corrections properly applied via API
    # Validation: /api/confirm-features endpoint works correctly
    # CRITICAL: Ensures user can fix any automatic suggestion errors
```

#### Step 3 Test: Data Validation  
```python
def test_step_3_isolated():
    # Input: Test data + feature definitions
    # Expected: Validation report with no critical errors
    # Validation: Data quality metrics, completeness checks
```

#### Step 4 Test: Data Preparation
```python
def test_step_4_isolated():
    # Input: Raw data + feature definitions
    # Expected: Properly encoded features + StandardScaler artifacts
    # Validation: Feature count (12 not 31), scaler presence
    # CRITICAL: Verify numeric features are scaled, not one-hot encoded
```

#### Step 5 Test: AutoML Training
```python
def test_step_5_isolated():
    # Input: Prepared data from Step 4
    # Expected: Trained model with correct feature count
    # Validation: Model accepts 12 features, reasonable performance
```

#### Step 6 Test: Model Explanation
```python
def test_step_6_isolated():
    # Input: Trained model + test data
    # Expected: SHAP explanations and feature importance
    # Validation: Explanation format, feature influence values
```

#### Step 7 Test: Prediction System
```python
def test_step_7_isolated_regression():
    # Input: User data + trained regression model
    # Expected: Predictions in ~$425k range (not ~$150M)
    # Validation: Prediction magnitude, confidence intervals
    # CRITICAL: Test column_mapper.encode_user_input_gcs()

def test_step_7_isolated_classification():
    # Input: User data + trained classification model
    # Expected: Predictions as probabilities (0.0-1.0 range)
    # Validation: Probability values, class predictions
    # CRITICAL: Test column_mapper.encode_user_input_gcs()
```

### Expected Outcomes
- ✅ Each step produces valid output in isolation
- ✅ Feature type classification is correct
- ✅ StandardScaler is created and applied
- ✅ Model training produces 12-feature model
- ✅ Predictions are in correct magnitude

### Critical Success Criteria
- Step 2: All numeric features classified correctly (both regression & classification)
- Step 4: Correct feature encoding (12 for regression, 8 for classification)
- Step 7: Valid predictions (regression: $200k-$800k, classification: 0.0-1.0)

## Phase 3: Data Flow Integration Testing 🔄

**Purpose**: Test data consistency between pipeline steps
**Duration**: ~1.5 hours
**Dependencies**: Phase 2 success

### Phase 3 Tests

#### Feature Schema Consistency
```python
def test_feature_schema_consistency():
    # Verify feature definitions match between:
    # - Step 2 output (feature_definitions.json)
    # - Step 4 input (encoding logic)
    # - Step 7 input (column mapping)
```

#### Encoding Consistency  
```python
def test_encoding_consistency():
    # Verify encoding applied in Step 4 matches Step 7 expectations:
    # - Same categorical mappings
    # - Same StandardScaler configuration
    # - Same feature order
```

#### Model Input Validation
```python
def test_model_input_validation():
    # Verify Step 7 provides exactly what model expects:
    # - Correct number of features (12)
    # - Proper feature names/order
    # - Appropriate data types and scaling
```

#### End-to-End Data Flow
```python
def test_end_to_end_data_flow():
    # Trace single data sample through entire pipeline:
    # Raw Input → Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 7
    # Validate data transformations at each step
```

### Expected Outcomes
- ✅ Feature schemas consistent across all steps
- ✅ Encoding transformations match between training and prediction
- ✅ Model receives correctly formatted input
- ✅ End-to-end data flow maintains consistency

### Critical Success Criteria
- Schema consistency: 100% match between steps
- Feature count consistency: Always 12 features
- Scaling consistency: Same StandardScaler applied

## Phase 4: API Endpoint Deep Testing 🌐

**Purpose**: Test exact frontend interaction patterns
**Duration**: ~1 hour  
**Dependencies**: Phase 3 success

### Phase 4 Tests

#### Endpoint Behavior Comparison
```python
def test_endpoint_behavior_comparison():
    # Test same input against all prediction endpoints:
    # - /api/predict (known working)
    # - /api/predict/single (suspected broken)
    # - /api/predict/explain (for completeness)
    # Validate identical results
```

#### Request/Response Format Validation
```python
def test_request_response_formats():
    # Verify exact request/response formats match frontend expectations:
    # - Input parameter names and types
    # - Response structure and data types
    # - Error handling and status codes
```

#### Column Mapping Verification
```python
def test_column_mapping_verification():
    # Deep test of column_mapper.py functions:
    # - encode_user_input_gcs() with known inputs
    # - generate_feature_slider_config() output format
    # - StandardScaler application verification
```

#### Frontend Simulation
```python
def test_frontend_simulation():
    # Simulate exact frontend requests:
    # - Use identical payload structure
    # - Test with realistic user input values
    # - Verify response processing
```

### Expected Outcomes
- ✅ All prediction endpoints return identical results
- ✅ Request/response formats match frontend expectations
- ✅ Column mapping functions work correctly
- ✅ Frontend simulation produces correct predictions

### Critical Success Criteria
- Endpoint consistency: <1% prediction variance (regression) or consistent class predictions (classification)
- Response format: 100% schema compliance for both task types
- Frontend simulation: Correct predictions (regression: ~$425k, classification: valid probabilities)

## Phase 5: Frontend Simulation Testing 🖥️

**Purpose**: Replicate complete user workflow
**Duration**: ~45 minutes
**Dependencies**: Phase 4 success

### Phase 5 Tests

#### Complete User Workflow
```python
def test_complete_user_workflow():
    # Replicate exact frontend user journey:
    # 1. Load prediction schema
    # 2. Submit user input
    # 3. Receive prediction
    # 4. Request explanation (optional)
    # Validate each step
```

#### Cross-Browser/Platform Testing
```python
def test_cross_platform_compatibility():
    # Test API calls from different contexts:
    # - Direct HTTP requests
    # - JavaScript fetch() calls
    # - Streamlit widget interactions
```

#### Error Scenario Testing
```python
def test_error_scenarios():
    # Test frontend error handling:
    # - Invalid input values
    # - Server timeout scenarios
    # - Malformed responses
```

#### Performance Under Load
```python
def test_performance_under_load():
    # Test system performance with:
    # - Multiple concurrent requests
    # - Large batch predictions
    # - Sustained load patterns
```

### Expected Outcomes
- ✅ Complete user workflow produces correct results
- ✅ Cross-platform compatibility verified
- ✅ Error scenarios handled gracefully
- ✅ Performance meets requirements

### Critical Success Criteria
- User workflow: 100% success rate
- Response time: <3 seconds per prediction
- Error handling: Graceful degradation

## Implementation Strategy

### Task Distribution
Each phase can be implemented as separate testing tasks:
- **Task 1**: Implement Phase 1 infrastructure tests
- **Task 2**: Implement Phase 2 component isolation tests  
- **Task 3**: Implement Phase 3 integration tests
- **Task 4**: Implement Phase 4 API endpoint tests
- **Task 5**: Implement Phase 5 frontend simulation tests

### Priority Order
1. **Phase 2** (highest priority) - Most likely to reveal core bugs
2. **Phase 4** (high priority) - Direct frontend integration issues
3. **Phase 1** (medium priority) - Foundation for other tests
4. **Phase 3** (medium priority) - Validates data flow consistency
5. **Phase 5** (lower priority) - Final validation and performance

### Bug Isolation Strategy
- **Run phases sequentially** - Each phase depends on previous success
- **Stop at first failure** - Fix critical bugs before proceeding
- **Document all findings** - Maintain detailed bug reports
- **Regression testing** - Re-run previous phases after fixes

## Expected Bug Discovery Areas

### Primary Suspects (Based on Analysis)
1. **Step 7 column_mapper.py** - Most likely source of prediction errors
2. **API endpoint routing** - Different code paths between endpoints
3. **StandardScaler application** - Scaling consistency issues

### Secondary Suspects  
1. **Step 4 encoding logic** - Feature preparation inconsistencies
2. **GCS artifact loading** - Model/scaler loading issues
3. **Frontend request formatting** - Input parameter mismatches

### Debugging Tools
- **Atomic testing** - Isolate exact failure points
- **Data flow tracing** - Track transformations step-by-step
- **Comparison testing** - Working vs broken endpoint analysis
- **Performance profiling** - Identify bottlenecks and errors

This plan provides systematic, comprehensive coverage for identifying and fixing the remaining prediction bugs while maintaining code quality and preventing regression. 