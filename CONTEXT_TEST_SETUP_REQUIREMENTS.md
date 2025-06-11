# CONTEXT: Test Setup Requirements

## CRITICAL PRINCIPLE: Tests Must Match Pipeline Reality

**Primary Rule**: Tests must validate what the pipeline **actually creates**, not what we assume it creates.

## Step 4 Data Preparation - Lessons Learned

### ❌ Wrong Test Assumptions (Caused Initial Failures)
1. **Metadata Location**: Looking for `encoders_scalers_info` at root level
   - **Reality**: Stored under `metadata['prep_info']['encoders_scalers_info']`

2. **Column Mapping File**: Expecting `column_mapping.json` in Step 4
   - **Reality**: Created in Step 5 AutoML, not Step 4

3. **Bundled Scalers**: Looking for single scaler files
   - **Reality**: Individual `{feature}_scaler.joblib` files (better design)

4. **Generic File Names**: Expecting standard names
   - **Reality**: Feature-specific names like `square_feet_scaler.joblib`

### ✅ Correct Test Setup Pattern

#### 1. Investigate First, Test Second
```python
# ALWAYS check what pipeline actually creates
metadata_bytes = download_run_file(run_id, 'metadata.json')
metadata = json.loads(metadata_bytes.decode('utf-8'))
print("Available keys:", list(metadata.keys()))

# Check nested structures
prep_info = metadata.get('prep_info', {})
print("prep_info keys:", list(prep_info.keys()))
```

#### 2. Validate Actual Structure
```python
# Step 4 stores prep information under prep_info
prep_info = metadata.get('prep_info', {})
if 'encoders_scalers_info' in prep_info:
    encoders_info = prep_info['encoders_scalers_info']
    # Validate what actually exists
```

#### 3. Test What Each Step Actually Creates

**Step 4 Creates:**
- Individual scaler files: `{feature}_scaler.joblib`
- Individual encoder files: `{feature}_onehot.joblib` 
- Metadata under `prep_info` key
- Cleaned data CSV
- Profiling report HTML

**Step 4 Does NOT Create:**
- `column_mapping.json` (that's Step 5)
- Bundled scaler files
- Root-level encoder metadata

## General Test Setup Guidelines

### 1. Environment Parity Requirements
Tests must create the **exact same file structure** as production:
```python
# REQUIRED for all tests
upload_file_to_gcs(run_id, "original_data.csv", csv_data)
upload_json_to_gcs(run_id, "status.json", status_data)
upload_json_to_gcs(run_id, "metadata.json", metadata_data)
```

### 2. Test Validation Pattern
```python
def validate_step_artifacts(run_id, expected_files, metadata_fields):
    # 1. Check files exist
    for file_name in expected_files:
        assert file_exists_in_gcs(run_id, file_name)
    
    # 2. Check metadata in correct location
    metadata = download_and_parse_metadata(run_id)
    step_info = metadata.get('{step}_info', {})  # e.g., 'prep_info'
    
    # 3. Validate what actually exists
    for field in metadata_fields:
        assert field in step_info, f"Missing {field} in {step}_info"
```

### 3. Debugging Failed Tests
When tests fail:
1. **First**: Check what files the pipeline actually created
2. **Second**: Check metadata structure and nesting
3. **Third**: Update test expectations to match reality
4. **Never**: Change pipeline to match wrong test expectations

### 4. Test File Naming Convention
- Individual tests: `test_phase_2_step_{N}_{description}.py`
- Test runs: `step{N}_{type}_{task}_{timestamp}`
- Metadata keys: `{step}_info` (e.g., `prep_info`, `automl_info`)

## Documentation Requirements

Every test file should include:
1. **Purpose**: What pipeline functionality is being tested
2. **Expected Artifacts**: List of files/metadata the step creates
3. **Validation Logic**: How success/failure is determined
4. **Cleanup**: How test artifacts are removed

## Key Success Metrics

**Step 4 Data Preparation:**
- Data shape expansion (features → encoded features)
- Individual scaler/encoder files created
- Metadata stored under `prep_info`
- Cleaned data CSV uploaded
- No column mapping file (that's Step 5)

**General Pipeline:**
- Status progression: pending → in-progress → completed
- Proper error handling and logging
- GCS file structure matches production exactly
- Test environment parity with production environment

**CRITICAL: Test Environment Parity Requirements**

This document defines how test files should be structured to properly validate pipeline functionality. Tests must validate what the pipeline **actually creates**, not what we assume it creates.

## Core Principle: Test What Actually Exists

### ❌ Wrong Approach
- Assume file names/structure without checking implementation
- Test for bundled files when pipeline creates individual files  
- Look for Step 5 artifacts in Step 4 tests
- Expect specific metadata field names without verifying

### ✅ Correct Approach
- **Examine pipeline implementation** before writing validation
- **Check actual GCS artifacts** created by each step
- **Validate what exists**, not what you think should exist
- **Match test expectations to implementation reality**

## Step-by-Step Test Setup Guidelines

### Step 1: Understand What Each Pipeline Step Creates

| Pipeline Step | Creates | Does NOT Create |
|---------------|---------|-----------------|
| Step 4 (Prep) | Individual `{feature}_scaler.joblib` files<br/>One-hot encoder info in metadata<br/>`cleaned_data.csv`<br/>Profiling reports | `column_mapping.json` (Step 5)<br/>Bundled scaler files<br/>Model artifacts |
| Step 5 (AutoML) | `column_mapping.json`<br/>Model files<br/>Training logs | Prediction endpoints<br/>User input validation |
| Step 7 (Predict) | Prediction results<br/>Confidence scores | Training artifacts |

### Step 2: Validation Pattern for Each Step

#### For Step 4 (Data Preparation)
```python
# ✅ CORRECT - Check what Step 4 actually creates
def validate_step_4_artifacts(test_run_id):
    # 1. Check metadata for encoders_scalers_info
    metadata = download_metadata(test_run_id)
    encoders_info = metadata.get('encoders_scalers_info', {})
    
    # 2. Validate individual scaler files exist
    for encoder_name, details in encoders_info.items():
        if details.get('type') == 'StandardScaler':
            gcs_path = details.get('gcs_path')  # e.g., "models/square_feet_scaler.joblib"
            assert download_run_file(test_run_id, gcs_path) is not None
    
    # 3. Check for cleaned data
    assert download_run_file(test_run_id, 'cleaned_data.csv') is not None
    
    # ❌ DON'T check for column_mapping.json (created in Step 5)
    # ❌ DON'T look for bundled "StandardScaler.pkl"
```

#### For Step 5 (AutoML Training)
```python
# ✅ CORRECT - Check what Step 5 actually creates  
def validate_step_5_artifacts(test_run_id):
    # 1. Check for column_mapping.json (created here, not Step 4)
    column_mapping = download_run_file(test_run_id, 'column_mapping.json')
    assert column_mapping is not None
    
    # 2. Check for model files
    # (implementation-specific model artifacts)
```

### Step 3: Test Environment Setup Requirements

#### Complete Test Run Initialization
```python
def setup_complete_test_run(test_run_id, test_data, task_type):
    """CRITICAL: Tests must create EXACT same file structure as production."""
    
    # 1. Upload original data
    upload_run_file(test_run_id, constants.DATA_FILENAME, data_csv_bytes)
    
    # 2. Create status.json (REQUIRED - production creates this)
    status = {
        "run_id": test_run_id,
        "stage": "step_1_ingest", 
        "status": "completed",
        "created_at": datetime.now().isoformat()
    }
    upload_run_file(test_run_id, constants.STATUS_FILENAME, json.dumps(status))
    
    # 3. Create metadata.json (REQUIRED - production creates this)
    metadata = {
        "run_id": test_run_id,
        "upload_timestamp": datetime.now().isoformat(),
        "original_shape": {"rows": len(test_data), "columns": len(test_data.columns)},
        "columns": list(test_data.columns)
    }
    upload_run_file(test_run_id, constants.METADATA_FILENAME, json.dumps(metadata))
    
    # 4. Simulate confirmations (target and features)
    simulate_target_confirmation(test_run_id, target_column, task_type, ml_type)
    simulate_feature_confirmation(test_run_id, test_data)
```

### Step 4: Validation Anti-Patterns to Avoid

#### ❌ Wrong: Hardcoded File Expectations
```python
# DON'T assume file names without checking implementation
scaler_patterns = ["scalers/StandardScaler.pkl", "scaler.pkl"]
for pattern in scaler_patterns:
    assert download_run_file(test_run_id, pattern)  # Will fail!
```

#### ✅ Right: Dynamic Validation Based on Metadata
```python
# Check metadata to see what files were actually created
metadata = download_metadata(test_run_id)
encoders_info = metadata.get('encoders_scalers_info', {})
for encoder_name, details in encoders_info.items():
    gcs_path = details.get('gcs_path')
    if gcs_path:  # Only check files that should exist
        assert download_run_file(test_run_id, gcs_path)
```

#### ❌ Wrong: Cross-Step Artifact Expectations
```python
# DON'T check for Step 5 artifacts in Step 4 tests
def test_step_4():
    # This will fail - column_mapping.json is created in Step 5!
    assert download_run_file(test_run_id, 'column_mapping.json')
```

#### ✅ Right: Step-Specific Validation
```python
def test_step_4():
    # Only check what Step 4 actually creates
    assert download_run_file(test_run_id, 'cleaned_data.csv')
    metadata = download_metadata(test_run_id)
    assert 'encoders_scalers_info' in metadata  # Step 4 creates this
    
def test_step_5():
    # Check for Step 5 artifacts
    assert download_run_file(test_run_id, 'column_mapping.json')  # Created here
```

### Step 5: Key Lessons from Step 4 Bug Fix

#### The Problem
- Tests were looking for `column_mapping.json` (created in Step 5)
- Tests expected bundled `StandardScaler.pkl` (Step 4 creates individual files)
- Tests looked for wrong metadata fields

#### The Solution
- **Check implementation first**: What does Step 4 actually create?
- **Match validation to reality**: Individual scaler files, not bundled
- **Validate correct artifacts**: `encoders_scalers_info` in metadata, not `column_mapping.json`

#### Critical Success Factors
1. **Implementation-First Validation**: Always check what the code creates before writing tests
2. **GCS Artifact Inspection**: Use `download_run_file()` to verify actual file structure
3. **Metadata-Driven Testing**: Use metadata to discover what files should exist
4. **Step Boundaries**: Don't cross-validate artifacts between pipeline steps

## Test Development Workflow

### Before Writing Any Test Validation:

1. **Read the pipeline step implementation**
   ```bash
   # Example: Before testing Step 4
   cat pipeline/step_4_prep/prep_runner.py
   cat pipeline/step_4_prep/encoding_logic.py
   ```

2. **Run the step manually and inspect GCS artifacts**
   ```python
   # Run step and check what files are created
   prep_runner.run_preparation_stage_gcs(test_run_id)
   # Then check metadata and GCS to see actual file structure
   ```

3. **Write validation that matches implementation**
   ```python
   # Validate only what you confirmed exists
   def validate_step_X_outputs():
       # Based on implementation inspection
   ```

## File Structure by Pipeline Step

### Step 4 (Data Preparation) Creates:
```
runs/{run_id}/
├── cleaned_data.csv                    ✅ Always created
├── metadata.json                       ✅ Updated with prep info  
├── status.json                         ✅ Updated to completed
├── {run_id}_profile.html              ✅ Profiling report
└── models/
    ├── {feature}_scaler.joblib         ✅ Individual feature scalers
    ├── {category}_onehot               ✅ One-hot encoder info (in metadata)
    └── {text_field}_tfidf_vectorizer.joblib  ✅ Text vectorizers (if any)
```

### Step 5 (AutoML Training) Creates:
```
runs/{run_id}/
├── column_mapping.json                 ✅ Created HERE, not Step 4
├── model_artifacts/                    ✅ Model files
└── training_logs/                      ✅ Training information
```

## Conclusion

**The root cause of test failures is almost always mismatch between test expectations and implementation reality.**

- ✅ **Before writing tests**: Check what the pipeline actually creates
- ✅ **Validate what exists**: Don't assume file names or structure  
- ✅ **Stay within step boundaries**: Don't cross-validate artifacts
- ✅ **Use metadata**: Let the pipeline tell you what files to expect
- ✅ **Implementation drives tests**: Not the other way around

This approach prevents "pipeline works perfectly but tests fail" scenarios and ensures tests validate actual functionality rather than assumptions.

## CRITICAL FIX: Step 7 Prediction Input Requirements

### ❌ **Root Cause of $150M Prediction Bug**

**Problem**: Test inputs were missing required one-hot encoded features, causing the model to fill missing features with 0 values, leading to massive prediction errors.

### ✅ **Solution Implemented**

**Before (Incomplete Input)**:
```python
test_input = {
    "square_feet": 2500,
    "property_type": "Single Family"  # Categorical string - WRONG!
}
```

**After (Complete Input)**:
```python
test_input = {
    "square_feet": 2500,
    "house_age_years": 15,           # All numeric features
    "school_district_rating": 7,
    "distance_to_city_miles": 5,
    # One-hot encoded categories (CRITICAL!)
    "property_type_Condo": 0,
    "property_type_Ranch": 0,
    "property_type_Single Family": 1,  # Only the selected type = 1
    "property_type_Townhouse": 0
}
```

### 📋 **Input Requirements for Predictions**

1. **Include ALL features the model was trained on**
2. **Use one-hot encoding for categorical features** (not string values)
3. **Only one category = 1, others = 0** for each categorical group
4. **Provide all numeric features** that were present during training

**Key Learning**: Always check `column_mapping.json` to see exact features the model expects! 