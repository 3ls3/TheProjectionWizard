# Task 6 Context & Key Learnings

## Critical Information for Future LLM Sessions

This document contains essential context, lessons learned, and debugging insights from implementing Task 6 (Data Preparation) of The Projection Wizard. This information will help future LLM sessions avoid common pitfalls and implement subsequent tasks more efficiently.

## Project Architecture Patterns Established

### File-Based Pipeline Communication (CRITICAL)

**Proven Pattern**: Each stage operates on file artifacts, never modifying previous stage outputs:

```
data/runs/{run_id}/
├── original_data.csv       # Stage 1 → Never modified by later stages
├── cleaned_data.csv        # Stage 6 → ML-ready data for Stage 7+
├── metadata.json          # Updated by each stage (atomic writes)
├── status.json           # Pipeline state tracking
├── {run_id}_profile.html # Data profiling report
└── model/                # Encoders/scalers for model deployment
    ├── target_column_label_encoder.joblib
    ├── scaler_feature.joblib
    └── text_feature_tfidf_vectorizer.joblib
```

**Key Insight**: This pattern enables:
- Stage re-runs without data corruption
- Parallel development of different stages
- Clear debugging when issues occur
- Easy rollback and comparison

### Pydantic Object Conversion Pattern (ESSENTIAL)

**Critical Understanding**: Pipeline stores Pydantic objects as JSON dictionaries, but business logic expects proper objects.

**Always Convert in Runners**:
```python
# CRITICAL: Load from file as dictionary
metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)

# CRITICAL: Convert to Pydantic objects
target_info_dict = metadata_dict.get('target_info')
if target_info_dict:
    target_info = schemas.TargetInfo(**target_info_dict)

feature_schemas_dict = metadata_dict.get('feature_schemas', {})
feature_schemas = {
    col_name: schemas.FeatureSchemaInfo(**schema_dict) 
    for col_name, schema_dict in feature_schemas_dict.items()
}
```

**Lesson Learned**: This pattern prevents runtime errors like `'dict' object has no attribute 'name'`.

### Schema Evolution Strategy

**Working Pattern**: Add new fields to existing schemas as Optional:

```python
# In common/schemas.py
class MetadataWithPrepInfo(MetadataWithFullSchema):
    prep_info: Optional[PrepInfo] = None  # ALWAYS Optional for backward compatibility

class PrepInfo(BaseModel):
    final_shape_after_prep: List[int]
    cleaning_steps_performed: List[str]
    encoders_scalers_info: Dict[str, Any]
    profiling_report_path: Optional[str] = None  # Optional for fallback cases
    prep_completed_at: str
```

## Data Processing Core Principles

### Schema-Driven Processing (FUNDAMENTAL)

**Key Insight**: All data operations should use confirmed feature schemas, not inferred types:

```python
# CORRECT: Schema-driven imputation
encoding_role = feature_schemas[column].encoding_role

if encoding_role in ['numeric-continuous', 'numeric-discrete']:
    # Median imputation for numeric
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)
elif encoding_role in ['categorical-nominal', 'categorical-ordinal']:
    # Mode imputation for categorical
    mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else "_UNKNOWN_"
    df[column].fillna(mode_value, inplace=True)

# WRONG: Type-based processing
if df[column].dtype == 'object':  # Don't do this
```

**Why This Matters**: User-confirmed schemas provide ML intent, not just data types.

### Encoding Role System (CRITICAL FOR AUTOML)

**Established Encoding Roles** (from `common/constants.py`):
```python
ENCODING_ROLES = [
    "numeric-continuous",    # StandardScaler
    "numeric-discrete",      # StandardScaler or keep as-is
    "categorical-nominal",   # OneHotEncoder
    "categorical-ordinal",   # OrdinalEncoder or label encoding
    "boolean",              # Direct conversion to 0/1
    "datetime",             # Extract temporal features
    "text",                 # TF-IDF vectorization
    "target"               # Special handling based on ML type
]
```

**Processing Strategy by Role**:
```python
# Numeric: StandardScaler + persistence
if encoding_role in ['numeric-continuous', 'numeric-discrete']:
    scaler = StandardScaler()
    df_encoded[columns] = scaler.fit_transform(df_encoded[columns])
    joblib.dump(scaler, model_dir / f"scaler_{safe_column_name}.joblib")

# Categorical Nominal: One-hot encoding
elif encoding_role == 'categorical-nominal':
    df_dummies = pd.get_dummies(df_encoded[column], prefix=column)
    df_encoded = pd.concat([df_encoded, df_dummies], axis=1)
    df_encoded.drop(columns=[column], inplace=True)

# Text: TF-IDF with feature limiting
elif encoding_role == 'text':
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df_encoded[column].astype(str))
    joblib.dump(vectorizer, model_dir / f"{safe_column_name}_tfidf_vectorizer.joblib")
```

### Target Variable Handling (ML-CRITICAL)

**Target ML Types** (from schema stage):
```python
# Binary classification with 0/1 values
if target_info.ml_type == 'binary_01':
    df_encoded[target_column] = df_encoded[target_column].astype(int)

# Binary/Multiclass with text labels
elif target_info.ml_type in ['binary_text_labels', 'multiclass_text_labels']:
    le = LabelEncoder()
    df_encoded[target_column] = le.fit_transform(df_encoded[target_column].astype(str))
    # CRITICAL: Save encoder for prediction reversal
    joblib.dump(le, model_dir / f"target_{target_column}_label_encoder.joblib")

# Regression - ensure numeric
elif target_info.ml_type == 'numeric_continuous':
    df_encoded[target_column] = pd.to_numeric(df_encoded[target_column], errors='coerce')
```

**Lesson Learned**: Target encoding is crucial for AutoML success. Always save label encoders for text targets.

## Critical Technical Patterns

### Encoder Persistence Strategy (DEPLOYMENT-CRITICAL)

**Why This Matters**: Models need these encoders for production predictions.

```python
# CRITICAL: Consistent encoder naming and metadata tracking
encoders_scalers_info = {}

# Example encoder persistence
scaler = StandardScaler()
df_encoded[columns] = scaler.fit_transform(df_encoded[columns])

# Save with descriptive name
encoder_name = f"scaler_{safe_column_name}"
encoder_path = model_dir / f"{encoder_name}.joblib"
joblib.dump(scaler, encoder_path)

# Track in metadata for model deployment
encoders_scalers_info[encoder_name] = {
    'type': 'StandardScaler',
    'file_path': str(encoder_path),
    'columns_affected': columns.tolist() if hasattr(columns, 'tolist') else [columns],
    'encoding_role': encoding_role,
    'original_column_name': column
}
```

### Atomic File Operations (DATA INTEGRITY)

**Critical Pattern**: Always use atomic writes for metadata files:

```python
# CORRECT: Atomic write prevents corruption
storage.write_json_atomic(run_dir / constants.METADATA_FILENAME, metadata_dict)
storage.write_json_atomic(run_dir / constants.STATUS_FILENAME, status_update)

# WRONG: Direct write can corrupt on failure
with open(metadata_path, 'w') as f:
    json.dump(metadata_dict, f)  # Don't do this
```

**Why This Matters**: Pipeline interruption during file writes can corrupt the entire run.

### Error Handling & Fallback Strategies

**Proven Pattern**: Always have fallback approaches for external dependencies:

```python
# Three-level fallback for ydata-profiling
def generate_profile_report_with_fallback(df, report_path, title):
    try:
        # Level 1: Full ydata-profiling with optimizations
        profile = ProfileReport(df, title=title, **optimized_config)
        profile.to_file(report_path)
        return True
    except Exception as e1:
        logger.warning(f"Full profiling failed: {e1}")
        try:
            # Level 2: Minimal ydata-profiling
            profile = ProfileReport(df, title=title, minimal=True)
            profile.to_file(report_path)
            return True
        except Exception as e2:
            logger.warning(f"Minimal profiling failed: {e2}")
            try:
                # Level 3: Custom HTML summary
                generate_basic_html_summary(df, report_path, title)
                return True
            except Exception as e3:
                logger.error(f"All profiling methods failed: {e3}")
                return False
```

**Lesson Learned**: External libraries (especially data science ones) can be unreliable. Always have fallbacks.

## Streamlit UI Best Practices

### Auto-Navigation Pattern (USER EXPERIENCE)

**Established Pattern**: Immediate auto-navigation after successful completion:

```python
# CORRECT: Auto-navigate for smooth flow
if st.button("🚀 Run Data Preparation", type="primary"):
    with st.spinner("Running data preparation..."):
        success = prep_runner.run_preparation_stage(run_id)
        
        if success:
            st.success("✅ Data preparation completed successfully!")
            st.balloons()  # Visual feedback
            # Auto-navigate immediately
            st.session_state['current_page'] = 'automl'
            st.rerun()

# WRONG: Manual navigation requires extra clicks
if success:
    st.success("Completed!")
    # Then user has to click "Continue" button - bad UX
```

### Button Layout Pattern

**Established Standard**:
```python
# CORRECT: Consistent 3:1 layout
col1, col2 = st.columns([3, 1])

with col1:
    if st.button("✅ Confirm Action", type="primary", use_container_width=True):
        # Process action

with col2:
    if st.button("← Back", use_container_width=True):
        # Navigate back
```

### Session State Management

**Critical Pattern**: Check for existing results to avoid re-runs:

```python
# ALWAYS check for existing completion
def show_prep_page():
    if 'run_id' not in st.session_state:
        st.error("No active run found.")
        return
    
    run_id = st.session_state['run_id']
    
    # Check if stage already completed
    metadata = storage.read_metadata(run_id)
    prep_completed = metadata and metadata.get('prep_info') is not None
    
    if prep_completed:
        # Show results, don't show run button
        display_prep_results(run_id)
    else:
        # Show run interface
        show_prep_execution_interface(run_id)
```

## Performance & Scalability Insights

### Memory Management for Large DataFrames

**Working Strategies**:
```python
# Avoid unnecessary copying
df_clean = df_original.copy()  # Only when modification needed

# Use inplace operations where safe
df_clean[column].fillna(median_value, inplace=True)

# Clean up intermediate objects
del df_intermediate  # Explicit cleanup for large DataFrames

# Process in chunks if needed (for future scaling)
chunk_size = 10000
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    process_chunk(chunk)
```

### Feature Expansion Management

**Real-World Example**: 80 → 315 columns (394% expansion)
- **One-hot encoding**: Categorical columns with many categories
- **DateTime features**: 4 features per datetime column (year, month, day, dayofweek)
- **TF-IDF features**: 50 features per text column (configurable)

**Optimization Strategy**:
```python
# Limit TF-IDF features to prevent explosion
vectorizer = TfidfVectorizer(
    max_features=50,  # Configurable limit
    stop_words='english',
    ngram_range=(1, 1)  # Only unigrams for performance
)

# Limit one-hot encoding for high-cardinality categoricals
if df[column].nunique() > 100:  # Threshold for cardinality
    # Use target encoding or frequency encoding instead
    # (Implementation for future enhancement)
```

## Integration Points for AutoML (Task 7)

### Expected Outputs for AutoML

**Files Created**:
```
data/runs/{run_id}/
├── cleaned_data.csv       # X and y ready for ML
├── metadata.json          # With prep_info containing encoder information
└── model/                 # 79+ encoder files for production deployment
    ├── target_label_encoder.joblib  # For label decoding
    ├── scaler_*.joblib             # For feature scaling
    └── *_tfidf_vectorizer.joblib   # For text processing
```

**Key Metadata Available**:
```python
prep_info = {
    'final_shape_after_prep': [1459, 315],  # rows × columns
    'cleaning_steps_performed': [
        'Imputed 58 missing values in column age using median',
        'Imputed 12 missing values in column category using mode',
        'Removed 5 duplicate rows'
    ],
    'encoders_scalers_info': {
        'scaler_numeric_feature': {
            'type': 'StandardScaler',
            'file_path': 'data/runs/.../model/scaler_numeric_feature.joblib',
            'columns_affected': ['numeric_feature'],
            'encoding_role': 'numeric-continuous'
        }
        # ... 79 more encoders
    },
    'profiling_report_path': '.taskmaster/data/runs/{run_id}/{run_id}_profile.html'
}
```

### AutoML Integration Strategy

**Target Information**:
```python
# Target is already encoded and ready for ML
target_column = target_info.name  # e.g., 'target_variable'
target_ml_type = target_info.ml_type  # 'binary_01', 'multiclass_text_labels', etc.

# For classification with text labels, decoder is available:
if target_ml_type in ['binary_text_labels', 'multiclass_text_labels']:
    label_encoder_path = f"model/target_{target_column}_label_encoder.joblib"
    # Use this to decode predictions back to original labels
```

**Feature Information**:
```python
# All features are numeric and ready for ML
df_ready = pd.read_csv(cleaned_data_path)
X = df_ready.drop(columns=[target_column])
y = df_ready[target_column]

# All preprocessing is done - can directly use with PyCaret/AutoML
```

## Common Pitfalls & Solutions

### Pitfall 1: Object Type Confusion

**Problem**: 
```python
'dict' object has no attribute 'name'
```

**Solution**: Always convert dictionary objects to Pydantic in runners:
```python
target_info = schemas.TargetInfo(**target_info_dict) if target_info_dict else None
```

### Pitfall 2: Memory Issues with Large DataFrames

**Problem**: Out of memory errors during processing

**Solution**: Monitor memory usage and use efficient operations:
```python
# Check memory usage
print(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Use efficient data types
df = df.astype({'category_column': 'category'})
```

### Pitfall 3: Encoder Path Issues

**Problem**: Encoders saved with unsafe file names

**Solution**: Sanitize column names for file paths:
```python
import re

def sanitize_filename(name: str) -> str:
    # Replace invalid characters with underscores
    return re.sub(r'[^\w\-_.]', '_', name)

safe_column_name = sanitize_filename(column_name)
encoder_path = model_dir / f"scaler_{safe_column_name}.joblib"
```

### Pitfall 4: Profile Generation Failures

**Problem**: ydata-profiling crashes on certain data patterns

**Solution**: Use three-level fallback strategy (implemented in profiling_logic.py)

### Pitfall 5: UI Navigation Issues

**Problem**: Multiple navigation buttons confusing users

**Solution**: Implement auto-navigation and conditional display:
```python
# Only show navigation when needed
if not stage_completed:
    # Show back button
    pass
# Auto-navigate on success, no manual navigation needed
```

## Testing Strategy for Data Pipeline

### Effective Testing Approach

**1. Unit Tests**: Test business logic functions in isolation
```python
def test_clean_data():
    # Create test DataFrame with known issues
    df_test = pd.DataFrame({
        'numeric_col': [1, 2, None, 4],
        'category_col': ['A', 'B', None, 'A']
    })
    # Test cleaning with specific schemas
    result, steps = cleaning_logic.clean_data(df_test, test_schemas, test_target)
    # Assert specific outcomes
```

**2. Integration Tests**: Test complete pipeline flows
```python
def test_prep_runner_end_to_end():
    # Use real run_id with actual artifacts
    success = prep_runner.run_preparation_stage('2025-01-06T120000Z_test')
    assert success
    # Verify all expected files created
```

**3. Real-World Tests**: Use actual datasets
```python
# Test with house prices dataset (1459 rows × 80 columns)
# This revealed memory optimization needs and edge cases
```

### Debugging Tools & Commands

**Essential Commands**:
```bash
# Activate environment
source .venv/bin/activate

# Test imports in isolation
python -c "from step_4_prep.cleaning_logic import clean_data; print('✓')"
python -c "from step_4_prep.encoding_logic import encode_features; print('✓')"
python -c "from step_4_prep.profiling_logic import generate_profile_report; print('✓')"

# Test runner without UI
python -c "
from step_4_prep.prep_runner import run_preparation_stage
result = run_preparation_stage('your_run_id_here')
print(f'Success: {result}')
"

# Check file artifacts
ls -la data/runs/{run_id}/
ls -la data/runs/{run_id}/model/

# Verify Streamlit page
streamlit run app.py --server.headless true --server.port 8502
```

**Log Analysis**:
```bash
# Monitor pipeline execution
tail -f data/runs/{run_id}/pipeline.log

# Check for specific errors
grep "ERROR" data/runs/{run_id}/pipeline.log
grep "Exception" data/runs/{run_id}/pipeline.log
```

## Technology Stack Context

### Current Working Versions
```
scikit-learn==1.5.2     # StandardScaler, LabelEncoder
pandas==2.2.3           # Core data processing
ydata-profiling==4.10.0 # Data profiling (with fallbacks)
streamlit==1.39.0       # UI framework
joblib==1.4.2           # Encoder persistence
pydantic==2.10.3        # Schema validation
```

### Library-Specific Insights

**ydata-profiling**: 
- Often fails on edge case data
- Memory intensive for large datasets
- Requires fallback strategies
- Performance optimizations essential

**scikit-learn**:
- Reliable for standard encoders
- StandardScaler works well with proper numeric conversion
- LabelEncoder essential for text targets

**pandas**:
- Memory efficient with proper data types
- `.copy()` usage should be minimized
- Vectorized operations preferred

## Future Task Preparation

### For Task 7 (Model Training/AutoML)

**Available Resources**:
- Clean, encoded, ML-ready dataset (`cleaned_data.csv`)
- Complete encoder registry for deployment
- Target information with ML task type
- Comprehensive data profile for insights

**Expected Integration Points**:
```python
# AutoML should be able to:
# 1. Load cleaned_data.csv directly
df_ml_ready = pd.read_csv(cleaned_data_path)
X = df_ml_ready.drop(columns=[target_column])
y = df_ml_ready[target_column]

# 2. Use target ML type for algorithm selection
if target_info.ml_type in ['binary_01', 'binary_text_labels']:
    # Classification setup
elif target_info.ml_type == 'numeric_continuous':
    # Regression setup

# 3. Access encoders for model deployment
encoders_info = metadata['prep_info']['encoders_scalers_info']
# Load specific encoders as needed
```

### For Task 8 (Model Explanation)

**Available Resources**:
- Trained model (from Task 7)
- Feature names and encoding information
- Original vs. encoded feature mapping
- Target label encoder (for text target decoding)

### For Future Deployment

**Critical Artifacts for Production**:
- All encoder/scaler files in `model/` directory
- Complete `encoders_scalers_info` metadata
- Target label encoder for prediction decoding
- Feature processing pipeline documentation

## Quick Reference Checklist

### Pre-Implementation Checklist
1. ✅ Virtual environment activated
2. ✅ Previous stage (validation) completed
3. ✅ Input files exist and are readable
4. ✅ Metadata contains target_info and feature_schemas
5. ✅ Storage/constants/schemas modules available

### Implementation Checklist
1. ✅ Pydantic object conversion in runner
2. ✅ Atomic file writes for metadata
3. ✅ Encoder persistence with descriptive names
4. ✅ Comprehensive error handling
5. ✅ Three-level fallback for external dependencies

### Testing Checklist
1. ✅ Unit tests for each logic module
2. ✅ Integration test with real data
3. ✅ UI component import verification
4. ✅ End-to-end pipeline test
5. ✅ Memory usage monitoring

### Debugging Checklist
1. ✅ Check all imports work in isolation
2. ✅ Verify file paths and permissions
3. ✅ Test runner function independently
4. ✅ Check logs for detailed error information
5. ✅ Verify session state in UI

This context should significantly accelerate Task 7 implementation by providing proven patterns, avoiding known pitfalls, and ensuring proper integration with the data preparation outputs. 