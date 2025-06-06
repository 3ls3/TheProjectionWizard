# Task 7 Context & Key Learnings

## Critical Information for Future LLM Sessions

This document contains essential context, lessons learned, and debugging insights from implementing Task 7 (AutoML with PyCaret) of The Projection Wizard. This information will help future LLM sessions avoid common pitfalls and implement subsequent tasks more efficiently.

## Project Architecture Patterns Established

### AutoML Pipeline Integration (CRITICAL)

**Proven Pattern**: AutoML stage seamlessly integrates with existing file-based communication:

```
data/runs/{run_id}/
├── cleaned_data.csv       # Input from Stage 6 (Data Preparation)
├── metadata.json          # Contains target_info, prep_info → Updated with automl_info
├── status.json           # Pipeline state tracking → Updated to automl stage
├── model/                # Contains encoders from prep + PyCaret pipeline
│   ├── target_*_label_encoder.joblib    # From prep stage
│   ├── scaler_*.joblib                  # From prep stage
│   ├── *_tfidf_vectorizer.joblib        # From prep stage
│   └── pycaret_pipeline.pkl             # NEW: Complete ML pipeline
└── pipeline.log          # Comprehensive logging
```

**Key Insight**: This pattern enables:
- Complete model deployment pipeline (preprocessing + ML model)
- Seamless handoff between data preparation and model training
- Model persistence with all required preprocessing components
- Clear debugging when AutoML fails

### PyCaret Integration Architecture (ESSENTIAL)

**Critical Understanding**: PyCaret handles preprocessing internally but we need external encoders for deployment.

**Working Integration Pattern**:
```python
# 1. Load ML-ready data (already processed by prep stage)
df_ml_ready = pd.read_csv(cleaned_data_path)
X = df_ml_ready.drop(columns=[target_column])
y = df_ml_ready[target_column]

# 2. PyCaret setup() does additional internal preprocessing
pc_setup = setup(
    data=df_ml_ready,
    target=target_column_name,
    session_id=session_id,
    log_experiment=False,  # Disable MLflow logging
    verbose=False,         # Reduce console output
    html=False,           # Disable HTML output
    use_gpu=False,        # CPU-only for compatibility
    train_size=0.8        # 80% for training, 20% for test
)

# 3. Model comparison and selection
top_models = compare_models(n_select=top_n_models_to_compare, verbose=False)
best_model = top_models[0] if isinstance(top_models, list) else top_models

# 4. Finalize model (train on full dataset)
final_pipeline = finalize_model(best_model)

# 5. Save complete pipeline (includes PyCaret's internal preprocessing + model)
save_model(final_pipeline, str(pycaret_model_dir / 'pycaret_pipeline'))
```

**Lesson Learned**: PyCaret's save_model() creates a complete pipeline that includes:
- PyCaret's internal preprocessing (additional to our prep stage)
- The trained ML model
- All necessary components for prediction on new data

### Dual Preprocessing Strategy (DEPLOYMENT-CRITICAL)

**Why Two Preprocessing Layers**:

```python
# Layer 1: Our prep stage (step_4_prep/)
# Purpose: Raw data → ML-ready features with saved encoders
# Output: cleaned_data.csv + encoder files for deployment on raw data

# Layer 2: PyCaret internal preprocessing
# Purpose: Additional ML optimizations (scaling, imputation, etc.)
# Output: Included in pycaret_pipeline.pkl for direct ML predictions
```

**For Deployment**:
```python
# Option 1: Raw data → Use our encoders → PyCaret pipeline
raw_data → [our_encoders] → cleaned_data → [pycaret_pipeline] → predictions

# Option 2: Pre-processed data → Direct PyCaret pipeline  
cleaned_data → [pycaret_pipeline] → predictions
```

**Lesson Learned**: This dual strategy provides maximum flexibility for deployment scenarios.

## PyCaret Core Principles

### Task Type Handling (FUNDAMENTAL)

**Dynamic Module Import Strategy**:
```python
def run_pycaret_experiment(df_ml_ready, target_column_name, task_type, ...):
    try:
        if task_type == 'classification':
            # Import classification modules
            from pycaret.classification import setup, compare_models, finalize_model, save_model, pull
        elif task_type == 'regression':
            # Import regression modules  
            from pycaret.regression import setup, compare_models, finalize_model, save_model, pull
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
```

**Why This Matters**: PyCaret has separate modules for classification and regression with identical APIs but different internal implementations.

### Model Selection Strategy (PROVEN)

**Optimal Configuration for MVP**:
```python
# Model comparison configuration
top_models = compare_models(
    n_select=3,                    # Compare top 3 models for speed
    include=None,                  # Try all available models
    exclude=['lightgbm', 'xgboost'] if not allow_lightgbm_and_xgboost else None,
    verbose=False,                 # Reduce output
    fold=5                        # 5-fold CV (PyCaret default)
)

# Model finalization (important for production)
final_pipeline = finalize_model(best_initial_model)  # Trains on full dataset
```

**Key Insights**:
- `compare_models()` uses cross-validation for reliable performance estimates
- `finalize_model()` retrains the best model on the complete dataset
- Model exclusion helps avoid installation/environment issues

### Performance Metrics Extraction (CRITICAL)

**Robust Metrics Extraction Pattern**:
```python
def extract_performance_metrics(task_type: str) -> Dict[str, float]:
    try:
        # Pull the results dataframe from PyCaret
        results_df = pull()
        
        if task_type == 'classification':
            # Classification metrics mapping
            metric_mapping = {
                'AUC': 'AUC',
                'Accuracy': 'Accuracy', 
                'F1': 'F1',
                'Recall': 'Recall',
                'Precision': 'Prec.'  # Note: PyCaret uses 'Prec.' not 'Precision'
            }
        elif task_type == 'regression':
            # Regression metrics mapping  
            metric_mapping = {
                'R2': 'R2',
                'RMSE': 'RMSE',
                'MAE': 'MAE',
                'MAPE': 'MAPE'
            }
        
        # Extract metrics from first row (best model)
        performance_metrics = {}
        if not results_df.empty:
            for our_name, pycaret_name in metric_mapping.items():
                if pycaret_name in results_df.columns:
                    value = results_df.iloc[0][pycaret_name]
                    if pd.notna(value):
                        performance_metrics[our_name] = float(value)
        
        return performance_metrics
    except Exception as e:
        logger.warning(f"Could not extract performance metrics: {e}")
        return {}
```

**Lesson Learned**: PyCaret's column names don't always match expected names (e.g., 'Prec.' vs 'Precision').

## Schema Evolution for AutoML

### AutoMLInfo Schema Design (PRODUCTION-READY)

```python
class AutoMLInfo(BaseModel):
    """Information about the AutoML stage results."""
    tool_used: str = Field(..., description="AutoML tool used (e.g., 'PyCaret')")
    best_model_name: Optional[str] = None
    pycaret_pipeline_path: Optional[str] = None  # Relative to run_dir
    performance_metrics: Optional[Dict[str, float]] = None
    automl_completed_at: Optional[str] = None
    target_column: Optional[str] = None           # For display convenience
    task_type: Optional[str] = None               # For display convenience
    dataset_shape_for_training: Optional[List[int]] = None  # [rows, cols]

class MetadataWithAutoML(MetadataWithFullSchema):
    """Metadata model that includes AutoML information."""
    automl_info: Optional[AutoMLInfo] = None
```

**Design Principles**:
- **Optional fields**: Backward compatibility with existing runs
- **Convenience fields**: Duplicate some info for easier UI display
- **Relative paths**: Portable across different environments
- **Structured metrics**: Dictionary for flexible metric storage

### Metadata Update Pattern (ATOMIC)

**Critical Pattern**: Always use atomic operations for metadata updates:

```python
# CORRECT: Load → Update → Save atomically
metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)

automl_info = {
    'tool_used': 'PyCaret',
    'best_model_name': model_name,
    'pycaret_pipeline_path': str(Path(constants.MODEL_DIR_NAME) / 'pycaret_pipeline.pkl'),
    'performance_metrics': metrics,
    'automl_completed_at': datetime.utcnow().isoformat(),
    'target_column': target_info.name,
    'task_type': target_info.task_type,
    'dataset_shape_for_training': list(df_ml_ready.shape)
}

metadata_dict['automl_info'] = automl_info
storage.write_json_atomic(run_dir / constants.METADATA_FILENAME, metadata_dict)
```

## UI Design Patterns for AutoML

### Results Display Strategy (USER-FRIENDLY)

**Performance Metrics Visualization**:
```python
# Primary display: st.metric with percentage formatting
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

# Secondary display: Detailed table in expander
with st.expander("📋 Detailed Performance Metrics", expanded=False):
    df_metrics = pd.DataFrame([
        {"Metric": k, "Value": f"{v:.4f}"} 
        for k, v in performance_metrics.items()
    ])
    st.table(df_metrics)
```

### Model Download Strategy (DEPLOYMENT-READY)

```python
# Download button with proper MIME type
with open(model_file_path, 'rb') as f:
    st.download_button(
        label="🤖 Download PyCaret Pipeline",
        data=f.read(),
        file_name=f"{run_id}_pycaret_pipeline.pkl",
        mime="application/octet-stream",  # For .pkl files
        use_container_width=True
    )

# File size information
file_size_kb = model_file_path.stat().st_size / 1024
st.caption(f"File size: {file_size_kb:.1f} KB")
```

### Auto-Navigation Pattern (CONSISTENT)

**Established Auto-Navigation Flow**:
```python
if stage_success:
    st.success("✅ AutoML training completed successfully!")
    
    # Show immediate results
    st.subheader("🎉 Training Results")
    # Display key metrics...
    
    # Auto-navigate with visual feedback
    st.balloons()
    st.success("🔍 Proceeding to Model Explanation...")
    st.session_state['current_page'] = 'explain'
    st.rerun()
```

## Error Handling & Debugging

### PyCaret Common Issues (CRITICAL)

**Issue 1: Silent Parameter Failures**
```python
# WRONG: These parameters don't exist in all PyCaret versions
pc_setup = setup(data=df, target=target, silent=True)  # ❌ 'silent' doesn't exist

# CORRECT: Use parameters that actually exist
pc_setup = setup(
    data=df,
    target=target,
    log_experiment=False,  # Disable MLflow
    verbose=False,         # Reduce output
    html=False            # Disable HTML reports
)
```

**Issue 2: Import Path Sensitivity**
```python
# WRONG: Import at module level can cause environment issues
from pycaret.classification import setup

# CORRECT: Import within function after task type is known
def run_pycaret_experiment(task_type, ...):
    if task_type == 'classification':
        from pycaret.classification import setup, compare_models
    # This allows dynamic loading and better error handling
```

**Issue 3: Model Type Handling**
```python
# WRONG: Assume compare_models always returns list
best_model = compare_models(n_select=1)[0]  # ❌ Can return single object

# CORRECT: Handle both cases
top_models = compare_models(n_select=n)
if isinstance(top_models, list):
    best_model = top_models[0]
else:
    best_model = top_models  # Single model returned
```

### Validation Strategy (ESSENTIAL)

**Input Validation Function**:
```python
def validate_pycaret_inputs(df: pd.DataFrame, target_column: str, task_type: str) -> Tuple[bool, List[str]]:
    """Comprehensive input validation before PyCaret execution."""
    issues = []
    
    # DataFrame checks
    if df.empty:
        issues.append("DataFrame is empty")
    
    if len(df) < 10:  # PyCaret needs minimum samples
        issues.append(f"Dataset too small: {len(df)} rows (minimum 10 required)")
    
    # Target column checks
    if target_column not in df.columns:
        issues.append(f"Target column '{target_column}' not found")
    
    if target_column in df.columns:
        if df[target_column].nunique() < 2:
            issues.append(f"Target column '{target_column}' has insufficient unique values")
    
    # Task type validation
    if task_type not in ['classification', 'regression']:
        issues.append(f"Invalid task type: {task_type}")
    
    return len(issues) == 0, issues
```

### Debugging Commands (ESSENTIAL)

**Critical Debugging Commands**:
```bash
# Test PyCaret imports in isolation
python -c "from pycaret.classification import setup; print('✓ Classification')"
python -c "from pycaret.regression import setup; print('✓ Regression')"

# Test AutoML runner without UI
python -c "
from step_5_automl.automl_runner import run_automl_stage
result = run_automl_stage('your_run_id_here')
print(f'Success: {result}')
"

# Test specific run validation
python -c "
from step_5_automl.automl_runner import validate_automl_stage_inputs
result = validate_automl_stage_inputs('your_run_id_here')
print(f'Valid: {result}')
"

# Check model file and size
ls -la data/runs/{run_id}/model/pycaret_pipeline.pkl

# Monitor pipeline execution
tail -f data/runs/{run_id}/pipeline.log | grep automl
```

## Integration with App Navigation

### App.py Integration Pattern (PROVEN)

**Navigation Logic Update**:
```python
# Check if data preparation is complete
prep_complete = (status_data.get('stage') == constants.PREP_STAGE and 
                status_data.get('status') == 'completed')

# Enable AutoML button only after prep completion
if prep_complete:
    if st.sidebar.button("🤖 Model Training", use_container_width=True):
        st.session_state['current_page'] = 'automl'
        st.rerun()
else:
    st.sidebar.button("🤖 Model Training", disabled=True, 
                     help="Complete data preparation first")

# Check if AutoML is complete for next stage
automl_complete = (status_data.get('stage') == constants.AUTOML_STAGE and 
                  status_data.get('status') == 'completed')

if automl_complete:
    st.sidebar.button("📊 Model Explanation", use_container_width=True)
else:
    st.sidebar.button("📊 Model Explanation", disabled=True, 
                     help="Complete model training first")
```

**Module Import Pattern**:
```python
# Use importlib for numeric filenames
automl_page_module = importlib.import_module('ui.06_automl_page')

# Route in main function
elif current_page == 'automl':
    automl_page_module.show_automl_page()
```

## Performance & Scalability Insights

### PyCaret Performance Tuning

**Working Configuration for Production**:
```python
# Optimized PyCaret setup for reasonable performance
pc_setup = setup(
    data=df_ml_ready,
    target=target_column_name,
    session_id=123,           # Reproducible results
    train_size=0.8,           # 80/20 split
    log_experiment=False,     # Disable MLflow logging overhead
    verbose=False,            # Reduce console output
    html=False,              # Disable HTML report generation
    use_gpu=False            # CPU-only for compatibility
)

# Limited model comparison for speed
top_models = compare_models(
    n_select=3,              # Compare only top 3 for speed
    fold=5,                  # 5-fold CV (default)
    sort='AUC',             # Primary metric for classification
    verbose=False           # Reduce output
)
```

### Memory Management

**Large Dataset Strategies**:
```python
# Monitor DataFrame memory usage
logger.info(f"ML-ready data memory usage: {df_ml_ready.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# For very large datasets, consider sampling for model comparison
if len(df_ml_ready) > 10000:
    logger.warning(f"Large dataset ({len(df_ml_ready)} rows). Consider sampling for initial experiments.")
```

### Real-World Performance Results

**Test Case: Titanic Dataset**
- **Input**: 891 rows × 904 columns (post feature engineering)
- **PyCaret Setup**: ~2 seconds
- **Model Comparison**: ~13 seconds (3 models)
- **Model Finalization**: ~1 second  
- **Total Time**: ~18 seconds end-to-end
- **Memory Usage**: Reasonable for development environments
- **Model Size**: 1,247 KB PyCaret pipeline

## Integration Points for Next Stages

### For Task 8 (Model Explanation)

**Available Resources**:
```python
# AutoML outputs ready for explanation
automl_info = {
    'tool_used': 'PyCaret',
    'best_model_name': 'RidgeClassifier',
    'pycaret_pipeline_path': 'model/pycaret_pipeline.pkl',
    'performance_metrics': {'AUC': 0.8816, 'Accuracy': 0.8426, ...},
    'target_column': 'Survived',
    'task_type': 'classification',
    'dataset_shape_for_training': [891, 904]
}

# Model pipeline for SHAP analysis
pipeline = joblib.load(pycaret_model_dir / 'pycaret_pipeline.pkl')

# Training data for explanation
df_ml_ready = pd.read_csv(cleaned_data_path)
X = df_ml_ready.drop(columns=[target_column])
```

**Expected Integration**:
- Load PyCaret pipeline for SHAP explanation
- Use training data for explanation context
- Generate global SHAP summary plots
- Save explanation artifacts alongside model

### For Future Deployment

**Complete Deployment Package**:
```
data/runs/{run_id}/model/
├── pycaret_pipeline.pkl           # Complete ML pipeline (PyCaret)
├── target_*_label_encoder.joblib  # Target decoding (our prep stage)
├── scaler_*.joblib               # Feature preprocessing (our prep stage)
└── *_tfidf_vectorizer.joblib     # Text preprocessing (our prep stage)
```

**Deployment Strategies**:
1. **Direct PyCaret**: Use only `pycaret_pipeline.pkl` on cleaned data
2. **Full Pipeline**: Raw data → our encoders → PyCaret pipeline
3. **Hybrid**: Partially processed data with selective encoder usage

## Common Pitfalls & Solutions

### Pitfall 1: PyCaret Environment Issues

**Problem**: PyCaret installation conflicts or missing dependencies
```bash
ModuleNotFoundError: No module named 'pycaret.classification'
```

**Solution**: Test imports early and provide clear error messages
```python
try:
    if task_type == 'classification':
        from pycaret.classification import setup
except ImportError as e:
    logger.error(f"PyCaret classification module not available: {e}")
    return None, None, f"PyCaret not properly installed: {e}"
```

### Pitfall 2: Data Type Issues

**Problem**: PyCaret expects specific data types
```
ValueError: could not convert string to float
```

**Solution**: Ensure data is properly preprocessed
```python
# Our prep stage should handle this, but validate anyway
numeric_columns = df_ml_ready.select_dtypes(include=[np.number]).columns
categorical_columns = df_ml_ready.select_dtypes(include=['object']).columns
logger.info(f"Numeric columns: {len(numeric_columns)}, Categorical: {len(categorical_columns)}")
```

### Pitfall 3: Target Variable Issues

**Problem**: Target not properly encoded for task type
```
ValueError: Classification target must be categorical
```

**Solution**: Validate target encoding before PyCaret
```python
if task_type == 'classification':
    unique_values = df_ml_ready[target_column].nunique()
    logger.info(f"Target has {unique_values} unique values for classification")
    if unique_values > 20:
        logger.warning(f"High cardinality target ({unique_values} classes)")
```

### Pitfall 4: Model Saving Issues

**Problem**: Model save fails silently or with unclear errors
```
Exception: Could not save model
```

**Solution**: Verify directory permissions and space
```python
# Ensure model directory exists and is writable
pycaret_model_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Model directory: {pycaret_model_dir}")
logger.info(f"Directory writable: {os.access(pycaret_model_dir, os.W_OK)}")

# Save with error handling
try:
    save_model(final_pipeline, str(pycaret_model_dir / 'pycaret_pipeline'))
    logger.info("Model saved successfully")
except Exception as e:
    logger.error(f"Model save failed: {e}")
    raise
```

## Quick Reference Checklist

### Pre-Implementation Checklist
1. ✅ Virtual environment activated with PyCaret installed
2. ✅ Previous stage (data preparation) completed successfully
3. ✅ Input files exist: cleaned_data.csv, metadata.json with prep_info
4. ✅ Target info and task type properly set
5. ✅ Model directory exists and is writable

### Implementation Checklist
1. ✅ Dynamic import based on task type
2. ✅ Input validation before PyCaret execution
3. ✅ Proper PyCaret setup with performance optimizations
4. ✅ Model comparison with error handling
5. ✅ Model finalization and persistence
6. ✅ Performance metrics extraction and validation
7. ✅ Metadata updates with atomic operations
8. ✅ Status updates with meaningful messages

### Testing Checklist
1. ✅ Unit tests for PyCaret logic functions
2. ✅ Integration test with real data end-to-end
3. ✅ UI component import verification
4. ✅ Navigation integration testing
5. ✅ Error handling validation

### Debugging Checklist
1. ✅ Check all imports work in isolation
2. ✅ Verify input files exist and are readable
3. ✅ Test runner function independently of UI
4. ✅ Check logs for detailed error information
5. ✅ Verify model file creation and accessibility

This context should significantly accelerate Task 8 implementation by providing proven patterns, avoiding known pitfalls, and ensuring proper integration with the established pipeline architecture. 