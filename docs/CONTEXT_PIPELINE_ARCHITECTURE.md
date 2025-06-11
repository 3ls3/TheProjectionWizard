# Pipeline Architecture & Data Flow

## 7-Step Pipeline Overview

```
Step 1: Data Ingest     → Step 2: Schema Definition → Step 3: Data Validation
        ↓                         ↓                          ↓
   Raw CSV Data            Feature Types              Validated Dataset
        ↓                         ↓                          ↓
Step 4: Data Prep       → Step 5: AutoML Training  → Step 6: Model Explain
        ↓                         ↓                          ↓
  Encoded Features          Trained Model            SHAP Explanations
        ↓                         ↓                          ↓
                    Step 7: Prediction System
                           ↓
                    Live Predictions
```

## Step-by-Step Breakdown

### Step 1: Data Ingest (`step_1_ingest/`)
**Purpose**: Load and store raw data in GCS
**Input**: CSV file path
**Output**: `{run_id}/original_data.csv` in GCS
**Key Files**: 
- `ingest_logic.py` - Main ingestion logic
- Validates CSV format and stores in GCS

### Step 2: Schema Definition (`step_2_schema/`)
**Purpose**: Define feature types and characteristics
**Input**: Raw data from Step 1
**Output**: `{run_id}/feature_definitions.json`
**Key Files**:
- `feature_definition_logic.py` - **CRITICAL**: Contains automatic feature suggestion logic
- `schema_logic.py` - Schema validation
**Important Behavior**: 
- **Automatic Suggestions**: May classify some continuous features as discrete (e.g., `square_feet`, `annual_income`, `loan_amount`)
- **User Override**: This is EXPECTED behavior - suggestions are meant to be reviewed and corrected by user via `/api/confirm-features` endpoint
- **Testing Strategy**: Tests validate that suggestions are reasonable, not perfect. Happy-path testing assumes user provides correct overrides

#### ⚠️ CRITICAL File Dependencies (Added 2025-06-11)
**LESSON LEARNED**: Step 2 schema confirmation functions have implicit file dependencies:

1. **`status.json`** - REQUIRED by `confirm_feature_schemas_gcs()`
   - Function downloads status, updates it, and uploads back to GCS
   - **Missing file causes**: "Schema confirmation failed" error
   - **Created by**: `/api/upload` endpoint during initial run setup
   - **Test requirement**: Tests must create this file or schema confirmation will fail

2. **`metadata.json`** - REQUIRED for run configuration
   - Contains initial data information and pipeline settings
   - Updated by schema confirmation with feature definitions
   - **Missing file causes**: "Metadata not found" errors in downstream steps

3. **`original_data.csv`** - Source data for analysis
   - Referenced by automatic suggestion algorithms
   - Required for feature type inference

**Testing Implication**: Tests cannot just upload CSV data - they must create the complete file structure that production creates, or schema confirmation will fail with cryptic error messages.

### Step 3: Data Validation (`step_3_validation/`)
**Purpose**: Validate data quality and completeness
**Input**: Raw data + feature definitions
**Output**: Validation report
**Key Files**:
- `validation_logic.py` - Data quality checks
- Ensures data meets pipeline requirements

### Step 4: Data Preparation (`step_4_prep/`)
**Purpose**: Encode features and prepare training data
**Input**: Validated data + feature definitions
**Output**: `{run_id}/cleaned_data.csv`, scalers in `{run_id}/scalers/`
**Key Files**:
- `encoding_logic.py` - **CRITICAL**: Feature encoding and StandardScaler creation
- `prep_logic.py` - Data preparation orchestration
**Critical Components**:
- One-hot encoding for categorical features
- StandardScaler for numeric features
- Feature interaction creation

### Step 5: AutoML Training (`step_5_automl/`)
**Purpose**: Train ML models using PyCaret
**Input**: Prepared data from Step 4
**Output**: `{run_id}/pycaret_pipeline.pkl`, model metrics
**Key Files**:
- `automl_logic.py` - PyCaret model training
- `model_logic.py` - Model selection and evaluation
**Critical Output**: Trained model with embedded preprocessing

### Step 6: Model Explanation (`step_6_explain/`)
**Purpose**: Generate SHAP explanations and feature importance
**Input**: Trained model from Step 5
**Output**: SHAP plots, feature importance data
**Key Files**:
- `explain_logic.py` - SHAP explanation generation
- Used for model interpretability

### Step 7: Prediction System (`step_7_predict/`)
**Purpose**: Make predictions on new data
**Input**: User input data
**Output**: Predictions with confidence intervals
**Key Files**:
- `column_mapper.py` - **CRITICAL**: Input transformation logic
- `predict_logic.py` - Prediction orchestration
**Critical Functions**:
- `encode_user_input_gcs()` - **WORKING**: Applies StandardScaler correctly
- `generate_feature_slider_config()` - **FIXED**: Was missing, now implemented

## Data Formats & Schemas

### Training Data Schemas

#### Regression Dataset (house_prices.csv)
```python
{
    'square_feet': 'numeric-continuous',      # 800-4000 range
    'bedrooms': 'numeric-discrete',           # 1-5 range  
    'bathrooms': 'numeric-discrete',          # 1.0-4.5 range
    'garage_spaces': 'numeric-discrete',      # 0-3 range
    'property_type': 'categorical-nominal',   # ['Single Family', 'Condo', 'Townhouse', 'Ranch']
    'neighborhood_quality_score': 'numeric-discrete',  # 1-10 range
    'price': 'target'                         # ~175k-787k range, mean ~424k
}
```

#### Classification Dataset (loan_approval.csv)
```python
{
    'applicant_age': 'numeric-discrete',      # 18-65 range
    'annual_income': 'numeric-continuous',    # 25k-285k range
    'credit_score': 'numeric-discrete',       # 391-850 range
    'employment_years': 'numeric-continuous', # 0.0-21.9 range
    'loan_amount': 'numeric-continuous',      # 50k-834k range
    'debt_to_income_ratio': 'numeric-continuous', # 0.051-0.399 range
    'education_level': 'categorical-nominal', # ['High School', 'Bachelors', 'Masters', 'PhD']
    'property_type': 'categorical-nominal',   # ['Single Family', 'Condo', 'Townhouse', 'Multi-family']
    'approved': 'target'                      # Binary: 0 (rejected), 1 (approved)
}
```

### Feature Definition Format
```json
{
    "feature_name": {
        "type": "numeric-continuous|numeric-discrete|categorical-nominal|categorical-ordinal",
        "description": "Human readable description",
        "values": ["possible", "values"],  # For categorical
        "min": 0,                          # For numeric
        "max": 100                         # For numeric
    }
}
```

### Column Mapping Format
```json
{
    "original_columns": ["col1", "col2"],
    "encoded_columns": ["col1", "col2_Cat1", "col2_Cat2"],
    "categorical_mappings": {
        "property_type": {
            "Single Family": [1, 0, 0, 0],
            "Condo": [0, 1, 0, 0]
        }
    },
    "scaler_features": ["square_feet", "bedrooms"]
}
```

## Critical Data Flow Issues

### Training vs Prediction Mismatch
**Problem**: Model trained with 31 features but prediction receives 12
**Root Cause**: Incorrect feature type classification in Step 2
**Solution**: Fixed heuristics in `feature_definition_logic.py`

### Scaling Issues
**Problem**: StandardScaler not applied during prediction
**Affected Endpoints**:
- ❌ `/api/predict/single` - Was bypassing scaler application
- ✅ `/api/predict` - Correctly applies scaling
**Solution**: Updated both endpoints to use `encode_user_input_gcs()`

## API Integration Points

### Primary Prediction Flow
```
User Input → /api/predict/single → column_mapper.encode_user_input_gcs() 
→ Model Prediction → Response
```

### Schema Generation Flow
```
Feature Definitions → column_mapper.generate_feature_slider_config() 
→ Frontend Schema
```

### Explanation Flow
```
User Input → /api/predict/explain → column_mapper.encode_user_input_gcs() 
→ Model + SHAP → Explanation Response
```

## GCS Artifact Dependencies

### Training Artifacts (Required for Prediction)
1. `{run_id}/pycaret_pipeline.pkl` - Trained model
2. `{run_id}/column_mapping.json` - Feature transformation rules
3. `{run_id}/scalers/` - StandardScaler objects
4. `{run_id}/metadata.json` - Pipeline configuration

### Data Artifacts
1. `{run_id}/original_data.csv` - Raw training data
2. `{run_id}/cleaned_data.csv` - Processed training data
3. `{run_id}/feature_definitions.json` - Feature schema

## Error Propagation Patterns

### Common Failure Points
1. **Step 2**: Incorrect feature type classification → Wrong encoding in Step 4
2. **Step 4**: Missing StandardScaler → Incorrect predictions in Step 7
3. **Step 7**: Column mapping mismatch → Model receives wrong features
4. **GCS**: Missing artifacts → Prediction system fails

### Debugging Strategy
1. Validate each step's output format
2. Check feature consistency between training and prediction
3. Verify StandardScaler application
4. Ensure GCS artifacts are complete

## Model Training Context

### Current Model Performance
- **Algorithm**: PyCaret AutoML (likely RandomForest or LightGBM)
- **Features**: 12 core features (was incorrectly 31)
- **Target Range**: $175k - $787k (mean ~$424k)
- **Training Size**: 300 samples (improved from 100)

### Feature Importance (Corrected)
1. `square_feet` - Primary driver (~60% importance)
2. `property_type` - Secondary driver (~25% importance)
3. `neighborhood_quality_score` - Tertiary (~10% importance)
4. Other features - Remaining ~5%

## Testing Implications

### Critical Test Points
1. **Feature Type Classification** - Ensure numeric features aren't categorized
2. **Scaling Consistency** - Verify StandardScaler application
3. **Column Mapping** - Check encoded feature alignment
4. **Prediction Magnitude** - Verify ~$425k not ~$150M predictions
5. **API Endpoint Parity** - All endpoints should return similar results

### Integration Test Requirements
- End-to-end pipeline execution
- API endpoint consistency validation
- GCS artifact verification
- Frontend-backend integration testing

This architecture provides the foundation for understanding data flow issues and implementing targeted tests. 