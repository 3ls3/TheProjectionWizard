# Step 4 - Data Preparation Module

This module contains the core data cleaning logic for The Projection Wizard's data preparation stage.

## Overview

The data preparation stage is responsible for:
1. **Data Cleaning**: Handling missing values and removing duplicates based on feature schemas ✅
2. **Data Encoding**: Converting features to ML-ready formats ✅
3. **Data Profiling**: Generating ydata-profiling reports (future sub-task)

## Current Implementation

**Status:** ✅ **All Sub-tasks Completed** - Data Preparation Module Ready for Production

### Sub-task 6.A: Core Cleaning Logic (`cleaning_logic.py`) ✅

**Status: COMPLETED**

#### Main Function: `clean_data()`

```python
def clean_data(df_original: pd.DataFrame, 
               feature_schemas: Dict[str, schemas.FeatureSchemaInfo], 
               target_info: schemas.TargetInfo,
               cleaning_config: Optional[dict] = None) -> Tuple[pd.DataFrame, List[str]]
```

**Purpose**: Clean the original DataFrame based on feature schemas and target information.

**Parameters**:
- `df_original`: The raw DataFrame loaded from `original_data.csv`
- `feature_schemas`: Dictionary of `FeatureSchemaInfo` objects from `metadata.json`
- `target_info`: `TargetInfo` object with target column information
- `cleaning_config`: Optional configuration dictionary (for future use)

**Returns**: 
- Tuple containing the cleaned DataFrame and a list of cleaning steps performed

#### Key Features

1. **Missing Value Imputation**:
   - **Numeric columns** (`numeric-continuous`, `numeric-discrete`): Imputed with median
   - **Categorical columns** (`categorical-nominal`, `categorical-ordinal`, `text`): Imputed with mode or `"_UNKNOWN_"`
   - **Boolean columns**: Imputed with mode or `False`
   - **Datetime columns**: Forward/backward fill or placeholder date
   - **Target column**: Uses target's ML type to determine strategy

2. **Duplicate Removal**:
   - Removes duplicate rows while keeping the first occurrence
   - Logs the number of rows removed

3. **Robust Error Handling**:
   - Handles columns not found in schemas with fallback inference
   - Gracefully handles edge cases (all NaN columns, empty data, etc.)
   - Comprehensive logging of all cleaning steps

#### Encoding Role Support

The cleaning logic supports all encoding roles defined in `common/constants.py`:
- `numeric-continuous`
- `numeric-discrete` 
- `categorical-nominal`
- `categorical-ordinal`
- `text`
- `datetime`
- `boolean`
- `target`

#### Example Usage

```python
from step_4_prep.cleaning_logic import clean_data
from common import schemas

# Load your data and schemas
df_original = pd.read_csv("data/runs/run_123/original_data.csv")
metadata = load_metadata("data/runs/run_123/metadata.json")

# Convert metadata to schema objects
target_info = schemas.TargetInfo(**metadata["target_info"])
feature_schemas = {
    col: schemas.FeatureSchemaInfo(**info) 
    for col, info in metadata["feature_schemas"].items()
}

# Clean the data
df_cleaned, cleaning_steps = clean_data(df_original, feature_schemas, target_info)

print("Cleaning steps performed:")
for step in cleaning_steps:
    print(f"  - {step}")
```

### Sub-task 6.B: Core Encoding Logic (`encoding_logic.py`) ✅

**Status: COMPLETED**

#### Main Function: `encode_features()`

```python
def encode_features(df_cleaned: pd.DataFrame, 
                   feature_schemas: Dict[str, schemas.FeatureSchemaInfo], 
                   target_info: schemas.TargetInfo,
                   run_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]
```

**Purpose**: Convert cleaned data into ML-ready formats based on feature schemas and target information.

**Parameters**:
- `df_cleaned`: The DataFrame after cleaning from `cleaning_logic.py`
- `feature_schemas`: Dictionary of `FeatureSchemaInfo` objects from `metadata.json`
- `target_info`: `TargetInfo` object with target information
- `run_id`: Current run ID, used for saving encoders/scalers

**Returns**: 
- Tuple containing the encoded DataFrame and encoders/scalers info dictionary

#### Target Variable Encoding

Based on `target_info.ml_type`:
- **`binary_01`**: Ensures column is int 0/1 (maps boolean/text if needed)
- **`multiclass_int_labels`**: Ensures column is int
- **`binary_text_labels`**, **`multiclass_text_labels`**: Uses `LabelEncoder` and saves fitted encoder
- **`numeric_continuous`**: Ensures column is float

#### Feature Encoding by Role

- **`numeric-continuous`**, **`numeric-discrete`**: `StandardScaler` (saves fitted scaler)
- **`categorical-nominal`**: One-hot encoding with `pd.get_dummies()`
- **`categorical-ordinal`**: Simple integer encoding using categorical codes (MVP)
- **`boolean`**: Ensures 0/1 format, handles text boolean values (Yes/No, True/False, etc.)
- **`datetime`**: Extracts year, month, day, dayofweek features
- **`text`**: `TfidfVectorizer` with max_features=50 (saves fitted vectorizer)
- **`identifier_ignore`**: Drops column completely

#### Encoder/Scaler Management

- All fitted encoders and scalers are saved to `data/runs/<run_id>/model/` directory
- Comprehensive metadata about each encoder is returned in `encoders_scalers_info`
- File paths, parameters, and transformation details are tracked for later use

#### Example Usage

```python
from step_4_prep.encoding_logic import encode_features

# After cleaning
df_encoded, encoders_info = encode_features(df_cleaned, feature_schemas, target_info, run_id)

print(f"Encoded shape: {df_encoded.shape}")
print(f"Saved encoders: {list(encoders_info.keys())}")
```

### Sub-task 6.C: Profiling Logic (`profiling_logic.py`) ✅

**Status: COMPLETED**

#### Main Function: `generate_profile_report()`

```python
def generate_profile_report(df_final_prepared: pd.DataFrame, 
                          report_path: Path, 
                          title: str) -> bool
```

**Purpose**: Generate a comprehensive ydata-profiling HTML report for the prepared DataFrame.

**Parameters**:
- `df_final_prepared`: The DataFrame after cleaning and encoding
- `report_path`: Full Path object where the HTML report should be saved
- `title`: Title for the ydata-profiling report

**Returns**: 
- `True` if report generation successful, `False` otherwise

#### Enhanced Function with Fallback: `generate_profile_report_with_fallback()`

```python
def generate_profile_report_with_fallback(df_final_prepared: pd.DataFrame, 
                                        report_path: Path, 
                                        title: str,
                                        run_id: Optional[str] = None) -> bool
```

**Additional Features**:
- Includes fallback mechanisms for better compatibility
- Optional run ID for better logging context
- Multiple fallback levels for robust profiling

#### Key Features

1. **Robustness**: 
   - Handles both `ydata-profiling` and legacy `pandas-profiling` packages
   - Multiple fallback strategies ensure report generation always succeeds
   - Conservative profiling settings to prevent memory/performance issues

2. **Performance Optimization**:
   - Selective correlation calculations (skip expensive Spearman, Kendall, phi_k, Cramers)
   - Limited sample sizes for performance
   - Disabled interactions for large datasets
   - Controlled duplicate analysis

3. **Three-Level Fallback Strategy**:
   - **Level 1**: Full ydata-profiling report with optimized settings
   - **Level 2**: Minimal ydata-profiling report with basic statistics only
   - **Level 3**: Custom HTML summary with basic descriptive statistics

4. **Comprehensive Error Handling**:
   - Cleanup of partial files on failure
   - Detailed logging of all attempts and failures
   - Graceful handling of edge cases (empty DataFrames, etc.)

#### Example Usage

```python
from step_4_prep.profiling_logic import generate_profile_report_with_fallback
from pathlib import Path

# Generate profile report with fallback
report_path = Path("data/runs/run_123/ydata_profile.html")
success = generate_profile_report_with_fallback(
    df_final_prepared=df_encoded,
    report_path=report_path,
    title="Data Preparation Profile Report",
    run_id="run_123"
)

if success:
    print(f"Profile report saved to: {report_path}")
else:
    print("Profile report generation failed")
```

#### Testing Results

Successfully tested with:
- Regular DataFrames (100 rows × 7 columns): Generated 1MB+ reports
- Large DataFrames (1000 rows × 6 columns): Handled efficiently
- Edge cases: Empty DataFrames, single columns, all missing values
- All fallback levels working as expected

## Design Principles

1. **Immutability**: Original DataFrame is copied before modification
2. **Transparency**: All cleaning steps are logged for auditability
3. **Schema-Driven**: Cleaning strategies are based on encoding roles from user/system decisions
4. **Fallback Safety**: Graceful handling when columns are missing from schemas
5. **MVP Focus**: Simple, reliable strategies that can be enhanced later

## Integration with Pipeline

This cleaning logic integrates with the broader pipeline through:
- Reading from `data/runs/<run_id>/original_data.csv`
- Using schemas from `data/runs/<run_id>/metadata.json`
- Will output to `data/runs/<run_id>/cleaned_data.csv` (in prep runner)
- Logging steps for inclusion in `data/runs/<run_id>/metadata.json`

## Future Enhancements

- Configurable cleaning strategies via `cleaning_config` parameter
- Advanced outlier detection and handling
- Custom imputation strategies per column
- Data validation before and after cleaning
- Performance optimization for large datasets

## Testing

The modules have been comprehensively tested with:

**Cleaning Logic:**
- Various data types and missing value patterns
- Edge cases (all NaN columns, no missing values, unknown columns)
- Integration with real metadata format from storage
- Datetime handling with pandas modern syntax

**Encoding Logic:**
- All encoding roles (numeric, categorical, boolean, datetime, text, identifier)
- All target ML types (binary_01, multiclass_int_labels, text_labels, numeric_continuous)
- Encoder/scaler saving and metadata tracking
- Edge cases (missing columns in schema, unknown roles)
- Complex transformations (TF-IDF, one-hot, datetime features)

**Profiling Logic:**
- ydata-profiling report generation with optimized settings
- Three-level fallback strategy (full → minimal → basic HTML)
- Robust handling of package compatibility issues
- Performance optimizations for large datasets
- Edge case handling (empty data, single columns, all missing values)

### Sub-task 6.D: Prep Stage Runner (`prep_runner.py`) ✅

**Status: COMPLETED**

#### Main Function: `run_preparation_stage()`

```python
def run_preparation_stage(run_id: str) -> bool
```

**Purpose**: Orchestrate the complete data preparation stage for a given run.

**Parameters**:
- `run_id`: Unique run identifier

**Returns**: 
- `True` if stage completes successfully, `False` otherwise

#### Stage Orchestration

The prep stage runner executes the following steps in sequence:

1. **Input Loading & Validation**:
   - Load and validate `metadata.json` with Pydantic schema conversion
   - Load `original_data.csv` 
   - Convert target_info and feature_schemas dictionaries to Pydantic objects
   - Comprehensive error handling for missing or invalid inputs

2. **Data Cleaning**:
   - Call `cleaning_logic.clean_data()` with loaded schemas
   - Handle missing values and remove duplicates
   - Log all cleaning steps performed

3. **Feature Encoding**:
   - Call `encoding_logic.encode_features()` with cleaned data
   - Transform features to ML-ready formats
   - Save encoders/scalers to `data/runs/<run_id>/model/`

4. **Data Output**:
   - Save final prepared data to `cleaned_data.csv`
   - Verify file creation and log final data shape

5. **Profiling Report**:
   - Generate ydata-profiling report with fallback handling
   - Save to `<run_id>_profile.html`
   - Non-critical step (continues if profiling fails)

6. **Metadata & Status Updates**:
   - Update `metadata.json` with comprehensive prep_info
   - Update `status.json` to completed state
   - Atomic file operations for data integrity

#### Key Features

1. **Robust Error Handling**:
   - Comprehensive try-catch blocks for each stage
   - Graceful failure with detailed error logging
   - Status updates on both success and failure

2. **Schema Validation**:
   - Critical conversion of dict metadata to Pydantic objects
   - Proper handling of TargetInfo.name vs column field mapping
   - Validation of required metadata fields

3. **Comprehensive Logging**:
   - Run-scoped logger with detailed progress tracking
   - Summary statistics and completion report
   - File-by-file output verification

4. **Status Management**:
   - Updates status.json at key stages (running → completed/failed)
   - Detailed error messages for debugging
   - Non-blocking status updates (continues on status write failures)

#### Additional Functions

**`validate_prep_stage_inputs(run_id: str) -> bool`**
- Pre-flight validation of all required inputs
- Checks file existence and metadata structure
- Validates Pydantic schema compatibility

**`get_prep_stage_summary(run_id: str) -> Optional[dict]`**
- Post-execution summary of prep stage results
- Returns key metrics and file availability status
- Useful for debugging and monitoring

#### Example Usage

```python
from step_4_prep.prep_runner import run_preparation_stage

# Run the complete prep stage
success = run_preparation_stage("2025-06-05T141742Z_c021f7b8")

if success:
    print("✅ Data preparation completed successfully!")
else:
    print("❌ Data preparation failed - check logs")
```

#### Testing Results

Successfully tested on real dataset:
- **Input**: House prices dataset (1459 rows × 80 columns)
- **Output**: Prepared dataset (1459 rows × 315 columns)
- **Processing**: 38 cleaning steps, 79 encoders/scalers saved
- **Profile Report**: 8.6MB HTML report generated
- **Execution Time**: ~3 minutes for full pipeline
- **All outputs verified**: cleaned_data.csv, updated metadata.json, status.json, profile report

### Sub-task 6.E: Prep UI (`ui/05_prep_page.py`) ✅

**Status: COMPLETED**

#### Main Function: `show_prep_page()`

```python
def show_prep_page() -> None
```

**Purpose**: Provide a complete Streamlit UI for data preparation stage with execution and results viewing.

#### Key Features

1. **Run Data Preparation Button**:
   - Calls `prep_runner.run_preparation_stage(run_id)` on click
   - Shows spinner with progress indication during execution
   - Displays immediate success/failure messages
   - Prevents multiple concurrent runs with session state management

2. **Existing Results Display** (when page is revisited):
   - Reads `status.json` to check if prep stage completed
   - Loads and displays comprehensive preparation summary from `metadata.json`
   - Key metrics: final data shape, cleaning steps count, encoders/scalers count
   - Completion timestamp display

3. **File Downloads & Viewing**:
   - **Cleaned Data Download**: `st.download_button` for cleaned_data.csv with file size info
   - **Profile Report Options**: 
     - Download button for HTML report
     - Optional iframe viewer for inline report viewing (with performance warning)
     - File size information

4. **Detailed Information Expandables**:
   - **Cleaning Steps**: Numbered list of all cleaning operations performed
   - **Encoding Details**: Grouped by encoder type with columns affected
   - **Model Artifacts**: List of saved encoder/scaler files with sizes

5. **Navigation & Flow Control**:
   - Back to Data Validation
   - Forward to Model Training (only enabled after successful prep)
   - Re-run Data Preparation option
   - Force rerun capability with session state

6. **Robust Error Handling**:
   - Graceful handling of missing files or corrupted metadata
   - Detailed error messages from status.json
   - Log file path suggestions for debugging
   - Session state cleanup on completion/failure

#### User Experience

1. **First Visit** (prep not run):
   - Shows "Run Data Preparation" button
   - Expandable section explaining what will happen
   - Progress spinner during execution (several minutes)
   - Immediate results display on completion

2. **Return Visit** (prep completed):
   - Green success banner with completion status
   - Comprehensive results dashboard
   - Download buttons and file information
   - Easy navigation to next step

3. **Debug Support**:
   - Optional debug info checkbox
   - File timestamp and size checking
   - Session state inspection
   - Error message details from status.json

#### Integration Points

- **UI State Management**: Uses `st.session_state['run_id']` and navigation flags
- **Backend Integration**: Calls `prep_runner.run_preparation_stage()` directly
- **File System**: Reads from `storage` API and `constants` for file paths
- **Schema Integration**: Uses `schemas` for metadata parsing and validation

#### Example User Flow

```python
# Navigate to prep page
st.session_state['current_page'] = 'prep'

# User clicks "Run Data Preparation"
# → prep_runner.run_preparation_stage(run_id) executes
# → Progress spinner shows during execution
# → Results displayed immediately on completion
# → Download buttons available
# → Navigation to next step enabled
```

#### Visual Features

- 📊 **Data metrics**: Shape, steps, encoders count with `st.metric`
- 📥 **Download section**: Organized download buttons with file info
- 📋 **Expandable details**: Cleaning steps, encoding info, model artifacts
- 🚀 **Action buttons**: Primary buttons for key actions
- ⚠️ **Status indicators**: Success, warning, error states
- 🔧 **Debug mode**: Optional technical information for developers

The UI provides a complete, user-friendly interface for the data preparation stage with comprehensive feedback, easy navigation, and robust error handling.

## Complete Task 6 Summary

**All Sub-tasks Completed Successfully:**

✅ **Sub-task 6.A**: Core Cleaning Logic (`cleaning_logic.py`)  
✅ **Sub-task 6.B**: Core Encoding Logic (`encoding_logic.py`)  
✅ **Sub-task 6.C**: Profiling Logic (`profiling_logic.py`)  
✅ **Sub-task 6.D**: Prep Stage Runner (`prep_runner.py`)  
✅ **Sub-task 6.E**: Prep UI (`ui/05_prep_page.py`)

The data preparation module is now fully functional with a complete end-to-end pipeline from raw data to ML-ready features, comprehensive profiling reports, and an intuitive user interface. 