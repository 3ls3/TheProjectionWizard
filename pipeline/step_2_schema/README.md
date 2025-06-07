# Step 2 Schema - Target Definition & Feature Schema Assist

This module contains the logic for Target Confirmation (Task 3) and Key Feature Schema Assist & Confirmation (Task 4) in The Projection Wizard pipeline.

## Functions

### `suggest_target_and_task(df: pd.DataFrame)`

Analyzes a DataFrame to suggest the most likely target column and task type based on heuristics.

**Parameters:**
- `df`: The pandas DataFrame loaded from original_data.csv

**Returns:**
- Tuple of (target_column_name, task_type, target_ml_type)

**Heuristics:**
- **Target Column**: Prioritizes columns with names like 'target', 'label', 'class', 'outcome', 'output', 'result', 'y', 'response'. Falls back to the last column if no obvious match.
- **Task Type**: 
  - Classification for categorical data, boolean data, or numeric data with few unique values
  - Regression for continuous numeric data with many unique values
- **ML Type**: Specific encoding format like 'binary_01', 'binary_text_labels', 'multiclass_int_labels', 'numeric_continuous', etc.

### `confirm_target_definition(run_id, confirmed_target_column, confirmed_task_type, confirmed_target_ml_type)`

Confirms the target definition and updates the run's metadata.json and status.json files.

**Parameters:**
- `run_id`: The ID of the current run
- `confirmed_target_column`: User-confirmed target column name
- `confirmed_task_type`: User-confirmed task type ('classification' or 'regression')
- `confirmed_target_ml_type`: User-confirmed ML-ready type

**Returns:**
- `True` if successful, `False` otherwise

**Side Effects:**
- Updates `metadata.json` with target_info section
- Updates `status.json` with stage completion status
- Writes to run-specific log file

## Feature Schema Functions

### `identify_key_features(df_original, target_info, num_features_to_surface=7)`

Identifies potentially important features using basic importance metrics.

**Parameters:**
- `df_original`: The original pandas DataFrame (loaded from original_data.csv)
- `target_info`: Dictionary with target column info (name, task_type, ml_type) from metadata.json
- `num_features_to_surface`: How many top features to suggest (default: 7)

**Returns:**
- List of top N feature column names

**Implementation:**
- Uses mutual information for classification tasks
- Uses f-regression for regression tasks
- Performs minimal stable cleaning for metric calculation stability
- Falls back to correlation-based methods if advanced metrics fail

### `suggest_initial_feature_schemas(df)`

Suggests initial data types and encoding roles for all columns based on heuristics.

**Parameters:**
- `df`: The original pandas DataFrame

**Returns:**
- Dictionary where keys are column names and values are dicts with 'initial_dtype' and 'suggested_encoding_role'

**Encoding Role Heuristics:**
- `numeric-continuous`: For numeric columns with many unique values
- `categorical-nominal`: For object/categorical columns with low cardinality
- `boolean`: For boolean dtype columns
- `datetime`: For datetime columns
- `text`: For high cardinality object columns or ID-like columns

### `confirm_feature_schemas(run_id, user_confirmed_schemas, all_initial_schemas)`

Confirms feature schemas and updates metadata.json with feature schema information.

**Parameters:**
- `run_id`: The ID of the current run
- `user_confirmed_schemas`: Dictionary of user-confirmed schemas (only for reviewed columns)
- `all_initial_schemas`: Full dictionary of initial suggestions for all columns

**Returns:**
- `True` if successful, `False` otherwise

**Side Effects:**
- Updates `metadata.json` with feature_schemas section
- Updates `status.json` with stage completion status
- Writes to run-specific log file

## Example Usage

### Target Definition
```python
import pandas as pd
from step_2_schema.target_definition_logic import suggest_target_and_task, confirm_target_definition

# Load data
df = pd.read_csv('path/to/original_data.csv')

# Get suggestions
target_col, task_type, ml_type = suggest_target_and_task(df)
print(f"Suggested target: {target_col}, Task: {task_type}, ML Type: {ml_type}")

# User confirms (in real usage, this would come from UI)
confirmed_target = target_col  # or user override
confirmed_task = task_type     # or user override
confirmed_ml_type = ml_type    # or user override

# Confirm and save
success = confirm_target_definition(
    run_id="your_run_id",
    confirmed_target_column=confirmed_target,
    confirmed_task_type=confirmed_task,
    confirmed_target_ml_type=confirmed_ml_type
)
```

### Feature Schema Assist
```python
import pandas as pd
from step_2_schema.feature_definition_logic import (
    identify_key_features, 
    suggest_initial_feature_schemas, 
    confirm_feature_schemas
)

# Load data
df = pd.read_csv('path/to/original_data.csv')

# Get initial schema suggestions for all columns
all_schemas = suggest_initial_feature_schemas(df)

# Identify key features based on target
target_info = {'name': 'target_column', 'task_type': 'classification', 'ml_type': 'binary_01'}
key_features = identify_key_features(df, target_info, num_features_to_surface=5)

# User reviews and confirms schemas (in real usage, this would come from UI)
user_confirmed = {
    'feature1': {'final_dtype': 'float64', 'final_encoding_role': 'numeric-continuous'},
    'feature2': {'final_dtype': 'object', 'final_encoding_role': 'categorical-ordinal'}
}

# Confirm and save
success = confirm_feature_schemas(
    run_id="your_run_id",
    user_confirmed_schemas=user_confirmed,
    all_initial_schemas=all_schemas
)
```

## ML Type Categories

- **Binary Classification**: `binary_01`, `binary_numeric`, `binary_text_labels`, `binary_boolean`
- **Multiclass Classification**: `multiclass_int_labels`, `multiclass_text_labels`, `high_cardinality_text`
- **Regression**: `numeric_continuous`
- **Other**: `unknown_type` 