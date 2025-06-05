# Step 2 Schema - Target Definition

This module contains the logic for Target Confirmation (Task 3) in The Projection Wizard pipeline.

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

## Example Usage

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

## ML Type Categories

- **Binary Classification**: `binary_01`, `binary_numeric`, `binary_text_labels`, `binary_boolean`
- **Multiclass Classification**: `multiclass_int_labels`, `multiclass_text_labels`, `high_cardinality_text`
- **Regression**: `numeric_continuous`
- **Other**: `unknown_type` 