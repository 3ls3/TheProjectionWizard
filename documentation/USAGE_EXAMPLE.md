# Step 2 Schema - Target Definition Usage Example

This document shows how to use the target definition functionality in a complete workflow.

## Basic Usage

```python
import pandas as pd
from step_2_schema.target_definition_logic import suggest_target_and_task, confirm_target_definition
from common.schemas import TargetInfo, MetadataWithTarget

# 1. Load your data
df = pd.read_csv('data/fixtures/sample_classification.csv')
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# 2. Get AI suggestions
target_col, task_type, ml_type = suggest_target_and_task(df)
print(f"AI suggests:")
print(f"  Target Column: {target_col}")
print(f"  Task Type: {task_type}")
print(f"  ML Type: {ml_type}")

# 3. User confirms (or overrides) the suggestions
confirmed_target = target_col  # User could change this
confirmed_task = task_type     # User could change this  
confirmed_ml_type = ml_type    # User could change this

# 4. Save the confirmed target definition
# Note: This requires a valid run_id from step 1 (ingest)
# success = confirm_target_definition(
#     run_id="your_run_id_here",
#     confirmed_target_column=confirmed_target,
#     confirmed_task_type=confirmed_task,
#     confirmed_target_ml_type=confirmed_ml_type
# )
```

## Schema Validation

```python
from common.schemas import TargetInfo, MetadataWithTarget
from datetime import datetime, timezone

# Create a TargetInfo object with validation
target_info = TargetInfo(
    name="target",
    task_type="classification",
    ml_type="binary_01",
    user_confirmed_at=datetime.now(timezone.utc)
)

print("TargetInfo created successfully!")
print(f"Validated target info: {target_info.model_dump()}")

# Create metadata with target info
metadata = MetadataWithTarget(
    run_id="test_run_123",
    timestamp=datetime.now(timezone.utc),
    original_filename="sample_data.csv",
    target_info=target_info,
    task_type="classification"  # Convenience field
)

print("MetadataWithTarget created successfully!")
print(f"Metadata: {metadata.model_dump()}")
```

## Expected Output

When you run the basic usage example with sample classification data:

```
Data shape: (10, 4)
Columns: ['feature1', 'feature2', 'feature3', 'target']
AI suggests:
  Target Column: target
  Task Type: classification
  ML Type: binary_01
```

When you run with sample regression data:

```
Data shape: (10, 4) 
Columns: ['feature1', 'feature2', 'feature3', 'price']
AI suggests:
  Target Column: price
  Task Type: regression
  ML Type: numeric_continuous
```

## Error Handling

The functions include comprehensive error handling:

```python
# Invalid task type will raise validation error
try:
    target_info = TargetInfo(
        name="target",
        task_type="invalid_task",  # This will fail validation
        ml_type="binary_01"
    )
except ValueError as e:
    print(f"Validation error: {e}")

# Invalid ML type will raise validation error  
try:
    target_info = TargetInfo(
        name="target", 
        task_type="classification",
        ml_type="invalid_ml_type"  # This will fail validation
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

## Integration with UI

The UI (ui/02_target_page.py) uses these functions to:

1. Load data from the run directory
2. Get AI suggestions 
3. Present options to the user
4. Validate user selections
5. Save confirmed target definition
6. Update run metadata and status

The UI provides helpful descriptions for each ML type option and shows data statistics to help users make informed decisions. 