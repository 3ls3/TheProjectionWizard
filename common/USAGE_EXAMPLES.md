# Common Utilities Usage Examples

This document provides examples of how to use the common utilities in The Projection Wizard.

## Stage Constants

The pipeline stages are now mapped to actual directory names for better clarity and robustness:

```python
from common.constants import (
    INGEST_STAGE, SCHEMA_STAGE, VALIDATION_STAGE, 
    PREP_STAGE, AUTOML_STAGE, EXPLAIN_STAGE, RESULTS_STAGE
)

# Stage constants map to actual directory names
print(INGEST_STAGE)        # "step_1_ingest"
print(SCHEMA_STAGE)        # "step_2_schema"
print(VALIDATION_STAGE)    # "step_3_validation"
print(PREP_STAGE)          # "step_4_prep"
print(AUTOML_STAGE)        # "step_5_automl"
print(EXPLAIN_STAGE)       # "step_6_explain"
print(RESULTS_STAGE)       # "results"
```

## Stage Helper Functions

```python
from common.constants import (
    get_stage_directory, get_stage_display_name, is_valid_stage
)

# Get directory name for a stage
directory = get_stage_directory(INGEST_STAGE)  # "step_1_ingest"
ui_directory = get_stage_directory(RESULTS_STAGE)  # "ui"

# Get human-readable display names
display_name = get_stage_display_name(INGEST_STAGE)  # "Data Ingestion"

# Validate stage names
is_valid = is_valid_stage("step_1_ingest")  # True
is_valid = is_valid_stage("invalid_stage")  # False
```

## Creating Status Objects

```python
from common.schemas import StageStatus
from common.constants import INGEST_STAGE

# Create a stage status with validated stage name
status = StageStatus(
    stage=INGEST_STAGE,  # Uses constant, validated against PIPELINE_STAGES
    status="completed",
    message="Data successfully ingested"
)
```

## Working with Run Storage

```python
from common.utils import generate_run_id
from common.storage import write_metadata, read_metadata, write_status
from common.logger import get_logger
from common.constants import INGEST_STAGE

# Generate a new run ID
run_id = generate_run_id()  # "2025-06-05T110343Z_fb74454d"

# Create a logger for the run
logger = get_logger(run_id, "ingest_stage", "INFO")

# Write metadata
metadata = {
    "run_id": run_id,
    "stage": INGEST_STAGE,
    "original_filename": "data.csv"
}
write_metadata(run_id, metadata)

# Write stage status
status_data = {
    "stage": INGEST_STAGE,
    "status": "completed",
    "message": "Ingestion successful"
}
write_status(run_id, status_data)

# Log with run context
logger.info(f"Completed {INGEST_STAGE} stage")
```

## Benefits of This Approach

1. **Consistency**: Stage names in logs/status match actual directory structure
2. **Type Safety**: Pydantic validation ensures only valid stages are used
3. **Maintainability**: Single source of truth for stage definitions
4. **Clarity**: Human-readable display names for UI components
5. **Robustness**: Helper functions prevent typos and provide validation

## Migration from Old Approach

If you have existing code using string literals like `"ingest"`, update to use constants:

```python
# Old approach (error-prone)
stage = "ingest"
status = StageStatus(stage="ingest", status="completed")

# New approach (robust)
from common.constants import INGEST_STAGE
stage = INGEST_STAGE
status = StageStatus(stage=INGEST_STAGE, status="completed")
``` 