# Module Interfaces & Integration Guide

This document defines the interfaces between different modules to enable safe parallel development.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI       │    │   CLI Runner    │
│                 │    │                 │    │                 │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Pipeline Engine     │
                    │                       │
                    │  step_1_ingest        │
                    │  step_2_schema        │
                    │  step_3_validation    │
                    │  step_4_prep          │
                    │  step_5_automl        │
                    │  step_6_explain       │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Common Services     │
                    │                       │
                    │  storage, logger,     │
                    │  schemas, constants   │
                    └───────────────────────┘
```

## 📡 API → Pipeline Interface

### **Pipeline Execution Functions**
```python
# Import these functions for API endpoints
from pipeline.step_1_ingest.ingest_logic import run_ingestion
from pipeline.step_2_schema.target_definition_logic import suggest_target_and_task, confirm_target_definition
from pipeline.step_2_schema.feature_definition_logic import suggest_initial_feature_schemas, confirm_feature_schemas
from pipeline.step_3_validation.validation_runner import run_validation_stage
from pipeline.step_4_prep.prep_runner import run_preparation_stage
from pipeline.step_5_automl.automl_runner import run_automl_stage
from pipeline.step_6_explain.explain_runner import run_explainability_stage
```

### **Function Signatures (DO NOT CHANGE)**
```python
# Stage 1: Ingestion
def run_ingestion(uploaded_file_object: Union[BinaryIO, object], base_runs_path_str: str) -> str:
    """Returns: run_id"""

# Stage 2: Schema - Target Definition
def suggest_target_and_task(run_id: str) -> Tuple[str, str]:
    """Returns: (suggested_target_column, suggested_task_type)"""

def confirm_target_definition(run_id: str, target_column: str, task_type: str, target_ml_type: str = None) -> bool:
    """Returns: success"""

# Stage 2: Schema - Feature Definition  
def suggest_initial_feature_schemas(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Returns: {column_name: {initial_dtype: str, suggested_encoding_role: str}}"""

def confirm_feature_schemas(run_id: str, feature_schemas: Dict[str, Dict[str, Any]]) -> bool:
    """Returns: success"""

# Stage 3: Validation
def run_validation_stage(run_id: str) -> bool:
    """Returns: success"""

# Stage 4: Preparation
def run_preparation_stage(run_id: str) -> bool:
    """Returns: success"""

# Stage 5: AutoML
def run_automl_stage(run_id: str, test_mode: bool = False) -> bool:
    """Returns: success"""

# Stage 6: Explanation
def run_explainability_stage(run_id: str) -> bool:
    """Returns: success"""
```

### **Status Checking Functions**
```python
from common.status_utils import get_pipeline_stage_status

# Check if a stage can run
status = get_pipeline_stage_status(run_id, "step_4_prep")
if status.can_proceed:
    # Safe to start next stage
    run_preparation_stage(run_id)
```

## 🗄️ Data Access Interface

### **Storage Functions (Safe for All)**
```python
from common import storage

# Reading data (safe for all to use)
df_original = storage.read_original_data(run_id)
df_cleaned = storage.read_cleaned_data(run_id)
metadata = storage.read_json(run_id, "metadata.json")
status = storage.read_json(run_id, "status.json")

# Writing data (coordinate if modifying shared patterns)
storage.write_json_atomic(run_id, "custom_data.json", data)
storage.write_metadata(run_id, metadata_dict)
storage.write_status(run_id, status_dict)

# Directory operations
run_dir = storage.get_run_dir(run_id)  # Creates if needed
```

### **Metadata Structure (READ-ONLY for API)**
```python
# Current metadata.json structure
{
    "run_id": str,
    "timestamp": datetime,
    "original_filename": str,
    "initial_rows": int,
    "initial_cols": int,
    "initial_dtypes": Dict[str, str],
    
    # Added by stage 2
    "target_info": {
        "name": str,
        "task_type": str,  # "classification" | "regression"
        "target_ml_type": str
    },
    "feature_schemas": Dict[str, Dict],
    
    # Added by stage 3
    "validation_info": Dict,
    
    # Added by stage 4
    "prep_info": {
        "final_shape_after_prep": List[int],
        "cleaning_steps_performed": List[str],
        "encoders_scalers_info": Dict,
        "profiling_report_path": str
    },
    
    # Added by stage 5
    "automl_info": {
        "tool_used": str,
        "best_model_name": str,
        "performance_metrics": Dict[str, float]
    },
    
    # Added by stage 6
    "explain_info": {
        "tool_used": str,
        "shap_summary_plot_path": str
    }
}
```

## 🧪 Testing Interface

### **Test Base Classes**
```python
from tests.unit.stage_tests.base_stage_test import BaseStageTest, TestResult

# Inherit from this for consistent testing
class MyStageTest(BaseStageTest):
    def __init__(self, test_run_id: str):
        super().__init__("my_stage", test_run_id)
    
    def run_test(self) -> Tuple[bool, Dict[str, Any]]:
        # Your test implementation
        pass
```

### **Test Utilities**
```python
from tests.integration.test_fixture_generator import TestFixtureGenerator

# Generate test data
generator = TestFixtureGenerator()
test_run_id = generator.setup_stage_1_ingestion("classification")
```

### **Assertion Patterns**
```python
# Standard test assertions to use
def test_stage_outputs(self):
    # 1. Validate input files exist
    input_valid, missing = self.validate_input_files(["metadata.json", "original_data.csv"])
    assert input_valid, f"Missing inputs: {missing}"
    
    # 2. Run stage
    success, result = self.run_stage_function(my_stage_function, run_id)
    assert success, f"Stage failed: {result}"
    
    # 3. Validate outputs
    output_valid, missing = self.validate_output_files(["status.json"])
    assert output_valid, f"Missing outputs: {missing}"
    
    # 4. Check status file
    status_valid, status_data = self.check_status_file("my_stage", "completed")
    assert status_valid, f"Status check failed: {status_data}"
```

## 🔄 Pipeline → Common Interface

### **Logging Patterns (Standardized)**
```python
from common import logger

# Get stage-specific logger
log = logger.get_stage_logger(run_id, "step_X_name")
structured_log = logger.get_stage_structured_logger(run_id, "step_X_name")

# Standard logging calls
log.info("Starting stage X processing")
log.error(f"Stage X failed: {error}")

# Structured events (for monitoring)
logger.log_structured_event(
    structured_log,
    "stage_started",
    {"stage": "step_X", "input_size": data_shape},
    "Stage X processing started"
)
```

### **Constants Usage (Read-Only)**
```python
from common import constants

# Use these constants, don't modify them
stage_name = constants.PREP_STAGE
status_file = constants.STATUS_FILENAME
pipeline_stages = constants.PIPELINE_STAGES

# If you need new constants, ADD them, don't modify existing
```

### **Schema Validation (Required)**
```python
from common import schemas

# Always validate data with Pydantic models
target_info = schemas.TargetInfo(**target_dict)
metadata = schemas.BaseMetadata(**metadata_dict)
status = schemas.StageStatus(**status_dict)

# If you need new schemas, ADD them, don't modify existing
```

## 🔌 Extension Points

### **Adding New API Endpoints**
```python
# api/routes/my_new_feature.py
from fastapi import APIRouter
from common import storage, constants

router = APIRouter()

@router.get("/my-endpoint/{run_id}")
async def my_endpoint(run_id: str):
    # Use established patterns
    if not storage.get_run_dir(run_id).exists():
        raise HTTPException(404, "Run not found")
    
    # Use common functions
    metadata = storage.read_json(run_id, constants.METADATA_FILENAME)
    return {"data": metadata}

# Register in api/main.py
app.include_router(my_new_feature.router, prefix="/api/v1", tags=["my_feature"])
```

### **Adding New Pipeline Features**
```python
# pipeline/step_X_name/my_enhancement.py
from common import logger, storage, constants

def enhanced_function(run_id: str, **kwargs) -> bool:
    # Follow established patterns
    log = logger.get_stage_logger(run_id, constants.CURRENT_STAGE)
    
    try:
        # Your enhancement logic
        result = do_enhancement()
        
        # Update metadata following pattern
        metadata = storage.read_metadata(run_id)
        metadata["my_enhancement_info"] = result
        storage.write_metadata(run_id, metadata)
        
        return True
    except Exception as e:
        log.error(f"Enhancement failed: {e}")
        return False
```

### **Adding New Tests**
```python
# tests/unit/stage_tests/test_my_stage.py
from tests.unit.stage_tests.base_stage_test import BaseStageTest

class MyStageTest(BaseStageTest):
    def __init__(self, test_run_id: str):
        super().__init__("my_stage", test_run_id)
    
    def run_test(self) -> Tuple[bool, Dict[str, Any]]:
        # Follow established test patterns
        validation_results = {}
        
        # 1. Input validation
        # 2. Run stage
        # 3. Output validation  
        # 4. Specific validations
        
        return success, validation_results
```

## ⚠️ Breaking Change Warnings

### **DO NOT MODIFY THESE WITHOUT COORDINATION**:
- Function signatures in pipeline stages
- Metadata structure keys (can add, don't remove/rename)
- File naming conventions in constants.py
- Pydantic model field names in schemas.py
- Storage function interfaces

### **SAFE TO EXTEND**:
- Add new API endpoints
- Add new pipeline functions
- Add new test cases
- Add new configuration options
- Add new utility functions

### **REQUIRES DISCUSSION**:
- New dependencies in requirements.txt
- Changes to shared constants
- Modifications to core schemas
- New file formats or naming conventions

---

**Last Updated**: [Current Date]
**Version**: 1.0
**Next Review**: [Weekly with team]

*This document should be updated whenever interfaces change.* 