# Repository Structure & File Organization

## Project Root Structure
```
TheProjectionWizard/
├── api/                          # FastAPI backend
│   ├── main.py                   # FastAPI application entry point & CORS config
│   ├── routes/                   # API endpoint definitions
│   │   ├── pipeline.py           # CRITICAL: Main ML pipeline endpoints (84KB, 2334 lines)
│   │   ├── schema.py             # Pydantic models for API contracts (18KB, 576 lines)
│   │   └── endpoints.py          # Schema-related endpoints (6.4KB, 207 lines)
│   ├── utils/                    # API utilities
│   │   ├── gcs_utils.py          # GCS operations and file management (12KB, 357 lines)
│   │   ├── io_helpers.py         # Data loading utilities (7.7KB, 275 lines)
│   │   └── __init__.py           # API utilities initialization
│   └── data/                     # API data handling
│       └── runs/                 # Runtime data storage
├── app/                          # Streamlit frontend
├── common/                       # Shared utilities
│   ├── constants.py              # System constants
│   ├── logger.py                 # Logging utilities
│   └── storage.py                # File/GCS operations
├── data/                         # Data storage
│   ├── fixtures/                 # Test/sample datasets
│   │   ├── house_prices.csv      # Main regression training dataset
│   │   └── loan_approval.csv     # Classification training dataset
│   └── runs/                     # Pipeline execution outputs
├── pipeline/                     # ML Pipeline (7 steps)
│   ├── orchestrator.py           # Pipeline orchestration
│   ├── step_1_ingest/            # Data ingestion
│   ├── step_2_schema/            # Feature schema definition
│   ├── step_3_validation/        # Data validation
│   ├── step_4_prep/              # Data preparation
│   ├── step_5_automl/            # Model training
│   ├── step_6_explain/           # Model explainability
│   └── step_7_predict/           # Prediction system
├── tests/                        # Test framework
│   ├── fixtures/                 # Test data generation
│   ├── integration/              # Integration tests
│   ├── unit/                     # Unit tests
│   └── reports/                  # Test reports
└── scripts/                      # Utility scripts
```

## Critical Files for Testing

### Pipeline Core Files
- `pipeline/orchestrator.py` - Orchestrates all pipeline steps
- `pipeline/step_7_predict/column_mapper.py` - Input transformation for predictions
- `pipeline/step_7_predict/predict_logic.py` - Core prediction logic
- `pipeline/step_4_prep/encoding_logic.py` - Feature encoding and scaling

### API Core Files (CRITICAL)
- `api/main.py` - FastAPI app initialization, CORS config, router registration
- `api/routes/pipeline.py` - **CRITICAL** Main pipeline endpoints (84KB):
  - `/api/upload` - CSV file upload
  - `/api/target-suggestion` - Target column AI suggestions  
  - `/api/confirm-target` - Target column confirmation
  - `/api/feature-suggestion` - Feature schema AI suggestions
  - `/api/confirm-features` - Feature schema confirmation & pipeline start
  - `/api/status` - Pipeline execution status
  - `/api/results` - Final pipeline results
  - `/api/prediction-schema` - Basic prediction input schema
  - `/api/predict` - **WORKING**: Returns correct ~$425k predictions
  - `/api/prediction-schema-enhanced` - Enhanced schema with sliders/dropdowns
  - `/api/predict/single` - **BROKEN**: Returns incorrect ~$150M predictions
  - `/api/predict/explain/{prediction_id}` - SHAP explanations
  - `/api/predict/batch` - Batch predictions
  - `/api/predict/compare` - Prediction comparisons
  - `/api/download/{run_id}/{filename}` - File downloads
- `api/routes/schema.py` - **CRITICAL** Pydantic API contracts (18KB):
  - All request/response models for frontend-backend communication
  - Enhanced prediction schemas with UI metadata
  - Batch prediction and explanation models
- `api/routes/endpoints.py` - Schema-related endpoints (6.4KB):
  - `/api/v1/feature_suggestions` - Feature analysis endpoints
  - `/api/v1/runs/{run_id}/info` - Run information
  - `/api/v1/runs` - List all runs

### API Utilities (CRITICAL)
- `api/utils/gcs_utils.py` - **CRITICAL** GCS operations (12KB):
  - All Google Cloud Storage interactions
  - File upload/download for pipeline artifacts
  - Run-specific file management
- `api/utils/io_helpers.py` - **CRITICAL** Data loading (7.7KB):
  - CSV and metadata loading from GCS
  - Run validation and file existence checks
  - Legacy compatibility with local storage

### Testing Infrastructure
- `tests/fixtures/fixture_generator.py` - Creates test data
- `tests/unit/stage_tests/base_stage_test.py` - Base test class
- `tests/integration/test_orchestrator.py` - Integration testing framework

### Common Utilities
- `common/constants.py` - All system constants
- `common/logger.py` - Structured logging
- `common/storage.py` - File operations

## Data Flow Locations

### Training Data Path
```
data/fixtures/house_prices.csv → step_1_ingest → step_2_schema → step_3_validation 
→ step_4_prep → step_5_automl → step_6_explain → GCS Storage
```

### Prediction Data Path
```
User Input → API Endpoint → step_7_predict/column_mapper.py → Model → Response
```

### Critical GCS Artifacts
- `{run_id}/original_data.csv` - Raw training data
- `{run_id}/cleaned_data.csv` - Processed training data
- `{run_id}/pycaret_pipeline.pkl` - Trained model
- `{run_id}/column_mapping.json` - Feature transformation rules
- `{run_id}/metadata.json` - Run configuration
- `{run_id}/scalers/` - StandardScaler objects

## File Naming Conventions

### Test Files
- `test_phase_{N}_{description}.py` - Phase-based testing
- `test_{step_name}_{component}.py` - Component testing
- `test_{api_endpoint_name}.py` - API testing

### Pipeline Artifacts
- `{run_id}/` - All artifacts for a pipeline run
- `status.json` - Current pipeline status
- `pipeline.log` - Detailed execution logs

## Import Patterns

### Pipeline Steps
```python
from pipeline.step_1_ingest import ingest_logic
from pipeline.step_4_prep import encoding_logic
from pipeline.step_7_predict import column_mapper, predict_logic
```

### Common Utilities
```python
from common import constants, logger, storage
from api.utils.gcs_utils import PROJECT_BUCKET_NAME
```

### Testing Framework
```python
from tests.fixtures.fixture_generator import TestFixtureGenerator
from tests.unit.stage_tests.base_stage_test import BaseStageTest, TestResult
```

## Directory Access Patterns

### Creating Test Runs
```python
test_run_dir = Path("data/test_runs") / test_run_id
test_run_dir.mkdir(parents=True, exist_ok=True)
```

### Accessing Pipeline Outputs
```python
run_dir = Path("data/runs") / run_id
model_path = run_dir / "pycaret_pipeline.pkl"
metadata_path = run_dir / "metadata.json"
```

### Test Reports
```python
report_dir = Path("tests/reports")
report_path = report_dir / f"test_report_{timestamp}.json"
```

## Environment Setup

### Virtual Environment
```bash
source .venv/bin/activate  # Always activate before running
```

### Key Dependencies
- `pandas` - Data manipulation
- `pycaret` - AutoML framework
- `fastapi` - API framework
- `streamlit` - Frontend framework
- `shap` - Model explainability
- `google-cloud-storage` - GCS operations

## Critical Constants

### From `common/constants.py`
- `ORIGINAL_DATA_FILENAME = "original_data.csv"`
- `CLEANED_DATA_FILENAME = "cleaned_data.csv"`
- `METADATA_FILENAME = "metadata.json"`
- `STATUS_FILENAME = "status.json"`
- `MODEL_DIR = "model"`
- `PLOTS_DIR = "plots"`

### GCS Configuration
- `PROJECT_BUCKET_NAME` - Main GCS bucket
- All pipeline artifacts stored in GCS with run_id prefix

## Testing Context

### Existing Test Infrastructure
- Comprehensive fixture generation system
- Base test classes for consistency
- Integration test orchestrator
- Structured logging for test results

### Known Working vs Broken
- ✅ `/api/predict` endpoint returns correct ~$425k predictions
- ❌ `/api/predict/single` endpoint returns incorrect ~$150M predictions
- ✅ StandardScaler fix implemented in `column_mapper.py`
- ❌ Frontend still receives incorrect values

This structure provides the foundation for implementing any isolated testing task within the repository. 