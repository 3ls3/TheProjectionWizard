# The Projection Wizard

A modular end-to-end machine learning pipeline for tabular data analysis and model building. The Projection Wizard guides users from data upload to trained models with explanations, designed for data analysts and citizen data scientists.

## Project Structure

```
TheProjectionWizard/
├── pipeline/               # Data processing pipeline
│   ├── step_1_ingest/      # Data ingestion and initial processing
│   ├── step_2_schema/      # Schema validation and target confirmation
│   ├── step_3_validation/  # Data validation with Great Expectations
│   ├── step_4_prep/        # Data preparation and cleaning
│   ├── step_5_automl/      # AutoML model training with PyCaret
│   └── step_6_explain/     # Model explainability with SHAP
├── app/                    # Streamlit application
│   ├── pages/              # Streamlit UI pages
│   └── main.py             # Main application entry point
├── common/                 # Shared utilities and schemas
├── scripts/                # Automation and testing scripts
│   ├── bash/               # Shell scripts for deployment
│   └── python/             # Python scripts for testing and CLI
├── tests/                  # Test suite
│   ├── unit/               # Unit tests for individual stages
│   ├── integration/        # Integration tests for full pipeline
│   ├── fixtures/           # Test fixtures and data
│   ├── data/               # Test data
│   └── reports/            # Test reports
├── data/
│   ├── runs/              # Run-specific artifacts
│   └── fixtures/          # Sample data for testing
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Pipeline Flow

1. **Data Ingestion**: Upload CSV and initial analysis
2. **Schema Confirmation**: Define target variable and data types
3. **Data Validation**: Validate data quality with Great Expectations
4. **Data Preparation**: Clean and encode features
5. **AutoML**: Train models with PyCaret
6. **Model Explanation**: Generate SHAP explanations
7. **Results**: View and download results

## Setup

### Prerequisites
- Python 3.10+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TheProjectionWizard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App
```bash
streamlit run app/main.py
```

### Docker Deployment

The project includes Docker support for containerized deployment to Google Cloud Run.

#### Local Docker Testing
```bash
# Build and run locally
make run-docker
# Opens at http://localhost:8501
```

#### Cloud Deployment
1. **Setup GCP Project** (one-time):
```bash
# Initialize GCP project and enable APIs
make gcp-init
```

2. **Configure Docker authentication** (one-time):
```bash
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

3. **Push image to Artifact Registry**:
```bash
# Push with default tag (wizard:latest)
make push-image

# Push with custom tag
IMAGE_TAG=wizard:v1.2.3 make push-image
```

4. **Deploy to Cloud Run**:
```bash
# Deploy the pushed image to Cloud Run
make deploy
```
This will deploy your application as a public web service with:
- 1 CPU, 2Gi memory
- 15 minute timeout
- Max 2 instances 
- Public access (no authentication required)

#### Environment Configuration
Copy `.env.example` to `.env` and set your GCP project details:
```bash
PROJECT_ID=your-gcp-project-id
REGION=europe-west1
IMAGE_TAG=wizard:v0.1  # Optional, defaults to wizard:latest
```

### Running the CLI Pipeline (Headless)

The CLI runner allows you to execute the entire pipeline from command line without UI interaction:

```bash
# Basic usage with auto-detection
python scripts/python/run_pipeline_cli.py --csv data/fixtures/sample_classification.csv

# Specify target column and task type
python scripts/python/run_pipeline_cli.py --csv data.csv --target price --task regression

# Full specification
python scripts/python/run_pipeline_cli.py --csv data.csv --target category --task classification --target-ml-type multiclass_text_labels
```

**CLI Options:**
- `--csv`: Path to input CSV file (required)
- `--target`: Target column name (optional, auto-detected if not provided)
- `--task`: Task type - `classification` or `regression` (optional, auto-detected)
- `--target-ml-type`: ML-ready type for target encoding (optional, auto-detected)
- `--output-dir`: Base directory for outputs (default: `data/runs`)

The CLI runner will:
1. Execute all pipeline stages sequentially
2. Generate all standard artifacts (cleaned data, models, reports)
3. Update the run index with results
4. Provide detailed progress output and error reporting

### Running Tests
```bash
# Run health check (comprehensive project validation)
python scripts/python/health_check.py

# Test the CLI runner functionality
python scripts/python/test_cli_runner.py

# Test common utilities
python scripts/python/test_common.py

# Individual component tests (examples)
python pipeline/step_1_ingest/test_ingest_logic.py
python pipeline/step_3_validation/test_ge_logic.py
```

## Development

### Code Style
This project uses:
- Black for code formatting
- Flake8 for linting
- MyPy for type checking

Run formatting and linting:
```bash
black .
flake8 .
mypy .
```

### Architecture Principles

- **Modularity**: Each pipeline stage is independent and testable
- **File-based Communication**: Stages communicate via JSON artifacts
- **Atomic Operations**: All file writes are atomic to prevent corruption
- **Immutable Artifacts**: Run artifacts are never modified, only appended

### Inter-Module Communication

Each run creates a unique directory under `data/runs/<run_id>/` containing:
- `metadata.json`: Run configuration and results
- `status.json`: Current pipeline status
- `original_data.csv`: Uploaded data
- `cleaned_data.csv`: Processed data
- `validation.json`: Data validation results
- `model/`: Trained model artifacts
- `plots/`: Generated visualizations

### Run Index

The system maintains a CSV-based run index at `data/runs/index.csv` that logs key details of each pipeline run:
- `run_id`: Unique identifier for the run
- `timestamp`: When the run was initiated
- `original_filename`: Name of the uploaded CSV file
- `status`: Final status (e.g., "Completed Successfully", "Failed at AutoML")

This index enables tracking of all pipeline executions and can be used for run history analysis.

### Testing Strategy

The project includes multiple levels of testing:

1. **Component Tests**: Individual module tests (e.g., `test_ingest_logic.py`, `test_ge_logic.py`)
2. **CLI Integration Test**: End-to-end pipeline testing via `test_cli_runner.py`
3. **Common Utilities Test**: Core functionality testing via `test_common.py`

Each step directory contains its own test files to verify the logic independent of the UI.

## Project Structure Details

### Step Modules
- `pipeline/step_1_ingest/`: Handles CSV upload and initial data analysis
- `pipeline/step_2_schema/`: Target definition and feature schema confirmation
- `pipeline/step_3_validation/`: Great Expectations data validation
- `pipeline/step_4_prep/`: Data cleaning and feature encoding
- `pipeline/step_5_automl/`: PyCaret model training and evaluation
- `pipeline/step_6_explain/`: SHAP-based model explainability

### Common Utilities
- `common/constants.py`: Project-wide constants and configuration
- `common/schemas.py`: Pydantic models for data validation
- `common/storage.py`: File I/O and atomic operations
- `common/logger.py`: Run-scoped logging utilities
- `common/utils.py`: General utility functions

### UI Pages
- `app/pages/01_upload_page.py` through `app/pages/08_results_page.py`: Streamlit page components
- `app/main.py`: Main Streamlit application entry point

## Contributing

1. Create feature branches for new functionality
2. Ensure all tests pass before submitting PRs
3. Follow the established code style guidelines
4. Update documentation for new features
5. Test both UI and CLI interfaces for any changes

## License

[Add license information here] 