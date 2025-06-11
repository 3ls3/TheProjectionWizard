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
│   ├── pages/              # Streamlit UI pages (01-08)
│   └── main.py             # Main application entry point
├── api/                    # FastAPI REST API (optional)
│   ├── routes/             # API route definitions
│   ├── utils/              # API utility functions
│   └── main.py             # FastAPI application entry point
├── common/                 # Shared utilities and schemas
│   ├── constants.py        # Project-wide constants
│   ├── schemas.py          # Pydantic data models
│   ├── storage.py          # File I/O and atomic operations
│   ├── logger.py           # Structured logging system
│   └── utils.py            # General utility functions
├── scripts/                # Automation and testing scripts
│   ├── bash/               # Shell scripts for deployment
│   └── python/             # Python scripts for testing and CLI
├── tests/                  # Test suite
│   ├── unit/               # Unit tests for individual stages
│   ├── integration/        # Integration tests for full pipeline
│   ├── fixtures/           # Test fixtures and data
│   ├── data/               # Test runs
│   └── reports/            # Test reports
├── data/
│   ├── runs/              # Run-specific artifacts
    └── fixtures/          # Sample data for testing
├── docs/                   # Project documentation
│   └── archive/           # Archived documentation
├── Dockerfile             # Container configuration
├── Makefile              # Build automation
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project configuration
└── README.md
```

## Pipeline Flow

1. **Data Ingestion**: Upload CSV and initial analysis
2. **Target Definition**: Define target variable and confirm task type
3. **Schema Confirmation**: Define feature types and data schemas
4. **Data Validation**: Validate data quality with Great Expectations
5. **Data Preparation**: Clean and encode features
6. **AutoML**: Train models with PyCaret
7. **Model Explanation**: Generate SHAP explanations
8. **Results**: View and download results

## Setup

### ⚠️ Python Version Requirements

**IMPORTANT**: This project requires **Python 3.10.x or 3.11.x ONLY**.

- ❌ Python 3.12+ is **NOT supported** (PyCaret compatibility)
- ❌ Python 3.9 and below are **NOT supported**
- ✅ Python 3.10.6 is **recommended**

### Prerequisites
- Python 3.10.6 or 3.11.x (see requirements above)
- pyenv (recommended for Python version management)
- pip

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd TheProjectionWizard
```

2. **Check Python compatibility** (REQUIRED):
```bash
# Run this BEFORE creating virtual environment
python setup_check.py
```

3. **Set correct Python version** (if needed):
```bash
# Install Python 3.10.6 using pyenv (if not already installed)
pyenv install 3.10.6
pyenv local 3.10.6

# Verify the change
python setup_check.py
```

4. **Create and activate virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

5. **Install dependencies**:
```bash
pip install -r requirements.txt
```

6. **Verify installation**:
```bash
# Use convenience script for full verification
./activate_env.sh
```

## Usage

### Running the Streamlit App
```bash
streamlit run app/main.py
```

The application will be available at `http://localhost:8501`

### Running the FastAPI Server (Optional)

The project includes an optional REST API for programmatic access:

```bash
# Production mode (cloud logging to stdout)
./start_backend.sh

# Development mode (local file logging enabled)
./start_backend_dev.sh

# Or manually with custom settings
export LOCAL_DEV_LOGGING=true  # Enable local file logging
uvicorn api.main:app --reload
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

**Key API Endpoints:**
- `GET /api/v1/runs` - List all pipeline runs
- `GET /api/v1/runs/{run_id}/info` - Get run information
- `GET /api/v1/feature_suggestions` - Get ML feature analysis

#### Local Development Logging

For pipeline debugging, enable local file logging with `LOCAL_DEV_LOGGING=true`:

```bash
# Enable local logging and start server
export LOCAL_DEV_LOGGING=true
./start_backend.sh

# Or use the dedicated development script
./start_backend_dev.sh
```

**Local logging features:**
- **Log files saved to**: `data/runs/{run_id}/logs/`
- **Human-readable logs**: `{stage}.log` (e.g., `automl.log`, `validation.log`)
- **Structured JSON logs**: `{stage}_structured.jsonl` 
- **Console output**: Real-time feedback with 🔮 prefixes
- **Persistent**: Log files survive server restarts

**Debugging utilities:**
```bash
# List all runs with logs
python scripts/python/debug_logs.py --list

# View logs for a specific run
python scripts/python/debug_logs.py --run abc123

# View specific stage logs
python scripts/python/debug_logs.py --run abc123 --stage automl

# Tail logs in real-time
python scripts/python/debug_logs.py --tail abc123

# View structured JSON logs
python scripts/python/debug_logs.py --run abc123 --json

# Clean up old logs
python scripts/python/debug_logs.py --clean
```

### Docker Deployment

The project supports containerized deployment with both FastAPI backend and React frontend using Docker Compose.

#### Architecture
- **Backend**: FastAPI server serving REST API endpoints (port 8000)
- **Frontend**: React/Vite application for modern UI (port 3000)
- **Communication**: Frontend communicates with backend via HTTP API calls

#### Quick Start
```bash
# Build and start both services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

**Access the application:**
- **Frontend**: http://localhost:3000 (Main UI)
- **Backend API**: http://localhost:8000 (FastAPI docs at /docs)

#### Development Workflow
```bash
# Start services
docker-compose up

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up --build
```

#### Environment Configuration
Configuration is handled via `.env` file:
```bash
# API Settings
VITE_API_URL=http://localhost:8000
BACKEND_PORT=8000
FRONTEND_PORT=3000

# Development settings
NODE_ENV=development
PYTHONUNBUFFERED=1
```

#### Service Architecture
- **backend**: FastAPI service with data volume mounting for persistent runs
- **frontend**: React application with Vite dev server
- **Networks**: Isolated Docker network for service communication
- **Health checks**: Backend health monitoring with automatic frontend dependency

#### Data Persistence
Upload data and pipeline runs are persisted via Docker volume:
```bash
# Data is mounted from host ./data to container /app/data
# Pipeline runs stored in ./data/runs/{run_id}/
```

#### Cloud Deployment (Optional)
For cloud deployment, update `.env` with your cloud endpoints:
```bash
VITE_API_URL=https://your-api-domain.com
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
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

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
- **Quality Gates**: Built-in validation prevents bad data from propagating downstream

### Technology Stack

**Core Technologies:**
- **Frontend**: Streamlit (UI), FastAPI (REST API)
- **ML/AI**: PyCaret (AutoML), Scikit-Learn, SHAP (Explainability)
- **Data Processing**: Pandas, Great Expectations (Validation), YData Profiling
- **Infrastructure**: Docker, Google Cloud Run
- **Development**: pytest, Black, Flake8, MyPy

### Inter-Module Communication

Each run creates a unique directory under `data/runs/<run_id>/` containing:
- `metadata.json`: Run configuration and results (evolves through pipeline)
- `status.json`: Current pipeline status and error tracking
- `original_data.csv`: Uploaded data
- `cleaned_data.csv`: Processed ML-ready data
- `validation.json`: Data validation results
- `model/`: Trained model artifacts and encoders
- `plots/`: Generated visualizations and reports

### Run Index

The system maintains a CSV-based run index at `data/runs/index.csv` that logs key details of each pipeline run:
- `run_id`: Unique identifier for the run
- `timestamp`: When the run was initiated
- `original_filename`: Name of the uploaded CSV file
- `status`: Final status (e.g., "Completed Successfully", "Failed at AutoML")

This index enables tracking of all pipeline executions and can be used for run history analysis.

### Testing Strategy

The project includes multiple levels of testing:

1. **Unit Tests**: Individual module tests in each `pipeline/step_X/` directory
2. **Integration Tests**: End-to-end pipeline testing via `test_cli_runner.py`
3. **Common Utilities Tests**: Core functionality testing via `test_common.py`
4. **Health Checks**: Comprehensive project validation via `health_check.py`

Each step directory contains its own test files to verify the logic independent of the UI.

## Project Structure Details

### Pipeline Modules
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
- `common/logger.py`: Run-scoped structured logging utilities
- `common/utils.py`: General utility functions

### UI Components
- `app/pages/01_upload_page.py`: Data upload interface
- `app/pages/02_target_page.py`: Target variable definition
- `app/pages/03_schema_page.py`: Feature schema confirmation
- `app/pages/04_validation_page.py`: Data validation interface
- `app/pages/05_prep_page.py`: Data preparation interface
- `app/pages/06_automl_page.py`: Model training interface
- `app/pages/07_explain_page.py`: Model explanation interface
- `app/pages/08_results_page.py`: Results and download interface
- `app/main.py`: Main Streamlit application entry point

### API Endpoints (Optional)
- `api/routes/`: REST API route definitions
- `api/utils/`: API-specific utilities
- `api/main.py`: FastAPI application with health checks and core endpoints

## Key Features

- **Progressive Workflow**: Guided step-by-step ML pipeline
- **Quality Assurance**: Built-in data validation and error handling
- **Model Explainability**: SHAP-based feature importance and explanations
- **Deployment Ready**: Docker containerization and cloud deployment
- **Dual Interface**: Both UI (Streamlit) and programmatic (CLI/API) access
- **Comprehensive Logging**: Structured logging with error tracking
- **Testing Coverage**: Unit, integration, and health check testing

## Contributing

1. Create feature branches for new functionality
2. Ensure all tests pass before submitting PRs
3. Follow the established code style guidelines (Black, Flake8, MyPy)
4. Update documentation for new features
5. Test both UI and CLI interfaces for any changes
6. Activate the virtual environment (`.venv`) before development

## License

[Add license information here] 