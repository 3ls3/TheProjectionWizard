# The Projection Wizard

A modular end-to-end machine learning pipeline for tabular data analysis and model building. The Projection Wizard guides users from data upload to trained models with explanations, designed for data analysts and citizen data scientists.

## Project Structure

```
TheProjectionWizard/
├── step_1_ingest/          # Data ingestion and initial processing
├── step_2_schema/          # Schema validation and target confirmation
├── step_3_validation/      # Data validation with Great Expectations
├── step_4_prep/            # Data preparation and cleaning
├── step_5_automl/          # AutoML model training with PyCaret
├── step_6_explain/         # Model explainability with SHAP
├── ui/                     # Streamlit UI pages
├── common/                 # Shared utilities and schemas
├── scripts/                # Automation and testing scripts
├── data/
│   ├── runs/              # Run-specific artifacts
│   └── fixtures/          # Sample data for testing
├── requirements.txt
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
streamlit run ui/main.py
```

### Running the CLI Pipeline (Headless)

The CLI runner allows you to execute the entire pipeline from command line without UI interaction:

```bash
# Basic usage with auto-detection
python scripts/run_pipeline_cli.py --csv data/fixtures/sample_classification.csv

# Specify target column and task type
python scripts/run_pipeline_cli.py --csv data.csv --target price --task regression

# Full specification
python scripts/run_pipeline_cli.py --csv data.csv --target category --task classification --target-ml-type multiclass_text_labels
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
# Run the main smoke test
python scripts/run_smoke_test.py

# Test the CLI runner specifically
python scripts/test_cli_runner.py
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

## Contributing

1. Create feature branches for new functionality
2. Ensure all tests pass before submitting PRs
3. Follow the established code style guidelines
4. Update documentation for new features

## License

[Add license information here] 