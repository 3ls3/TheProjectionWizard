# The Projection Wizard API

REST API for The Projection Wizard ML pipeline functionality.

## Quick Start

### Running the API

From the project root directory:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install FastAPI if not already installed
pip install fastapi uvicorn

# Run the API server
cd api
uvicorn main:app --reload
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Alternative: Run from project root

```bash
# From project root
uvicorn api.main:app --reload
```

## Available Endpoints

### Core Endpoints

- **GET** `/` - API information and version
- **GET** `/health` - Health check endpoint

### Schema Analysis

- **GET** `/api/v1/feature_suggestions?run_id={run_id}&num_features={num}` 
  - Get feature importance analysis and schema suggestions
  - **Parameters**:
    - `run_id` (required): The ID of the run to analyze
    - `num_features` (optional): Number of key features to identify (default: 7, max: 50)
  - **Returns**:
    ```json
    {
      "key_features": ["feature1", "feature2", ...],
      "initial_suggestions": {
        "feature1": {
          "initial_dtype": "int64", 
          "suggested_encoding_role": "numeric-continuous"
        },
        ...
      },
      "metadata": {
        "run_id": "...",
        "total_columns": 10,
        "total_rows": 1000,
        "target_column": "target",
        "task_type": "classification"
      }
    }
    ```

### Run Management

- **GET** `/api/v1/runs` - List all available runs
- **GET** `/api/v1/runs/{run_id}/info` - Get information about a specific run

## Example Usage

```bash
# List all runs
curl http://localhost:8000/api/v1/runs

# Get run information
curl http://localhost:8000/api/v1/runs/your_run_id/info

# Get feature suggestions
curl "http://localhost:8000/api/v1/feature_suggestions?run_id=your_run_id&num_features=5"
```

## Requirements

The API requires the following files to exist for each run:
- `data/runs/{run_id}/original_data.csv` - The original dataset
- `data/runs/{run_id}/metadata.json` - Run metadata including target column information

## Error Handling

The API provides detailed error messages for common issues:
- `404` - Run not found or required files missing
- `400` - Invalid parameters or missing target information
- `500` - Internal server errors with detailed messages

## Development

For development with auto-reload:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API automatically includes CORS middleware for development. Configure appropriately for production use. 