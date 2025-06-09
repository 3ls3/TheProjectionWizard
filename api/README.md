# The Projection Wizard API

REST API for The Projection Wizard ML pipeline functionality. This API provides a complete interface for running the ML pipeline from data upload through model results.

## Quick Start

### Running the API

From the project root directory:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Using Docker Compose (recommended)
docker-compose up --build

# Backend will be available at http://localhost:8000
# Frontend will be available at http://localhost:3000
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Core System Endpoints

#### `GET /`
- **Description**: API information and version
- **Response**:
  ```json
  {
    "message": "The Projection Wizard API",
    "version": "1.0.0",
    "docs": "/docs",
    "redoc": "/redoc"
  }
  ```

#### `GET /health`
- **Description**: Health check endpoint
- **Response**:
  ```json
  {
    "status": "healthy"
  }
  ```

### Pipeline Workflow Endpoints

The API follows a sequential workflow for ML pipeline execution:

#### 1. `POST /api/upload`
- **Description**: Upload CSV file and create a new pipeline run
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (required): CSV file to upload
- **Response**:
  ```json
  {
    "run_id": "a1b2c3d4",
    "shape": [1000, 12],
    "preview": [
      ["col1", "col2", "target"],
      ["value1", "value2", "0"],
      ["value3", "value4", "1"]
    ]
  }
  ```
- **Example**:
  ```bash
  curl -X POST "http://localhost:8000/api/upload" \
    -F "file=@data.csv"
  ```

#### 2. `GET /api/target-suggestion`
- **Description**: Get AI-suggested target column for ML task
- **Parameters**:
  - `run_id` (required): Run ID from upload response
- **Response**:
  ```json
  {
    "suggested_column": "Survived",
    "task_type": "classification",
    "confidence": 0.85
  }
  ```
- **Example**:
  ```bash
  curl "http://localhost:8000/api/target-suggestion?run_id=a1b2c3d4"
  ```

#### 3. `POST /api/confirm-target`
- **Description**: Confirm target column and task type
- **Request Body**:
  ```json
  {
    "run_id": "a1b2c3d4",
    "confirmed_column": "Survived",
    "task_type": "classification"
  }
  ```
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```
- **Example**:
  ```bash
  curl -X POST "http://localhost:8000/api/confirm-target" \
    -H "Content-Type: application/json" \
    -d '{"run_id":"a1b2c3d4","confirmed_column":"Survived","task_type":"classification"}'
  ```

#### 4. `GET /api/feature-suggestion`
- **Description**: Get AI-suggested feature encodings and roles
- **Parameters**:
  - `run_id` (required): Run ID
- **Response**:
  ```json
  {
    "Age": {
      "initial_dtype": "float64",
      "suggested_encoding_role": "numeric-continuous"
    },
    "Cabin": {
      "initial_dtype": "object",
      "suggested_encoding_role": "text"
    },
    "Sex": {
      "initial_dtype": "object",
      "suggested_encoding_role": "categorical"
    }
  }
  ```
- **Example**:
  ```bash
  curl "http://localhost:8000/api/feature-suggestion?run_id=a1b2c3d4"
  ```

#### 5. `POST /api/confirm-features`
- **Description**: Confirm feature encodings and trigger pipeline execution (steps 3-7)
- **Request Body**:
  ```json
  {
    "run_id": "a1b2c3d4",
    "confirmed_features": {
      "Age": {
        "initial_dtype": "float64",
        "suggested_encoding_role": "numeric-continuous"
      },
      "Sex": {
        "initial_dtype": "object",
        "suggested_encoding_role": "categorical"
      }
    }
  }
  ```
- **Response**:
  ```json
  {
    "status": "ok",
    "message": "Pipeline started in background"
  }
  ```
- **Example**:
  ```bash
  curl -X POST "http://localhost:8000/api/confirm-features" \
    -H "Content-Type: application/json" \
    -d '{"run_id":"a1b2c3d4","confirmed_features":{...}}'
  ```

#### 6. `GET /api/status`
- **Description**: Get current pipeline execution status
- **Parameters**:
  - `run_id` (required): Run ID
- **Response**:
  ```json
  {
    "status": "running",
    "stage": "automl",
    "progress_pct": 70,
    "message": "Training AutoML models..."
  }
  ```
- **Possible Status Values**:
  - `pending`: Pipeline not started
  - `running`: Pipeline executing
  - `completed`: Pipeline finished successfully
  - `failed`: Pipeline encountered an error
- **Example**:
  ```bash
  curl "http://localhost:8000/api/status?run_id=a1b2c3d4"
  ```

#### 7. `GET /api/results`
- **Description**: Get final pipeline results (only available when status is 'completed')
- **Parameters**:
  - `run_id` (required): Run ID
- **Response**:
  ```json
  {
    "model_metrics": {
      "accuracy": 0.825,
      "f1_score": 0.810,
      "precision": 0.795,
      "recall": 0.826
    },
    "top_features": [
      {"feature": "Age", "importance": 0.342},
      {"feature": "Sex", "importance": 0.298},
      {"feature": "Pclass", "importance": 0.234}
    ],
    "explainability": {
      "shap_available": true,
      "explanation_summary": "Age and Sex are the most predictive features..."
    }
  }
  ```
- **Example**:
  ```bash
  curl "http://localhost:8000/api/results?run_id=a1b2c3d4"
  ```

## Error Handling

The API returns consistent error responses with structured details:

```json
{
  "detail": {
    "detail": "Run 'nonexistent' not found",
    "code": "run_not_found",
    "context": {
      "run_id": "nonexistent"
    }
  }
}
```

### Common Error Codes

- `run_not_found`: The specified run ID doesn't exist
- `invalid_csv`: Uploaded file is not a valid CSV or is corrupted
- `target_not_confirmed`: Target column must be confirmed before proceeding
- `pipeline_failed`: Pipeline execution encountered an error
- `results_not_ready`: Results not available yet (pipeline still running or failed)

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters, validation errors)
- `404`: Not Found (run not found, files missing)
- `500`: Internal Server Error (unexpected server errors)

## Workflow Example

Complete example of using the API:

```bash
# 1. Upload CSV
UPLOAD_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/upload" -F "file=@titanic.csv")
RUN_ID=$(echo $UPLOAD_RESPONSE | jq -r '.run_id')

# 2. Get target suggestion
curl "http://localhost:8000/api/target-suggestion?run_id=$RUN_ID"

# 3. Confirm target
curl -X POST "http://localhost:8000/api/confirm-target" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\":\"$RUN_ID\",\"confirmed_column\":\"Survived\",\"task_type\":\"classification\"}"

# 4. Get feature suggestions
FEATURES=$(curl -s "http://localhost:8000/api/feature-suggestion?run_id=$RUN_ID")

# 5. Confirm features (triggers pipeline)
curl -X POST "http://localhost:8000/api/confirm-features" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\":\"$RUN_ID\",\"confirmed_features\":$FEATURES}"

# 6. Poll status until complete
while true; do
  STATUS=$(curl -s "http://localhost:8000/api/status?run_id=$RUN_ID" | jq -r '.status')
  echo "Status: $STATUS"
  [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]] && break
  sleep 5
done

# 7. Get results
curl "http://localhost:8000/api/results?run_id=$RUN_ID"
```

## Data Storage

### Run Directory Structure
```
data/runs/{run_id}/
├── original_data.csv              # Uploaded CSV file
├── metadata.json                  # Run metadata and configuration
├── pipeline_structured.jsonl     # Structured logs
├── step_3_validation/            # Validation artifacts
├── step_4_prep/                  # Prepared data
├── step_5_automl/               # Trained models
└── step_6_explain/              # SHAP explanations
```

### Persistence
- All run data is persisted in `data/runs/{run_id}/`
- API is stateless - all state stored in run directories
- Run directories can be backed up and restored independently

## Development

### Local Development
```bash
# Run with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/

# Check API with curl or HTTPie
http GET localhost:8000/health
```

### Production Considerations
- Configure CORS origins appropriately
- Add authentication if needed
- Set up proper logging and monitoring
- Use reverse proxy (nginx) for production deployment
- Configure file upload size limits

## Dependencies

Key dependencies for file upload functionality:
- `fastapi>=0.104.0` - Web framework
- `uvicorn>=0.24.0` - ASGI server
- `python-multipart>=0.0.6` - **Required** for file uploads
- `pydantic>=2.4.0` - Data validation

## Frontend Integration

This API is designed to work with the React frontend available at `http://localhost:3000` when using Docker Compose. The frontend automatically uses the `VITE_API_URL` environment variable to communicate with this API. 