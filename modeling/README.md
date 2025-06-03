# Team B - Modeling Pipeline

**⚠️ This directory is reserved for Team B's implementation.**

Team A should not edit files in this directory to maintain clean boundaries between our work.

## Expected Team B Components

Team B will handle:

- **Model Training**: AutoML comparisons using PyCaret or AutoGluon
- **Model Evaluation**: Performance metrics and model selection
- **Explainability**: SHAP, LIME analysis
- **Prediction Output**: Final model inference
- **Main Streamlit App**: Integration with Team A's cleaned data

## Integration Points

Team B should expect to receive cleaned and validated data from Team A in:
- `data/processed/` - Cleaned CSV files
- `data/processed/` - Data validation reports and metadata

## Data Handoff Format

Team A will provide:
```
data/processed/
├── {dataset_name}_cleaned.csv          # Main cleaned dataset
├── {dataset_name}_cleaned.json         # Cleaning report
├── {dataset_name}_validation_results.json  # Validation results
└── {dataset_name}_metadata.json        # Pipeline metadata
```

Team B can import our pipeline functions if needed:
```python
from eda_validation import cleaning, utils
from eda_validation.validation import run_validation
``` 