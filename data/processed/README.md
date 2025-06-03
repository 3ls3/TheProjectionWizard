# Processed Data Directory

This directory contains outputs from Team A's EDA and validation pipeline, ready for Team B's modeling work.

## Generated Files

For each processed dataset `{dataset_name}`, you'll find:

### Core Data Files
- **`{dataset_name}_cleaned.csv`** - Main cleaned dataset ready for modeling
- **`{dataset_name}_profile.html`** - Comprehensive EDA report (ydata-profiling)

### Reports and Metadata
- **`{dataset_name}_cleaned.json`** - Data cleaning report with transformation details
- **`{dataset_name}_validation_results.json`** - Data validation results from Great Expectations
- **`{dataset_name}_expectations.json`** - Generated expectation suite for future validation
- **`{dataset_name}_metadata.json`** - Pipeline metadata and quality scores

## File Contents

### Cleaned Data (`_cleaned.csv`)
- Missing values handled according to specified strategy
- Column names standardized to snake_case
- Duplicates removed
- Data types optimized
- Ready for direct use in modeling

### Cleaning Report (`_cleaned.json`)
```json
{
  "original_shape": [1000, 15],
  "final_shape": [950, 15],
  "steps_performed": ["handled_missing_values_drop", "standardized_column_names", "removed_duplicates"],
  "rows_removed": 50,
  "columns_removed": 0,
  "missing_values_original": {...},
  "missing_values_final": {...}
}
```

### Validation Results (`_validation_results.json`)
```json
{
  "suite_name": "dataset_validation_suite",
  "total_expectations": 25,
  "successful_expectations": 23,
  "failed_expectations": 2,
  "overall_success": false,
  "success_rate": 0.92,
  "expectation_results": [...]
}
```

## Usage for Team B

Team B can directly use the cleaned data:

```python
import pandas as pd
import json

# Load cleaned data
df = pd.read_csv('data/processed/dataset_cleaned.csv')

# Load metadata for context
with open('data/processed/dataset_metadata.json', 'r') as f:
    metadata = json.load(f)

# Check data quality score
quality_score = metadata['quality_score']
print(f"Data quality score: {quality_score}/100")
```

## Quality Assurance

All files in this directory have passed Team A's validation pipeline:
- ✅ Data types verified
- ✅ Missing values handled
- ✅ Duplicates removed  
- ✅ Column names standardized
- ✅ Expectations validated
- ✅ Quality score calculated 