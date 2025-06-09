# Test Data Fixtures

This directory contains synthetic CSV datasets generated for testing the ML pipeline.
Each dataset is designed to test specific scenarios and edge cases.

## Generated Fixtures

| Filename | Rows | Columns | Description |
|----------|------|---------|-------------|
| `valid_small.csv` | 100 | 4 | Generate a clean, simple binary classification dataset. |
| `regression_nans.csv` | 100 | 4 | Generate regression data with missing values. |
| `imbalanced.csv` | 100 | 4 | Generate dataset with highly imbalanced target classes. |
| `outliers.csv` | 100 | 4 | Generate dataset with extreme outliers. |
| `multiclass.csv` | 100 | 4 | Generate multiclass classification dataset. |
| `missing_column.csv` | 100 | 3 | Generate dataset missing the 'income' column. |
| `wrong_dtype.csv` | 100 | 4 | Generate dataset with wrong data types. |
| `nan_target.csv` | 100 | 4 | Generate dataset with NaN values in target column. |
| `duplicates.csv` | 120 | 4 | Generate dataset with exact duplicate rows. |
| `empty.csv` | 0 | 4 | Generate empty dataset for edge case testing. |
| `validation_failure.csv` | 100 | 7 | Generate dataset specifically designed to fail multiple validation checks. |
| `validation_failure_correct_columns.csv` | 100 | 4 | Generate dataset with correct column names but failing data validation. |
| `customer_churn.csv` | 100 | 8 | Generate realistic customer churn prediction dataset. |
| `loan_approval.csv` | 100 | 9 | Generate realistic loan approval prediction dataset. |
| `house_prices.csv` | 100 | 10 | Generate realistic house price prediction dataset. |
| `sales_forecast.csv` | 100 | 10 | Generate realistic sales forecasting dataset. |

## Usage

These fixtures are used by the test suite to validate:
- Data ingestion and validation
- Pipeline robustness with edge cases
- Model training on various data patterns
- Error handling for malformed data

## Regenerating Fixtures

To regenerate all fixtures:
```bash
python scripts/generate_fixtures.py
```

To generate with custom parameters:
```bash
python scripts/generate_fixtures.py --rows 200 --output-dir custom/path
```
