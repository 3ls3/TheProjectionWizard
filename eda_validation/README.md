# EDA & Validation Pipeline

This module provides comprehensive data exploration, validation, and cleaning capabilities for The Projection Wizard. It's designed to be used both programmatically and through the Streamlit interface.

## 🚀 Quick Start

### Using the Streamlit Interface

1. Start the Streamlit app:
```bash
streamlit run app/streamlit_team_a.py
```

2. Follow the guided pipeline:
   - Upload your CSV file
   - Review and adjust data types
   - Select target variable
   - Run data validation
   - Clean and preprocess data
   - Generate final EDA report

### Programmatic Usage

```python
from eda_validation import cleaning, ydata_profile, utils
from eda_validation.validation import setup_expectations, run_validation

# Load and clean data
df = pd.read_csv('data/raw/dataset.csv')
df_clean, report = cleaning.clean_dataframe(
    df,
    missing_strategy='drop',
    missing_threshold=0.5,
    standardize_columns=True,
    remove_dups=True,
    convert_dtypes=True
)

# Generate EDA profile
profile = ydata_profile.generate_profile(df_clean, title="My Dataset")
ydata_profile.save_profile_report(profile, "profile.html")

# Setup and run validation
expectations = setup_expectations.create_typed_expectation_suite(df_clean)
success, results = run_validation.validate_dataframe_with_suite(df_clean, expectations)
```

## 📦 Module Structure

```
eda_validation/
├── __init__.py                 # Module initialization
├── ydata_profile.py           # EDA via ydata-profiling
├── cleaning.py                # Data cleaning & preprocessing
├── utils.py                   # Utility functions
└── validation/
    ├── __init__.py
    ├── setup_expectations.py  # Great Expectations setup
    └── run_validation.py      # Validation execution
```

## 🔧 Key Features

### Data Cleaning (`cleaning.py`)
- Missing value handling (drop, fill, forward fill)
- Column name standardization
- Duplicate removal
- Automatic data type conversion
- Target column preservation

### EDA Profiling (`ydata_profile.py`)
- Comprehensive data quality analysis
- Statistical summaries
- Correlation analysis
- Distribution visualizations
- HTML report generation

### Data Validation (`validation/`)
- Type-based validation rules
- Data quality checks
- Custom expectation suites
- Validation report generation

## 📊 Output Files

The pipeline generates several output files in the `data/processed/` directory:

- `{dataset_name}_cleaned.csv` - Cleaned dataset
- `{dataset_name}_cleaned.json` - Cleaning report
- `{dataset_name}_validation_results.json` - Validation results
- `{dataset_name}_metadata.json` - Pipeline metadata

## 🛠️ Dependencies

### Required
```
streamlit==1.45.1
pandas==1.5.3
numpy==1.25.2
great-expectations==0.18.11
ydata-profiling==4.6.3
visions==0.7.5
matplotlib==3.8.4
seaborn==0.12.2
```

### Optional
```
openpyxl  # For Excel file support
plotly    # For interactive visualizations
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/test_eda_validation.py
```

## 📝 Notes

- The pipeline is designed to work with CSV files up to 200MB
- All data transformations are logged in the cleaning report
- Validation rules are automatically generated based on data types
- The Streamlit interface provides detailed debugging information
