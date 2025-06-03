# ProjectionWizard

**AutoML Data Science Pipeline for Tabular Data**

A collaborative data science bootcamp project implementing an end-to-end AutoML pipeline with clean team boundaries and modular architecture.

## 🏗️ Project Structure

```
TheProjectionWizard/
├── app/
│   ├── main.py                     # Shared Streamlit entry (Team B)
│   ├── streamlit_team_a.py         # Team A's temporary frontend
│   ├── predictor.py                # Team B's predictor (existing)
│   └── utils.py                    # Shared utilities
│
├── data/
│   ├── raw/                        # User-uploaded data
│   ├── mock/                       # Synthetic test datasets  
│   └── processed/                  # Team A's cleaned outputs
│
├── eda_validation/                 # ✅ Team A's Main Module
│   ├── __init__.py                 # Module initialization
│   ├── ydata_profile.py           # EDA via ydata-profiling
│   ├── cleaning.py                # Data cleaning & preprocessing
│   ├── utils.py                   # Utility functions
│   └── validation/
│       ├── __init__.py
│       ├── setup_expectations.py  # Great Expectations setup
│       └── run_validation.py      # Validation execution
│
├── modeling/                      # Team B's folder (do not edit)
│   └── README.md                  # Team B integration guide
│
├── tests/
│   ├── test_eda_validation.py     # Team A's test suite
│   ├── test_print.py              # Existing tests
│   └── test_tim.py                # Existing tests
│
├── requirements.txt               # Python dependencies
├── app.yaml                       # App configuration
└── README.md                      # This file
```

## 👥 Team Responsibilities

### Team A (EDA & Validation)
- **Data Profiling**: Comprehensive EDA using ydata-profiling
- **Data Validation**: Quality checks with Great Expectations
- **Data Cleaning**: Missing values, duplicates, standardization
- **Pipeline Output**: Cleaned data ready for modeling

### Team B (Modeling & Prediction)
- **AutoML Training**: PyCaret/AutoGluon model comparison
- **Model Evaluation**: Performance metrics and selection
- **Explainability**: SHAP/LIME analysis
- **Main App Integration**: Streamlit UI with Team A's data

## 🚀 Quick Start

### Team A Pipeline Usage

1. **Upload raw data** to `data/raw/`

2. **Run EDA profiling**:
   ```bash
   python eda_validation/ydata_profile.py data/raw/your_file.csv
   ```

3. **Clean the data**:
   ```bash
   python eda_validation/cleaning.py data/raw/your_file.csv --missing-strategy drop
   ```

4. **Setup validation expectations**:
   ```bash
   python eda_validation/validation/setup_expectations.py data/raw/your_file.csv
   ```

5. **Run data validation**:
   ```bash
   python eda_validation/validation/run_validation.py data/raw/your_file.csv -s expectations.json
   ```

6. **Use Team A's temporary frontend**:
   ```bash
   streamlit run app/streamlit_team_a.py
   ```

### Programmatic Usage

```python
from eda_validation import cleaning, ydata_profile, utils
from eda_validation.validation import setup_expectations, run_validation

# Load and clean data
df = pd.read_csv('data/raw/dataset.csv')
df_clean, report = cleaning.clean_dataframe(df)

# Generate EDA profile
profile = ydata_profile.generate_profile(df_clean, title="My Dataset")
ydata_profile.save_profile_report(profile, "profile.html")

# Setup and run validation
expectations = setup_expectations.create_basic_expectation_suite(df_clean)
success, results = run_validation.validate_dataframe_with_suite(df_clean, expectations)
```

## 📦 Dependencies

### Core Requirements
```
pandas
numpy
streamlit
pathlib
```

### Optional (Enhanced Features)
```
ydata-profiling  # For comprehensive EDA reports
great-expectations  # For data validation
openpyxl  # For Excel file support
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🧪 Testing

Run Team A's test suite:
```bash
# Using unittest
python tests/test_eda_validation.py

# Using pytest (if installed)
python -m pytest tests/test_eda_validation.py -v
```

## 📁 Data Flow

```
Raw Data → EDA Profiling → Data Cleaning → Validation → Processed Data → Team B
    ↓           ↓              ↓             ↓             ↓
data/raw/   reports/    data/processed/ validation/  data/processed/
```

### Team A Outputs

For each dataset `{name}`, Team A generates:
- `{name}_cleaned.csv` - Clean data ready for modeling
- `{name}_profile.html` - EDA report
- `{name}_cleaned.json` - Cleaning report  
- `{name}_validation_results.json` - Validation results
- `{name}_expectations.json` - Expectation suite

## 🔧 Configuration

### Cleaning Options
- **Missing value strategies**: `drop`, `fill_mean`, `fill_median`, `fill_mode`, `forward_fill`
- **Column standardization**: `snake_case`, `camel_case`, `lower`
- **Duplicate handling**: Configurable keep strategy

### Validation Options
- **Automatic expectation generation** based on data analysis
- **Custom expectation suites** via JSON configuration
- **Validation reporting** with detailed failure analysis

## 🤝 Team Integration

### For Team B

Team A provides clean handoff through `data/processed/`:

```python
# Load Team A's cleaned data
df = pd.read_csv('data/processed/dataset_cleaned.csv')

# Load metadata for context
with open('data/processed/dataset_metadata.json') as f:
    metadata = json.load(f)

# Check data quality
quality_score = metadata['quality_score']  # 0-100 scale
```

### Module Boundaries

- **Team A**: `eda_validation/` module only
- **Team B**: `modeling/` and `app/main.py`
- **Shared**: `data/processed/` for handoff

## 📋 CLI Reference

### EDA Profiling
```bash
python eda_validation/ydata_profile.py INPUT_FILE [-o OUTPUT] [-t TITLE]
```

### Data Cleaning  
```bash
python eda_validation/cleaning.py INPUT_FILE [-o OUTPUT] [--missing-strategy STRATEGY] [--missing-threshold THRESHOLD]
```

### Validation Setup
```bash
python eda_validation/validation/setup_expectations.py INPUT_FILE [-n NAME] [-o OUTPUT]
```

### Validation Execution
```bash
python eda_validation/validation/run_validation.py INPUT_FILE -s SUITE_FILE [-o OUTPUT]
```

## 🎯 Quality Metrics

Team A's pipeline generates quality scores based on:
- **Missing values**: Percentage of missing data
- **Duplicates**: Duplicate row percentage  
- **Validation**: Great Expectations success rate
- **Overall Score**: Composite 0-100 quality rating

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from project root
2. **Missing dependencies**: Install optional packages for full functionality
3. **File path issues**: Use relative paths from project root
4. **Memory issues**: Consider sampling large datasets for initial exploration

### Debug Mode
```bash
# Enable detailed logging
python eda_validation/cleaning.py data.csv --verbose
```

## 📝 License

Data Science Bootcamp Project - Educational Use

---

**Team A Contributors**: [Your Names Here]  
**Team B Contributors**: [Team B Names Here]
