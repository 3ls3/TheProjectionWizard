# The Projection Wizard

A data science bootcamp project to build an end-to-end ML pipeline. Users upload a tabular CSV (non-time-series), and receive:
→ a trained model
→ predictions
→ data profiling reports
→ basic model explanations

## Project Structure

The pipeline consists of two teams:
- **Team A (Current Focus)**: Handles EDA, data validation, and light cleaning
  - Goal: Output a cleaned_data.csv and schema.json ready for AutoML
- **Team B**: Takes the cleaned data and runs AutoML (e.g. PyCaret), model comparison, and explainability

## New Feature: Type Override UI

### Overview
We've implemented a user-first approach where users can review and override auto-detected data types and select their target variable before running intensive EDA and validation processes.

### How It Works

1. **Upload CSV**: Users upload their CSV file as before
2. **Data Preview**: View basic file info, column information, and data preview
3. **🆕 Type Override & Target Selection**: 
   - Review pandas-inferred data types for each column
   - Override types using dropdown menus (string/object, integer, float, boolean, category, datetime)
   - Select target variable using radio buttons (only one allowed)
   - Confirm selections with the "Confirm Types & Target" button
4. **Processing**: DataFrame is updated with confirmed types and stored in session state
5. **Next Steps**: EDA, validation, and cleaning steps now use the processed DataFrame with correct types

### Benefits

- **More Accurate EDA**: Reports are generated on correctly typed data
- **Better Validation**: Great Expectations rules apply to the intended data schema
- **Improved Cleaning**: Operations work on properly typed columns
- **User Control**: Users explicitly confirm their data schema before analysis

### UI Components

- **Column Type Grid**: Shows column name, inferred type, new type dropdown, and target selection
- **Real-time Summary**: Displays selected target column and number of type changes
- **Confirmation System**: Requires explicit confirmation before proceeding
- **Error Handling**: Graceful handling of type conversion failures
- **State Management**: Preserves user selections and processed data across navigation

### Technical Implementation

- **Session State**: Uses `st.session_state` to maintain user selections and processed DataFrame
- **Type Conversion**: Robust type casting with error handling for edge cases
- **Integration**: All subsequent pipeline steps check for confirmed types before proceeding
- **Modularity**: Clean separation of concerns with dedicated functions

## Running the Application

```bash
# Activate virtual environment
source .venv/bin/activate

# Run Team A's Streamlit interface
streamlit run app/streamlit_team_a.py
```

## 🔄 Pipeline Flow (Continuous UX)

### Main Pipeline (Continuous Flow)
1. **📁 Upload Data** → Upload CSV and review basic file information
2. **📊 Basic EDA** → Quick statistical overview and data quality checks  
3. **🎯 Type Override** → Confirm column types and target selection
4. **✅ Data Validation** → Automatic Great Expectations validation after confirmation
5. **🧹 Data Cleaning** → Clean and preprocess data (placeholder for now)
6. **📋 Final EDA** → Complete analysis with downloadable cleaned_data.csv

### Debug Sidebar (Detailed Information)
- **Upload Data Details** → Session state, DataFrame info, column analysis
- **Type Override Details** → Type changes, processed DataFrame details
- **Data Validation Details** → Validation results, expectation details
- **Data Cleaning Details** → Cleaning operations (to be implemented)
- **EDA Profiling Details** → Comprehensive reports (to be implemented)

## Sample Data

Test the application with the sample data file:
- `data/mock/sample_data.csv` - Contains various data types for testing

## ✅ Latest Updates: Continuous Pipeline UX

### 🎯 New Continuous Flow Design

We've completely restructured the user experience to provide a **seamless, continuous pipeline flow**:

#### **📊 Progress Tracking**
- **Visual Progress Bar**: Clear 6-stage progress indicator at the top
- **Stage-based Navigation**: Each stage builds on the previous one
- **State Persistence**: Session state maintained when switching to debug views
- **Smart Advancement**: Automatic progression when stages are completed

#### **🔄 Main Pipeline Flow**
1. **Upload** → File upload with immediate validation and preview
2. **Basic EDA** → Quick statistical overview and data quality insights
3. **Type Override** → Interactive type confirmation and target selection
4. **Validation** → Automatic Great Expectations validation with clear pass/fail
5. **Cleaning** → Data cleaning operations (placeholder implementation)
6. **Final EDA** → Complete analysis with downloadable outputs

#### **🔍 Debug Sidebar**
- **Detailed Logging**: Comprehensive debugging information for each stage
- **Session State Inspection**: Real-time view of all stored data and settings
- **Validation Details**: In-depth validation results and expectation analysis
- **Development Support**: Essential debugging tools for development and testing

#### **🚀 User Experience Improvements**
- **No Lost Progress**: Switching between main flow and debug views preserves all progress
- **Clear Error Handling**: Actionable error messages with options to fix or continue
- **Download Ready**: Final cleaned_data.csv and metadata.json ready for Team B
- **Restart Option**: Easy reset to process multiple datasets in sequence

## Next Steps

- ✅ **COMPLETED**: Great Expectations validation with confirmed schema
- Implement ydata-profiling EDA reports using processed DataFrame
- Implement data cleaning operations
- Create export functionality for cleaned_data.csv and schema.json

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
