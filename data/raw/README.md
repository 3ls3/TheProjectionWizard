# Raw Data Directory

This directory contains user-uploaded, unprocessed data files.

## Supported File Types

Team A's pipeline supports:
- **CSV files** (`.csv`, `.tsv`)
- **Excel files** (`.xlsx`, `.xls`)
- **JSON files** (`.json`)
- **Parquet files** (`.parquet`, `.pq`)

## Usage

Place your raw datasets here for processing through Team A's EDA and validation pipeline.

### Example Structure
```
data/raw/
├── customer_data.csv
├── sales_records.xlsx
└── product_catalog.json
```

### Processing Flow
1. Upload raw data files here
2. Run Team A's pipeline:
   ```bash
   # Generate EDA profile
   python eda_validation/ydata_profile.py data/raw/your_file.csv
   
   # Clean data
   python eda_validation/cleaning.py data/raw/your_file.csv
   
   # Setup and run validation
   python eda_validation/validation/setup_expectations.py data/raw/your_file.csv
   python eda_validation/validation/run_validation.py data/raw/your_file.csv -s expectations.json
   ```
3. Cleaned data appears in `data/processed/`

### File Size Considerations
- Large files (>100MB) may take longer to process
- Consider sampling large datasets for initial EDA exploration
- The pipeline automatically handles memory-efficient processing where possible 