#!/usr/bin/env python3
"""
Synthetic Data Fixture Generator for ML Pipeline Testing

This script generates various synthetic CSV datasets to test different aspects
of the ML pipeline, including edge cases and pathological data scenarios.

TODO: Extend Fixture Generator

This script is designed to evolve into a comprehensive synthetic data generator for testing the full ML pipeline.

Future capabilities:
- Pathology simulation (NaNs, outliers, wrong types, etc.)
- CLI options for dataset type, rows, features, pathology flags
- Auto-tagging and scenario metadata
- CI integration to run pipeline on all fixtures

Current version: MVP to generate core testing datasets
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Callable, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_classification(n_rows: int = 100) -> pd.DataFrame:
    """Generate a clean, simple binary classification dataset."""
    np.random.seed(42)
    
    ages = np.random.randint(18, 80, n_rows)
    incomes = np.random.normal(50000, 20000, n_rows).astype(int)
    incomes = np.clip(incomes, 20000, 150000)  # Reasonable income range
    
    genders = np.random.choice(['Male', 'Female', 'Other'], n_rows, p=[0.45, 0.45, 0.10])
    
    # Create target with some logical correlation
    purchase_prob = (ages / 100) + (incomes / 200000) + np.random.normal(0, 0.1, n_rows)
    purchased = (purchase_prob > 0.6).astype(int)
    
    return pd.DataFrame({
        'age': ages,
        'income': incomes,
        'gender': genders,
        'purchased': purchased
    })

def regression_with_nans(n_rows: int = 100) -> pd.DataFrame:
    """Generate regression data with missing values."""
    np.random.seed(123)
    
    ages = np.random.randint(18, 80, n_rows)
    incomes = np.random.normal(50000, 20000, n_rows).astype(int)
    genders = np.random.choice(['Male', 'Female', 'Other'], n_rows)
    
    # Introduce NaN values randomly (about 10% missing)
    nan_indices = np.random.choice(n_rows, size=int(n_rows * 0.1), replace=False)
    incomes = incomes.astype(float)
    incomes[nan_indices] = np.nan
    
    # Target is continuous (spending amount)
    spending = ages * 50 + incomes * 0.02 + np.random.normal(0, 500, n_rows)
    spending = np.clip(spending, 0, 10000)
    
    return pd.DataFrame({
        'age': ages,
        'income': incomes,
        'gender': genders,
        'spending': spending
    })

def imbalanced_classes(n_rows: int = 100) -> pd.DataFrame:
    """Generate dataset with highly imbalanced target classes."""
    np.random.seed(456)
    
    ages = np.random.randint(18, 80, n_rows)
    incomes = np.random.normal(50000, 20000, n_rows).astype(int)
    genders = np.random.choice(['Male', 'Female', 'Other'], n_rows)
    
    # Create highly imbalanced target (95% class 0, 5% class 1)
    purchased = np.random.choice([0, 1], n_rows, p=[0.95, 0.05])
    
    return pd.DataFrame({
        'age': ages,
        'income': incomes,
        'gender': genders,
        'purchased': purchased
    })

def outliers_dataset(n_rows: int = 100) -> pd.DataFrame:
    """Generate dataset with extreme outliers."""
    np.random.seed(789)
    
    ages = np.random.randint(18, 80, n_rows)
    incomes = np.random.normal(50000, 20000, n_rows).astype(int)
    genders = np.random.choice(['Male', 'Female', 'Other'], n_rows)
    
    # Introduce extreme outliers
    outlier_indices = np.random.choice(n_rows, size=5, replace=False)
    ages[outlier_indices] = [200, 300, -50, 999, 150]
    incomes[outlier_indices] = [1000000, -50000, 2000000, 0, 5000000]
    
    purchased = np.random.choice([0, 1], n_rows, p=[0.7, 0.3])
    
    return pd.DataFrame({
        'age': ages,
        'income': incomes,
        'gender': genders,
        'purchased': purchased
    })

def multiclass_dataset(n_rows: int = 100) -> pd.DataFrame:
    """Generate multiclass classification dataset."""
    np.random.seed(101)
    
    ages = np.random.randint(18, 80, n_rows)
    incomes = np.random.normal(50000, 20000, n_rows).astype(int)
    genders = np.random.choice(['Male', 'Female', 'Other'], n_rows)
    
    # Multi-class target: customer segment
    segments = np.random.choice(['Budget', 'Premium', 'Luxury', 'Enterprise'], 
                               n_rows, p=[0.4, 0.3, 0.2, 0.1])
    
    return pd.DataFrame({
        'age': ages,
        'income': incomes,
        'gender': genders,
        'segment': segments
    })

# Edge case datasets for testing data validation
def missing_column_dataset(n_rows: int = 100) -> pd.DataFrame:
    """Generate dataset missing the 'income' column."""
    df = simple_classification(n_rows)
    return df.drop('income', axis=1)

def wrong_dtype_dataset(n_rows: int = 100) -> pd.DataFrame:
    """Generate dataset with wrong data types."""
    df = simple_classification(n_rows)
    # Make age column contain strings
    df['age'] = df['age'].astype(str) + '_years'
    # Make income contain mixed types - convert to object dtype first
    df['income'] = df['income'].astype(object)
    df.loc[0:10, 'income'] = 'unknown'
    return df

def nan_target_dataset(n_rows: int = 100) -> pd.DataFrame:
    """Generate dataset with NaN values in target column."""
    df = simple_classification(n_rows)
    # Introduce NaN in target
    nan_indices = np.random.choice(n_rows, size=int(n_rows * 0.15), replace=False)
    df.loc[nan_indices, 'purchased'] = np.nan
    return df

def duplicates_dataset(n_rows: int = 100) -> pd.DataFrame:
    """Generate dataset with exact duplicate rows."""
    df = simple_classification(n_rows)
    # Add exact duplicates (repeat 20% of rows)
    duplicate_indices = np.random.choice(n_rows, size=int(n_rows * 0.2), replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    df_with_duplicates = pd.concat([df, duplicates], ignore_index=True)
    return df_with_duplicates

def empty_dataset(n_rows: int = 0) -> pd.DataFrame:
    """Generate empty dataset for edge case testing."""
    return pd.DataFrame(columns=['age', 'income', 'gender', 'purchased'])

# Fixture registry mapping names to generation functions
FIXTURE_REGISTRY: Dict[str, Callable] = {
    'valid_small': simple_classification,
    'regression_nans': regression_with_nans,
    'imbalanced': imbalanced_classes,
    'outliers': outliers_dataset,
    'multiclass': multiclass_dataset,
    'missing_column': missing_column_dataset,
    'wrong_dtype': wrong_dtype_dataset,
    'nan_target': nan_target_dataset,
    'duplicates': duplicates_dataset,
    'empty': empty_dataset,
}

def generate_all_fixtures(output_dir: str, n_rows: int = 100):
    """Generate all fixtures and save to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating fixtures in: {output_path}")
    
    fixture_descriptions = {}
    
    for name, generator_func in FIXTURE_REGISTRY.items():
        try:
            if name == 'empty':
                df = generator_func(0)  # Empty dataset
            else:
                df = generator_func(n_rows)
            
            filename = f"{name}.csv"
            filepath = output_path / filename
            df.to_csv(filepath, index=False)
            
            logger.info(f"✅ Generated {filename}: {len(df)} rows, {len(df.columns)} columns -> {filepath}")
            
            # Store description for README
            fixture_descriptions[name] = {
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'description': generator_func.__doc__.strip() if generator_func.__doc__ else "No description available"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate {name}: {str(e)}")
    
    # Generate README
    generate_readme(output_path, fixture_descriptions)
    
    logger.info(f"🎉 Successfully generated {len(fixture_descriptions)} fixtures!")

def generate_readme(output_dir: Path, descriptions: Dict):
    """Generate README.md explaining each fixture."""
    readme_content = """# Test Data Fixtures

This directory contains synthetic CSV datasets generated for testing the ML pipeline.
Each dataset is designed to test specific scenarios and edge cases.

## Generated Fixtures

| Filename | Rows | Columns | Description |
|----------|------|---------|-------------|
"""
    
    for name, info in descriptions.items():
        readme_content += f"| `{info['filename']}` | {info['rows']} | {info['columns']} | {info['description']} |\n"
    
    readme_content += """
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
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"📝 Generated README.md: {readme_path}")

def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Generate synthetic test fixtures for ML pipeline")
    parser.add_argument(
        '--rows', 
        type=int, 
        default=100, 
        help='Number of rows to generate for each dataset (default: 100)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='tests/data/fixtures', 
        help='Output directory for fixtures (default: tests/data/fixtures)'
    )
    parser.add_argument(
        '--list', 
        action='store_true', 
        help='List available fixture types and exit'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available fixture types:")
        for name, func in FIXTURE_REGISTRY.items():
            print(f"  {name}: {func.__doc__.strip() if func.__doc__ else 'No description'}")
        return
    
    generate_all_fixtures(args.output_dir, args.rows)

if __name__ == "__main__":
    main() 