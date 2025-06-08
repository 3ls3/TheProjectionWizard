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

# NEW: Datasets designed to fail validation
def validation_failure_dataset(n_rows: int = 100) -> pd.DataFrame:
    """Generate dataset specifically designed to fail multiple validation checks."""
    np.random.seed(999)
    
    # PROBLEM 1: Wrong column order and missing expected columns
    # Create completely different columns than expected (age, income, gender, purchased)
    wrong_columns = np.random.randn(n_rows)
    unexpected_data = np.random.choice(['A', 'B', 'C'], n_rows)
    
    # PROBLEM 2: Extreme missing values - 80% missing in key columns  
    ages = np.full(n_rows, np.nan)  # Almost all NaN
    ages[0:int(n_rows * 0.1)] = np.random.randint(18, 80, int(n_rows * 0.1))  # Only 10% valid
    
    incomes = np.full(n_rows, np.nan)  # Almost all NaN  
    incomes[0:int(n_rows * 0.05)] = np.random.normal(50000, 20000, int(n_rows * 0.05))  # Only 5% valid
    
    # PROBLEM 3: Target column completely wrong
    # For binary classification, use completely non-binary values
    purchased = ['invalid'] * n_rows  # All invalid strings
    
    # PROBLEM 4: Categorical column with extreme cardinality (more than 50)
    # Create 80 unique gender categories (way above threshold)
    gender_categories = [f'InvalidGender_{i}' for i in range(80)]
    genders = np.random.choice(gender_categories, n_rows)
    
    # PROBLEM 5: Add columns that completely break expectations
    # Negative ages, impossible values
    ages = ages.astype(object)
    
    # Add invalid string values to some age entries
    invalid_strings = ['not_a_number', 'invalid', 'NaN', '???', 'ERROR']
    for i in range(int(n_rows * 0.1), min(int(n_rows * 0.2), n_rows)):
        ages[i] = np.random.choice(invalid_strings)
    
    # Add impossible numeric values to some age entries  
    impossible_values = [-999, -100, 999999, 0.5, None]
    for i in range(int(n_rows * 0.2), min(int(n_rows * 0.3), n_rows)):
        ages[i] = np.random.choice(impossible_values)
    
    # PROBLEM 6: Income with impossible values and mixed types
    incomes = incomes.astype(object)
    
    # Add invalid string values to some income entries
    invalid_income_strings = ['negative_income', 'unknown', 'error', '$$$', None]
    for i in range(int(n_rows * 0.05), min(int(n_rows * 0.15), n_rows)):
        incomes[i] = np.random.choice(invalid_income_strings)
    
    # Add impossible numeric values to some income entries
    impossible_income_values = [-999999, 0, -1, float('inf'), float('-inf')]
    for i in range(int(n_rows * 0.15), min(int(n_rows * 0.25), n_rows)):
        incomes[i] = np.random.choice(impossible_income_values)
    
    # PROBLEM 7: Create a DataFrame that will fail table-level expectations
    # Wrong column names entirely
    return pd.DataFrame({
        'wrong_age_column': ages,  # Wrong column name
        'wrong_income_column': incomes,  # Wrong column name  
        'wrong_gender_column': genders,  # Wrong column name
        'wrong_target_column': purchased,  # Wrong column name
        'completely_unexpected_column_1': wrong_columns,
        'completely_unexpected_column_2': unexpected_data,
        'another_wrong_column': np.random.randn(n_rows)
    })

def validation_failure_correct_columns_dataset(n_rows: int = 100) -> pd.DataFrame:
    """Generate dataset with correct column names but failing data validation."""
    np.random.seed(888)
    
    # Correct column names but problematic data
    ages = np.random.randint(18, 80, n_rows).astype(object)
    incomes = np.random.normal(50000, 20000, n_rows).astype(object)
    genders = np.random.choice(['Male', 'Female', 'Other'], n_rows).astype(object)
    purchased = np.random.choice([0, 1], n_rows).astype(object)
    
    # PROBLEM 1: Make 90% of ages invalid strings/mixed types
    invalid_age_indices = np.random.choice(n_rows, size=int(n_rows * 0.9), replace=False)
    invalid_age_values = ['invalid_age', 'not_a_number', 'error', 'NaN', 'unknown', 
                         'negative', '999999', 'too_old', 'baby', 'ancient']
    for i in invalid_age_indices:
        ages[i] = np.random.choice(invalid_age_values)
    
    # PROBLEM 2: Make 85% of incomes missing (way above 30% threshold)
    missing_income_indices = np.random.choice(n_rows, size=int(n_rows * 0.85), replace=False)
    for i in missing_income_indices:
        incomes[i] = np.nan
    
    # PROBLEM 3: Make remaining incomes mixed invalid types
    valid_income_indices = [i for i in range(n_rows) if i not in missing_income_indices]
    for i in valid_income_indices[:len(valid_income_indices)//2]:
        incomes[i] = np.random.choice(['negative_income', 'unknown_income', 'invalid', '$$$'])
    
    # PROBLEM 4: Create way too many gender categories (70+ unique values)
    gender_categories = [f'InvalidGender_{i}' for i in range(75)]
    for i in range(n_rows):
        genders[i] = np.random.choice(gender_categories)
    
    # PROBLEM 5: Make target completely invalid for binary classification
    invalid_target_values = ['maybe', 'perhaps', 'unknown', 'invalid', 'error', 
                           'definitely_not', 'absolutely', 'never', 'always', 'sometimes']
    for i in range(n_rows):
        purchased[i] = np.random.choice(invalid_target_values)
    
    return pd.DataFrame({
        'age': ages,
        'income': incomes, 
        'gender': genders,
        'purchased': purchased
    })

# NEW: Realistic Classification Datasets
def customer_churn_dataset(n_rows: int = 1000) -> pd.DataFrame:
    """Generate realistic customer churn prediction dataset."""
    np.random.seed(42)
    
    # Customer demographics
    ages = np.random.normal(45, 15, n_rows).astype(int)
    ages = np.clip(ages, 18, 85)
    
    # Account tenure in months
    tenure_months = np.random.exponential(24, n_rows).astype(int)
    tenure_months = np.clip(tenure_months, 1, 120)
    
    # Monthly charges with realistic distribution
    monthly_charges = np.random.lognormal(np.log(75), 0.4, n_rows)
    monthly_charges = np.clip(monthly_charges, 20, 300).round(2)
    
    # Contract type affects churn
    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                     n_rows, p=[0.5, 0.3, 0.2])
    
    # Internet service
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                       n_rows, p=[0.4, 0.5, 0.1])
    
    # Support tickets (Poisson distribution)
    support_tickets = np.random.poisson(2, n_rows)
    support_tickets = np.clip(support_tickets, 0, 20)
    
    # Payment method
    payment_methods = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
                                      n_rows, p=[0.3, 0.2, 0.25, 0.25])
    
    # Create churn with logical relationships
    churn_prob = (
        (contract_types == 'Month-to-month') * 0.3 +
        (monthly_charges > 80) * 0.2 +
        (support_tickets > 5) * 0.3 +
        (tenure_months < 12) * 0.2 +
        np.random.normal(0, 0.1, n_rows)
    )
    
    churned = (churn_prob > 0.4).astype(int)
    
    return pd.DataFrame({
        'customer_age': ages,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'contract_type': contract_types,
        'internet_service': internet_service,
        'support_tickets': support_tickets,
        'payment_method': payment_methods,
        'churned': churned
    })

def loan_approval_dataset(n_rows: int = 1000) -> pd.DataFrame:
    """Generate realistic loan approval prediction dataset."""
    np.random.seed(123)
    
    # Applicant demographics
    ages = np.random.normal(35, 12, n_rows).astype(int)
    ages = np.clip(ages, 18, 75)
    
    # Annual income (log-normal distribution)
    annual_income = np.random.lognormal(np.log(60000), 0.6, n_rows).astype(int)
    annual_income = np.clip(annual_income, 25000, 500000)
    
    # Credit score (normal distribution around 650)
    credit_scores = np.random.normal(650, 120, n_rows).astype(int)
    credit_scores = np.clip(credit_scores, 300, 850)
    
    # Employment length in years
    employment_years = np.random.exponential(5, n_rows)
    employment_years = np.clip(employment_years, 0, 40).round(1)
    
    # Loan amount requested
    loan_amounts = np.random.lognormal(np.log(200000), 0.8, n_rows).astype(int)
    loan_amounts = np.clip(loan_amounts, 50000, 1000000)
    
    # Debt-to-income ratio
    monthly_debt = annual_income * np.random.uniform(0.05, 0.4, n_rows) / 12
    debt_to_income = (monthly_debt * 12 / annual_income).round(3)
    
    # Education level
    education_levels = np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'],
                                       n_rows, p=[0.3, 0.4, 0.25, 0.05])
    
    # Property type
    property_types = np.random.choice(['Single Family', 'Condo', 'Townhouse', 'Multi-family'],
                                     n_rows, p=[0.6, 0.2, 0.15, 0.05])
    
    # Create approval with realistic logic
    approval_score = (
        (credit_scores - 300) / 550 * 0.4 +  # Credit score weight
        (annual_income / 100000) * 0.3 +      # Income weight
        (employment_years / 20) * 0.1 +       # Employment stability
        (1 - debt_to_income) * 0.2 +          # Low debt ratio is good
        np.random.normal(0, 0.1, n_rows)      # Random noise
    )
    
    approved = (approval_score > 0.5).astype(int)
    
    return pd.DataFrame({
        'applicant_age': ages,
        'annual_income': annual_income,
        'credit_score': credit_scores,
        'employment_years': employment_years,
        'loan_amount': loan_amounts,
        'debt_to_income_ratio': debt_to_income,
        'education_level': education_levels,
        'property_type': property_types,
        'approved': approved
    })

# NEW: Realistic Regression Datasets
def house_prices_dataset(n_rows: int = 1000) -> pd.DataFrame:
    """Generate realistic house price prediction dataset."""
    np.random.seed(456)
    
    # House characteristics
    square_feet = np.random.normal(2000, 800, n_rows).astype(int)
    square_feet = np.clip(square_feet, 800, 6000)
    
    # Number of bedrooms (correlated with square footage)
    bedrooms = np.round(square_feet / 500 + np.random.normal(0, 0.5, n_rows)).astype(int)
    bedrooms = np.clip(bedrooms, 1, 8)
    
    # Number of bathrooms
    bathrooms = bedrooms * 0.75 + np.random.normal(0, 0.5, n_rows)
    bathrooms = np.clip(bathrooms, 1, 6).round(1)
    
    # House age
    house_age = np.random.exponential(15, n_rows).astype(int)
    house_age = np.clip(house_age, 0, 100)
    
    # Garage spaces
    garage_spaces = np.random.choice([0, 1, 2, 3], n_rows, p=[0.1, 0.3, 0.5, 0.1])
    
    # Neighborhood quality (1-10 scale)
    neighborhood_quality = np.random.choice(range(1, 11), n_rows, 
                                          p=[0.05, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.05, 0.03, 0.02])
    
    # School district rating
    school_rating = np.random.normal(7, 2, n_rows)
    school_rating = np.clip(school_rating, 1, 10).round(1)
    
    # Distance to city center (miles)
    distance_to_city = np.random.exponential(10, n_rows).round(1)
    distance_to_city = np.clip(distance_to_city, 1, 50)
    
    # Property type
    property_types = np.random.choice(['Single Family', 'Townhouse', 'Condo', 'Ranch'],
                                     n_rows, p=[0.6, 0.2, 0.15, 0.05])
    
    # Calculate price with realistic factors
    base_price = (
        square_feet * 120 +                    # $120 per sq ft base
        bedrooms * 8000 +                      # Bedroom premium
        bathrooms * 5000 +                     # Bathroom premium
        garage_spaces * 3000 +                 # Garage value
        neighborhood_quality * 15000 +         # Neighborhood premium
        school_rating * 8000 +                 # School district value
        -house_age * 500 +                     # Depreciation
        -distance_to_city * 1000 +             # Location penalty
        np.random.normal(0, 25000, n_rows)     # Market noise
    )
    
    # Ensure reasonable price range
    house_price = np.clip(base_price, 100000, 1500000).astype(int)
    
    return pd.DataFrame({
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'house_age_years': house_age,
        'garage_spaces': garage_spaces,
        'neighborhood_quality_score': neighborhood_quality,
        'school_district_rating': school_rating,
        'distance_to_city_miles': distance_to_city,
        'property_type': property_types,
        'sale_price': house_price
    })

def sales_forecast_dataset(n_rows: int = 1000) -> pd.DataFrame:
    """Generate realistic sales forecasting dataset."""
    np.random.seed(789)
    
    # Time-based features (simulate monthly data over several years)
    months = np.random.randint(1, 13, n_rows)
    years = np.random.randint(2020, 2024, n_rows)
    
    # Marketing spend
    marketing_spend = np.random.lognormal(np.log(10000), 0.6, n_rows).astype(int)
    marketing_spend = np.clip(marketing_spend, 2000, 100000)
    
    # Number of sales reps
    sales_reps = np.random.poisson(8, n_rows)
    sales_reps = np.clip(sales_reps, 3, 25)
    
    # Average deal size
    avg_deal_size = np.random.lognormal(np.log(5000), 0.4, n_rows).astype(int)
    avg_deal_size = np.clip(avg_deal_size, 1000, 50000)
    
    # Lead generation activities
    leads_generated = np.random.poisson(100, n_rows)
    leads_generated = np.clip(leads_generated, 20, 500)
    
    # Market conditions (1-10 scale)
    market_conditions = np.random.choice(range(1, 11), n_rows,
                                        p=[0.02, 0.03, 0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.08, 0.02])
    
    # Product category
    product_categories = np.random.choice(['Software', 'Hardware', 'Services', 'Consulting'],
                                         n_rows, p=[0.4, 0.25, 0.2, 0.15])
    
    # Region
    regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'],
                              n_rows, p=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    # Seasonal effects
    seasonal_multiplier = np.where(np.isin(months, [11, 12]), 1.3,  # Holiday boost
                          np.where(np.isin(months, [6, 7, 8]), 0.8,  # Summer slowdown
                                  1.0))
    
    # Calculate sales with realistic relationships
    base_sales = (
        marketing_spend * 0.1 +                # Marketing ROI
        sales_reps * 8000 +                    # Rep productivity
        leads_generated * 50 +                 # Lead conversion
        avg_deal_size * 0.5 +                  # Deal size impact
        market_conditions * 5000 +             # Market conditions
        np.random.normal(0, 10000, n_rows)     # Random variation
    ) * seasonal_multiplier
    
    # Ensure positive sales
    monthly_sales = np.clip(base_sales, 10000, 500000).astype(int)
    
    return pd.DataFrame({
        'month': months,
        'year': years,
        'marketing_spend': marketing_spend,
        'sales_reps_count': sales_reps,
        'avg_deal_size': avg_deal_size,
        'leads_generated': leads_generated,
        'market_conditions_score': market_conditions,
        'product_category': product_categories,
        'region': regions,
        'monthly_sales': monthly_sales
    })

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
    # NEW: Validation failure datasets
    'validation_failure': validation_failure_dataset,
    'validation_failure_correct_columns': validation_failure_correct_columns_dataset,
    # NEW: Realistic classification datasets
    'customer_churn': customer_churn_dataset,
    'loan_approval': loan_approval_dataset,
    # NEW: Realistic regression datasets
    'house_prices': house_prices_dataset,
    'sales_forecast': sales_forecast_dataset,
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