"""
Data Cleaning Module
===================

Handles basic data cleaning operations like dropping NA values, renaming columns,
and basic preprocessing steps.

Usage:
    # As a module
    from eda_validation.cleaning import clean_dataframe, handle_missing_values
    
    # As CLI
    python eda_validation/cleaning.py data/raw/sample.csv
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
    threshold: float = 0.5,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values 
                       ('drop', 'fill_mean', 'fill_median', 'fill_mode', 'forward_fill')
        threshold (float): For 'drop' strategy, drop rows/cols with more than this fraction of NAs
        columns (List[str], optional): Specific columns to process, if None processes all
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
        
    Example:
        >>> df_clean = handle_missing_values(df, strategy="fill_mean", threshold=0.8)
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns.tolist()
    
    logger.info(f"Handling missing values with strategy: {strategy}")
    logger.info(f"Missing values before cleaning: {df_clean[columns].isnull().sum().sum()}")
    
    try:
        if strategy == "drop":
            # Drop columns with too many missing values
            cols_to_drop = []
            for col in columns:
                missing_fraction = df_clean[col].isnull().sum() / len(df_clean)
                if missing_fraction > threshold:
                    cols_to_drop.append(col)
                    logger.info(f"Dropping column '{col}' with {missing_fraction:.2%} missing values")
            
            df_clean = df_clean.drop(columns=cols_to_drop)
            
            # Drop rows with any remaining missing values
            df_clean = df_clean.dropna()
            
        elif strategy == "fill_mean":
            for col in columns:
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    
        elif strategy == "fill_median":
            for col in columns:
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    
        elif strategy == "fill_mode":
            for col in columns:
                mode_value = df_clean[col].mode()
                if len(mode_value) > 0:
                    df_clean[col].fillna(mode_value[0], inplace=True)
                    
        elif strategy == "forward_fill":
            df_clean[columns] = df_clean[columns].fillna(method='ffill')
            
        else:
            logger.error(f"Unknown strategy: {strategy}")
            return df
        
        logger.info(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error handling missing values: {str(e)}")
        raise


def standardize_column_names(
    df: pd.DataFrame,
    naming_convention: str = "snake_case"
) -> pd.DataFrame:
    """
    Standardize column names according to specified convention.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        naming_convention (str): Naming convention ('snake_case', 'camel_case', 'lower')
        
    Returns:
        pd.DataFrame: DataFrame with standardized column names
        
    Example:
        >>> df_clean = standardize_column_names(df, "snake_case")
    """
    df_clean = df.copy()
    
    logger.info(f"Standardizing column names to: {naming_convention}")
    logger.info(f"Original columns: {list(df.columns)}")
    
    try:
        if naming_convention == "snake_case":
            # Convert to snake_case
            new_columns = []
            for col in df_clean.columns:
                # Replace spaces and special chars with underscores
                new_col = str(col).lower()
                new_col = new_col.replace(' ', '_')
                new_col = new_col.replace('-', '_')
                # Remove multiple underscores
                while '__' in new_col:
                    new_col = new_col.replace('__', '_')
                new_col = new_col.strip('_')
                new_columns.append(new_col)
                
        elif naming_convention == "lower":
            new_columns = [str(col).lower().strip() for col in df_clean.columns]
            
        elif naming_convention == "camel_case":
            # Convert to camelCase
            new_columns = []
            for col in df_clean.columns:
                words = str(col).lower().replace('_', ' ').replace('-', ' ').split()
                if words:
                    new_col = words[0] + ''.join(word.capitalize() for word in words[1:])
                    new_columns.append(new_col)
                else:
                    new_columns.append(str(col))
        else:
            logger.error(f"Unknown naming convention: {naming_convention}")
            return df
        
        df_clean.columns = new_columns
        logger.info(f"New columns: {list(df_clean.columns)}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error standardizing column names: {str(e)}")
        raise


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first"
) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (List[str], optional): Columns to consider for duplicates
        keep (str): Which duplicates to keep ('first', 'last', False)
        
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
        
    Example:
        >>> df_clean = remove_duplicates(df, keep="first")
    """
    df_clean = df.copy()
    
    initial_rows = len(df_clean)
    logger.info(f"Removing duplicates. Initial rows: {initial_rows}")
    
    try:
        df_clean = df_clean.drop_duplicates(subset=subset, keep=keep)
        final_rows = len(df_clean)
        removed_rows = initial_rows - final_rows
        
        logger.info(f"Removed {removed_rows} duplicate rows. Final rows: {final_rows}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error removing duplicates: {str(e)}")
        raise


def detect_and_convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically detect and convert data types.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with optimized data types
        
    Example:
        >>> df_clean = detect_and_convert_dtypes(df)
    """
    df_clean = df.copy()
    
    logger.info("Detecting and converting data types")
    logger.info(f"Original dtypes:\n{df.dtypes}")
    
    try:
        # Try to convert to numeric where possible
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                
                # If most values can be converted, use numeric type
                non_null_original = df_clean[col].notna().sum()
                non_null_numeric = numeric_series.notna().sum()
                
                if non_null_numeric / non_null_original > 0.8:  # 80% threshold
                    df_clean[col] = numeric_series
                    logger.info(f"Converted column '{col}' to numeric")
                    
                # Try to convert to datetime
                elif col.lower() in ['date', 'time', 'timestamp'] or 'date' in col.lower():
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                        logger.info(f"Converted column '{col}' to datetime")
                    except:
                        pass
        
        logger.info(f"Final dtypes:\n{df_clean.dtypes}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error detecting/converting dtypes: {str(e)}")
        raise


def clean_dataframe(
    df: pd.DataFrame,
    missing_strategy: str = "drop",
    missing_threshold: float = 0.5,
    standardize_columns: bool = True,
    remove_dups: bool = True,
    convert_dtypes: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for missing values
        missing_threshold (float): Threshold for dropping columns/rows
        standardize_columns (bool): Whether to standardize column names
        remove_dups (bool): Whether to remove duplicates
        convert_dtypes (bool): Whether to auto-convert data types
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Cleaned DataFrame and cleaning report
        
    Example:
        >>> df_clean, report = clean_dataframe(df)
    """
    logger.info("Starting complete data cleaning pipeline")
    
    # Initialize report
    report = {
        "original_shape": df.shape,
        "original_columns": list(df.columns),
        "original_dtypes": df.dtypes.to_dict(),
        "missing_values_original": df.isnull().sum().to_dict(),
        "steps_performed": []
    }
    
    df_clean = df.copy()
    
    try:
        # Step 1: Handle missing values
        if missing_strategy != "none":
            df_clean = handle_missing_values(df_clean, missing_strategy, missing_threshold)
            report["steps_performed"].append(f"handled_missing_values_{missing_strategy}")
        
        # Step 2: Standardize column names
        if standardize_columns:
            df_clean = standardize_column_names(df_clean, "snake_case")
            report["steps_performed"].append("standardized_column_names")
        
        # Step 3: Remove duplicates
        if remove_dups:
            df_clean = remove_duplicates(df_clean)
            report["steps_performed"].append("removed_duplicates")
        
        # Step 4: Convert data types
        if convert_dtypes:
            df_clean = detect_and_convert_dtypes(df_clean)
            report["steps_performed"].append("converted_dtypes")
        
        # Update report with final state
        report.update({
            "final_shape": df_clean.shape,
            "final_columns": list(df_clean.columns),
            "final_dtypes": df_clean.dtypes.to_dict(),
            "missing_values_final": df_clean.isnull().sum().to_dict(),
            "rows_removed": df.shape[0] - df_clean.shape[0],
            "columns_removed": df.shape[1] - df_clean.shape[1]
        })
        
        logger.info(f"Cleaning completed. Shape changed from {df.shape} to {df_clean.shape}")
        
        return df_clean, report
        
    except Exception as e:
        logger.error(f"Error in cleaning pipeline: {str(e)}")
        raise


def clean_csv_file(
    input_path: str,
    output_path: Optional[str] = None,
    **cleaning_params
) -> bool:
    """
    Complete pipeline: load CSV, clean data, and save results.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str, optional): Path for output CSV file
        **cleaning_params: Parameters for clean_dataframe function
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load data
        logger.info(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Clean data
        df_clean, report = clean_dataframe(df, **cleaning_params)
        
        # Determine output path
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = f"data/processed/{input_path_obj.stem}_cleaned.csv"
        
        # Save cleaned data
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        df_clean.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to: {output_path}")
        
        # Save cleaning report
        report_path = output_path_obj.with_suffix('.json')
        import json
        with open(report_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_report = {}
            for k, v in report.items():
                if isinstance(v, dict):
                    serializable_report[k] = {str(k2): str(v2) for k2, v2 in v.items()}
                else:
                    serializable_report[k] = str(v) if not isinstance(v, (list, str, int, float)) else v
            json.dump(serializable_report, f, indent=2)
        
        logger.info(f"Cleaning report saved to: {report_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in clean_csv_file: {str(e)}")
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Clean CSV data with various preprocessing steps"
    )
    parser.add_argument(
        "input_path",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for cleaned CSV (default: auto-generated)"
    )
    parser.add_argument(
        "--missing-strategy",
        choices=["drop", "fill_mean", "fill_median", "fill_mode", "forward_fill"],
        default="drop",
        help="Strategy for handling missing values"
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.5,
        help="Threshold for dropping columns/rows with missing values"
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Skip column name standardization"
    )
    parser.add_argument(
        "--no-remove-duplicates",
        action="store_true",
        help="Skip duplicate removal"
    )
    parser.add_argument(
        "--no-convert-dtypes",
        action="store_true",
        help="Skip automatic data type conversion"
    )
    
    args = parser.parse_args()
    
    # Run cleaning
    success = clean_csv_file(
        input_path=args.input_path,
        output_path=args.output,
        missing_strategy=args.missing_strategy,
        missing_threshold=args.missing_threshold,
        standardize_columns=not args.no_standardize,
        remove_dups=not args.no_remove_duplicates,
        convert_dtypes=not args.no_convert_dtypes
    )
    
    if success:
        print("✅ Data cleaning completed successfully")
    else:
        print("❌ Data cleaning failed")
        exit(1)


if __name__ == "__main__":
    main() 