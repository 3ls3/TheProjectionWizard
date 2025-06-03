"""
Utility Functions Module
========================

Reusable helper functions for the EDA validation pipeline.

This module provides:
- File handling utilities
- Data type detection and conversion helpers
- Logging and reporting utilities
- Configuration management

Usage:
    from eda_validation.utils import detect_file_type, setup_logging
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import hashlib
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for the pipeline.
    
    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): Path to log file
        format_string (str, optional): Custom format string
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = setup_logging("DEBUG", "pipeline.log")
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )
    
    logger = logging.getLogger("eda_validation")
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def detect_file_type(file_path: str) -> str:
    """
    Detect the type of data file based on extension and content.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: File type ('csv', 'json', 'excel', 'parquet', 'unknown')
        
    Example:
        >>> file_type = detect_file_type("data.csv")
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    type_mapping = {
        '.csv': 'csv',
        '.tsv': 'csv',
        '.txt': 'csv',
        '.json': 'json',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.parquet': 'parquet',
        '.pq': 'parquet'
    }
    
    return type_mapping.get(extension, 'unknown')


def load_data_file(
    file_path: str,
    file_type: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load data file into pandas DataFrame with automatic type detection.
    
    Args:
        file_path (str): Path to the data file
        file_type (str, optional): File type override
        **kwargs: Additional arguments for pandas readers
        
    Returns:
        pd.DataFrame: Loaded data
        
    Example:
        >>> df = load_data_file("data.csv")
    """
    if file_type is None:
        file_type = detect_file_type(file_path)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading {file_type} file: {file_path}")
    
    try:
        if file_type == 'csv':
            # Try to detect separator
            if 'sep' not in kwargs:
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                    if '\t' in first_line:
                        kwargs['sep'] = '\t'
                    elif ';' in first_line:
                        kwargs['sep'] = ';'
                    else:
                        kwargs['sep'] = ','
                        
            df = pd.read_csv(file_path, **kwargs)
            
        elif file_type == 'json':
            df = pd.read_json(file_path, **kwargs)
            
        elif file_type == 'excel':
            df = pd.read_excel(file_path, **kwargs)
            
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path, **kwargs)
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file for integrity checking.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: MD5 hash of the file
        
    Example:
        >>> hash_value = calculate_file_hash("data.csv")
    """
    hash_md5 = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        Dict with DataFrame statistics and information
        
    Example:
        >>> info = get_dataframe_info(df)
    """
    info = {
        "shape": df.shape,
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "column_count": len(df.columns),
        "row_count": len(df),
        "duplicate_rows": df.duplicated().sum(),
        "total_missing_values": df.isnull().sum().sum(),
        "columns": {},
        "dtypes": df.dtypes.to_dict()
    }
    
    # Per-column information
    for col in df.columns:
        col_data = df[col]
        
        col_info = {
            "dtype": str(col_data.dtype),
            "missing_count": col_data.isnull().sum(),
            "missing_percentage": col_data.isnull().sum() / len(df) * 100,
            "unique_count": col_data.nunique(),
            "unique_percentage": col_data.nunique() / len(df) * 100
        }
        
        # Add type-specific statistics
        if col_data.dtype in ['int64', 'float64']:
            col_info.update({
                "min": col_data.min(),
                "max": col_data.max(),
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std()
            })
        elif col_data.dtype == 'object':
            # String statistics
            str_lengths = col_data.dropna().astype(str).str.len()
            if len(str_lengths) > 0:
                col_info.update({
                    "avg_length": str_lengths.mean(),
                    "min_length": str_lengths.min(),
                    "max_length": str_lengths.max()
                })
        
        info["columns"][col] = col_info
    
    return info


def detect_column_types(df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, str]:
    """
    Detect semantic column types (beyond pandas dtypes).
    
    Args:
        df (pd.DataFrame): Input DataFrame
        sample_size (int): Number of rows to sample for detection
        
    Returns:
        Dict mapping column names to detected types
        
    Example:
        >>> types = detect_column_types(df)
    """
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df
    
    column_types = {}
    
    for col in sample_df.columns:
        col_data = sample_df[col].dropna()
        
        if len(col_data) == 0:
            column_types[col] = "empty"
            continue
        
        # Check for ID columns
        if col.lower() in ['id', 'index', 'key'] or col.lower().endswith('_id'):
            if col_data.nunique() / len(col_data) > 0.95:
                column_types[col] = "identifier"
                continue
        
        # Check for categorical data
        unique_ratio = col_data.nunique() / len(col_data)
        if unique_ratio < 0.05 or col_data.nunique() < 20:
            column_types[col] = "categorical"
            continue
        
        # Check data type
        if col_data.dtype in ['int64', 'float64']:
            # Check if it looks like a year
            if col.lower() in ['year', 'yr'] or 'year' in col.lower():
                if col_data.min() > 1900 and col_data.max() < 2100:
                    column_types[col] = "year"
                    continue
            
            # Check for binary numeric
            unique_values = set(col_data.unique())
            if unique_values.issubset({0, 1}) or unique_values.issubset({0.0, 1.0}):
                column_types[col] = "binary"
                continue
            
            column_types[col] = "numeric"
            
        elif col_data.dtype == 'object':
            # Try to detect dates
            if col.lower() in ['date', 'time', 'timestamp'] or any(word in col.lower() for word in ['date', 'time']):
                try:
                    pd.to_datetime(col_data.iloc[:100], errors='raise')
                    column_types[col] = "datetime"
                    continue
                except:
                    pass
            
            # Check for email-like patterns
            if col.lower() in ['email', 'mail', 'e_mail']:
                if col_data.str.contains('@').any():
                    column_types[col] = "email"
                    continue
            
            # Check for URL-like patterns
            if col.lower() in ['url', 'link', 'website']:
                if col_data.str.contains('http').any():
                    column_types[col] = "url"
                    continue
            
            # Default to text
            column_types[col] = "text"
        
        else:
            column_types[col] = "other"
    
    return column_types


def save_pipeline_metadata(
    metadata: Dict[str, Any],
    output_path: str
) -> bool:
    """
    Save pipeline metadata and configuration.
    
    Args:
        metadata (Dict): Metadata to save
        output_path (str): Path to save metadata file
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> save_pipeline_metadata(meta, "pipeline_metadata.json")
    """
    try:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        metadata["saved_at"] = datetime.now().isoformat()
        metadata["pipeline_version"] = "1.0.0"
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error saving metadata: {str(e)}")
        return False


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    Validate file path and check existence.
    
    Args:
        file_path (str): Path to validate
        must_exist (bool): Whether the file must exist
        
    Returns:
        bool: True if valid, False otherwise
        
    Example:
        >>> is_valid = validate_file_path("data.csv", must_exist=True)
    """
    try:
        path = Path(file_path)
        
        if must_exist and not path.exists():
            return False
        
        if must_exist and path.is_dir():
            return False
        
        # Check if parent directory exists (for output files)
        if not must_exist and not path.parent.exists():
            return False
        
        return True
        
    except Exception:
        return False


def create_summary_report(
    input_file: str,
    dataframe_info: Dict[str, Any],
    cleaning_report: Optional[Dict[str, Any]] = None,
    validation_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive summary report of the pipeline run.
    
    Args:
        input_file (str): Path to input file
        dataframe_info (Dict): DataFrame information
        cleaning_report (Dict, optional): Data cleaning report
        validation_results (Dict, optional): Validation results
        
    Returns:
        Dict containing summary report
        
    Example:
        >>> report = create_summary_report("data.csv", df_info, cleaning, validation)
    """
    report = {
        "pipeline_run": {
            "timestamp": datetime.now().isoformat(),
            "input_file": input_file,
            "file_hash": calculate_file_hash(input_file) if Path(input_file).exists() else None
        },
        "data_summary": dataframe_info,
        "processing_steps": []
    }
    
    if cleaning_report:
        report["cleaning_report"] = cleaning_report
        report["processing_steps"].append("data_cleaning")
    
    if validation_results:
        report["validation_results"] = validation_results
        report["processing_steps"].append("data_validation")
    
    # Calculate overall quality score
    quality_score = 100
    
    if dataframe_info.get("total_missing_values", 0) > 0:
        missing_percentage = (dataframe_info["total_missing_values"] / 
                            (dataframe_info["row_count"] * dataframe_info["column_count"]) * 100)
        quality_score -= min(missing_percentage, 20)  # Max 20 point deduction
    
    if dataframe_info.get("duplicate_rows", 0) > 0:
        duplicate_percentage = dataframe_info["duplicate_rows"] / dataframe_info["row_count"] * 100
        quality_score -= min(duplicate_percentage, 15)  # Max 15 point deduction
    
    if validation_results and not validation_results.get("overall_success", True):
        failed_percentage = (validation_results.get("failed_expectations", 0) / 
                           validation_results.get("total_expectations", 1) * 100)
        quality_score -= min(failed_percentage, 30)  # Max 30 point deduction
    
    report["quality_score"] = max(0, quality_score)
    
    return report 