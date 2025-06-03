"""
EDA Validation Module
====================

Team A's module for exploratory data analysis, validation, and cleaning.

This module provides:
- EDA profiling via ydata-profiling
- Data validation using Great Expectations
- Basic data cleaning utilities
- Helper functions for data processing

Usage:
    from eda_validation import ydata_profile, cleaning, utils
    from eda_validation.validation import setup_expectations, run_validation
"""

# Version info
__version__ = "0.1.0"
__author__ = "Team A"

# Make key functions available at module level
from . import ydata_profile
from . import cleaning
from . import utils

__all__ = [
    'ydata_profile',
    'cleaning',
    'utils',
] 