"""
Validation Submodule
===================

Great Expectations based data validation for Team A's pipeline.

This submodule provides:
- Setup and configuration of Great Expectations
- Running validation checks against user data
- Generating validation reports

Usage:
    from eda_validation.validation import setup_expectations, run_validation
"""

from . import setup_expectations
from . import run_validation

__all__ = [
    'setup_expectations',
    'run_validation',
] 