"""
Test suite for the data preparation stage of The Projection Wizard pipeline.

This module contains tests for validating data cleaning, preprocessing, and feature engineering.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tests.unit.stage_tests.base_stage_test import BaseStageTest, TestResult


class PrepStageTest(BaseStageTest):
    """Test suite for the data preparation stage."""

    def __init__(self, test_run_id: str):
        """Initialize the preparation stage test.
        
        Args:
            test_run_id: Unique identifier for this test run
        """
        super().__init__("prep", test_run_id)
        self.required_input_files = [
            "original_data.csv",
            "metadata.json",
            "validation.json"
        ]
        self.required_output_files = [
            "cleaned_data.csv",
            "prep_metadata.json",
            "status.json"
        ]

    def validate_input_files(self) -> Tuple[bool, str]:
        """Validate that all required input files exist and are properly formatted.
        
        Returns:
            Tuple of (success, error_message)
        """
        # Check if files exist
        for file in self.required_input_files:
            if not (self.test_run_dir / file).exists():
                return False, f"Required input file {file} not found"

        # Validate CSV format
        try:
            df = pd.read_csv(self.test_run_dir / "original_data.csv")
            if df.empty:
                return False, "Input CSV file is empty"
        except Exception as e:
            return False, f"Error reading CSV file: {str(e)}"

        # Validate metadata format
        try:
            with open(self.test_run_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                return False, "Metadata must be a JSON object"
        except Exception as e:
            return False, f"Error reading metadata file: {str(e)}"

        # Validate validation results
        try:
            with open(self.test_run_dir / "validation.json", 'r') as f:
                validation = json.load(f)
            if not isinstance(validation, dict):
                return False, "Validation results must be a JSON object"
            if "success" not in validation:
                return False, "Validation results missing 'success' field"
        except Exception as e:
            return False, f"Error reading validation file: {str(e)}"

        return True, ""

    def validate_output_files(self) -> Tuple[bool, str]:
        """Validate that all required output files exist and are properly formatted.
        
        Returns:
            Tuple of (success, error_message)
        """
        # Check if files exist
        for file in self.required_output_files:
            if not (self.test_run_dir / file).exists():
                return False, f"Required output file {file} not found"

        # Validate cleaned data
        try:
            df = pd.read_csv(self.test_run_dir / "cleaned_data.csv")
            if df.empty:
                return False, "Cleaned data file is empty"
        except Exception as e:
            return False, f"Error reading cleaned data file: {str(e)}"

        # Validate prep metadata
        try:
            with open(self.test_run_dir / "prep_metadata.json", 'r') as f:
                prep_metadata = json.load(f)
            if not isinstance(prep_metadata, dict):
                return False, "Prep metadata must be a JSON object"
            required_fields = ["preprocessing_steps", "feature_engineering", "encoders"]
            for field in required_fields:
                if field not in prep_metadata:
                    return False, f"Prep metadata missing required field: {field}"
        except Exception as e:
            return False, f"Error reading prep metadata file: {str(e)}"

        # Validate status file
        try:
            with open(self.test_run_dir / "status.json", 'r') as f:
                status = json.load(f)
            if not isinstance(status, dict):
                return False, "Status must be a JSON object"
            if "status" not in status:
                return False, "Status file missing 'status' field"
            if status["status"] != "completed":
                return False, "Prep stage not marked as completed"
        except Exception as e:
            return False, f"Error reading status file: {str(e)}"

        return True, ""

    def run_prep_checks(self) -> Tuple[bool, str]:
        """Run specific preparation checks for this stage.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Load the data and metadata
            original_df = pd.read_csv(self.test_run_dir / "original_data.csv")
            cleaned_df = pd.read_csv(self.test_run_dir / "cleaned_data.csv")
            with open(self.test_run_dir / "prep_metadata.json", 'r') as f:
                prep_metadata = json.load(f)

            # Check data shape
            if cleaned_df.shape[0] != original_df.shape[0]:
                return False, "Number of rows changed during preparation"

            # Check for missing values
            if cleaned_df.isnull().any().any():
                return False, "Cleaned data contains missing values"

            # Check preprocessing steps
            if "preprocessing_steps" not in prep_metadata:
                return False, "Missing preprocessing steps in metadata"

            # Check feature engineering
            if "feature_engineering" not in prep_metadata:
                return False, "Missing feature engineering info in metadata"

            # Check encoders
            if "encoders" not in prep_metadata:
                return False, "Missing encoder information in metadata"

            # Verify categorical encoding
            for col, encoder_info in prep_metadata["encoders"].items():
                if col in cleaned_df.columns:
                    if encoder_info["type"] == "label":
                        # Check if values are numeric
                        if not pd.to_numeric(cleaned_df[col], errors='coerce').notnull().all():
                            return False, f"Label encoding failed for column {col}"
                    elif encoder_info["type"] == "onehot":
                        # Check if one-hot columns exist
                        prefix = f"{col}_"
                        onehot_cols = [c for c in cleaned_df.columns if c.startswith(prefix)]
                        if not onehot_cols:
                            return False, f"One-hot encoding columns missing for {col}"

            return True, ""

        except Exception as e:
            return False, f"Error during preparation checks: {str(e)}"

    def validate_prep_metadata(self) -> Tuple[bool, str]:
        """Validate that prep_metadata.json contains all required fields."""
        try:
            with open(self.test_run_dir / "prep_metadata.json", 'r') as f:
                prep_metadata = json.load(f)
            if not isinstance(prep_metadata, dict):
                return False, "Prep metadata must be a JSON object"
            required_fields = [
                "preprocessing_steps",
                "feature_engineering",
                "encoders",
                "task_type",
                "target_column"
            ]
            for field in required_fields:
                if field not in prep_metadata:
                    return False, f"Prep metadata missing required field: {field}"
        except Exception as e:
            return False, f"Error reading prep metadata file: {str(e)}"
        return True, ""

    def run_test(self) -> TestResult:
        """Run the preparation stage test."""
        self.log_test_start()
        start_time = time.time()
        
        # Validate input files
        input_success, input_error = self.validate_input_files()
        if not input_success:
            return TestResult(
                stage_name="prep",
                success=False,
                duration=time.time() - start_time,
                details={"input_validation": False},
                error_message=input_error
            )
        
        # Validate output files
        output_success, output_error = self.validate_output_files()
        if not output_success:
            return TestResult(
                stage_name="prep",
                success=False,
                duration=time.time() - start_time,
                details={"output_validation": False},
                error_message=output_error
            )
        
        # Validate prep metadata
        metadata_success, metadata_error = self.validate_prep_metadata()
        if not metadata_success:
            return TestResult(
                stage_name="prep",
                success=False,
                duration=time.time() - start_time,
                details={"prep_metadata": False},
                error_message=metadata_error
            )
        
        # Validate status file
        status_success, status_data = self.check_status_file("prep")
        if not status_success:
            return TestResult(
                stage_name="prep",
                success=False,
                duration=time.time() - start_time,
                details={"status_check": False},
                error_message="Status validation failed"
            )
        
        # All validations passed
        self.log_test_end(True, time.time() - start_time)
        return TestResult(
            stage_name="prep",
            success=True,
            duration=time.time() - start_time,
            details={
                "input_validation": True,
                "output_validation": True,
                "prep_metadata": True,
                "status_check": True
            }
        )


def run_prep_test(test_run_id: str) -> TestResult:
    """Run the preparation stage test for a specific test run.
    
    Args:
        test_run_id: Unique identifier for the test run
        
    Returns:
        TestResult object containing test results
    """
    test = PrepStageTest(test_run_id)
    return test.run_test() 