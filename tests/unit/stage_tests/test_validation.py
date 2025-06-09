"""
Test suite for the data validation stage of The Projection Wizard pipeline.

This module contains tests for validating data quality and schema compliance
using Great Expectations.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import time

import pandas as pd
from great_expectations.core import ExpectationSuite

from tests.unit.stage_tests.base_stage_test import BaseStageTest, TestResult


class ValidationStageTest(BaseStageTest):
    """Test suite for the data validation stage."""

    def __init__(self, test_run_id: str):
        """Initialize the validation stage test.
        
        Args:
            test_run_id: Unique identifier for this test run
        """
        super().__init__("validation", test_run_id)
        self.required_input_files = [
            "original_data.csv",
            "metadata.json"
        ]
        self.required_output_files = [
            "validation.json",
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

        # Validate status file
        try:
            with open(self.test_run_dir / "status.json", 'r') as f:
                status = json.load(f)
            if not isinstance(status, dict):
                return False, "Status must be a JSON object"
            if "status" not in status:
                return False, "Status file missing 'status' field"
            if status["status"] != "completed":
                return False, "Validation stage not marked as completed"
        except Exception as e:
            return False, f"Error reading status file: {str(e)}"

        return True, ""

    def run_validation_checks(self) -> Tuple[bool, str]:
        """Run specific validation checks for this stage.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Load the data and metadata
            df = pd.read_csv(self.test_run_dir / "original_data.csv")
            with open(self.test_run_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)

            # Get validation results
            with open(self.test_run_dir / "validation.json", 'r') as f:
                validation_results = json.load(f)

            # Check if all required validations were performed
            required_checks = [
                "columns_exist",
                "no_missing_values",
                "data_types_match"
            ]
            
            for check in required_checks:
                if check not in validation_results:
                    return False, f"Missing required validation check: {check}"

            # Verify validation results match actual data
            if validation_results["columns_exist"]:
                for col in df.columns:
                    if col not in validation_results["columns_exist"]:
                        return False, f"Column {col} not validated"

            return True, ""

        except Exception as e:
            return False, f"Error during validation checks: {str(e)}"

    def run_test(self) -> TestResult:
        """Run the validation stage test."""
        self.log_test_start()
        start_time = time.time()
        
        # Validate input files
        input_success, input_error = self.validate_input_files()
        if not input_success:
            return TestResult(
                stage_name="validation",
                success=False,
                duration=time.time() - start_time,
                details={"input_validation": False},
                error_message=input_error
            )
        
        # Validate output files
        output_success, output_error = self.validate_output_files()
        if not output_success:
            return TestResult(
                stage_name="validation",
                success=False,
                duration=time.time() - start_time,
                details={"output_validation": False},
                error_message=output_error
            )
        
        # Validate validation results
        validation_success, validation_error = self.run_validation_checks()
        if not validation_success:
            return TestResult(
                stage_name="validation",
                success=False,
                duration=time.time() - start_time,
                details={"validation_results": False},
                error_message=validation_error
            )
        
        # Validate status file
        status_success, status_data = self.check_status_file("validation")
        if not status_success:
            return TestResult(
                stage_name="validation",
                success=False,
                duration=time.time() - start_time,
                details={"status_check": False},
                error_message="Status validation failed"
            )
        
        # All validations passed
        self.log_test_end(True, time.time() - start_time)
        return TestResult(
            stage_name="validation",
            success=True,
            duration=time.time() - start_time,
            details={
                "input_validation": True,
                "output_validation": True,
                "validation_results": True,
                "status_check": True
            }
        )


def run_validation_test(test_run_id: str) -> TestResult:
    """Run the validation stage test for a specific test run.
    
    Args:
        test_run_id: Unique identifier for the test run
        
    Returns:
        TestResult object containing test results
    """
    test = ValidationStageTest(test_run_id)
    return test.run_test() 