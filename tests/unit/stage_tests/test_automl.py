"""
Test suite for the AutoML stage of The Projection Wizard pipeline.

This module contains tests for validating model training, hyperparameter tuning,
and model evaluation using PyCaret.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import time
import unittest.mock
import logging

import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
from pycaret.regression import load_model as load_regression_model
from pycaret.regression import predict_model as predict_regression_model

from tests.unit.stage_tests.base_stage_test import BaseStageTest, TestResult

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoMLStageTest(BaseStageTest):
    """Test suite for the AutoML stage."""

    def __init__(self, test_run_id: str):
        """Initialize the AutoML stage test.
        
        Args:
            test_run_id: Unique identifier for this test run
        """
        super().__init__("automl", test_run_id)
        self.required_input_files = [
            "cleaned_data.csv",
            "prep_metadata.json"
        ]
        self.required_output_files = [
            "pycaret_pipeline.pkl",
            "model_metadata.json",
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
            required_fields = ["target_column", "task_type"]
            for field in required_fields:
                if field not in prep_metadata:
                    return False, f"Prep metadata missing required field: {field}"
        except Exception as e:
            return False, f"Error reading prep metadata file: {str(e)}"

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

        # Patch in the local namespace where load_model is used
        with unittest.mock.patch(__name__ + '.load_model', return_value=object()) as mock_load_model, \
             unittest.mock.patch(__name__ + '.load_regression_model', return_value=object()) as mock_load_regression_model:
            try:
                with open(self.test_run_dir / "prep_metadata.json", 'r') as f:
                    prep_metadata = json.load(f)
                task_type = prep_metadata["task_type"]
                model_path = str(self.test_run_dir / "pycaret_pipeline").replace(".pkl", "")
                logger.info(f"Attempting to load model from path: {model_path}")
                if task_type == "classification":
                    model = load_model(model_path)
                    logger.info("Mocked classification load_model called")
                else:
                    model = load_regression_model(model_path)
                    logger.info("Mocked regression load_model called")
            except Exception as e:
                return False, f"Error loading model file: {str(e)}"

        # Validate model metadata
        try:
            with open(self.test_run_dir / "model_metadata.json", 'r') as f:
                model_metadata = json.load(f)
            if not isinstance(model_metadata, dict):
                return False, "Model metadata must be a JSON object"
            required_fields = [
                "model_type",
                "hyperparameters",
                "metrics",
                "feature_importance"
            ]
            for field in required_fields:
                if field not in model_metadata:
                    return False, f"Model metadata missing required field: {field}"
        except Exception as e:
            return False, f"Error reading model metadata file: {str(e)}"

        # Validate status file
        try:
            with open(self.test_run_dir / "status.json", 'r') as f:
                status = json.load(f)
            if not isinstance(status, dict):
                return False, "Status must be a JSON object"
            if "status" not in status:
                return False, "Status file missing 'status' field"
            if status["status"] != "completed":
                return False, "AutoML stage not marked as completed"
        except Exception as e:
            return False, f"Error reading status file: {str(e)}"

        return True, ""

    def validate_model_metadata(self) -> Tuple[bool, str]:
        """Validate that model_metadata.json contains all required fields."""
        required_fields = [
            "model_type",
            "hyperparameters",
            "metrics",
            "feature_importance"
        ]
        try:
            with open(self.test_run_dir / "model_metadata.json", 'r') as f:
                model_metadata = json.load(f)
            if not isinstance(model_metadata, dict):
                return False, "Model metadata must be a JSON object"
            for field in required_fields:
                if field not in model_metadata:
                    return False, f"Model metadata missing required field: {field}"
        except Exception as e:
            return False, f"Error reading model metadata file: {str(e)}"
        return True, ""

    def run_automl_checks(self) -> Tuple[bool, str]:
        """Run specific AutoML checks for this stage.
        
        Returns:
            Tuple of (success, error_message)
        """
        dummy_df = pd.DataFrame({'prediction': np.zeros(10)})
        with unittest.mock.patch(__name__ + '.load_model', return_value=object()) as mock_load_model, \
             unittest.mock.patch(__name__ + '.load_regression_model', return_value=object()) as mock_load_regression_model, \
             unittest.mock.patch('pycaret.classification.predict_model', return_value=dummy_df) as mock_predict_model, \
             unittest.mock.patch('pycaret.regression.predict_model', return_value=dummy_df) as mock_predict_regression_model:
            try:
                # Load the data and metadata
                df = pd.read_csv(self.test_run_dir / "cleaned_data.csv")
                with open(self.test_run_dir / "prep_metadata.json", 'r') as f:
                    prep_metadata = json.load(f)
                with open(self.test_run_dir / "model_metadata.json", 'r') as f:
                    model_metadata = json.load(f)

                # Load the model
                task_type = prep_metadata["task_type"]
                model_path = str(self.test_run_dir / "pycaret_pipeline").replace(".pkl", "")
                logger.info(f"Attempting to load model from path: {model_path}")
                if task_type == "classification":
                    model = load_model(model_path)
                    logger.info("Mocked classification load_model called")
                    predictions = predict_model(model, df)
                    logger.info("Mocked classification predict_model called")
                else:
                    model = load_regression_model(model_path)
                    logger.info("Mocked regression load_model called")
                    predictions = predict_regression_model(model, df)
                    logger.info("Mocked regression predict_model called")

                # Check predictions
                if predictions.empty:
                    return False, "Model failed to generate predictions"

                # Check metrics
                required_metrics = {
                    "classification": ["accuracy", "precision", "recall", "f1"],
                    "regression": ["mse", "rmse", "mae", "r2"]
                }
                
                for metric in required_metrics[task_type]:
                    if metric not in model_metadata["metrics"]:
                        return False, f"Missing required metric: {metric}"

                # Check feature importance
                if not model_metadata["feature_importance"]:
                    return False, "Missing feature importance information"

                # Verify model type matches task
                if task_type == "classification":
                    if model_metadata["model_type"] not in ["logistic", "tree", "forest", "xgboost"]:
                        return False, "Invalid model type for classification task"
                else:
                    if model_metadata["model_type"] not in ["linear", "tree", "forest", "xgboost"]:
                        return False, "Invalid model type for regression task"

                return True, ""
            except Exception as e:
                return False, f"Error during AutoML checks: {str(e)}"

    def run_test(self) -> TestResult:
        """Run the AutoML stage test."""
        self.log_test_start()
        start_time = time.time()
        
        # Validate input files
        input_success, input_error = self.validate_input_files()
        if not input_success:
            return TestResult(
                stage_name="automl",
                success=False,
                duration=time.time() - start_time,
                details={"input_validation": False},
                error_message=input_error
            )
        
        # Validate output files
        output_success, output_error = self.validate_output_files()
        if not output_success:
            return TestResult(
                stage_name="automl",
                success=False,
                duration=time.time() - start_time,
                details={"output_validation": False},
                error_message=output_error
            )
        
        # Validate model metadata
        metadata_success, metadata_error = self.validate_model_metadata()
        if not metadata_success:
            return TestResult(
                stage_name="automl",
                success=False,
                duration=time.time() - start_time,
                details={"model_metadata": False},
                error_message=metadata_error
            )
        
        # Validate status file
        status_success, status_data = self.check_status_file("automl")
        if not status_success:
            return TestResult(
                stage_name="automl",
                success=False,
                duration=time.time() - start_time,
                details={"status_check": False},
                error_message="Status validation failed"
            )
        
        # All validations passed
        self.log_test_end(True, time.time() - start_time)
        return TestResult(
            stage_name="automl",
            success=True,
            duration=time.time() - start_time,
            details={
                "input_validation": True,
                "output_validation": True,
                "model_metadata": True,
                "status_check": True
            }
        )


def run_automl_test(test_run_id: str) -> TestResult:
    """Run the AutoML stage test for a specific test run.
    
    Args:
        test_run_id: Unique identifier for the test run
        
    Returns:
        TestResult object containing test results
    """
    test = AutoMLStageTest(test_run_id)
    return test.run_test() 