"""
Test fixture generator for The Projection Wizard pipeline.

This module provides utilities to create controlled test environments for each pipeline stage.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any
import uuid
from datetime import datetime

import pandas as pd


class TestFixtureGenerator:
    """Generator for test fixtures for each pipeline stage."""

    def __init__(self):
        """Initialize the fixture generator."""
        self.base_dir = Path("data/test_runs")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.sample_data_path = Path(__file__).parent.parent / "data" / "sample_classification.csv"

    def setup_ingestion_fixture(self, task_type: str) -> str:
        """
        Create fixtures for the ingestion stage.

        Args:
            task_type: Type of ML task ("classification" or "regression")

        Returns:
            Test run ID for the ingestion stage
        """
        test_run_id = f"ingestion_{task_type}_{os.urandom(4).hex()}"
        test_dir = self.base_dir / test_run_id
        test_dir.mkdir(parents=True, exist_ok=True)

        # Copy sample data file
        sample_file = Path(__file__).parent.parent / "data" / f"sample_{task_type}.csv"
        shutil.copy(sample_file, test_dir / "original_data.csv")

        # Create minimal metadata
        metadata = {
            "task_type": task_type,
            "target_column": "target" if task_type == "classification" else "price"
        }
        with open(test_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return test_run_id

    def setup_schema_fixture(self, task_type: str) -> str:
        """
        Create fixtures for the schema stage.

        Args:
            task_type: Type of ML task ("classification" or "regression")

        Returns:
            Test run ID for the schema stage
        """
        test_run_id = f"schema_{task_type}_{os.urandom(4).hex()}"
        test_dir = self.base_dir / test_run_id
        test_dir.mkdir(parents=True, exist_ok=True)

        # Copy sample data file
        sample_file = Path(__file__).parent.parent / "data" / f"sample_{task_type}.csv"
        shutil.copy(sample_file, test_dir / "original_data.csv")

        # Create minimal metadata with schema
        metadata = {
            "task_type": task_type,
            "target_column": "target" if task_type == "classification" else "price",
            "schema": {
                "target": {"type": "binary"} if task_type == "classification" else {"type": "numeric"},
                "features": {
                    "age": {"type": "numeric"},
                    "income": {"type": "numeric"},
                    "education_level": {"type": "numeric"}
                }
            }
        }
        with open(test_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return test_run_id

    def setup_validation_fixture(self, task_type: str) -> str:
        """Set up validation stage fixture."""
        test_run_id = f"validation_{task_type}_{uuid.uuid4().hex[:8]}"
        test_dir = self.base_dir / test_run_id
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy sample data
        shutil.copy(self.sample_data_path, test_dir / "original_data.csv")
        
        # Create metadata.json
        metadata = {
            "task_type": task_type,
            "stage": "validation",
            "timestamp": datetime.now().isoformat()
        }
        with open(test_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Read columns from sample data
        df = pd.read_csv(self.sample_data_path)
        columns = list(df.columns)
        
        # Create validation.json (expected output)
        validation = {
            "status": "success",
            "success": True,
            "columns_exist": columns,
            "no_missing_values": True,
            "data_types_match": True,
            "validations": [
                {"name": "data_quality", "status": "passed"},
                {"name": "schema_check", "status": "passed"}
            ]
        }
        with open(test_dir / "validation.json", 'w') as f:
            json.dump(validation, f, indent=2)
        
        # Create status.json
        status = {
            "stage": "validation",
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        with open(test_dir / "status.json", 'w') as f:
            json.dump(status, f, indent=2)
        
        return test_run_id

    def setup_prep_fixture(self, task_type: str, validation_test_run_id: str) -> str:
        """Set up preparation stage fixture."""
        test_run_id = f"prep_{task_type}_{uuid.uuid4().hex[:8]}"
        test_dir = self.base_dir / test_run_id
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy sample data
        shutil.copy(self.sample_data_path, test_dir / "original_data.csv")
        
        # Copy validation.json from validation stage
        validation_dir = self.base_dir / validation_test_run_id
        shutil.copy(validation_dir / "validation.json", test_dir / "validation.json")
        
        # Create metadata.json
        metadata = {
            "task_type": task_type,
            "stage": "prep",
            "timestamp": datetime.now().isoformat()
        }
        with open(test_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create prep_metadata.json (expected output)
        prep_metadata = {
            "preprocessing_steps": ["cleaning", "encoding"],
            "feature_count": 5,
            "feature_engineering": ["pca", "scaling"],
            "encoders": ["onehot", "label"],
            "task_type": task_type,
            "target_column": "target" if task_type == "classification" else "price"
        }
        with open(test_dir / "prep_metadata.json", 'w') as f:
            json.dump(prep_metadata, f, indent=2)
        
        # Create cleaned_data.csv (expected output)
        df = pd.read_csv(self.sample_data_path)
        df.to_csv(test_dir / "cleaned_data.csv", index=False)
        
        # Create status.json
        status = {
            "stage": "prep",
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        with open(test_dir / "status.json", 'w') as f:
            json.dump(status, f, indent=2)
        
        return test_run_id

    def setup_automl_fixture(self, task_type: str) -> str:
        """Set up AutoML stage fixture."""
        test_run_id = f"automl_{task_type}_{uuid.uuid4().hex[:8]}"
        test_dir = self.base_dir / test_run_id
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy sample data
        shutil.copy(self.sample_data_path, test_dir / "cleaned_data.csv")
        
        # Create prep_metadata.json
        prep_metadata = {
            "preprocessing_steps": ["cleaning", "encoding"],
            "feature_count": 5,
            "feature_engineering": ["pca", "scaling"],
            "encoders": ["onehot", "label"],
            "task_type": task_type,
            "target_column": "target" if task_type == "classification" else "price"
        }
        with open(test_dir / "prep_metadata.json", 'w') as f:
            json.dump(prep_metadata, f, indent=2)
        
        # Create pycaret_pipeline.pkl (expected output)
        with open(test_dir / "pycaret_pipeline.pkl", 'wb') as f:
            f.write(b'')
        
        # Create model_metadata.json (expected output)
        model_metadata = {
            "model_type": "random_forest",
            "accuracy": 0.95,
            "task_type": task_type,
            "hyperparameters": {"n_estimators": 100, "max_depth": 5},
            "metrics": {"accuracy": 0.95, "f1": 0.93},
            "feature_importance": {"age": 0.4, "income": 0.35, "education_level": 0.25}
        }
        with open(test_dir / "model_metadata.json", 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Create status.json
        status = {
            "stage": "automl",
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        with open(test_dir / "status.json", 'w') as f:
            json.dump(status, f, indent=2)
        
        return test_run_id

    def setup_explain_fixture(self, task_type: str) -> str:
        """
        Create fixtures for the explain stage.

        Args:
            task_type: Type of ML task ("classification" or "regression")

        Returns:
            Test run ID for the explain stage
        """
        test_run_id = f"explain_{task_type}_{os.urandom(4).hex()}"
        test_dir = self.base_dir / test_run_id
        test_dir.mkdir(parents=True, exist_ok=True)

        # Copy sample data file
        sample_file = Path(__file__).parent.parent / "data" / f"sample_{task_type}.csv"
        shutil.copy(sample_file, test_dir / "cleaned_data.csv")

        # Create minimal metadata with schema
        metadata = {
            "task_type": task_type,
            "target_column": "target" if task_type == "classification" else "price",
            "schema": {
                "target": {"type": "binary"} if task_type == "classification" else {"type": "numeric"},
                "features": {
                    "age": {"type": "numeric"},
                    "income": {"type": "numeric"},
                    "education_level": {"type": "numeric"}
                }
            }
        }
        with open(test_dir / "prep_metadata.json", "w") as f:
            json.dump(metadata, f)

        return test_run_id

    def cleanup_test_runs(self) -> None:
        """Clean up old test runs to save space."""
        for item in self.base_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)


def create_all_stage_fixtures(task_type: str = "classification") -> Dict[str, str]:
    """
    Create fixtures for all pipeline stages.
    
    Args:
        task_type: Type of task (classification or regression)
        
    Returns:
        Dictionary mapping stage names to test run IDs
    """
    generator = TestFixtureGenerator()
    validation_id = generator.setup_validation_fixture(task_type)
    prep_id = generator.setup_prep_fixture(task_type, validation_id)
    fixtures = {
        "ingestion": generator.setup_ingestion_fixture(task_type),
        "schema": generator.setup_schema_fixture(task_type),
        "validation": validation_id,
        "prep": prep_id,
        "automl": generator.setup_automl_fixture(task_type),
        "explain": generator.setup_explain_fixture(task_type)
    }
    print(f"✅ Generated fixtures for stages: {list(fixtures.keys())}")
    return fixtures 