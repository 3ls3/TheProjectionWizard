"""
Fixture Generator for The Projection Wizard Testing Framework.

This module provides utilities to create controlled test environments for each pipeline stage,
enabling isolated testing and debugging of individual components.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Literal
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from common import constants, schemas, storage, utils


class TestFixtureGenerator:
    """
    Generator for creating test fixtures for pipeline stage testing.
    
    This class can create controlled test environments with minimal, valid
    input files for testing specific pipeline stages in isolation.
    """
    
    def __init__(self, base_fixtures_dir: Optional[Path] = None):
        """
        Initialize the fixture generator.
        
        Args:
            base_fixtures_dir: Directory containing base test data files
        """
        if base_fixtures_dir is None:
            base_fixtures_dir = Path(__file__).parent.parent.parent / "data" / "fixtures"
        
        self.base_fixtures_dir = Path(base_fixtures_dir)
        self.test_runs_dir = Path(__file__).parent.parent.parent / "data" / "test_runs"
        
        # Ensure test runs directory exists
        self.test_runs_dir.mkdir(exist_ok=True)
    
    def create_test_run_id(self, stage: str, task_type: str) -> str:
        """
        Create a test run ID with stage and task type context.
        
        Args:
            stage: Pipeline stage name
            task_type: 'classification' or 'regression'
            
        Returns:
            Test run ID string
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"test_{stage}_{task_type}_{timestamp}"
    
    def get_sample_data(self, task_type: Literal["classification", "regression"]) -> pd.DataFrame:
        """
        Load sample data for testing.
        
        Args:
            task_type: Type of ML task
            
        Returns:
            Sample DataFrame for testing
        """
        if task_type == "classification":
            file_path = self.base_fixtures_dir / "sample_classification.csv"
        elif task_type == "regression":
            file_path = self.base_fixtures_dir / "sample_regression.csv"
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return pd.read_csv(file_path)
    
    def create_base_metadata(self, run_id: str, task_type: str, target_column: str) -> Dict[str, Any]:
        """
        Create base metadata structure for testing.
        
        Args:
            run_id: Test run identifier
            task_type: 'classification' or 'regression'
            target_column: Name of target column
            
        Returns:
            Base metadata dictionary
        """
        df = self.get_sample_data(task_type)
        
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_filename": f"sample_{task_type}.csv",
            "initial_rows": len(df),
            "initial_cols": len(df.columns),
            "initial_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "task_type": task_type,
            "target_column": target_column
        }
        
        return metadata
    
    def create_target_info(self, task_type: str, target_column: str) -> Dict[str, Any]:
        """
        Create target information for schema stage testing.
        
        Args:
            task_type: 'classification' or 'regression'
            target_column: Name of target column
            
        Returns:
            Target info dictionary
        """
        df = self.get_sample_data(task_type)
        target_series = df[target_column]
        
        if task_type == "classification":
            unique_values = sorted(target_series.unique().tolist())
            target_info = {
                "column_name": target_column,
                "task_type": "classification",
                "ml_type": "binary_01" if len(unique_values) == 2 else "multiclass_int_labels",
                "unique_values": unique_values,
                "value_counts": target_series.value_counts().to_dict(),
                "is_confirmed": True
            }
        else:  # regression
            target_info = {
                "column_name": target_column,
                "task_type": "regression", 
                "ml_type": "numeric_continuous",
                "min_value": float(target_series.min()),
                "max_value": float(target_series.max()),
                "mean_value": float(target_series.mean()),
                "is_confirmed": True
            }
        
        return target_info
    
    def create_feature_schemas(self, task_type: str, target_column: str) -> Dict[str, Any]:
        """
        Create feature schema information for testing.
        
        Args:
            task_type: 'classification' or 'regression'
            target_column: Name of target column
            
        Returns:
            Feature schemas dictionary
        """
        df = self.get_sample_data(task_type)
        feature_schemas = {}
        
        for col in df.columns:
            if col == target_column:
                continue
                
            col_data = df[col]
            
            # Determine data type and role
            if col_data.dtype in ['int64', 'float64']:
                if col_data.nunique() <= 10:
                    role = "categorical-nominal"
                    data_type = "categorical"
                else:
                    role = "numeric-continuous"
                    data_type = "numeric"
            else:
                role = "categorical-nominal"
                data_type = "categorical"
            
            feature_schemas[col] = {
                "column_name": col,
                "data_type": data_type,
                "encoding_role": role,
                "unique_values": int(col_data.nunique()),
                "missing_count": int(col_data.isnull().sum()),
                "is_confirmed": True
            }
        
        return feature_schemas
    
    def setup_stage_1_ingestion(self, task_type: str = "classification") -> str:
        """
        Set up test environment for Stage 1 (Ingestion).
        Only requires the original CSV file.
        
        Args:
            task_type: Type of ML task to test
            
        Returns:
            Test run ID
        """
        run_id = self.create_test_run_id("ingestion", task_type)
        run_dir = self.test_runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Copy sample data as original_data.csv
        sample_file = self.base_fixtures_dir / f"sample_{task_type}.csv"
        original_data_path = run_dir / constants.ORIGINAL_DATA_FILENAME
        shutil.copy2(sample_file, original_data_path)
        
        print(f"✅ Created Stage 1 (Ingestion) test fixture: {run_id}")
        print(f"   Files: {original_data_path}")
        
        return run_id
    
    def setup_stage_2_schema(self, task_type: str = "classification") -> str:
        """
        Set up test environment for Stage 2 (Schema Definition).
        Requires original_data.csv and basic metadata.json.
        
        Args:
            task_type: Type of ML task to test
            
        Returns:
            Test run ID
        """
        run_id = self.create_test_run_id("schema", task_type)
        run_dir = self.test_runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Copy sample data
        sample_file = self.base_fixtures_dir / f"sample_{task_type}.csv"
        original_data_path = run_dir / constants.ORIGINAL_DATA_FILENAME
        shutil.copy2(sample_file, original_data_path)
        
        # Create basic metadata from ingestion stage
        target_col = "target" if task_type == "classification" else "price"
        metadata = self.create_base_metadata(run_id, task_type, target_col)
        
        metadata_path = run_dir / constants.METADATA_FILENAME
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create completed ingestion status
        status_data = {
            "stage": constants.INGEST_STAGE,
            "status": "completed",
            "message": "Test ingestion completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        status_path = run_dir / constants.STATUS_FILENAME
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        print(f"✅ Created Stage 2 (Schema) test fixture: {run_id}")
        print(f"   Files: {original_data_path}, {metadata_path}, {status_path}")
        
        return run_id
    
    def setup_stage_3_validation(self, task_type: str = "classification") -> str:
        """
        Set up test environment for Stage 3 (Validation).
        Requires original_data.csv and metadata.json with complete schemas.
        
        Args:
            task_type: Type of ML task to test
            
        Returns:
            Test run ID
        """
        run_id = self.create_test_run_id("validation", task_type)
        run_dir = self.test_runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Copy sample data
        sample_file = self.base_fixtures_dir / f"sample_{task_type}.csv"
        original_data_path = run_dir / constants.ORIGINAL_DATA_FILENAME
        shutil.copy2(sample_file, original_data_path)
        
        # Create metadata with schemas
        target_col = "target" if task_type == "classification" else "price"
        metadata = self.create_base_metadata(run_id, task_type, target_col)
        metadata["target_info"] = self.create_target_info(task_type, target_col)
        metadata["feature_schemas"] = self.create_feature_schemas(task_type, target_col)
        
        metadata_path = run_dir / constants.METADATA_FILENAME
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create completed schema status
        status_data = {
            "stage": constants.SCHEMA_STAGE,
            "status": "completed",
            "message": "Test schema definition completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        status_path = run_dir / constants.STATUS_FILENAME
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        print(f"✅ Created Stage 3 (Validation) test fixture: {run_id}")
        print(f"   Files: {original_data_path}, {metadata_path}, {status_path}")
        
        return run_id
    
    def setup_stage_4_prep(self, task_type: str = "classification") -> str:
        """
        Set up test environment for Stage 4 (Data Preparation).
        Same as validation stage - requires complete schemas.
        
        Args:
            task_type: Type of ML task to test
            
        Returns:
            Test run ID
        """
        run_id = self.create_test_run_id("prep", task_type)
        run_dir = self.test_runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Copy sample data
        sample_file = self.base_fixtures_dir / f"sample_{task_type}.csv"
        original_data_path = run_dir / constants.ORIGINAL_DATA_FILENAME
        shutil.copy2(sample_file, original_data_path)
        
        # Create metadata with schemas
        target_col = "target" if task_type == "classification" else "price"
        metadata = self.create_base_metadata(run_id, task_type, target_col)
        metadata["target_info"] = self.create_target_info(task_type, target_col)
        metadata["feature_schemas"] = self.create_feature_schemas(task_type, target_col)
        
        # Add validation info (as if validation passed)
        metadata["validation_info"] = {
            "passed": True,
            "report_path": constants.VALIDATION_FILENAME,
            "total_expectations_evaluated": 5,
            "successful_expectations": 5
        }
        
        metadata_path = run_dir / constants.METADATA_FILENAME
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create completed validation status
        status_data = {
            "stage": constants.VALIDATION_STAGE,
            "status": "completed",
            "message": "Test validation completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        status_path = run_dir / constants.STATUS_FILENAME
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        print(f"✅ Created Stage 4 (Prep) test fixture: {run_id}")
        print(f"   Files: {original_data_path}, {metadata_path}, {status_path}")
        
        return run_id
    
    def setup_stage_5_automl(self, task_type: str = "classification") -> str:
        """
        Set up test environment for Stage 5 (AutoML).
        Requires cleaned_data.csv and metadata with prep info.
        
        Args:
            task_type: Type of ML task to test
            
        Returns:
            Test run ID
        """
        run_id = self.create_test_run_id("automl", task_type)
        run_dir = self.test_runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Create model directory
        model_dir = run_dir / constants.MODEL_DIR
        model_dir.mkdir(exist_ok=True)
        
        # Copy and prepare cleaned data (same as original for testing)
        sample_file = self.base_fixtures_dir / f"sample_{task_type}.csv"
        original_data_path = run_dir / constants.ORIGINAL_DATA_FILENAME
        cleaned_data_path = run_dir / constants.CLEANED_DATA_FILE
        shutil.copy2(sample_file, original_data_path)
        shutil.copy2(sample_file, cleaned_data_path)
        
        # Create metadata with prep info
        target_col = "target" if task_type == "classification" else "price"
        metadata = self.create_base_metadata(run_id, task_type, target_col)
        metadata["target_info"] = self.create_target_info(task_type, target_col)
        metadata["feature_schemas"] = self.create_feature_schemas(task_type, target_col)
        metadata["validation_info"] = {
            "passed": True,
            "report_path": constants.VALIDATION_FILENAME,
            "total_expectations_evaluated": 5,
            "successful_expectations": 5
        }
        
        # Add prep info
        metadata["prep_info"] = {
            "cleaned_data_path": constants.CLEANED_DATA_FILE,
            "initial_rows": metadata["initial_rows"],
            "final_rows": metadata["initial_rows"],
            "rows_removed": 0,
            "duplicates_removed": 0,
            "missing_values_imputed": 0,
            "encoders_saved": []
        }
        
        metadata_path = run_dir / constants.METADATA_FILENAME
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create completed prep status
        status_data = {
            "stage": constants.PREP_STAGE,
            "status": "completed",
            "message": "Test preparation completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        status_path = run_dir / constants.STATUS_FILENAME
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        print(f"✅ Created Stage 5 (AutoML) test fixture: {run_id}")
        print(f"   Files: {original_data_path}, {cleaned_data_path}, {metadata_path}, {status_path}")
        
        return run_id
    
    def setup_stage_6_explain(self, task_type: str = "classification") -> str:
        """
        Set up test environment for Stage 6 (Explanation).
        Requires model artifacts and cleaned data.
        
        Args:
            task_type: Type of ML task to test
            
        Returns:
            Test run ID
        """
        run_id = self.create_test_run_id("explain", task_type)
        run_dir = self.test_runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Create required directories
        model_dir = run_dir / constants.MODEL_DIR
        plots_dir = run_dir / constants.PLOTS_DIR
        model_dir.mkdir(exist_ok=True)
        plots_dir.mkdir(exist_ok=True)
        
        # Copy data files
        sample_file = self.base_fixtures_dir / f"sample_{task_type}.csv"
        original_data_path = run_dir / constants.ORIGINAL_DATA_FILENAME
        cleaned_data_path = run_dir / constants.CLEANED_DATA_FILE
        shutil.copy2(sample_file, original_data_path)
        shutil.copy2(sample_file, cleaned_data_path)
        
        # Create dummy model file (for testing - won't be functional)
        model_path = model_dir / "pycaret_pipeline.pkl"
        with open(model_path, 'w') as f:
            f.write("# Dummy model file for testing")
        
        # Create metadata with AutoML info
        target_col = "target" if task_type == "classification" else "price"
        metadata = self.create_base_metadata(run_id, task_type, target_col)
        metadata["target_info"] = self.create_target_info(task_type, target_col)
        metadata["feature_schemas"] = self.create_feature_schemas(task_type, target_col)
        metadata["validation_info"] = {
            "passed": True,
            "report_path": constants.VALIDATION_FILENAME,
            "total_expectations_evaluated": 5,
            "successful_expectations": 5
        }
        metadata["prep_info"] = {
            "cleaned_data_path": constants.CLEANED_DATA_FILE,
            "initial_rows": metadata["initial_rows"],
            "final_rows": metadata["initial_rows"],
            "rows_removed": 0,
            "duplicates_removed": 0,
            "missing_values_imputed": 0,
            "encoders_saved": []
        }
        
        # Add AutoML info
        metadata["automl_info"] = {
            "best_model_name": "Random Forest" if task_type == "classification" else "Random Forest Regressor",
            "model_path": str(model_path),
            "training_score": 0.95,
            "cv_score": 0.90,
            "training_time_seconds": 30.5
        }
        
        metadata_path = run_dir / constants.METADATA_FILENAME
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create completed AutoML status
        status_data = {
            "stage": constants.AUTOML_STAGE,
            "status": "completed",
            "message": "Test AutoML completed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        status_path = run_dir / constants.STATUS_FILENAME
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        print(f"✅ Created Stage 6 (Explain) test fixture: {run_id}")
        print(f"   Files: {original_data_path}, {cleaned_data_path}, {metadata_path}, {status_path}, {model_path}")
        
        return run_id
    
    def cleanup_test_runs(self, keep_latest: int = 5) -> None:
        """
        Clean up old test runs, keeping only the most recent ones.
        
        Args:
            keep_latest: Number of most recent test runs to keep per stage/type
        """
        if not self.test_runs_dir.exists():
            return
        
        # Group test runs by stage and type
        test_runs = {}
        for run_dir in self.test_runs_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("test_"):
                parts = run_dir.name.split("_")
                if len(parts) >= 3:
                    stage = parts[1]
                    task_type = parts[2]
                    key = f"{stage}_{task_type}"
                    
                    if key not in test_runs:
                        test_runs[key] = []
                    test_runs[key].append(run_dir)
        
        # Keep only the most recent runs for each stage/type
        deleted_count = 0
        for key, runs in test_runs.items():
            # Sort by modification time (newest first)
            runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove older runs
            for run_dir in runs[keep_latest:]:
                shutil.rmtree(run_dir)
                deleted_count += 1
                print(f"🗑️  Cleaned up old test run: {run_dir.name}")
        
        if deleted_count > 0:
            print(f"✅ Cleaned up {deleted_count} old test runs")
        else:
            print("✅ No old test runs to clean up")


def create_all_stage_fixtures(task_type: str = "classification") -> Dict[str, str]:
    """
    Convenience function to create test fixtures for all pipeline stages.
    
    Args:
        task_type: Type of ML task to test
        
    Returns:
        Dictionary mapping stage names to run IDs
    """
    generator = TestFixtureGenerator()
    
    fixtures = {
        "ingestion": generator.setup_stage_1_ingestion(task_type),
        "schema": generator.setup_stage_2_schema(task_type),
        "validation": generator.setup_stage_3_validation(task_type),
        "prep": generator.setup_stage_4_prep(task_type),
        "automl": generator.setup_stage_5_automl(task_type),
        "explain": generator.setup_stage_6_explain(task_type)
    }
    
    print(f"\n🎯 Created complete test fixture suite for {task_type}")
    print("Stage -> Run ID mapping:")
    for stage, run_id in fixtures.items():
        print(f"  {stage:12} -> {run_id}")
    
    return fixtures


if __name__ == "__main__":
    # Demo usage
    print("🧪 Creating test fixtures for both task types...")
    
    # Create fixtures for classification
    print("\n" + "="*60)
    print("CLASSIFICATION FIXTURES")
    print("="*60)
    classification_fixtures = create_all_stage_fixtures("classification")
    
    # Create fixtures for regression  
    print("\n" + "="*60)
    print("REGRESSION FIXTURES")
    print("="*60)
    regression_fixtures = create_all_stage_fixtures("regression")
    
    print(f"\n🎉 Test fixture creation completed!")
    print(f"Total fixtures created: {len(classification_fixtures) + len(regression_fixtures)}") 