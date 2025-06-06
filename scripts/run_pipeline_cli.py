#!/usr/bin/env python3
"""
Headless CLI Runner for The Projection Wizard Pipeline.

This script executes the entire data pipeline from data ingestion through model
explainability, producing all standard artefacts without UI interaction.

Usage:
    python scripts/run_pipeline_cli.py --csv path/to/data.csv [options]
    
    Or as a module:
    python -m scripts.run_pipeline_cli --csv path/to/data.csv [options]

Examples:
    # Basic usage with auto-detection
    python scripts/run_pipeline_cli.py --csv data/fixtures/sample.csv
    
    # Specify target column and task type
    python scripts/run_pipeline_cli.py --csv data.csv --target price --task regression
    
    # Full specification
    python scripts/run_pipeline_cli.py --csv data.csv --target category --task classification --target-ml-type multiclass_text_labels
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import pipeline stage functions
from step_1_ingest.ingest_logic import run_ingestion
from step_2_schema.target_definition_logic import suggest_target_and_task, confirm_target_definition
from step_2_schema.feature_definition_logic import suggest_initial_feature_schemas, confirm_feature_schemas
from step_3_validation.validation_runner import run_validation_stage
from step_4_prep.prep_runner import run_preparation_stage
from step_5_automl.automl_runner import run_automl_stage
from step_6_explain.explain_runner import run_explainability_stage

# Import common modules
from common import constants, utils, storage, logger, schemas


class MockUploadedFile:
    """Mock file object to simulate Streamlit's UploadedFile for CLI usage."""
    
    def __init__(self, filepath: Path):
        self.path = Path(filepath)
        self.name = self.path.name
        self._file = None
        
    def __enter__(self):
        self._file = open(self.path, 'rb')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
            
    def read(self, size: int = -1):
        return self._file.read(size)
        
    def seek(self, offset: int, whence: int = 0):
        return self._file.seek(offset, whence)
        
    def tell(self):
        return self._file.tell()
        
    def getvalue(self):
        """Streamlit compatibility method."""
        current_pos = self.tell()
        self.seek(0)
        content = self.read()
        self.seek(current_pos)
        return content


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run The Projection Wizard pipeline from command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Examples:')[1] if 'Examples:' in __doc__ else ""
    )
    
    # Required arguments
    parser.add_argument(
        '--csv', 
        required=True,
        type=str,
        help='Path to the input CSV file'
    )
    
    # Optional arguments for target configuration
    parser.add_argument(
        '--target',
        type=str,
        help='Name of the target column (if omitted, auto-detection will be used)'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['classification', 'regression'],
        help='Task type: classification or regression (if omitted, auto-detection will be used)'
    )
    
    parser.add_argument(
        '--target-ml-type',
        type=str,
        choices=constants.TARGET_ML_TYPES,
        help='ML-ready type for the target column (if omitted, auto-detection will be used)'
    )
    
    # Optional output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Path(constants.DATA_DIR_NAME) / constants.RUNS_DIR_NAME),
        help='Base directory for pipeline outputs (default: data/runs)'
    )
    
    return parser.parse_args()


def validate_input_file(csv_path: str) -> Path:
    """Validate that the input CSV file exists and is readable."""
    file_path = Path(csv_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")
        
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {csv_path}")
        
    if file_path.suffix.lower() != '.csv':
        print(f"Warning: File does not have .csv extension: {csv_path}")
        
    # Test that we can read the file
    try:
        with open(file_path, 'rb') as f:
            # Read first few bytes to ensure it's readable
            f.read(1024)
    except Exception as e:
        raise IOError(f"Cannot read input file {csv_path}: {e}")
        
    return file_path


def check_stage_status(run_id: str, expected_stage: str, cli_logger) -> bool:
    """
    Check if the current stage completed successfully.
    
    Returns:
        True if stage completed successfully, False if failed
    """
    try:
        status_dict = storage.read_status(run_id)
        if not status_dict:
            cli_logger.error(f"Could not read status.json for run {run_id}")
            return False
            
        current_status = schemas.StageStatus(**status_dict)
        
        if current_status.status == 'failed':
            cli_logger.error(f"Stage {current_status.stage} failed. Errors: {current_status.errors}")
            return False
        elif current_status.status == 'completed':
            cli_logger.info(f"Stage {current_status.stage} completed successfully")
            return True
        else:
            cli_logger.warning(f"Stage {current_status.stage} has unexpected status: {current_status.status}")
            return False
            
    except Exception as e:
        cli_logger.error(f"Error checking stage status: {e}")
        return False


def update_run_index_entry(run_id: str, status: str, cli_logger):
    """Update the run index with the current status."""
    try:
        # Read metadata to get basic info
        metadata = storage.read_metadata(run_id)
        if not metadata:
            cli_logger.warning("Could not read metadata for run index update")
            return
            
        run_entry_data = {
            'run_id': run_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'original_filename': metadata.get('original_filename', 'unknown'),
            'status': status
        }
        
        storage.append_to_run_index(run_entry_data)
        cli_logger.info(f"Updated run index with status: {status}")
        
    except Exception as e:
        cli_logger.warning(f"Failed to update run index: {e}")


def main():
    """Main CLI execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate input file
        print(f"Validating input file: {args.csv}")
        csv_path = validate_input_file(args.csv)
        print(f"✓ Input file validated: {csv_path}")
        
        print("\n" + "="*80)
        print("STARTING PROJECTION WIZARD CLI PIPELINE")
        print("="*80)
        
        # ========================================
        # Stage 1: Data Ingestion
        # ========================================
        print(f"\n📥 Stage 1: Data Ingestion")
        print(f"Processing file: {csv_path}")
        
        try:
            with MockUploadedFile(csv_path) as mock_file:
                run_id = run_ingestion(mock_file, args.output_dir)
            
            # Get logger for this run
            cli_logger = logger.get_logger(run_id=run_id, logger_name="cli_runner")
            cli_logger.info(f"CLI Pipeline started for file: {csv_path}")
            cli_logger.info(f"Generated run_id: {run_id}")
            
            print(f"✓ Generated run ID: {run_id}")
            
            # Check ingestion status
            if not check_stage_status(run_id, constants.INGEST_STAGE, cli_logger):
                cli_logger.error("Ingestion stage failed")
                update_run_index_entry(run_id, "Failed at Ingestion", cli_logger)
                return 1
                
        except Exception as e:
            print(f"✗ Ingestion failed: {e}")
            return 1
        
        # ========================================
        # Stage 2A: Target Definition
        # ========================================
        print(f"\n🎯 Stage 2A: Target Definition")
        
        try:
            # Read the uploaded data for target suggestion
            import pandas as pd
            df = storage.read_original_data(run_id)
            if df is None:
                raise Exception("Could not read original data for target analysis")
            
            # Use CLI args or auto-detection
            if args.target and args.task and args.target_ml_type:
                # Use provided arguments
                target_column = args.target
                task_type = args.task
                target_ml_type = args.target_ml_type
                print(f"Using provided target configuration:")
                print(f"  Target: {target_column}")
                print(f"  Task: {task_type}")
                print(f"  ML Type: {target_ml_type}")
                cli_logger.info(f"Using CLI-provided target: {target_column}, task: {task_type}, ml_type: {target_ml_type}")
            else:
                # Auto-detect target
                print("Auto-detecting target column and task type...")
                suggested_target, suggested_task, suggested_ml_type = suggest_target_and_task(df)
                
                # Use suggestions or defaults
                target_column = args.target or suggested_target
                task_type = args.task or suggested_task  
                target_ml_type = args.target_ml_type or suggested_ml_type
                
                if not target_column:
                    raise Exception("Could not determine target column automatically. Please specify --target")
                if not task_type:
                    raise Exception("Could not determine task type automatically. Please specify --task")
                if not target_ml_type:
                    raise Exception("Could not determine target ML type automatically. Please specify --target-ml-type")
                
                print(f"Auto-detected target configuration:")
                print(f"  Target: {target_column}")
                print(f"  Task: {task_type}")
                print(f"  ML Type: {target_ml_type}")
                cli_logger.info(f"Auto-detected target: {target_column}, task: {task_type}, ml_type: {target_ml_type}")
            
            # Confirm target definition
            success = confirm_target_definition(
                run_id=run_id,
                confirmed_target_column=target_column,
                confirmed_task_type=task_type,
                confirmed_target_ml_type=target_ml_type
            )
            
            if not success:
                raise Exception("Target definition confirmation failed")
                
            print(f"✓ Target definition confirmed")
            cli_logger.info("Target definition stage completed")
            
        except Exception as e:
            cli_logger.error(f"Target definition failed: {e}")
            print(f"✗ Target definition failed: {e}")
            update_run_index_entry(run_id, "Failed at Target Definition", cli_logger)
            return 1
        
        # ========================================
        # Stage 2B: Feature Schema Definition  
        # ========================================
        print(f"\n📋 Stage 2B: Feature Schema Definition")
        
        try:
            print("Auto-detecting feature schemas...")
            
            # Get suggested schemas for all columns
            suggested_schemas = suggest_initial_feature_schemas(df)
            cli_logger.info(f"Generated feature schemas for {len(suggested_schemas)} columns")
            
            # For CLI, we'll use the suggested schemas as-is
            # In the future, this could be enhanced to accept a JSON config file
            print(f"Using auto-detected schemas for {len(suggested_schemas)} features")
            
            # Convert suggested schemas to the format expected by confirm_feature_schemas
            # For CLI mode, we'll use all suggested schemas as user confirmed
            user_confirmed_schemas = {}
            for col, schema in suggested_schemas.items():
                user_confirmed_schemas[col] = {
                    'final_dtype': schema['initial_dtype'],
                    'final_encoding_role': schema['suggested_encoding_role']
                }
            
            # Confirm feature schemas  
            success = confirm_feature_schemas(
                run_id=run_id,
                user_confirmed_schemas=user_confirmed_schemas,
                all_initial_schemas=suggested_schemas
            )
            
            if not success:
                raise Exception("Feature schema confirmation failed")
                
            print(f"✓ Feature schemas confirmed")
            cli_logger.info("Feature schema definition stage completed")
            
        except Exception as e:
            cli_logger.error(f"Feature schema definition failed: {e}")
            print(f"✗ Feature schema definition failed: {e}")
            update_run_index_entry(run_id, "Failed at Feature Schema", cli_logger)
            return 1
        
        # ========================================
        # Stage 3: Data Validation
        # ========================================
        print(f"\n✅ Stage 3: Data Validation")
        
        try:
            cli_logger.info("Starting data validation stage")
            success = run_validation_stage(run_id)
            
            if not success:
                if not check_stage_status(run_id, constants.VALIDATION_STAGE, cli_logger):
                    raise Exception("Validation stage failed")
                    
            print(f"✓ Data validation completed")
            cli_logger.info("Data validation stage completed")
            
        except Exception as e:
            cli_logger.error(f"Data validation failed: {e}")
            print(f"✗ Data validation failed: {e}")
            update_run_index_entry(run_id, "Failed at Validation", cli_logger)
            return 1
        
        # ========================================
        # Stage 4: Data Preparation
        # ========================================
        print(f"\n🔧 Stage 4: Data Preparation")
        
        try:
            cli_logger.info("Starting data preparation stage")
            success = run_preparation_stage(run_id)
            
            if not success:
                if not check_stage_status(run_id, constants.PREP_STAGE, cli_logger):
                    raise Exception("Data preparation stage failed")
                    
            print(f"✓ Data preparation completed")
            cli_logger.info("Data preparation stage completed")
            
        except Exception as e:
            cli_logger.error(f"Data preparation failed: {e}")
            print(f"✗ Data preparation failed: {e}")
            update_run_index_entry(run_id, "Failed at Preparation", cli_logger)
            return 1
        
        # ========================================
        # Stage 5: AutoML Training
        # ========================================
        print(f"\n🤖 Stage 5: AutoML Training")
        
        try:
            cli_logger.info("Starting AutoML training stage")
            success = run_automl_stage(run_id)
            
            if not success:
                if not check_stage_status(run_id, constants.AUTOML_STAGE, cli_logger):
                    raise Exception("AutoML stage failed")
                    
            print(f"✓ AutoML training completed")
            cli_logger.info("AutoML training stage completed")
            
        except Exception as e:
            cli_logger.error(f"AutoML training failed: {e}")
            print(f"✗ AutoML training failed: {e}")
            update_run_index_entry(run_id, "Failed at AutoML", cli_logger)
            return 1
        
        # ========================================
        # Stage 6: Model Explainability
        # ========================================
        print(f"\n🔍 Stage 6: Model Explainability")
        
        try:
            cli_logger.info("Starting model explainability stage")
            success = run_explainability_stage(run_id)
            
            if not success:
                if not check_stage_status(run_id, constants.EXPLAIN_STAGE, cli_logger):
                    raise Exception("Explainability stage failed")
                    
            print(f"✓ Model explainability completed")
            cli_logger.info("Model explainability stage completed")
            
        except Exception as e:
            cli_logger.error(f"Model explainability failed: {e}")
            print(f"✗ Model explainability failed: {e}")
            update_run_index_entry(run_id, "Failed at Explainability", cli_logger)
            return 1
        
        # ========================================
        # Pipeline Completion
        # ========================================
        print(f"\n🎉 Pipeline Completed Successfully!")
        print("="*80)
        
        run_dir = storage.get_run_dir(run_id)
        print(f"Run ID: {run_id}")
        print(f"Output directory: {run_dir}")
        print(f"Log file: {run_dir / constants.PIPELINE_LOG_FILENAME}")
        
        print(f"\nGenerated artifacts:")
        artifacts = [
            constants.ORIGINAL_DATA_FILE,
            constants.CLEANED_DATA_FILE,
            constants.METADATA_FILE,
            constants.STATUS_FILE,
            constants.VALIDATION_FILE,
            constants.PROFILE_REPORT_FILE,
            f"{constants.MODEL_DIR}/{constants.MODEL_FILE}",
            f"{constants.PLOTS_DIR}/{constants.SHAP_SUMMARY_PLOT}"
        ]
        
        for artifact in artifacts:
            artifact_path = run_dir / artifact
            if artifact_path.exists():
                print(f"  ✓ {artifact}")
            else:
                print(f"  ✗ {artifact} (missing)")
        
        cli_logger.info("CLI pipeline completed successfully")
        update_run_index_entry(run_id, "Completed Successfully", cli_logger)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with unexpected error: {e}")
        if 'cli_logger' in locals():
            cli_logger.error(f"CLI pipeline failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 