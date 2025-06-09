"""
Test runner for Stage 1: Data Ingestion.
Tests the ingestion logic in isolation with controlled inputs.
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from common import constants
from pipeline.step_1_ingest.ingest_logic import run_ingestion
from tests.unit.stage_tests.base_stage_test import BaseStageTest, TestResult


class MockUploadedFile:
    """Mock uploaded file object for testing ingestion."""
    
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.name = self.file_path.name
        
    def getvalue(self) -> bytes:
        """Return file content as bytes."""
        with open(self.file_path, 'rb') as f:
            return f.read()


class IngestionStageTest(BaseStageTest):
    """Test runner for the data ingestion stage."""
    
    def __init__(self, test_run_id: str):
        super().__init__(test_run_id, "ingestion")
        
        # Find the original CSV file in the test directory
        self.original_csv_path = self.test_run_dir / constants.ORIGINAL_DATA_FILENAME
    
    def run_test(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run the ingestion stage test.
        
        Returns:
            Tuple of (success, test_results)
        """
        start_time = time.time()
        self.log_test_start()
        
        validation_results = {}
        success = True
        
        try:
            # 1. Validate input files exist
            required_files = [constants.ORIGINAL_DATA_FILENAME]
            input_valid, missing_files = self.validate_input_files(required_files)
            validation_results["input_validation"] = {
                "success": input_valid,
                "missing_files": missing_files
            }
            
            if not input_valid:
                success = False
                self.test_logger.error("Input validation failed")
                return success, validation_results
            
            # 2. Clear existing metadata and status files (simulate fresh ingestion)
            metadata_path = self.test_run_dir / constants.METADATA_FILENAME
            status_path = self.test_run_dir / constants.STATUS_FILENAME
            
            if metadata_path.exists():
                metadata_path.unlink()
                self.test_logger.info("Removed existing metadata.json for fresh test")
            
            if status_path.exists():
                status_path.unlink()
                self.test_logger.info("Removed existing status.json for fresh test")
            
            # 3. Run the ingestion logic
            mock_uploaded_file = MockUploadedFile(self.original_csv_path)
            base_runs_path = str(self.test_run_dir.parent)
            
            # Note: The run_ingestion function will create a new run_id,
            # but we want to test with our controlled test_run_id
            # So we'll test the ingestion logic components directly
            
            # Import and test the ingestion logic directly
            from pipeline.step_1_ingest.ingest_logic import run_ingestion
            
            # For testing, we'll modify the approach to work with our test directory
            stage_success, actual_run_id = self.run_ingestion_with_controlled_id()
            
            validation_results["stage_execution"] = {
                "success": stage_success,
                "run_id": actual_run_id if stage_success else None
            }
            
            if not stage_success:
                success = False
                self.test_logger.error("Ingestion stage execution failed")
                return success, validation_results
            
            # 4. Validate output files were created
            expected_files = [constants.METADATA_FILENAME, constants.STATUS_FILENAME]
            output_valid, missing_output = self.validate_output_files(expected_files)
            validation_results["output_validation"] = {
                "success": output_valid,
                "missing_files": missing_output
            }
            
            if not output_valid:
                success = False
            
            # 5. Validate status file content
            status_valid, status_data = self.check_status_file(constants.INGEST_STAGE, "completed")
            validation_results["status_validation"] = {
                "success": status_valid,
                "status_data": status_data
            }
            
            if not status_valid:
                success = False
            
            # 6. Validate metadata content
            expected_metadata_keys = ["run_id", "timestamp", "original_filename", "initial_rows", "initial_cols", "initial_dtypes"]
            metadata_valid, metadata_data = self.check_metadata_updates(expected_metadata_keys)
            validation_results["metadata_validation"] = {
                "success": metadata_valid,
                "found_keys": list(metadata_data.keys()) if metadata_data else []
            }
            
            if not metadata_valid:
                success = False
            
            # 7. Additional ingestion-specific validations
            ingestion_validations = self.validate_ingestion_specifics(metadata_data)
            validation_results["ingestion_specifics"] = ingestion_validations
            
            if not ingestion_validations["success"]:
                success = False
            
        except Exception as e:
            success = False
            self.test_logger.error(f"Test execution failed: {e}")
            validation_results["execution_error"] = str(e)
        
        # Log test completion
        duration = time.time() - start_time
        self.log_test_end(success, duration)
        
        # Create comprehensive test report
        test_report = self.create_test_report(success, duration, validation_results)
        
        return success, test_report
    
    def run_ingestion_with_controlled_id(self) -> Tuple[bool, str]:
        """
        Run ingestion logic in a way that works with our controlled test environment.
        
        Returns:
            Tuple of (success, run_id)
        """
        try:
            # Create a mock uploaded file
            mock_file = MockUploadedFile(self.original_csv_path)
            
            # We'll use the actual ingestion function but point it to our test area
            from pipeline.step_1_ingest.ingest_logic import run_ingestion
            
            # Since run_ingestion creates its own run_id, we'll use it as-is
            # and then copy the results to our test directory for validation
            actual_run_id = run_ingestion(mock_file, str(self.test_run_dir.parent))
            
            # Copy the generated files to our test directory for validation
            actual_run_dir = self.test_run_dir.parent / actual_run_id
            if actual_run_dir.exists():
                import shutil
                for file_name in [constants.METADATA_FILENAME, constants.STATUS_FILENAME]:
                    src_file = actual_run_dir / file_name
                    dst_file = self.test_run_dir / file_name
                    if src_file.exists():
                        shutil.copy2(src_file, dst_file)
                        self.test_logger.info(f"Copied {file_name} from actual run to test directory")
                
                # Clean up the actual run directory since it's just for testing
                shutil.rmtree(actual_run_dir)
                self.test_logger.info(f"Cleaned up temporary run directory: {actual_run_id}")
            
            return True, actual_run_id
            
        except Exception as e:
            self.test_logger.error(f"Controlled ingestion execution failed: {e}")
            return False, ""
    
    def validate_ingestion_specifics(self, metadata_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ingestion-specific requirements.
        
        Args:
            metadata_data: Metadata dictionary from the test
            
        Returns:
            Validation results dictionary
        """
        results = {"success": True, "checks": {}}
        
        try:
            # Check that initial_rows is a positive integer
            initial_rows = metadata_data.get("initial_rows")
            if not isinstance(initial_rows, int) or initial_rows <= 0:
                results["success"] = False
                results["checks"]["initial_rows"] = f"Invalid initial_rows: {initial_rows}"
                self.test_logger.error(f"Invalid initial_rows: {initial_rows}")
            else:
                results["checks"]["initial_rows"] = "✅ Valid"
                self.test_logger.info(f"✅ Valid initial_rows: {initial_rows}")
            
            # Check that initial_cols is a positive integer
            initial_cols = metadata_data.get("initial_cols")
            if not isinstance(initial_cols, int) or initial_cols <= 0:
                results["success"] = False
                results["checks"]["initial_cols"] = f"Invalid initial_cols: {initial_cols}"
                self.test_logger.error(f"Invalid initial_cols: {initial_cols}")
            else:
                results["checks"]["initial_cols"] = "✅ Valid"
                self.test_logger.info(f"✅ Valid initial_cols: {initial_cols}")
            
            # Check that initial_dtypes is a dictionary
            initial_dtypes = metadata_data.get("initial_dtypes")
            if not isinstance(initial_dtypes, dict):
                results["success"] = False
                results["checks"]["initial_dtypes"] = f"Invalid initial_dtypes type: {type(initial_dtypes)}"
                self.test_logger.error(f"Invalid initial_dtypes type: {type(initial_dtypes)}")
            else:
                results["checks"]["initial_dtypes"] = "✅ Valid"
                self.test_logger.info(f"✅ Valid initial_dtypes with {len(initial_dtypes)} columns")
            
            # Check that original_filename is set
            original_filename = metadata_data.get("original_filename")
            if not original_filename or not isinstance(original_filename, str):
                results["success"] = False
                results["checks"]["original_filename"] = f"Invalid original_filename: {original_filename}"
                self.test_logger.error(f"Invalid original_filename: {original_filename}")
            else:
                results["checks"]["original_filename"] = "✅ Valid"
                self.test_logger.info(f"✅ Valid original_filename: {original_filename}")
            
        except Exception as e:
            results["success"] = False
            results["checks"]["validation_error"] = str(e)
            self.test_logger.error(f"Ingestion validation error: {e}")
        
        return results


def run_ingestion_test(test_run_id: str) -> TestResult:
    """
    Convenience function to run the ingestion stage test.
    
    Args:
        test_run_id: Test run identifier
        
    Returns:
        TestResult object with test outcomes
    """
    try:
        test_runner = IngestionStageTest(test_run_id)
        success, details = test_runner.run_test()
        
        # Extract duration from details
        duration = details.get("test_info", {}).get("duration_seconds", 0.0)
        
        # Extract errors
        errors = []
        if not success:
            validation_results = details.get("validation_results", {})
            for check_name, check_result in validation_results.items():
                if isinstance(check_result, dict) and not check_result.get("success", True):
                    errors.append(f"{check_name}: {check_result}")
        
        return TestResult("ingestion", success, duration, details, errors)
        
    except Exception as e:
        return TestResult("ingestion", False, 0.0, {"error": str(e)}, [str(e)])


if __name__ == "__main__":
    # Demo usage
    from tests.fixtures.fixture_generator import TestFixtureGenerator
    
    print("🧪 Testing Ingestion Stage...")
    
    # Create test fixture
    generator = TestFixtureGenerator()
    test_run_id = generator.setup_stage_1_ingestion("classification")
    
    # Run test
    print(f"\n🔬 Running ingestion test for: {test_run_id}")
    result = run_ingestion_test(test_run_id)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"INGESTION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration:.2f}s")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    
    print(f"\nTest completed at: {result.timestamp}")
    print(f"See test logs in: data/test_runs/{test_run_id}/test_ingestion.log")

def create_all_stage_fixtures(task_type: str = "classification") -> Dict[str, str]:
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