"""
Base test runner for pipeline stage testing.
Provides common functionality and structure for all stage tests.
"""

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from common import constants, logger, storage


class BaseStageTest:
    """
    Base class for testing individual pipeline stages.
    
    Provides common functionality for setting up tests, running stages,
    and validating results in a consistent manner.
    """
    
    def __init__(self, stage_name: str, test_run_id: str):
        """
        Initialize the stage test.
        
        Args:
            stage_name: Name of the pipeline stage being tested
            test_run_id: Test run identifier
        """
        self.stage_name = stage_name
        self.test_run_id = test_run_id
        self.test_logger = logger.get_stage_logger(test_run_id, f"test_{stage_name}")
        
        # Get test run directory
        self.test_run_dir = Path(__file__).parent.parent.parent / "data" / "test_runs" / test_run_id
        
        if not self.test_run_dir.exists():
            raise ValueError(f"Test run directory does not exist: {self.test_run_dir}")
    
    def log_test_start(self) -> None:
        """Log the start of the test."""
        self.test_logger.info(f"🧪 Starting {self.stage_name} stage test")
        self.test_logger.info(f"Test run ID: {self.test_run_id}")
        self.test_logger.info(f"Test directory: {self.test_run_dir}")
    
    def log_test_end(self, success: bool, duration: float) -> None:
        """Log the end of the test."""
        status = "✅ PASSED" if success else "❌ FAILED"
        self.test_logger.info(f"{status} - {self.stage_name} stage test completed in {duration:.2f}s")
    
    def validate_input_files(self, required_files: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that required input files exist for the stage.
        
        Args:
            required_files: List of required file names
            
        Returns:
            Tuple of (success, missing_files)
        """
        missing_files = []
        
        for file_name in required_files:
            file_path = self.test_run_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
                self.test_logger.error(f"Required input file missing: {file_name}")
        
        if missing_files:
            self.test_logger.error(f"Input validation failed. Missing files: {missing_files}")
            return False, missing_files
        else:
            self.test_logger.info(f"✅ All required input files present: {required_files}")
            return True, []
    
    def validate_output_files(self, expected_files: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that expected output files were created by the stage.
        
        Args:
            expected_files: List of expected output file names
            
        Returns:
            Tuple of (success, missing_files)
        """
        missing_files = []
        
        for file_name in expected_files:
            file_path = self.test_run_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
                self.test_logger.error(f"Expected output file not created: {file_name}")
        
        if missing_files:
            self.test_logger.error(f"Output validation failed. Missing files: {missing_files}")
            return False, missing_files
        else:
            self.test_logger.info(f"✅ All expected output files created: {expected_files}")
            return True, []
    
    def check_status_file(self, expected_stage: str, expected_status: str = "completed") -> Tuple[bool, Dict[str, Any]]:
        """
        Check the status.json file for expected stage and status.
        
        Args:
            expected_stage: Expected stage name in status
            expected_status: Expected status value
            
        Returns:
            Tuple of (success, status_data)
        """
        try:
            status_path = self.test_run_dir / constants.STATUS_FILENAME
            
            if not status_path.exists():
                self.test_logger.error("status.json file not found")
                return False, {}
            
            with open(status_path, 'r') as f:
                status_data = json.load(f)
            
            # Validate stage
            actual_stage = status_data.get('stage')
            if actual_stage != expected_stage:
                self.test_logger.error(f"Status stage mismatch. Expected: {expected_stage}, Got: {actual_stage}")
                return False, status_data
            
            # Validate status
            actual_status = status_data.get('status')
            if actual_status != expected_status:
                self.test_logger.error(f"Status value mismatch. Expected: {expected_status}, Got: {actual_status}")
                return False, status_data
            
            self.test_logger.info(f"✅ Status validation passed: {expected_stage} -> {expected_status}")
            return True, status_data
            
        except Exception as e:
            self.test_logger.error(f"Error reading status.json: {e}")
            return False, {}
    
    def check_metadata_updates(self, expected_keys: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check that metadata.json was updated with expected keys.
        
        Args:
            expected_keys: List of keys that should be present in metadata
            
        Returns:
            Tuple of (success, metadata_data)
        """
        try:
            metadata_path = self.test_run_dir / constants.METADATA_FILENAME
            
            if not metadata_path.exists():
                self.test_logger.error("metadata.json file not found")
                return False, {}
            
            with open(metadata_path, 'r') as f:
                metadata_data = json.load(f)
            
            missing_keys = []
            for key in expected_keys:
                if key not in metadata_data:
                    missing_keys.append(key)
            
            if missing_keys:
                self.test_logger.error(f"Metadata validation failed. Missing keys: {missing_keys}")
                return False, metadata_data
            else:
                self.test_logger.info(f"✅ Metadata validation passed. Found keys: {expected_keys}")
                return True, metadata_data
                
        except Exception as e:
            self.test_logger.error(f"Error reading metadata.json: {e}")
            return False, {}
    
    def run_test(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run the stage test. Should be implemented by subclasses.
        
        Returns:
            Tuple of (success, test_results)
        """
        raise NotImplementedError("Subclasses must implement run_test()")
    
    def run_stage_function(self, stage_function, *args, **kwargs) -> Tuple[bool, Any]:
        """
        Run a stage function with error handling and logging.
        
        Args:
            stage_function: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Tuple of (success, result)
        """
        try:
            self.test_logger.info(f"Executing stage function: {stage_function.__name__}")
            result = stage_function(*args, **kwargs)
            
            if isinstance(result, bool):
                success = result
                self.test_logger.info(f"Stage function returned: {success}")
            else:
                success = True
                self.test_logger.info(f"Stage function completed, returned: {type(result)}")
            
            return success, result
            
        except Exception as e:
            self.test_logger.error(f"Stage function failed: {e}")
            self.test_logger.error(f"Traceback: {traceback.format_exc()}")
            return False, str(e)
    
    def create_test_report(self, success: bool, duration: float, 
                          validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive test report.
        
        Args:
            success: Overall test success
            duration: Test duration in seconds
            validation_results: Results from various validation checks
            
        Returns:
            Test report dictionary
        """
        report = {
            "test_info": {
                "stage_name": self.stage_name,
                "test_run_id": self.test_run_id,
                "test_directory": str(self.test_run_dir),
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
                "success": success
            },
            "validation_results": validation_results,
            "files": {
                "test_directory_contents": [
                    str(p.name) for p in self.test_run_dir.iterdir() if p.is_file()
                ]
            }
        }
        
        return report


class TestResult:
    """Container for test results with structured information."""
    
    def __init__(self, stage_name: str, success: bool, duration: float, 
                 details: Dict[str, Any], errors: List[str] = None):
        self.stage_name = stage_name
        self.success = success
        self.duration = duration
        self.details = details
        self.errors = errors or []
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "success": self.success,
            "duration": self.duration,
            "details": self.details,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat()
        } 