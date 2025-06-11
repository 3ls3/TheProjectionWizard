"""
Phase 1 Infrastructure Validation Tests
Implements systematic testing of foundational system components.

This module provides comprehensive testing for:
- GCS connectivity and permissions
- Artifact upload/download operations
- Run directory structure validation

Part of the 5-phase testing plan for fixing prediction pipeline bugs.
"""

import json
import time
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from io import BytesIO

# System path setup for imports
import sys
sys.path.append('.')

# GCS utilities
from api.utils.gcs_utils import (
    PROJECT_BUCKET_NAME, upload_to_gcs, download_from_gcs, 
    check_gcs_file_exists, list_gcs_files, delete_gcs_file,
    GCSError, _gcs_client
)

# Common utilities
from common import logger


class TestResult:
    """
    Enhanced TestResult class matching workflow documentation requirements.
    Supports measurements, assertions, and structured reporting.
    """
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.status = "pending"
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        
        # Data storage
        self.measurements = {}
        self.info = {}
        self.assertions = []
        self.errors = []
        self.messages = []
        
    def add_measurement(self, key: str, value: float, unit: str = ""):
        """Add a performance measurement."""
        self.measurements[key] = {
            "value": value,
            "unit": unit,
            "timestamp": time.time()
        }
        
    def add_info(self, key: str, value: Any):
        """Add contextual information."""
        self.info[key] = value
        
    def assert_equals(self, expected: Any, actual: Any, message: str = ""):
        """Assert two values are equal."""
        assertion = {
            "type": "equals",
            "expected": expected,
            "actual": actual,
            "message": message,
            "passed": expected == actual,
            "timestamp": time.time()
        }
        self.assertions.append(assertion)
        
        if not assertion["passed"]:
            error_msg = f"Assertion failed: {message}. Expected: {expected}, Got: {actual}"
            self.errors.append(error_msg)
            
    def assert_in_range(self, value: Any, min_val: Any, max_val: Any, message: str = ""):
        """Assert value is within specified range."""
        in_range = min_val <= value <= max_val
        assertion = {
            "type": "in_range",
            "value": value,
            "min": min_val,
            "max": max_val,
            "message": message,
            "passed": in_range,
            "timestamp": time.time()
        }
        self.assertions.append(assertion)
        
        if not assertion["passed"]:
            error_msg = f"Range assertion failed: {message}. Value {value} not in range [{min_val}, {max_val}]"
            self.errors.append(error_msg)
            
    def assert_true(self, condition: bool, message: str = ""):
        """Assert condition is true."""
        assertion = {
            "type": "true",
            "condition": condition,
            "message": message,
            "passed": condition,
            "timestamp": time.time()
        }
        self.assertions.append(assertion)
        
        if not assertion["passed"]:
            error_msg = f"Truth assertion failed: {message}"
            self.errors.append(error_msg)
            
    def success(self, message: str):
        """Mark test as successful with message."""
        self.status = "success"
        self.messages.append(f"SUCCESS: {message}")
        self._finalize()
        
    def error(self, message: str):
        """Mark test as failed with error message."""
        self.status = "error"
        self.errors.append(message)
        self.messages.append(f"ERROR: {message}")
        self._finalize()
        
    def _finalize(self):
        """Finalize test timing."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        if self.end_time is None:
            self._finalize()
            
        return {
            "test_name": self.test_name,
            "status": self.status,
            "duration": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "measurements": self.measurements,
            "info": self.info,
            "assertions": self.assertions,
            "errors": self.errors,
            "messages": self.messages,
            "summary": {
                "total_assertions": len(self.assertions),
                "passed_assertions": len([a for a in self.assertions if a["passed"]]),
                "failed_assertions": len([a for a in self.assertions if not a["passed"]]),
                "total_errors": len(self.errors),
                "has_measurements": len(self.measurements) > 0
            }
        }


class Phase1InfrastructureTest:
    """
    Phase 1 Infrastructure Validation Test Suite.
    
    Tests foundational system components that other tests depend on:
    1. GCS connectivity and permissions
    2. Artifact upload/download operations  
    3. Run directory structure validation
    """
    
    def __init__(self):
        self.test_logger = logger.get_stage_logger("test_infra", "PHASE1")
        self.cleanup_files = []  # Track files for cleanup
        
    def test_gcs_connectivity(self) -> TestResult:
        """
        Comprehensive GCS connectivity test with three sub-tests:
        1. Bucket access & permissions
        2. Artifact upload/download operations
        3. Run directory structure validation
        """
        result = TestResult("gcs_connectivity")
        
        try:
            self.test_logger.info("🧪 Starting Phase 1 GCS Connectivity Test")
            result.add_info("bucket_name", PROJECT_BUCKET_NAME)
            result.add_info("test_timestamp", datetime.now().isoformat())
            
            # Sub-test 1: Bucket Access & Permissions
            bucket_success = self._test_bucket_access(result)
            
            # Sub-test 2: Artifact Upload/Download Operations
            artifact_success = self._test_artifact_operations(result) if bucket_success else False
            
            # Sub-test 3: Run Directory Structure
            directory_success = self._test_run_directory_structure(result) if bucket_success else False
            
            # Overall success evaluation
            overall_success = bucket_success and artifact_success and directory_success
            
            if overall_success:
                result.success("All GCS connectivity tests passed")
                self.test_logger.info("✅ Phase 1 GCS Connectivity Test: SUCCESS")
            else:
                failed_tests = []
                if not bucket_success:
                    failed_tests.append("bucket_access")
                if not artifact_success:
                    failed_tests.append("artifact_operations")
                if not directory_success:
                    failed_tests.append("directory_structure")
                    
                result.error(f"GCS connectivity test failed. Failed sub-tests: {failed_tests}")
                self.test_logger.error(f"❌ Phase 1 GCS Connectivity Test: FAILED ({failed_tests})")
                
        except Exception as e:
            result.error(f"Test execution failed: {str(e)}")
            result.add_info("exception_traceback", traceback.format_exc())
            self.test_logger.error(f"❌ Phase 1 GCS Connectivity Test: EXCEPTION - {str(e)}")
            
        finally:
            # Cleanup temporary files
            self._cleanup_test_files()
            
        return result
    
    def _test_bucket_access(self, result: TestResult) -> bool:
        """Test basic bucket access and permissions."""
        self.test_logger.info("🔍 Sub-test 1: Bucket Access & Permissions")
        
        try:
            start_time = time.time()
            
            # Check if GCS client is initialized
            result.assert_true(_gcs_client is not None, "GCS client should be initialized")
            
            if _gcs_client is None:
                result.error("GCS client not initialized - check credentials")
                return False
            
            # Test bucket access by attempting to list files with a safe prefix
            test_prefix = f"test_gcs_connectivity_temp/"
            file_list = list_gcs_files(PROJECT_BUCKET_NAME, test_prefix)
            
            # Add timing measurement
            access_time = time.time() - start_time
            result.add_measurement("bucket_access_time", access_time, "seconds")
            
            # Validate access was successful (even if no files found)
            result.assert_true(isinstance(file_list, list), "GCS file listing should return a list")
            result.add_info("bucket_accessible", True)
            result.add_info("test_prefix_files_found", len(file_list))
            
            self.test_logger.info(f"✅ Bucket access successful (found {len(file_list)} files with test prefix)")
            return True
            
        except GCSError as e:
            result.error(f"GCS permission error: {str(e)}")
            result.add_info("bucket_accessible", False)
            self.test_logger.error(f"❌ Bucket access failed: {str(e)}")
            return False
            
        except Exception as e:
            result.error(f"Bucket access test failed: {str(e)}")
            result.add_info("bucket_accessible", False)
            self.test_logger.error(f"❌ Bucket access exception: {str(e)}")
            return False
    
    def _test_artifact_operations(self, result: TestResult) -> bool:
        """Test upload/download operations with integrity verification."""
        self.test_logger.info("🔍 Sub-test 2: Artifact Upload/Download Operations")
        
        try:
            # Create test content
            test_content = "Phase 1 GCS Test File\nTimestamp: {}\nTest ID: gcs_connectivity_test".format(
                datetime.now().isoformat()
            )
            test_filename = f"test_gcs_connectivity_temp/test_upload_{int(time.time())}.txt"
            
            # Track for cleanup
            self.cleanup_files.append(test_filename)
            
            # Sub-test 2a: Upload operation
            upload_start = time.time()
            
            test_buffer = BytesIO(test_content.encode('utf-8'))
            upload_success = upload_to_gcs(PROJECT_BUCKET_NAME, test_buffer, test_filename)
            
            upload_time = time.time() - upload_start
            result.add_measurement("upload_time", upload_time, "seconds")
            result.assert_true(upload_success, "File upload should succeed")
            
            if not upload_success:
                result.error("File upload failed")
                return False
                
            self.test_logger.info(f"✅ Upload successful ({upload_time:.3f}s)")
            
            # Sub-test 2b: File existence check
            exists_start = time.time()
            
            file_exists = check_gcs_file_exists(PROJECT_BUCKET_NAME, test_filename)
            
            exists_time = time.time() - exists_start
            result.add_measurement("exists_check_time", exists_time, "seconds")
            result.assert_true(file_exists, "Uploaded file should exist")
            
            if not file_exists:
                result.error("Uploaded file not found")
                return False
                
            self.test_logger.info(f"✅ File existence confirmed ({exists_time:.3f}s)")
            
            # Sub-test 2c: Download operation
            download_start = time.time()
            
            downloaded_content = download_from_gcs(PROJECT_BUCKET_NAME, test_filename)
            
            download_time = time.time() - download_start
            result.add_measurement("download_time", download_time, "seconds")
            result.assert_true(downloaded_content is not None, "File download should succeed")
            
            if downloaded_content is None:
                result.error("File download failed")
                return False
                
            self.test_logger.info(f"✅ Download successful ({download_time:.3f}s)")
            
            # Sub-test 2d: Content integrity verification
            downloaded_text = downloaded_content.decode('utf-8')
            result.assert_equals(test_content, downloaded_text, "Downloaded content should match uploaded content")
            
            if test_content != downloaded_text:
                result.error("Content integrity check failed")
                result.add_info("original_content_length", len(test_content))
                result.add_info("downloaded_content_length", len(downloaded_text))
                return False
                
            self.test_logger.info("✅ Content integrity verified")
            
            # Add summary info
            total_operation_time = upload_time + exists_time + download_time
            result.add_measurement("total_operation_time", total_operation_time, "seconds")
            result.add_info("test_file_size", len(test_content))
            result.add_info("artifact_operations_successful", True)
            
            return True
            
        except Exception as e:
            result.error(f"Artifact operations test failed: {str(e)}")
            result.add_info("artifact_operations_successful", False)
            self.test_logger.error(f"❌ Artifact operations failed: {str(e)}")
            return False
    
    def _test_run_directory_structure(self, result: TestResult) -> bool:
        """Test run directory structure interactions."""
        self.test_logger.info("🔍 Sub-test 3: Run Directory Structure")
        
        try:
            # Test listing runs directory structure
            runs_start = time.time()
            
            runs_prefix = "runs/"
            runs_files = list_gcs_files(PROJECT_BUCKET_NAME, runs_prefix)
            
            runs_time = time.time() - runs_start
            result.add_measurement("runs_directory_list_time", runs_time, "seconds")
            
            # Validate runs directory can be accessed
            result.assert_true(isinstance(runs_files, list), "Runs directory listing should return a list")
            result.add_info("runs_directory_accessible", True)
            result.add_info("existing_runs_count", len(runs_files))
            
            self.test_logger.info(f"✅ Runs directory accessible ({len(runs_files)} files found)")
            
            # Test creating a test run directory structure
            test_run_id = f"test_gcs_connectivity_{int(time.time())}"
            test_run_prefix = f"runs/{test_run_id}/"
            
            # Create a test metadata file in the run directory
            test_metadata = {
                "run_id": test_run_id,
                "test_type": "gcs_connectivity",
                "timestamp": datetime.now().isoformat(),
                "purpose": "Phase 1 infrastructure validation"
            }
            
            metadata_filename = f"{test_run_prefix}test_metadata.json"
            self.cleanup_files.append(metadata_filename)
            
            metadata_content = json.dumps(test_metadata, indent=2)
            metadata_buffer = BytesIO(metadata_content.encode('utf-8'))
            
            metadata_upload_start = time.time()
            metadata_success = upload_to_gcs(PROJECT_BUCKET_NAME, metadata_buffer, metadata_filename)
            metadata_upload_time = time.time() - metadata_upload_start
            
            result.add_measurement("test_run_creation_time", metadata_upload_time, "seconds")
            result.assert_true(metadata_success, "Test run directory creation should succeed")
            
            if not metadata_success:
                result.error("Test run directory creation failed")
                return False
                
            # Verify the test run structure
            test_run_files = list_gcs_files(PROJECT_BUCKET_NAME, test_run_prefix)
            result.assert_true(len(test_run_files) > 0, "Test run directory should contain files")
            result.assert_true(metadata_filename in test_run_files, "Test metadata file should be in run directory")
            
            self.test_logger.info(f"✅ Test run directory created and verified")
            
            result.add_info("test_run_id", test_run_id)
            result.add_info("test_run_files_created", len(test_run_files))
            result.add_info("run_directory_structure_working", True)
            
            return True
            
        except Exception as e:
            result.error(f"Run directory structure test failed: {str(e)}")
            result.add_info("run_directory_structure_working", False)
            self.test_logger.error(f"❌ Run directory structure test failed: {str(e)}")
            return False
    
    def _cleanup_test_files(self):
        """Clean up temporary test files from GCS."""
        self.test_logger.info("🧹 Cleaning up test files")
        
        for filename in self.cleanup_files:
            try:
                delete_success = delete_gcs_file(PROJECT_BUCKET_NAME, filename)
                if delete_success:
                    self.test_logger.info(f"✅ Cleaned up: {filename}")
                else:
                    self.test_logger.warning(f"⚠️  Could not clean up: {filename} (file may not exist)")
            except Exception as e:
                self.test_logger.warning(f"⚠️  Cleanup error for {filename}: {str(e)}")
        
        self.cleanup_files.clear()


def run_phase_1_infrastructure_test() -> TestResult:
    """
    Convenience function to run Phase 1 infrastructure test.
    
    Returns:
        TestResult: Comprehensive test results
    """
    test_runner = Phase1InfrastructureTest()
    return test_runner.test_gcs_connectivity()


def main():
    """Main function for running the test directly."""
    print("🚀 Starting Phase 1 Infrastructure Validation")
    print(f"Target bucket: {PROJECT_BUCKET_NAME}")
    print("-" * 60)
    
    # Run the test
    result = run_phase_1_infrastructure_test()
    
    # Generate report
    report_data = result.to_dict()
    
    # Save report
    reports_dir = Path("tests/reports")
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"phase_1_infrastructure_test_{timestamp}.json"
    report_path = reports_dir / report_filename
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PHASE 1 INFRASTRUCTURE TEST SUMMARY")
    print("=" * 60)
    print(f"Status: {'✅ SUCCESS' if result.status == 'success' else '❌ FAILED'}")
    print(f"Duration: {result.duration:.3f} seconds")
    print(f"Assertions: {result.to_dict()['summary']['passed_assertions']}/{result.to_dict()['summary']['total_assertions']} passed")
    print(f"Errors: {result.to_dict()['summary']['total_errors']}")
    
    if result.measurements:
        print("\n🔍 Key Metrics:")
        for name, measurement in result.measurements.items():
            print(f"  • {name}: {measurement['value']:.3f} {measurement['unit']}")
    
    if result.errors:
        print("\n❌ Errors:")
        for error in result.errors:
            print(f"  • {error}")
    
    print(f"\n📄 Full report saved: {report_path}")
    
    return result.status == "success"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 