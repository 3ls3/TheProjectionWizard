#!/usr/bin/env python3
"""
Test script for step_1_ingest/ingest_logic.py
Verifies that the ingestion logic works correctly.
"""

import sys
import tempfile
import pandas as pd
from pathlib import Path
from io import BytesIO

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.step_1_ingest.ingest_logic import run_ingestion
from common import storage, constants
from common.storage import read_json


class MockUploadedFile:
    """Mock Streamlit UploadedFile for testing."""
    def __init__(self, content: bytes, filename: str):
        self._content = content
        self.name = filename
    
    def getvalue(self) -> bytes:
        return self._content


def create_test_csv() -> bytes:
    """Create a simple test CSV file."""
    test_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Marketing']
    }
    
    df = pd.DataFrame(test_data)
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def test_successful_ingestion():
    """Test successful CSV ingestion."""
    print("Testing successful CSV ingestion...")
    
    # Create test CSV content
    csv_content = create_test_csv()
    mock_file = MockUploadedFile(csv_content, "test_data.csv")
    
    # Run ingestion
    run_id = run_ingestion(mock_file, "data/runs")
    
    # Verify run_id was generated
    assert run_id is not None and len(run_id) > 0, "Run ID should be generated"
    print(f"Generated run_id: {run_id}")
    
    # Verify run directory was created
    run_dir = storage.get_run_dir(run_id)
    assert run_dir.exists(), "Run directory should exist"
    
    # Verify original data file was saved
    original_data_path = run_dir / constants.ORIGINAL_DATA_FILE
    assert original_data_path.exists(), "Original data file should exist"
    
    # Verify we can read the saved CSV
    df = pd.read_csv(original_data_path)
    assert df.shape == (5, 5), "CSV should have correct dimensions"
    assert 'name' in df.columns, "CSV should have expected columns"
    
    # Verify metadata.json was created
    metadata_path = run_dir / constants.METADATA_FILE
    assert metadata_path.exists(), "Metadata file should exist"
    
    metadata = read_json(run_id, constants.METADATA_FILE)
    assert metadata is not None, "Metadata should be readable"
    assert metadata['run_id'] == run_id, "Metadata should have correct run_id"
    assert metadata['original_filename'] == "test_data.csv", "Metadata should have correct filename"
    assert metadata['initial_rows'] == 5, "Metadata should have correct row count"
    assert metadata['initial_cols'] == 5, "Metadata should have correct column count"
    assert metadata['initial_dtypes'] is not None, "Metadata should have dtypes"
    
    # Verify status.json was created
    status_path = run_dir / constants.STATUS_FILE
    assert status_path.exists(), "Status file should exist"
    
    status = read_json(run_id, constants.STATUS_FILE)
    assert status is not None, "Status should be readable"
    assert status['stage'] == constants.INGEST_STAGE, "Status should have correct stage"
    assert status['status'] == 'completed', "Status should be completed"
    assert status['message'] == 'Ingestion successful.', "Status should have success message"
    
    # Verify log file was created
    log_path = run_dir / constants.PIPELINE_LOG_FILE
    assert log_path.exists(), "Log file should exist"
    
    print("✓ Successful ingestion test passed")
    return run_id


def test_invalid_csv_handling():
    """Test handling of invalid CSV content."""
    print("Testing invalid CSV handling...")
    
    # Create invalid CSV content that will actually cause pandas to fail
    # Using binary content that causes UnicodeDecodeError
    invalid_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00'  # PNG header (binary)
    mock_file = MockUploadedFile(invalid_content, "invalid.csv")
    
    # Run ingestion
    run_id = run_ingestion(mock_file, "data/runs")
    
    # Verify run_id was still generated
    assert run_id is not None and len(run_id) > 0, "Run ID should be generated even for invalid CSV"
    
    # Verify run directory was created
    run_dir = storage.get_run_dir(run_id)
    assert run_dir.exists(), "Run directory should exist"
    
    # Verify original data file was saved (even if invalid)
    original_data_path = run_dir / constants.ORIGINAL_DATA_FILE
    assert original_data_path.exists(), "Original data file should exist"
    
    # Verify status.json reflects the failure
    status = read_json(run_id, constants.STATUS_FILE)
    assert status is not None, "Status should be readable"
    assert status['status'] == 'failed', "Status should be failed for invalid CSV"
    assert 'Failed to read or parse uploaded CSV' in status['message'], "Status should indicate CSV parse failure"
    assert status['errors'] is not None and len(status['errors']) > 0, "Status should have error details"
    
    # Verify metadata.json was still created with limited info
    metadata = read_json(run_id, constants.METADATA_FILE)
    assert metadata is not None, "Metadata should be readable"
    assert metadata['initial_rows'] is None, "Metadata should have None for rows due to parse failure"
    assert metadata['initial_cols'] is None, "Metadata should have None for cols due to parse failure"
    assert metadata['initial_dtypes'] is None, "Metadata should have None for dtypes due to parse failure"
    
    print("✓ Invalid CSV handling test passed")
    return run_id


def test_file_object_handling():
    """Test handling of standard file objects."""
    print("Testing standard file object handling...")
    
    # Create test CSV content
    csv_content = create_test_csv()
    
    # Create a standard file-like object
    file_obj = BytesIO(csv_content)
    file_obj.name = "standard_file.csv"
    
    # Run ingestion
    run_id = run_ingestion(file_obj, "data/runs")
    
    # Verify basic success
    assert run_id is not None and len(run_id) > 0, "Run ID should be generated"
    
    # Verify status shows success
    status = read_json(run_id, constants.STATUS_FILE)
    assert status['status'] == 'completed', "Status should be completed"
    
    print("✓ Standard file object handling test passed")
    return run_id


def main():
    """Run all tests."""
    print("Running tests for step_1_ingest/ingest_logic.py")
    print("=" * 60)
    
    try:
        # Run tests
        test_successful_ingestion()
        test_invalid_csv_handling() 
        test_file_object_handling()
        
        print("=" * 60)
        print("✅ All ingestion tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 