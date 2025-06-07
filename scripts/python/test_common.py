#!/usr/bin/env python3
"""
Test script for common utilities.
Verifies that the implemented functions work correctly.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from common.utils import generate_run_id
from common.storage import (
    get_run_dir, write_json_atomic, read_json, 
    append_to_run_index, write_metadata, read_metadata
)
from common.logger import get_logger
from common.schemas import BaseMetadata, StageStatus, RunIndexEntry
from common.constants import (
    INGEST_STAGE, SCHEMA_STAGE, get_stage_directory, 
    get_stage_display_name, is_valid_stage
)


def test_generate_run_id():
    """Test run ID generation."""
    print("Testing generate_run_id()...")
    
    run_id = generate_run_id()
    print(f"Generated run_id: {run_id}")
    
    # Check format: YYYY-MM-DDTHH-MM-SSZ_shortUUID
    parts = run_id.split('_')
    assert len(parts) == 2, "Run ID should have timestamp and UUID parts"
    assert len(parts[1]) == 8, "UUID part should be 8 characters"
    assert 'T' in parts[0] and 'Z' in parts[0], "Timestamp should be ISO format"
    
    print("✓ generate_run_id() test passed")


def test_storage_functions():
    """Test storage functions."""
    print("Testing storage functions...")
    
    # Generate test run ID
    run_id = generate_run_id()
    
    # Test get_run_dir
    run_dir = get_run_dir(run_id)
    assert run_dir.exists(), "Run directory should be created"
    assert (run_dir / "model").exists(), "Model subdirectory should exist"
    assert (run_dir / "plots").exists(), "Plots subdirectory should exist"
    
    # Test write_json_atomic and read_json
    test_data = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
    write_json_atomic(run_id, "test.json", test_data)
    
    read_data = read_json(run_id, "test.json")
    assert read_data == test_data, "Read data should match written data"
    
    # Test metadata convenience functions
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "original_filename": "test.csv"
    }
    write_metadata(run_id, metadata)
    read_metadata_result = read_metadata(run_id)
    assert read_metadata_result == metadata, "Metadata should match"
    
    print("✓ Storage functions test passed")


def test_run_index():
    """Test run index functionality."""
    print("Testing run index...")
    
    # Test data
    run_entry = {
        "run_id": generate_run_id(),
        "timestamp": datetime.utcnow().isoformat(),
        "original_filename": "test.csv",
        "status": "completed"
    }
    
    # Test append_to_run_index
    append_to_run_index(run_entry)
    
    # Verify CSV file was created and has content
    index_path = Path("data/runs/index.csv")
    assert index_path.exists(), "Index CSV should be created"
    
    with open(index_path, 'r') as f:
        content = f.read()
        assert "run_id" in content, "Header should be present"
        assert run_entry["run_id"] in content, "Entry should be in file"
    
    print("✓ Run index test passed")


def test_logger():
    """Test logger functionality."""
    print("Testing logger...")
    
    run_id = generate_run_id()
    logger = get_logger(run_id, "test_logger", "INFO")
    
    logger.info("Test log message")
    logger.warning("Test warning message")
    
    # Verify log file was created
    run_dir = get_run_dir(run_id)
    log_file = run_dir / "pipeline.log"
    assert log_file.exists(), "Log file should be created"
    
    # Check log content
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Test log message" in content, "Log message should be in file"
        assert run_id in content, "Run ID should be in log format"
    
    print("✓ Logger test passed")


def test_pydantic_models():
    """Test Pydantic models."""
    print("Testing Pydantic models...")
    
    # Test BaseMetadata
    metadata = BaseMetadata(
        run_id=generate_run_id(),
        timestamp=datetime.utcnow(),
        original_filename="test.csv",
        initial_rows=100,
        initial_cols=10
    )
    assert metadata.run_id is not None
    
    # Test StageStatus
    status = StageStatus(
        stage=INGEST_STAGE,
        status="completed",
        message="Successfully ingested data"
    )
    assert status.stage == INGEST_STAGE
    assert status.status == "completed"
    
    # Test stage validation
    try:
        invalid_status = StageStatus(
            stage="invalid_stage",
            status="completed"
        )
        assert False, "Should have raised validation error for invalid stage"
    except ValueError as e:
        assert "stage must be one of" in str(e)
    
    # Test RunIndexEntry
    entry = RunIndexEntry(
        run_id=generate_run_id(),
        timestamp=datetime.utcnow(),
        original_filename="test.csv",
        status="completed"
    )
    assert entry.run_id is not None
    
    print("✓ Pydantic models test passed")


def test_stage_constants():
    """Test stage constants and helper functions."""
    print("Testing stage constants...")
    
    # Test helper functions
    assert get_stage_directory(INGEST_STAGE) == INGEST_STAGE
    assert get_stage_directory("results") == "ui"
    
    assert get_stage_display_name(INGEST_STAGE) == "Data Ingestion"
    assert get_stage_display_name(SCHEMA_STAGE) == "Schema Validation"
    
    assert is_valid_stage(INGEST_STAGE) == True
    assert is_valid_stage("invalid_stage") == False
    
    # Test invalid stage directory lookup
    try:
        get_stage_directory("invalid_stage")
        assert False, "Should have raised ValueError for invalid stage"
    except ValueError as e:
        assert "Unknown stage" in str(e)
    
    print("✓ Stage constants test passed")


def main():
    """Run all tests."""
    print("Running tests for common utilities...")
    print("=" * 50)
    
    try:
        test_generate_run_id()
        test_storage_functions()
        test_run_index()
        test_logger()
        test_pydantic_models()
        test_stage_constants()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 