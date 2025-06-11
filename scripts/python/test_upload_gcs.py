#!/usr/bin/env python3
"""
Test script to verify the modified /api/upload endpoint works with GCS.
This will test the upload functionality without requiring the full frontend.
"""

import sys
import requests
import json
from pathlib import Path
from io import StringIO
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_csv():
    """Create a simple test CSV file in memory."""
    # Create a simple dataset
    data = {
        'id': [1, 2, 3, 4, 5],
        'feature_a': [10.5, 20.3, 15.7, 8.9, 12.1],
        'feature_b': ['A', 'B', 'A', 'C', 'B'],
        'target': [1, 0, 1, 0, 1]
    }
    
    df = pd.DataFrame(data)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    return csv_content

def test_upload_endpoint():
    """Test the /api/upload endpoint with GCS integration."""
    
    print("🧪 Testing /api/upload endpoint with GCS integration...")
    
    # Create test CSV data
    csv_content = create_test_csv()
    print(f"📄 Created test CSV with {len(csv_content)} characters")
    
    # Prepare the file upload
    files = {
        'file': ('test_data.csv', csv_content, 'text/csv')
    }
    
    # Make request to upload endpoint
    try:
        print("📡 Sending POST request to http://localhost:8000/api/upload...")
        response = requests.post(
            "http://localhost:8000/api/upload",
            files=files,
            timeout=30
        )
        
        print(f"📈 Response status: {response.status_code}")
        
        if response.status_code == 201:
            # Parse response
            result = response.json()
            print("✅ Upload successful!")
            print(f"   Run ID: {result.get('run_id')}")
            print(f"   Shape: {result.get('shape')}")
            print(f"   Preview rows: {len(result.get('preview', []))}")
            
            # Verify GCS upload using our utilities
            run_id = result.get('run_id')
            if run_id:
                print(f"\n🔍 Verifying GCS upload for run {run_id}...")
                verify_gcs_files(run_id)
                
            return True
            
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is the FastAPI server running on localhost:8000?")
        print("   Start the server with: uvicorn api.main:app --reload")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def verify_gcs_files(run_id: str):
    """Verify that files were uploaded to GCS correctly."""
    try:
        from api.utils.gcs_utils import (
            check_run_file_exists, 
            download_run_file, 
            list_run_files
        )
        
        # Check which files exist
        files_to_check = ["original_data.csv", "metadata.json", "status.json"]
        for filename in files_to_check:
            exists = check_run_file_exists(run_id, filename)
            print(f"   📂 {filename}: {'✅ exists' if exists else '❌ missing'}")
        
        # List all files in the run directory
        all_files = list_run_files(run_id)
        print(f"   📋 Total files in GCS: {len(all_files)}")
        for file in all_files:
            print(f"      - {file}")
        
        # Download and check metadata.json
        metadata_bytes = download_run_file(run_id, "metadata.json")
        if metadata_bytes:
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            print(f"   📊 Metadata check:")
            print(f"      - Run ID: {metadata.get('run_id')}")
            print(f"      - Original filename: {metadata.get('original_filename')}")
            print(f"      - Storage type: {metadata.get('storage', {}).get('type')}")
            print(f"      - Bucket: {metadata.get('storage', {}).get('bucket')}")
        
        # Download and check status.json
        status_bytes = download_run_file(run_id, "status.json")
        if status_bytes:
            status = json.loads(status_bytes.decode('utf-8'))
            print(f"   📈 Status check:")
            print(f"      - Stage: {status.get('stage')}")
            print(f"      - Status: {status.get('status')}")
            print(f"      - Progress: {status.get('progress_pct')}%")
        
        print("✅ GCS verification completed successfully!")
        
    except Exception as e:
        print(f"❌ GCS verification failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("GCS Upload Endpoint Test")
    print("=" * 60)
    print()
    
    success = test_upload_endpoint()
    
    print()
    if success:
        print("🎉 Upload endpoint test passed!")
        print("✅ GCS integration is working correctly!")
    else:
        print("❌ Upload endpoint test failed!")
        print("🔧 Check server logs and GCS configuration")
    
    print("=" * 60) 