#!/usr/bin/env python3
"""
Debug script for the prediction issue in /api/predict/single endpoint.
This script will reproduce the exact flow and trace where input values become 0.0.
"""

import json
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from api.utils.io_helpers import (
    validate_run_exists_gcs,
    load_original_data_csv_gcs,
    load_metadata_json_gcs
)
from pipeline.step_7_predict.predict_logic import (
    load_pipeline_gcs,
    generate_enhanced_prediction_with_shap
)
from pipeline.step_7_predict.column_mapper import encode_user_input_gcs
from api.utils.gcs_utils import PROJECT_BUCKET_NAME

def debug_prediction_pipeline():
    """Debug the entire prediction pipeline to find where values become 0.0."""
    
    run_id = "dd495cb4"
    
    # Original user input from the bug report
    input_data = {
        "square_feet": 3032,
        "bedrooms": 4,
        "bathrooms": 3.1500000000000004,
        "house_age_years": 8,
        "garage_spaces": 2,
        "neighborhood_quality_score": 5,
        "school_district_rating": 6.6,
        "distance_to_city_miles": 5.699999999999999,
        "property_type": "Single Family"
    }
    
    print("=== DEBUGGING PREDICTION PIPELINE ===")
    print(f"Run ID: {run_id}")
    print(f"Original input data: {input_data}")
    print()
    
    # Step 1: Validate run exists
    print("Step 1: Validating run exists...")
    if not validate_run_exists_gcs(run_id, PROJECT_BUCKET_NAME):
        print(f"ERROR: Run '{run_id}' not found in GCS")
        return
    print("✓ Run exists in GCS")
    print()
    
    # Step 2: Load required components
    print("Step 2: Loading required components...")
    try:
        model = load_pipeline_gcs(run_id, PROJECT_BUCKET_NAME)
        metadata = load_metadata_json_gcs(run_id, PROJECT_BUCKET_NAME)
        df_original = load_original_data_csv_gcs(run_id, PROJECT_BUCKET_NAME)
        
        if model is None:
            print("ERROR: Model could not be loaded")
            return
        if metadata is None:
            print("ERROR: Metadata could not be loaded")
            return
        if df_original is None:
            print("ERROR: Original data could not be loaded")
            return
            
        print("✓ Model loaded successfully")
        print("✓ Metadata loaded successfully")
        print("✓ Original data loaded successfully")
        print(f"Original data shape: {df_original.shape}")
        print(f"Original data columns: {list(df_original.columns)}")
        
        target_column = metadata.get('target_info', {}).get('name')
        task_type = metadata.get('target_info', {}).get('task_type', 'unknown')
        print(f"Target column: {target_column}")
        print(f"Task type: {task_type}")
        print()
        
    except Exception as e:
        print(f"ERROR: Failed to load components: {e}")
        return
    
    # Step 3: Encode user input
    print("Step 3: Encoding user input...")
    try:
        encoded_df, encoding_issues = encode_user_input_gcs(
            input_data, run_id, df_original, target_column
        )
        
        if encoded_df is None or encoding_issues:
            print(f"ERROR: Input encoding failed: {encoding_issues}")
            return
            
        print("✓ Input encoding successful")
        print(f"Encoded DataFrame shape: {encoded_df.shape}")
        print(f"Encoded DataFrame columns: {list(encoded_df.columns)}")
        print("Encoded values (first 5 columns):")
        print(encoded_df.iloc[0, :5].to_dict())
        print()
        
    except Exception as e:
        print(f"ERROR: Input encoding failed: {e}")
        return
    
    # Step 4: Generate enhanced prediction
    print("Step 4: Generating enhanced prediction...")
    try:
        result = generate_enhanced_prediction_with_shap(model, encoded_df, target_column, task_type)
        
        print("✓ Enhanced prediction successful")
        print(f"Prediction value: {result.get('prediction_value')}")
        print(f"Task type: {result.get('task_type')}")
        print(f"Target column: {result.get('target_column')}")
        print()
        
        # Check the problematic fields
        print("=== DEBUGGING PROBLEMATIC FIELDS ===")
        input_features = result.get('input_features', {})
        processed_features = result.get('processed_features', {})
        
        print("Input features (should be original user input):")
        for key, value in list(input_features.items())[:5]:
            print(f"  {key}: {value}")
        
        print("\nProcessed features (should be encoded values):")
        for key, value in list(processed_features.items())[:5]:
            print(f"  {key}: {value}")
        
        print("\nFeature contributions (first 5):")
        contributions = result.get('feature_contributions', [])
        for contrib in contributions[:5]:
            print(f"  {contrib['feature_name']}: value={contrib['feature_value']}, contribution={contrib['contribution_value']}")
        
        # Check if values are all zeros
        input_values = list(input_features.values())
        processed_values = list(processed_features.values())
        
        input_all_zeros = all(v == 0 or v == 0.0 for v in input_values)
        processed_all_zeros = all(v == 0 or v == 0.0 for v in processed_values)
        
        print(f"\nDEBUG RESULTS:")
        print(f"Input features all zeros: {input_all_zeros}")
        print(f"Processed features all zeros: {processed_all_zeros}")
        
        if input_all_zeros and processed_all_zeros:
            print("🐛 BUG CONFIRMED: Both input and processed features are all zeros!")
        else:
            print("✓ Values look correct - bug might be elsewhere")
            
    except Exception as e:
        print(f"ERROR: Enhanced prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    debug_prediction_pipeline() 