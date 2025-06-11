#!/usr/bin/env python3
"""
Test script to verify the /api/predict/single endpoint fix.
This will test that input_features and feature_contributions now show correct original values.
"""

import requests
import json
import time

def test_single_predict_fix():
    """Test the fixed /api/predict/single endpoint."""
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Test data from the bug report
    run_id = "dd495cb4"
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
    
    print("=== TESTING FIXED /api/predict/single ENDPOINT ===")
    print(f"Testing with run ID: {run_id}")
    print(f"Input data: {input_data}")
    print()
    
    # Make the API call
    url = f"http://localhost:8000/api/predict/single?run_id={run_id}"
    
    try:
        response = requests.post(url, json=input_data, timeout=30)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ REQUEST SUCCESSFUL")
            print(f"Prediction value: {result.get('prediction_value')}")
            print(f"Task type: {result.get('task_type')}")
            print(f"Target column: {result.get('target_column')}")
            print()
            
            # Check the fix: input_features should show original values
            input_features = result.get('input_features', {})
            processed_features = result.get('processed_features', {})
            feature_contributions = result.get('feature_contributions', [])
            
            print("=== CHECKING THE FIX ===")
            
            # Check input_features
            print("Input features (should show original user input):")
            for key, value in input_features.items():
                original_value = input_data.get(key, 'N/A')
                match_status = "✅" if value == original_value else "❌"
                print(f"  {key}: {value} (original: {original_value}) {match_status}")
            
            print("\nProcessed features (first 5, should show encoded values):")
            processed_items = list(processed_features.items())[:5]
            for key, value in processed_items:
                print(f"  {key}: {value}")
            
            print("\nFeature contributions (first 5):")
            for i, contrib in enumerate(feature_contributions[:5]):
                feature_name = contrib['feature_name']
                feature_value = contrib['feature_value']
                contribution = contrib['contribution_value']
                
                # Check if this is a categorical feature
                if '_' in feature_name:
                    parts = feature_name.split('_', 1)
                    if len(parts) == 2:
                        original_col, encoded_value = parts
                        if original_col in input_data:
                            expected_value = 1 if str(input_data[original_col]) == encoded_value else 0
                            match_status = "✅" if feature_value == expected_value else "❌"
                        else:
                            match_status = "?"
                    else:
                        match_status = "?"
                else:
                    # Numeric feature
                    original_value = input_data.get(feature_name, 'N/A')
                    match_status = "✅" if feature_value == original_value else "❌"
                
                print(f"  {feature_name}: value={feature_value}, contribution={contribution:.4f} {match_status}")
            
            # Overall assessment
            input_all_zeros = all(v == 0 or v == 0.0 for v in input_features.values())
            contrib_all_zeros = all(contrib['feature_value'] == 0 for contrib in feature_contributions)
            
            print(f"\n=== ASSESSMENT ===")
            print(f"Input features all zeros: {input_all_zeros} {'❌ STILL BROKEN' if input_all_zeros else '✅ FIXED'}")
            print(f"Feature contributions all zeros: {contrib_all_zeros} {'❌ STILL BROKEN' if contrib_all_zeros else '✅ FIXED'}")
            
            if not input_all_zeros and not contrib_all_zeros:
                print("🎉 BUG APPEARS TO BE FIXED!")
            else:
                print("🐛 Bug still present - values are still showing as zeros")
                
        else:
            print(f"❌ REQUEST FAILED")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_single_predict_fix() 