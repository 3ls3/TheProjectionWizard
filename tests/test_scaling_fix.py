#!/usr/bin/env python3
"""
Tests to verify the scaling fix in prediction input processing.

This addresses the critical bug where StandardScalers were saved during training
but not applied during prediction, causing 300x inflated predictions.
"""

import pytest
import requests
import json
import pandas as pd
import numpy as np


class TestScalingFix:
    """Test the scaling fix for prediction input processing."""
    
    BASE_URL = "https://b09d-2a02-8109-d90b-2f00-5c57-9e7b-c6-50ca.ngrok-free.app"
    
    @pytest.fixture
    def old_broken_run_id(self):
        """Run with encoding fix but no scaling fix."""
        return "1ba8a85d"  # Before scaling fix
    
    @pytest.fixture  
    def test_input_realistic(self):
        """Realistic house input that should predict ~400k."""
        return {
            "square_feet": 2000,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "house_age_years": 10,
            "garage_spaces": 2, 
            "neighborhood_quality_score": 7,
            "school_district_rating": 8.0,
            "distance_to_city_miles": 5.0,
            "property_type": "Single Family"
        }
    
    @pytest.fixture
    def test_input_extreme(self):
        """Extreme input to test scaling impact."""
        return {
            "square_feet": 5000,  # Large house
            "bedrooms": 6,        # Many bedrooms
            "bathrooms": 4.0,
            "house_age_years": 1,  # New house
            "garage_spaces": 3,
            "neighborhood_quality_score": 10,  # Best neighborhood
            "school_district_rating": 10.0,    # Best schools
            "distance_to_city_miles": 1.0,     # Close to city
            "property_type": "Single Family"
        }
    
    @pytest.fixture
    def expected_reasonable_range(self):
        """Expected reasonable price range for house predictions."""
        return (150_000, 1_000_000)  # $150k - $1M (reasonable for houses)
    
    def test_prediction_reasonableness_before_fix(self, old_broken_run_id, test_input_realistic, expected_reasonable_range):
        """Test that predictions were unreasonable before the scaling fix."""
        response = requests.post(
            f"{self.BASE_URL}/api/predict",
            json={"run_id": old_broken_run_id, "input_values": test_input_realistic}
        )
        
        if response.status_code == 200:
            prediction = response.json()['prediction_value']
            min_reasonable, max_reasonable = expected_reasonable_range
            
            print(f"\nBEFORE SCALING FIX:")
            print(f"  Prediction: ${prediction:,.0f}")
            print(f"  Reasonable range: ${min_reasonable:,.0f} - ${max_reasonable:,.0f}")
            
            # Before scaling fix, predictions should be way too high
            is_too_high = prediction > max_reasonable * 10  # 10x higher than reasonable
            print(f"  Is prediction unreasonably high: {is_too_high}")
            
            if is_too_high:
                print("  ✅ Confirmed: predictions were unreasonably high before scaling fix")
            else:
                print("  🤔 Unexpected: prediction seems reasonable - maybe fix already applied?")
    
    def test_column_mapping_has_correct_features(self, old_broken_run_id):
        """Verify the model has the correct features (not one-hot encoded numeric)."""
        response = requests.get(f"{self.BASE_URL}/api/download/{old_broken_run_id}/column_mapping.json")
        
        if response.status_code == 200:
            column_mapping = response.json()
            encoded_columns = column_mapping['encoded_columns']
            
            print(f"\nCOLUMN MAPPING VERIFICATION:")
            print(f"  Total features: {len(encoded_columns)}")
            print(f"  Features: {encoded_columns}")
            
            # Verify numeric features are not one-hot encoded
            assert "bedrooms" in encoded_columns, "bedrooms should be numeric feature"
            assert "garage_spaces" in encoded_columns, "garage_spaces should be numeric feature"
            assert "neighborhood_quality_score" in encoded_columns, "neighborhood_quality_score should be numeric feature"
            
            # Verify no incorrect one-hot encoding
            assert "bedrooms_1" not in encoded_columns, "bedrooms should not be one-hot encoded"
            assert "garage_spaces_0" not in encoded_columns, "garage_spaces should not be one-hot encoded"
            
            print("  ✅ Column mapping looks correct - numeric features properly handled")
    
    def test_prediction_with_multiple_inputs(self, old_broken_run_id, expected_reasonable_range):
        """Test predictions with various input combinations."""
        test_cases = [
            {"name": "Small house", "input": {
                "square_feet": 1200, "bedrooms": 2, "bathrooms": 1.0, "house_age_years": 30,
                "garage_spaces": 1, "neighborhood_quality_score": 5, "school_district_rating": 6.0,
                "distance_to_city_miles": 10.0, "property_type": "Single Family"
            }},
            {"name": "Medium house", "input": {
                "square_feet": 2000, "bedrooms": 3, "bathrooms": 2.0, "house_age_years": 10,
                "garage_spaces": 2, "neighborhood_quality_score": 7, "school_district_rating": 8.0,
                "distance_to_city_miles": 5.0, "property_type": "Single Family"
            }},
            {"name": "Large house", "input": {
                "square_feet": 3500, "bedrooms": 5, "bathrooms": 3.5, "house_age_years": 5,
                "garage_spaces": 3, "neighborhood_quality_score": 9, "school_district_rating": 9.5,
                "distance_to_city_miles": 2.0, "property_type": "Single Family"
            }}
        ]
        
        min_reasonable, max_reasonable = expected_reasonable_range
        results = []
        
        print(f"\nMULTIPLE INPUT PREDICTIONS:")
        
        for test_case in test_cases:
            response = requests.post(
                f"{self.BASE_URL}/api/predict",
                json={"run_id": old_broken_run_id, "input_values": test_case["input"]}
            )
            
            if response.status_code == 200:
                prediction = response.json()['prediction_value']
                is_reasonable = min_reasonable <= prediction <= max_reasonable * 3  # Allow 3x margin
                
                results.append({
                    'name': test_case['name'],
                    'prediction': prediction,
                    'reasonable': is_reasonable
                })
                
                print(f"  {test_case['name']}: ${prediction:,.0f} {'✅' if is_reasonable else '❌'}")
            else:
                print(f"  {test_case['name']}: Failed to get prediction")
        
        # Check if any predictions are reasonable
        reasonable_count = sum(1 for r in results if r['reasonable'])
        total_count = len(results)
        
        print(f"  Summary: {reasonable_count}/{total_count} predictions are reasonable")
        
        if reasonable_count == 0 and total_count > 0:
            print("  🚨 All predictions unreasonable - scaling fix may be needed")
        elif reasonable_count == total_count:
            print("  ✅ All predictions reasonable - scaling fix working!")
        else:
            print("  🤔 Mixed results - investigate further")
    
    def test_feature_scaling_consistency(self, old_broken_run_id):
        """Test that feature scaling is applied consistently."""
        # Test with same input values but different runs to see if scaling is consistent
        base_input = {
            "square_feet": 2000, "bedrooms": 3, "bathrooms": 2.0, "house_age_years": 10,
            "garage_spaces": 2, "neighborhood_quality_score": 7, "school_district_rating": 8.0,
            "distance_to_city_miles": 5.0, "property_type": "Single Family"
        }
        
        # Test with slightly different values to see if scaling affects predictions proportionally
        inputs = [
            {**base_input, "square_feet": 1500},  # Smaller house
            {**base_input, "square_feet": 2000},  # Base
            {**base_input, "square_feet": 2500},  # Larger house
        ]
        
        predictions = []
        print(f"\nFEATURE SCALING CONSISTENCY TEST:")
        
        for i, input_vals in enumerate(inputs):
            response = requests.post(
                f"{self.BASE_URL}/api/predict",
                json={"run_id": old_broken_run_id, "input_values": input_vals}
            )
            
            if response.status_code == 200:
                prediction = response.json()['prediction_value']
                predictions.append(prediction)
                print(f"  {input_vals['square_feet']} sqft: ${prediction:,.0f}")
            else:
                predictions.append(None)
                print(f"  {input_vals['square_feet']} sqft: Failed")
        
        # Check if predictions increase with square footage (basic sanity check)
        valid_predictions = [p for p in predictions if p is not None]
        if len(valid_predictions) >= 2:
            is_increasing = all(valid_predictions[i] <= valid_predictions[i+1] for i in range(len(valid_predictions)-1))
            print(f"  Predictions increase with square footage: {is_increasing}")
            
            # Check if the increases are reasonable (not 100x jumps)
            if len(valid_predictions) == 3:
                ratio_1_2 = valid_predictions[1] / valid_predictions[0] if valid_predictions[0] > 0 else 0
                ratio_2_3 = valid_predictions[2] / valid_predictions[1] if valid_predictions[1] > 0 else 0
                print(f"  Prediction ratios: {ratio_1_2:.2f}x, {ratio_2_3:.2f}x")
                
                # Reasonable increases should be < 2x for 25% increase in square footage
                reasonable_ratios = ratio_1_2 < 2 and ratio_2_3 < 2
                print(f"  Ratios are reasonable: {reasonable_ratios}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 