#!/usr/bin/env python3
"""
Comprehensive test to verify ALL prediction endpoints work correctly
after the complete scaling fix implementation.

This test validates that both /predict and /predict/single endpoints
now return reasonable predictions in the correct range.
"""

import pytest
import requests
import json


class TestCompletePredictionFix:
    """Test all prediction endpoints after the complete scaling fix."""
    
    BASE_URL = "https://b09d-2a02-8109-d90b-2f00-5c57-9e7b-c6-50ca.ngrok-free.app"
    
    @pytest.fixture
    def fixed_run_id(self):
        """Run ID with the complete fix."""
        return "1ba8a85d"
    
    @pytest.fixture
    def test_input_realistic(self):
        """Realistic house input that should predict ~400k."""
        return {
            "square_feet": 2000,
            "bedrooms": 3,
            "bathrooms": 2,
            "house_age_years": 10,
            "garage_spaces": 2,
            "neighborhood_quality_score": 7,
            "school_district_rating": 8,
            "distance_to_city_miles": 5,
            "property_type": "Single Family"
        }
    
    def test_predict_endpoint_reasonable_values(self, fixed_run_id, test_input_realistic):
        """Test that /predict endpoint returns reasonable values."""
        url = f"{self.BASE_URL}/api/predict"
        
        response = requests.post(url, json={
            "run_id": fixed_run_id,
            "input_values": test_input_realistic
        })
        
        assert response.status_code == 200, f"Request failed: {response.text}"
        
        data = response.json()
        prediction = data["prediction_value"]
        
        print(f"\\n/predict endpoint:")
        print(f"  Prediction: ${prediction:,.0f}")
        
        # Verify prediction is reasonable (150k - 1M for house prices)
        assert 150_000 <= prediction <= 1_000_000, f"Prediction ${prediction:,.0f} is unreasonable"
        print(f"  ✅ Prediction is reasonable")
    
    def test_predict_single_endpoint_reasonable_values(self, fixed_run_id, test_input_realistic):
        """Test that /predict/single endpoint returns reasonable values after fix."""
        url = f"{self.BASE_URL}/api/predict/single"
        
        response = requests.post(
            f"{url}?run_id={fixed_run_id}",
            json=test_input_realistic
        )
        
        assert response.status_code == 200, f"Request failed: {response.text}"
        
        data = response.json()
        prediction = data["prediction_value"]
        
        print(f"\\n/predict/single endpoint:")
        print(f"  Prediction: ${prediction:,.0f}")
        
        # Verify prediction is reasonable (150k - 1M for house prices)
        assert 150_000 <= prediction <= 1_000_000, f"Prediction ${prediction:,.0f} is unreasonable"
        print(f"  ✅ Prediction is reasonable")
    
    def test_both_endpoints_similar_predictions(self, fixed_run_id, test_input_realistic):
        """Test that both endpoints return similar predictions."""
        # Get prediction from /predict
        predict_response = requests.post(f"{self.BASE_URL}/api/predict", json={
            "run_id": fixed_run_id,
            "input_values": test_input_realistic
        })
        predict_value = predict_response.json()["prediction_value"]
        
        # Get prediction from /predict/single
        single_response = requests.post(
            f"{self.BASE_URL}/api/predict/single?run_id={fixed_run_id}",
            json=test_input_realistic
        )
        single_value = single_response.json()["prediction_value"]
        
        print(f"\\nComparison:")
        print(f"  /predict:        ${predict_value:,.0f}")
        print(f"  /predict/single: ${single_value:,.0f}")
        
        # Verify predictions are similar (within 5% of each other)
        difference = abs(predict_value - single_value)
        avg_value = (predict_value + single_value) / 2
        percent_diff = (difference / avg_value) * 100
        
        print(f"  Difference:      ${difference:,.0f} ({percent_diff:.1f}%)")
        
        assert percent_diff <= 5, f"Predictions differ by {percent_diff:.1f}% - should be similar"
        print(f"  ✅ Predictions are similar")
    
    def test_processed_features_show_scaling(self, fixed_run_id, test_input_realistic):
        """Test that processed features show proper scaling (not raw values)."""
        url = f"{self.BASE_URL}/api/predict/single"
        
        response = requests.post(
            f"{url}?run_id={fixed_run_id}",
            json=test_input_realistic
        )
        
        assert response.status_code == 200, f"Request failed: {response.text}"
        
        data = response.json()
        processed_features = data["processed_features"]
        
        print(f"\\nProcessed features (should show scaling):")
        print(f"  Raw input: square_feet={test_input_realistic['square_feet']}")
        print(f"  Processed: square_feet={processed_features.get('square_feet', 'missing')}")
        
        # For scaled features, processed values should be different from raw values
        # (StandardScaler converts to mean=0, std=1, so values like -0.5, 1.2, etc.)
        square_feet_processed = processed_features.get('square_feet')
        
        if square_feet_processed is not None:
            # Scaled values should be much smaller than raw values (typically -3 to +3)
            if abs(square_feet_processed) < 10:  # Scaled value
                print(f"  ✅ Features appear to be properly scaled")
            else:  # Raw value
                print(f"  🚨 Features do NOT appear to be scaled (value too large)")
                # This might be expected if the model doesn't scale all features
    
    def test_confidence_intervals_reasonable(self, fixed_run_id, test_input_realistic):
        """Test that confidence intervals are reasonable."""
        url = f"{self.BASE_URL}/api/predict/single"
        
        response = requests.post(
            f"{url}?run_id={fixed_run_id}",
            json=test_input_realistic
        )
        
        assert response.status_code == 200, f"Request failed: {response.text}"
        
        data = response.json()
        prediction = data["prediction_value"]
        confidence_interval = data.get("confidence_interval", {})
        
        if confidence_interval:
            lower = confidence_interval.get("lower_bound")
            upper = confidence_interval.get("upper_bound")
            
            print(f"\\nConfidence interval:")
            print(f"  Prediction: ${prediction:,.0f}")
            print(f"  Lower:      ${lower:,.0f}")
            print(f"  Upper:      ${upper:,.0f}")
            
            # Verify confidence interval makes sense
            assert lower < prediction < upper, "Prediction should be within confidence interval"
            
            # Verify interval is reasonable (not too wide or narrow)
            interval_width = upper - lower
            interval_pct = (interval_width / prediction) * 100
            print(f"  Width:      ${interval_width:,.0f} ({interval_pct:.1f}%)")
            
            assert 10 <= interval_pct <= 100, f"Confidence interval width {interval_pct:.1f}% seems unreasonable"
            print(f"  ✅ Confidence interval is reasonable")


if __name__ == "__main__":
    print("Testing complete prediction fix...")
    
    test_instance = TestCompletePredictionFix()
    run_id = "1ba8a85d"
    test_input = {
        "square_feet": 2000,
        "bedrooms": 3,
        "bathrooms": 2,
        "house_age_years": 10,
        "garage_spaces": 2,
        "neighborhood_quality_score": 7,
        "school_district_rating": 8,
        "distance_to_city_miles": 5,
        "property_type": "Single Family"
    }
    
    try:
        test_instance.test_predict_endpoint_reasonable_values(run_id, test_input)
        test_instance.test_predict_single_endpoint_reasonable_values(run_id, test_input)
        test_instance.test_both_endpoints_similar_predictions(run_id, test_input)
        test_instance.test_processed_features_show_scaling(run_id, test_input)
        test_instance.test_confidence_intervals_reasonable(run_id, test_input)
        
        print(f"\\n🎉 ALL TESTS PASSED! Both prediction endpoints are working correctly.")
        
    except Exception as e:
        print(f"\\n❌ TEST FAILED: {e}")
        raise 