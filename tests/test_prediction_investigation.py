#!/usr/bin/env python3
"""
Tests to investigate the remaining prediction issue.
Despite fixing feature encoding, predictions are still 300x too high.
"""

import pytest
import requests
import json
import pandas as pd


class TestPredictionInvestigation:
    """Investigate why predictions are still incorrect despite fixed encoding."""
    
    BASE_URL = "https://b09d-2a02-8109-d90b-2f00-5c57-9e7b-c6-50ca.ngrok-free.app"
    
    @pytest.fixture
    def broken_run_id(self):
        """The old broken run with incorrect feature encoding."""
        return "cd984c23"
    
    @pytest.fixture
    def fixed_run_id(self):
        """The new run with fixed feature encoding."""
        return "1ba8a85d"
    
    @pytest.fixture
    def test_input(self):
        """Standard test input for predictions."""
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
    
    @pytest.fixture
    def expected_price_range(self):
        """Expected price range based on dataset (175k - 787k)."""
        return (150_000, 800_000)
    
    def test_column_mapping_comparison(self, broken_run_id, fixed_run_id):
        """Compare column mappings between broken and fixed models."""
        # Get broken model columns
        broken_resp = requests.get(f"{self.BASE_URL}/api/download/{broken_run_id}/column_mapping.json")
        broken_columns = broken_resp.json()['encoded_columns']
        
        # Get fixed model columns
        fixed_resp = requests.get(f"{self.BASE_URL}/api/download/{fixed_run_id}/column_mapping.json")
        fixed_columns = fixed_resp.json()['encoded_columns']
        
        print(f"\nBROKEN MODEL ({broken_run_id}):")
        print(f"  Features: {len(broken_columns)}")
        print(f"  Columns: {broken_columns}")
        
        print(f"\nFIXED MODEL ({fixed_run_id}):")
        print(f"  Features: {len(fixed_columns)}")
        print(f"  Columns: {fixed_columns}")
        
        # Verify the fix worked
        assert len(fixed_columns) < len(broken_columns), "Fixed model should have fewer features"
        assert "bedrooms_1" not in fixed_columns, "bedrooms should not be one-hot encoded"
        assert "garage_spaces_0" not in fixed_columns, "garage_spaces should not be one-hot encoded"
        assert "bedrooms" in fixed_columns, "bedrooms should be numeric"
        assert "garage_spaces" in fixed_columns, "garage_spaces should be numeric"
    
    def test_prediction_comparison(self, broken_run_id, fixed_run_id, test_input, expected_price_range):
        """Compare predictions between broken and fixed models."""
        # Test broken model
        broken_pred = requests.post(
            f"{self.BASE_URL}/api/predict",
            json={"run_id": broken_run_id, "input_values": test_input}
        ).json()
        
        # Test fixed model
        fixed_pred = requests.post(
            f"{self.BASE_URL}/api/predict", 
            json={"run_id": fixed_run_id, "input_values": test_input}
        ).json()
        
        broken_value = broken_pred['prediction_value']
        fixed_value = fixed_pred['prediction_value']
        
        print(f"\nPREDICTION COMPARISON:")
        print(f"  Broken model: ${broken_value:,.0f}")
        print(f"  Fixed model:  ${fixed_value:,.0f}")
        print(f"  Expected range: ${expected_price_range[0]:,.0f} - ${expected_price_range[1]:,.0f}")
        print(f"  Fixed is better: {abs(fixed_value - 400_000) < abs(broken_value - 400_000)}")
        
        # Both models are still way off - investigate further
        min_expected, max_expected = expected_price_range
        
        # Check if either prediction is reasonable
        broken_reasonable = min_expected <= broken_value <= max_expected * 3  # Allow 3x margin
        fixed_reasonable = min_expected <= fixed_value <= max_expected * 3
        
        if not fixed_reasonable:
            print("\n🚨 FIXED MODEL STILL PREDICTING INCORRECTLY")
            print("   This suggests there's another issue beyond feature encoding.")
    
    def test_model_performance_metrics(self, broken_run_id, fixed_run_id):
        """Compare model performance metrics between runs."""
        # Get broken model results
        broken_results = requests.get(f"{self.BASE_URL}/api/results?run_id={broken_run_id}").json()
        
        # Get fixed model results  
        fixed_results = requests.get(f"{self.BASE_URL}/api/results?run_id={fixed_run_id}").json()
        
        print(f"\nMODEL PERFORMANCE COMPARISON:")
        
        # Extract metrics
        broken_metrics = broken_results.get('automl_summary', {}).get('performance_metrics', {})
        fixed_metrics = fixed_results.get('automl_summary', {}).get('performance_metrics', {})
        
        print(f"  Broken model metrics: {broken_metrics}")
        print(f"  Fixed model metrics: {fixed_metrics}")
        
        # Compare RMSE if available
        if 'RMSE' in broken_metrics and 'RMSE' in fixed_metrics:
            broken_rmse = broken_metrics['RMSE']
            fixed_rmse = fixed_metrics['RMSE']
            print(f"  RMSE comparison: {broken_rmse:.0f} -> {fixed_rmse:.0f}")
            
            # RMSE should be in reasonable range for house prices (thousands, not millions)
            assert fixed_rmse < 1_000_000, f"RMSE {fixed_rmse} is too high - suggests scaling issue"
    
    def test_target_value_investigation(self, fixed_run_id):
        """Investigate if there's a target value scaling issue."""
        # Download the cleaned data to check target values
        try:
            cleaned_data_resp = requests.get(f"{self.BASE_URL}/api/download/{fixed_run_id}/cleaned_data.csv")
            
            if cleaned_data_resp.status_code == 200:
                # Parse CSV content
                lines = cleaned_data_resp.text.strip().split('\n')
                header = lines[0].split(',')
                
                # Find sale_price column
                if 'sale_price' in header:
                    price_idx = header.index('sale_price')
                    
                    # Get some price values
                    sample_prices = []
                    for line in lines[1:6]:  # First 5 data rows
                        values = line.split(',')
                        if len(values) > price_idx:
                            try:
                                price = float(values[price_idx])
                                sample_prices.append(price)
                            except ValueError:
                                continue
                    
                    print(f"\nTARGET VALUE INVESTIGATION:")
                    print(f"  Sample target values: {sample_prices}")
                    print(f"  Range: {min(sample_prices):,.0f} - {max(sample_prices):,.0f}")
                    
                    # Check if target values are reasonable
                    avg_target = sum(sample_prices) / len(sample_prices)
                    print(f"  Average target: {avg_target:,.0f}")
                    
                    # If targets are in 300k-500k range but predictions are 150M,
                    # there's likely a scaling or unit issue
                    if avg_target < 1_000_000:  # Targets in reasonable range
                        print("  ✅ Target values are in reasonable range")
                        print("  🚨 But predictions are 300x higher - suggests model/scaling issue")
                    else:
                        print("  🚨 Target values are unexpectedly high")
        
        except Exception as e:
            print(f"Could not analyze target values: {e}")
    
    def test_feature_importance_analysis(self, fixed_run_id):
        """Analyze feature importance to identify suspicious patterns."""
        try:
            results = requests.get(f"{self.BASE_URL}/api/results?run_id={fixed_run_id}").json()
            
            # Look for feature importance data
            if 'automl_summary' in results:
                automl = results['automl_summary']
                print(f"\nFEATURE IMPORTANCE ANALYSIS:")
                print(f"  Model type: {automl.get('best_model_name', 'Unknown')}")
                print(f"  Training shape: {automl.get('dataset_shape_for_training', 'Unknown')}")
                
                # Check if any suspicious feature has very high importance
                feature_importance = automl.get('feature_importance', {})
                if feature_importance:
                    sorted_features = sorted(feature_importance.items(), 
                                           key=lambda x: abs(x[1]), reverse=True)
                    
                    print("  Top important features:")
                    for feature, importance in sorted_features[:5]:
                        print(f"    {feature}: {importance:.4f}")
                        
                        # Check for suspicious patterns
                        if abs(importance) > 0.8:  # Very high importance
                            print(f"      🚨 Very high importance - potential issue")
        
        except Exception as e:
            print(f"Could not analyze feature importance: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 