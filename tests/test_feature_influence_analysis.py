#!/usr/bin/env python3
"""
Feature Influence Analysis Tests

This test suite methodically analyzes whether features actually influence 
predictions in meaningful ways. Tests for:
1. Feature sensitivity - do feature changes affect predictions reasonably?
2. Feature importance analysis - are the right features important?
3. Model behavior analysis - is the model just predicting the mean?
4. SHAP value validation - do explanations make sense?
"""

import pytest
import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class TestFeatureInfluenceAnalysis:
    """Comprehensive analysis of feature influence on predictions."""
    
    BASE_URL = "https://b09d-2a02-8109-d90b-2f00-5c57-9e7b-c6-50ca.ngrok-free.app"
    
    @pytest.fixture
    def run_id(self):
        return "1ba8a85d"
    
    @pytest.fixture
    def baseline_input(self):
        """Baseline realistic house input."""
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
    
    def get_prediction(self, run_id: str, input_data: Dict) -> float:
        """Get prediction value from API."""
        response = requests.post(f"{self.BASE_URL}/api/predict", json={
            "run_id": run_id,
            "input_values": input_data
        })
        assert response.status_code == 200, f"Prediction failed: {response.text}"
        return response.json()["prediction_value"]
    
    def test_square_footage_sensitivity(self, run_id, baseline_input):
        """Test if square footage changes affect predictions logically."""
        print("\\n=== SQUARE FOOTAGE SENSITIVITY TEST ===")
        
        square_feet_values = [1000, 1500, 2000, 2500, 3000, 4000]
        predictions = []
        
        for sqft in square_feet_values:
            test_input = baseline_input.copy()
            test_input["square_feet"] = sqft
            prediction = self.get_prediction(run_id, test_input)
            predictions.append(prediction)
            print(f"  {sqft:4d} sq ft → ${prediction:8,.0f}")
        
        # Analyze relationship
        sqft_vs_price = list(zip(square_feet_values, predictions))
        
        # Check if predictions generally increase with square footage
        increasing_pairs = 0
        total_pairs = 0
        for i in range(len(sqft_vs_price) - 1):
            for j in range(i + 1, len(sqft_vs_price)):
                if sqft_vs_price[j][0] > sqft_vs_price[i][0]:  # Larger house
                    total_pairs += 1
                    if sqft_vs_price[j][1] > sqft_vs_price[i][1]:  # Higher price
                        increasing_pairs += 1
        
        monotonic_percentage = (increasing_pairs / total_pairs) * 100
        print(f"\\n  Analysis:")
        print(f"    Monotonic relationship: {monotonic_percentage:.1f}% of pairs")
        print(f"    Price range: ${min(predictions):,.0f} - ${max(predictions):,.0f}")
        print(f"    Price variability: {(max(predictions) - min(predictions)) / np.mean(predictions) * 100:.1f}%")
        
        # Assertions
        assert monotonic_percentage >= 70, f"Square footage should generally increase price (only {monotonic_percentage:.1f}% monotonic)"
        price_variability = (max(predictions) - min(predictions)) / np.mean(predictions) * 100
        assert price_variability >= 20, f"Price should vary significantly with square footage (only {price_variability:.1f}% variability)"
        
        print(f"  ✅ Square footage shows reasonable influence")
        return sqft_vs_price
    
    def test_bedroom_sensitivity(self, run_id, baseline_input):
        """Test if bedroom count affects predictions logically."""
        print("\\n=== BEDROOM COUNT SENSITIVITY TEST ===")
        
        bedroom_values = [1, 2, 3, 4, 5, 6]
        predictions = []
        
        for bedrooms in bedroom_values:
            test_input = baseline_input.copy()
            test_input["bedrooms"] = bedrooms
            prediction = self.get_prediction(run_id, test_input)
            predictions.append(prediction)
            print(f"  {bedrooms} bedrooms → ${prediction:8,.0f}")
        
        # Check for reasonable relationship
        price_range = max(predictions) - min(predictions)
        price_variability = (price_range / np.mean(predictions)) * 100
        
        print(f"\\n  Analysis:")
        print(f"    Price range: ${min(predictions):,.0f} - ${max(predictions):,.0f}")
        print(f"    Price variability: {price_variability:.1f}%")
        
        # More bedrooms should generally mean higher price (with some exceptions)
        correlation = np.corrcoef(bedroom_values, predictions)[0, 1]
        print(f"    Correlation with bedrooms: {correlation:.3f}")
        
        assert price_variability >= 5, f"Bedroom count should affect price (only {price_variability:.1f}% variability)"
        assert correlation >= 0.3, f"More bedrooms should generally increase price (correlation: {correlation:.3f})"
        
        print(f"  ✅ Bedroom count shows reasonable influence")
        return list(zip(bedroom_values, predictions))
    
    def test_property_type_sensitivity(self, run_id, baseline_input):
        """Test if property type affects predictions logically."""
        print("\\n=== PROPERTY TYPE SENSITIVITY TEST ===")
        
        property_types = ["Condo", "Townhouse", "Single Family", "Ranch"]
        predictions = []
        
        for prop_type in property_types:
            test_input = baseline_input.copy()
            test_input["property_type"] = prop_type
            prediction = self.get_prediction(run_id, test_input)
            predictions.append(prediction)
            print(f"  {prop_type:12s} → ${prediction:8,.0f}")
        
        price_range = max(predictions) - min(predictions)
        price_variability = (price_range / np.mean(predictions)) * 100
        
        print(f"\\n  Analysis:")
        print(f"    Price range: ${min(predictions):,.0f} - ${max(predictions):,.0f}")
        print(f"    Price variability: {price_variability:.1f}%")
        
        assert price_variability >= 5, f"Property type should affect price (only {price_variability:.1f}% variability)"
        
        print(f"  ✅ Property type shows reasonable influence")
        return list(zip(property_types, predictions))
    
    def test_model_not_just_predicting_mean(self, run_id, baseline_input):
        """Test if model is actually using features vs just predicting the mean."""
        print("\\n=== MODEL BEHAVIOR ANALYSIS ===")
        
        # Generate diverse test cases
        test_cases = [
            # Small, old house
            {**baseline_input, "square_feet": 800, "bedrooms": 1, "bathrooms": 1, 
             "house_age_years": 50, "garage_spaces": 0, "neighborhood_quality_score": 4},
            
            # Medium house
            baseline_input,
            
            # Large, new house
            {**baseline_input, "square_feet": 4000, "bedrooms": 5, "bathrooms": 4, 
             "house_age_years": 2, "garage_spaces": 3, "neighborhood_quality_score": 9},
            
            # Luxury house
            {**baseline_input, "square_feet": 5000, "bedrooms": 6, "bathrooms": 5, 
             "house_age_years": 1, "garage_spaces": 3, "neighborhood_quality_score": 10,
             "school_district_rating": 10},
            
            # Basic house
            {**baseline_input, "square_feet": 1200, "bedrooms": 2, "bathrooms": 1, 
             "house_age_years": 30, "garage_spaces": 1, "neighborhood_quality_score": 5,
             "school_district_rating": 5}
        ]
        
        case_names = ["Small/Old", "Medium", "Large/New", "Luxury", "Basic"]
        predictions = []
        
        for i, test_case in enumerate(test_cases):
            prediction = self.get_prediction(run_id, test_case)
            predictions.append(prediction)
            print(f"  {case_names[i]:10s} → ${prediction:8,.0f}")
        
        # Analysis
        mean_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)
        cv = std_prediction / mean_prediction  # Coefficient of variation
        
        print(f"\\n  Statistical Analysis:")
        print(f"    Mean prediction: ${mean_prediction:8,.0f}")
        print(f"    Std deviation:   ${std_prediction:8,.0f}")
        print(f"    Coefficient of variation: {cv:.3f}")
        print(f"    Min prediction:  ${min(predictions):8,.0f}")
        print(f"    Max prediction:  ${max(predictions):8,.0f}")
        print(f"    Range:           ${max(predictions) - min(predictions):8,.0f}")
        
        # Assertions
        assert cv >= 0.15, f"Model shows too little variation (CV: {cv:.3f}) - might be just predicting mean"
        
        # Check logical ordering
        luxury_idx = case_names.index("Luxury")
        basic_idx = case_names.index("Basic")
        assert predictions[luxury_idx] > predictions[basic_idx], "Luxury house should cost more than basic house"
        
        large_idx = case_names.index("Large/New")
        small_idx = case_names.index("Small/Old")
        assert predictions[large_idx] > predictions[small_idx], "Large/new house should cost more than small/old house"
        
        print(f"  ✅ Model shows meaningful variation and logical ordering")
        return predictions
    
    def test_shap_values_make_sense(self, run_id, baseline_input):
        """Test if SHAP values provide reasonable explanations."""
        print("\\n=== SHAP VALUES ANALYSIS ===")
        
        # Get SHAP explanation
        response = requests.post(
            f"{self.BASE_URL}/api/predict/single?run_id={run_id}",
            json=baseline_input
        )
        assert response.status_code == 200, f"SHAP request failed: {response.text}"
        
        data = response.json()
        prediction = data["prediction_value"]
        feature_contributions = data.get("feature_contributions", [])
        
        print(f"  Prediction: ${prediction:,.0f}")
        print(f"  \\n  Feature Contributions:")
        
        # Sort by absolute contribution
        sorted_contribs = sorted(feature_contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        
        total_shap_magnitude = sum(abs(contrib["shap_value"]) for contrib in feature_contributions)
        
        for i, contrib in enumerate(sorted_contribs[:8]):  # Top 8 features
            feature = contrib["feature_name"]
            shap_val = contrib["shap_value"]
            feature_val = contrib["feature_value"]
            direction = contrib["contribution_direction"]
            pct_of_total = (abs(shap_val) / total_shap_magnitude * 100) if total_shap_magnitude > 0 else 0
            
            print(f"    {i+1:2d}. {feature:25s}: {shap_val:8.0f} ({direction:8s}) | value={feature_val} | {pct_of_total:4.1f}%")
        
        # Analysis
        print(f"\\n  SHAP Analysis:")
        print(f"    Total SHAP magnitude: {total_shap_magnitude:8.0f}")
        print(f"    Non-zero contributions: {len([c for c in feature_contributions if abs(c['shap_value']) > 1]):2d}/{len(feature_contributions)}")
        
        # Check if important features have meaningful contributions
        important_features = ["square_feet", "bedrooms", "bathrooms", "neighborhood_quality_score"]
        important_contribs = [c for c in feature_contributions if c["feature_name"] in important_features]
        
        meaningful_contribs = [c for c in important_contribs if abs(c["shap_value"]) > 1000]
        
        print(f"    Important features with meaningful contributions: {len(meaningful_contribs)}/{len(important_features)}")
        
        # Assertions
        assert total_shap_magnitude > 1000, f"SHAP values too small (total: {total_shap_magnitude}) - model might not be using features"
        assert len(meaningful_contribs) >= 2, f"Not enough important features have meaningful SHAP values ({len(meaningful_contribs)}/4)"
        
        print(f"  ✅ SHAP values show meaningful feature usage")
        return feature_contributions
    
    def test_feature_importance_consistency(self, run_id, baseline_input):
        """Test if different feature importance methods are consistent."""
        print("\\n=== FEATURE IMPORTANCE CONSISTENCY TEST ===")
        
        # Get multiple predictions with feature variations to estimate importance
        feature_importance_empirical = {}
        
        # Test each numeric feature
        numeric_features = ["square_feet", "bedrooms", "bathrooms", "house_age_years", 
                          "garage_spaces", "neighborhood_quality_score", "school_district_rating"]
        
        baseline_prediction = self.get_prediction(run_id, baseline_input)
        
        for feature in numeric_features:
            # Test with +20% and -20% changes
            low_input = baseline_input.copy()
            high_input = baseline_input.copy()
            
            baseline_val = baseline_input[feature]
            low_input[feature] = max(1, int(baseline_val * 0.8))  # -20%
            high_input[feature] = int(baseline_val * 1.2)  # +20%
            
            low_pred = self.get_prediction(run_id, low_input)
            high_pred = self.get_prediction(run_id, high_input)
            
            # Calculate sensitivity (price change per unit feature change)
            feature_change = high_input[feature] - low_input[feature]
            price_change = abs(high_pred - low_pred)
            sensitivity = price_change / feature_change if feature_change > 0 else 0
            
            feature_importance_empirical[feature] = sensitivity
            
            print(f"  {feature:25s}: {low_input[feature]:3.0f} → ${low_pred:6,.0f} | {high_input[feature]:3.0f} → ${high_pred:6,.0f} | sensitivity: {sensitivity:6.0f}")
        
        # Sort by empirical importance
        sorted_importance = sorted(feature_importance_empirical.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\\n  Empirical Feature Importance Ranking:")
        for i, (feature, importance) in enumerate(sorted_importance):
            print(f"    {i+1:2d}. {feature:25s}: {importance:8.0f}")
        
        # Check if square_feet is among top features (it should be for house prices)
        sqft_rank = next(i for i, (f, _) in enumerate(sorted_importance) if f == "square_feet") + 1
        
        assert sqft_rank <= 3, f"Square footage should be a top-3 feature (rank: {sqft_rank})"
        
        # Check if there's meaningful variation in importance
        max_importance = max(feature_importance_empirical.values())
        min_importance = min(feature_importance_empirical.values())
        importance_ratio = max_importance / min_importance if min_importance > 0 else float('inf')
        
        print(f"\\n  Importance ratio (max/min): {importance_ratio:.1f}")
        assert importance_ratio >= 2, f"Features should have varied importance (ratio: {importance_ratio:.1f})"
        
        print(f"  ✅ Feature importance shows reasonable patterns")
        return sorted_importance


if __name__ == "__main__":
    print("🔍 COMPREHENSIVE FEATURE INFLUENCE ANALYSIS")
    print("=" * 60)
    
    test_instance = TestFeatureInfluenceAnalysis()
    run_id = "1ba8a85d"
    baseline_input = {
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
        # Run all tests
        test_instance.test_square_footage_sensitivity(run_id, baseline_input)
        test_instance.test_bedroom_sensitivity(run_id, baseline_input)
        test_instance.test_property_type_sensitivity(run_id, baseline_input)
        test_instance.test_model_not_just_predicting_mean(run_id, baseline_input)
        test_instance.test_shap_values_make_sense(run_id, baseline_input)
        test_instance.test_feature_importance_consistency(run_id, baseline_input)
        
        print("\\n" + "=" * 60)
        print("🎉 ALL FEATURE INFLUENCE TESTS PASSED!")
        print("✅ Model shows meaningful feature relationships")
        
    except Exception as e:
        print(f"\\n❌ FEATURE INFLUENCE TEST FAILED: {e}")
        print("🔧 This indicates the model may need pipeline improvements")
        raise 