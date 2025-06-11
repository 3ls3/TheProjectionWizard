#!/usr/bin/env python3
"""
Tests for the feature encoding fix that prevents numeric features
from being incorrectly classified as categorical.

This addresses the bug where features like bedrooms, garage_spaces, 
and neighborhood_quality_score were one-hot encoded instead of 
treated as numeric features.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.step_2_schema import feature_definition_logic


class TestFeatureEncodingFix:
    """Test cases for the feature encoding fix."""
    
    @pytest.fixture
    def sample_house_data(self):
        """Create sample house price data similar to the real dataset."""
        return pd.DataFrame({
            'square_feet': [1465, 1601, 2494, 2454, 3080],
            'bedrooms': [3, 3, 4, 5, 6],  # 1-8 unique values - should be numeric
            'bathrooms': [2.5, 1.8, 2.4, 4.1, 3.8],
            'house_age_years': [1, 2, 7, 7, 68],
            'garage_spaces': [2, 0, 2, 1, 0],  # 0-3 unique values - should be numeric  
            'neighborhood_quality_score': [5, 6, 7, 5, 5],  # 1-10 scale - should be numeric
            'school_district_rating': [9.9, 5.3, 8.4, 9.2, 5.6],
            'distance_to_city_miles': [1.1, 6.6, 10.5, 7.8, 6.9],
            'property_type': ['Townhouse', 'Single Family', 'Single Family', 'Single Family', 'Townhouse'],
            'sale_price': [369929, 367039, 470672, 502876, 487537]
        })
    
    @pytest.fixture
    def problematic_numeric_features(self):
        """Features that were incorrectly classified before the fix."""
        return ['bedrooms', 'garage_spaces', 'neighborhood_quality_score']
    
    @pytest.fixture
    def should_be_categorical_features(self):
        """Features that should correctly be classified as categorical."""
        return ['property_type']
    
    def test_problematic_features_now_numeric(self, sample_house_data, problematic_numeric_features):
        """Test that previously problematic features are now classified as numeric."""
        schemas = feature_definition_logic.suggest_initial_feature_schemas(sample_house_data)
        
        for feature in problematic_numeric_features:
            encoding_role = schemas[feature]['suggested_encoding_role']
            assert encoding_role in ['numeric-discrete', 'numeric-continuous'], \
                f"Feature '{feature}' should be numeric but got '{encoding_role}'"
    
    def test_true_categorical_features_remain_categorical(self, sample_house_data, should_be_categorical_features):
        """Test that truly categorical features are still classified correctly."""
        schemas = feature_definition_logic.suggest_initial_feature_schemas(sample_house_data)
        
        for feature in should_be_categorical_features:
            encoding_role = schemas[feature]['suggested_encoding_role']
            assert encoding_role in ['categorical-nominal', 'categorical-ordinal'], \
                f"Feature '{feature}' should be categorical but got '{encoding_role}'"
    
    def test_numeric_pattern_recognition(self):
        """Test that column names with numeric indicators are recognized correctly."""
        # Create data with column names that suggest numeric nature
        numeric_indicator_data = pd.DataFrame({
            'bedroom_count': [1, 2, 3, 4, 5],  # Should be numeric despite few unique values
            'room_number': [1, 2, 3, 4, 5],    # Should be numeric
            'quality_score': [1, 2, 3, 4, 5],  # Should be numeric
            'rating_value': [1, 2, 3, 4, 5],   # Should be numeric
            'age_years': [1, 2, 3, 4, 5],      # Should be numeric
            'area_sqft': [1, 2, 3, 4, 5],      # Should be numeric
        })
        
        schemas = feature_definition_logic.suggest_initial_feature_schemas(numeric_indicator_data)
        
        for column in numeric_indicator_data.columns:
            encoding_role = schemas[column]['suggested_encoding_role']
            assert encoding_role in ['numeric-discrete', 'numeric-continuous'], \
                f"Column '{column}' with numeric indicator should be numeric but got '{encoding_role}'"
    
    def test_consecutive_integers_are_numeric(self):
        """Test that consecutive integer sequences are treated as numeric."""
        consecutive_data = pd.DataFrame({
            'consecutive_1_5': [1, 2, 3, 4, 5],      # Consecutive from 1
            'consecutive_0_4': [0, 1, 2, 3, 4],      # Consecutive from 0
            'consecutive_2_6': [2, 3, 4, 5, 6],      # Consecutive from 2
        })
        
        schemas = feature_definition_logic.suggest_initial_feature_schemas(consecutive_data)
        
        for column in consecutive_data.columns:
            encoding_role = schemas[column]['suggested_encoding_role']
            assert encoding_role == 'numeric-discrete', \
                f"Consecutive integers '{column}' should be numeric-discrete but got '{encoding_role}'"
    
    def test_non_consecutive_integers_may_be_categorical(self):
        """Test that non-consecutive integers without numeric indicators might be categorical."""
        non_consecutive_data = pd.DataFrame({
            'weird_ids': [1, 3, 7, 15, 31],      # Non-consecutive, no numeric indicators
            'status_codes': [100, 200, 404, 500, 503],  # Non-consecutive
        })
        
        schemas = feature_definition_logic.suggest_initial_feature_schemas(non_consecutive_data)
        
        # These might be categorical since they're non-consecutive and have no numeric indicators
        for column in non_consecutive_data.columns:
            encoding_role = schemas[column]['suggested_encoding_role']
            # Could be either, but the important thing is the logic considers them
            assert encoding_role in ['numeric-discrete', 'categorical-nominal'], \
                f"Non-consecutive integers '{column}' got unexpected role '{encoding_role}'"
    
    def test_float_columns_always_numeric(self):
        """Test that float columns are always treated as numeric regardless of unique count."""
        float_data = pd.DataFrame({
            'few_floats': [1.0, 2.0, 3.0],           # Few unique floats
            'many_floats': [1.1, 2.2, 3.3]          # Same length
        })
        
        schemas = feature_definition_logic.suggest_initial_feature_schemas(float_data)
        
        for column in float_data.columns:
            encoding_role = schemas[column]['suggested_encoding_role']
            assert encoding_role == 'numeric-continuous', \
                f"Float column '{column}' should be numeric-continuous but got '{encoding_role}'"
    
    def test_real_house_prices_dataset(self):
        """Test with the actual house prices dataset to ensure fix works."""
        # Load the real fixture data
        df = pd.read_csv('../data/fixtures/house_prices.csv')
        schemas = feature_definition_logic.suggest_initial_feature_schemas(df)
        
        # Verify specific columns that were problematic
        expected_numeric = {
            'square_feet': ['numeric-discrete', 'numeric-continuous'],
            'bedrooms': ['numeric-discrete'], 
            'bathrooms': ['numeric-continuous'],
            'house_age_years': ['numeric-discrete', 'numeric-continuous'],
            'garage_spaces': ['numeric-discrete'],
            'neighborhood_quality_score': ['numeric-discrete'],
            'school_district_rating': ['numeric-continuous'],
            'distance_to_city_miles': ['numeric-continuous']
        }
        
        expected_categorical = {
            'property_type': ['categorical-nominal']
        }
        
        # Check numeric features
        for column, valid_roles in expected_numeric.items():
            if column in schemas:
                actual_role = schemas[column]['suggested_encoding_role']
                assert actual_role in valid_roles, \
                    f"Column '{column}' expected {valid_roles} but got '{actual_role}'"
        
        # Check categorical features  
        for column, valid_roles in expected_categorical.items():
            if column in schemas:
                actual_role = schemas[column]['suggested_encoding_role']
                assert actual_role in valid_roles, \
                    f"Column '{column}' expected {valid_roles} but got '{actual_role}'"
    
    def test_encoding_consistency_across_runs(self, sample_house_data):
        """Test that the same data produces consistent encoding suggestions."""
        schemas1 = feature_definition_logic.suggest_initial_feature_schemas(sample_house_data)
        schemas2 = feature_definition_logic.suggest_initial_feature_schemas(sample_house_data.copy())
        
        for column in sample_house_data.columns:
            role1 = schemas1[column]['suggested_encoding_role']
            role2 = schemas2[column]['suggested_encoding_role']
            assert role1 == role2, \
                f"Inconsistent encoding for '{column}': '{role1}' vs '{role2}'"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"]) 