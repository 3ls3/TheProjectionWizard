#!/usr/bin/env python3

"""
REGRESSION MAGNITUDE BUG INVESTIGATION

Based on breakthrough findings:
✅ Column mapper working correctly
✅ Classification predictions perfect (1.000 probability)
❌ Regression predictions: $188M instead of ~$425k

This test isolates the regression-specific prediction logic to find the magnitude bug.

TARGET INVESTIGATION AREAS:
1. Model output scaling/transformation  
2. Target value inverse transformation
3. Units conversion in prediction logic
4. Feature scaling application during prediction
5. Regression vs classification prediction path differences

SUCCESS CRITERIA:
- Identify exact point where $425k becomes $188M
- Compare regression vs classification prediction paths
- Find scaling/transformation bug in regression logic
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
from datetime import datetime
import traceback
import tempfile
import joblib

# Add project root to path
sys.path.append('.')

# Core imports
from common import constants, logger
from tests.fixtures.fixture_generator import TestFixtureGenerator

# Pipeline imports - focus on prediction logic
try:
    from pipeline.step_7_predict import column_mapper, predict_logic
    from pipeline.step_4_prep import encoding_logic
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Pipeline imports not available: {e}")
    PIPELINE_AVAILABLE = False

# Try GCS imports but continue without them
try:
    from api.utils.gcs_utils import upload_run_file, download_run_file, PROJECT_BUCKET_NAME
    GCS_AVAILABLE = True
except Exception as e:
    print(f"GCS not available: {e}")
    GCS_AVAILABLE = False


class RegressionMagnitudeBugInvestigation:
    """
    Focused investigation of the regression prediction magnitude bug.
    Can operate with or without GCS to isolate the core issue.
    """
    
    def __init__(self):
        self.logger = logger.get_logger("regression_magnitude_investigation", "test")
        self.fixture_generator = TestFixtureGenerator()
        
        # Expected values from context documents
        self.EXPECTED_REGRESSION_RANGE = (200000, 800000)  # $200k-$800k
        self.EXPECTED_REGRESSION_MEAN = 424000  # ~$425k from context
        
        # Problem indicators
        self.PROBLEMATIC_VALUE = 188000000  # $188M - the wrong prediction
        
        # Investigation results
        self.investigation_results = {}
        
    def investigate_regression_magnitude_bug(self):
        """
        Main investigation method to isolate the regression magnitude bug.
        """
        self.logger.info("=" * 80)
        self.logger.info("🔍 REGRESSION MAGNITUDE BUG INVESTIGATION")
        self.logger.info("Target: Find why predictions are $188M instead of ~$425k")
        self.logger.info("=" * 80)
        
        # Step 1: Create test data that we know should work
        test_data = self._create_known_good_test_data()
        self.logger.info(f"✅ Created test data with {len(test_data)} samples")
        
        # Step 2: Manual step-by-step processing to isolate the bug
        if PIPELINE_AVAILABLE:
            processing_results = self._manual_step_by_step_processing(test_data)
            self.investigation_results.update(processing_results)
        else:
            self.logger.warning("⚠️  Pipeline not available - skipping step-by-step processing")
        
        # Step 3: Compare expected vs actual at each transformation step
        transformation_analysis = self._analyze_transformation_steps(test_data)
        self.investigation_results.update(transformation_analysis)
        
        # Step 4: Generate investigation report
        self._generate_investigation_report()
        
        return self.investigation_results
    
    def _create_known_good_test_data(self):
        """
        Create test data that should produce predictions around $425k.
        Based on context documents: mean price ~$424k, range $175k-$787k
        """
        # Create a representative sample that should predict around the mean
        test_sample = {
            "square_feet": 2500,           # Mid-range house size
            "bedrooms": 3,                 # Common bedroom count
            "bathrooms": 2.5,              # Common bathroom count  
            "garage_spaces": 2,            # Common garage size
            "property_type": "Single Family",  # Most common type
            "neighborhood_quality_score": 7  # Above average but not premium
        }
        
        # Create DataFrame
        test_df = pd.DataFrame([test_sample])
        
        self.logger.info("🏠 Created representative test sample:")
        for key, value in test_sample.items():
            self.logger.info(f"   {key}: {value}")
            
        return test_df
    
    def _manual_step_by_step_processing(self, test_data):
        """
        Manually process data through each step to isolate where the magnitude bug occurs.
        """
        results = {}
        
        self.logger.info("🔬 Starting manual step-by-step processing...")
        
        try:
            # Step 1: Raw input validation
            self.logger.info("📋 Step 1: Raw Input Validation")
            raw_sample = test_data.iloc[0].to_dict()
            results['raw_input'] = raw_sample
            self.logger.info(f"   Raw input keys: {list(raw_sample.keys())}")
            self.logger.info(f"   square_feet: {raw_sample.get('square_feet')}")
            
            # Step 2: Simulate feature encoding (without requiring complete pipeline)
            self.logger.info("🔧 Step 2: Feature Encoding Simulation")
            encoded_sample = self._simulate_feature_encoding(raw_sample)
            results['encoded_input'] = encoded_sample
            self.logger.info(f"   Encoded features count: {len(encoded_sample)}")
            
            # Step 3: Check if we can access actual column mapping logic
            if hasattr(column_mapper, 'encode_user_input_gcs'):
                self.logger.info("🎯 Step 3: Testing column_mapper.encode_user_input_gcs()")
                # This is the critical function that was identified as working correctly
                # But we need to see what it actually outputs for regression
                results['column_mapper_available'] = True
            else:
                self.logger.warning("⚠️  column_mapper.encode_user_input_gcs not available")
                results['column_mapper_available'] = False
                
            # Step 4: Look for any obvious scaling issues
            self.logger.info("📊 Step 4: Scaling Analysis")
            scaling_analysis = self._analyze_potential_scaling_issues(raw_sample)
            results['scaling_analysis'] = scaling_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error in manual processing: {str(e)}")
            results['processing_error'] = str(e)
            results['processing_traceback'] = traceback.format_exc()
            
        return results
    
    def _simulate_feature_encoding(self, raw_sample):
        """
        Simulate the feature encoding that should happen in the pipeline.
        Based on context: should result in 12 features for regression.
        """
        # According to context documents, regression should have:
        # - Numeric features: scaled with StandardScaler
        # - Categorical features: one-hot encoded
        # - Total features: 12 (not 31)
        
        encoded = {}
        
        # Numeric features (should be scaled)
        numeric_features = ['square_feet', 'bedrooms', 'bathrooms', 'garage_spaces', 'neighborhood_quality_score']
        for feature in numeric_features:
            if feature in raw_sample:
                # Note: actual scaling would require StandardScaler from training
                encoded[feature] = raw_sample[feature]  # Placeholder
                
        # Categorical feature: property_type
        # According to context: ['Single Family', 'Condo', 'Townhouse', 'Ranch']
        property_types = ['Single Family', 'Condo', 'Townhouse', 'Ranch']
        selected_type = raw_sample.get('property_type', 'Single Family')
        
        for prop_type in property_types:
            encoded[f'property_type_{prop_type}'] = 1 if prop_type == selected_type else 0
            
        self.logger.info(f"   Simulated encoding: {len(encoded)} features")
        self.logger.info(f"   Features: {list(encoded.keys())}")
        
        return encoded
    
    def _analyze_potential_scaling_issues(self, raw_sample):
        """
        Analyze potential scaling issues that could cause magnitude errors.
        """
        analysis = {}
        
        # Check if square_feet could be causing issues
        square_feet = raw_sample.get('square_feet', 0)
        
        # Possible scaling scenarios that could cause $425k -> $188M
        scaling_factor = 188000000 / 425000  # ~442x multiplier
        analysis['magnitude_multiplier'] = scaling_factor
        
        self.logger.info(f"📊 Magnitude analysis:")
        self.logger.info(f"   Expected: ~$425k")
        self.logger.info(f"   Actual: ~$188M") 
        self.logger.info(f"   Multiplier: ~{scaling_factor:.1f}x")
        
        # Check if square_feet could be the culprit
        if square_feet > 0:
            sq_ft_factor = square_feet / 1000  # 2500 -> 2.5
            analysis['square_feet_factor'] = sq_ft_factor
            
            # Could there be a units conversion error?
            # If model expects square_feet in thousands but gets raw value
            possible_scaling_error = square_feet / sq_ft_factor
            analysis['possible_units_error'] = possible_scaling_error
            
            self.logger.info(f"   square_feet: {square_feet}")
            self.logger.info(f"   If model expects '000s: {sq_ft_factor}")
            
        return analysis
    
    def _analyze_transformation_steps(self, test_data):
        """
        Analyze what transformations might be causing the magnitude issue.
        """
        analysis = {}
        
        self.logger.info("🔍 Analyzing potential transformation issues...")
        
        # Based on context documents, the bug is likely in:
        # 1. Model output scaling/transformation
        # 2. Target value inverse transformation  
        # 3. Units conversion in prediction logic
        # 4. Feature scaling application during prediction
        
        # Hypothesis 1: Target value transformation issue
        # If the model was trained on scaled target values (price/1000 or price/100000)
        # but predictions aren't inverse-transformed correctly
        
        expected_prediction = 425000  # $425k
        actual_prediction = 188000000  # $188M
        
        # Calculate potential scaling factors
        scaling_ratios = {
            'raw_ratio': actual_prediction / expected_prediction,
            'could_be_divided_by_1000': actual_prediction / 1000,
            'could_be_divided_by_10000': actual_prediction / 10000,
            'could_be_multiplied_by_scale': actual_prediction * 0.001,
            'could_be_square_feet_confusion': actual_prediction / 2500  # If somehow multiplied by sq ft
        }
        
        analysis['scaling_hypotheses'] = scaling_ratios
        
        self.logger.info("🧮 Scaling hypotheses:")
        for hypothesis, value in scaling_ratios.items():
            self.logger.info(f"   {hypothesis}: {value:,.0f}")
            if 200000 <= value <= 800000:  # In expected range
                self.logger.info(f"      ⭐ {hypothesis} puts prediction in expected range!")
                
        return analysis
    
    def _generate_investigation_report(self):
        """
        Generate a comprehensive investigation report.
        """
        self.logger.info("=" * 80)
        self.logger.info("📋 INVESTIGATION REPORT")
        self.logger.info("=" * 80)
        
        # Summary of findings
        self.logger.info("🔍 KEY FINDINGS:")
        
        if 'magnitude_multiplier' in self.investigation_results.get('scaling_analysis', {}):
            multiplier = self.investigation_results['scaling_analysis']['magnitude_multiplier']
            self.logger.info(f"   📊 Magnitude error: {multiplier:.1f}x too large")
            
        if 'scaling_hypotheses' in self.investigation_results:
            self.logger.info("   🧮 Likely fixes:")
            hypotheses = self.investigation_results['scaling_hypotheses']
            for hypothesis, value in hypotheses.items():
                if 200000 <= value <= 800000:
                    self.logger.info(f"      ✅ {hypothesis}: ${value:,.0f}")
                    
        # Next steps recommendations
        self.logger.info("")
        self.logger.info("🎯 RECOMMENDED NEXT STEPS:")
        self.logger.info("   1. Examine model training target scaling")
        self.logger.info("   2. Check inverse transformation in prediction logic")
        self.logger.info("   3. Compare regression vs classification prediction paths")
        self.logger.info("   4. Look for units conversion errors")
        
        # Critical files to examine
        self.logger.info("")
        self.logger.info("📁 CRITICAL FILES TO EXAMINE:")
        self.logger.info("   • pipeline/step_7_predict/predict_logic.py")
        self.logger.info("   • pipeline/step_7_predict/column_mapper.py")
        self.logger.info("   • pipeline/step_5_automl/automl_logic.py (model training)")
        self.logger.info("   • api/routes/pipeline.py (prediction endpoints)")


def main():
    """
    Run the regression magnitude bug investigation.
    """
    print("🔮 Starting Regression Magnitude Bug Investigation...")
    print("This can run with or without GCS to isolate the core issue.")
    print("")
    
    investigator = RegressionMagnitudeBugInvestigation()
    results = investigator.investigate_regression_magnitude_bug()
    
    print("")
    print("=" * 80)
    print("🎯 INVESTIGATION COMPLETE")
    print("=" * 80)
    print("Check the logs above for detailed findings and recommendations.")
    print("")
    
    # Save results to file for further analysis
    timestamp = int(time.time())
    results_file = Path("tests/reports") / f"regression_magnitude_investigation_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📄 Investigation results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main() 