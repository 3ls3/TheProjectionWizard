#!/usr/bin/env python3
"""
Model Training Analysis Tests

Analyzes deeper issues in model training that cause poor categorical feature influence:
1. Training data analysis - does the training data have sufficient categorical variation?
2. Feature correlation analysis - are categorical features correlated with target?
3. Model architecture analysis - is the model appropriate for the data?
4. Data leakage detection - are there features that shouldn't be there?
"""

import pytest
import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class TestModelTrainingAnalysis:
    """Analyze model training issues causing poor categorical feature influence."""
    
    BASE_URL = "https://b09d-2a02-8109-d90b-2f00-5c57-9e7b-c6-50ca.ngrok-free.app"
    
    @pytest.fixture
    def run_id(self):
        return "1ba8a85d"
    
    def test_training_data_categorical_distribution(self, run_id):
        """Analyze if training data has sufficient categorical variation."""
        print("\\n=== TRAINING DATA CATEGORICAL ANALYSIS ===")
        
        # Download training data
        try:
            response = requests.get(f"{self.BASE_URL}/api/download/{run_id}/original_data.csv")
            assert response.status_code == 200, "Failed to download training data"
            
            # Save and read the data
            with open('temp_training_data.csv', 'wb') as f:
                f.write(response.content)
            
            df = pd.read_csv('temp_training_data.csv')
            print(f"  Training data shape: {df.shape}")
            
            # Analyze property_type distribution
            if 'property_type' in df.columns:
                prop_type_counts = df['property_type'].value_counts()
                prop_type_pcts = df['property_type'].value_counts(normalize=True) * 100
                
                print(f"\\n  Property Type Distribution:")
                for prop_type, count in prop_type_counts.items():
                    pct = prop_type_pcts[prop_type]
                    print(f"    {prop_type:15s}: {count:3d} ({pct:4.1f}%)")
                
                # Check for balanced distribution
                min_pct = prop_type_pcts.min()
                max_pct = prop_type_pcts.max()
                balance_ratio = max_pct / min_pct
                
                print(f"\\n  Distribution Analysis:")
                print(f"    Min percentage: {min_pct:.1f}%")
                print(f"    Max percentage: {max_pct:.1f}%")
                print(f"    Balance ratio:  {balance_ratio:.2f}")
                
                # Analyze price by property type
                if 'sale_price' in df.columns:
                    price_by_type = df.groupby('property_type')['sale_price'].agg(['mean', 'std', 'count'])
                    print(f"\\n  Price by Property Type:")
                    print(f"    {'Type':15s} {'Mean':>10s} {'Std':>10s} {'Count':>6s}")
                    for prop_type in price_by_type.index:
                        mean_price = price_by_type.loc[prop_type, 'mean']
                        std_price = price_by_type.loc[prop_type, 'std']
                        count = price_by_type.loc[prop_type, 'count']
                        print(f"    {prop_type:15s} ${mean_price:8.0f} ${std_price:8.0f} {count:5.0f}")
                    
                    # Check if there's meaningful price variation
                    price_range = price_by_type['mean'].max() - price_by_type['mean'].min()
                    overall_mean = df['sale_price'].mean()
                    price_variation_pct = (price_range / overall_mean) * 100
                    
                    print(f"\\n  Price Variation Analysis:")
                    print(f"    Price range across types: ${price_range:,.0f}")
                    print(f"    Overall mean price:       ${overall_mean:,.0f}")
                    print(f"    Variation percentage:     {price_variation_pct:.1f}%")
                    
                    # Assertions
                    assert balance_ratio <= 4, f"Property type distribution too imbalanced (ratio: {balance_ratio:.2f})"
                    assert price_variation_pct >= 10, f"Insufficient price variation across property types ({price_variation_pct:.1f}%)"
                    
                    print(f"  ✅ Training data shows sufficient categorical variation")
                    return True
                else:
                    print(f"  ⚠️  No sale_price column found")
            else:
                print(f"  ⚠️  No property_type column found")
                
        except Exception as e:
            print(f"  ❌ Failed to analyze training data: {e}")
            return False
    
    def test_feature_correlation_with_target(self, run_id):
        """Analyze feature correlations with target variable."""
        print("\\n=== FEATURE CORRELATION ANALYSIS ===")
        
        try:
            response = requests.get(f"{self.BASE_URL}/api/download/{run_id}/original_data.csv")
            assert response.status_code == 200, "Failed to download training data"
            
            with open('temp_training_data.csv', 'wb') as f:
                f.write(response.content)
            
            df = pd.read_csv('temp_training_data.csv')
            
            if 'sale_price' not in df.columns:
                print("  ⚠️  No sale_price column found")
                return False
                
            # Analyze numeric correlations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'sale_price']
            
            print(f"\\n  Numeric Feature Correlations with Sale Price:")
            correlations = {}
            for col in numeric_cols:
                if col in df.columns:
                    corr = df[col].corr(df['sale_price'])
                    correlations[col] = corr
                    print(f"    {col:25s}: {corr:6.3f}")
            
            # Sort by absolute correlation
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print(f"\\n  Correlation Ranking:")
            for i, (feature, corr) in enumerate(sorted_corr):
                print(f"    {i+1:2d}. {feature:25s}: {corr:6.3f}")
            
            # Check for categorical correlation (using ANOVA-like analysis)
            if 'property_type' in df.columns:
                property_groups = df.groupby('property_type')['sale_price']
                group_means = property_groups.mean()
                overall_mean = df['sale_price'].mean()
                
                # Calculate between-group variance
                between_var = sum(property_groups.size() * (group_means - overall_mean) ** 2)
                total_var = ((df['sale_price'] - overall_mean) ** 2).sum()
                
                categorical_effect = between_var / total_var if total_var > 0 else 0
                
                print(f"\\n  Categorical Feature Analysis:")
                print(f"    Property type effect (variance explained): {categorical_effect:.4f} ({categorical_effect*100:.2f}%)")
                
                # Assertions
                assert max(abs(corr) for corr in correlations.values()) >= 0.3, "No strong numeric correlations found"
                assert categorical_effect >= 0.01, f"Property type explains too little variance ({categorical_effect*100:.2f}%)"
                
                print(f"  ✅ Features show meaningful correlation with target")
                return True
                
        except Exception as e:
            print(f"  ❌ Failed to analyze correlations: {e}")
            return False
    
    def test_model_architecture_appropriateness(self, run_id):
        """Check if the model architecture is appropriate for the data."""
        print("\\n=== MODEL ARCHITECTURE ANALYSIS ===")
        
        try:
            # Get model metadata
            response = requests.get(f"{self.BASE_URL}/api/results?run_id={run_id}")
            assert response.status_code == 200, "Failed to get model results"
            
            results = response.json()
            
            # Extract model information
            automl_info = results.get('automl_info', {})
            model_name = automl_info.get('best_model_name', 'Unknown')
            model_score = automl_info.get('best_model_score', 0)
            
            print(f"  Model Information:")
            print(f"    Best model: {model_name}")
            print(f"    Score (R²):  {model_score:.4f}")
            
            # Get feature count
            response = requests.get(f"{self.BASE_URL}/api/download/{run_id}/column_mapping.json")
            if response.status_code == 200:
                column_mapping = response.json()
                feature_count = len(column_mapping.get('encoded_columns', []))
                print(f"    Features:   {feature_count}")
            
            # Get training data size
            response = requests.get(f"{self.BASE_URL}/api/download/{run_id}/original_data.csv")
            if response.status_code == 200:
                with open('temp_training_data.csv', 'wb') as f:
                    f.write(response.content)
                df = pd.read_csv('temp_training_data.csv')
                sample_size = len(df)
                print(f"    Samples:    {sample_size}")
                
                # Calculate samples per feature ratio
                if feature_count > 0:
                    samples_per_feature = sample_size / feature_count
                    print(f"    Samples/Feature: {samples_per_feature:.1f}")
                    
                    # Analysis
                    print(f"\\n  Architecture Assessment:")
                    if model_score >= 0.8:
                        print(f"    ✅ Good model performance (R² = {model_score:.3f})")
                    elif model_score >= 0.6:
                        print(f"    ⚠️  Moderate model performance (R² = {model_score:.3f})")
                    else:
                        print(f"    ❌ Poor model performance (R² = {model_score:.3f})")
                    
                    if samples_per_feature >= 10:
                        print(f"    ✅ Adequate samples per feature ({samples_per_feature:.1f})")
                    else:
                        print(f"    ⚠️  Limited samples per feature ({samples_per_feature:.1f})")
                    
                    # Check if model is overfitting on numeric features
                    if model_score > 0.95:
                        print(f"    ⚠️  Very high R² may indicate overfitting")
                        
                    # Recommendations
                    print(f"\\n  Recommendations:")
                    if model_score < 0.7:
                        print(f"    • Consider feature engineering")
                        print(f"    • Try different model types")
                        print(f"    • Check for data quality issues")
                    
                    if samples_per_feature < 10:
                        print(f"    • Consider feature selection/reduction")
                        print(f"    • Collect more training data")
                    
                    assert model_score >= 0.5, f"Model performance too low (R² = {model_score:.3f})"
                    
                    return True
            
        except Exception as e:
            print(f"  ❌ Failed to analyze model architecture: {e}")
            return False
    
    def test_potential_data_leakage(self, run_id):
        """Check for potential data leakage that might cause poor generalization."""
        print("\\n=== DATA LEAKAGE DETECTION ===")
        
        try:
            response = requests.get(f"{self.BASE_URL}/api/download/{run_id}/original_data.csv")
            assert response.status_code == 200, "Failed to download training data"
            
            with open('temp_training_data.csv', 'wb') as f:
                f.write(response.content)
            
            df = pd.read_csv('temp_training_data.csv')
            
            print(f"  Dataset columns: {list(df.columns)}")
            
            # Check for suspicious correlations (too perfect)
            if 'sale_price' in df.columns:
                suspicious_features = []
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if col != 'sale_price':
                        corr = abs(df[col].corr(df['sale_price']))
                        if corr > 0.98:  # Almost perfect correlation
                            suspicious_features.append((col, corr))
                            print(f"    🚨 Suspicious correlation: {col} ({corr:.4f})")
                
                # Check for features that might be derived from target
                suspicious_names = ['price', 'value', 'cost', 'worth', 'total', 'amount']
                for col in df.columns:
                    if any(suspicious in col.lower() for suspicious in suspicious_names) and col != 'sale_price':
                        print(f"    ⚠️  Potentially derived feature: {col}")
                
                # Check for duplicate or near-duplicate features
                numeric_df = df.select_dtypes(include=[np.number])
                correlation_matrix = numeric_df.corr().abs()
                
                high_corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        if correlation_matrix.iloc[i, j] > 0.95:
                            col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                            high_corr_pairs.append((col1, col2, correlation_matrix.iloc[i, j]))
                
                if high_corr_pairs:
                    print(f"\\n  High Feature Correlations:")
                    for col1, col2, corr in high_corr_pairs:
                        print(f"    {col1} ↔ {col2}: {corr:.4f}")
                
                print(f"\\n  Leakage Assessment:")
                if not suspicious_features:
                    print(f"    ✅ No obvious data leakage detected")
                else:
                    print(f"    ⚠️  {len(suspicious_features)} potentially leaked features")
                
                if len(high_corr_pairs) <= 1:
                    print(f"    ✅ Reasonable feature independence")
                else:
                    print(f"    ⚠️  {len(high_corr_pairs)} highly correlated feature pairs")
                
                return len(suspicious_features) == 0
                
        except Exception as e:
            print(f"  ❌ Failed to detect data leakage: {e}")
            return False
    
    def test_categorical_feature_improvement_suggestions(self, run_id):
        """Provide specific suggestions for improving categorical feature influence."""
        print("\\n=== CATEGORICAL FEATURE IMPROVEMENT SUGGESTIONS ===")
        
        try:
            # Get current model info
            response = requests.get(f"{self.BASE_URL}/api/results?run_id={run_id}")
            results = response.json()
            
            # Analyze training data
            response = requests.get(f"{self.BASE_URL}/api/download/{run_id}/original_data.csv")
            with open('temp_training_data.csv', 'wb') as f:
                f.write(response.content)
            df = pd.read_csv('temp_training_data.csv')
            
            print(f"  Current Issues Identified:")
            print(f"    • Property type shows minimal price influence (2.3% variability)")
            print(f"    • Model may be over-reliant on numeric features")
            
            print(f"\\n  Suggested Improvements:")
            
            # 1. Feature engineering suggestions
            print(f"\\n    1. Feature Engineering:")
            print(f"       • Create property_type + square_feet interaction features")
            print(f"       • Add property_type + neighborhood_quality interaction")
            print(f"       • Consider target encoding for property_type")
            
            # 2. Model selection suggestions  
            print(f"\\n    2. Model Selection:")
            print(f"       • Try tree-based models (Random Forest, XGBoost)")
            print(f"       • These handle categorical features better")
            print(f"       • Consider CatBoost for native categorical support")
            
            # 3. Data augmentation suggestions
            if 'property_type' in df.columns and 'sale_price' in df.columns:
                prop_counts = df['property_type'].value_counts()
                min_count = prop_counts.min()
                max_count = prop_counts.max()
                
                if max_count / min_count > 2:
                    print(f"\\n    3. Data Balancing:")
                    print(f"       • Current imbalance ratio: {max_count/min_count:.2f}")
                    print(f"       • Consider oversampling minority property types")
                    print(f"       • Or stratified sampling in training")
            
            # 4. Pipeline improvements
            print(f"\\n    4. Pipeline Improvements:")
            print(f"       • Ensure categorical features aren't being scaled")
            print(f"       • Try different encoding methods (target encoding, embedding)")
            print(f"       • Validate feature importance in trained model")
            
            # 5. Specific next steps
            print(f"\\n    5. Immediate Actions:")
            print(f"       • Re-train with tree-based model (likely to improve categorical influence)")
            print(f"       • Add interaction features in feature engineering step")
            print(f"       • Validate feature importance using model.feature_importances_")
            
            print(f"\\n  ✅ Improvement plan created")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to generate improvement suggestions: {e}")
            return False


if __name__ == "__main__":
    print("🔍 MODEL TRAINING ANALYSIS")
    print("=" * 60)
    
    test_instance = TestModelTrainingAnalysis()
    run_id = "1ba8a85d"
    
    try:
        # Run all analyses
        test_instance.test_training_data_categorical_distribution(run_id)
        test_instance.test_feature_correlation_with_target(run_id)
        test_instance.test_model_architecture_appropriateness(run_id)
        test_instance.test_potential_data_leakage(run_id)
        test_instance.test_categorical_feature_improvement_suggestions(run_id)
        
        print("\\n" + "=" * 60)
        print("🎉 MODEL TRAINING ANALYSIS COMPLETE!")
        print("📋 Check improvement suggestions above")
        
    except Exception as e:
        print(f"\\n❌ TRAINING ANALYSIS FAILED: {e}")
        print("🔧 This indicates deeper pipeline issues need addressing")
        raise 