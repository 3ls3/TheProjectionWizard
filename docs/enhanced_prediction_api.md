# Enhanced Prediction API Documentation

## Overview

The Enhanced Prediction API provides a comprehensive interface for making predictions with rich explanations, real-time feature importance, and detailed analysis. This API has been **significantly enhanced** with real SHAP (SHapley Additive exPlanations) integration, providing authentic feature importance and contribution analysis.

## Key Features

### 🎯 **Real SHAP Integration**
- **Authentic SHAP Values**: Uses the existing SHAP logic from `pipeline/step_6_explain/shap_logic.py`
- **Multiple Explainer Support**: Automatically selects best explainer (TreeExplainer → Explainer → KernelExplainer)
- **Graceful Fallback**: Falls back to model-based importance if SHAP computation fails
- **Feature Contribution Analysis**: Real SHAP values drive feature contribution rankings

### 📊 **Enhanced Schema with Feature Importance**
- **SHAP-Based Feature Ranking**: Features ordered by real SHAP importance scores
- **Smart UI Guidance**: High/medium/low priority features for optimal UX
- **Automated Categorization**: Features grouped by type (demographic, financial, behavioral, etc.)
- **Slider Configuration**: Intelligent min/max/step values with suggested defaults

### ⚡ **Comprehensive Prediction Analysis**
- **Single Predictions**: Enhanced predictions with real SHAP explanations
- **Batch Processing**: Up to 100 predictions with performance summaries
- **Scenario Comparison**: Side-by-side analysis with sensitivity insights
- **Confidence Intervals**: Statistical confidence bounds for predictions

## API Endpoints

### 1. Enhanced Prediction Schema

```http
GET /api/prediction-schema-enhanced?run_id={run_id}
```

**New Features:**
- Real SHAP-based feature importance when model is available
- Falls back to statistical measures if SHAP unavailable
- Features sorted by actual importance scores
- UI guidance based on feature contribution potential

**Response Enhancement:**
```json
{
  "features": [
    {
      "column_name": "income",
      "importance_score": 0.85,
      "importance_rank": 1,
      "shap_available": true,
      "ui_guidance": {
        "priority": "high",
        "recommendation": "Key feature - small changes may significantly impact predictions",
        "highlight": true
      }
    }
  ],
  "schema_metadata": {
    "importance_method": "shap",
    "supports_shap_explanations": true
  },
  "ui_recommendations": {
    "enable_real_time_shap": true,
    "primary_features": ["income", "age", "credit_score"]
  }
}
```

### 2. Enhanced Single Prediction

```http
POST /api/predict/single?run_id={run_id}
```

**Real SHAP Integration:**
- Calculates actual SHAP values for the prediction
- Uses existing SHAP logic with multiple explainer fallbacks
- Provides feature contributions based on real SHAP values
- Includes SHAP base value for complete explanation

**Request:**
```json
{
  "income": 50000,
  "age": 35,
  "credit_score": 720
}
```

**Response:**
```json
{
  "prediction_value": 0.85,
  "prediction_id": "pred_abc123",
  "shap_values": {
    "income": 0.12,
    "age": -0.05,
    "credit_score": 0.18
  },
  "shap_base_value": 0.6,
  "feature_contributions": [
    {
      "feature_name": "credit_score",
      "contribution_value": 0.18,
      "feature_value": 720,
      "contribution_direction": "positive",
      "shap_value": 0.18
    }
  ],
  "shap_available": true,
  "shap_fallback_used": false,
  "top_contributing_features": ["credit_score", "income"]
}
```

### 3. SHAP Explanation Endpoint

```http
GET /api/predict/explain/{prediction_id}?run_id={run_id}
```

**Real SHAP Explanations:**
- Generates on-demand SHAP explanations for any input
- Uses the same SHAP logic as the batch explanation system
- Provides detailed feature-by-feature contribution analysis
- Includes explanation confidence and methodology used

**Request Body:**
```json
{
  "income": 50000,
  "age": 35,
  "credit_score": 720
}
```

**Response:**
```json
{
  "prediction_id": "pred_abc123",
  "shap_values": {
    "income": 0.12,
    "age": -0.05,
    "credit_score": 0.18
  },
  "shap_base_value": 0.6,
  "feature_contributions": [
    {
      "feature_name": "credit_score",
      "shap_value": 0.18,
      "contribution_direction": "positive"
    }
  ],
  "explanation_summary": "SHAP explanation for prediction pred_abc123. Real SHAP values calculated. Top contributing features: credit_score, income, age",
  "shap_available": true,
  "fallback_used": false,
  "explanation_timestamp": "2024-01-15T10:30:00Z"
}
```

## Technical Implementation

### SHAP Integration Architecture

1. **Model Loading**: Loads trained PyCaret pipeline from GCS
2. **Explainer Selection**: Automatically selects optimal SHAP explainer:
   - **TreeExplainer**: For tree-based models (fastest)
   - **Explainer**: General-purpose explainer for any model
   - **KernelExplainer**: Fallback for complex models
3. **SHAP Calculation**: Computes real SHAP values using existing logic
4. **Graceful Fallback**: Uses model-based importance if SHAP fails

### Feature Importance Calculation

```python
def get_global_feature_importance_from_shap(model, sample_data, target_column, task_type):
    """
    Calculate global feature importance using SHAP on a sample of data.
    
    Uses existing SHAP logic with multiple explainer fallbacks.
    """
    # Try TreeExplainer first (fastest for tree models)
    # Fall back to general Explainer
    # Final fallback to KernelExplainer
    # Calculate mean absolute SHAP values as importance
```

### Enhanced Schema Generation

```python
def generate_enhanced_prediction_schema(df, target_column, metadata):
    """
    Generate enhanced schema with real SHAP-based feature importance.
    
    - Loads model from metadata if available
    - Calculates SHAP-based importance scores
    - Sorts features by importance
    - Provides UI guidance based on importance
    """
```

## Performance Considerations

### SHAP Computation Optimization

1. **Sample Size Limiting**: 
   - Single predictions: Direct calculation
   - Feature importance: Limited to 100 samples
   - Background samples: Adaptive sizing

2. **Explainer Selection**:
   - TreeExplainer: Fastest for tree models
   - Kernel fallback: Minimal background samples (10)
   - Error handling: Graceful degradation

3. **Caching Strategy**:
   - Feature importance cached per model
   - SHAP values calculated on-demand
   - Fallback to statistical measures when needed

## Error Handling

### SHAP Fallback Mechanism

1. **SHAP Computation Fails**: Falls back to model-based importance
2. **Model Unavailable**: Uses statistical measures (coefficient of variation)
3. **Explainer Errors**: Tries multiple explainer types before giving up
4. **Graceful Degradation**: Always provides some form of feature ranking

### Response Indicators

- `shap_available`: Whether real SHAP values are provided
- `shap_fallback_used`: Whether fallback importance was used
- `importance_method`: "shap" or "statistical"
- `supports_shap_explanations`: Whether model supports SHAP

## Frontend Integration

### Real-Time SHAP Updates

```javascript
// Enable real-time SHAP updates based on schema
if (schema.ui_recommendations.enable_real_time_shap) {
  // Set up slider change handlers for instant SHAP feedback
  setupRealTimeShapUpdates(schema.features);
}

// Highlight high-importance features
schema.features
  .filter(f => f.ui_guidance.highlight)
  .forEach(f => highlightFeature(f.column_name));
```

### Feature Importance Visualization

```javascript
// Sort features by importance for display
const sortedFeatures = schema.features
  .sort((a, b) => b.importance_score - a.importance_score);

// Create importance indicators
sortedFeatures.forEach(feature => {
  const indicator = createImportanceIndicator(
    feature.importance_score,
    feature.ui_guidance.priority
  );
});
```

## Benefits of Real SHAP Integration

### 🎯 **Authentic Explanations**
- Real SHAP values instead of approximations
- Mathematically sound feature contributions
- Consistent with model's actual decision process

### 🚀 **Enhanced User Experience**
- Features ordered by actual importance
- Smart UI guidance based on real impact
- Confidence in explanation quality

### 🔧 **Robust Implementation**
- Multiple fallback mechanisms
- Graceful error handling
- Backward compatibility maintained

### 📊 **Accurate Insights**
- Global feature importance from SHAP
- Local explanations for each prediction
- Consistent methodology across endpoints

## Backward Compatibility

All existing endpoints continue to work unchanged. The enhanced functionality is additive:

- Existing `/predict` endpoint unchanged
- New `/predict/single` provides enhanced features
- Schema endpoints provide additional metadata
- Graceful fallback ensures functionality without SHAP

## Next Steps

1. **Frontend Development**: Implement rich slider interfaces using the enhanced schema
2. **Real-Time Updates**: Build responsive UI that updates SHAP explanations as users adjust sliders
3. **Visualization**: Create SHAP-based feature importance charts
4. **Caching**: Implement intelligent caching for frequently-used SHAP calculations

The Enhanced Prediction API now provides a solid foundation for building sophisticated, explainable ML interfaces with real SHAP integration and comprehensive feature analysis.

## Migration Path

The enhanced API endpoints coexist with existing endpoints:
- Existing `/api/predict` continues to work
- New endpoints are additive, no breaking changes
- Frontend can gradually adopt enhanced features
- Legacy integrations remain functional
