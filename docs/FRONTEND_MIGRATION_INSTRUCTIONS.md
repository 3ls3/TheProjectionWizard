# Frontend Migration Instructions: Simplified Prediction API

## Overview
The backend has been simplified to remove problematic prediction endpoints and replace them with a reliable `/api/predict/enhanced` endpoint. This document provides step-by-step instructions for updating the frontend.

## 🚨 **CRITICAL CHANGES**

### **Removed Endpoints (DO NOT USE):**
- ❌ `/api/predict/single` - Had zero feature value issues
- ❌ `/api/predict/explain/{prediction_id}` - Complex SHAP calculations
- ❌ `/api/predict/batch` - Complex batch processing  
- ❌ `/api/predict/compare` - Complex comparison logic
- ❌ `/api/prediction-schema-enhanced` - Complex schema generation

### **New/Updated Endpoints:**
- ✅ `/api/predict/enhanced` - **NEW**: Reliable prediction with feature importance
- ✅ `/api/predict` - **UNCHANGED**: Basic prediction (still works)
- ✅ `/api/prediction-schema` - **UNCHANGED**: Basic schema (still works)
- ✅ `/api/results` - **UNCHANGED**: Get feature importance and SHAP plots

---

## 📝 **Step 1: Update `apiclient.ts`**

### **1.1 Remove Deprecated Functions**
Remove these functions entirely:
```typescript
// ❌ REMOVE THESE FUNCTIONS:
export async function getEnhancedPredictionSchema(runId: string): Promise<EnhancedPredictionSchemaResponse>
export async function makeSinglePrediction(runId: string, inputValues: Record<string, any>): Promise<SinglePredictionResponse>
export async function getShapExplanation(runId: string, predictionId: string, inputValues: Record<string, any>): Promise<ShapExplanationResponse>
export async function makeBatchPredictions(...)
export async function comparePredictionScenarios(...)
```

### **1.2 Add New Enhanced Prediction Function**
Add this new function:
```typescript
/**
 * Make an enhanced prediction with feature importance from results
 * Replaces the problematic makeSinglePrediction function
 */
export async function makeEnhancedPrediction(request: PredictionInputRequest): Promise<EnhancedPredictionResponse> {
  return apiFetch<EnhancedPredictionResponse>(`/predict/enhanced`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}
```

### **1.3 Update Imports**
Remove these from the imports:
```typescript
// ❌ REMOVE FROM IMPORTS:
EnhancedPredictionSchemaResponse,
SinglePredictionResponse,
ShapExplanationResponse
```

Add this to the imports:
```typescript
// ✅ ADD TO IMPORTS:
EnhancedPredictionResponse
```

---

## 📝 **Step 2: Update `types.ts`**

### **2.1 Remove Deprecated Types**
Remove these type definitions entirely:
```typescript
// ❌ REMOVE THESE TYPES:
export interface EnhancedPredictionSchemaResponse { ... }
export interface SinglePredictionResponse { ... }
export interface ShapExplanationResponse { ... }
export interface FeatureMetadata { ... }
export interface SliderConfig { ... }
export interface CategoricalConfig { ... }
export interface FeatureContribution { ... }
```

### **2.2 Add New Enhanced Prediction Response Type**
Add this new type:
```typescript
// ✅ ADD THIS NEW TYPE:
export interface EnhancedPredictionResponse {
  api_version: "v1";
  prediction_value: any;
  confidence?: number;
  input_features: Record<string, any>;
  feature_importance: string[];
  feature_importance_scores: Record<string, number>;
  task_type: string;
  target_column: string;
  model_name?: string;
  prediction_timestamp: string;
  shap_plot_available: boolean;
  explainability_available: boolean;
}
```

---

## 📝 **Step 3: Update `PredictionInterface.tsx`**

### **3.1 Complete Rewrite Required**
The current `PredictionInterface.tsx` needs to be completely rewritten because it depends on the removed enhanced schema endpoint. Here's the new approach:

### **3.2 New Implementation Strategy**
```typescript
import React, { useEffect, useState, useCallback } from 'react';
import { 
  getPredictionSchema,  // ✅ Use basic schema instead
  makePrediction,       // ✅ Use basic prediction for real-time updates
  makeEnhancedPrediction, // ✅ Use new enhanced prediction for feature importance
  getResults           // ✅ Get feature importance from results
} from '@/lib/apiClient';
import { 
  PredictionSchemaResponse, 
  PredictionResponse,
  EnhancedPredictionResponse,
  FinalResultsResponse 
} from '@/lib/types';

interface PredictionInterfaceProps {
  runId: string;
}

const PredictionInterface: React.FC<PredictionInterfaceProps> = ({ runId }) => {
  // ✅ NEW STATE STRUCTURE:
  const [schema, setSchema] = useState<PredictionSchemaResponse | null>(null);
  const [results, setResults] = useState<FinalResultsResponse | null>(null);
  const [inputValues, setInputValues] = useState<Record<string, string | number>>({});
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [enhancedPrediction, setEnhancedPrediction] = useState<EnhancedPredictionResponse | null>(null);
  
  // ✅ LOAD BASIC SCHEMA AND RESULTS:
  useEffect(() => {
    const loadData = async () => {
      try {
        // Load basic schema for input controls
        const schemaResponse = await getPredictionSchema(runId);
        setSchema(schemaResponse);
        
        // Load results for feature importance
        const resultsResponse = await getResults(runId);
        setResults(resultsResponse);
        
        // Initialize input values with basic defaults
        const defaults: Record<string, string | number> = {};
        Object.entries(schemaResponse.numeric_columns).forEach(([column, config]) => {
          defaults[column] = config.mean; // Use mean as default
        });
        Object.entries(schemaResponse.categorical_columns).forEach(([column, config]) => {
          defaults[column] = config.default;
        });
        setInputValues(defaults);
        
      } catch (err) {
        console.error('Error loading prediction data:', err);
      }
    };

    if (runId) {
      loadData();
    }
  }, [runId]);

  // ✅ MAKE PREDICTIONS:
  const makePredictionCall = useCallback(async (values: Record<string, string | number>) => {
    if (!schema) return;
    
    try {
      // Make basic prediction for real-time updates
      const basicPrediction = await makePrediction({
        run_id: runId,
        input_values: values
      });
      setPrediction(basicPrediction);
      
      // Make enhanced prediction for feature importance
      const enhanced = await makeEnhancedPrediction({
        run_id: runId,
        input_values: values
      });
      setEnhancedPrediction(enhanced);
      
    } catch (err) {
      console.error('Error making prediction:', err);
    }
  }, [runId, schema]);

  // ✅ RENDER SIMPLIFIED UI:
  return (
    <div className="space-y-6">
      {/* Basic Input Controls using schema.numeric_columns and schema.categorical_columns */}
      {/* Display prediction.prediction_value */}
      {/* Display feature importance from enhancedPrediction.feature_importance */}
      {/* Show SHAP plot availability from enhancedPrediction.shap_plot_available */}
    </div>
  );
};
```

### **3.3 Key Changes in UI Logic**

1. **Feature Importance**: Get from `enhancedPrediction.feature_importance` array and `enhancedPrediction.feature_importance_scores` object
2. **Input Controls**: Build from basic `schema.numeric_columns` and `schema.categorical_columns`
3. **SHAP Data**: Check `enhancedPrediction.shap_plot_available` and use `/api/results` for SHAP plots
4. **Real-time Updates**: Use basic `/api/predict` for fast updates, enhanced for feature importance

---

## 📝 **Step 4: Update Other Components**

### **4.1 Any Component Using Removed Endpoints**
Search for these patterns and update:

```typescript
// ❌ FIND AND REPLACE:
getEnhancedPredictionSchema → getPredictionSchema + getResults
makeSinglePrediction → makeEnhancedPrediction  
getShapExplanation → getResults (for SHAP plots)
makeBatchPredictions → Multiple makeEnhancedPrediction calls
comparePredictionScenarios → Multiple makeEnhancedPrediction calls + client-side comparison
```

### **4.2 Import Updates**
Update imports in all files:
```typescript
// ❌ REMOVE:
import { getEnhancedPredictionSchema, makeSinglePrediction } from '@/lib/apiClient';

// ✅ REPLACE WITH:
import { getPredictionSchema, makeEnhancedPrediction, getResults } from '@/lib/apiClient';
```

---

## 📝 **Step 5: Testing Checklist**

### **5.1 Verify These Work:**
- [ ] Basic prediction: `POST /api/predict` 
- [ ] Enhanced prediction: `POST /api/predict/enhanced`
- [ ] Basic schema: `GET /api/prediction-schema`
- [ ] Results with feature importance: `GET /api/results`

### **5.2 Verify These Are Removed:**
- [ ] No calls to `/api/predict/single`
- [ ] No calls to `/api/prediction-schema-enhanced`
- [ ] No calls to `/api/predict/explain/*`
- [ ] No calls to `/api/predict/batch`
- [ ] No calls to `/api/predict/compare`

### **5.3 Test Data Flow:**
1. Load basic schema → Build input controls
2. Load results → Get feature importance ranking
3. Make enhanced prediction → Get prediction + feature importance scores
4. Display prediction value and feature importance
5. Show SHAP plot availability

---

## 📝 **Step 6: Example Complete Implementation**

Here's a minimal working example for the new `PredictionInterface`:

```typescript
const PredictionInterface: React.FC<PredictionInterfaceProps> = ({ runId }) => {
  const [schema, setSchema] = useState<PredictionSchemaResponse | null>(null);
  const [results, setResults] = useState<FinalResultsResponse | null>(null);
  const [inputValues, setInputValues] = useState<Record<string, string | number>>({});
  const [enhancedPrediction, setEnhancedPrediction] = useState<EnhancedPredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Load schema and results
  useEffect(() => {
    const loadData = async () => {
      try {
        const [schemaRes, resultsRes] = await Promise.all([
          getPredictionSchema(runId),
          getResults(runId)
        ]);
        setSchema(schemaRes);
        setResults(resultsRes);
        
        // Set defaults
        const defaults: Record<string, string | number> = {};
        Object.entries(schemaRes.numeric_columns).forEach(([col, config]) => {
          defaults[col] = config.mean;
        });
        Object.entries(schemaRes.categorical_columns).forEach(([col, config]) => {
          defaults[col] = config.default;
        });
        setInputValues(defaults);
      } catch (err) {
        console.error('Error loading data:', err);
      }
    };
    
    if (runId) loadData();
  }, [runId]);

  // Make enhanced prediction
  const predict = useCallback(async (values: Record<string, string | number>) => {
    if (!schema) return;
    
    try {
      setIsLoading(true);
      const enhanced = await makeEnhancedPrediction({
        run_id: runId,
        input_values: values
      });
      setEnhancedPrediction(enhanced);
    } catch (err) {
      console.error('Prediction error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [runId, schema]);

  // Auto-predict on input changes
  useEffect(() => {
    if (Object.keys(inputValues).length > 0) {
      predict(inputValues);
    }
  }, [inputValues, predict]);

  if (!schema || !results) {
    return <div>Loading...</div>;
  }

  return (
    <div className="space-y-6">
      {/* Input Controls */}
      <Card>
        <CardHeader>
          <CardTitle>Input Features</CardTitle>
        </CardHeader>
        <CardContent>
          {/* Numeric inputs */}
          {Object.entries(schema.numeric_columns).map(([name, config]) => (
            <div key={name}>
              <Label>{name}</Label>
              <Slider
                value={[Number(inputValues[name]) || config.mean]}
                onValueChange={([value]) => setInputValues(prev => ({ ...prev, [name]: value }))}
                min={config.min}
                max={config.max}
              />
            </div>
          ))}
          
          {/* Categorical inputs */}
          {Object.entries(schema.categorical_columns).map(([name, config]) => (
            <div key={name}>
              <Label>{name}</Label>
              <Select
                value={String(inputValues[name])}
                onValueChange={(value) => setInputValues(prev => ({ ...prev, [name]: value }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {config.options.map(option => (
                    <SelectItem key={option} value={option}>{option}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Prediction Results */}
      <Card>
        <CardHeader>
          <CardTitle>Prediction Results</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div>Calculating...</div>
          ) : enhancedPrediction ? (
            <div>
              <div className="text-2xl font-bold">
                {enhancedPrediction.prediction_value}
              </div>
              
              {/* Feature Importance */}
              <div className="mt-4">
                <h4>Feature Importance</h4>
                {enhancedPrediction.feature_importance.slice(0, 5).map(feature => (
                  <div key={feature} className="flex justify-between">
                    <span>{feature}</span>
                    <span>{(enhancedPrediction.feature_importance_scores[feature] * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              
              {/* SHAP Availability */}
              {enhancedPrediction.shap_plot_available && (
                <Badge>SHAP Plot Available</Badge>
              )}
            </div>
          ) : (
            <div>No prediction yet</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
```

---

## 🎯 **Summary**

The migration simplifies the prediction interface by:

1. **Removing complex endpoints** that had reliability issues
2. **Using basic schema + results** instead of complex enhanced schema  
3. **Combining basic prediction + enhanced prediction** for optimal performance
4. **Getting feature importance from results** instead of real-time SHAP calculations
5. **Maintaining all core functionality** with better reliability

The new approach is more reliable, faster, and easier to maintain while providing the same user experience. 