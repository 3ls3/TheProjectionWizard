# Workflow Recommendations for Optimal Task Implementation

## Recommended Implementation Workflow

When you receive individual testing tasks, follow this workflow to deliver maximum value:

### 1. Context Loading Phase (2-3 minutes)
```bash
# Read all context documents first
cat CONTEXT_REPO_STRUCTURE.md
cat CONTEXT_PIPELINE_ARCHITECTURE.md  
cat CONTEXT_TESTING_FRAMEWORK.md
cat CONTEXT_COMPREHENSIVE_TESTING_PLAN.md
```

**Key Information to Extract:**
- Current working directory and file locations
- Relevant import patterns for the task
- Expected data formats and schemas
- Testing patterns that apply to the task
- Critical failure points related to the task

### 2. Task Analysis Phase (3-5 minutes)

**Questions to Ask Yourself:**
1. **What specific component/system is being tested?**
   - Map task to pipeline step or API endpoint
   - Identify critical files and dependencies
   - Understand data flow requirements

2. **What testing pattern applies?**
   - Atomic testing (single component)
   - Integration testing (multiple components)
   - API testing (endpoint behavior)
   - Data flow testing (transformations)

3. **What are the expected success criteria?**
   - **Regression**: Prediction magnitude (~$425k not ~$150M), 12 features
   - **Classification**: Valid probabilities (0.0-1.0), reasonable class predictions, 8 features
   - Response format compliance for both task types
   - Performance requirements

4. **What are the likely failure modes?**
   - StandardScaler not applied
   - Feature type misclassification
   - Column mapping inconsistencies
   - GCS artifact loading issues

### 3. Environment Setup Phase (2-3 minutes)

**Always Do This First:**
```python
# Activate virtual environment
source .venv/bin/activate

# Set up test environment
import sys
sys.path.append('.')
from pathlib import Path
import time

# Create isolated test environment
test_run_id = f"test_{int(time.time())}"
test_dir = Path("tests/data/test_runs") / test_run_id
test_dir.mkdir(parents=True, exist_ok=True)

# Import testing framework
from tests.unit.stage_tests.base_stage_test import BaseStageTest, TestResult
```

### 4. Implementation Strategy

#### For Phase 1 Tasks (Infrastructure)
```python
# Focus on basic connectivity and initialization
# Test GCS, API, model loading
# Use simple assertions and timing measurements
# Report infrastructure status clearly

def test_infrastructure_component():
    result = TestResult("infrastructure_test")
    start_time = time.time()
    
    try:
        # Test basic functionality
        component_status = test_component_initialization()
        result.add_measurement("init_time", time.time() - start_time)
        
        if component_status:
            result.success("Component initialized successfully")
        else:
            result.error("Component initialization failed")
            
    except Exception as e:
        result.error(f"Infrastructure test failed: {str(e)}")
    
    return result
```

#### For Phase 2 Tasks (Component Isolation)
```python
# Focus on testing single pipeline steps
# Use controlled inputs with known expected outputs
# Validate data transformations and formats
# Check feature count and type consistency

def test_pipeline_step_isolated():
    result = TestResult("step_isolation_test")
    
    # Create controlled test input
    test_input = create_known_test_data()
    
    # Execute step in isolation
    step_output = execute_pipeline_step(test_input)
    
    # Validate output format and content
    result.assert_equals(len(step_output.columns), expected_column_count)
    result.assert_in_range(step_output.shape[0], min_rows, max_rows)
    
    return result
```

#### For Phase 3 Tasks (Integration)
```python
# Focus on data flow between components
# Test consistency of transformations
# Verify end-to-end data integrity
# Check artifact dependencies

def test_integration_flow():
    result = TestResult("integration_test")
    
    # Track data through multiple steps
    data_checkpoints = {}
    
    for step in pipeline_steps:
        step_output = execute_step(step, previous_output)
        data_checkpoints[step] = validate_step_output(step_output)
        
    # Verify consistency across checkpoints
    verify_data_flow_consistency(data_checkpoints)
    
    return result
```

#### For Phase 4 Tasks (API Testing)
```python
# Focus on API endpoint behavior
# Test request/response formats
# Validate endpoint consistency
# Check frontend integration points

def test_api_endpoint():
    result = TestResult("api_test")
    
    # Prepare realistic test payload
    test_payload = create_frontend_like_payload()
    
    # Test endpoint
    response = call_api_endpoint(endpoint_url, test_payload)
    
    # Validate response
    result.assert_equals(response.status_code, 200)
    result.assert_prediction_in_range(response.json()['prediction'], 200000, 800000)
    
    return result
```

#### For Phase 5 Tasks (Frontend Simulation)
```python
# Focus on complete user workflows
# Simulate exact frontend behavior
# Test error scenarios
# Validate performance requirements

def test_frontend_simulation():
    result = TestResult("frontend_simulation")
    
    # Simulate complete user journey
    workflow_steps = [
        'load_schema',
        'submit_input', 
        'receive_prediction',
        'request_explanation'
    ]
    
    for step in workflow_steps:
        step_result = simulate_user_action(step)
        if not step_result.success:
            result.error(f"Workflow failed at step: {step}")
            return result
    
    result.success("Complete user workflow successful")
    return result
```

### 5. Quality Assurance Checklist

**Before Completing Any Task:**
- [ ] **Imports are correct** - All required modules imported
- [ ] **Error handling is comprehensive** - Try/catch with detailed logging
- [ ] **Test isolation is maintained** - No interference between tests
- [ ] **Assertions are meaningful** - Test actual expected behaviors
- [ ] **Cleanup is performed** - Remove test artifacts
- [ ] **Results are structured** - Use TestResult for reporting
- [ ] **Performance is measured** - Include timing data
- [ ] **Edge cases are considered** - Test boundary conditions

### 6. Common Implementation Patterns

#### Data Validation Pattern
```python
def validate_data_format(data, expected_schema):
    """Validate data matches expected format"""
    checks = {
        'column_count': len(data.columns) == expected_schema['column_count'],
        'column_names': set(data.columns) == set(expected_schema['columns']),
        'data_types': all(data[col].dtype == expected_schema['types'][col] 
                         for col in data.columns),
        'value_ranges': all(data[col].between(min_val, max_val).all()
                           for col, (min_val, max_val) in expected_schema['ranges'].items())
    }
    
    return all(checks.values()), checks
```

#### Model Testing Pattern
```python
def test_model_predictions(model_path, test_data):
    """Test model predictions are reasonable"""
    model = load_model(model_path)
    predictions = model.predict(test_data)
    
    # Validate prediction characteristics
    checks = {
        'non_negative': all(p >= 0 for p in predictions),
        'reasonable_range': all(200000 <= p <= 800000 for p in predictions),
        'not_constant': len(set(predictions)) > 1,
        'finite_values': all(np.isfinite(p) for p in predictions)
    }
    
    return checks
```

#### API Testing Pattern
```python
def test_api_consistency(endpoints, test_payload):
    """Test multiple endpoints return consistent results"""
    results = {}
    
    for endpoint in endpoints:
        try:
            response = requests.post(f"{API_BASE_URL}{endpoint}", 
                                   json=test_payload, timeout=30)
            results[endpoint] = {
                'status': response.status_code,
                'prediction': response.json().get('prediction'),
                'error': None
            }
        except Exception as e:
            results[endpoint] = {
                'status': None,
                'prediction': None, 
                'error': str(e)
            }
    
    # Check consistency
    predictions = [r['prediction'] for r in results.values() 
                  if r['prediction'] is not None]
    
    if len(predictions) > 1:
        max_deviation = max(predictions) - min(predictions)
        relative_deviation = max_deviation / np.mean(predictions)
        consistent = relative_deviation < 0.01  # 1% tolerance
    else:
        consistent = False
    
    return results, consistent
```

## Key Success Factors

### 1. **Always Start with Context**
- Read all context documents before coding
- Understand the specific bug being targeted
- Know the expected vs actual behavior

### 2. **Use Systematic Debugging**
- Create controlled test inputs
- Validate outputs at each step
- Compare expected vs actual results
- Document all findings

### 3. **Leverage Existing Infrastructure**
- Use BaseStageTest and TestResult classes
- Follow established import patterns
- Reuse fixture generation utilities
- Maintain consistent error handling

### 4. **Focus on Critical Bugs**
- **Primary**: Prediction magnitude and validity
  - Regression: ~$425k not ~$150M
  - Classification: Valid probabilities (0.0-1.0) and reasonable class decisions
- **Secondary**: Feature count consistency 
  - Regression: 12 features not 31
  - Classification: 8 features correctly encoded
- **Tertiary**: API endpoint parity across both task types

### 5. **Provide Actionable Results**
- Clear success/failure indication
- Detailed error messages with context
- Performance measurements
- Suggestions for fixes

## Anti-Patterns to Avoid

### ❌ Don't Do This:
- Skip reading context documents
- Write tests without understanding the bug
- Use hardcoded values without explanation
- Ignore error handling
- Skip cleanup of test artifacts
- Write tests that interfere with each other
- Focus on symptoms rather than root causes

### ✅ Do This Instead:
- Load context first, then analyze task
- Understand the specific bug being targeted
- Use configurable test parameters
- Include comprehensive error handling
- Clean up all test artifacts
- Ensure test isolation
- Target root causes with precise validation

This workflow ensures consistent, high-quality test implementation that effectively isolates and fixes bugs in the prediction pipeline. 