# The Projection Wizard Testing Framework

A comprehensive, methodical testing framework for The Projection Wizard pipeline that enables step-by-step testing to surface errors and bugs at each stage.

## 🎯 Overview

This testing framework provides:

- **🔧 Fixture Generation**: Automated creation of controlled test environments for each pipeline stage
- **🧪 Stage-Specific Testing**: Isolated testing of individual pipeline components
- **🔬 Integration Testing**: Full pipeline testing with data flowing through stages
- **📊 Regression Testing**: Testing with multiple datasets (classification & regression)
- **📋 Detailed Logging**: Stage-specific logs for granular debugging
- **📄 Comprehensive Reporting**: JSON reports with detailed test results

## 🏗️ Architecture

```
tests/
├── fixtures/                    # Test fixture generation
│   ├── __init__.py
│   └── fixture_generator.py     # Main fixture creation utilities
├── stage_tests/                 # Individual stage test runners
│   ├── __init__.py
│   ├── base_stage_test.py       # Base class for all stage tests
│   ├── test_ingestion.py        # Ingestion stage test runner
│   └── test_*.py               # Additional stage test runners (to be implemented)
├── reports/                     # Test reports output directory
├── test_orchestrator.py         # Main test orchestration and management
├── quick_test.py               # Quick demo/validation script
└── README.md                   # This documentation
```

## 🚀 Quick Start

### 1. Run the Demo

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the quick demo to see everything working
cd tests
python quick_test.py
```

### 2. Generate Test Fixtures

```python
from tests.fixtures.fixture_generator import TestFixtureGenerator, create_all_stage_fixtures

# Create fixtures for all stages
classification_fixtures = create_all_stage_fixtures("classification")
regression_fixtures = create_all_stage_fixtures("regression")
```

### 3. Run Individual Stage Tests

```python
from tests.stage_tests.test_ingestion import run_ingestion_test

# Test ingestion stage
result = run_ingestion_test(test_run_id)
print(f"Success: {result.success}, Duration: {result.duration:.2f}s")
```

### 4. Use the Test Orchestrator

```python
from tests.test_orchestrator import TestOrchestrator

orchestrator = TestOrchestrator()

# Run individual stage test
result = orchestrator.run_individual_stage_test("ingestion", "classification")

# Run all stages in sequence
results = orchestrator.run_sequential_pipeline_test("classification")

# Run full integration test
integration_result = orchestrator.run_full_integration_test("classification")
```

## 📋 Command Line Interface

The test orchestrator includes a comprehensive CLI for various testing modes:

```bash
# Run all test modes
python test_orchestrator.py --mode all

# Run individual stage test
python test_orchestrator.py --mode individual --stage ingestion --task-type classification

# Run sequential testing
python test_orchestrator.py --mode sequential --task-type regression

# Run integration testing
python test_orchestrator.py --mode integration --task-type classification

# Run regression testing (both task types)
python test_orchestrator.py --mode regression

# Clean up old test runs
python test_orchestrator.py --cleanup
```

## 🔧 Test Fixtures

### Fixture Types

The framework creates controlled test environments for each pipeline stage:

| Stage | Required Files | Purpose |
|-------|---------------|---------|
| **Ingestion** | `original_data.csv` | Test CSV file upload and metadata creation |
| **Schema** | `original_data.csv`, `metadata.json` | Test target/feature schema definition |
| **Validation** | `original_data.csv`, `metadata.json` (with schemas) | Test Great Expectations validation |
| **Prep** | Same as Validation + `validation.json` | Test data cleaning and encoding |
| **AutoML** | `cleaned_data.csv`, full metadata | Test PyCaret model training |
| **Explain** | Model artifacts, `cleaned_data.csv` | Test SHAP explanations |

### Sample Datasets

The framework uses consistent sample datasets:

- **Classification**: `sample_classification.csv` (10 rows, 4 columns, binary target)
- **Regression**: `sample_regression.csv` (10 rows, 4 columns, continuous target)

### Fixture Generation

```python
generator = TestFixtureGenerator()

# Create stage-specific fixtures
ingestion_run_id = generator.setup_stage_1_ingestion("classification")
schema_run_id = generator.setup_stage_2_schema("classification")
validation_run_id = generator.setup_stage_3_validation("classification")
prep_run_id = generator.setup_stage_4_prep("classification")
automl_run_id = generator.setup_stage_5_automl("classification")
explain_run_id = generator.setup_stage_6_explain("classification")
```

## 🧪 Stage Testing

### Base Test Class

All stage tests inherit from `BaseStageTest` which provides:

- Input file validation
- Output file validation
- Status file checking
- Metadata validation
- Error handling and logging
- Test reporting

### Example: Ingestion Stage Test

```python
from tests.stage_tests.test_ingestion import IngestionStageTest

# Create and run test
test = IngestionStageTest(test_run_id)
success, results = test.run_test()

# The test validates:
# 1. Input file exists (original_data.csv)
# 2. Ingestion logic executes successfully
# 3. Output files created (metadata.json, status.json)
# 4. Status shows completed
# 5. Metadata has required fields
# 6. Data types and counts are correct
```

### Test Validation Layers

Each stage test performs multiple validation layers:

1. **Input Validation**: Required files exist and are readable
2. **Execution Validation**: Stage function runs without errors
3. **Output Validation**: Expected files are created
4. **Content Validation**: File contents meet specifications
5. **Stage-Specific Validation**: Custom checks for each stage
6. **Integration Validation**: Compatibility with next stage

## 📊 Logging & Debugging

### Stage-Specific Logs

Each test creates separate log files in the test run directory:

```
data/test_runs/{test_run_id}/
├── original_data.csv
├── metadata.json
├── status.json
├── test_ingestion.log      # Test-specific log
├── ingestion.log          # Stage execution log
└── ...
```

### Log Format

```
2025-06-06 11:31:42 | test_run_id | Stage | Logger | Level | Message
```

### Debugging Failed Tests

1. **Check test logs**: `data/test_runs/{test_run_id}/test_{stage}.log`
2. **Check stage logs**: `data/test_runs/{test_run_id}/{stage}.log`
3. **Inspect test reports**: `tests/reports/`
4. **Validate fixtures**: Ensure test data is properly set up

## 📄 Test Reports

### Report Types

- **Individual Test Reports**: Single stage test results
- **Sequential Test Reports**: All stages tested independently
- **Integration Test Reports**: Full pipeline flow results
- **Regression Test Reports**: Multiple task types results

### Report Format

```json
{
  "test_type": "individual_test",
  "timestamp": "2025-06-06T11:31:42.123456",
  "result": {
    "stage_name": "ingestion",
    "success": true,
    "duration": 2.45,
    "details": {...},
    "errors": []
  }
}
```

## 🔄 Reusability & Extensibility

### Adding New Stage Tests

1. Create new test file: `tests/stage_tests/test_new_stage.py`
2. Inherit from `BaseStageTest`
3. Implement stage-specific validation logic
4. Add to test orchestrator
5. Update fixture generator if needed

### Example New Stage Test

```python
from tests.stage_tests.base_stage_test import BaseStageTest

class NewStageTest(BaseStageTest):
    def __init__(self, test_run_id: str):
        super().__init__("new_stage", test_run_id)
    
    def run_test(self) -> Tuple[bool, Dict[str, Any]]:
        # Implement stage-specific testing logic
        pass
```

### Modifying for Code Changes

The framework is designed to be robust against pipeline code changes:

1. **Fixture Generator**: Update metadata/file structures as needed
2. **Base Test Class**: Common validation logic remains stable
3. **Stage Tests**: Update stage-specific validations
4. **Orchestrator**: Add/remove stages as pipeline evolves

## 🎯 Best Practices

### Test Development

1. **Start with Fixtures**: Always create proper test fixtures first
2. **Validate Inputs**: Check all required files exist before testing
3. **Test in Isolation**: Each stage should be testable independently
4. **Comprehensive Validation**: Check files, content, and side effects
5. **Clear Error Messages**: Provide actionable error information

### Debugging

1. **Use Stage-Specific Logs**: Check the right log file for your issue
2. **Validate Fixtures**: Ensure test environment is set up correctly
3. **Run Individual Tests**: Isolate problems to specific stages
4. **Check Integration**: Verify stage outputs work with next stage

### Maintenance

1. **Regular Cleanup**: Remove old test runs to save space
2. **Update Fixtures**: Keep test data relevant to pipeline changes
3. **Monitor Reports**: Review test reports for patterns
4. **Version Control**: Track test results over time

## 📈 Performance & Scaling

### Test Run Management

- **Automatic Cleanup**: Keeps only recent test runs per stage/type
- **Parallel Testing**: Framework supports concurrent test execution
- **Resource Monitoring**: Track test duration and resource usage

### Scaling Considerations

- Test fixtures are lightweight (small sample datasets)
- Log files are stage-specific to reduce size
- Reports are JSON for easy parsing and analysis
- Framework supports both quick validation and comprehensive testing

## 🔮 Future Enhancements

### Planned Features

1. **Performance Benchmarking**: Track test execution times over code changes
2. **Continuous Integration**: GitHub Actions integration
3. **Visual Test Reports**: HTML reports with charts and graphs
4. **Test Data Management**: Larger, more realistic test datasets
5. **Mock External Services**: Test with simulated external dependencies

### Extension Points

- **Custom Validators**: Add domain-specific validation logic
- **Additional Datasets**: Support for more complex test scenarios
- **Test Scheduling**: Automated testing on code changes
- **Integration with Monitoring**: Real-time test result tracking

---

## 🎉 Getting Started

Ready to start testing? Run the quick demo:

```bash
cd tests && python quick_test.py
```

This will demonstrate all framework capabilities and create example test runs you can inspect and learn from.

Happy testing! 🧪✨ 