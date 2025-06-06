"""
Test Orchestrator for The Projection Wizard.

This module provides a comprehensive testing framework that can run:
1. Individual stage tests in isolation
2. Sequential pipeline testing (stage by stage)
3. Full integration tests
4. Regression testing with multiple datasets
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

from common import constants, logger
from tests.fixtures.fixture_generator import TestFixtureGenerator, create_all_stage_fixtures
from tests.stage_tests.base_stage_test import TestResult


class TestOrchestrator:
    """
    Orchestrates comprehensive testing of The Projection Wizard pipeline.
    
    Provides multiple testing modes:
    - Individual stage testing
    - Sequential pipeline testing  
    - Full integration testing
    - Regression testing
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the test orchestrator.
        
        Args:
            output_dir: Directory for test reports (default: tests/reports)
        """
        self.base_dir = Path(__file__).parent.parent
        self.output_dir = output_dir or (Path(__file__).parent / "reports")
        self.output_dir.mkdir(exist_ok=True)
        
        self.fixture_generator = TestFixtureGenerator()
        self.test_results: List[TestResult] = []
        
        # Create orchestrator logger
        self.logger = logger.get_logger("test_orchestrator", "orchestrator")
    
    def run_individual_stage_test(self, stage_name: str, task_type: str = "classification") -> TestResult:
        """
        Run a test for an individual pipeline stage.
        
        Args:
            stage_name: Name of the stage to test (e.g., "ingestion", "schema", etc.)
            task_type: Type of ML task ("classification" or "regression")
            
        Returns:
            TestResult object with test outcomes
        """
        self.logger.info(f"🔬 Running individual stage test: {stage_name} ({task_type})")
        
        try:
            # Create fixture for the stage
            if stage_name == "ingestion":
                test_run_id = self.fixture_generator.setup_stage_1_ingestion(task_type)
                from tests.stage_tests.test_ingestion import run_ingestion_test
                result = run_ingestion_test(test_run_id)
            
            elif stage_name == "schema":
                test_run_id = self.fixture_generator.setup_stage_2_schema(task_type)
                # TODO: Import and run schema test when created
                result = TestResult(stage_name, False, 0.0, {"error": "Schema test not implemented yet"}, ["Not implemented"])
            
            elif stage_name == "validation":
                test_run_id = self.fixture_generator.setup_stage_3_validation(task_type)
                # TODO: Import and run validation test when created
                result = TestResult(stage_name, False, 0.0, {"error": "Validation test not implemented yet"}, ["Not implemented"])
            
            elif stage_name == "prep":
                test_run_id = self.fixture_generator.setup_stage_4_prep(task_type)
                # TODO: Import and run prep test when created
                result = TestResult(stage_name, False, 0.0, {"error": "Prep test not implemented yet"}, ["Not implemented"])
            
            elif stage_name == "automl":
                test_run_id = self.fixture_generator.setup_stage_5_automl(task_type)
                # TODO: Import and run automl test when created
                result = TestResult(stage_name, False, 0.0, {"error": "AutoML test not implemented yet"}, ["Not implemented"])
            
            elif stage_name == "explain":
                test_run_id = self.fixture_generator.setup_stage_6_explain(task_type)
                # TODO: Import and run explain test when created
                result = TestResult(stage_name, False, 0.0, {"error": "Explain test not implemented yet"}, ["Not implemented"])
            
            else:
                raise ValueError(f"Unknown stage name: {stage_name}")
            
            # Add to results
            self.test_results.append(result)
            
            # Log result
            status = "✅ PASSED" if result.success else "❌ FAILED"
            self.logger.info(f"{status} - {stage_name} test completed in {result.duration:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Individual stage test failed: {e}")
            result = TestResult(stage_name, False, 0.0, {"error": str(e)}, [str(e)])
            self.test_results.append(result)
            return result
    
    def run_sequential_pipeline_test(self, task_type: str = "classification") -> Dict[str, TestResult]:
        """
        Run tests for all pipeline stages in sequence.
        
        This tests each stage independently with pre-created fixtures,
        not as a flowing pipeline.
        
        Args:
            task_type: Type of ML task ("classification" or "regression")
            
        Returns:
            Dictionary mapping stage names to TestResult objects
        """
        self.logger.info(f"🧪 Running sequential pipeline test ({task_type})")
        
        stages = ["ingestion", "schema", "validation", "prep", "automl", "explain"]
        results = {}
        
        start_time = time.time()
        
        for stage in stages:
            self.logger.info(f"Testing stage: {stage}")
            result = self.run_individual_stage_test(stage, task_type)
            results[stage] = result
            
            # Log intermediate result
            if result.success:
                self.logger.info(f"✅ {stage} test passed")
            else:
                self.logger.error(f"❌ {stage} test failed: {result.errors}")
        
        duration = time.time() - start_time
        
        # Summary
        passed_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)
        
        self.logger.info(f"📊 Sequential test summary: {passed_count}/{total_count} stages passed in {duration:.2f}s")
        
        return results
    
    def run_full_integration_test(self, task_type: str = "classification") -> TestResult:
        """
        Run a full integration test of the entire pipeline.
        
        This creates a single test run and executes all stages in sequence,
        allowing data to flow through the actual pipeline.
        
        Args:
            task_type: Type of ML task ("classification" or "regression")
            
        Returns:
            TestResult for the full integration test
        """
        self.logger.info(f"🚀 Running full integration test ({task_type})")
        
        start_time = time.time()
        
        try:
            # Create initial fixture (just the CSV file)
            test_run_id = self.fixture_generator.setup_stage_1_ingestion(task_type)
            self.logger.info(f"Created integration test run: {test_run_id}")
            
            # Run each stage in sequence, allowing data to flow through
            stages_to_run = [
                ("ingestion", "step_1_ingest.ingest_logic", "run_ingestion"),
                ("schema", "step_2_schema", None),  # TODO: Add when available
                ("validation", "step_3_validation.validation_runner", "run_validation_stage"),
                ("prep", "step_4_prep.prep_runner", "run_preparation_stage"),
                ("automl", "step_5_automl.automl_runner", "run_automl_stage"),
                ("explain", "step_6_explain.explain_runner", "run_explainability_stage")
            ]
            
            stage_results = {}
            overall_success = True
            
            for stage_name, module_path, function_name in stages_to_run:
                if function_name is None:
                    # Skip stages not yet implemented
                    stage_results[stage_name] = {"success": False, "error": "Not implemented", "skipped": True}
                    continue
                
                self.logger.info(f"Running integration stage: {stage_name}")
                
                try:
                    # Import and run the stage function
                    if stage_name == "ingestion":
                        # Special handling for ingestion
                        from tests.stage_tests.test_ingestion import MockUploadedFile
                        from step_1_ingest.ingest_logic import run_ingestion
                        
                        csv_path = Path(__file__).parent.parent / "data" / "test_runs" / test_run_id / constants.ORIGINAL_DATA_FILENAME
                        mock_file = MockUploadedFile(csv_path)
                        actual_run_id = run_ingestion(mock_file, str(csv_path.parent.parent))
                        
                        # Move the generated run to our test location
                        import shutil
                        actual_run_dir = csv_path.parent.parent / actual_run_id
                        test_run_dir = csv_path.parent
                        
                        if actual_run_dir.exists():
                            for item in actual_run_dir.iterdir():
                                if item.is_file():
                                    shutil.copy2(item, test_run_dir / item.name)
                            shutil.rmtree(actual_run_dir)
                        
                        stage_success = True
                        
                    else:
                        # Import the module dynamically
                        module = __import__(module_path, fromlist=[function_name])
                        stage_function = getattr(module, function_name)
                        
                        # Run the stage function
                        stage_success = stage_function(test_run_id)
                    
                    stage_results[stage_name] = {"success": stage_success}
                    
                    if not stage_success:
                        overall_success = False
                        self.logger.error(f"Integration stage {stage_name} failed")
                        break
                    else:
                        self.logger.info(f"✅ Integration stage {stage_name} passed")
                
                except Exception as e:
                    stage_results[stage_name] = {"success": False, "error": str(e)}
                    overall_success = False
                    self.logger.error(f"Integration stage {stage_name} error: {e}")
                    break
            
            duration = time.time() - start_time
            
            # Create result
            result = TestResult(
                stage_name=f"full_integration_{task_type}",
                success=overall_success,
                duration=duration,
                details={
                    "test_run_id": test_run_id,
                    "stage_results": stage_results,
                    "task_type": task_type
                },
                errors=[f"{k}: {v.get('error', 'Failed')}" for k, v in stage_results.items() 
                       if not v.get("success", True) and not v.get("skipped", False)]
            )
            
            self.test_results.append(result)
            
            status = "✅ PASSED" if overall_success else "❌ FAILED"
            self.logger.info(f"{status} - Full integration test completed in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Full integration test failed: {e}")
            result = TestResult(
                stage_name=f"full_integration_{task_type}",
                success=False,
                duration=duration,
                details={"error": str(e)},
                errors=[str(e)]
            )
            self.test_results.append(result)
            return result
    
    def run_regression_test(self) -> Dict[str, Dict[str, TestResult]]:
        """
        Run regression tests with both classification and regression datasets.
        
        Returns:
            Dictionary with results for both task types
        """
        self.logger.info("🔄 Running regression tests (both task types)")
        
        results = {}
        
        for task_type in ["classification", "regression"]:
            self.logger.info(f"Running regression test for {task_type}")
            results[task_type] = self.run_sequential_pipeline_test(task_type)
        
        # Summary
        all_passed = all(
            result.success 
            for task_results in results.values() 
            for result in task_results.values()
        )
        
        status = "✅ ALL PASSED" if all_passed else "❌ SOME FAILED"
        self.logger.info(f"{status} - Regression tests completed")
        
        return results
    
    def create_test_report(self, test_name: str, results: Any) -> str:
        """
        Create a detailed test report and save it to file.
        
        Args:
            test_name: Name of the test run
            results: Test results data
            
        Returns:
            Path to the created report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"{test_name}_{timestamp}.json"
        
        # Prepare report data
        if isinstance(results, dict) and "classification" in results:
            # Regression test results
            report_data = {
                "test_type": "regression_test",
                "timestamp": datetime.now().isoformat(),
                "results": {
                    task_type: {
                        stage: result.to_dict() 
                        for stage, result in task_results.items()
                    }
                    for task_type, task_results in results.items()
                }
            }
        elif isinstance(results, dict):
            # Sequential test results
            report_data = {
                "test_type": "sequential_test",
                "timestamp": datetime.now().isoformat(),
                "results": {
                    stage: result.to_dict() 
                    for stage, result in results.items()
                }
            }
        elif isinstance(results, TestResult):
            # Single test result
            report_data = {
                "test_type": "single_test",
                "timestamp": datetime.now().isoformat(),
                "result": results.to_dict()
            }
        else:
            # Unknown format
            report_data = {
                "test_type": "unknown",
                "timestamp": datetime.now().isoformat(),
                "results": str(results)
            }
        
        # Write report
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"📄 Test report saved: {report_file}")
        return str(report_file)
    
    def cleanup_old_test_runs(self) -> None:
        """Clean up old test runs to save space."""
        self.fixture_generator.cleanup_test_runs()


def main():
    """Main function for running tests from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Orchestrator for The Projection Wizard")
    parser.add_argument("--mode", choices=["individual", "sequential", "integration", "regression", "all"], 
                       default="all", help="Test mode to run")
    parser.add_argument("--stage", help="Stage name for individual testing")
    parser.add_argument("--task-type", choices=["classification", "regression"], 
                       default="classification", help="Task type for testing")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old test runs")
    
    args = parser.parse_args()
    
    orchestrator = TestOrchestrator()
    
    if args.cleanup:
        orchestrator.cleanup_old_test_runs()
    
    if args.mode == "individual":
        if not args.stage:
            print("Error: --stage is required for individual mode")
            return
        
        result = orchestrator.run_individual_stage_test(args.stage, args.task_type)
        report_file = orchestrator.create_test_report(f"individual_{args.stage}_{args.task_type}", result)
        
        print(f"\n{'='*60}")
        print(f"INDIVIDUAL TEST RESULTS ({args.stage})")
        print(f"{'='*60}")
        print(f"Success: {result.success}")
        print(f"Duration: {result.duration:.2f}s")
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
        print(f"Report: {report_file}")
    
    elif args.mode == "sequential":
        results = orchestrator.run_sequential_pipeline_test(args.task_type)
        report_file = orchestrator.create_test_report(f"sequential_{args.task_type}", results)
        
        print(f"\n{'='*60}")
        print(f"SEQUENTIAL TEST RESULTS ({args.task_type})")
        print(f"{'='*60}")
        for stage, result in results.items():
            status = "✅ PASSED" if result.success else "❌ FAILED"
            print(f"{stage:12} - {status} ({result.duration:.2f}s)")
        print(f"Report: {report_file}")
    
    elif args.mode == "integration":
        result = orchestrator.run_full_integration_test(args.task_type)
        report_file = orchestrator.create_test_report(f"integration_{args.task_type}", result)
        
        print(f"\n{'='*60}")
        print(f"INTEGRATION TEST RESULTS ({args.task_type})")
        print(f"{'='*60}")
        print(f"Success: {result.success}")
        print(f"Duration: {result.duration:.2f}s")
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
        print(f"Report: {report_file}")
    
    elif args.mode == "regression":
        results = orchestrator.run_regression_test()
        report_file = orchestrator.create_test_report("regression", results)
        
        print(f"\n{'='*60}")
        print("REGRESSION TEST RESULTS")
        print(f"{'='*60}")
        for task_type, task_results in results.items():
            print(f"\n{task_type.upper()}:")
            for stage, result in task_results.items():
                status = "✅ PASSED" if result.success else "❌ FAILED"
                print(f"  {stage:12} - {status} ({result.duration:.2f}s)")
        print(f"\nReport: {report_file}")
    
    elif args.mode == "all":
        print("🧪 Running all test modes...")
        
        # Run all modes
        print("\n1. Individual tests...")
        for stage in ["ingestion"]:  # Only implemented stages
            for task_type in ["classification", "regression"]:
                result = orchestrator.run_individual_stage_test(stage, task_type)
                status = "✅" if result.success else "❌"
                print(f"   {stage} ({task_type}): {status}")
        
        print("\n2. Sequential tests...")
        for task_type in ["classification", "regression"]:
            results = orchestrator.run_sequential_pipeline_test(task_type)
            passed = sum(1 for r in results.values() if r.success)
            total = len(results)
            print(f"   {task_type}: {passed}/{total} stages passed")
        
        print("\n3. Integration tests...")
        for task_type in ["classification", "regression"]:
            result = orchestrator.run_full_integration_test(task_type)
            status = "✅" if result.success else "❌"
            print(f"   {task_type}: {status}")
        
        # Create summary report
        report_file = orchestrator.create_test_report("complete_test_suite", orchestrator.test_results)
        print(f"\n📄 Complete test suite report: {report_file}")


if __name__ == "__main__":
    main() 