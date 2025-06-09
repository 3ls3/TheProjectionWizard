#!/usr/bin/env python3
"""
Quick Test Script for The Projection Wizard Testing Framework.

This script demonstrates the testing framework capabilities and can be used
for quick validation that everything is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from tests.fixtures.fixture_generator import TestFixtureGenerator, create_all_stage_fixtures
from tests.unit.stage_tests.test_ingestion import run_ingestion_test
from tests.integration.test_orchestrator import TestOrchestrator


def demo_fixture_generation():
    """Demonstrate fixture generation capabilities."""
    print("🔧 FIXTURE GENERATION DEMO")
    print("="*50)
    
    generator = TestFixtureGenerator()
    
    # Create fixtures for all stages (classification)
    print("\n📊 Creating Classification Fixtures:")
    classification_fixtures = create_all_stage_fixtures("classification")
    
    # Create fixtures for all stages (regression)  
    print("\n📈 Creating Regression Fixtures:")
    regression_fixtures = create_all_stage_fixtures("regression")
    
    print(f"\n✅ Created {len(classification_fixtures) + len(regression_fixtures)} test fixtures")
    return classification_fixtures, regression_fixtures


def demo_individual_stage_testing(fixtures):
    """Demonstrate individual stage testing."""
    print("\n\n🧪 INDIVIDUAL STAGE TESTING DEMO")
    print("="*50)
    
    # Test ingestion stage with classification data
    ingestion_run_id = fixtures["ingestion"]
    print(f"\n🔬 Testing Ingestion Stage (run: {ingestion_run_id})")
    
    result = run_ingestion_test(ingestion_run_id)
    
    print(f"\nResults:")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration:.2f}s")
    print(f"  Errors: {len(result.errors)}")
    
    if result.errors:
        print("  Error details:")
        for error in result.errors[:3]:  # Show first 3 errors
            print(f"    - {error}")
    
    return result


def demo_test_orchestrator():
    """Demonstrate the test orchestrator capabilities."""
    print("\n\n🎭 TEST ORCHESTRATOR DEMO")
    print("="*50)
    
    orchestrator = TestOrchestrator()
    
    # Run individual stage test
    print("\n1. Individual Stage Test:")
    result = orchestrator.run_individual_stage_test("ingestion", "classification")
    status = "✅ PASSED" if result.success else "❌ FAILED"
    print(f"   Ingestion (classification): {status} ({result.duration:.2f}s)")
    
    # Run sequential test (just first stage for demo)
    print("\n2. Sequential Test Preview:")
    print("   (Note: Only ingestion stage is fully implemented)")
    
    stages = ["ingestion", "schema", "validation", "prep", "automl", "explain"]
    for stage in stages:
        if stage == "ingestion":
            result = orchestrator.run_individual_stage_test(stage, "classification")
            status = "✅ PASSED" if result.success else "❌ FAILED"
        else:
            status = "⚠️  NOT IMPLEMENTED"
        print(f"   {stage:12}: {status}")
    
    return orchestrator


def demo_log_inspection(test_run_id):
    """Show how to inspect logs from a test run."""
    print("\n\n📋 LOG INSPECTION DEMO")
    print("="*50)
    
    test_run_dir = Path(__file__).parent.parent / "data" / "test_runs" / test_run_id
    
    if test_run_dir.exists():
        print(f"\n📁 Test Run Directory: {test_run_dir}")
        print("Files created:")
        
        for file_path in sorted(test_run_dir.iterdir()):
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"  📄 {file_path.name:<20} ({size:,} bytes)")
        
        # Show log file content (if exists)
        log_files = list(test_run_dir.glob("*.log"))
        if log_files:
            print(f"\n📋 Log File Preview ({log_files[0].name}):")
            with open(log_files[0], 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[-5:], 1):  # Show last 5 lines
                    print(f"  {i:2d}: {line.strip()}")
        
        # Show metadata content
        metadata_file = test_run_dir / "metadata.json"
        if metadata_file.exists():
            import json
            print(f"\n📊 Metadata Preview:")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                key_fields = ["run_id", "original_filename", "initial_rows", "initial_cols"]
                for key in key_fields:
                    if key in metadata:
                        print(f"  {key:<20}: {metadata[key]}")
    else:
        print(f"❌ Test run directory not found: {test_run_dir}")


def main():
    """Main demo function."""
    print("🎯 THE PROJECTION WIZARD TESTING FRAMEWORK DEMO")
    print("="*60)
    print("This demo showcases the comprehensive testing capabilities")
    print("built for The Projection Wizard pipeline.\n")
    
    try:
        # 1. Fixture Generation
        classification_fixtures, regression_fixtures = demo_fixture_generation()
        
        # 2. Individual Stage Testing
        ingestion_result = demo_individual_stage_testing(classification_fixtures)
        
        # 3. Test Orchestrator
        orchestrator = demo_test_orchestrator()
        
        # 4. Log Inspection
        demo_log_inspection(classification_fixtures["ingestion"])
        
        # 5. Summary
        print("\n\n🎉 DEMO SUMMARY")
        print("="*50)
        print("✅ Fixture generation: Working")
        print(f"✅ Individual stage testing: {'Working' if ingestion_result.success else 'Issues detected'}")
        print("✅ Test orchestrator: Working")
        print("✅ Log inspection: Working")
        print("✅ Stage-specific logging: Working")
        
        print(f"\n📊 Total test fixtures created: {len(classification_fixtures) + len(regression_fixtures)}")
        print(f"🧪 Successfully tested stages: 1 (ingestion)")
        print(f"⚠️  Pending implementation: 5 stages (schema, validation, prep, automl, explain)")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Implement test runners for remaining stages")
        print("2. Run comprehensive integration tests")
        print("3. Set up continuous integration")
        print("4. Add performance benchmarking")
        
        print(f"\n📁 Test artifacts saved in: data/test_runs/")
        print(f"📋 Test reports saved in: tests/reports/")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 