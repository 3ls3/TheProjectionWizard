"""
Pipeline orchestrator for The Projection Wizard.
Provides a simple interface for running multiple pipeline stages sequentially.
"""

from common import logger


def run_from_schema_confirm(run_id: str) -> bool:
    """
    Run steps 3-7 synchronously after feature schemas are confirmed.
    Returns True on full success, False otherwise.

    Args:
        run_id: The ID of the run to process

    Returns:
        True if all stages completed successfully, False if any stage failed
    """
    # Get logger for orchestration
    orchestrator_logger = logger.get_logger(run_id, "api_orchestrator")

    orchestrator_logger.info("Starting pipeline execution: stages 3-7")

    try:
        # Import stage runners
        from pipeline.step_3_validation import validation_runner
        from pipeline.step_4_prep import prep_runner
        from pipeline.step_5_automl import automl_runner
        from pipeline.step_6_explain import explain_runner

        # Run each stage in sequence; bail on first failure

        # Stage 3: Validation
        orchestrator_logger.info("Starting stage 3: Validation")
        if not validation_runner.run_validation_stage(run_id):
            orchestrator_logger.error("Stage 3 (validation) failed")
            return False
        orchestrator_logger.info("Stage 3 (validation) completed successfully")

        # Stage 4: Data Preparation
        orchestrator_logger.info("Starting stage 4: Data Preparation")
        if not prep_runner.run_preparation_stage(run_id):
            orchestrator_logger.error("Stage 4 (preparation) failed")
            return False
        orchestrator_logger.info(
            "Stage 4 (preparation) completed successfully"
        )

        # Stage 5: AutoML
        orchestrator_logger.info("Starting stage 5: AutoML")
        if not automl_runner.run_automl_stage(run_id):
            orchestrator_logger.error("Stage 5 (automl) failed")
            return False
        orchestrator_logger.info("Stage 5 (automl) completed successfully")

        # Stage 6: Explainability
        orchestrator_logger.info("Starting stage 6: Explainability")
        if not explain_runner.run_explainability_stage(run_id):
            orchestrator_logger.error("Stage 6 (explain) failed")
            return False
        orchestrator_logger.info("Stage 6 (explain) completed successfully")

        orchestrator_logger.info("All pipeline stages completed successfully")
        return True

    except Exception as e:
        orchestrator_logger.error(
            f"Unexpected error in pipeline orchestrator: {str(e)}"
        )
        return False
