"""
Prediction Runner for The Projection Wizard.
Orchestrates prediction operations including model loading, data processing, and result generation.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime

from common import logger, storage, constants, schemas
from . import predict_logic, plot_utils


def run_prediction_stage(
    run_id: str,
    input_data: Union[pd.DataFrame, str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    generate_plots: bool = True
) -> bool:
    """
    Execute the complete prediction stage for a given run with input data.

    This function orchestrates:
    1. Loading the trained model and metadata
    2. Processing input data and generating predictions
    3. Creating visualizations (if enabled)
    4. Saving results to files

    Args:
        run_id: Unique run identifier
        input_data: Input data for predictions - can be:
            - pd.DataFrame: Direct DataFrame input
            - str/Path: Path to CSV file to load
        output_dir: Directory to save results (defaults to run directory)
        generate_plots: Whether to generate visualization plots

    Returns:
        True if prediction completes successfully, False otherwise
    """
    # Get loggers for this run
    log = logger.get_stage_logger(run_id, "PREDICT")

    try:
        log.info(f"Starting prediction stage for run {run_id}")

        # =============================
        # 1. LOAD METADATA AND MODEL
        # =============================
        log.info("Loading metadata and trained model...")

        # Load metadata
        metadata = storage.read_json(run_id, constants.METADATA_FILENAME)
        if not metadata:
            raise Exception("Could not load metadata")

        target_info = schemas.TargetInfo(**metadata['target_info'])
        log.info(f"Target: {target_info.name}, Task: {target_info.task_type}")

        # Load trained model
        run_dir = storage.get_run_dir(run_id)
        model = predict_logic.load_pipeline(run_dir)
        log.info("Model loaded successfully")

        # =============================
        # 2. LOAD AND VALIDATE INPUT DATA
        # =============================
        log.info("Processing input data...")

        if isinstance(input_data, (str, Path)):
            # Load from file
            input_df = pd.read_csv(input_data)
            log.info(f"Loaded input data from file: {input_data}")
        elif isinstance(input_data, pd.DataFrame):
            # Use DataFrame directly
            input_df = input_data.copy()
            log.info("Using provided DataFrame input")
        else:
            raise ValueError(f"Unsupported input_data type: {type(input_data)}")

        log.info(f"Input data shape: {input_df.shape}")

        # Validate inputs
        is_valid, issues = predict_logic.validate_prediction_inputs(model, input_df)
        if not is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(issues)}")

        # =============================
        # 3. GENERATE PREDICTIONS
        # =============================
        log.info("Generating predictions...")

        result_df = predict_logic.generate_predictions(model, input_df)
        log.info(f"Generated {len(result_df)} predictions")

        # Get prediction summary
        summary = predict_logic.get_prediction_summary(result_df, target_info.task_type)
        log.info(f"Prediction summary: {summary}")

        # =============================
        # 4. SAVE RESULTS
        # =============================
        log.info("Saving prediction results...")

        # Determine output directory
        if output_dir is None:
            output_dir = run_dir / "predictions"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f"predictions_{timestamp}.csv"
        result_df.to_csv(results_file, index=False)
        log.info(f"Results saved to: {results_file}")

        # Save summary JSON
        summary_file = output_dir / f"prediction_summary_{timestamp}.json"
        storage.write_json_atomic(str(summary_file.parent), summary_file.name, summary)
        log.info(f"Summary saved to: {summary_file}")

        # =============================
        # 5. GENERATE PLOTS (IF ENABLED)
        # =============================
        plot_files = []

        if generate_plots:
            log.info("Generating visualization plots...")

            try:
                if target_info.task_type.lower() == "classification":
                    # Create prediction distribution plot
                    plot_file = output_dir / f"prediction_distribution_{timestamp}.png"
                    plot_utils.create_prediction_summary_plot(
                        result_df, target_info.task_type, plot_file,
                        title=f"Prediction Distribution - {target_info.name}"
                    )
                    plot_files.append(plot_file)

                elif target_info.task_type.lower() == "regression":
                    # Create prediction distribution plot
                    plot_file = output_dir / f"prediction_distribution_{timestamp}.png"
                    plot_utils.create_prediction_summary_plot(
                        result_df, target_info.task_type, plot_file,
                        title=f"Prediction Distribution - {target_info.name}"
                    )
                    plot_files.append(plot_file)

                log.info(f"Generated {len(plot_files)} visualization plots")

            except Exception as e:
                log.warning(f"Failed to generate plots: {e}")

        # =============================
        # 6. LOG COMPLETION
        # =============================
        log.info("="*50)
        log.info("PREDICTION STAGE COMPLETED SUCCESSFULLY")
        log.info("="*50)
        log.info(f"Input shape: {input_df.shape}")
        log.info(f"Predictions generated: {len(result_df)}")
        log.info(f"Task type: {target_info.task_type}")
        log.info(f"Target column: {target_info.name}")
        log.info(f"Output files:")
        log.info(f"  - {results_file.name}")
        log.info(f"  - {summary_file.name}")
        for plot_file in plot_files:
            log.info(f"  - {plot_file.name}")
        log.info("="*50)

        return True

    except Exception as e:
        log.error(f"Prediction stage failed: {e}")
        log.error("Full traceback:", exc_info=True)
        return False


def make_single_prediction(
    run_id: str,
    feature_values: Dict[str, Any],
    return_details: bool = False
) -> Union[Any, Dict[str, Any]]:
    """
    Make a single prediction with provided feature values.

    Args:
        run_id: Unique run identifier
        feature_values: Dictionary of feature_name -> value
        return_details: Whether to return detailed prediction information

    Returns:
        Prediction value (if return_details=False) or detailed dictionary (if True)
    """
    # Load model and metadata
    metadata = storage.read_json(run_id, constants.METADATA_FILENAME)
    if not metadata:
        raise Exception("Could not load metadata")

    run_dir = storage.get_run_dir(run_id)
    model = predict_logic.load_pipeline(run_dir)

    # Convert feature values to DataFrame
    input_df = pd.DataFrame([feature_values])

    # Generate prediction
    result_df = predict_logic.generate_predictions(model, input_df)
    prediction_value = result_df['prediction'].iloc[0]

    if return_details:
        target_info = schemas.TargetInfo(**metadata['target_info'])
        return {
            'prediction': prediction_value,
            'target_column': target_info.name,
            'task_type': target_info.task_type,
            'input_features': feature_values,
            'aligned_features': result_df.drop(columns=['prediction']).iloc[0].to_dict(),
            'timestamp': datetime.now().isoformat()
        }
    else:
        return prediction_value


def validate_prediction_stage_inputs(run_id: str) -> bool:
    """
    Validate that all required inputs for the prediction stage are available.

    Args:
        run_id: Run identifier

    Returns:
        True if all inputs are valid, False otherwise
    """
    log = logger.get_stage_logger(run_id, "PREDICT")

    try:
        # Check if run directory exists
        run_dir = storage.get_run_dir(run_id)
        if not run_dir.exists():
            log.error(f"Run directory does not exist: {run_dir}")
            return False

        # Check if metadata.json exists
        metadata_path = run_dir / constants.METADATA_FILENAME
        if not metadata_path.exists():
            log.error(f"Metadata file does not exist: {metadata_path}")
            return False

        # Check if model file exists
        model_path = run_dir / constants.MODEL_DIR / "pycaret_pipeline.pkl"
        if not model_path.exists():
            log.error(f"Model file does not exist: {model_path}")
            return False

        # Try to load and validate metadata structure
        try:
            metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)

            # Check for required keys
            if 'target_info' not in metadata_dict:
                log.error("Missing 'target_info' in metadata")
                return False

            if 'automl_info' not in metadata_dict:
                log.error("Missing 'automl_info' in metadata")
                return False

            # Validate target_info can be parsed
            target_info_dict = metadata_dict['target_info']
            schemas.TargetInfo(**target_info_dict)

            log.info("All prediction stage inputs validated successfully")
            return True

        except Exception as e:
            log.error(f"Metadata validation failed: {e}")
            return False

    except Exception as e:
        log.error(f"Input validation failed: {e}")
        return False


def get_prediction_stage_summary(run_id: str) -> Optional[dict]:
    """
    Get a summary of the prediction stage capabilities for a given run.

    Args:
        run_id: Run identifier

    Returns:
        Dictionary with prediction stage summary or None if not available
    """
    try:
        if not validate_prediction_stage_inputs(run_id):
            return None

        metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
        target_info = schemas.TargetInfo(**metadata_dict['target_info'])
        automl_info = metadata_dict.get('automl_info', {})

        # Check if class labels are available (for classification)
        class_labels_available = False
        try:
            class_labels_data = storage.read_json(run_id, 'class_labels.json')
            class_labels_available = class_labels_data is not None
        except:
            pass

        return {
            'run_id': run_id,
            'task_type': target_info.task_type,
            'target_column': target_info.name,
            'model_name': automl_info.get('best_model_name', 'Unknown'),
            'model_available': True,
            'class_labels_available': class_labels_available,
            'can_make_predictions': True
        }

    except Exception:
        return None
