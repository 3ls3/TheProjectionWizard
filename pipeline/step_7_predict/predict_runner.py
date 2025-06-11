"""
Prediction Runner for The Projection Wizard.
Orchestrates prediction operations including model loading, data processing, and result generation.
Refactored for GCS-based storage.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import tempfile
import io

from common import logger, storage, constants, schemas
from api.utils.gcs_utils import (
    download_run_file, upload_run_file, check_run_file_exists, 
    PROJECT_BUCKET_NAME
)
from . import predict_logic, plot_utils


def run_prediction_stage_gcs(
    run_id: str,
    input_data: Union[pd.DataFrame, str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    generate_plots: bool = True,
    gcs_bucket_name: str = PROJECT_BUCKET_NAME
) -> bool:
    """
    Execute the complete prediction stage for a given run with input data using GCS storage.

    This function orchestrates:
    1. Loading the trained model and metadata from GCS
    2. Processing input data and generating predictions
    3. Creating visualizations (if enabled) and saving to GCS
    4. Saving results to GCS

    Args:
        run_id: Unique run identifier
        input_data: Input data for predictions - can be:
            - pd.DataFrame: Direct DataFrame input
            - str/Path: Path to CSV file to load
        output_dir: Directory to save results (ignored in GCS version, uses GCS paths)
        generate_plots: Whether to generate visualization plots
        gcs_bucket_name: GCS bucket name for storage

    Returns:
        True if prediction completes successfully, False otherwise
    """
    # Get loggers for this run
    log = logger.get_stage_logger(run_id, "PREDICT")

    try:
        log.info(f"Starting prediction stage for run {run_id} (GCS-based)")

        # =============================
        # 1. LOAD METADATA AND MODEL FROM GCS
        # =============================
        log.info("Loading metadata and trained model from GCS...")

        # Load metadata
        metadata = storage.read_json(run_id, constants.METADATA_FILENAME)
        if not metadata:
            raise Exception("Could not load metadata from GCS")

        target_info = schemas.TargetInfo(**metadata['target_info'])
        log.info(f"Target: {target_info.name}, Task: {target_info.task_type}")

        # Load trained model from GCS
        model = predict_logic.load_pipeline_gcs(run_id, gcs_bucket_name)
        log.info("Model loaded successfully from GCS")

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

        result_df = predict_logic.generate_predictions(model, input_df, target_info.name)
        log.info(f"Generated {len(result_df)} predictions")

        # Get prediction summary
        summary = predict_logic.get_prediction_summary(result_df, target_info.task_type)
        log.info(f"Prediction summary: {summary}")

        # =============================
        # 4. SAVE RESULTS TO GCS
        # =============================
        log.info("Saving prediction results to GCS...")

        # Generate timestamp for file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save results CSV to GCS
        results_filename = f"predictions/predictions_{timestamp}.csv"
        results_csv_bytes = io.BytesIO()
        result_df.to_csv(results_csv_bytes, index=False)
        upload_success = upload_run_file(run_id, results_filename, results_csv_bytes)
        
        if upload_success:
            log.info(f"Results saved to GCS: {results_filename}")
        else:
            raise Exception("Failed to upload results to GCS")

        # Save summary JSON to GCS
        summary_filename = f"predictions/prediction_summary_{timestamp}.json"
        summary_json_bytes = io.BytesIO(str(summary).encode())
        upload_success = upload_run_file(run_id, summary_filename, summary_json_bytes)
        
        if upload_success:
            log.info(f"Summary saved to GCS: {summary_filename}")
        else:
            log.warning("Failed to upload summary to GCS")

        # =============================
        # 5. GENERATE PLOTS AND SAVE TO GCS (IF ENABLED)
        # =============================
        plot_files = []

        if generate_plots:
            log.info("Generating visualization plots...")

            try:
                if target_info.task_type.lower() == "classification":
                    # Create prediction distribution plot
                    plot_gcs_path = f"predictions/plots/prediction_distribution_{timestamp}.png"
                    plot_success = plot_utils.create_prediction_summary_plot_gcs(
                        result_df, target_info.task_type, run_id, plot_gcs_path,
                        title=f"Prediction Distribution - {target_info.name}",
                        gcs_bucket_name=gcs_bucket_name
                    )
                    if plot_success:
                        plot_files.append(plot_gcs_path)

                elif target_info.task_type.lower() == "regression":
                    # Create prediction distribution plot
                    plot_gcs_path = f"predictions/plots/prediction_distribution_{timestamp}.png"
                    plot_success = plot_utils.create_prediction_summary_plot_gcs(
                        result_df, target_info.task_type, run_id, plot_gcs_path,
                        title=f"Prediction Distribution - {target_info.name}",
                        gcs_bucket_name=gcs_bucket_name
                    )
                    if plot_success:
                        plot_files.append(plot_gcs_path)

                log.info(f"Generated {len(plot_files)} visualization plots in GCS")

            except Exception as e:
                log.warning(f"Failed to generate plots: {e}")

        # =============================
        # 6. LOG COMPLETION
        # =============================
        log.info("="*50)
        log.info("PREDICTION STAGE COMPLETED SUCCESSFULLY (GCS)")
        log.info("="*50)
        log.info(f"Input shape: {input_df.shape}")
        log.info(f"Predictions generated: {len(result_df)}")
        log.info(f"Task type: {target_info.task_type}")
        log.info(f"Target column: {target_info.name}")
        log.info(f"GCS output files:")
        log.info(f"  - {results_filename}")
        log.info(f"  - {summary_filename}")
        for plot_file in plot_files:
            log.info(f"  - {plot_file}")
        log.info("="*50)

        return True

    except Exception as e:
        log.error(f"Prediction stage failed: {e}")
        log.error("Full traceback:", exc_info=True)
        return False


def run_prediction_stage(
    run_id: str,
    input_data: Union[pd.DataFrame, str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    generate_plots: bool = True
) -> bool:
    """
    Legacy compatibility function - redirects to GCS version.
    
    Args:
        run_id: Unique run identifier
        input_data: Input data for predictions
        output_dir: Directory to save results (ignored in GCS version)
        generate_plots: Whether to generate visualization plots

    Returns:
        True if prediction completes successfully, False otherwise
    """
    logger_instance = logger.get_stage_logger(run_id, "PREDICT")
    logger_instance.warning("Using legacy run_prediction_stage function - redirecting to GCS version")
    return run_prediction_stage_gcs(run_id, input_data, output_dir, generate_plots)


def make_single_prediction_gcs(
    run_id: str,
    feature_values: Dict[str, Any],
    return_details: bool = False,
    gcs_bucket_name: str = PROJECT_BUCKET_NAME
) -> Union[Any, Dict[str, Any]]:
    """
    Make a single prediction with provided feature values using GCS storage.

    Args:
        run_id: Unique run identifier
        feature_values: Dictionary of feature_name -> value
        return_details: Whether to return detailed prediction information
        gcs_bucket_name: GCS bucket name for storage

    Returns:
        Prediction value (if return_details=False) or detailed dictionary (if True)
    """
    # Load model and metadata from GCS
    metadata = storage.read_json(run_id, constants.METADATA_FILENAME)
    if not metadata:
        raise Exception("Could not load metadata from GCS")

    target_info = schemas.TargetInfo(**metadata['target_info'])
    model = predict_logic.load_pipeline_gcs(run_id, gcs_bucket_name)

    # Convert feature values to DataFrame
    input_df = pd.DataFrame([feature_values])

    # Generate prediction
    result_df = predict_logic.generate_predictions(model, input_df, target_info.name)
    prediction_value = result_df['prediction'].iloc[0]

    if return_details:
        target_info = schemas.TargetInfo(**metadata['target_info'])
        return {
            'prediction': prediction_value,
            'target_column': target_info.name,
            'task_type': target_info.task_type,
            'input_features': feature_values,
            'aligned_features': result_df.drop(columns=['prediction']).iloc[0].to_dict(),
            'timestamp': datetime.now().isoformat(),
            'storage_type': 'gcs',
            'gcs_bucket': gcs_bucket_name
        }
    else:
        return prediction_value


def make_single_prediction(
    run_id: str,
    feature_values: Dict[str, Any],
    return_details: bool = False
) -> Union[Any, Dict[str, Any]]:
    """
    Legacy compatibility function - redirects to GCS version.

    Args:
        run_id: Unique run identifier
        feature_values: Dictionary of feature_name -> value
        return_details: Whether to return detailed prediction information

    Returns:
        Prediction value (if return_details=False) or detailed dictionary (if True)
    """
    return make_single_prediction_gcs(run_id, feature_values, return_details)


def validate_prediction_stage_inputs_gcs(run_id: str,
                                        gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> bool:
    """
    Validate that all required inputs for the prediction stage are available in GCS.

    Args:
        run_id: Run identifier
        gcs_bucket_name: GCS bucket name

    Returns:
        True if all inputs are valid, False otherwise
    """
    log = logger.get_stage_logger(run_id, "PREDICT")

    try:
        # Check if metadata.json exists in GCS
        if not check_run_file_exists(run_id, constants.METADATA_FILENAME):
            log.error(f"Metadata file does not exist in GCS: {constants.METADATA_FILENAME}")
            return False

        # Check if model file exists in GCS
        # Try both GCS path and legacy path
        model_gcs_path = f"{constants.MODEL_DIR}/pycaret_pipeline.pkl"
        if not check_run_file_exists(run_id, model_gcs_path):
            log.error(f"Model file does not exist in GCS: {model_gcs_path}")
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

            log.info("All prediction stage inputs validated successfully (GCS)")
            return True

        except Exception as e:
            log.error(f"Metadata validation failed: {e}")
            return False

    except Exception as e:
        log.error(f"Input validation failed: {e}")
        return False


def validate_prediction_stage_inputs(run_id: str) -> bool:
    """
    Legacy compatibility function - redirects to GCS version.

    Args:
        run_id: Run identifier

    Returns:
        True if all inputs are valid, False otherwise
    """
    logger_instance = logger.get_stage_logger(run_id, "PREDICT")
    logger_instance.warning("Using legacy validate_prediction_stage_inputs function - redirecting to GCS version")
    return validate_prediction_stage_inputs_gcs(run_id)


def get_prediction_stage_summary_gcs(run_id: str) -> Optional[dict]:
    """
    Get a summary of the prediction stage capabilities for a given run (GCS version).

    Args:
        run_id: Run identifier

    Returns:
        Dictionary with prediction stage summary or None if not available
    """
    try:
        if not validate_prediction_stage_inputs_gcs(run_id):
            return None

        metadata_dict = storage.read_json(run_id, constants.METADATA_FILENAME)
        target_info = schemas.TargetInfo(**metadata_dict['target_info'])
        automl_info = metadata_dict.get('automl_info', {})

        # Check if class labels are available (for classification)
        class_labels_available = False
        try:
            if check_run_file_exists(run_id, 'class_labels.json'):
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
            'can_make_predictions': True,
            'storage_type': 'gcs',
            'gcs_bucket': PROJECT_BUCKET_NAME
        }

    except Exception:
        return None


def get_prediction_stage_summary(run_id: str) -> Optional[dict]:
    """
    Legacy compatibility function - redirects to GCS version.

    Args:
        run_id: Run identifier

    Returns:
        Dictionary with prediction stage summary or None if not available
    """
    return get_prediction_stage_summary_gcs(run_id)
