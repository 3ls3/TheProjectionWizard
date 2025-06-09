"""
Logging utilities for The Projection Wizard.
Provides run-scoped logging that writes to stage-specific log files.
Includes both human-readable summaries and machine-parseable JSON logs.
"""

import logging
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import numpy as np

from . import storage
from .constants import PIPELINE_LOG_FILENAME, STAGE_LOG_FILENAMES, PIPELINE_STAGES


def json_serializer(obj):
    """
    Custom JSON serializer that handles numpy types and other non-serializable objects.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy types to Python types
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):  # Custom objects
        return obj.__dict__
    else:
        # Fallback: convert to string
        return str(obj)


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    Produces clean JSON-line format suitable for machine parsing.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON line.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON string representation of the log record
        """
        # Base log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        # Extract run_id and stage from logger name if available
        # Logger names follow pattern: projection_wizard.{logger_type}.{stage}.{run_id}
        name_parts = record.name.split('.')
        if len(name_parts) >= 4 and name_parts[0] == 'projection_wizard':
            log_entry["run_id"] = name_parts[-1]
            if len(name_parts) >= 3:
                log_entry["stage"] = name_parts[-2]
        
        # Add any extra JSON data from the log record
        if hasattr(record, 'extra_json') and isinstance(record.extra_json, dict):
            log_entry.update(record.extra_json)
        
        # Handle exceptions
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False, separators=(',', ':'), default=json_serializer)


def get_stage_log_filename(stage: str) -> str:
    """
    Get the log filename for a specific stage.
    
    Args:
        stage: Pipeline stage name
        
    Returns:
        Log filename for the stage (defaults to pipeline.log if stage not recognized)
    """
    return STAGE_LOG_FILENAMES.get(stage, PIPELINE_LOG_FILENAME)


def get_structured_log_filename(stage: str) -> str:
    """
    Get the structured JSON log filename for a specific stage.
    
    Args:
        stage: Pipeline stage name
        
    Returns:
        JSONL filename for the stage
    """
    base_filename = get_stage_log_filename(stage)
    # Replace .log extension with _structured.jsonl
    if base_filename.endswith('.log'):
        return base_filename.replace('.log', '_structured.jsonl')
    else:
        return f"{base_filename}_structured.jsonl"


def get_logger(run_id: str, logger_name: str = 'pipeline', log_level: str = 'INFO', 
               stage: Optional[str] = None) -> logging.Logger:
    """
    Get a run-scoped logger that writes to stage-specific log files.
    
    Args:
        run_id: Unique run identifier
        logger_name: Name for the logger (default: 'pipeline')
        log_level: Logging level as string (default: 'INFO')
        stage: Pipeline stage name for stage-specific logging (optional)
        
    Returns:
        Configured logger instance
    """
    # Create a unique logger name that includes run_id and optionally stage
    if stage:
        full_logger_name = f"projection_wizard.{logger_name}.{stage}.{run_id}"
    else:
        full_logger_name = f"projection_wizard.{logger_name}.{run_id}"
    
    logger = logging.getLogger(full_logger_name)
    
    # Don't add handlers if logger already exists and has handlers
    if logger.handlers:
        return logger
        
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Get run directory (creates if necessary)
    run_dir = storage.get_run_dir(run_id)
    
    # Determine log file based on stage
    if stage:
        log_filename = get_stage_log_filename(stage)
    else:
        log_filename = PIPELINE_LOG_FILENAME
        
    log_file = run_dir / log_filename
    
    # File handler for stage-specific or general log
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    
    # Console handler for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create formatter with run_id and stage context
    if stage:
        stage_display = stage.replace('step_', '').replace('_', ' ').title()
        formatter = logging.Formatter(
            fmt=f'%(asctime)s | {run_id} | {stage_display} | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            fmt=f'%(asctime)s | {run_id} | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger


def get_structured_logger(run_id: str, logger_name: str = 'structured', log_level: str = 'INFO',
                         stage: Optional[str] = None) -> logging.Logger:
    """
    Get a run-scoped structured logger that writes JSON lines to stage-specific JSONL files.
    
    Args:
        run_id: Unique run identifier
        logger_name: Name for the logger (default: 'structured')
        log_level: Logging level as string (default: 'INFO')
        stage: Pipeline stage name for stage-specific logging (optional)
        
    Returns:
        Configured logger instance that outputs JSON lines
    """
    # Create a unique logger name for structured logging
    if stage:
        full_logger_name = f"projection_wizard.{logger_name}.{stage}.{run_id}"
    else:
        full_logger_name = f"projection_wizard.{logger_name}.{run_id}"
    
    logger = logging.getLogger(full_logger_name)
    
    # Don't add handlers if logger already exists and has handlers
    if logger.handlers:
        return logger
        
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Get run directory (creates if necessary)
    run_dir = storage.get_run_dir(run_id)
    
    # Determine JSONL file based on stage
    if stage:
        jsonl_filename = get_structured_log_filename(stage)
    else:
        jsonl_filename = "pipeline_structured.jsonl"
        
    jsonl_file = run_dir / jsonl_filename
    
    # File handler for JSONL output
    file_handler = logging.FileHandler(jsonl_file)
    file_handler.setLevel(numeric_level)
    
    # Use JSON formatter
    json_formatter = JSONFormatter()
    file_handler.setFormatter(json_formatter)
    
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_stage_logger(run_id: str, stage: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a stage-specific logger for a run that writes to a stage-specific log file.
    
    Args:
        run_id: Unique run identifier
        stage: Pipeline stage name
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance with stage context and stage-specific log file
    """
    logger = get_logger(run_id, f"stage.{stage}", logging.getLevelName(level), stage=stage)
    return logger


def get_stage_structured_logger(run_id: str, stage: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a stage-specific structured logger that writes JSON lines to stage-specific JSONL files.
    
    Args:
        run_id: Unique run identifier
        stage: Pipeline stage name
        level: Logging level (default: INFO)
        
    Returns:
        Configured structured logger instance
    """
    logger = get_structured_logger(run_id, f"structured.{stage}", logging.getLevelName(level), stage=stage)
    return logger


def get_general_logger(run_id: str, logger_name: str = 'pipeline', level: int = logging.INFO) -> logging.Logger:
    """
    Get a general logger for a run that writes to the main pipeline.log file.
    Useful for cross-stage logging or general pipeline operations.
    
    Args:
        run_id: Unique run identifier
        logger_name: Name for the logger (default: 'pipeline')
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance that writes to pipeline.log
    """
    logger = get_logger(run_id, logger_name, logging.getLevelName(level), stage=None)
    return logger


def setup_root_logger(level: int = logging.WARNING) -> None:
    """
    Setup root logger to suppress verbose output from dependencies.
    
    Args:
        level: Logging level for root logger (default: WARNING)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Suppress specific noisy loggers
    noisy_loggers = [
        'urllib3.connectionpool',
        'matplotlib',
        'PIL.PngImagePlugin',
        'pycaret'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def log_stage_start(logger: logging.Logger, stage: str, run_id: str) -> None:
    """Log the start of a pipeline stage."""
    logger.info(f"Starting stage '{stage}' for run {run_id}")


def log_stage_end(logger: logging.Logger, stage: str, run_id: str, 
                  duration_seconds: float) -> None:
    """Log the completion of a pipeline stage."""
    logger.info(f"Completed stage '{stage}' for run {run_id} in {duration_seconds:.2f}s")


def log_error(logger: logging.Logger, stage: str, error: Exception, run_id: str) -> None:
    """Log an error during a pipeline stage."""
    logger.error(f"Error in stage '{stage}' for run {run_id}: {str(error)}", exc_info=True)


# =============================
# STRUCTURED LOGGING HELPERS
# =============================

def log_structured_event(logger: logging.Logger, event_type: str, 
                         event_data: Optional[Dict[str, Any]] = None,
                         message: str = "") -> None:
    """
    Log a structured event with consistent schema.
    
    Args:
        logger: Structured logger instance
        event_type: Type of event (e.g., 'ingestion_complete', 'model_trained')
        event_data: Additional event-specific data
        message: Human-readable message (optional)
    """
    extra_json = {
        "event": event_type,
        "event_data": event_data or {}
    }
    
    logger.info(message or f"Event: {event_type}", extra={"extra_json": extra_json})


def log_structured_metric(logger: logging.Logger, metric_name: str, 
                         metric_value: Union[float, int, str],
                         metric_type: str = "performance",
                         additional_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Log a structured metric.
    
    Args:
        logger: Structured logger instance
        metric_name: Name of the metric
        metric_value: Value of the metric
        metric_type: Type of metric (e.g., 'performance', 'data_quality', 'timing')
        additional_data: Additional metric context
    """
    extra_json = {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_type": metric_type
    }
    
    if additional_data:
        extra_json.update(additional_data)
    
    logger.info(f"Metric: {metric_name} = {metric_value}", extra={"extra_json": extra_json})


def log_structured_error(logger: logging.Logger, error_type: str, error_message: str,
                        error_context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log a structured error event.
    
    Args:
        logger: Structured logger instance
        error_type: Type of error (e.g., 'validation_failed', 'model_training_error')
        error_message: Error message
        error_context: Additional error context
    """
    extra_json = {
        "error_type": error_type,
        "error_context": error_context or {}
    }
    
    logger.error(error_message, extra={"extra_json": extra_json})


# =============================
# PIPELINE SUMMARY LOGGING
# =============================

class PipelineSummaryLogger:
    """
    High-level pipeline summary logger that creates clear, concise summaries
    of pipeline execution including key metrics and transformations.
    Also emits structured events alongside human-readable summaries.
    """
    
    def __init__(self, run_id: str):
        """Initialize the pipeline summary logger."""
        self.run_id = run_id
        self.logger = get_general_logger(run_id, "SUMMARY")
        self.structured_logger = get_structured_logger(run_id, "pipeline_events")
        self.start_time = datetime.now()
        
    def log_pipeline_start(self, original_filename: str) -> None:
        """Log the start of the pipeline with input file information."""
        self.logger.info("="*80)
        self.logger.info("🔮 PROJECTION WIZARD PIPELINE STARTED")
        self.logger.info("="*80)
        self.logger.info(f"📁 Input File: {original_filename}")
        self.logger.info(f"🆔 Run ID: {self.run_id}")
        self.logger.info(f"⏰ Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")
        
        # Structured event
        log_structured_event(
            self.structured_logger,
            "pipeline_started",
            {
                "input_filename": original_filename,
                "started_at": self.start_time.isoformat()
            },
            f"Pipeline started with file: {original_filename}"
        )
        
    def log_ingestion_summary(self, rows: int, cols: int, filename: str, 
                             warnings: Optional[List[str]] = None) -> None:
        """Log a summary of the data ingestion stage."""
        self.logger.info("📥 DATA INGESTION COMPLETE")
        self.logger.info(f"   • File: {filename}")
        self.logger.info(f"   • Shape: {rows:,} rows × {cols} columns")
        
        if warnings:
            self.logger.info(f"   ⚠️  Warnings: {len(warnings)}")
            for warning in warnings:
                self.logger.info(f"     - {warning}")
        else:
            self.logger.info("   ✅ No issues detected")
        self.logger.info("")
        
        # Structured event
        log_structured_event(
            self.structured_logger,
            "ingestion_complete",
            {
                "filename": filename,
                "rows": rows,
                "columns": cols,
                "warnings": warnings or [],
                "warning_count": len(warnings) if warnings else 0
            },
            f"Data ingestion complete: {rows:,} rows × {cols} columns"
        )
        
    def log_schema_summary(self, target_column: str, task_type: str, 
                          feature_count: int, categorical_count: int, 
                          numeric_count: int) -> None:
        """Log a summary of the schema definition stage."""
        self.logger.info("🎯 SCHEMA DEFINITION COMPLETE")
        self.logger.info(f"   • Target Column: '{target_column}' ({task_type})")
        self.logger.info(f"   • Features: {feature_count} total")
        self.logger.info(f"     - Categorical: {categorical_count}")
        self.logger.info(f"     - Numeric: {numeric_count}")
        self.logger.info("")
        
        # Structured event
        log_structured_event(
            self.structured_logger,
            "schema_definition_complete",
            {
                "target_column": target_column,
                "task_type": task_type,
                "feature_count": feature_count,
                "categorical_count": categorical_count,
                "numeric_count": numeric_count
            },
            f"Schema defined: {target_column} ({task_type}) with {feature_count} features"
        )
        
    def log_validation_summary(self, validation_results: Dict[str, Any]) -> None:
        """Log a summary of the data validation stage."""
        passed = validation_results.get('expectations_passed', 0)
        total = validation_results.get('total_expectations', 0)
        success_rate = (passed / total * 100) if total > 0 else 0
        
        self.logger.info("✅ DATA VALIDATION COMPLETE")
        self.logger.info(f"   • Validation Score: {success_rate:.1f}% ({passed}/{total} checks passed)")
        
        # Log key data quality metrics
        quality_metrics = validation_results.get('quality_metrics', {})
        if quality_metrics:
            missing_pct = quality_metrics.get('missing_percentage', 0)
            duplicate_pct = quality_metrics.get('duplicate_percentage', 0)
            self.logger.info(f"   • Missing Values: {missing_pct:.1f}%")
            self.logger.info(f"   • Duplicate Rows: {duplicate_pct:.1f}%")
            
        warnings = validation_results.get('warnings', [])
        if warnings:
            self.logger.info(f"   ⚠️  Data Quality Warnings: {len(warnings)}")
            for warning in warnings[:3]:  # Show first 3 warnings
                self.logger.info(f"     - {warning}")
            if len(warnings) > 3:
                self.logger.info(f"     ... and {len(warnings) - 3} more")
        self.logger.info("")
        
        # Structured event
        log_structured_event(
            self.structured_logger,
            "validation_complete",
            {
                "validation_score": success_rate,
                "expectations_passed": passed,
                "total_expectations": total,
                "quality_metrics": quality_metrics,
                "warnings": warnings,
                "warning_count": len(warnings)
            },
            f"Data validation complete: {success_rate:.1f}% score ({passed}/{total} checks passed)"
        )
        
    def log_preparation_summary(self, input_shape: tuple, output_shape: tuple,
                               cleaning_steps: List[str], encoding_steps: List[str]) -> None:
        """Log a summary of the data preparation stage."""
        self.logger.info("🔧 DATA PREPARATION COMPLETE")
        self.logger.info(f"   • Input Shape: {input_shape[0]:,} rows × {input_shape[1]} columns")
        self.logger.info(f"   • Output Shape: {output_shape[0]:,} rows × {output_shape[1]} columns")
        
        row_change = output_shape[0] - input_shape[0]
        col_change = output_shape[1] - input_shape[1]
        
        if row_change != 0:
            self.logger.info(f"   • Rows {'Added' if row_change > 0 else 'Removed'}: {abs(row_change):,}")
        if col_change != 0:
            self.logger.info(f"   • Columns {'Added' if col_change > 0 else 'Removed'}: {abs(col_change)}")
            
        self.logger.info(f"   • Cleaning Steps: {', '.join(cleaning_steps) if cleaning_steps else 'None'}")
        self.logger.info(f"   • Encoding Steps: {', '.join(encoding_steps) if encoding_steps else 'None'}")
        self.logger.info("")
        
        # Structured event
        log_structured_event(
            self.structured_logger,
            "preparation_complete",
            {
                "input_shape": {"rows": input_shape[0], "columns": input_shape[1]},
                "output_shape": {"rows": output_shape[0], "columns": output_shape[1]},
                "rows_changed": row_change,
                "columns_changed": col_change,
                "cleaning_steps": cleaning_steps,
                "encoding_steps": encoding_steps
            },
            f"Data preparation complete: {input_shape} → {output_shape}"
        )
        
    def log_automl_summary(self, model_name: str, task_type: str, target_column: str,
                          training_shape: tuple, metrics: Dict[str, float],
                          training_duration: Optional[float] = None) -> None:
        """Log a summary of the AutoML training stage."""
        self.logger.info("🤖 MODEL TRAINING COMPLETE")
        self.logger.info(f"   • Algorithm: {model_name}")
        self.logger.info(f"   • Task Type: {task_type.title()}")
        self.logger.info(f"   • Target: '{target_column}'")
        self.logger.info(f"   • Training Data: {training_shape[0]:,} samples × {training_shape[1]} features")
        
        if training_duration:
            self.logger.info(f"   • Training Time: {training_duration:.1f}s")
            
        # Log performance metrics in a clean format
        self.logger.info("   • Performance Metrics:")
        for metric_name, value in metrics.items():
            # Format different types of metrics appropriately
            if isinstance(value, float):
                if 0 <= value <= 1:
                    self.logger.info(f"     - {metric_name}: {value:.4f}")
                else:
                    self.logger.info(f"     - {metric_name}: {value:.2f}")
            else:
                self.logger.info(f"     - {metric_name}: {value}")
        self.logger.info("")
        
        # Structured event
        log_structured_event(
            self.structured_logger,
            "automl_training_complete",
            {
                "model_name": model_name,
                "task_type": task_type,
                "target_column": target_column,
                "training_shape": {"samples": training_shape[0], "features": training_shape[1]},
                "metrics": metrics,
                "training_duration": training_duration
            },
            f"AutoML training complete: {model_name} with {len(metrics)} metrics"
        )
        
    def log_explainability_summary(self, plots_generated: List[str], 
                                  top_features: Optional[List[str]] = None) -> None:
        """Log a summary of the explainability stage."""
        self.logger.info("🔍 MODEL EXPLAINABILITY COMPLETE")
        self.logger.info(f"   • Plots Generated: {len(plots_generated)}")
        for plot in plots_generated:
            self.logger.info(f"     - {plot}")
            
        if top_features:
            self.logger.info(f"   • Top Important Features:")
            for i, feature in enumerate(top_features[:5], 1):  # Show top 5
                self.logger.info(f"     {i}. {feature}")
        self.logger.info("")
        
        # Structured event
        log_structured_event(
            self.structured_logger,
            "explainability_complete",
            {
                "plots_generated": plots_generated,
                "plot_count": len(plots_generated),
                "top_features": top_features or [],
                "feature_count": len(top_features) if top_features else 0
            },
            f"Model explainability complete: {len(plots_generated)} plots generated"
        )
        
    def log_pipeline_completion(self, success: bool = True, error_message: Optional[str] = None) -> None:
        """Log the completion of the entire pipeline."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if success:
            self.logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*80)
            self.logger.info(f"⏱️  Total Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            self.logger.info(f"🆔 Run ID: {self.run_id}")
            self.logger.info(f"📁 Results saved to: data/runs/{self.run_id}/")
            self.logger.info("")
            self.logger.info("Generated Artifacts:")
            
            # List expected artifacts
            artifacts = [
                "📊 original_data.csv - Raw uploaded data",
                "🧹 cleaned_data.csv - Preprocessed data ready for ML", 
                "📋 metadata.json - Complete pipeline metadata",
                "🤖 model/ - Trained model files",
                "📈 plots/ - Explainability visualizations",
                "📝 *.log - Detailed execution logs"
            ]
            
            for artifact in artifacts:
                self.logger.info(f"   • {artifact}")
        else:
            self.logger.info("❌ PIPELINE FAILED")
            self.logger.info("="*80)
            self.logger.info(f"⏱️  Duration before failure: {duration:.1f} seconds")
            self.logger.info(f"🆔 Run ID: {self.run_id}")
            if error_message:
                self.logger.info(f"❗ Error: {error_message}")
                
        self.logger.info("="*80)
        self.logger.info("")
        
        # Structured event
        event_type = "pipeline_completed" if success else "pipeline_failed"
        event_data = {
            "success": success,
            "duration_seconds": duration,
            "completed_at": end_time.isoformat()
        }
        
        if error_message:
            event_data["error_message"] = error_message
            
        log_structured_event(
            self.structured_logger,
            event_type,
            event_data,
            f"Pipeline {'completed successfully' if success else 'failed'} in {duration:.1f}s"
        )


def get_pipeline_summary_logger(run_id: str) -> PipelineSummaryLogger:
    """
    Get a pipeline summary logger instance.
    
    Args:
        run_id: Unique run identifier
        
    Returns:
        PipelineSummaryLogger instance
    """
    return PipelineSummaryLogger(run_id) 