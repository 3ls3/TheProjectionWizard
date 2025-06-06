"""
Logging utilities for The Projection Wizard.
Provides run-scoped logging that writes to stage-specific log files.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from . import storage
from .constants import PIPELINE_LOG_FILENAME, STAGE_LOG_FILENAMES, PIPELINE_STAGES


def get_stage_log_filename(stage: str) -> str:
    """
    Get the log filename for a specific stage.
    
    Args:
        stage: Pipeline stage name
        
    Returns:
        Log filename for the stage (defaults to pipeline.log if stage not recognized)
    """
    return STAGE_LOG_FILENAMES.get(stage, PIPELINE_LOG_FILENAME)


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