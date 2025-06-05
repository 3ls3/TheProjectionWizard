"""
Logging utilities for The Projection Wizard.
Provides run-scoped logging that writes to run-specific log files.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .constants import RUNS_DIR, PIPELINE_LOG_FILE


def get_logger(run_id: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a run-scoped logger that writes to the run's pipeline.log file.
    
    Args:
        run_id: Unique run identifier
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger_name = f"projection_wizard.{run_id}"
    logger = logging.getLogger(logger_name)
    
    # Don't add handlers if logger already exists and has handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create run directory if it doesn't exist
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # File handler for run-specific log
    log_file = run_dir / PIPELINE_LOG_FILE
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Console handler for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
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
    Get a stage-specific logger for a run.
    
    Args:
        run_id: Unique run identifier
        stage: Pipeline stage name
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance with stage context
    """
    logger = get_logger(run_id, level)
    
    # Create a stage-specific adapter
    return logging.LoggerAdapter(logger, {'stage': stage})


class PipelineLoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter that adds pipeline context to log messages."""
    
    def process(self, msg, kwargs):
        stage = self.extra.get('stage', 'unknown')
        return f"[{stage.upper()}] {msg}", kwargs


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