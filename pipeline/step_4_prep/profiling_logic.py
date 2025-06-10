"""
Data profiling logic for The Projection Wizard.
Contains functions for generating ydata-profiling reports of prepared data.
Refactored for GCS-based storage.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import warnings
import signal
import time
import tempfile
import io
import logging

from common import constants
from api.utils.gcs_utils import upload_run_file, PROJECT_BUCKET_NAME

# Configure logging for this module
logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Profile generation timed out")


def generate_profile_report_gcs_with_timeout(df_final_prepared: pd.DataFrame, 
                                           report_filename: str,
                                           run_id: str,
                                           title: str,
                                           timeout_seconds: int = 300,
                                           gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> bool:
    """
    Generate profile report with timeout protection and save to GCS.
    
    Args:
        df_final_prepared: The DataFrame after cleaning and encoding
        report_filename: Filename for the HTML report in GCS
        run_id: Run ID for GCS path
        title: Title for the ydata-profiling report
        timeout_seconds: Maximum time to allow for profile generation (default: 5 minutes)
        gcs_bucket_name: GCS bucket name
        
    Returns:
        True if report generation successful, False otherwise
    """
    try:
        # Set up timeout signal (Unix/Mac only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        # Generate the profile
        result = generate_profile_report_gcs(df_final_prepared, report_filename, run_id, title, gcs_bucket_name)
        
        # Cancel the alarm if we succeeded
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            
        return result
        
    except TimeoutError:
        logger.error(f"Profile generation timed out after {timeout_seconds} seconds")
        return False
        
    except Exception as e:
        # Cancel the alarm on any exception
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        logger.error(f"Profile generation failed with exception: {str(e)}")
        return False
    
    finally:
        # Ensure alarm is always cancelled
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def generate_profile_report_gcs(df_final_prepared: pd.DataFrame, 
                               report_filename: str,
                               run_id: str, 
                               title: str,
                               gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> bool:
    """
    Generate a ydata-profiling HTML report for the prepared DataFrame and save to GCS.
    
    Args:
        df_final_prepared: The DataFrame after cleaning and encoding
        report_filename: Filename for the HTML report in GCS
        run_id: Run ID for GCS path
        title: Title for the ydata-profiling report
        gcs_bucket_name: GCS bucket name
        
    Returns:
        True if report generation successful, False otherwise
    """
    try:
        # Import ydata-profiling - handle potential import issues
        try:
            from ydata_profiling import ProfileReport
        except ImportError:
            # Fallback to older pandas-profiling package name
            try:
                from pandas_profiling import ProfileReport
            except ImportError:
                logger.error("Neither ydata-profiling nor pandas-profiling is installed. "
                           "Please install with: pip install ydata-profiling")
                return False
        
        # Log start of profiling
        logger.info(f"Starting profile report generation for '{title}'")
        logger.info(f"DataFrame shape: {df_final_prepared.shape}")
        logger.info(f"Report will be saved to GCS: runs/{run_id}/{report_filename}")
        
        # Configure profiling settings for robustness and performance
        # Suppress warnings during profiling to avoid cluttering logs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Generate profile with conservative settings for stability
            profile = ProfileReport(
                df_final_prepared,
                title=title,
                explorative=True,
                # Additional settings for robustness
                minimal=False,  # Full report
                samples={"head": 5, "tail": 5},  # Limit sample size for performance
                correlations={
                    "auto": {"calculate": False},      # DISABLE - can hang on large datasets
                    "pearson": {"calculate": False},   # DISABLE - can hang on large datasets
                    "spearman": {"calculate": False},  # DISABLE - can hang on large datasets
                    "kendall": {"calculate": False},   # DISABLE - can hang on large datasets
                    "phi_k": {"calculate": False},     # DISABLE - can hang on large datasets
                    "cramers": {"calculate": False}    # DISABLE - can hang on large datasets
                },
                missing_diagrams={
                    "matrix": True,
                    "bar": True,
                    "heatmap": False  # Skip heatmap for performance with large datasets
                },
                interactions={"continuous": False, "targets": []},  # Disable interactions for performance
                duplicates={"head": 5}  # Limit duplicate examples
            )
        
        # Save the report to a temporary file, then upload to GCS
        logger.info("Generating and saving profile report...")
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        try:
            # Generate report to temporary file
            profile.to_file(temp_path)
            
            # Upload to GCS
            upload_success = upload_run_file(run_id, report_filename, temp_path)
            
            if upload_success:
                # Get file size for logging
                temp_size_mb = Path(temp_path).stat().st_size / (1024 * 1024)
                logger.info(f"Profile report successfully generated and uploaded to GCS")
                logger.info(f"Report file size: {temp_size_mb:.2f} MB")
                return True
            else:
                logger.error("Failed to upload profile report to GCS")
                return False
                
        finally:
            # Clean up temporary file
            try:
                Path(temp_path).unlink()
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up temporary file: {cleanup_error}")
            
    except Exception as e:
        logger.error(f"Failed to generate profile report: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return False


def generate_profile_report_gcs_with_fallback(df_final_prepared: pd.DataFrame, 
                                             report_filename: str,
                                             run_id: str,
                                             title: str,
                                             gcs_bucket_name: str = PROJECT_BUCKET_NAME) -> bool:
    """
    Generate a ydata-profiling report with fallback options for better compatibility and save to GCS.
    
    Args:
        df_final_prepared: The DataFrame after cleaning and encoding
        report_filename: Filename for the HTML report in GCS
        run_id: Run ID for GCS path
        title: Title for the ydata-profiling report
        gcs_bucket_name: GCS bucket name
        
    Returns:
        True if report generation successful, False otherwise
    """
    # First attempt: Full profile report with timeout protection
    logger.info(f"Attempting full profile report generation for '{title}' (with 5-minute timeout)")
    success = generate_profile_report_gcs_with_timeout(
        df_final_prepared, report_filename, run_id, title, timeout_seconds=300, gcs_bucket_name=gcs_bucket_name
    )
    
    if success:
        return True
    
    # Fallback 1: Try with minimal profile
    logger.warning("Full profile failed, attempting minimal profile...")
    try:
        from ydata_profiling import ProfileReport
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Minimal profile with basic statistics only - ULTRA SAFE
            profile = ProfileReport(
                df_final_prepared,
                title=f"{title} (Minimal)",
                minimal=True,  # Minimal report
                explorative=False,
                # DISABLE ALL CORRELATIONS - these cause hangs
                correlations={
                    "auto": {"calculate": False},
                    "pearson": {"calculate": False},
                    "spearman": {"calculate": False},
                    "kendall": {"calculate": False},
                    "phi_k": {"calculate": False},
                    "cramers": {"calculate": False}
                },
                missing_diagrams={"matrix": False, "bar": True, "heatmap": False},
                interactions={"continuous": False, "targets": []},
                duplicates={"head": 0},
                # Additional safety settings
                samples={"head": 0, "tail": 0},  # No sample display
                infer_dtypes=False  # Skip dtype inference
            )
        
        # Save to temporary file and upload to GCS
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        try:
            profile.to_file(temp_path)
            upload_success = upload_run_file(run_id, report_filename, temp_path)
            
            if upload_success:
                logger.info(f"Minimal profile report successfully generated and uploaded to GCS")
                return True
        finally:
            # Clean up temporary file
            try:
                Path(temp_path).unlink()
            except Exception:
                pass
                
    except Exception as e:
        logger.error(f"Minimal profile also failed: {str(e)}")
    
    # Fallback 2: Create basic HTML summary
    logger.warning("All profiling attempts failed, creating basic HTML summary...")
    try:
        basic_html = _create_basic_html_summary(df_final_prepared, title)
        html_bytes = basic_html.encode('utf-8')
        
        upload_success = upload_run_file(run_id, report_filename, io.BytesIO(html_bytes))
        
        if upload_success:
            logger.info(f"Basic HTML summary created and uploaded to GCS")
            return True
            
    except Exception as e:
        logger.error(f"Even basic HTML generation failed: {str(e)}")
    
    return False


def _create_basic_html_summary(df: pd.DataFrame, title: str) -> str:
    """
    Create a basic HTML summary as a fallback when ydata-profiling fails.
    
    Args:
        df: DataFrame to summarize
        title: Title for the report
        
    Returns:
        HTML string with basic statistics
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <p><strong>Basic Data Summary Report</strong></p>
            <p><em>Generated as fallback - full profiling was not available</em></p>
        </div>
        
        <div class="warning">
            <strong>Note:</strong> This is a basic summary report. For detailed profiling, 
            please ensure ydata-profiling is properly installed and compatible.
        </div>
        
        <div class="section">
            <h2>Dataset Overview</h2>
            <ul>
                <li><strong>Rows:</strong> {df.shape[0]:,}</li>
                <li><strong>Columns:</strong> {df.shape[1]:,}</li>
                <li><strong>Memory Usage:</strong> {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>Column Information</h2>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Data Type</th>
                    <th>Non-Null Count</th>
                    <th>Missing %</th>
                </tr>
    """
    
    # Add column information
    for col in df.columns:
        non_null_count = df[col].count()
        missing_pct = (1 - non_null_count / len(df)) * 100
        html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{df[col].dtype}</td>
                    <td>{non_null_count:,}</td>
                    <td>{missing_pct:.1f}%</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Numeric Columns Summary</h2>
    """
    
    # Add numeric summary if any numeric columns exist
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        html += f"<p>Found {len(numeric_cols)} numeric columns</p>"
        desc_stats = df[numeric_cols].describe()
        html += desc_stats.to_html(classes='', table_id='numeric-summary')
    else:
        html += "<p>No numeric columns found</p>"
    
    html += """
        </div>
        
        <div class="section">
            <h2>Categorical Columns Summary</h2>
    """
    
    # Add categorical summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        html += f"<p>Found {len(categorical_cols)} categorical columns</p>"
        html += "<ul>"
        for col in categorical_cols[:10]:  # Limit to first 10
            unique_count = df[col].nunique()
            html += f"<li><strong>{col}:</strong> {unique_count} unique values</li>"
        if len(categorical_cols) > 10:
            html += f"<li><em>... and {len(categorical_cols) - 10} more columns</em></li>"
        html += "</ul>"
    else:
        html += "<p>No categorical columns found</p>"
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html


# Legacy compatibility functions (redirects to GCS versions)
def generate_profile_report_with_fallback(df_final_prepared: pd.DataFrame, 
                                        report_path: Path, 
                                        title: str,
                                        run_id: Optional[str] = None) -> bool:
    """
    Legacy compatibility function - redirects to GCS version.
    """
    logger.warning("Using legacy generate_profile_report_with_fallback function - redirecting to GCS version")
    if run_id is None:
        logger.error("run_id is required for GCS version")
        return False
    
    report_filename = report_path.name
    return generate_profile_report_gcs_with_fallback(df_final_prepared, report_filename, run_id, title)


def generate_profile_report_with_timeout(df_final_prepared: pd.DataFrame, 
                                       report_path: Path, 
                                       title: str,
                                       timeout_seconds: int = 300,
                                       run_id: Optional[str] = None) -> bool:
    """
    Legacy compatibility function - redirects to GCS version.
    """
    logger.warning("Using legacy generate_profile_report_with_timeout function - redirecting to GCS version")
    if run_id is None:
        logger.error("run_id is required for GCS version")
        return False
    
    report_filename = report_path.name
    return generate_profile_report_gcs_with_timeout(df_final_prepared, report_filename, run_id, title, timeout_seconds)


def generate_profile_report(df_final_prepared: pd.DataFrame, 
                          report_path: Path, 
                          title: str,
                          run_id: Optional[str] = None) -> bool:
    """
    Legacy compatibility function - redirects to GCS version.
    """
    logger.warning("Using legacy generate_profile_report function - redirecting to GCS version")
    if run_id is None:
        logger.error("run_id is required for GCS version")
        return False
    
    report_filename = report_path.name
    return generate_profile_report_gcs(df_final_prepared, report_filename, run_id, title) 