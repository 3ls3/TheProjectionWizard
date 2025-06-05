"""
General utility functions for The Projection Wizard.
"""

import uuid
from datetime import datetime


def generate_run_id() -> str:
    """
    Generate a unique run ID combining ISO 8601 timestamp (UTC) and short UUID.
    
    Format: YYYY-MM-DDTHH-MM-SSZ_shortUUID
    Example: 2025-06-07T103045Z_a1b2c3d4
    
    Returns:
        Unique run identifier string
    """
    # Get current UTC timestamp in ISO 8601 format with custom formatting
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    
    # Generate short UUID (first 8 characters)
    short_uuid = uuid.uuid4().hex[:8]
    
    return f"{timestamp}_{short_uuid}" 