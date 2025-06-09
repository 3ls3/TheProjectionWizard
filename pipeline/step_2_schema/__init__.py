"""
Step 2 Schema module for The Projection Wizard.
Contains target definition and schema confirmation logic.
"""

from .target_definition_logic import suggest_target_and_task, confirm_target_definition
from .feature_definition_logic import (
    identify_key_features,
    suggest_initial_feature_schemas,
    confirm_feature_schemas
)

__all__ = [
    'suggest_target_and_task', 
    'confirm_target_definition',
    'identify_key_features',
    'suggest_initial_feature_schemas',
    'confirm_feature_schemas'
]
