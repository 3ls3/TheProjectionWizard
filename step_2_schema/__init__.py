"""
Step 2 Schema module for The Projection Wizard.
Contains target definition and schema confirmation logic.
"""

from .target_definition_logic import suggest_target_and_task, confirm_target_definition

__all__ = ['suggest_target_and_task', 'confirm_target_definition']
