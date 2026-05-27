"""
Groundwater Statistical Analysis Module
=====================================
Interactive statistical analysis tool for groundwater levels with visualisation.

This module has been refactored into a modular structure:
- statistical_tests/styles.py: UI styles
- statistical_tests/components.py: UI components (guide, dropdowns)
- statistical_tests/layout.py: Dashboard layout
- statistical_tests/callbacks.py: Dash callbacks
- statistical_tests/statistics.py: Statistical calculations
- statistical_tests/plotting.py: Plot creation functions
- statistical_tests/__init__.py: Main entry point
"""

# Import from modular structure for backward compatibility
# Note: The folder name is 'statistical_tests', so we import from it directly
from .statistical_tests import run_statistical_analysis

__all__ = ['run_statistical_analysis']