"""
Statistics Module for Groundwater Data
=====================================
Monthly statistics visualization with customizable percentile ranges.

This module has been refactored into a modular structure:
- statistics/constants.py: Constants (colors, month order)
- statistics/statistical_calculations.py: Statistical calculation functions
- statistics/data_processing.py: Data preparation and transformation
- statistics/plotting.py: Plot creation and configuration functions
- statistics/interactive.py: Jupyter notebook interactive widgets
- statistics/__init__.py: Main entry point
"""

# Import from modular structure for backward compatibility
from .statistics import run_statistics, interactive_run_statistics, calculate_significance

__all__ = ['run_statistics', 'interactive_run_statistics', 'calculate_significance']