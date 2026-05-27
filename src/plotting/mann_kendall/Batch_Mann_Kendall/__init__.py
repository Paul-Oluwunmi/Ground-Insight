"""
Batch Mann-Kendall Wrapper Functions

This module contains comprehensive batch wrapper functions that orchestrate
multiple Mann-Kendall analysis types and handle file organization.
"""

# Use absolute imports - standalone, no fallback to original file
# All imports are from the modular structure only
import sys
import os

# Ensure parent directory is in sys.path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import directly from modular structure - standalone, no fallback
from src.plotting.mann_kendall.Batch_Mann_Kendall.yearly_statistics import run_yearly_water_level_statistics
from src.plotting.mann_kendall.Batch_Mann_Kendall.hydrological_year import run_hydrological_year_analysis_both_types
from src.plotting.mann_kendall.Batch_Mann_Kendall.original_mk import run_comprehensive_original_mk_batch_analysis_separate_files
from src.plotting.mann_kendall.Batch_Mann_Kendall.seasonal_mk import run_comprehensive_seasonal_mk_batch_analysis_separate_files

__all__ = [
    'run_yearly_water_level_statistics',
    'run_hydrological_year_analysis_both_types',
    'run_comprehensive_original_mk_batch_analysis_separate_files',
    'run_comprehensive_seasonal_mk_batch_analysis_separate_files',
]

