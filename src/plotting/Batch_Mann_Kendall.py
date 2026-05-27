"""
Batch Mann-Kendall Analysis Module - Backward Compatibility Wrapper
===================================================================

This file provides a simple wrapper for batch Mann-Kendall analysis functions.
It imports from the modular structure in mann_kendall/Batch_Mann_Kendall/.

USAGE:
    from src.plotting.Batch_Mann_Kendall import (
        run_yearly_water_level_statistics,
        run_hydrological_year_analysis_both_types,
        run_comprehensive_original_mk_batch_analysis_separate_files,
        run_comprehensive_seasonal_mk_batch_analysis_separate_files
    )
"""

# Import batch functions from the modular structure
# Import directly from the Batch_Mann_Kendall package files to avoid circular imports
import sys
import os

# Get the directory containing this file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_mann_kendall_dir = os.path.join(_current_dir, 'mann_kendall')
_batch_mk_dir = os.path.join(_mann_kendall_dir, 'Batch_Mann_Kendall')

# Ensure parent directory is in sys.path for absolute imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# CRITICAL: Create mock src.plotting package BEFORE any imports
# This prevents Python from importing the real src.plotting.__init__.py which has circular imports
if 'src.plotting' not in sys.modules:
    from types import ModuleType
    plotting_pkg = ModuleType('src.plotting')
    plotting_pkg.__path__ = [_current_dir]
    plotting_pkg.__name__ = 'src.plotting'
    plotting_pkg.__package__ = 'src.plotting'
    sys.modules['src.plotting'] = plotting_pkg

# Now import directly from the Batch_Mann_Kendall module files
# Python will use our mock src.plotting instead of importing the real one
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

