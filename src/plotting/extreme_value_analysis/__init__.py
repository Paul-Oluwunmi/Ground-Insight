"""
Extreme Value Analysis Module for Groundwater Data

This module has been refactored into a modular structure:
- constants.py: Colors, season colors, global store
- utils.py: Utility functions (period conversions, formatting, block period selection)
- data_processing.py: Data processing and resampling functions
- block_maxima.py: Block maxima extraction
- seasonal_analysis.py: Seasonal pattern analysis
- eva_analysis.py: EVA statistical analysis functions (stationary, Bayesian GEV)
- plotting/: Plotting functions (time series, EVA plots, corner plots, PPC)
- dashboard/: Dashboard layout and callbacks

Note: This is separate from statistics.py and statistical_tests.py modules.
"""

# Import core functions from original file for now (will be migrated gradually)
# This maintains backward compatibility while establishing the modular structure
import sys
import os

# Add parent directory to path to import from original file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_original_file = os.path.join(_parent_dir, 'Extreme_Value_Analysis.py')

# Import from modules we've created
from .constants import SEASON_COLORS, COLORS, global_store
from .utils import (
    get_period_hours, get_block_size_days, format_resample_period,
    convert_time_unit, get_suitable_block_periods, get_default_block_period,
    get_resample_options, validate_block_period
)
from .data_processing import get_processed_data, get_global_data, find_peaks_polars
from .block_maxima import get_block_maxima
from .seasonal_analysis import analyse_seasonal_patterns, add_seasonal_annotation, get_seasons_numpy
from .eva_analysis import (
    perform_extreme_value_analysis,
    perform_bayesian_eva_stationary,
    perform_bayesian_eva_non_stationary,
    perform_non_stationary_analysis,
    calculate_eva_stats,
    calculate_non_stationary_return_periods
)
from .plotting import (
    create_time_series_plot,
    create_eva_plots,
    create_stationary_eva_plots,
    create_non_stationary_eva_plots,
    create_corner_plot,
    generate_posterior_predictive_checks
)
from .dashboard import init_eva_dashboard, register_callbacks, run_eva_analysis

__all__ = [
    'run_eva_analysis',
    'perform_extreme_value_analysis',
    'perform_bayesian_eva_stationary',
    'perform_bayesian_eva_non_stationary',
    'perform_non_stationary_analysis',
    'calculate_eva_stats',
    'calculate_non_stationary_return_periods',
    'create_time_series_plot',
    'create_eva_plots',
    'create_stationary_eva_plots',
    'create_non_stationary_eva_plots',
    'create_corner_plot',
    'generate_posterior_predictive_checks',
    'init_eva_dashboard',
    'register_callbacks',
    'get_block_maxima',
    'get_processed_data',
    'get_global_data',
    'find_peaks_polars',
    'analyse_seasonal_patterns',
    'add_seasonal_annotation',
    'get_seasons_numpy',
    'get_suitable_block_periods',
    'get_resample_options',
    'get_default_block_period',
    'get_block_size_days',
    'format_resample_period',
    'get_period_hours',
    'convert_time_unit',
    'validate_block_period',
    'SEASON_COLORS',
    'COLORS',
    'global_store'
]

