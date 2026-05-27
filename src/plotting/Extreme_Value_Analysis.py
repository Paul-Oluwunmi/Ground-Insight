"""
Extreme Value Analysis Module - Backward Compatibility Wrapper
================================================================

This file has been refactored into a modular structure in the extreme_value_analysis/ folder.
This wrapper maintains backward compatibility for existing imports.

NEW USAGE (Recommended):
    from src.plotting.extreme_value_analysis import run_eva_analysis

OLD USAGE (Still works):
    from src.plotting.Extreme_Value_Analysis import run_eva_analysis

The modular structure includes:
- extreme_value_analysis/constants.py: Colors and constants
- extreme_value_analysis/utils.py: Utility functions
- extreme_value_analysis/data_processing.py: Data processing
- extreme_value_analysis/block_maxima.py: Block maxima extraction
- extreme_value_analysis/seasonal_analysis.py: Seasonal analysis
- extreme_value_analysis/eva_analysis.py: Statistical analysis functions
- extreme_value_analysis/plotting/: Plotting functions
- extreme_value_analysis/dashboard/: Dashboard functions
"""

# Import all functions from the modular structure for backward compatibility
from .extreme_value_analysis import (
    run_eva_analysis,
    perform_extreme_value_analysis,
    perform_bayesian_eva_stationary,
    perform_bayesian_eva_non_stationary,
    perform_non_stationary_analysis,
    calculate_eva_stats,
    calculate_non_stationary_return_periods,
    create_time_series_plot,
    create_eva_plots,
    create_stationary_eva_plots,
    create_non_stationary_eva_plots,
    create_corner_plot,
    generate_posterior_predictive_checks,
    init_eva_dashboard,
    register_callbacks,
    get_block_maxima,
    get_processed_data,
    get_global_data,
    find_peaks_polars,
    analyse_seasonal_patterns,
    add_seasonal_annotation,
    get_seasons_numpy,
    get_suitable_block_periods,
    get_resample_options,
    get_default_block_period,
    get_block_size_days,
    format_resample_period,
    get_period_hours,
    convert_time_unit,
    validate_block_period,
    SEASON_COLORS,
    COLORS,
    global_store
)

# Export everything for backward compatibility
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
