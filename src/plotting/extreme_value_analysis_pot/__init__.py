"""
Extreme Value Analysis POT Module
==================================

POT (Peaks Over Threshold) analysis for extreme value analysis.

Main Functions:
--------------
- run_pot_dashboard: Main entry point for POT analysis dashboard
- create_pot_dashboard: Create POT analysis dashboard
"""

# Import from modules
from .constants import COLORS, SEASON_COLORS, global_store_pot
from .data_processing import find_peaks_data, calculate_exceedances
from .statistical_calculations import (
    calculate_return_periods,
    calculate_confidence_intervals,
    fit_gpd_by_season,
    calculate_return_levels_by_season,
    fit_gpd_with_trend,
    calculate_return_levels_trend,
    fit_gpd_by_covariate,
    calculate_return_levels_covariate
)
from .bayesian_analysis import (
    perform_bayesian_gpd_stationary,
    perform_bayesian_gpd_non_stationary
)
from .plotting import (
    update_eva_plots,
    create_corner_plot_gpd,
    generate_posterior_predictive_checks_gpd
)

# Import dashboard functions (will be available after dashboard migration)
try:
    from .dashboard import run_pot_dashboard, create_pot_dashboard
    __all__ = [
        'run_pot_dashboard',
        'create_pot_dashboard',
        'COLORS',
        'SEASON_COLORS',
        'global_store_pot',
        'find_peaks_data',
        'calculate_exceedances',
        'calculate_return_periods',
        'calculate_confidence_intervals',
        'fit_gpd_by_season',
        'calculate_return_levels_by_season',
        'fit_gpd_with_trend',
        'calculate_return_levels_trend',
        'fit_gpd_by_covariate',
        'calculate_return_levels_covariate',
        'perform_bayesian_gpd_stationary',
        'perform_bayesian_gpd_non_stationary',
        'update_eva_plots',
        'create_corner_plot_gpd',
        'generate_posterior_predictive_checks_gpd'
    ]
except ImportError:
    # Dashboard not yet migrated - import from original file temporarily
    import sys
    import os
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_current_dir)
    _original_file = os.path.join(_parent_dir, 'Extreme_Value_Analysis_POT.py')
    
    # Import dashboard functions from original file
    import importlib.util
    spec = importlib.util.spec_from_file_location("pot_dashboard", _original_file)
    pot_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pot_module)
    
    run_pot_dashboard = pot_module.run_pot_dashboard
    create_pot_dashboard = pot_module.create_pot_dashboard
    
    __all__ = [
        'run_pot_dashboard',
        'create_pot_dashboard',
        'COLORS',
        'SEASON_COLORS',
        'global_store_pot',
        'find_peaks_data',
        'calculate_exceedances',
        'calculate_return_periods',
        'calculate_confidence_intervals',
        'fit_gpd_by_season',
        'calculate_return_levels_by_season',
        'fit_gpd_with_trend',
        'calculate_return_levels_trend',
        'fit_gpd_by_covariate',
        'calculate_return_levels_covariate',
        'perform_bayesian_gpd_stationary',
        'perform_bayesian_gpd_non_stationary',
        'update_eva_plots',
        'create_corner_plot_gpd',
        'generate_posterior_predictive_checks_gpd'
    ]

