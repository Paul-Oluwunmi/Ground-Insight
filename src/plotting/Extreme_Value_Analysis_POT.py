"""
Extreme Value Analysis POT Module - Backward Compatibility Wrapper
==================================================================

This file provides backward compatibility for the original Extreme_Value_Analysis_POT.py script.
Helper functions have been moved to the modular extreme_value_analysis_pot package.
The dashboard functions remain here but import from the modular structure.

USAGE:
    from src.plotting.Extreme_Value_Analysis_POT import run_pot_dashboard
    run_pot_dashboard(data, location_columns)
"""

# Import helper functions from modular structure
from .extreme_value_analysis_pot.constants import COLORS, SEASON_COLORS, global_store_pot
from .extreme_value_analysis_pot.data_processing import find_peaks_data, calculate_exceedances
from .extreme_value_analysis_pot.statistical_calculations import (
    calculate_return_periods,
    calculate_confidence_intervals,
    fit_gpd_by_season,
    calculate_return_levels_by_season,
    fit_gpd_with_trend,
    calculate_return_levels_trend,
    fit_gpd_by_covariate,
    calculate_return_levels_covariate
)
from .extreme_value_analysis_pot.bayesian_analysis import (
    perform_bayesian_gpd_stationary,
    perform_bayesian_gpd_non_stationary
)
from .extreme_value_analysis_pot.plotting import (
    update_eva_plots,
    create_corner_plot_gpd,
    generate_posterior_predictive_checks_gpd
)

# Import dashboard functions (will be migrated later)
from .extreme_value_analysis_pot.dashboard import run_pot_dashboard, create_pot_dashboard

__all__ = ['run_pot_dashboard', 'create_pot_dashboard']
