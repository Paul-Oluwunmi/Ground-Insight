"""
Plotting Module

This module contains functions for creating plots for Mann-Kendall analysis.
"""

from .individual_plots import (
    create_gwl_rolling_mean_plot,
    create_seasonal_analysis_plot,
    create_mk_trend_plot,
    create_duration_analysis_plot,
    create_low_flow_analysis_plot,
    create_gwl_rolling_mean_plot_simple,
    create_individual_well_plots
)
from .batch_plots import (
    run_individual_well_plotting,
    run_gwl_rolling_mean_plots
)

__all__ = [
    'create_gwl_rolling_mean_plot',
    'create_seasonal_analysis_plot',
    'create_mk_trend_plot',
    'create_duration_analysis_plot',
    'create_low_flow_analysis_plot',
    'create_gwl_rolling_mean_plot_simple',
    'create_individual_well_plots',
    'run_individual_well_plotting',
    'run_gwl_rolling_mean_plots'
]
