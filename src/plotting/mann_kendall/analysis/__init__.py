"""
Analysis Module

This module contains various analysis functions for Mann-Kendall analysis.
"""

from .duration_low_flow import (
    calculate_yearly_max_duration_metrics,
    calculate_percentile_low_flow_metrics
)
from .yearly_statistics import (
    calculate_yearly_water_level_statistics
)

from .seasonal import (
    perform_segmented_enhanced_seasonal_analysis,
    perform_comprehensive_seasonal_analysis
)
from .hydrological_year import (
    calculate_hydrological_year_statistics,
    calculate_hydrological_year_statistics_rolling
)
from .time_evolution import (
    calculate_simple_time_evolution_analysis,
    run_time_evolution_analysis,
    run_simple_time_evolution_analysis
)

# TODO: Import functions as they are migrated from original file
# from .hydrological_year import run_hydrological_year_analysis_both_types
# from .yearly_statistics import run_yearly_water_level_statistics

__all__ = [
    'calculate_yearly_max_duration_metrics',
    'calculate_percentile_low_flow_metrics',
    'calculate_yearly_water_level_statistics',
    'perform_segmented_enhanced_seasonal_analysis',
    'perform_comprehensive_seasonal_analysis',
    'calculate_hydrological_year_statistics',
    'calculate_hydrological_year_statistics_rolling',
    'calculate_simple_time_evolution_analysis',
    'run_time_evolution_analysis',
    'run_simple_time_evolution_analysis'
]
