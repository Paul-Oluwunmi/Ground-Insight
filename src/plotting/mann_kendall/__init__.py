"""
Mann-Kendall Analysis Module

This module has been refactored into a modular structure:
- constants.py: Configuration parameters
- utils.py: Helper functions (dates, seasons, rolling windows)
- core_tests.py: Core Mann-Kendall test implementations
- batch_analysis.py: Batch processing functions
- analysis/: Analysis modules (seasonal, time_evolution, hydrological_year, duration_low_flow, yearly_statistics)
- plotting/: Plotting functions (individual_plots, batch_plots)
- export/: Export functions (csv_export, excel_export)
- dashboard.py: Main dashboard function
"""

# Import constants
from .constants import (
    USE_ROLLING_MEAN,
    ROLLING_WINDOW_SIZE,
    ALPHA_LEVEL,
    RESAMPLE_FREQUENCY,
    CUSTOM_SAVE_LOCATION,
    CREATE_FOLDERS,
    SAVE_PLOTS,
    RUN_ORIGINAL_MK,
    RUN_SEASONAL_MK,
    RUN_TIME_EVOLUTION,
    RUN_HYDROLOGICAL_YEAR,
    RUN_DURATION_LOW_FLOW,
    RUN_YEARLY_STATISTICS,
    TIME_EVOLUTION_WINDOW_SIZE,
    TIME_EVOLUTION_STEP_SIZE,
    TIME_EVOLUTION_OVERLAP,
    TIME_EVOLUTION_MIN_POINTS,
    IGNORE_ZERO_VALUES_IN_TREND
)

# Import utility functions
from .utils import (
    safe_calculate_years_from_dates,
    safe_calculate_duration_days,
    calculate_strict_rolling_window,
    get_nz_season,
    check_rolling_mean_compatibility,
    robust_callback,
    convert_slope_to_mm_per_year,
    format_slope_mm,
    generate_well_filename,
    generate_well_folder_name,
    generate_well_display_name,
    get_well_folder_name_consistent,
    ensure_datetime_index,
    handle_timestamp_interval_errors,
    safe_check_data_structure,
    safe_analyze_timestamp_intervals,
    detect_data_format,
    assess_data_quality_for_mk
)

# Import core test functions
from .core_tests import (
    mann_kendall_test,
    seasonal_mann_kendall_test
)

# Import batch analysis functions
from .batch_analysis import (
    run_mann_kendall_analysis
)

# Import export functions
from .export import (
    save_basic_mk_analysis_csv,
    save_duration_analysis_csv,
    save_yearly_water_level_statistics_csv,
    save_low_flow_analysis_csv,
    create_all_wells_duration_analysis_csv,
    create_all_wells_yearly_water_level_statistics_csv,
    create_all_wells_low_flow_analysis_csv,
    save_simple_time_evolution_analysis_csv,
    create_analysis_summary_excel,
    update_analysis_status_excel
)

# Import analysis functions
from .analysis import (
    calculate_yearly_max_duration_metrics,
    calculate_percentile_low_flow_metrics,
    calculate_yearly_water_level_statistics,
    perform_segmented_enhanced_seasonal_analysis,
    perform_comprehensive_seasonal_analysis,
    calculate_hydrological_year_statistics,
    calculate_hydrological_year_statistics_rolling,
    calculate_simple_time_evolution_analysis,
    run_time_evolution_analysis,
    run_simple_time_evolution_analysis
)

# Import plotting functions
from .plotting import (
    create_gwl_rolling_mean_plot,
    create_seasonal_analysis_plot,
    create_mk_trend_plot,
    create_duration_analysis_plot,
    create_low_flow_analysis_plot,
    create_gwl_rolling_mean_plot_simple,
    create_individual_well_plots,
    run_individual_well_plotting,
    run_gwl_rolling_mean_plots
)

# Import batch wrapper functions - use try/except to handle import errors
try:
    from .Batch_Mann_Kendall import (
        run_yearly_water_level_statistics,
        run_hydrological_year_analysis_both_types,
        run_comprehensive_original_mk_batch_analysis_separate_files,
        run_comprehensive_seasonal_mk_batch_analysis_separate_files
    )
except (ImportError, ValueError) as e:
    # If import fails, set to None - functions will be imported directly when needed
    run_yearly_water_level_statistics = None
    run_hydrological_year_analysis_both_types = None
    run_comprehensive_original_mk_batch_analysis_separate_files = None
    run_comprehensive_seasonal_mk_batch_analysis_separate_files = None

# Import dashboard function
from .dashboard import run_mann_kendall_dashboard

# Note: Other functions will be imported as they are migrated from the original file
# For now, they remain in the original mann_kendall.py file which acts as a backward compatibility wrapper

__all__ = [
    # Constants
    'USE_ROLLING_MEAN',
    'ROLLING_WINDOW_SIZE',
    'ALPHA_LEVEL',
    'RESAMPLE_FREQUENCY',
    'CUSTOM_SAVE_LOCATION',
    'CREATE_FOLDERS',
    'SAVE_PLOTS',
    'RUN_ORIGINAL_MK',
    'RUN_SEASONAL_MK',
    'RUN_TIME_EVOLUTION',
    'RUN_HYDROLOGICAL_YEAR',
    'RUN_DURATION_LOW_FLOW',
    'RUN_YEARLY_STATISTICS',
    'TIME_EVOLUTION_WINDOW_SIZE',
    'TIME_EVOLUTION_STEP_SIZE',
    'TIME_EVOLUTION_OVERLAP',
    'TIME_EVOLUTION_MIN_POINTS',
    'IGNORE_ZERO_VALUES_IN_TREND',
    # Utility functions
    'safe_calculate_years_from_dates',
    'safe_calculate_duration_days',
    'calculate_strict_rolling_window',
    'get_nz_season',
    'check_rolling_mean_compatibility',
    'robust_callback',
    'convert_slope_to_mm_per_year',
    'format_slope_mm',
    'generate_well_filename',
    'generate_well_folder_name',
    'generate_well_display_name',
    'get_well_folder_name_consistent',
    'ensure_datetime_index',
    'handle_timestamp_interval_errors',
    'safe_check_data_structure',
    'safe_analyze_timestamp_intervals',
    'detect_data_format',
    'assess_data_quality_for_mk',
    'check_well_data_availability',
    'display_well_data_availability',
    # Core tests
    'mann_kendall_test',
    'seasonal_mann_kendall_test',
    # Batch analysis
    'run_mann_kendall_analysis',
    # Export functions
    'save_basic_mk_analysis_csv',
    'save_duration_analysis_csv',
    'save_yearly_water_level_statistics_csv',
    'save_low_flow_analysis_csv',
    'create_all_wells_duration_analysis_csv',
    'create_all_wells_yearly_water_level_statistics_csv',
    'create_all_wells_low_flow_analysis_csv',
    'save_simple_time_evolution_analysis_csv',
    'create_analysis_summary_excel',
    'update_analysis_status_excel',
    # Analysis functions
    'calculate_yearly_max_duration_metrics',
    'calculate_percentile_low_flow_metrics',
    'calculate_yearly_water_level_statistics',
    'perform_segmented_enhanced_seasonal_analysis',
    'perform_comprehensive_seasonal_analysis',
    'calculate_hydrological_year_statistics',
    'calculate_hydrological_year_statistics_rolling',
    'calculate_simple_time_evolution_analysis',
    'run_time_evolution_analysis',
    'run_simple_time_evolution_analysis',
    # Plotting functions
    'create_gwl_rolling_mean_plot',
    'create_seasonal_analysis_plot',
    'create_mk_trend_plot',
    'create_duration_analysis_plot',
    'create_low_flow_analysis_plot',
    'create_gwl_rolling_mean_plot_simple',
    'create_individual_well_plots',
    'run_individual_well_plotting',
    'run_gwl_rolling_mean_plots',
    # Batch wrapper functions
    'run_yearly_water_level_statistics',
    'run_hydrological_year_analysis_both_types',
    'run_comprehensive_original_mk_batch_analysis_separate_files',
    'run_comprehensive_seasonal_mk_batch_analysis_separate_files',
    # Dashboard function
    'run_mann_kendall_dashboard',
]
