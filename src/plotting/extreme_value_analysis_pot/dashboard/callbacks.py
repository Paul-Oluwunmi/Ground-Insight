"""
POT Dashboard Callbacks
=======================

Dash callbacks for the POT analysis dashboard.
This file contains all callback functions extracted from the original dashboard.

NOTE: The main callback `update_eva_analysis` is very large (~3000 lines).
It will be extracted in parts due to size constraints.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import logging
import textwrap
import traceback

from ..constants import SEASON_COLORS
from ..data_processing import find_peaks_data, calculate_exceedances
from ..statistical_calculations import (
    calculate_return_periods,
    calculate_confidence_intervals,
    fit_gpd_by_season,
    calculate_return_levels_by_season,
    fit_gpd_with_trend,
    calculate_return_levels_trend,
    fit_gpd_by_covariate,
    calculate_return_levels_covariate
)
from ..bayesian_analysis import (
    perform_bayesian_gpd_stationary,
    perform_bayesian_gpd_non_stationary
)
from ..plotting import (
    update_eva_plots,
    create_full_eva_figure,
    create_corner_plot_gpd,
    generate_posterior_predictive_checks_gpd
)
from .helpers import create_time_series_plot, calculate_threshold


def _run_eva_modular(data, threshold_method, threshold_value, data_type, location,
                     sampling_interval, is_drought, gpd_model_type, covariate_selector=None):
    """
    Build EVA main plot and related outputs using modular functions (no backup file).
    Returns (main_fig, eva_style, corner_fig, ppc_fig, ppc_table, bayesian_style).
    """
    empty = go.Figure(), {'display': 'none'}, go.Figure(), go.Figure(), [], {'display': 'none'}
    if threshold_value is None:
        return empty
    try:
        # Single location for EVA
        loc = location[0] if isinstance(location, list) and location else location
        if data_type == 'gwl' and not loc:
            return empty
        column_to_analyse = 'mm_Rain' if data_type == 'rain' else loc
        if data_type == 'rain':
            if 'mm_Rain' not in data.columns:
                return empty
            if sampling_interval != 'original':
                work = (data[['DateTime', 'mm_Rain']].set_index('DateTime')
                        .resample(sampling_interval).sum().reset_index())
            else:
                work = data[['DateTime', 'mm_Rain']].copy()
            series = work['mm_Rain']
        else:
            if loc not in data.columns:
                return empty
            if sampling_interval != 'original':
                work = (data[['DateTime', loc]].set_index('DateTime')
                        .resample(sampling_interval).mean().reset_index())
            else:
                work = data[['DateTime', loc]].copy()
            series = work.set_index('DateTime')[loc]
        threshold = calculate_threshold(
            series, threshold_method, threshold_value, sampling_interval, data_type, is_drought
        )
        peaks_data = find_peaks_data(
            work, threshold, data_type, column=None if data_type == 'rain' else loc, is_drought=is_drought
        )
        if peaks_data is None or len(peaks_data) < 2:
            return go.Figure(), {'display': 'block'}, go.Figure(), go.Figure(), [], {'display': 'none'}
        # Add Season column (expected by update_eva_plots); find_peaks_data does not add it
        if 'Season' not in peaks_data.columns and 'DateTime' in peaks_data.columns:
            peaks_data = peaks_data.copy()
            peaks_data['Season'] = peaks_data['DateTime'].dt.month.map({
                12: 'Summer', 1: 'Summer', 2: 'Summer',
                3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
                6: 'Winter', 7: 'Winter', 8: 'Winter',
                9: 'Spring', 10: 'Spring', 11: 'Spring'
            })
        exceedances = calculate_exceedances(peaks_data, threshold, data_type, is_drought=is_drought)
        main_fig = create_full_eva_figure(
            exceedances, data_type, peaks_data=peaks_data, threshold=threshold,
            column_to_analyse=column_to_analyse, is_drought=is_drought
        )
        if main_fig is None:
            return empty
        return (main_fig, {'display': 'block'}, go.Figure(), go.Figure(), [], {'display': 'none'})
    except Exception as e:
        logging.error(f"Error in modular EVA callback: {str(e)}\n{traceback.format_exc()}")
        return empty


def register_callbacks(app, data, location_columns):
    """
    Register all callbacks for the POT dashboard.
    
    Parameters:
    -----------
    app : dash.Dash
        Dash application instance
    data : pd.DataFrame
        Input data containing DateTime and location columns
    location_columns : list
        List of location column names
    """
    
    # Callback to show/hide drought toggle
    @app.callback(
        Output('drought-toggle-container', 'style'),
        Input('data-type', 'value')
    )
    def toggle_drought_visibility(data_type):
        """Show/hide drought toggle based on data type"""
        if data_type == 'gwl':
            return {'display': 'inline-flex', 'alignItems': 'center', 'marginLeft': '20px'}
        return {'display': 'none'}

    # Time series callback — same Inputs as main EVA callback (below) so both update together
    # when user changes Data Type, Location, Threshold, Sampling interval, or Drought toggle.
    @app.callback(
        Output('time-series-plot', 'figure'),
        [Input('data-type', 'value'),
         Input('location-selector', 'value'),
         Input('threshold-method', 'value'),
         Input('threshold-value', 'value'),
         Input('sampling-interval', 'value'),
         Input('drought-toggle', 'value')]  
    )
    def update_time_series(data_type, locations, threshold_method, threshold_value, sampling_interval, is_drought):
        """Update time series plot (connected to POT Analysis via shared Inputs)."""
        try:
            return create_time_series_plot(
                data_type, 
                locations, 
                threshold_method, 
                threshold_value, 
                sampling_interval,
                data,
                is_drought,
                calculate_threshold
            )
        
        except Exception as e:
            logging.error(f"Error in time series plot: {str(e)}")
            return go.Figure()

    # Callback for threshold guidance
    @app.callback(
        [Output('threshold-help-text', 'children'),
         Output('threshold-range-text', 'children'),
         Output('threshold-value', 'min'),
         Output('threshold-value', 'max'),
         Output('threshold-value', 'step')],
        [Input('threshold-method', 'value'),
         Input('data-type', 'value')]
    )
    def update_threshold_guidance(method, data_type):
        """Update help text and input constraints based on selected method"""
        if method == 'percentile':
            help_text = """
            Enter a percentile value between 0 and 100.
            - Common values: 90-99 for high extremes, 1-10 for low extremes
            - Example: 95 means the threshold will be at the 95th percentile
            """
            range_text = "Valid range: 0 to 100"
            min_val = 0
            max_val = 100
            step = 0.1
        elif method == 'fixed':
            if data_type == 'gwl':
                help_text = """
                Enter the actual groundwater level threshold in meters.
                - Use a value within the range of your data
                - Based on physical or regulatory requirements
                """
                range_text = "Enter actual groundwater level (m)"
            else:
                help_text = """
                Enter the actual rainfall threshold in millimeters.
                - Use a value within the range of your data
                - Based on physical or regulatory requirements
                """
                range_text = "Enter actual rainfall amount (mm)"
            min_val = None
            max_val = None
            step = 0.01
        elif method == 'mean_std':
            help_text = """
            Enter the number of standard deviations.
            - Typical values: 1 to 3
            - Example: 2 means threshold = mean + (2 × standard deviation)
            """
            range_text = "Recommended range: 1 to 3"
            min_val = 0
            max_val = 5
            step = 0.1
        elif method == 'moving_avg':
            help_text = """
            Enter the number of standard deviations.
            - Uses a 30-day moving average as the baseline
            - Adds N standard deviations to detect extremes
            """
            range_text = "Recommended range: 1 to 3"
            min_val = 0
            max_val = 5
            step = 0.1
        elif method == 'seasonal_percentile':
            help_text = """
            Enter the percentile (0-100).
            - Calculates separate thresholds for each season
            - Accounts for seasonal variations in the data
            """
            range_text = "Valid range: 0 to 100"
            min_val = 0
            max_val = 100
            step = 0.1
        elif method == 'pot':
            help_text = """
            Enter the average number of peaks per year.
            - Typical values: 3 to 5 events per year
            - Threshold will be adjusted to achieve this frequency
            """
            range_text = "Recommended range: 1 to 10"
            min_val = 1
            max_val = 10
            step = 0.5
        elif method == 'declustering':
            help_text = """
            Enter the minimum time between independent events (days).
            - Helps identify independent extreme events
            - Typical values: 3-7 days for rainfall, 7-14 days for GWL
            """
            range_text = "Enter separation time in days"
            min_val = 1
            max_val = 30
            step = 1
        else:  # min_annual_max
            help_text = """
            Enter the number of years for baseline.
            - Uses minimum of annual maxima as threshold
            - Recommended: 5-10 years of data minimum
            """
            range_text = "Enter number of years"
            min_val = 1
            max_val = 50
            step = 1

        return html.Pre(help_text, style={'whiteSpace': 'pre-wrap'}), range_text, min_val, max_val, step

    # Callback for location selector state
    @app.callback(
        Output('location-selector', 'disabled'),
        Input('data-type', 'value')
    )
    def update_location_selector_state(data_type):
        """Enable/disable location selector based on data type"""
        return data_type == 'rain'

    # Callback for initial location
    @app.callback(
        Output('location-selector', 'value'),
        [Input('data-type', 'value')],
        [State('location-selector', 'options')]
    )
    def set_initial_location(data_type, options):
        """Set initial location value based on data type"""
        if data_type == 'rain':
            return None
        elif options:
            return options[0]['value']
        return None

    # Callback for covariate selector visibility
    @app.callback(
        Output('covariate-selector-container', 'style'),
        Input('gpd-model-type', 'value')
    )
    def toggle_covariate_selector(gpd_model_type):
        """Show/hide covariate selector based on model type"""
        if gpd_model_type == 'covariate':
            return {'display': 'block', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'flex': '1', 'marginRight': '15px'}
        return {'display': 'none'}

    # Callback for covariate options
    @app.callback(
        Output('covariate-selector', 'options'),
        [Input('data-type', 'value')]
    )
    def update_covariate_options(data_type):
        """Update covariate options based on available columns"""
        # Use all numeric columns except DateTime and main value column
        cols = [col for col in data.columns if col not in ['DateTime', 'mm_Rain'] and np.issubdtype(data[col].dtype, np.number)]
        return [{'label': col, 'value': col} for col in cols]

    # Main EVA analysis callback — uses same Inputs as time series (threshold-method, threshold-value,
    # data-type, location-selector, sampling-interval, drought-toggle) so POT Analysis plots stay
    # in sync with the time series when any control changes. Adds gpd-model-type and covariate (State).
    # This is very large (~3000 lines) in the original; we load from backup or use modular fallback.
    import os
    _dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    _plotting_dir = os.path.dirname(os.path.dirname(_dashboard_dir))
    _backup_candidates = [
        os.path.join(_dashboard_dir, 'Extreme_Value_Analysis_POT_backup.py'),
        os.path.join(_plotting_dir, 'Extreme_Value_Analysis_POT_backup.py'),
        os.path.join(_plotting_dir, 'Extreme_Value_Analysis_POT.py'),
        os.path.join(os.path.expanduser('~'), 'Downloads', 'groundwater_module', 'src', 'plotting', 'Extreme_Value_Analysis_POT.py'),
    ]
    _backup_file = None
    for _candidate in _backup_candidates:
        if os.path.exists(_candidate):
            _backup_file = _candidate
            break
    import importlib.util
    from plotly.subplots import make_subplots
    from scipy.stats import genpareto
    from scipy import stats
    
    # Import COLORS from constants
    from ..constants import COLORS
    
    # In the full original file, def update_eva_analysis starts at line 1653, ends at 4701 (1-based).
    # We extract only the function (not @app.callback) so exec doesn't need 'app' in namespace.
    _CALLBACK_START = 1652   # 0-based index (line 1653 in file: def update_eva_analysis)
    _CALLBACK_END = 4702     # 0-based exclusive (through line 4701)
    
    if _backup_file and os.path.exists(_backup_file):
        # Read backup file and extract callback function
        with open(_backup_file, 'r', encoding='utf-8') as f:
            backup_lines = f.readlines()
        # Only use backup if file is long enough (full original ~5977 lines; stub ~40)
        if len(backup_lines) >= _CALLBACK_END:
            callback_lines = backup_lines[_CALLBACK_START:_CALLBACK_END]  # def update_eva_analysis ... end of function
        else:
            callback_lines = []
        callback_code = ''.join(callback_lines) if callback_lines else ''
        if callback_code:
            callback_code = textwrap.dedent(callback_code)  # Remove leading indent so exec() works
        
        # Create namespace with our modular imports
        callback_namespace = {
            'go': go,
            'pd': pd,
            'np': np,
            'make_subplots': make_subplots,
            'genpareto': genpareto,
            'stats': stats,
            'logging': logging,
            'traceback': traceback,
            'SEASON_COLORS': SEASON_COLORS,
            'COLORS': COLORS,
            'find_peaks_data': find_peaks_data,
            'calculate_exceedances': calculate_exceedances,
            'calculate_return_periods': calculate_return_periods,
            'calculate_confidence_intervals': calculate_confidence_intervals,
            'fit_gpd_by_season': fit_gpd_by_season,
            'calculate_return_levels_by_season': calculate_return_levels_by_season,
            'fit_gpd_with_trend': fit_gpd_with_trend,
            'calculate_return_levels_trend': calculate_return_levels_trend,
            'fit_gpd_by_covariate': fit_gpd_by_covariate,
            'calculate_return_levels_covariate': calculate_return_levels_covariate,
            'perform_bayesian_gpd_stationary': perform_bayesian_gpd_stationary,
            'perform_bayesian_gpd_non_stationary': perform_bayesian_gpd_non_stationary,
            'create_corner_plot_gpd': create_corner_plot_gpd,
            'generate_posterior_predictive_checks_gpd': generate_posterior_predictive_checks_gpd,
            'calculate_threshold': calculate_threshold,
            'data': data,
        }
        
        # Fix covariate_selector issue: add it as a parameter to the callback function
        # Replace the function signature to include covariate_selector
        callback_code = callback_code.replace(
            'def update_eva_analysis(threshold_method, threshold_value, data_type, location, \n                          sampling_interval, is_drought, gpd_model_type):',
            'def update_eva_analysis(threshold_method, threshold_value, data_type, location, \n                          sampling_interval, is_drought, gpd_model_type, covariate_selector=None):'
        )
        
        # Execute callback code to create the function
        try:
            exec(callback_code, callback_namespace)
            callback_func = callback_namespace.get('update_eva_analysis')
        except Exception as e:
            logging.error(f"Error executing callback code: {str(e)}\n{traceback.format_exc()}")
            callback_func = None
        
        # Register the callback
        if callback_func:
            @app.callback(
                [Output('main-plot', 'figure'),
                 Output('eva-content', 'style'),
                 Output('corner-plot-gpd', 'figure'),
                 Output('ppc-ecdf-plot-gpd', 'figure'),
                 Output('ppc-summary-table-gpd', 'data'),
                 Output('bayesian-plots-container-gpd', 'style')],
                [Input('threshold-method', 'value'),
                 Input('threshold-value', 'value'),
                 Input('data-type', 'value'),
                 Input('location-selector', 'value'),
                 Input('sampling-interval', 'value'),
                 Input('drought-toggle', 'value'),
                 Input('gpd-model-type', 'value')],
                [State('covariate-selector', 'value')]  # Add covariate_selector as State
            )
            def update_eva_analysis(threshold_method, threshold_value, data_type, location, 
                                  sampling_interval, is_drought, gpd_model_type, covariate_selector=None):
                """Main EVA analysis callback - extracted from backup with modular imports"""
                return callback_func(
                    threshold_method, threshold_value, data_type, location,
                    sampling_interval, is_drought, gpd_model_type, covariate_selector
                )
        else:
            # Fallback: use modular EVA when extraction failed
            @app.callback(
                [Output('main-plot', 'figure'),
                 Output('eva-content', 'style'),
                 Output('corner-plot-gpd', 'figure'),
                 Output('ppc-ecdf-plot-gpd', 'figure'),
                 Output('ppc-summary-table-gpd', 'data'),
                 Output('bayesian-plots-container-gpd', 'style')],
                [Input('threshold-method', 'value'),
                 Input('threshold-value', 'value'),
                 Input('data-type', 'value'),
                 Input('location-selector', 'value'),
                 Input('sampling-interval', 'value'),
                 Input('drought-toggle', 'value'),
                 Input('gpd-model-type', 'value')],
                [State('covariate-selector', 'value')]
            )
            def update_eva_analysis(threshold_method, threshold_value, data_type, location, 
                                  sampling_interval, is_drought, gpd_model_type, covariate_selector=None):
                """Modular EVA when callback extraction failed"""
                return _run_eva_modular(
                    data, threshold_method, threshold_value, data_type, location,
                    sampling_interval, is_drought, gpd_model_type, covariate_selector
                )
    else:
        # Fallback if backup file doesn't exist: use modular EVA so plots still load
        @app.callback(
            [Output('main-plot', 'figure'),
             Output('eva-content', 'style'),
             Output('corner-plot-gpd', 'figure'),
             Output('ppc-ecdf-plot-gpd', 'figure'),
             Output('ppc-summary-table-gpd', 'data'),
             Output('bayesian-plots-container-gpd', 'style')],
            [Input('threshold-method', 'value'),
             Input('threshold-value', 'value'),
             Input('data-type', 'value'),
             Input('location-selector', 'value'),
             Input('sampling-interval', 'value'),
             Input('drought-toggle', 'value'),
             Input('gpd-model-type', 'value')]
        )
        def update_eva_analysis(threshold_method, threshold_value, data_type, location, 
                              sampling_interval, is_drought, gpd_model_type):
            """Modular EVA callback when backup file not found"""
            return _run_eva_modular(
                data, threshold_method, threshold_value, data_type, location,
                sampling_interval, is_drought, gpd_model_type, None
            )

