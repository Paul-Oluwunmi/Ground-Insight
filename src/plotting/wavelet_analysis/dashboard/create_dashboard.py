"""
Wavelet Analysis Dashboard Creation
====================================

Creates the wavelet analysis dashboard with layout and callbacks.
This file imports helper functions from the modular structure.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pywt
import pandas as pd
import datetime
import scipy.ndimage
import functools
import io
import plotly.io as pio
import base64
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy import signal as scipy_signal
import scipy.stats
import scipy.signal
import statsmodels.api as sm
import traceback
import os
import importlib.util

# Import from modular structure
from ..constants import (
    RESAMPLE_LABELS, pwc_selections, xwt_selections, 
    phase_analysis_settings, DEFAULT_WAVELET_TYPE, DEFAULT_WAVELET, DEFAULT_SCALE
)
from ..utils import (
    get_phase_settings, get_flat_wavelet_options, summary_to_dash_table,
    get_nz_season_vrects, get_nz_season_for_dates
)
from ..data_processing import (
    remove_confounders, compute_wavelet_cwt, cached_data_resample,
    cached_percentile_calculation, cached_coherence_computation,
    cached_wavelet_periods
)
from .layout import create_layout
from .callbacks import register_callbacks


def create_wavelet_dashboard(data, well_columns, rainfall_col=None, pumping_col=None, 
                             tide_col=None, remove_rainfall=False, remove_pumping=False, 
                             remove_tide=False, confounder_lags=7, date_range=None):
    """
    Create an interactive wavelet analysis dashboard.
    
    This function uses the modular layout and callbacks extracted from the original file.
    """
    # Process data (same as original)
    data_proc = data.copy()
    if date_range is not None:
        start_date, end_date = date_range
        data_proc['DateTime'] = pd.to_datetime(data_proc['DateTime'])
        mask = (data_proc['DateTime'] >= pd.to_datetime(start_date)) & (data_proc['DateTime'] <= pd.to_datetime(end_date))
        data_proc = data_proc.loc[mask].reset_index(drop=True)
    
    # Align all time series on the same time index
    align_cols = ['DateTime'] + well_columns
    
    # Auto-detect rainfall column if not specified
    if rainfall_col is None and 'mm_Rain' in data_proc.columns:
        rainfall_col = 'mm_Rain'
    
    if rainfall_col and rainfall_col in data_proc.columns:
        align_cols.append(rainfall_col)
    if pumping_col and pumping_col in data_proc.columns:
        align_cols.append(pumping_col)
    if tide_col and tide_col in data_proc.columns:
        align_cols.append(tide_col)
    
    data_proc = data_proc[align_cols]
    # Interpolate missing values (linear)
    data_proc = data_proc.sort_values('DateTime').reset_index(drop=True)
    data_proc = data_proc.interpolate(method='linear', limit_direction='both')
    
    # Remove selected confounders for each well
    for well in well_columns:
        conf_cols = []
        if remove_rainfall and rainfall_col and rainfall_col in data_proc.columns:
            conf_cols.append(rainfall_col)
        if remove_pumping and pumping_col and pumping_col in data_proc.columns:
            conf_cols.append(pumping_col)
        if remove_tide and tide_col and tide_col in data_proc.columns:
            conf_cols.append(tide_col)
        if conf_cols:
            data_proc[well] = remove_confounders(data_proc, well, conf_cols, lags=confounder_lags)
    
    # Prepare options
    well_options = [{'label': w, 'value': w} for w in well_columns]
    data_type_options = [
        {'label': 'Groundwater Level', 'value': 'gwl'},
        {'label': 'Rainfall', 'value': 'rain'}
    ]
    wavelet_type_options = [
        {'label': 'Continuous (CWT)', 'value': 'cwt'},
        {'label': 'Discrete (DWT)', 'value': 'dwt'},
        {'label': 'Stationary (SWT)', 'value': 'swt'}
    ]
    
    # Create Dash app
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    # Set the layout
    app.layout = create_layout(well_options, data_type_options, wavelet_type_options, data_proc)
    
    # Register callbacks
    register_callbacks(app, data_proc, well_columns, rainfall_col, pumping_col, tide_col)
    
    return app

