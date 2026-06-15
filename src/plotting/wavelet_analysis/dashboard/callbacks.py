"""
Wavelet Analysis Dashboard Callbacks
=====================================

Dash callbacks for the wavelet analysis dashboard.
This file extracts callbacks from the original Wavelet_Analysis.py file.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
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
import scipy
import statsmodels.api as sm
import traceback
import os
import logging

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
from ..helpers import (
    sync_coherence_series1_with_well, get_auto_interpretation_mode,
    extract_events, event_stats
)


def register_callbacks(app, data, well_columns, rainfall_col=None, pumping_col=None, tide_col=None):
    """
    Register all callbacks for the wavelet analysis dashboard.
    
    Parameters:
    -----------
    app : dash.Dash
        The Dash application instance.
    data : pd.DataFrame
        The main dataset.
    well_columns : list
        List of well column names.
    rainfall_col : str or None
        Column name for rainfall data.
    pumping_col : str or None
        Column name for pumping data.
    tide_col : str or None
        Column name for tidal data.
    """
    # Extract callbacks from the monolithic source (lines 712-end).
    # Prefer the copy bundled in this package; fall back to env var / legacy paths.
    _here = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.normpath(os.path.join(_here, '..', '_wavelet_analysis_full.py')),
        os.environ.get('WAVELET_ANALYSIS_FILE', ''),
        r'c:\Users\oluwunmi\Downloads\SCox_update\groundwater_module\src\plotting\Wavelet_Analysis.py',
        r'c:\Users\oluwunmi\Downloads\groundwater_module\src\plotting\Wavelet_Analysis.py',
    ]
    _original_file = next((p for p in _candidates if p and os.path.exists(p)), None)
    
    if _original_file:
        # Read original file
        with open(_original_file, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()
        
        # Extract callback code from the function body (from line 712 onward).
        # The source lives inside run_wavelet_analysis(), so the block is indented;
        # dedent it to valid top-level code and drop the trailing app.run() call
        # (the app is created and served elsewhere).
        import textwrap, re
        callback_lines = original_lines[711:]
        callback_code = textwrap.dedent(''.join(callback_lines))
        callback_code = re.sub(r'(?m)^app\.run\b.*$', '', callback_code)
        
        # Create namespace with modular imports
        callback_namespace = {
            'app': app,
            'dash': dash,
            'dcc': dcc,
            'html': html,
            'Input': Input,
            'Output': Output,
            'State': State,
            'callback_context': callback_context,
            'go': go,
            'np': np,
            'pywt': pywt,
            'pd': pd,
            'datetime': datetime,
            'scipy': scipy.stats,  # Use scipy.stats as scipy
            'scipy.ndimage': scipy.ndimage,
            'scipy.signal': scipy_signal,
            'scipy.stats': scipy.stats,
            'functools': functools,
            'io': io,
            'pio': pio,
            'base64': base64,
            'interp1d': interp1d,
            'uniform_filter1d': uniform_filter1d,
            'find_peaks': find_peaks,
            'sm': sm,
            'traceback': traceback,
            'logging': logging,
            'RESAMPLE_LABELS': RESAMPLE_LABELS,
            'pwc_selections': pwc_selections,
            'xwt_selections': xwt_selections,
            'phase_analysis_settings': phase_analysis_settings,
            'get_phase_settings': get_phase_settings,
            'get_flat_wavelet_options': get_flat_wavelet_options,
            'summary_to_dash_table': summary_to_dash_table,
            'get_nz_season_vrects': get_nz_season_vrects,
            'get_nz_season_for_dates': get_nz_season_for_dates,
            'remove_confounders': remove_confounders,
            'compute_wavelet_cwt': compute_wavelet_cwt,
            'cached_data_resample': cached_data_resample,
            'cached_percentile_calculation': cached_percentile_calculation,
            'cached_coherence_computation': cached_coherence_computation,
            'cached_wavelet_periods': cached_wavelet_periods,
            'sync_coherence_series1_with_well': sync_coherence_series1_with_well,
            'no_update': dash.no_update,
            'get_auto_interpretation_mode': get_auto_interpretation_mode,
            'extract_events': extract_events,
            'event_stats': event_stats,
            'data': data,
            'well_columns': well_columns,
            'rainfall_col': rainfall_col,
            'pumping_col': pumping_col,
            'tide_col': tide_col,
            'DEFAULT_WAVELET_TYPE': DEFAULT_WAVELET_TYPE,
            'DEFAULT_WAVELET': DEFAULT_WAVELET,
            'DEFAULT_SCALE': DEFAULT_SCALE,
        }
        
        # Execute callback code
        try:
            exec(callback_code, callback_namespace)
        except Exception as e:
            logging.error(f"Error executing callback code: {str(e)}\n{traceback.format_exc()}")
            raise
    else:
        _searched = ', '.join(p for p in _candidates if p)
        logging.error(f"Wavelet source file not found - callbacks cannot be loaded. Searched: {_searched}")
        raise FileNotFoundError(f"Wavelet source file not found. Searched: {_searched}")

