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
import re
import textwrap


def _find_wavelet_source():
    """Locate the monolithic Wavelet_Analysis source file."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(here, '..', '_wavelet_analysis_full.py')),
        os.environ.get('WAVELET_ANALYSIS_FILE', ''),
        r'c:\Users\oluwunmi\Downloads\SCox_update\groundwater_module\src\plotting\Wavelet_Analysis.py',
        r'c:\Users\oluwunmi\Downloads\groundwater_module\src\plotting\Wavelet_Analysis.py',
    ]
    return next((p for p in candidates if p and os.path.exists(p)), None)


def create_wavelet_dashboard(data, well_columns, rainfall_col=None, pumping_col=None, 
                             tide_col=None, remove_rainfall=False, remove_pumping=False, 
                             remove_tide=False, confounder_lags=7, date_range=None):
    """
    Create an interactive wavelet analysis dashboard.

    The dashboard logic lives in the monolithic ``Wavelet_Analysis.py`` source
    (bundled as ``_wavelet_analysis_full.py``). Rather than slicing it, we run
    its module-level imports and the body of ``run_wavelet_analysis`` in a single
    namespace, stopping before the final ``app.run(...)`` call, and return the
    constructed Dash ``app``. This guarantees every helper, local variable, and
    callback resolves exactly as the original author intended.
    """
    source_file = _find_wavelet_source()
    if not source_file:
        raise FileNotFoundError(
            "Wavelet source file not found. Expected "
            "src/plotting/wavelet_analysis/_wavelet_analysis_full.py "
            "or set the WAVELET_ANALYSIS_FILE environment variable."
        )

    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Split into module-level code (imports/constants) and the function body.
    def_idx = next(
        (i for i, line in enumerate(lines) if line.startswith('def run_wavelet_analysis')),
        None,
    )
    if def_idx is None:
        raise ValueError("Could not find run_wavelet_analysis in the wavelet source file.")

    module_top = ''.join(lines[:def_idx])
    body = textwrap.dedent(''.join(lines[def_idx + 1:]))
    # Do not start the server here; run_dashboard.py runs the returned app.
    body = re.sub(r'(?m)^app\.run\b.*$', '', body)

    namespace = {'__name__': 'wavelet_dashboard'}
    # Module-level imports and constants (dash, numpy, PreventUpdate, etc.).
    exec(compile(module_top, source_file, 'exec'), namespace)
    # Provide the function parameters as locals for the body.
    namespace.update({
        'data': data,
        'well_columns': well_columns,
        'rainfall_col': rainfall_col,
        'pumping_col': pumping_col,
        'tide_col': tide_col,
        'remove_rainfall': remove_rainfall,
        'remove_pumping': remove_pumping,
        'remove_tide': remove_tide,
        'confounder_lags': confounder_lags,
        'date_range': date_range,
    })
    # Run the dashboard-building body (builds data_proc, layout, callbacks, app).
    exec(compile(body, source_file, 'exec'), namespace)

    app = namespace.get('app')
    if app is None:
        raise RuntimeError("Wavelet dashboard source did not define an 'app'.")
    return app

