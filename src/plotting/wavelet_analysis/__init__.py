"""
Wavelet Analysis Module
=======================

Wavelet analysis dashboard for groundwater and rainfall data.

Main Functions:
--------------
- run_wavelet_analysis: Main entry point for wavelet analysis dashboard
- get_phase_settings: Safely get phase analysis settings

Constants:
----------
- RESAMPLE_LABELS: Dictionary mapping resample frequencies to labels
- pwc_selections: Global variable for PWC selections
- xwt_selections: Global variable for XWT selections
- phase_analysis_settings: Global variable for phase analysis settings
"""

# Import main function
from .dashboard import run_wavelet_analysis

# Import constants and utilities for backward compatibility
from .constants import (
    RESAMPLE_LABELS,
    pwc_selections,
    xwt_selections,
    phase_analysis_settings,
    DEFAULT_WAVELET_TYPE,
    DEFAULT_WAVELET,
    DEFAULT_SCALE
)
from .utils import get_phase_settings

__all__ = [
    'run_wavelet_analysis',
    'get_phase_settings',
    'RESAMPLE_LABELS',
    'pwc_selections',
    'xwt_selections',
    'phase_analysis_settings',
    'DEFAULT_WAVELET_TYPE',
    'DEFAULT_WAVELET',
    'DEFAULT_SCALE'
]

