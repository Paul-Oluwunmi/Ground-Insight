"""
Wavelet Analysis Module - Backward Compatibility Wrapper
========================================================

This file has been refactored into a modular structure in the wavelet_analysis/ folder.
This wrapper maintains backward compatibility for existing imports.

NEW USAGE (Recommended):
    from src.plotting.wavelet_analysis import run_wavelet_analysis

OLD USAGE (Still works):
    from src.plotting.Wavelet_Analysis import run_wavelet_analysis

The modular structure includes:
- wavelet_analysis/constants.py: Constants and default settings
- wavelet_analysis/utils.py: Utility functions
- wavelet_analysis/dashboard/: Dashboard layout and callbacks
"""

# Import from the modular package for backward compatibility
from .wavelet_analysis import (
    run_wavelet_analysis,
    get_phase_settings,
    RESAMPLE_LABELS,
    pwc_selections,
    xwt_selections,
    phase_analysis_settings,
    DEFAULT_WAVELET_TYPE,
    DEFAULT_WAVELET,
    DEFAULT_SCALE
)

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
