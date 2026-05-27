"""
Wavelet Analysis Constants
===========================

Global constants and configuration for wavelet analysis.
"""

RESAMPLE_LABELS = {
    '10T': '10-min',
    '15T': '15-min',
    '30T': '30-min',
    '1H': 'Hourly',
    '1D': 'Daily',
    '1W': 'Weekly',
    '2W': '2-Weekly',
    '1M': 'Monthly',
    '3M': '3-Monthly',
    '6M': '6-Monthly',
    '1Y': 'Yearly',
}

# Global variable to store PWC selections
pwc_selections = {}

# Global variable to store XWT selections  
xwt_selections = {}

# Global variable to store phase analysis settings
# Initialize phase analysis settings globally at module level
phase_analysis_settings = {
    'target_period': 12,
    'show_plots': []
}

# Default wavelet settings
DEFAULT_WAVELET_TYPE = 'cwt'
DEFAULT_WAVELET = 'morl'
DEFAULT_SCALE = 32

