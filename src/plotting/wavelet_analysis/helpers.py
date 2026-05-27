"""
Wavelet Analysis Helper Functions
==================================

Additional helper functions for wavelet analysis dashboard.
"""

import numpy as np
import pandas as pd
from dash import callback_context, no_update


def sync_coherence_series1_with_well(well_value, coh_series1_value):
    """Sync coherence series1 dropdown with well selection."""
    if well_value != coh_series1_value:
        return well_value
    return no_update


def get_auto_interpretation_mode(series1_name, series2_name):
    """
    Automatically determine interpretation mode based on series names.
    
    Parameters:
    -----------
    series1_name : str
        Name of first series
    series2_name : str
        Name of second series
    
    Returns:
    --------
    str
        Interpretation mode string
    """
    series1_lower = series1_name.lower()
    series2_lower = series2_name.lower()
    
    # Check for rainfall combinations
    if 'mm_rain' in [series1_name, series2_name] or 'rain' in series1_lower or 'rain' in series2_lower:
        return 'recharge_discharge'
    
    # Check for tidal combinations
    if 'tide' in series1_lower or 'tide' in series2_lower or 'tidal' in series1_lower or 'tidal' in series2_lower:
        return 'tidal_response'
    
    # Check for pumping combinations
    if 'pump' in series1_lower or 'pump' in series2_lower or 'pumping' in series1_lower or 'pumping' in series2_lower:
        return 'pumping_impact'
    
    # If both are wells (default case)
    return 'hydraulic_connectivity'


def extract_events(mask, x, y, event_type):
    """
    Extracts contiguous events from a boolean mask.
    Returns a list of dicts with start, end, duration, min/max value, and magnitude.
    
    Parameters:
    -----------
    mask : np.ndarray or pd.Series
        Boolean mask indicating event periods
    x : pd.Series or np.ndarray
        Time index
    y : pd.Series or np.ndarray
        Values
    event_type : str
        Type of event (e.g., 'drought', 'flood')
    
    Returns:
    --------
    list
        List of event dictionaries
    """
    events = []
    in_event = False
    for i in range(len(mask)):
        if mask[i] and not in_event:
            in_event = True
            start_idx = i
        elif not mask[i] and in_event:
            in_event = False
            end_idx = i - 1
            event = {
                'Type': event_type,
                'Start': x[start_idx],
                'End': x[end_idx],
                'Duration': (x[end_idx] - x[start_idx]).astype('timedelta64[m]').item() // 60 if hasattr(x[end_idx] - x[start_idx], 'astype') else (x[end_idx] - x[start_idx]).total_seconds() // 3600,
                'Min': float(np.min(y[start_idx:end_idx+1])),
                'Max': float(np.max(y[start_idx:end_idx+1])),
                'Magnitude': float(np.abs(np.max(y[start_idx:end_idx+1]) - np.min(y[start_idx:end_idx+1])))
            }
            events.append(event)
    # Handle if event goes to end
    if in_event:
        end_idx = len(mask) - 1
        event = {
            'Type': event_type,
            'Start': x[start_idx],
            'End': x[end_idx],
            'Duration': (x[end_idx] - x[start_idx]).astype('timedelta64[m]').item() // 60 if hasattr(x[end_idx] - x[start_idx], 'astype') else (x[end_idx] - x[start_idx]).total_seconds() // 3600,
            'Min': float(np.min(y[start_idx:end_idx+1])),
            'Max': float(np.max(y[start_idx:end_idx+1])),
            'Magnitude': float(np.abs(np.max(y[start_idx:end_idx+1]) - np.min(y[start_idx:end_idx+1])))
        }
        events.append(event)
    return events


def event_stats(events):
    """
    Calculate statistics for a list of events.
    
    Parameters:
    -----------
    events : list
        List of event dictionaries
    
    Returns:
    --------
    dict
        Dictionary of event statistics
    """
    if not events:
        return {}
    durations = [e['Duration'] for e in events]
    magnitudes = [e['Magnitude'] for e in events]
    return {
        'Count': len(events),
        'Mean Duration (h)': np.mean(durations),
        'Mean Magnitude': np.mean(magnitudes),
        'Max Magnitude': np.max(magnitudes),
        'Min Magnitude': np.min(magnitudes)
    }

