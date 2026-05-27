"""
POT Data Processing
===================

Functions for finding peaks and calculating exceedances from threshold.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks
import logging


def find_peaks_data(data, threshold, data_type, column=None, is_drought=False):
    """Find peaks in the data based on threshold
    
    Args:
        data: DataFrame with DateTime and values
        threshold: float or Series, threshold value(s)
        data_type: str, either 'gwl' or 'rain'
        column: str, column name for GWL data
        is_drought: bool, whether to analyze drought conditions
    
    Returns:
        DataFrame with peaks data
    """
    try:
        # Select appropriate column
        if data_type == 'rain':
            values = data['mm_Rain'].values
            dates = data['DateTime']
        else:
            values = data[column].values
            dates = data['DateTime']

        # Find peaks based on drought/non-drought analysis
        if isinstance(threshold, pd.Series):
            if is_drought:
                # For drought analysis
                peak_mask = values < threshold.values
                if any(peak_mask):
                    masked_peaks = scipy_find_peaks(
                        -values[peak_mask],
                        height=-threshold.values[peak_mask]
                    )[0]
                    if len(masked_peaks) > 0:
                        peak_indices = np.where(peak_mask)[0][masked_peaks]
                    else:
                        peak_indices = np.array([], dtype=int)
                else:
                    peak_indices = np.array([], dtype=int)
            else:
                # For high extremes
                peak_mask = values > threshold.values
                if any(peak_mask):
                    masked_peaks = scipy_find_peaks(
                        values[peak_mask],
                        height=threshold.values[peak_mask]
                    )[0]
                    if len(masked_peaks) > 0:
                        peak_indices = np.where(peak_mask)[0][masked_peaks]
                    else:
                        peak_indices = np.array([], dtype=int)
                else:
                    peak_indices = np.array([], dtype=int)
        else:
            if is_drought:
                # For drought analysis with fixed threshold
                peaks_result = scipy_find_peaks(-values, height=-threshold)
                peak_indices = peaks_result[0] if len(peaks_result[0]) > 0 else np.array([], dtype=int)
            else:
                # For high extremes with fixed threshold
                peaks_result = scipy_find_peaks(values, height=threshold)
                peak_indices = peaks_result[0] if len(peaks_result[0]) > 0 else np.array([], dtype=int)

        # Create peaks DataFrame
        if len(peak_indices) > 0:
            peaks_data = pd.DataFrame({
                'DateTime': dates.iloc[peak_indices],
                'Value': values[peak_indices]
            })
            if data_type == 'rain':
                peaks_data.columns = ['DateTime', 'mm_Rain']
            else:
                peaks_data.columns = ['DateTime', column]
        else:
            # Create empty DataFrame with correct columns
            if data_type == 'rain':
                peaks_data = pd.DataFrame({
                    'DateTime': pd.Series(dtype='datetime64[ns]'),
                    'mm_Rain': pd.Series(dtype='float64')
                })
            else:
                peaks_data = pd.DataFrame({
                    'DateTime': pd.Series(dtype='datetime64[ns]'),
                    column: pd.Series(dtype='float64')
                })

        return peaks_data

    except Exception as e:
        logging.error(f"Error in find_peaks_data: {str(e)}")
        raise


def calculate_exceedances(peaks_data, threshold, data_type, is_drought=False):
    """Calculate exceedances from threshold
    
    Args:
        peaks_data: DataFrame with peaks information
        threshold: float or Series for threshold value(s)
        data_type: str, either 'gwl' or 'rain'
        is_drought: bool, whether to analyze drought conditions
    
    Returns:
        Series with exceedance values
    """
    try:
        # Select appropriate column based on data type
        if data_type == 'rain':
            values = peaks_data['mm_Rain']
        else:
            values = peaks_data[peaks_data.columns[1]]  

        # Calculate exceedances
        if isinstance(threshold, pd.Series):
            matched_threshold = threshold[peaks_data.index]
            if is_drought:
                exceedances = matched_threshold - values  # Positive values for drought
            else:
                exceedances = values - matched_threshold  # Positive values for high extremes
        else:
            if is_drought:
                exceedances = threshold - values  # Positive values for drought
            else:
                exceedances = values - threshold  # Positive values for high extremes

        return exceedances

    except Exception as e:
        logging.error(f"Error in calculate_exceedances: {str(e)}")
        raise

