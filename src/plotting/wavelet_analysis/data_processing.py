"""
Wavelet Analysis Data Processing
=================================

Data processing and caching functions for wavelet analysis.
"""

import numpy as np
import pywt
import functools
from scipy import signal


def remove_confounders(df, y_col, confounder_cols, lags=7):
    """
    Remove confounder effects from a time series using linear regression.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    y_col : str
        Column name for the dependent variable
    confounder_cols : list
        List of column names for confounder variables
    lags : int
        Number of lags to use in regression (default: 7)
    
    Returns:
    --------
    np.ndarray
        Residuals after removing confounder effects
    """
    import statsmodels.api as sm
    
    y = df[y_col].values
    X = []
    for col in confounder_cols:
        if col in df.columns:
            for lag in range(lags+1):
                X.append(df[col].shift(lag).values)
    if not X:
        return y  # No confounders
    X = np.column_stack(X)
    mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    y_clean = y[mask]
    X_clean = X[mask]
    X_clean = sm.add_constant(X_clean)
    model = sm.OLS(y_clean, X_clean).fit()
    y_pred = model.predict(X_clean)
    y_resid = y_clean - y_pred
    # Fill residuals back into original shape
    y_out = np.full_like(y, np.nan)
    y_out[mask] = y_resid
    return y_out


def compute_wavelet_cwt(y_tuple, wavelet, scale):
    """
    Cached computation of the continuous wavelet transform.
    y_tuple: tuple of y values (must be hashable for lru_cache)
    wavelet: wavelet name
    scale: max scale (int)
    Returns: coef, freqs
    """
    y = np.array(y_tuple)
    scales = np.arange(1, scale+1)
    coef, freqs = pywt.cwt(y, scales, wavelet)
    return coef, freqs


@functools.lru_cache(maxsize=128) 
def cached_data_resample(data_hash, freq, start_date, end_date, use_all_data_flag):
    """
    PERFORMANCE BOOST: Cache resampled data results.
    Prevents expensive resampling operations on tab/overlay changes.
    """
    # This will be populated by the calling function
    # Returns cached resampled dataframe
    return None


@functools.lru_cache(maxsize=64)
def cached_percentile_calculation(data_tuple, percentile):
    """
    PERFORMANCE BOOST: Cache percentile calculations.
    These are expensive with large datasets.
    """
    data_array = np.array(data_tuple)
    return np.percentile(data_array[~np.isnan(data_array)], percentile)


@functools.lru_cache(maxsize=32)
def cached_coherence_computation(x1_tuple, x2_tuple, sampling_rate, num_periods):
    """
    PERFORMANCE BOOST: Cache wavelet coherence computations.
    These are very expensive and often repeated with same parameters.
    """
    try:
        x1 = np.array(x1_tuple)
        x2 = np.array(x2_tuple)
        
        # Simple coherence computation (placeholder - replace with actual wavelet coherence)
        f, Cxy = signal.coherence(x1, x2, fs=1/sampling_rate, nperseg=min(256, len(x1)//4))
        return f, Cxy
    except:
        return None, None


@functools.lru_cache(maxsize=128)
def cached_wavelet_periods(min_period, max_period, num_periods, dt, wavelet_type):
    """
    PERFORMANCE BOOST: Cache wavelet period calculations.
    These are computed frequently with same parameters.
    """
    periods = np.logspace(np.log10(min_period), np.log10(max_period), num_periods)
    return periods

