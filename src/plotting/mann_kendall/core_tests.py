"""
Core Mann-Kendall test implementations

This module contains the core Mann-Kendall test functions including:
- Original Mann-Kendall test
- Seasonal Mann-Kendall test
- Caching mechanisms for performance
"""

import pandas as pd
import numpy as np
import pymannkendall as mk
from .constants import (
    USE_ROLLING_MEAN,
    ROLLING_WINDOW_SIZE,
    ALPHA_LEVEL,
    IGNORE_ZERO_VALUES_IN_TREND
)


# =============================================================================
# CACHING MECHANISM
# =============================================================================

# Simple in-memory cache for Mann-Kendall test results
_mk_cache = {}
_mk_cache_max_size = 512


def _get_cache_key(series, alpha, use_rolling_mean, rolling_window):
    """
    Create a cache key for Mann-Kendall test results.
    """
    if series is None or len(series) == 0:
        return None
    
    # Create a hashable key from the series data and parameters
    values_hash = hash(tuple(series.values))
    length = len(series)
    key = (values_hash, length, alpha, use_rolling_mean, rolling_window)
    return key


def _cached_mk_test(series, alpha, use_rolling_mean, rolling_window):
    """
    Cached Mann-Kendall test using in-memory dictionary cache.
    """
    cache_key = _get_cache_key(series, alpha, use_rolling_mean, rolling_window)
    
    if cache_key is None:
        return None
    
    # Check cache
    if cache_key in _mk_cache:
        return _mk_cache[cache_key]
    
    # Perform the test
    result = _perform_mk_test(series, alpha, use_rolling_mean, rolling_window)
    
    # Cache the result
    if len(_mk_cache) >= _mk_cache_max_size:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(_mk_cache))
        del _mk_cache[oldest_key]
    
    _mk_cache[cache_key] = result
    return result


def _perform_mk_test(series, alpha, use_rolling_mean, rolling_window):
    """
    Perform the actual Mann-Kendall test.
    """
    try:
        # Use global defaults if not specified
        if use_rolling_mean is None:
            use_rolling_mean = USE_ROLLING_MEAN
        if rolling_window is None:
            rolling_window = ROLLING_WINDOW_SIZE
        
        # Check if we have enough data
        if len(series) < 1:
            return {
                'trend': 'insufficient_data',
                'p_value': None,
                'z_score': None,
                'S': None,
                'var_S': None,
                'n_points': len(series),
                'slope': None,
                'slope_mm_year': None,
                'rolling_mean_applied': False,
                'rolling_window': None,
                'error': 'Insufficient data (need at least 1 point)'
            }
        
        # Apply rolling mean if requested (no data length restriction)
        if use_rolling_mean:
            # Calculate rolling mean
            rolling_mean = series.rolling(window=rolling_window, center=False).mean()
            # Remove NaN values from rolling mean
            clean_rolling_mean = rolling_mean.dropna()
            
            # Check if rolling mean produced any data
            if len(clean_rolling_mean) < 1:
                # Fall back to original data if rolling mean is empty
                analysis_series = series
                rolling_mean_applied = False
                actual_window = None
            else:
                # Use rolling mean result
                analysis_series = clean_rolling_mean
                rolling_mean_applied = True
                actual_window = rolling_window
        else:
            analysis_series = series
            rolling_mean_applied = False
            actual_window = None
        
        # Perform Mann-Kendall test
        try:
            # Check if we have enough data points for MK test
            if len(analysis_series) < 2:
                return {
                    'trend': 'insufficient_data',
                    'p_value': None,
                    'z_score': None,
                    'S': None,
                    'var_S': None,
                    'n_points': len(analysis_series),
                    'slope': None,
                    'slope_mm_year': None,
                    'rolling_mean_applied': rolling_mean_applied,
                    'rolling_window': actual_window,
                    'error': 'Need at least 2 data points for Mann-Kendall test'
                }
            
            # Add safety checks before Mann-Kendall test to prevent division by zero
            if len(analysis_series) < 3:
                return {
                    'trend': 'no trend',
                    'p_value': 1.0,
                    'z_score': 0.0,
                    'S': 0,
                    'var_S': 0,
                    'n_points': len(analysis_series),
                    'slope': 0.0,
                    'slope_mm_year': 0.0,
                    'rolling_mean_applied': rolling_mean_applied,
                    'rolling_window': actual_window,
                    'error': f'Insufficient data for Mann-Kendall trend analysis: {len(analysis_series)} points available, need at least 3 points for meaningful trend detection'
                }
            
            # Check for identical values (no variance)
            if analysis_series.nunique() == 1:
                return {
                    'trend': 'no trend',
                    'p_value': 1.0,
                    'z_score': 0.0,
                    'S': 0,
                    'var_S': 0,
                    'n_points': len(analysis_series),
                    'slope': 0.0,
                    'slope_mm_year': 0.0,
                    'rolling_mean_applied': rolling_mean_applied,
                    'rolling_window': actual_window,
                    'error': f'Cannot perform Mann-Kendall trend analysis: All {len(analysis_series)} data points are identical ({analysis_series.iloc[0]:.3f}) - no variance detected for trend detection'
                }
            
            # Check for very low variance
            variance = analysis_series.var()
            if variance < 1e-12:
                return {
                    'trend': 'no trend',
                    'p_value': 1.0,
                    'z_score': 0.0,
                    'S': 0,
                    'var_S': 0,
                    'n_points': len(analysis_series),
                    'slope': 0.0,
                    'slope_mm_year': 0.0,
                    'rolling_mean_applied': rolling_mean_applied,
                    'rolling_window': actual_window,
                    'error': f'Cannot perform Mann-Kendall trend analysis: Very low variance ({variance:.2e}) - data shows insufficient variability for meaningful trend detection'
                }
            
            try:
                mk_result = mk.original_test(analysis_series.values, alpha=alpha)
            except (ZeroDivisionError, ValueError, RuntimeError) as mk_error:
                return {
                    'trend': 'no trend',
                    'p_value': 1.0,
                    'z_score': 0.0,
                    'S': 0,
                    'var_S': 0,
                    'n_points': len(analysis_series),
                    'slope': 0.0,
                    'slope_mm_year': 0.0,
                    'rolling_mean_applied': rolling_mean_applied,
                    'rolling_window': actual_window,
                    'error': f'Mann-Kendall calculation error: {str(mk_error)}'
                }
            
            # Extract results with error checking
            try:
                trend = mk_result.trend
                p_value = mk_result.p
                tau = mk_result.Tau
                z_score = mk_result.z
                S = mk_result.s
                var_S = mk_result.var_s
                n_points = len(analysis_series)
                slope = mk_result.slope
                # Prevent division by zero in slope conversion
                if slope is not None and not (isinstance(slope, (int, float)) and slope == 0):
                    slope_mm_year = slope * 365.25 * 1000  # Convert to mm/year
                else:
                    slope_mm_year = None
            except (AttributeError, TypeError, ZeroDivisionError) as e:
                # Handle missing attributes or calculation errors
                return {
                    'trend': 'calculation_error',
                    'p_value': None,
                    'z_score': None,
                    'S': None,
                    'var_S': None,
                    'n_points': len(analysis_series),
                    'slope': None,
                    'slope_mm_year': None,
                    'rolling_mean_applied': rolling_mean_applied,
                    'rolling_window': actual_window,
                    'error': f'Result extraction failed: {str(e)}'
                }
            
            return {
                'trend': trend,
                'p_value': p_value,
                'z_score': z_score,
                'S': S,
                'var_S': var_S,
                'n_points': n_points,
                'slope': slope,
                'slope_mm_year': slope_mm_year,
                'rolling_mean_applied': rolling_mean_applied,
                'rolling_window': actual_window,
                'error': None
            }
            
        except Exception as e:
            return {
                'trend': 'error',
                'p_value': None,
                'z_score': None,
                'S': None,
                'var_S': None,
                'n_points': len(analysis_series),
                'slope': None,
                'slope_mm_year': None,
                'rolling_mean_applied': rolling_mean_applied,
                'rolling_window': actual_window,
                'error': f'Mann-Kendall test failed: {str(e)}'
            }
            
    except Exception as e:
        return {
            'trend': 'error',
            'p_value': None,
            'z_score': None,
            'S': None,
            'var_S': None,
            'n_points': len(series) if series is not None else 0,
            'slope': None,
            'slope_mm_year': None,
            'rolling_mean_applied': False,
            'rolling_window': None,
            'error': f'Function error: {str(e)}'
        }


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def mann_kendall_test(series, alpha=None, use_rolling_mean=None, rolling_window=None):
    """
    Perform Mann-Kendall trend test on a time series with caching.
    
    Args:
        series: pandas Series with datetime index and groundwater levels
        alpha: Significance level (default: 0.05)
        use_rolling_mean: Whether to apply rolling mean (default: uses global USE_ROLLING_MEAN)
        rolling_window: Rolling window size (default: uses global ROLLING_WINDOW_SIZE)
    
    Returns:
        dict: MK test results including trend, p-value, tau, slope, etc.
    """
    try:
        if series is None or len(series) < 1:
            return {'error': 'Insufficient data for MK test'}
        
        # Remove NaN values
        clean_series = series.dropna()
        if len(clean_series) < 1:
            return {'error': 'Insufficient data after removing NaN values'}
        
        # Handle zero values based on configuration
        original_length = len(clean_series)
        zero_values_count = 0
        if IGNORE_ZERO_VALUES_IN_TREND:
            # Filter out zero values
            non_zero_series = clean_series[clean_series != 0]
            zero_values_count = len(clean_series) - len(non_zero_series)
            clean_series = non_zero_series
            if len(clean_series) < 1:
                return {'error': 'Insufficient data after removing zero values'}
        
        # Apply rolling mean if enabled and sufficient data available
        if use_rolling_mean is None:
            use_rolling_mean = USE_ROLLING_MEAN
        if rolling_window is None:
            rolling_window = ROLLING_WINDOW_SIZE
            
        # Use global alpha if none specified
        if alpha is None:
            alpha = ALPHA_LEVEL
        
        # Simple rolling mean - no adaptive logic
        if use_rolling_mean and rolling_window > 1:
            # Apply rolling mean with the specified window size
            smoothed_series = clean_series.rolling(
                window=rolling_window, center=False
            ).mean().dropna()
            
            # Always use rolling mean result - no minimum point restriction
            analysis_series = smoothed_series
            smoothing_applied = True
            smoothing_info = f"Rolling {rolling_window}-day mean applied"
        else:
            # Use original data
            analysis_series = clean_series
            smoothing_applied = False
            smoothing_info = "No smoothing applied"
        
        # Use cached version if possible
        try:
            cached_result = _cached_mk_test(analysis_series, alpha, use_rolling_mean, rolling_window)
            if cached_result is not None and 'error' not in cached_result:
                # Add additional metadata
                cached_result['interpretation'] = cached_result.get('trend', 'Unknown')
                cached_result['original_n_points'] = len(clean_series)
                cached_result['alpha'] = alpha
                cached_result['smoothing_applied'] = smoothing_applied
                cached_result['smoothing_info'] = smoothing_info
                cached_result['rolling_window_size'] = rolling_window if smoothing_applied else None
                cached_result['zero_values_excluded'] = zero_values_count if IGNORE_ZERO_VALUES_IN_TREND else 0
                cached_result['zero_values_handling'] = 'Zero values excluded' if IGNORE_ZERO_VALUES_IN_TREND else 'Zero values included'
                return cached_result
        except:
            pass  # Fall back to direct calculation if caching fails

        # Perform Mann-Kendall test directly if caching failed
        try:
            # Add safety checks before Mann-Kendall test to prevent division by zero
            if len(analysis_series) < 3:
                return {
                    'trend': 'no trend',
                    'p_value': 1.0,
                    'tau': 0.0,
                    'z_score': 0.0,
                    'S': 0,
                    'var_S': 0,
                    'slope': 0.0,
                    'slope_mm_year': 0.0,
                    'interpretation': 'No trend - insufficient data',
                    'n_points': len(analysis_series),
                    'original_n_points': len(clean_series),
                    'alpha': alpha,
                    'smoothing_applied': smoothing_applied,
                    'smoothing_info': smoothing_info,
                    'rolling_window_size': rolling_window if smoothing_applied else None,
                    'zero_values_excluded': zero_values_count if IGNORE_ZERO_VALUES_IN_TREND else 0,
                    'zero_values_handling': 'Zero values excluded' if IGNORE_ZERO_VALUES_IN_TREND else 'Zero values included',
                    'error': f'Cannot perform Mann-Kendall trend analysis: {len(analysis_series)} data points available, need at least 3 points for meaningful trend detection'
                }
            
            # Check for identical values (no variance)
            if analysis_series.nunique() == 1:
                return {
                    'trend': 'no trend',
                    'p_value': 1.0,
                    'tau': 0.0,
                    'z_score': 0.0,
                    'S': 0,
                    'var_S': 0,
                    'slope': 0.0,
                    'slope_mm_year': 0.0,
                    'interpretation': 'No trend - all values identical',
                    'n_points': len(analysis_series),
                    'original_n_points': len(clean_series),
                    'alpha': alpha,
                    'smoothing_applied': smoothing_applied,
                    'smoothing_info': smoothing_info,
                    'rolling_window_size': rolling_window if smoothing_applied else None,
                    'zero_values_excluded': zero_values_count if IGNORE_ZERO_VALUES_IN_TREND else 0,
                    'zero_values_handling': 'Zero values excluded' if IGNORE_ZERO_VALUES_IN_TREND else 'Zero values included',
                    'error': f'Cannot perform Mann-Kendall trend analysis: All {len(analysis_series)} data points are identical ({analysis_series.iloc[0]:.3f}) - no variance detected for trend detection'
                }
            
            # Check for very low variance
            variance = analysis_series.var()
            if variance < 1e-12:
                return {
                    'trend': 'no trend',
                    'p_value': 1.0,
                    'tau': 0.0,
                    'z_score': 0.0,
                    'S': 0,
                    'var_S': 0,
                    'slope': 0.0,
                    'slope_mm_year': 0.0,
                    'interpretation': 'No trend - very low variance',
                    'n_points': len(analysis_series),
                    'original_n_points': len(clean_series),
                    'alpha': alpha,
                    'smoothing_applied': smoothing_applied,
                    'smoothing_info': smoothing_info,
                    'rolling_window_size': rolling_window if smoothing_applied else None,
                    'zero_values_excluded': zero_values_count if IGNORE_ZERO_VALUES_IN_TREND else 0,
                    'zero_values_handling': 'Zero values excluded' if IGNORE_ZERO_VALUES_IN_TREND else 'Zero values included',
                    'error': f'Cannot perform Mann-Kendall trend analysis: Very low variance ({variance:.2e}) - data shows insufficient variability for meaningful trend detection'
                }
            
            try:
                mk_result = mk.original_test(analysis_series.values, alpha=alpha)
            except (ZeroDivisionError, ValueError, RuntimeError) as mk_error:
                return {
                    'trend': 'no trend',
                    'p_value': 1.0,
                    'tau': 0.0,
                    'z_score': 0.0,
                    'S': 0,
                    'var_S': 0,
                    'slope': 0.0,
                    'slope_mm_year': 0.0,
                    'interpretation': 'No trend - calculation error',
                    'n_points': len(analysis_series),
                    'original_n_points': len(clean_series),
                    'alpha': alpha,
                    'smoothing_applied': smoothing_applied,
                    'smoothing_info': smoothing_info,
                    'rolling_window_size': rolling_window if smoothing_applied else None,
                    'zero_values_excluded': zero_values_count if IGNORE_ZERO_VALUES_IN_TREND else 0,
                    'zero_values_handling': 'Zero values excluded' if IGNORE_ZERO_VALUES_IN_TREND else 'Zero values included',
                    'error': f'Mann-Kendall calculation error: {str(mk_error)}'
                }
            
            # Extract results using pure pymannkendall with error handling
            try:
                trend = mk_result.trend
                p_value = mk_result.p
                tau = mk_result.Tau
                z_score = mk_result.z
                S = mk_result.s
                var_S = mk_result.var_s
                slope = mk_result.slope
                
                # Prevent division by zero in slope conversion
                if slope is not None and not (isinstance(slope, (int, float)) and slope == 0):
                    slope_mm_year = slope * 365.25 * 1000  # Convert to mm/year
                else:
                    slope_mm_year = None
                
                # Determine trend interpretation
                if p_value <= alpha:
                    if trend == 'increasing':
                        interpretation = 'Significant increasing trend'
                    elif trend == 'decreasing':
                        interpretation = 'Significant decreasing trend'
                    else:
                        interpretation = 'No significant trend'
                else:
                    interpretation = 'No significant trend'
                
                return {
                    'trend': trend,
                    'p_value': p_value,
                    'tau': tau,
                    'z_score': z_score,
                    'S': S,
                    'var_S': var_S,
                    'slope': slope,
                    'slope_mm_year': slope_mm_year,
                    'interpretation': interpretation,
                    'n_points': len(analysis_series),
                    'original_n_points': len(clean_series),
                    'alpha': alpha,
                    'smoothing_applied': smoothing_applied,
                    'smoothing_info': smoothing_info,
                    'rolling_window_size': rolling_window if smoothing_applied else None,
                    'zero_values_excluded': zero_values_count if IGNORE_ZERO_VALUES_IN_TREND else 0,
                    'zero_values_handling': 'Zero values excluded' if IGNORE_ZERO_VALUES_IN_TREND else 'Zero values included'
                }
                
            except (AttributeError, TypeError, ZeroDivisionError) as e:
                # Handle missing attributes or calculation errors
                return {
                    'trend': 'calculation_error',
                    'p_value': None,
                    'tau': None,
                    'z_score': None,
                    'S': None,
                    'var_S': None,
                    'slope': None,
                    'slope_mm_year': None,
                    'interpretation': 'Calculation error',
                    'n_points': len(analysis_series),
                    'original_n_points': len(clean_series),
                    'alpha': alpha,
                    'smoothing_applied': smoothing_applied,
                    'smoothing_info': smoothing_info,
                    'rolling_window_size': rolling_window if smoothing_applied else None,
                    'zero_values_excluded': zero_values_count if IGNORE_ZERO_VALUES_IN_TREND else 0,
                    'zero_values_handling': 'Zero values excluded' if IGNORE_ZERO_VALUES_IN_TREND else 'Zero values included',
                    'error': f'Result extraction failed: {str(e)}'
                }
                
        except Exception as e:
            # Handle MK test execution errors
            return {
                'trend': 'error',
                'p_value': None,
                'tau': None,
                'z_score': None,
                'S': None,
                'var_S': None,
                'slope': None,
                'slope_mm_year': None,
                'interpretation': 'Test execution failed',
                'n_points': len(analysis_series),
                'original_n_points': len(clean_series),
                'alpha': alpha,
                'smoothing_applied': smoothing_applied,
                'smoothing_info': smoothing_info,
                'rolling_window_size': rolling_window if smoothing_applied else None,
                'zero_values_excluded': zero_values_count if IGNORE_ZERO_VALUES_IN_TREND else 0,
                'zero_values_handling': 'Zero values excluded' if IGNORE_ZERO_VALUES_IN_TREND else 'Zero values included',
                'error': f'Mann-Kendall test failed: {str(e)}'
            }
        
    except Exception as e:
        print(f"Error in mann_kendall_test: {e}")
        return {'error': str(e)}


def seasonal_mann_kendall_test(series, alpha=None, seasonal_period=12):
    """
    Perform Seasonal Mann-Kendall trend test on a time series.
    
    Args:
        series: pandas Series with datetime index and groundwater levels
        alpha: Significance level (default: 0.05)
        seasonal_period: Number of seasons (default: 12 for monthly)
    
    Returns:
        dict: Seasonal MK test results
    """
    try:
        if series is None or len(series) < 1:
            return {'error': 'Insufficient data for seasonal MK test'}
        
        # Remove NaN values
        clean_series = series.dropna()
        if len(clean_series) < 1:
            return {'error': 'Insufficient data after removing NaN values'}
        
        # Handle zero values based on configuration
        original_length = len(clean_series)
        zero_values_count = 0
        if IGNORE_ZERO_VALUES_IN_TREND:
            # Filter out zero values
            non_zero_series = clean_series[clean_series != 0]
            zero_values_count = len(clean_series) - len(non_zero_series)
            clean_series = non_zero_series
            if len(clean_series) < 1:
                return {'error': 'Insufficient data after removing zero values'}
        
        # Use global alpha if none specified
        if alpha is None:
            alpha = ALPHA_LEVEL
        
        # Seasonal MK test uses original data only (no rolling mean)
        analysis_series = clean_series
        smoothing_applied = False
        smoothing_info = "No smoothing applied - Seasonal MK uses original data"
        
        # Perform Seasonal Mann-Kendall test
        try:
            mk_result = mk.seasonal_test(analysis_series.values, period=seasonal_period)
            
            # Extract results using pure pymannkendall with error handling
            try:
                trend = mk_result.trend
                p_value = mk_result.p
                tau = mk_result.Tau
                z_score = mk_result.z
                S = mk_result.s
                var_S = mk_result.var_s
                slope = mk_result.slope
                
                # Prevent division by zero in slope conversion
                if slope is not None and not (isinstance(slope, (int, float)) and slope == 0):
                    slope_mm_year = slope * 365.25 * 1000  # Convert to mm/year
                else:
                    slope_mm_year = None
                
                # Determine trend interpretation
                if p_value <= alpha:
                    if trend == 'increasing':
                        interpretation = 'Significant increasing trend (seasonal)'
                    elif trend == 'decreasing':
                        interpretation = 'Significant decreasing trend (seasonal)'
                    else:
                        interpretation = 'No significant trend (seasonal)'
                else:
                    interpretation = 'No significant trend (seasonal)'
                
                return {
                    'trend': trend,
                    'p_value': p_value,
                    'tau': tau,
                    'z_score': z_score,
                    'S': S,
                    'var_S': var_S,
                    'slope': slope,
                    'slope_mm_year': slope_mm_year,
                    'interpretation': interpretation,
                    'n_points': len(analysis_series),
                    'original_n_points': len(clean_series),
                    'alpha': alpha,
                    'seasonal_period': seasonal_period,
                    'smoothing_applied': smoothing_applied,
                    'smoothing_info': smoothing_info,
                    'zero_values_excluded': zero_values_count if IGNORE_ZERO_VALUES_IN_TREND else 0,
                    'zero_values_handling': 'Zero values excluded' if IGNORE_ZERO_VALUES_IN_TREND else 'Zero values included'
                }
                
            except (AttributeError, TypeError, ZeroDivisionError) as e:
                return {
                    'trend': 'calculation_error',
                    'p_value': None,
                    'tau': None,
                    'z_score': None,
                    'S': None,
                    'var_S': None,
                    'slope': None,
                    'slope_mm_year': None,
                    'interpretation': 'Calculation error',
                    'n_points': len(analysis_series),
                    'original_n_points': len(clean_series),
                    'alpha': alpha,
                    'seasonal_period': seasonal_period,
                    'smoothing_applied': smoothing_applied,
                    'smoothing_info': smoothing_info,
                    'zero_values_excluded': zero_values_count if IGNORE_ZERO_VALUES_IN_TREND else 0,
                    'zero_values_handling': 'Zero values excluded' if IGNORE_ZERO_VALUES_IN_TREND else 'Zero values included',
                    'error': f'Result extraction failed: {str(e)}'
                }
                
        except Exception as e:
            return {
                'trend': 'error',
                'p_value': None,
                'tau': None,
                'z_score': None,
                'S': None,
                'var_S': None,
                'slope': None,
                'slope_mm_year': None,
                'interpretation': 'Test execution failed',
                'n_points': len(analysis_series),
                'original_n_points': len(clean_series),
                'alpha': alpha,
                'seasonal_period': seasonal_period,
                'smoothing_applied': smoothing_applied,
                'smoothing_info': smoothing_info,
                'zero_values_excluded': zero_values_count if IGNORE_ZERO_VALUES_IN_TREND else 0,
                'zero_values_handling': 'Zero values excluded' if IGNORE_ZERO_VALUES_IN_TREND else 'Zero values included',
                'error': f'Seasonal Mann-Kendall test failed: {str(e)}'
            }
        
    except Exception as e:
        print(f"Error in seasonal_mann_kendall_test: {e}")
        return {'error': str(e)}


__all__ = [
    'mann_kendall_test',
    'seasonal_mann_kendall_test',
    '_get_cache_key',
    '_cached_mk_test',
    '_perform_mk_test'
]

