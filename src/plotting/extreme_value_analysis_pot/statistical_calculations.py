"""
POT Statistical Calculations
=============================

Functions for GPD fitting, return period calculations, and confidence intervals.
"""

import pandas as pd
import numpy as np
from scipy.stats import genpareto
from scipy import stats
from scipy.optimize import minimize
import logging


def calculate_return_periods(peaks_data: pd.Series, data_type: str, is_drought: bool = False) -> dict:
    """Calculate return periods and confidence intervals using GPD"""
    try:
        peaks = peaks_data.values
        n = len(peaks)

        sorted_peaks = np.sort(peaks) if is_drought else np.sort(peaks)[::-1]
        
        # Calculate plotting positions (Weibull plotting position)
        if data_type == 'rain':
            # Gringorten plotting position for rainfall
            plotting_positions = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
        else:
            # Weibull plotting position for GWL
            plotting_positions = np.arange(1, n + 1) / (n + 1)
        
        # Calculate exceedance probabilities and return periods
        # ISSUE: Probability calculation needs to be reversed for droughts
        prob = plotting_positions if is_drought else (1 - plotting_positions)
        return_periods = 1 / prob
        
        # Fit GPD to peaks
        # ISSUE: For droughts, we should fit to negative values to maintain right-tail
        peaks_for_fit = -peaks if is_drought else peaks
        shape, loc, scale = genpareto.fit(peaks_for_fit)
        
        # Calculate theoretical return levels
        theoretical_periods = np.logspace(0, 2, 100)  # 1 to 100 years
        prob = 1/theoretical_periods if is_drought else 1 - 1/theoretical_periods
        # ISSUE: Need to negate return levels for droughts
        return_levels = -genpareto.ppf(prob, shape, loc, scale) if is_drought else genpareto.ppf(prob, shape, loc, scale)
        
        # Calculate confidence intervals
        alpha = 0.05
        z = stats.norm.ppf(1 - alpha/2)
        std_error = scale / np.sqrt(n)
        
        ci_lower = return_levels - z * std_error
        ci_upper = return_levels + z * std_error
        
        return {
            'empirical_periods': return_periods,
            'empirical_levels': sorted_peaks,
            'theoretical_periods': theoretical_periods,
            'theoretical_levels': return_levels,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'gpd_params': (shape, loc, scale)
        }
        
    except Exception as e:
        logging.error(f"Error in calculate_return_periods: {str(e)}")
        return None


def calculate_confidence_intervals(peaks: pd.Series, params, return_periods, is_drought: bool = False, alpha=0.05):
    """Calculate return periods and confidence intervals using GPD"""
    try:
        n = len(peaks)
        shape, loc, scale = params
        
        # Calculate return levels based on peaks
        prob = 1/return_periods if is_drought else 1 - 1/return_periods
        
        # For droughts, negate values for GPD fit and then negate back
        if is_drought:
            return_levels = -genpareto.ppf(prob, shape, loc, scale)
        else:
            return_levels = genpareto.ppf(prob, shape, loc, scale)
        
        # Calculate standard errors for return levels
        # Note: The standard error calculation remains the same as it's based on the GPD parameters
        std_error = np.sqrt(scale**2 / n * (1 + shape * np.log(return_periods))**2)
        
        # Calculate confidence intervals
        z = stats.norm.ppf(1 - alpha/2)
        ci_lower = return_levels - z * std_error
        ci_upper = return_levels + z * std_error
        
        return ci_lower, ci_upper
        
    except Exception as e:
        logging.error(f"Error in calculate_confidence_intervals: {str(e)}")
        return None, None


def fit_gpd_by_season(exceedances, seasons):
    """
    Fit GPD to exceedances for each season.
    Returns a dict: {season: (shape, loc, scale)}
    """
    results = {}
    unique_seasons = np.unique(seasons)
    for season in unique_seasons:
        season_mask = (seasons == season)
        if np.sum(season_mask) > 1:  # Need at least 2 points to fit
            params = genpareto.fit(exceedances[season_mask])
            results[season] = params
        else:
            results[season] = (np.nan, np.nan, np.nan)
    return results


def calculate_return_levels_by_season(gpd_params_by_season, periods):
    """
    Calculate return levels for each season.
    Returns a dict: {season: return_levels}
    """
    levels = {}
    for season, params in gpd_params_by_season.items():
        if not np.any(np.isnan(params)):
            levels[season] = genpareto.ppf(1 - 1/periods, *params)
        else:
            levels[season] = np.full_like(periods, np.nan)
    return levels


def fit_gpd_with_trend(exceedances, times):
    """
    Fit GPD with scale parameter as a linear function of time: sigma(t) = sigma0 + sigma1 * t
    Returns (shape, loc, sigma0, sigma1)
    """
    # Normalize times for stability
    t = (times - np.min(times)) / (np.max(times) - np.min(times))
    def nll(params):
        shape, loc, sigma0, sigma1 = params
        sigma = sigma0 + sigma1 * t
        if np.any(sigma <= 0):
            return np.inf
        return -np.sum(genpareto.logpdf(exceedances, shape, loc, sigma))
    # Initial guess: stationary fit + small slope
    shape0, loc0, scale0 = genpareto.fit(exceedances)
    res = minimize(nll, [shape0, loc0, scale0, 0.01], method='L-BFGS-B')
    return res.x  # shape, loc, sigma0, sigma1


def calculate_return_levels_trend(params, times, periods):
    """
    Calculate return levels for a range of times and periods.
    Returns: dict {time: return_levels}
    """
    shape, loc, sigma0, sigma1 = params
    t_norm = (times - np.min(times)) / (np.max(times) - np.min(times))
    levels = {}
    for i, t in enumerate(t_norm):
        sigma = sigma0 + sigma1 * t
        levels[times[i]] = genpareto.ppf(1 - 1/periods, shape, loc, sigma)
    return levels


def fit_gpd_by_covariate(exceedances, covariate):
    """
    Fit GPD with scale parameter as a linear function of covariate: sigma(x) = sigma0 + sigma1 * x
    Returns (shape, loc, sigma0, sigma1)
    """
    x = (covariate - np.min(covariate)) / (np.max(covariate) - np.min(covariate))
    def nll(params):
        shape, loc, sigma0, sigma1 = params
        sigma = sigma0 + sigma1 * x
        if np.any(sigma <= 0):
            return np.inf
        return -np.sum(genpareto.logpdf(exceedances, shape, loc, sigma))
    shape0, loc0, scale0 = genpareto.fit(exceedances)
    res = minimize(nll, [shape0, loc0, scale0, 0.01], method='L-BFGS-B')
    return res.x


def calculate_return_levels_covariate(params, covariate_values, periods):
    """
    Calculate return levels for a range of covariate values and periods.
    Returns: dict {covariate_value: return_levels}
    """
    shape, loc, sigma0, sigma1 = params
    x_norm = (covariate_values - np.min(covariate_values)) / (np.max(covariate_values) - np.min(covariate_values))
    levels = {}
    for i, x in enumerate(x_norm):
        sigma = sigma0 + sigma1 * x
        levels[covariate_values[i]] = genpareto.ppf(1 - 1/periods, shape, loc, sigma)
    return levels

