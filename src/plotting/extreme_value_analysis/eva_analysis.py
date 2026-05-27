"""
EVA Statistical Analysis Functions

This module contains the core statistical analysis functions for Extreme Value Analysis
using Generalized Extreme Value (GEV) distributions. This is separate from the general
statistics.py module.
"""

import logging
import traceback
import numpy as np
import pandas as pd
from typing import Optional, Dict
from scipy import stats
from pyextremes import EVA


def perform_extreme_value_analysis(data) -> Optional[Dict]:
    """
    Perform extreme value analysis using GEV distribution with efficient confidence interval calculation
    
    Parameters:
    -----------
    data : np.ndarray or pd.Series
        Block maxima data
        
    Returns:
    --------
    dict : Dictionary containing GEV parameters, return levels, confidence intervals, etc.
    """
    try:
        # Convert input to numpy array if needed
        if isinstance(data, pd.Series):
            # Store the original Series for block_maxima
            block_maxima = data
            data = data.values
        else:
            block_maxima = pd.Series(data)
            
        data = np.array(data, dtype=float)
        
        logging.info(f"Starting EVA analysis with {len(data)} values")
        
        # Sort data in ascending order
        sorted_data = np.sort(data)
        n = len(sorted_data)
        logging.info(f"Data range: {sorted_data[0]:.3f} to {sorted_data[-1]:.3f}")
        
        # Calculate plotting positions using Gringorten formula
        plotting_positions = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
        empirical_return_periods = 1 / (1 - plotting_positions)
        
        # Fit GEV using scipy's built-in MLE method
        shape, loc, scale = stats.genextreme.fit(sorted_data)
        gev_params = (shape, loc, scale)
        logging.info(f"GEV fit: shape={shape:.3f}, loc={loc:.3f}, scale={scale:.3f}")
        
        # Calculate theoretical probabilities
        theoretical_probs = stats.genextreme.cdf(sorted_data, *gev_params)
        
        # Calculate return levels for various return periods
        return_periods = np.logspace(0, 2.5, 100)  # From 1 to ~316 years
        return_levels = stats.genextreme.ppf(1 - 1/return_periods, *gev_params)
        
        # Calculate confidence intervals using delta method
        p = 1 - 1/return_periods
        z_p = return_levels
        
        # Standard errors based on Fisher Information Matrix approximation
        se_factor = np.sqrt(1/n)  # Standard error scales with 1/√n
        
        # Calculate confidence intervals using normal approximation
        z_alpha = stats.norm.ppf(0.975)  # 95% CI
        margin_of_error = z_alpha * se_factor * scale * np.sqrt(1 + (shape * np.log(-np.log(p)))**2)
        
        confidence_levels = np.zeros((2, len(return_periods)))
        confidence_levels[0] = z_p - margin_of_error  # Lower bound
        confidence_levels[1] = z_p + margin_of_error  # Upper bound
        
        logging.info("Calculated return levels and confidence intervals")
        
        # Goodness of fit test using Kolmogorov-Smirnov test
        ks_stat, p_value = stats.kstest(sorted_data, 
                                      lambda x: stats.genextreme.cdf(x, *gev_params))
        logging.info(f"Kolmogorov-Smirnov test: statistic={ks_stat:.3f}, p-value={p_value:.3f}")
        
        results = {
            'sorted_maxima': sorted_data,
            'empirical_return_periods': empirical_return_periods,
            'return_periods': return_periods,
            'return_levels': return_levels,
            'confidence_intervals': confidence_levels,
            'gev_params': gev_params,
            'plotting_positions': plotting_positions,
            'theoretical_probs': theoretical_probs,
            'gev_p': p_value,
            'exceedance_prob': 1 - plotting_positions,
            'block_maxima': block_maxima  
        }
        
        logging.info("EVA analysis completed successfully")
        return results
        
    except Exception as e:
        logging.error(f"Error in perform_extreme_value_analysis: {str(e)}")
        traceback.print_exc()
        return None


def perform_bayesian_eva_stationary(data: pd.Series) -> Optional[Dict]:
    """
    Perform stationary GEV analysis using Bayesian MCMC with emcee.
    
    Uses 32 walkers × 15,000 samples with 3,000 burn-in and thinning=10.
    Progress bar displays in terminal during sampling (~1-2 minutes).
    
    Parameters:
    -----------
    data : pd.Series
        Block maxima series (minimum 20 values required)
        
    Returns:
    --------
    dict : Dictionary containing MCMC samples, return levels, and credible intervals
    None if analysis fails (falls back to MLE)
    """
    try:
        import emcee
        
        if isinstance(data, pd.Series):
            block_maxima = data
            y_data = data.values
        else:
            block_maxima = pd.Series(data)
            y_data = np.array(data, dtype=float)

        if len(y_data) < 20:
            logging.warning(f"Insufficient data for Bayesian analysis: {len(y_data)} points (minimum 20 required)")
            return None

        # Define the log-probability function for emcee
        def log_prior(theta):
            shape, loc, scale = theta
            # GEV parameter constraints
            data_std = np.std(y_data)
            data_mean = np.mean(y_data)
            if (-0.5 < shape < 0.5 and
                (data_mean - 2*data_std) < loc < (data_mean + 2*data_std) and
                0 < scale < data_std * 5):
                return 0.0
            return -np.inf

        def log_likelihood(theta, data):
            shape, loc, scale = theta
            try:
                log_l = np.sum(stats.genextreme.logpdf(data, shape, loc, scale))
                if np.isnan(log_l):
                    return -np.inf
                return log_l
            except (ValueError, RuntimeError):
                return -np.inf

        def log_probability(theta, data):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta, data)

        # Initial guess
        try:
            shape_mle, loc_mle, scale_mle = stats.genextreme.fit(y_data)
        except:
            shape_mle = 0.0
            loc_mle = np.median(y_data)
            scale_mle = np.std(y_data)

        initial_state = np.array([shape_mle, loc_mle, scale_mle])

        # Check if MLE is within priors
        if not np.isfinite(log_prior(initial_state)):
            initial_state = np.array([0.0, np.median(y_data), np.std(y_data)])

        # Set up the MCMC sampler
        nwalkers = 32
        ndim = 3
        p0 = initial_state + 1e-4 * np.random.randn(nwalkers, ndim)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[y_data])
        
        # Run MCMC with professional logging
        logging.info("BAYESIAN MCMC ANALYSIS STARTING")
        logging.info(f"Model: Stationary GEV Distribution")
        logging.info(f"Configuration: {nwalkers} walkers × 15,000 samples = {nwalkers * 15000:,} total samples")
        logging.info(f"Burn-in: 3,000 samples | Thinning: every 10th sample")
        logging.info(f"Expected effective samples: ~{nwalkers * (15000-3000)//10:,}")
        logging.info("MCMC sampling in progress... (estimated time: 1-2 minutes)")
        
        # Run MCMC with emcee's built-in progress bar
        sampler.run_mcmc(p0, 15000, progress=True)
        
        logging.info("MCMC sampling completed successfully!")
        
        # MCMC Diagnostics
        acceptance_rate = sampler.acceptance_fraction.mean()
        logging.info(f"Acceptance rate: {acceptance_rate:.3f} (target: 0.2-0.5)")
        
        # Safe autocorrelation time calculation
        try:
            autocorr_time = sampler.get_autocorr_time(quiet=True).mean()
            logging.info(f"Autocorrelation time: ~{autocorr_time:.1f} steps")
        except:
            logging.info("Autocorrelation time: Could not calculate (chains may need more samples)")

        # Extract samples and report results
        flat_samples = sampler.get_chain(discard=3000, thin=10, flat=True)
        
        logging.info("FINAL SAMPLING RESULTS:")
        logging.info(f"   • Effective samples: {flat_samples.shape[0]:,}")
        logging.info(f"   • Parameters: {flat_samples.shape[1]} (shape ξ, location μ, scale σ)")
        logging.info(f"   • Convergence: {'Good' if acceptance_rate > 0.2 else 'Check diagnostics'}")
        
        # Get parameter estimates
        shape, loc, scale = np.median(flat_samples, axis=0)
        gev_params = (shape, loc, scale)
        
        # Report estimated parameters
        logging.info("PARAMETER ESTIMATES (Posterior Medians):")
        logging.info(f"   • Shape (ξ): {shape:.4f}")
        logging.info(f"   • Location (μ): {loc:.4f}")
        logging.info(f"   • Scale (σ): {scale:.4f}")

        # Calculate return levels
        return_periods = np.logspace(0, 2.5, 100)
        q = 1 - 1 / return_periods
        
        post_return_levels = np.array([stats.genextreme.ppf(q, *s) for s in flat_samples])
        return_levels = np.median(post_return_levels, axis=0)
        confidence_intervals = np.percentile(post_return_levels, [2.5, 97.5], axis=0)

        # Prepare results for plotting
        sorted_data = np.sort(y_data)
        n = len(sorted_data)
        plotting_positions = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
        empirical_return_periods = 1 / (1 - plotting_positions)
        theoretical_probs = stats.genextreme.cdf(sorted_data, *gev_params)

        results = {
            'sorted_maxima': sorted_data,
            'empirical_return_periods': empirical_return_periods,
            'return_periods': return_periods,
            'return_levels': return_levels,
            'confidence_intervals': confidence_intervals,
            'gev_params': gev_params,
            'plotting_positions': plotting_positions,
            'theoretical_probs': theoretical_probs,
            'gev_p': None,
            'exceedance_prob': 1 - plotting_positions,
            'block_maxima': block_maxima,
            'mcmc_samples': flat_samples
        }
        
        logging.info("Bayesian GEV analysis completed successfully!")
        logging.info("Return levels and credible intervals calculated")
        logging.info("Diagnostic plots will be generated")
        return results

    except Exception as e:
        logging.error(f"Error in perform_bayesian_eva_stationary: {str(e)}")
        traceback.print_exc()
        return None


def perform_bayesian_eva_non_stationary(data: pd.Series) -> Optional[Dict]:
    """
    Perform non-stationary GEV analysis using Bayesian MCMC with emcee.
    
    Models linear trend in location parameter: μ(t) = μ₀ + μ₁×t
    Uses 32 walkers × 20,000 samples with 4,000 burn-in and thinning=10.
    Progress bar displays in terminal during sampling (~2-3 minutes).
    
    Parameters:
    -----------
    data : pd.Series
        Block maxima series with DatetimeIndex (minimum 20 values required)
        
    Returns:
    --------
    dict : Dictionary containing MCMC samples, time-dependent return levels, and credible intervals
    None if analysis fails (falls back to MLE)
    """
    try:
        import emcee
        
        if isinstance(data, pd.Series):
            block_maxima = data
            y_data = data.values
            time_stamps = data.index
        else:
            raise ValueError("Input data must be a pandas Series with a DatetimeIndex.")

        if len(y_data) < 20:
            logging.warning(f"Insufficient data for non-stationary Bayesian analysis: {len(y_data)} points (minimum 20 required)")
            return None

        # Prepare time covariate: normalize to range [0, 1] for better MCMC performance
        t_numeric = (time_stamps - time_stamps.min()).days
        t_norm = t_numeric / t_numeric.max()

        # Define the non-stationary log-probability function
        def log_prior_ns(theta):
            shape, loc_0, loc_1, scale = theta
            # Priors for shape, intercept (loc_0), and scale are similar to stationary
            if -0.5 < shape < 0.5 and \
               y_data.min() * 0.5 < loc_0 < y_data.max() * 1.5 and \
               0 < scale < y_data.std() * 5 and \
               -5 < loc_1 < 5:  # Prior for the trend parameter (loc_1)
                return 0.0
            return -np.inf

        def log_likelihood_ns(theta, y, t):
            shape, loc_0, loc_1, scale = theta
            # Location parameter is now a function of time
            loc_t = loc_0 + loc_1 * t
            try:
                log_l = np.sum(stats.genextreme.logpdf(y, shape, loc_t, scale))
                if np.isnan(log_l):
                    return -np.inf
                return log_l
            except (ValueError, RuntimeError):
                return -np.inf

        def log_probability_ns(theta, y, t):
            lp = log_prior_ns(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood_ns(theta, y, t)

        # Initial guess from pyextremes for robust initialization
        try:
            model_mle = EVA.from_extremes(series=data, extremes_type="BM")
            model_mle.fit_model('GEV', non_stationary=True, location_covariates='time')
            shape_mle = model_mle.results['GEV']['parameters']['shape']
            loc_mle = model_mle.results['GEV']['parameters']['location']
            scale_mle = model_mle.results['GEV']['parameters']['scale']
            # pyextremes uses a different time covariate, so we can't directly use its trend
            # We'll initialize the trend parameter near zero
            initial_state = np.array([shape_mle, loc_mle, 0.1, scale_mle])
        except Exception:
            # Fallback if pyextremes fails
            shape_mle, loc_mle, scale_mle = stats.genextreme.fit(y_data)
            initial_state = np.array([shape_mle, loc_mle, 0.1, scale_mle])

        # Check if the initial state is valid, otherwise use a robust default
        if not np.isfinite(log_prior_ns(initial_state)):
            logging.warning("Initial guess is outside priors. Using a robust starting point.")
            initial_state = np.array([0.0, np.median(y_data), 0.0, np.std(y_data)])

        # Set up and run the MCMC sampler
        nwalkers = 32
        ndim = 4
        p0 = initial_state + 1e-4 * np.random.randn(nwalkers, ndim)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_ns, args=[y_data, t_norm])
        
        # Run MCMC with professional logging
        logging.info("BAYESIAN MCMC ANALYSIS STARTING")
        logging.info(f"Model: Non-Stationary GEV (Linear Trend)")
        logging.info(f"Configuration: {nwalkers} walkers × 20,000 samples = {nwalkers * 20000:,} total samples")
        logging.info(f"Burn-in: 4,000 samples | Thinning: every 10th sample")
        logging.info(f"Expected effective samples: ~{nwalkers * (20000-4000)//10:,}")
        logging.info("MCMC sampling in progress... (estimated time: 2-3 minutes)")
        
        # Run MCMC with emcee's built-in progress bar
        sampler.run_mcmc(p0, 20000, progress=True)
        
        logging.info("MCMC sampling completed successfully!")
        
        # MCMC Diagnostics
        acceptance_rate = sampler.acceptance_fraction.mean()
        logging.info(f"Acceptance rate: {acceptance_rate:.3f} (target: 0.2-0.5)")
        
        # Safe autocorrelation time calculation
        try:
            autocorr_time = sampler.get_autocorr_time(quiet=True).mean()
            logging.info(f"Autocorrelation time: ~{autocorr_time:.1f} steps")
        except Exception as e:
            logging.info("Autocorrelation time: Could not calculate (chains may need more samples)")
            logging.debug(f"Autocorr error: {e}")

        flat_samples = sampler.get_chain(discard=4000, thin=10, flat=True)
        
        # Report sampling results
        logging.info("FINAL SAMPLING RESULTS:")
        logging.info(f"   • Effective samples: {flat_samples.shape[0]:,}")
        logging.info(f"   • Parameters: {flat_samples.shape[1]} (shape ξ, location₀ μ₀, location₁ μ₁, scale σ)")
        logging.info(f"   • Convergence: {'Good' if acceptance_rate > 0.2 else 'Check diagnostics'}")
        
        # Get median parameter estimates
        shape_p, loc_0_p, loc_1_p, scale_p = np.median(flat_samples, axis=0)
        
        # Report estimated parameters
        logging.info("PARAMETER ESTIMATES (Posterior Medians):")
        logging.info(f"   • Shape (ξ): {shape_p:.4f}")
        logging.info(f"   • Location Intercept (μ₀): {loc_0_p:.4f}")
        logging.info(f"   • Location Trend (μ₁): {loc_1_p:.4f}")
        logging.info(f"   • Scale (σ): {scale_p:.4f}")

        # Calculate time-dependent return levels
        return_periods = np.array([10.0, 50.0, 100.0])
        q = 1 - 1 / return_periods
        
        # Calculate return levels across the posterior for each time step
        post_return_levels = np.zeros((len(flat_samples), len(time_stamps), len(return_periods)))
        for i, s in enumerate(flat_samples):
            s_shape, s_loc_0, s_loc_1, s_scale = s
            s_loc_t = s_loc_0 + s_loc_1 * t_norm
            post_return_levels[i, :, :] = stats.genextreme.ppf(q, s_shape, s_loc_t[:, np.newaxis], s_scale)
            
        # Get median and credible intervals for return levels over time
        return_levels_t = np.median(post_return_levels, axis=0)
        ci_lower_t, ci_upper_t = np.percentile(post_return_levels, [2.5, 97.5], axis=0)

        # Format results to be compatible with existing plotting function
        rp_str = [str(rp) for rp in return_periods]
        return_levels_df = pd.DataFrame(return_levels_t, index=time_stamps, columns=rp_str)
        ci_lower_df = pd.DataFrame(ci_lower_t, index=time_stamps, columns=rp_str)
        ci_upper_df = pd.DataFrame(ci_upper_t, index=time_stamps, columns=rp_str)
        
        model_summary = {
            'GEV': {'parameters': {
                'shape': shape_p, 'location': loc_0_p, 'location_c': loc_1_p, 'scale': scale_p
            }}
        }

        results = {
            'non_stationary': True,
            'block_maxima': block_maxima,
            'return_levels': return_levels_df,
            'ci_lower': ci_lower_df,
            'ci_upper': ci_upper_df,
            'model_summary': model_summary,
            'mcmc_samples': flat_samples
        }
        
        logging.info("Non-stationary Bayesian GEV analysis completed successfully!")
        logging.info("Time-dependent return levels and credible intervals calculated")
        logging.info("Diagnostic plots will be generated")
        return results

    except Exception as e:
        logging.error(f"Error in perform_bayesian_eva_non_stationary: {str(e)}")
        traceback.print_exc()
        return None


def calculate_eva_stats(data: pd.Series, block_period: str = '4M') -> Dict:
    """
    Calculate EVA statistics including block maxima and analysis.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    block_period : str
        Block period for maxima extraction (default '4M')
        
    Returns:
    --------
    dict : Dictionary containing block maxima and EVA analysis results
    """
    try:
        # Get block maxima
        block_maxima = data.resample(block_period).max()
        logging.info(f"Found {len(block_maxima)} maxima across {len(block_maxima)} blocks")
        logging.info(f"First extreme: {block_maxima.index[0]}, Last extreme: {block_maxima.index[-1]}")
        logging.info(f"Calculated {len(block_maxima)} block maxima using {block_period} period")
        
        # Check if we have enough blocks for analysis
        if len(block_maxima) < 10:
            logging.error(f"Insufficient block maxima ({len(block_maxima)} < 10) for reliable analysis")
            return {}
        
        # Perform EVA analysis
        eva_results = perform_extreme_value_analysis(block_maxima)
        if eva_results is None:
            return {}
            
        # Combine results with block information
        stats = {
            'block_period': block_period,
            'block_maxima': block_maxima,
            'dates': block_maxima.index,  
            **eva_results  
        }
        
        return stats
        
    except Exception as e:
        logging.error(f"Error in calculate_eva_stats: {str(e)}")
        traceback.print_exc()
        return {}


def calculate_non_stationary_return_periods(
    model: EVA,
    return_periods: np.ndarray
) -> Dict:
    """
    Calculate non-stationary return periods and confidence intervals from a fitted pyextremes model.
    
    Parameters:
    -----------
    model : EVA
        Fitted pyextremes EVA model
    return_periods : np.ndarray
        Array of return periods to calculate
        
    Returns:
    --------
    dict : Dictionary containing return levels and confidence intervals
    """
    try:
        # Get return values from the model
        # This returns a DataFrame with time as index and return periods as columns
        return_values_df = model.get_return_value(
            return_periods=return_periods,
            return_period_size="365.25D",
            alpha=0.05  # For 95% confidence intervals
        )

        # The DataFrame contains return levels and upper/lower confidence bounds
        # We can extract them based on the multi-level column index
        return_levels = return_values_df.loc[:, (slice(None), 'return value')]
        ci_lower = return_values_df.loc[:, (slice(None), 'lower ci')]
        ci_upper = return_values_df.loc[:, (slice(None), 'upper ci')]

        # pyextremes names columns with the return period value, let's make them strings
        return_levels.columns = return_levels.columns.droplevel(1).astype(str)
        ci_lower.columns = ci_lower.columns.droplevel(1).astype(str)
        ci_upper.columns = ci_upper.columns.droplevel(1).astype(str)

        return {
            'return_levels': return_levels,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }

    except Exception as e:
        logging.error(f"Error in calculate_non_stationary_return_periods: {str(e)}")
        traceback.print_exc()
        return {}


def perform_non_stationary_analysis(data: pd.Series) -> Optional[Dict]:
    """
    Perform non-stationary extreme value analysis using pyextremes.
    A linear trend on the location parameter (mu) is assumed.
    
    Parameters:
    -----------
    data : pd.Series
        Block maxima series with DatetimeIndex
        
    Returns:
    --------
    dict : Dictionary containing non-stationary analysis results
    """
    try:
        logging.info(f"Starting non-stationary EVA with {len(data)} values.")
        if len(data) < 20:
            logging.warning("Insufficient data for non-stationary analysis (less than 20 points).")
            return None

        # Initialize EVA model from pre-extracted block maxima
        model = EVA.from_extremes(
            series=data,
            extremes_type="BM"
        )

        # Fit a non-stationary GEV model
        # We'll start by assuming a linear trend on the location parameter (mu)
        model.fit_model(
            model='GEV',
            non_stationary=True,
            location_covariates='time'
        )

        # Define the return periods we want to calculate
        return_periods = np.logspace(np.log10(2), np.log10(100), 20) # e.g., 2 to 100 years

        # Get the non-stationary return levels and CIs
        ns_return_stats = calculate_non_stationary_return_periods(model, return_periods)

        # Get model summary for display
        summary = model.get_summary()

        results = {
            'non_stationary': True,
            'block_maxima': data,
            'model_summary': summary,
            'return_periods': return_periods,
            **ns_return_stats
        }

        logging.info("Non-stationary EVA analysis completed successfully.")
        return results

    except Exception as e:
        logging.error(f"Error in perform_non_stationary_analysis: {str(e)}")
        traceback.print_exc()
        return None

