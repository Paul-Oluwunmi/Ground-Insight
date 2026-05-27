"""
POT Bayesian Analysis
=====================

Bayesian MCMC analysis for GPD models using emcee.
"""

import numpy as np
from scipy.stats import genpareto
import emcee
import logging
import traceback
from typing import Optional, Dict


def perform_bayesian_gpd_stationary(exceedances: np.ndarray) -> Optional[Dict]:
    """
    Perform stationary GPD analysis using Bayesian MCMC with emcee.
    """
    try:
        if len(exceedances) < 20:
            logging.warning("Insufficient data for Bayesian GPD analysis (< 20 points).")
            return None

        # Define the log-probability function for emcee
        def log_prior(theta):
            shape, scale = theta
            # Uniform priors: sensible ranges for GPD parameters
            # Shape: allow reasonable range including heavy and light tails
            # Scale: must be positive
            if -0.5 < shape < 0.5 and 0 < scale < np.std(exceedances) * 5:
                return 0.0
            return -np.inf

        def log_likelihood(theta, data):
            shape, scale = theta
            try:
                # GPD has location=0 for exceedances, so only shape and scale
                log_l = np.sum(genpareto.logpdf(data, shape, loc=0, scale=scale))
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

        # Initial guess from MLE for robust initialization
        shape_mle, loc_mle, scale_mle = genpareto.fit(exceedances)
        initial_state = np.array([shape_mle, scale_mle])  # Only shape and scale for GPD

        # Check if MLE is within priors, if not, use a more generic starting point
        if not np.isfinite(log_prior(initial_state)):
            initial_state = np.array([0.0, np.std(exceedances)])

        # Set up the MCMC sampler
        nwalkers = 32  # Standard number of walkers for robust sampling
        ndim = 2
        # Initialize walkers in a small ball around the starting estimate
        p0 = initial_state + 1e-4 * np.random.randn(nwalkers, ndim)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[exceedances])
        
        # Run MCMC with built-in progress bar
        logging.info(f"Starting Bayesian MCMC for stationary GPD model...")
        logging.info(f"Configuration: {nwalkers} walkers × 15,000 samples = {nwalkers * 15000:,} total samples")
        logging.info(f"Burn-in: 3,000 samples | Thinning: every 10th sample")
        logging.info(f"Expected effective samples: ~{nwalkers * (15000-3000)//10:,}")
        logging.info("MCMC sampling in progress... (this may take 1-2 minutes)")
        
        # Run MCMC with emcee's built-in progress bar
        sampler.run_mcmc(p0, 15000, progress=True)
        
        logging.info("MCMC sampling complete!")
        logging.info(f"Acceptance fraction per walker: {sampler.acceptance_fraction.mean():.3f} (target: 0.2-0.5)")
        logging.info(f"Autocorrelation time: ~{sampler.get_autocorr_time(quiet=True).mean():.1f} steps")

        # Discard burn-in and flatten the chains
        flat_samples = sampler.get_chain(discard=3000, thin=10, flat=True)
        
        # Report final sampling statistics
        logging.info(f"Final sampling results:")
        logging.info(f"   • Total effective samples: {flat_samples.shape[0]:,}")
        logging.info(f"   • Parameters estimated: {flat_samples.shape[1]} (shape ξ, scale σ)")
        logging.info(f"   • Chain convergence: {'Good' if sampler.acceptance_fraction.mean() > 0.2 else 'Check'}")
        
        # Get the posterior median as the parameter estimate
        shape, scale = np.median(flat_samples, axis=0)
        gpd_params = (shape, 0, scale)  # Add location=0 for consistency

        # Calculate return levels from the posterior samples for uncertainty
        return_periods = np.logspace(0, 2.5, 100)
        q = 1 - 1 / return_periods
        
        # Calculate return levels for each sample in the chain
        post_return_levels = np.array([genpareto.ppf(q, s[0], 0, s[1]) for s in flat_samples])
        
        # Get the median and 95% credible interval for the return levels
        return_levels = np.median(post_return_levels, axis=0)
        confidence_intervals = np.percentile(post_return_levels, [2.5, 97.5], axis=0)

        # For plotting, we need sorted data and plotting positions
        sorted_data = np.sort(exceedances)
        n = len(sorted_data)
        plotting_positions = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
        empirical_return_periods = 1 / (1 - plotting_positions)
        theoretical_probs = genpareto.cdf(sorted_data, *gpd_params)

        results = {
            'empirical_levels': sorted_data,
            'empirical_return_periods': empirical_return_periods,
            'theoretical_periods': return_periods,
            'theoretical_levels': return_levels,
            'ci_lower': confidence_intervals[0],
            'ci_upper': confidence_intervals[1],
            'gpd_params': gpd_params,
            'plotting_positions': plotting_positions,
            'theoretical_probs': theoretical_probs,
            'gev_p': None,  # p-value is a frequentist concept
            'exceedance_prob': 1 - plotting_positions,
            'mcmc_samples': flat_samples  # For potential diagnostics
        }
        
        return results

    except Exception as e:
        logging.error(f"Error in perform_bayesian_gpd_stationary: {str(e)}")
        traceback.print_exc()
        return None


def perform_bayesian_gpd_non_stationary(exceedances: np.ndarray, times: np.ndarray) -> Optional[Dict]:
    """
    Perform non-stationary GPD analysis using Bayesian MCMC with emcee.
    Assumes a linear trend on the scale parameter.
    """
    try:
        if len(exceedances) < 20:
            logging.warning("Insufficient data for non-stationary Bayesian GPD analysis (< 20 points).")
            return None

        # Normalize time to range [0, 1] for better MCMC performance
        t_norm = (times - np.min(times)) / (np.max(times) - np.min(times))

        # Define the non-stationary log-probability function
        def log_prior_ns(theta):
            shape, scale_0, scale_1 = theta
            # Priors for shape and scale parameters
            if -0.5 < shape < 0.5 and \
               0 < scale_0 < np.std(exceedances) * 5 and \
               -5 < scale_1 < 5:  # Prior for the trend parameter (scale_1)
                return 0.0
            return -np.inf

        def log_likelihood_ns(theta, data, t):
            shape, scale_0, scale_1 = theta
            # Scale parameter is now a function of time
            scale_t = scale_0 + scale_1 * t
            # Ensure all scale values are positive
            if np.any(scale_t <= 0):
                return -np.inf
            try:
                log_l = np.sum(genpareto.logpdf(data, shape, 0, scale_t))
                if np.isnan(log_l):
                    return -np.inf
                return log_l
            except (ValueError, RuntimeError):
                return -np.inf

        def log_probability_ns(theta, data, t):
            lp = log_prior_ns(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood_ns(theta, data, t)

        # Initial guess from MLE for robust initialization
        shape_mle, loc_mle, scale_mle = genpareto.fit(exceedances)
        initial_state = np.array([shape_mle, scale_mle, 0.1])  # shape, scale_0, scale_1

        # Check if the initial state is valid, otherwise use a robust default
        if not np.isfinite(log_prior_ns(initial_state)):
            initial_state = np.array([0.0, np.std(exceedances), 0.0])

        # Set up and run the MCMC sampler
        nwalkers = 32  # Standard number of walkers for robust sampling
        ndim = 3
        p0 = initial_state + 1e-4 * np.random.randn(nwalkers, ndim)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_ns, args=[exceedances, t_norm])
        
        # Run MCMC with built-in progress bar
        logging.info(f"Starting Bayesian MCMC for non-stationary GPD model...")
        logging.info(f"Configuration: {nwalkers} walkers × 20,000 samples = {nwalkers * 20000:,} total samples")
        logging.info(f"Burn-in: 4,000 samples | Thinning: every 10th sample")
        logging.info(f"Expected effective samples: ~{nwalkers * (20000-4000)//10:,}")
        logging.info("MCMC sampling in progress... (this may take 2-3 minutes)")
        
        # Run MCMC with emcee's built-in progress bar
        sampler.run_mcmc(p0, 20000, progress=True)
        
        logging.info("MCMC sampling complete!")
        logging.info(f"Acceptance fraction per walker: {sampler.acceptance_fraction.mean():.3f} (target: 0.2-0.5)")
        logging.info(f"Autocorrelation time: ~{sampler.get_autocorr_time(quiet=True).mean():.1f} steps")

        flat_samples = sampler.get_chain(discard=4000, thin=10, flat=True)
        
        # Report final sampling statistics
        logging.info(f"Final sampling results:")
        logging.info(f"   • Total effective samples: {flat_samples.shape[0]:,}")
        logging.info(f"   • Parameters estimated: {flat_samples.shape[1]} (shape ξ, scale₀ σ₀, scale₁ σ₁)")
        logging.info(f"   • Chain convergence: {'Good' if sampler.acceptance_fraction.mean() > 0.2 else 'Check'}")
        
        # Get median parameter estimates
        shape_p, scale_0_p, scale_1_p = np.median(flat_samples, axis=0)

        # Calculate time-dependent return levels
        return_periods = np.array([10.0, 50.0, 100.0])
        q = 1 - 1 / return_periods
        
        # Calculate return levels across the posterior for each time step
        post_return_levels = np.zeros((len(flat_samples), len(times), len(return_periods)))
        for i, s in enumerate(flat_samples):
            s_shape, s_scale_0, s_scale_1 = s
            s_scale_t = s_scale_0 + s_scale_1 * t_norm
            # Ensure positive scales
            s_scale_t = np.maximum(s_scale_t, 1e-6)
            post_return_levels[i, :, :] = genpareto.ppf(q, s_shape, 0, s_scale_t[:, np.newaxis])
            
        # Get median and credible intervals for return levels over time
        return_levels_t = np.median(post_return_levels, axis=0)
        ci_lower_t, ci_upper_t = np.percentile(post_return_levels, [2.5, 97.5], axis=0)

        # For plotting compatibility, we need to provide stationary-style results
        # Use the median time point for plotting
        median_time_idx = len(times) // 2
        
        # Get return levels at median time point
        return_levels_median = return_levels_t[median_time_idx, :]
        ci_lower_median = ci_lower_t[median_time_idx, :]
        ci_upper_median = ci_upper_t[median_time_idx, :]
        
        # Create empirical data for plotting
        sorted_data = np.sort(exceedances)
        n = len(sorted_data)
        plotting_positions = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
        empirical_return_periods = 1 / (1 - plotting_positions)
        
        # GPD parameters at median time
        scale_median = scale_0_p + scale_1_p * t_norm[median_time_idx]
        gpd_params = (shape_p, 0, scale_median)
        
        # Calculate theoretical probabilities
        theoretical_probs = genpareto.cdf(sorted_data, *gpd_params)

        results = {
            'empirical_levels': sorted_data,
            'empirical_return_periods': empirical_return_periods,
            'theoretical_periods': return_periods,
            'theoretical_levels': return_levels_median,
            'ci_lower': ci_lower_median,
            'ci_upper': ci_upper_median,
            'gpd_params': gpd_params,
            'plotting_positions': plotting_positions,
            'theoretical_probs': theoretical_probs,
            'gev_p': None,
            'exceedance_prob': 1 - plotting_positions,
            'mcmc_samples': flat_samples,
            'non_stationary': True,
            'return_levels_full': return_levels_t,
            'ci_lower_full': ci_lower_t,
            'ci_upper_full': ci_upper_t,
            'times': times,
            'scale_params': (scale_0_p, scale_1_p)
        }
        
        return results

    except Exception as e:
        logging.error(f"Error in perform_bayesian_gpd_non_stationary: {str(e)}")
        traceback.print_exc()
        return None

