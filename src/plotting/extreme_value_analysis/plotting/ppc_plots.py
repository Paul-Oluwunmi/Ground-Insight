"""
Posterior Predictive Check (PPC) plotting functions
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Tuple, Optional
import pandas as pd


def generate_posterior_predictive_checks(
    block_maxima: pd.Series,
    stats: Dict,
    analysis_type: str,
    n_samples: int = 100
) -> Tuple[go.Figure, Dict]:
    """
    Generate posterior predictive check plots.
    
    Parameters:
    -----------
    block_maxima : pd.Series
        Observed block maxima data
    stats : Dict
        Dictionary containing MCMC samples and fitted parameters
    analysis_type : str
        Type of analysis ('stationary' or 'non_stationary')
    n_samples : int
        Number of posterior samples to use for predictions
        
    Returns:
    --------
    Tuple[go.Figure, Dict]
        Plotly figure and dictionary with PPC statistics
    """
    try:
        from scipy import stats as scipy_stats
        
        observed = block_maxima.values
        n_obs = len(observed)
        
        # Get parameter estimates
        params = stats.get('parameters', {})
        loc = params.get('loc', np.mean(observed))
        scale = params.get('scale', np.std(observed))
        shape = params.get('shape', 0.1)
        
        # Generate posterior predictive samples
        ppc_samples = []
        for _ in range(n_samples):
            # Add some uncertainty to parameters
            loc_sample = loc + np.random.normal(0, scale * 0.1)
            scale_sample = scale * np.exp(np.random.normal(0, 0.1))
            shape_sample = shape + np.random.normal(0, 0.05)
            
            # Generate samples from GEV
            sample = scipy_stats.genextreme.rvs(
                c=-shape_sample, loc=loc_sample, scale=scale_sample, size=n_obs
            )
            ppc_samples.append(sample)
        
        ppc_samples = np.array(ppc_samples)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Observed vs Predicted Distribution', 'Q-Q Plot')
        )
        
        # Plot 1: Histogram comparison
        fig.add_trace(
            go.Histogram(x=observed, name='Observed', nbinsx=20,
                        marker_color='steelblue', opacity=0.7),
            row=1, col=1
        )
        
        # Add predicted distribution envelope
        ppc_flat = ppc_samples.flatten()
        fig.add_trace(
            go.Histogram(x=ppc_flat, name='Predicted', nbinsx=20,
                        marker_color='coral', opacity=0.5),
            row=1, col=1
        )
        
        # Plot 2: Q-Q plot
        observed_sorted = np.sort(observed)
        theoretical_quantiles = scipy_stats.genextreme.ppf(
            np.linspace(0.01, 0.99, n_obs), c=-shape, loc=loc, scale=scale
        )
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=observed_sorted,
                      mode='markers', name='Q-Q Points',
                      marker=dict(color='steelblue', size=8)),
            row=1, col=2
        )
        
        # Add 1:1 line
        min_val = min(theoretical_quantiles.min(), observed_sorted.min())
        max_val = max(theoretical_quantiles.max(), observed_sorted.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='1:1 Line',
                      line=dict(color='red', dash='dash')),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(text=f"Posterior Predictive Checks - {analysis_type.replace('_', ' ').title()}",
                      x=0.5, font=dict(size=16)),
            height=400,
            width=900,
            showlegend=True,
            plot_bgcolor='white'
        )
        
        fig.update_xaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Observed Quantiles", row=1, col=2)
        
        # Calculate PPC statistics
        ppc_stats = {
            'mean_observed': float(np.mean(observed)),
            'mean_predicted': float(np.mean(ppc_samples)),
            'std_observed': float(np.std(observed)),
            'std_predicted': float(np.std(ppc_samples)),
            'p_value': float(np.mean(np.mean(ppc_samples, axis=1) > np.mean(observed)))
        }
        
        return fig, ppc_stats
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error generating PPC: {str(e)}",
                         xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig, {'error': str(e)}


__all__ = ['generate_posterior_predictive_checks']

