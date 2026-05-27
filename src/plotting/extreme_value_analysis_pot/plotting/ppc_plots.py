"""
POT Posterior Predictive Checks
================================

Functions for generating posterior predictive checks for Bayesian GPD models.
"""

import numpy as np
from scipy.stats import genpareto
import plotly.graph_objects as go
import logging
import traceback

from ..constants import COLORS


def generate_posterior_predictive_checks_gpd(stats, observed_exceedances, analysis_type, n_sim=100):
    """
    Generate posterior predictive checks for Bayesian GPD models.
    """
    try:
        np.random.seed(42)
        n_obs = len(observed_exceedances)
        sim_data = []
        samples = stats.get('mcmc_samples')
        if samples is None:
            return go.Figure().update_layout(title="No Bayesian samples available"), {}

        # Randomly select n_sim samples from the posterior
        idx = np.random.choice(samples.shape[0], min(n_sim, samples.shape[0]), replace=False)
        
        if analysis_type == 'bayesian_stationary':
            for i in idx:
                shape, scale = samples[i]
                sim = genpareto.rvs(shape, loc=0, scale=scale, size=n_obs)
                sim_data.append(sim)
        elif analysis_type == 'bayesian_non_stationary':
            # For non-stationary, need time index
            t = np.linspace(0, 1, n_obs)  # Normalized time
            for i in idx:
                shape, scale_0, scale_1 = samples[i]
                scales = scale_0 + scale_1 * t
                scales = np.maximum(scales, 1e-6)  # Ensure positive scales
                sim = genpareto.rvs(shape, loc=0, scale=scales, size=n_obs)
                sim_data.append(sim)
        else:
            return go.Figure().update_layout(title="PPC only for Bayesian models"), {}

        sim_data = np.array(sim_data)

        # Create ECDF overlay plot
        fig = go.Figure()
        
        # Calculate percentiles for confidence bands
        x_range = np.linspace(0, np.max(sim_data), 200)
        ecdf_percentiles = []
        
        for x_val in x_range:
            y_vals = []
            for sim in sim_data:
                y_vals.append(np.mean(sim <= x_val))
            ecdf_percentiles.append(y_vals)
        
        ecdf_percentiles = np.array(ecdf_percentiles)
        
        # Add confidence bands (5th and 95th percentiles)
        p5 = np.percentile(ecdf_percentiles, 5, axis=1)
        p95 = np.percentile(ecdf_percentiles, 95, axis=1)
        median = np.percentile(ecdf_percentiles, 50, axis=1)
        
        # Add confidence band
        fig.add_trace(go.Scatter(
            x=x_range,
            y=p5,
            mode='lines',
            line=dict(color='rgba(128, 128, 128, 0)', width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=p95,
            mode='lines',
            line=dict(color='rgba(128, 128, 128, 0)', width=0),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.2)',
            name='90% Posterior Interval',
            hovertemplate="Value: %{x:.2f}<br>ECDF Range: %{y:.3f}<extra></extra>"
        ))
        
        # Add median line
        fig.add_trace(go.Scatter(
            x=x_range,
            y=median,
            mode='lines',
            line=dict(color=COLORS['secondary'], width=2, dash='dash'),
            name='Posterior Median ECDF',
            hovertemplate="Value: %{x:.2f}<br>Median ECDF: %{y:.3f}<extra></extra>"
        ))
        
        # Add observed ECDF
        obs_sorted = np.sort(observed_exceedances)
        obs_ecdf = np.linspace(0, 1, n_obs)
        
        fig.add_trace(go.Scatter(
            x=obs_sorted,
            y=obs_ecdf,
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=4, color=COLORS['primary']),
            name='Observed Exceedances ECDF',
            hovertemplate="Value: %{x:.2f}<br>Observed ECDF: %{y:.3f}<extra></extra>"
        ))
        
        # Professional styling
        analysis_title = "Stationary" if analysis_type == 'bayesian_stationary' else "Non-Stationary"
        
        fig.update_layout(
            title=dict(
                text=f"<b>Posterior Predictive Check: ECDF Comparison</b><br><i>{analysis_title} Bayesian GPD Model</i>",
                x=0.5,
                font=dict(size=16, weight='bold')
            ),
            xaxis=dict(
                title=dict(
                    text="Exceedance Values",
                    font=dict(size=12, weight='bold')
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickfont=dict(size=10, weight='bold'),
                zeroline=True,
                zerolinecolor='rgba(128, 128, 128, 0.3)'
            ),
            yaxis=dict(
                title=dict(
                    text="Empirical Cumulative Distribution",
                    font=dict(size=12, weight='bold')
                ),
                range=[0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickfont=dict(size=10, weight='bold'),
                tickformat='.2f',
                zeroline=True,
                zerolinecolor='rgba(128, 128, 128, 0.3)'
            ),
            height=600,
            width=1000,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                y=0.02,
                xanchor="right",
                x=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor="black",
                borderwidth=1,
                font=dict(size=11, weight='bold')
            ),
            margin=dict(l=80, r=80, t=80, b=60)
        )

        # Compute summary statistics
        obs_mean = float(np.mean(observed_exceedances))
        sim_means = np.mean(sim_data, axis=1)
        obs_var = float(np.var(observed_exceedances))
        sim_vars = np.var(sim_data, axis=1)
        obs_q95 = float(np.quantile(observed_exceedances, 0.95))
        sim_q95 = np.quantile(sim_data, 0.95, axis=1)
        obs_min = float(np.min(observed_exceedances))
        sim_min = np.min(sim_data, axis=1)
        obs_max = float(np.max(observed_exceedances))
        sim_max = np.max(sim_data, axis=1)

        summary = {
            'Mean': f"{obs_mean:.3f}",
            'Mean (90% CI)': f"({np.percentile(sim_means, 5):.3f}, {np.percentile(sim_means, 95):.3f})",
            'Variance': f"{obs_var:.3f}",
            'Variance (90% CI)': f"({np.percentile(sim_vars, 5):.3f}, {np.percentile(sim_vars, 95):.3f})",
            '95th Percentile': f"{obs_q95:.3f}",
            '95th Percentile (90% CI)': f"({np.percentile(sim_q95, 5):.3f}, {np.percentile(sim_q95, 95):.3f})",
            'Minimum': f"{obs_min:.3f}",
            'Minimum (90% CI)': f"({np.percentile(sim_min, 5):.3f}, {np.percentile(sim_min, 95):.3f})",
            'Maximum': f"{obs_max:.3f}",
            'Maximum (90% CI)': f"({np.percentile(sim_max, 5):.3f}, {np.percentile(sim_max, 95):.3f})",
        }

        return fig, summary

    except Exception as e:
        logging.error(f"Error in generate_posterior_predictive_checks_gpd: {str(e)}")
        traceback.print_exc()
        return go.Figure().update_layout(title=f"Error creating PPC: {e}"), {}

