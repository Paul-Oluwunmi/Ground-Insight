"""
Plotting functions for Extreme Value Analysis

This module contains plotting functions for EVA visualizations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import genextreme
from scipy import stats as scipy_stats
import logging
import traceback
import base64
import io
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from ..constants import SEASON_COLORS, COLORS

# Import plotting functions from submodules
from .time_series import create_time_series_plot


def analyse_seasonal_patterns(block_maxima: pd.Series) -> dict:
    """Analyse seasonal patterns in block maxima"""
    try:
        # Ensure we're working with a Series that has a datetime index
        if not isinstance(block_maxima.index, pd.DatetimeIndex):
            block_maxima.index = pd.to_datetime(block_maxima.index)

        # Create DataFrame with the values and their dates
        block_maxima_df = pd.DataFrame({
            'value': block_maxima,
            'month': block_maxima.index.month,
            'date': block_maxima.index
        })

        # Define NZ seasons (months are 1-12)
        nz_seasons = {
            'Summer': [12, 1, 2],
            'Autumn': [3, 4, 5],
            'Winter': [6, 7, 8],
            'Spring': [9, 10, 11]
        }
        
        def get_nz_season(month):
            for season, months in nz_seasons.items():
                if month in months:
                    return season
            return 'Unknown'
            
        block_maxima_df['season'] = block_maxima_df['month'].map(get_nz_season)

        # Calculate statistics for each season
        seasonal_stats = {'seasons': {}}
        
        for season in nz_seasons.keys():
            season_data = block_maxima_df[block_maxima_df['season'] == season]
            if not season_data.empty:
                seasonal_stats['seasons'][season] = {
                    'count': len(season_data),
                    'mean': season_data['value'].mean(),
                    'max': season_data['value'].max(),
                    'min': season_data['value'].min(),
                    'std': season_data['value'].std(),
                    'dates': season_data['date'].tolist(),
                    'values': season_data['value'].tolist()
                }

        seasonal_stats['total_points'] = len(block_maxima_df)
        return seasonal_stats
        
    except Exception as e:
        logging.error(f"Error in analyse_seasonal_patterns: {str(e)}")
        return {'seasons': {}, 'total_points': 0}


def add_seasonal_annotation(fig, seasonal_stats, data_type):
    """Add seasonal analysis to the plot"""
    try:
        # Collect all dates and values from each season
        all_dates = []
        all_values = []
        for season, stats in seasonal_stats['seasons'].items():
            if stats['count'] > 0:
                all_dates.extend(stats['dates'])
                all_values.extend(stats['values'])

        # Sort by date
        sorted_indices = np.argsort(all_dates)
        all_dates = np.array(all_dates)[sorted_indices]
        all_values = np.array(all_values)[sorted_indices]

        # Add time series line
        fig.add_trace(
            go.Scatter(
                x=all_dates,
                y=all_values,
                mode='lines',
                name='Time Series',
                line=dict(color='gray', width=1),
                showlegend=True
            ),
            row=1, col=1
        )

        # Add seasonal maxima points
        for season, stats in seasonal_stats['seasons'].items():
            if stats['count'] > 0:
                fig.add_trace(
                    go.Scatter(
                        x=stats['dates'],
                        y=stats['values'],
                        mode='markers',
                        name=f'{season} Maxima (n={stats["count"]})',
                        marker=dict(
                            color=SEASON_COLORS[season],
                            size=8,
                            symbol='circle'
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )

        # Update axes for seasonal plot
        fig.update_xaxes(title_text="Date", row=1, col=1, tickformat="%Y")
        
        value_label = 'Rainfall (mm)' if data_type == 'rain' else 'Groundwater Level (m)'
        fig.update_yaxes(title_text=f"Maximum {value_label}", row=1, col=1)

        return fig

    except Exception as e:
        logging.error(f"Error in add_seasonal_annotation: {str(e)}")
        return fig


def create_eva_plots(stats: Dict, well: str, data_type: str, 
                     analysis_type: str = 'stationary', 
                     warning: Optional[str] = None) -> go.Figure:
    """Create EVA plots for stationary or non-stationary analysis."""
    if analysis_type == 'stationary':
        return create_stationary_eva_plots(stats, well, data_type, warning=warning)
    elif analysis_type == 'bayesian_non_stationary':
        return create_non_stationary_eva_plots(stats, well, data_type, warning=warning)
    elif analysis_type == 'bayesian_stationary':
        return create_stationary_eva_plots(stats, well, data_type, warning=warning)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")


def create_stationary_eva_plots(stats: Dict, well: str, data_type: str, 
                                warning: Optional[str] = None) -> go.Figure:
    """Create EVA plots with seasonal analysis"""
    try:
        if not stats:
            raise ValueError("No statistics available for plotting")
            
        # Define labels based on data type
        if data_type == 'rain':
            value_label = 'Rainfall (mm)'
            return_level_label = 'Return Level (mm)'
        else:
            value_label = 'Groundwater Level (m)'
            return_level_label = 'Return Level (m)'
            
        # Get block maxima with dates
        block_maxima = stats.get('block_maxima')
        if block_maxima is None:
            raise ValueError("Block maxima not found in statistics")
        
        # Get seasonal categorization
        seasonal_stats = analyse_seasonal_patterns(block_maxima)
        if not seasonal_stats or not seasonal_stats.get('seasons'):
            raise ValueError("No seasonal data available for plotting")

        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '<b>Seasonal Analysis</b>', '<b>Probability Density</b>',
                '<b>Return Level Plot</b>', '<b>Exceedance Probability</b>',
                '<b>Q-Q Plot</b>', '<b>P-P Plot</b>'
            ),
            specs=[[{'b': 0.05}, {'b': 0.05}],
                [{'b': 0.05}, {'b': 0.05}],
                [{'b': 0.05}, {'b': 0.05}]],
            horizontal_spacing=0.15,
            vertical_spacing=0.1
        )

        # 1. Seasonal Analysis (row=1, col=1)
        add_seasonal_annotation(fig, seasonal_stats, data_type)
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray', row=1, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray', row=1, col=1)

        # 2. Probability Density (row=1, col=2)
        def get_optimal_bins(data):
            if len(data) <= 1:
                return 10
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            if iqr == 0:
                return 10
            bin_width = 2 * iqr / (len(data) ** (1/3))
            data_range = np.max(data) - np.min(data)
            if bin_width <= 0 or data_range <= 0:
                return 10
            n_bins = int(np.ceil(data_range / bin_width))
            return max(10, min(n_bins, 50))

        all_values = np.concatenate([s_stats['values'] for s_stats in seasonal_stats['seasons'].values() if 'values' in s_stats])
        
        if len(all_values) == 0:
            fig = go.Figure()
            fig.update_layout(title_text="Not enough seasonal data to generate plots.")
            return fig

        n_bins = get_optimal_bins(all_values)
        hist, bin_edges = np.histogram(all_values, bins=n_bins, density=True)

        # Add seasonal histograms
        for season, season_stats in seasonal_stats['seasons'].items():
            if season_stats['count'] > 0:
                fig.add_trace(
                    go.Histogram(
                        x=season_stats['values'],
                        name=f'{season} (n={season_stats["count"]})',
                        marker=dict(color=SEASON_COLORS[season], line=dict(color='white', width=0.5)),
                        opacity=0.6,
                        nbinsx=n_bins,
                        histnorm='probability density',
                        showlegend=True
                    ),
                    row=1, col=2
                )

        # Calculate GEV PDF
        data_range = np.ptp(all_values)
        x_range = np.linspace(np.min(all_values) - 0.2 * data_range, np.max(all_values) + 0.2 * data_range, 500)
        shape, loc, scale = stats['gev_params']
        pdf = genextreme.pdf(x_range, shape, loc, scale)
        hist_max = np.max(hist)
        pdf_max = np.max(pdf)
        scaling_factor = hist_max / pdf_max if pdf_max > 0 else 1
        pdf_scaled = pdf * scaling_factor

        fig.add_trace(
            go.Scatter(x=x_range, y=pdf_scaled, mode='lines', name=f'GEV PDF (ξ={shape:.3f})',
                      line=dict(color='black', width=2.5), showlegend=True),
            row=1, col=2
        )

        fig.update_xaxes(title_text=value_label, row=1, col=2, showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        fig.update_yaxes(title_text="Probability Density", row=1, col=2, rangemode='tozero', showgrid=True, gridwidth=0.5, gridcolor='lightgray')

        # 3. Return Level Plot (row=2, col=1)
        data_range = max(stats['return_levels']) - min(stats['sorted_maxima'])
        y_min = min(stats['sorted_maxima']) - 0.02 * data_range
        y_max = max(stats['return_levels']) + 0.02 * data_range

        rp_zones = [
            (1, 2, 'rgba(240, 248, 255, 0.3)', 'Frequent (1-2 years)'),
            (2, 5, 'rgba(176, 224, 230, 0.3)', 'Common (2-5 years)'),
            (5, 10, 'rgba(135, 206, 235, 0.3)', 'Uncommon (5-10 years)'),
            (10, 25, 'rgba(70, 130, 180, 0.3)', 'Rare (10-25 years)'),
            (25, 50, 'rgba(0, 0, 139, 0.3)', 'Very Rare (25-50 years)'),
            (50, np.max(stats['return_periods']), 'rgba(25, 25, 112, 0.3)', 'Extreme (50+ years)')
        ]

        for (start, end, color, label) in rp_zones:
            fig.add_trace(
                go.Scatter(x=[start, start, end, end], y=[y_min, y_max, y_max, y_min],
                          fill='toself', fillcolor=color, line=dict(width=0),
                          name=label, hoverinfo='name', showlegend=True),
                row=2, col=1
            )

        # Add seasonal return level points
        for season, season_stats_data in seasonal_stats['seasons'].items():
            if season_stats_data['count'] > 0:
                season_values = season_stats_data['values']
                season_dates = season_stats_data['dates']
                season_periods = [
                    stats['empirical_return_periods'][list(stats['sorted_maxima']).index(val)]
                    for val in season_values
                ]
                formatted_dates = [date.strftime('%d %b %Y') for date in season_dates]
                
                fig.add_trace(
                    go.Scatter(x=season_periods, y=season_values, mode='markers',
                              name=f'{season} (n={season_stats_data["count"]})',
                              marker=dict(color=SEASON_COLORS[season], size=8, line=dict(color='white', width=1)),
                              showlegend=True, customdata=formatted_dates,
                              hovertemplate=f"{season}<br>Date: %{{customdata}}<br>{value_label}: %{{y:.2f}}<extra></extra>"),
                    row=2, col=1
                )

        # Add GEV fit line
        unit = 'm' if data_type == 'gwl' else 'mm'
        fig.add_trace(
            go.Scatter(x=stats['return_periods'], y=stats['return_levels'], mode='lines',
                      name='GEV Fit', line=dict(color='black', width=2), showlegend=True),
            row=2, col=1
        )

        # Add confidence intervals
        ci_lower, ci_upper = stats['confidence_intervals']
        fig.add_trace(
            go.Scatter(x=stats['return_periods'], y=ci_lower, mode='lines',
                      line=dict(color='gray', dash='dash'), name='95% CI', showlegend=True),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=stats['return_periods'], y=ci_upper, mode='lines',
                      line=dict(color='gray', dash='dash'), fill='tonexty',
                      fillcolor='rgba(128, 128, 128, 0.2)', showlegend=False),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Return Period (years)", type="log", range=[np.log10(1), np.log10(100)],
                        row=2, col=1, gridcolor='lightgray', gridwidth=0.5, showgrid=True)
        fig.update_yaxes(title_text=value_label, range=[y_min, y_max], row=2, col=1,
                        gridcolor='lightgray', gridwidth=0.5, showgrid=True)

        # 4. Exceedance Probability (row=2, col=2)
        exceedance_range = np.linspace(0, 1, 100)
        fitted_values = genextreme.ppf(1 - exceedance_range, *stats['gev_params'])

        fig.add_trace(
            go.Scatter(x=exceedance_range * 100, y=fitted_values, mode='lines',
                      name='GEV Fit', line=dict(color='black', width=2), showlegend=False),
            row=2, col=2
        )

        for season, season_stats_data in seasonal_stats['seasons'].items():
            if season_stats_data['count'] > 0:
                season_values = season_stats_data['values']
                season_dates = season_stats_data['dates']
                season_probs = [
                    stats['exceedance_prob'][list(stats['sorted_maxima']).index(val)] * 100
                    for val in season_values
                ]
                formatted_dates = [date.strftime('%d %b %Y') for date in season_dates]
                
                fig.add_trace(
                    go.Scatter(x=season_probs, y=season_values, mode='markers',
                              name=f'{season}', marker=dict(color=SEASON_COLORS[season], size=8),
                              showlegend=False, customdata=formatted_dates,
                              hovertemplate=f"{season}<br>Date: %{{customdata}}<br>Exceedance: %{{x:.1f}}%<br>{value_label}: %{{y:.2f}}<extra></extra>"),
                    row=2, col=2
                )

        fig.update_xaxes(title_text="Exceedance Probability (%)", range=[0, 100], row=2, col=2,
                        showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        fig.update_yaxes(title_text=value_label, row=2, col=2, showgrid=True, gridwidth=0.5, gridcolor='lightgray')

        # 5. Q-Q Plot (row=3, col=1)
        theoretical_quantiles = genextreme.ppf(stats['plotting_positions'], *stats['gev_params'])
        all_vals = np.concatenate([theoretical_quantiles, stats['sorted_maxima']])
        min_val = np.min(all_vals) - 0.05 * (np.max(all_vals) - np.min(all_vals))
        max_val = np.max(all_vals) + 0.05 * (np.max(all_vals) - np.min(all_vals))

        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                      line=dict(color='black', dash='dash', width=2), name='1:1 Line', showlegend=False),
            row=3, col=1
        )

        for season, season_stats_data in seasonal_stats['seasons'].items():
            if season_stats_data['count'] > 0:
                season_values = season_stats_data['values']
                season_dates = season_stats_data['dates']
                season_quantiles = [
                    theoretical_quantiles[list(stats['sorted_maxima']).index(val)]
                    for val in season_values
                ]
                formatted_dates = [date.strftime('%d %b %Y') for date in season_dates]
                
                fig.add_trace(
                    go.Scatter(x=season_quantiles, y=season_values, mode='markers',
                              name=f'{season}', marker=dict(color=SEASON_COLORS[season], size=8),
                              showlegend=False, customdata=formatted_dates,
                              hovertemplate=f"{season}<br>Date: %{{customdata}}<br>Theoretical: %{{x:.2f}}<br>Sample: %{{y:.2f}}<extra></extra>"),
                    row=3, col=1
                )

        fig.update_xaxes(title_text="Theoretical Quantiles", row=3, col=1, showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        fig.update_yaxes(title_text="Sample Quantiles", row=3, col=1, showgrid=True, gridwidth=0.5, gridcolor='lightgray')

        # 6. P-P Plot (row=3, col=2)
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      line=dict(color='black', dash='dash', width=2), showlegend=False),
            row=3, col=2
        )

        for season, season_stats_data in seasonal_stats['seasons'].items():
            if season_stats_data['count'] > 0:
                season_values = season_stats_data['values']
                season_dates = season_stats_data['dates']
                season_empirical = [
                    stats['plotting_positions'][list(stats['sorted_maxima']).index(val)]
                    for val in season_values
                ]
                season_theoretical = [
                    stats['theoretical_probs'][list(stats['sorted_maxima']).index(val)]
                    for val in season_values
                ]
                formatted_dates = [date.strftime('%d %b %Y') for date in season_dates]
                
                fig.add_trace(
                    go.Scatter(x=season_empirical, y=season_theoretical, mode='markers',
                              name=f'{season}', marker=dict(color=SEASON_COLORS[season], size=8),
                              showlegend=False, customdata=formatted_dates,
                              hovertemplate=f"{season}<br>Date: %{{customdata}}<br>Empirical: %{{x:.3f}}<br>Theoretical: %{{y:.3f}}<extra></extra>"),
                    row=3, col=2
                )

        fig.update_xaxes(title_text="Empirical Probability", row=3, col=2, showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        fig.update_yaxes(title_text="Theoretical Probability", row=3, col=2, showgrid=True, gridwidth=0.5, gridcolor='lightgray')

        # Update layout
        title_text = f"Extreme Value Analysis - {well}"
        if warning:
            title_text += f"<br><i>{warning}</i>"
            
        fig.update_layout(
            title=dict(text=title_text, x=0.5, y=0.99, font=dict(size=18, weight='bold')),
            height=1400,
            width=1400,
            showlegend=True,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            margin=dict(l=200, r=120, t=120, b=60),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.10, font=dict(size=12, weight='bold'))
        )

        # Add GEV parameters annotation
        gev_params_text = (
            f"GEV Parameters:<br>ξ={stats['gev_params'][0]:.3f}<br>"
            f"μ={stats['gev_params'][1]:.3f}<br>"
            f"σ={stats['gev_params'][2]:.3f}"
        )
        if stats.get('gev_p') is not None:
            gev_params_text += f"<br>p={stats['gev_p']:.3f}"
        else:
            gev_params_text += "<br>(Bayesian Fit)"

        fig.add_annotation(
            text=gev_params_text,
            xref="paper", yref="paper",
            x=1.06, y=0.98,
            showarrow=False,
            font=dict(size=10, weight='bold'),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            align="left"
        )

        return fig
        
    except Exception as e:
        logging.error(f"Error in create_stationary_eva_plots: {str(e)}")
        traceback.print_exc()
        return go.Figure()


def create_non_stationary_eva_plots(stats: Dict, well: str, data_type: str, 
                                    warning: Optional[str] = None) -> go.Figure:
    """Create plots for non-stationary EVA results."""
    try:
        if not stats or not ('return_levels' in stats and not stats['return_levels'].empty):
            raise ValueError("Statistics for non-stationary analysis are not available.")

        fig = go.Figure()
        
        if data_type == 'rain':
            value_label = 'Rainfall (mm)'
            return_level_label = 'Return Level (mm)'
        else:
            value_label = 'Groundwater Level (m)'
            return_level_label = 'Return Level (m)'

        # Plot observed block maxima
        block_maxima = stats['block_maxima']
        fig.add_trace(go.Scatter(
            x=block_maxima.index,
            y=block_maxima.values,
            mode='markers',
            name='Observed Block Maxima',
            marker=dict(color=COLORS['primary'], opacity=0.6)
        ))

        # Plot non-stationary return levels
        return_levels = stats['return_levels']
        ci_lower = stats['ci_lower']
        ci_upper = stats['ci_upper']
        
        for rp in [10.0, 50.0, 100.0]:
            rp_str = str(rp)
            if rp_str in return_levels.columns:
                fig.add_trace(go.Scatter(
                    x=return_levels.index, y=return_levels[rp_str],
                    mode='lines', name=f'{int(rp)}-Year Return Level', line=dict(width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=ci_lower.index, y=ci_lower[rp_str],
                    mode='lines', line=dict(width=1, dash='dash'),
                    showlegend=False, name=f'{int(rp)}-Year CI'
                ))
                fig.add_trace(go.Scatter(
                    x=ci_upper.index, y=ci_upper[rp_str],
                    mode='lines', line=dict(width=1, dash='dash'),
                    fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)',
                    showlegend=False, name=f'{int(rp)}-Year CI'
                ))

        title_text = f"Non-Stationary Extreme Value Analysis - {well}"
        if warning:
            title_text += f"<br><i>{warning}</i>"
            
        fig.update_layout(
            title=title_text,
            xaxis_title="Date",
            yaxis_title=return_level_label,
            legend_title="Return Period",
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            height=600
        )
        
        # Add model summary annotation
        summary = stats.get('model_summary', {})
        if summary:
            params = summary.get('GEV', {}).get('parameters', {})
            mu_0 = params.get('location', 'N/A')
            mu_1 = params.get('location_c', 'N/A')
            sigma = params.get('scale', 'N/A')
            xi = params.get('shape', 'N/A')

            annotation_text = (
                f"<b>Non-Stationary GEV Parameters:</b><br>"
                f"μ(t) = μ₀ + μ₁*t<br>"
                f"μ₀ (intercept) = {mu_0:.3f}<br>"
                f"μ₁ (trend) = {mu_1:.4f}<br>"
                f"σ (scale) = {sigma:.3f}<br>"
                f"ξ (shape) = {xi:.3f}"
            )

            fig.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=1.02, y=1.0,
                showarrow=False,
                font=dict(size=10, weight='bold'),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                align="left",
                xanchor="left",
                yanchor="top"
            )

        return fig

    except Exception as e:
        logging.error(f"Error in create_non_stationary_eva_plots: {str(e)}")
        traceback.print_exc()
        return go.Figure()


def create_corner_plot(stats: Dict, analysis_type: str) -> go.Figure:
    """Create enhanced corner plot for GEV MCMC diagnostics."""
    try:
        import corner
        
        if 'mcmc_samples' not in stats or stats['mcmc_samples'] is None:
            return go.Figure().update_layout(
                title="Corner Plot (Not available)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )

        samples = stats['mcmc_samples']
        
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples)
        
        if analysis_type == 'bayesian_stationary':
            labels = ["Shape (ξ)", "Location (μ)", "Scale (σ)"]
            title_text = "Corner Plot: Stationary GEV Parameters"
        elif analysis_type == 'bayesian_non_stationary':
            labels = ["Shape (ξ)", "Intercept (μ₀)", "Trend (μ₁)", "Scale (σ)"]
            title_text = "Corner Plot: Non-Stationary GEV Parameters"
        else:
            return go.Figure()

        if samples.size == 0:
            return go.Figure().update_layout(title="No MCMC samples available")

        plt.style.use('default')
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.5
        
        fig_mpl = corner.corner(
            samples, 
            labels=labels, 
            bins=35,
            truths=np.median(samples, axis=0),
            truth_color='red',
            label_kwargs=dict(fontsize=18, weight='bold'),
            title_kwargs=dict(fontsize=16, weight='bold'),
            hist_kwargs=dict(alpha=1.0, color='#1E88E5', linewidth=1.0),
            scatter_kwargs=dict(alpha=0.7, s=1.5, color='#1E88E5', rasterized=False),
            contour_kwargs=dict(colors=['#1E88E5', '#D32F2F', '#FF9800'], linewidths=[2.5, 2.0, 1.5], alpha=1.0),
            contourf_kwargs=dict(colors=['#1E88E5'], alpha=0.4, levels=18),
            plot_density=True,
            plot_datapoints=True,
            plot_contours=True,
            fill_contours=True,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.4f',
            smooth=1.8,
            smooth1d=1.8,
            max_n_ticks=6
        )
        
        fig_mpl.subplots_adjust(left=0.10, bottom=0.10, right=0.90, top=0.90, wspace=0.28, hspace=0.28)
        fig_mpl.set_size_inches(15, 13)
        
        buf = io.BytesIO()
        fig_mpl.savefig(buf, format='png', dpi=220, bbox_inches='tight', facecolor='white',
                       edgecolor='none', pad_inches=0.35, transparent=False)
        buf.seek(0)
        plt.close(fig_mpl)
        
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        fig_go = go.Figure()
        fig_go.add_layout_image(
            dict(
                source=f'data:image/png;base64,{encoded_image}',
                xref="paper", yref="paper",
                x=0, y=1,
                sizex=1, sizey=1,
                xanchor="left", yanchor="top",
                sizing="contain",
                layer="below"
            )
        )
        
        fig_go.update_layout(
            title=dict(
                text=f"<b>{title_text}</b><br><i>Diagonal: Marginal Histograms | Off-diagonal: Parameter Correlations</i>",
                x=0.5,
                font=dict(size=22, weight='bold', color='#1565C0')
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False, range=[0, 1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False, range=[0, 1]),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=900,
            width=1350,
            margin=dict(l=60, r=370, t=130, b=60),
            dragmode=False,
            showlegend=False
        )
        
        # Add parameter summary
        summary_text = []
        for i, label in enumerate(labels):
            param_samples = samples[:, i]
            median_val = np.median(param_samples)
            ci_lower = np.percentile(param_samples, 16)
            ci_upper = np.percentile(param_samples, 84)
            summary_text.append(f"{label}: {median_val:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

        fig_go.add_annotation(
            text="<b>GEV Parameter Summary (68% CIs):</b><br><br>" + "<br>".join(summary_text),
            xref="paper", yref="paper",
            x=1.02, y=0.40,
            showarrow=False,
            font=dict(size=11, family="monospace"),
            bgcolor="rgba(240, 248, 255, 0.95)",
            bordercolor="#1565C0",
            borderwidth=2,
            align="left",
            xanchor="left",
            yanchor="top"
        )
        
        return fig_go

    except Exception as e:
        logging.error(f"Error creating corner plot: {str(e)}")
        traceback.print_exc()
        return go.Figure().update_layout(title=f"Error creating corner plot: {e}")


def generate_posterior_predictive_checks(stats, observed_data, analysis_type, n_sim=100):
    """Generate posterior predictive checks for Bayesian GEV models."""
    try:
        np.random.seed(42)
        n_obs = len(observed_data)
        sim_data = []
        samples = stats.get('mcmc_samples')
        
        if samples is None:
            return go.Figure().update_layout(title="No Bayesian samples available"), {}

        if not isinstance(samples, np.ndarray):
            samples = np.array(samples)
        
        if samples.size == 0:
            return go.Figure().update_layout(title="No MCMC samples available"), {}

        idx = np.random.choice(samples.shape[0], min(n_sim, samples.shape[0]), replace=False)
        
        if analysis_type == 'bayesian_stationary':
            for i in idx:
                shape, loc, scale = samples[i]
                sim = genextreme.rvs(-shape, loc=loc, scale=scale, size=n_obs)
                sim_data.append(sim)
        elif analysis_type == 'bayesian_non_stationary':
            t = np.arange(n_obs)
            for i in idx:
                shape, mu0, mu1, scale = samples[i]
                locs = mu0 + mu1 * t
                sim = genextreme.rvs(-shape, loc=locs, scale=scale, size=n_obs)
                sim_data.append(sim)
        else:
            return go.Figure().update_layout(title="PPC only for Bayesian models"), {}
            
        sim_data = np.array(sim_data)

        fig = go.Figure()
        
        x_range = np.linspace(np.min(sim_data), np.max(sim_data), 200)
        ecdf_percentiles = []
        
        for x_val in x_range:
            y_vals = [np.mean(sim <= x_val) for sim in sim_data]
            ecdf_percentiles.append(y_vals)
        
        ecdf_percentiles = np.array(ecdf_percentiles)
        
        p5 = np.percentile(ecdf_percentiles, 5, axis=1)
        p95 = np.percentile(ecdf_percentiles, 95, axis=1)
        median = np.percentile(ecdf_percentiles, 50, axis=1)
        
        fig.add_trace(go.Scatter(x=x_range, y=p5, mode='lines',
                                line=dict(color='rgba(128, 128, 128, 0)', width=0),
                                showlegend=False, hoverinfo='skip'))
        
        fig.add_trace(go.Scatter(x=x_range, y=p95, mode='lines',
                                line=dict(color='rgba(128, 128, 128, 0)', width=0),
                                fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)',
                                name='90% Posterior Interval'))
        
        fig.add_trace(go.Scatter(x=x_range, y=median, mode='lines',
                                line=dict(color=COLORS['secondary'], width=2, dash='dash'),
                                name='Posterior Median ECDF'))
        
        obs_sorted = np.sort(observed_data)
        obs_ecdf = np.linspace(0, 1, n_obs)
        
        fig.add_trace(go.Scatter(x=obs_sorted, y=obs_ecdf, mode='lines+markers',
                                line=dict(color=COLORS['primary'], width=3),
                                marker=dict(size=4, color=COLORS['primary']),
                                name='Observed Data ECDF'))
        
        analysis_title = "Stationary" if analysis_type == 'bayesian_stationary' else "Non-Stationary"
        
        fig.update_layout(
            title=dict(
                text=f"<b>Posterior Predictive Check: ECDF Comparison</b><br><i>{analysis_title} Bayesian GEV Model</i>",
                x=0.5, font=dict(size=16, weight='bold')
            ),
            xaxis=dict(title="Block Maxima Values", showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
            yaxis=dict(title="Empirical Cumulative Distribution", range=[0, 1], showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)'),
            height=600, width=1000,
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            hovermode='x unified',
            showlegend=True,
            legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98,
                       bgcolor='rgba(255, 255, 255, 0.8)', bordercolor="black", borderwidth=1)
        )

        # Compute summary statistics
        obs_mean = float(np.mean(observed_data))
        sim_means = np.mean(sim_data, axis=1)
        obs_var = float(np.var(observed_data))
        sim_vars = np.var(sim_data, axis=1)
        obs_q95 = float(np.quantile(observed_data, 0.95))
        sim_q95 = np.quantile(sim_data, 0.95, axis=1)
        obs_min = float(np.min(observed_data))
        sim_min = np.min(sim_data, axis=1)
        obs_max = float(np.max(observed_data))
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
        logging.error(f"Error in generate_posterior_predictive_checks: {str(e)}")
        traceback.print_exc()
        return go.Figure().update_layout(title=f"Error generating PPC: {e}"), {}


__all__ = [
    'create_time_series_plot',
    'create_eva_plots',
    'create_stationary_eva_plots',
    'create_non_stationary_eva_plots',
    'create_corner_plot',
    'generate_posterior_predictive_checks',
    'analyse_seasonal_patterns',
    'add_seasonal_annotation'
]
