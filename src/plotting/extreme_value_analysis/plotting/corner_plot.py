"""
Corner plot functions for MCMC diagnostics
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict


def create_corner_plot(stats: Dict, analysis_type: str) -> go.Figure:
    """
    Create corner plot for MCMC diagnostics showing parameter distributions.
    
    Parameters:
    -----------
    stats : Dict
        Dictionary containing MCMC samples and statistics
    analysis_type : str
        Type of analysis ('stationary' or 'non_stationary')
        
    Returns:
    --------
    go.Figure
        Plotly figure with corner plot
    """
    try:
        # Get samples from stats
        samples = stats.get('samples', {})
        if not samples:
            fig = go.Figure()
            fig.add_annotation(text="No MCMC samples available", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return fig
        
        # Get parameter names and values
        param_names = list(samples.keys())
        n_params = len(param_names)
        
        if n_params == 0:
            fig = go.Figure()
            fig.add_annotation(text="No parameters to plot", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return fig
        
        # Create subplots for corner plot
        fig = make_subplots(
            rows=n_params, cols=n_params,
            horizontal_spacing=0.05,
            vertical_spacing=0.05
        )
        
        for i, param_i in enumerate(param_names):
            for j, param_j in enumerate(param_names):
                row, col = i + 1, j + 1
                
                if i == j:
                    # Diagonal: histogram of parameter
                    values = np.array(samples[param_i]).flatten()
                    fig.add_trace(
                        go.Histogram(x=values, nbinsx=30, name=param_i,
                                   marker_color='steelblue', showlegend=False),
                        row=row, col=col
                    )
                elif i > j:
                    # Lower triangle: 2D scatter/density
                    x_vals = np.array(samples[param_j]).flatten()
                    y_vals = np.array(samples[param_i]).flatten()
                    fig.add_trace(
                        go.Scatter(x=x_vals, y=y_vals, mode='markers',
                                 marker=dict(size=2, opacity=0.3, color='steelblue'),
                                 showlegend=False),
                        row=row, col=col
                    )
        
        # Update layout
        fig.update_layout(
            title=dict(text=f"Corner Plot - {analysis_type.replace('_', ' ').title()}", 
                      x=0.5, font=dict(size=16)),
            height=200 * n_params,
            width=200 * n_params,
            showlegend=False,
            plot_bgcolor='white'
        )
        
        # Add axis labels
        for i, param in enumerate(param_names):
            fig.update_xaxes(title_text=param, row=n_params, col=i+1)
            fig.update_yaxes(title_text=param, row=i+1, col=1)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating corner plot: {str(e)}", 
                         xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig


__all__ = ['create_corner_plot']

