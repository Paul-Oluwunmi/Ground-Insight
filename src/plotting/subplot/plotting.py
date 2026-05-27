"""
Subplot Plotting Functions
===========================

Functions for creating plot traces and figures.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from .constants import SUBPLOT_COLORS


def get_nz_season_with_color(date):
    """Get the NZ season with color for a given date."""
    month = date.month
    if month in [12, 1, 2]:
        return "Summer", "#FF6B35"  # Warm orange-red
    elif month in [3, 4, 5]:
        return "Autumn", "#8B4513"  # Saddle brown
    elif month in [6, 7, 8]:
        return "Winter", "#4682B4"  # Steel blue
    else:  # 9, 10, 11
        return "Spring", "#32CD32"  # Lime green


def create_trace_enhanced(x, y, name, color, interval):
    """Create an enhanced plotly trace with seasonal hover information."""
    # Get seasons with colors for all dates
    seasons_with_colors = [get_nz_season_with_color(pd.Timestamp(d)) for d in x]
    season_texts = [f'<span style="color:{color}">{season}</span>' for season, color in seasons_with_colors]
    
    return go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=name,
        line=dict(
            color=color,
            width=2,
            shape='linear'
        ),
        marker=dict(
            size=4,
            color=color,
            symbol='circle',
            line=dict(width=1, color='white'),
            opacity=0.8
        ),
        hovertemplate=(
            f"<b>{name}</b><br>" +
            "<b>Date: %{x}</b><br>" +
            "<b>Level: %{y:.2f} m</b><br>" +
            "<b>Season: %{customdata}</b><br>" +
            "<b>Interval: " + interval + "</b><br>" +
            "<extra></extra>"
        ),
        customdata=season_texts,
        connectgaps=False,
        opacity=0.9,
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=color,
            font=dict(size=10, color="black")
        )
    )


def create_plot_figure(selected_well, resampled_data, interval, plot_index):
    """
    Create a plotly figure for a single subplot.
    
    Parameters:
    -----------
    selected_well : str or None
        Selected well name
    resampled_data : dict or pd.DataFrame
        Resampled data dictionary or DataFrame
    interval : str
        Selected time interval
    plot_index : int
        Index of the plot being created
        
    Returns:
    --------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    if selected_well and resampled_data:
        # Convert resampled_data to DataFrame if needed
        if isinstance(resampled_data, dict):
            plot_data = pd.DataFrame(resampled_data)
        elif isinstance(resampled_data, list):
            # If it's a list, try to convert to DataFrame
            plot_data = pd.DataFrame(resampled_data)
        elif isinstance(resampled_data, pd.DataFrame):
            plot_data = resampled_data
        else:
            # Unknown type, try to convert
            try:
                plot_data = pd.DataFrame(resampled_data)
            except Exception:
                # If conversion fails, return empty plot
                plot_data = pd.DataFrame()
        
        # Ensure plot_data is a DataFrame and has required columns
        if not isinstance(plot_data, pd.DataFrame) or plot_data.empty:
            # Return empty plot
            fig.update_layout(
                title=dict(
                    text=f"Well {selected_well} - No Data Available",
                    x=0.5,
                    font=dict(size=16, color='#6C757D', family='Arial, sans-serif')
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[{
                    'text': 'No data available for selected well and interval',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': dict(size=14, color='#6C757D')
                }]
            )
            return fig
        
        # Check if required columns exist
        if 'DateTime' not in plot_data.columns or selected_well not in plot_data.columns:
            # Return empty plot with error message
            fig.update_layout(
                title=dict(
                    text=f"Well {selected_well} - Missing Data Columns",
                    x=0.5,
                    font=dict(size=16, color='#6C757D', family='Arial, sans-serif')
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[{
                    'text': f'Required columns not found. Available: {list(plot_data.columns)}',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': dict(size=14, color='#6C757D')
                }]
            )
            return fig
        
        # Handle missing data points
        well_data = plot_data[['DateTime', selected_well]].dropna()
        
        if len(well_data) > 0:
            # Convert to numpy arrays for plotting
            x_data = np.array(well_data['DateTime'])
            y_data = np.array(well_data[selected_well])
            
            # Convert interval codes to display text
            interval_display = {
                'original': 'Original Data',
                'hourly': 'Hourly',  
                'daily': 'Daily',
                'monthly': 'Monthly',
                '6-monthly': '6-Monthly',
                'yearly': 'Yearly'  
            }
            
            # Create enhanced trace with seasonal hover
            trace = create_trace_enhanced(
                x_data, y_data, selected_well, 
                SUBPLOT_COLORS[plot_index], interval_display.get(interval, interval)
            )
            fig.add_trace(trace)
            
            # Update layout with enhanced styling
            fig.update_layout(
                title=dict(
                    text=f"Well {selected_well} ({interval_display.get(interval, interval)})",
                    x=0.5,
                    font=dict(size=16, color='#2C3E50', family='Arial, sans-serif', weight='bold')  
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',  
                font=dict(color='#495057', family='Arial, sans-serif'),
                margin=dict(l=60, r=20, t=60, b=60),
                showlegend=False,
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(52, 73, 94, 0.15)',
                    gridwidth=0.8,
                    tickfont=dict(color='#495057', family='Arial, sans-serif', weight='bold'),  
                    zeroline=False,
                    showline=True,
                    linecolor='#BDC3C7',
                    linewidth=1,
                    title=dict(text="Date" if plot_index == 3 else "", font=dict(size=14, weight='bold', color='#2C3E50'))
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(52, 73, 94, 0.15)',
                    gridwidth=0.8,
                    tickfont=dict(color='#495057', family='Arial, sans-serif', weight='bold'),  
                    zeroline=False,
                    showline=True,
                    linecolor='#BDC3C7',
                    linewidth=1,
                    title=dict(text="Groundwater Level (m)", font=dict(size=14, weight='bold', color='#2C3E50'))
                ),
                hovermode='closest'
            )
        else:
            # Empty plot when no data available
            fig.update_layout(
                title=dict(
                    text=f"Well {selected_well} - No Data Available",
                    x=0.5,
                    font=dict(size=16, color='#6C757D', family='Arial, sans-serif')
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[{
                    'text': 'No data available for selected well and interval',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': dict(size=14, color='#6C757D')
                }]
            )
    else:
        # Default empty plot styling
        fig.update_layout(
            title=dict(
                text=f"Plot {plot_index + 1} - Select a Well",
                x=0.5,
                font=dict(size=16, color='#6C757D', family='Arial, sans-serif')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[{
                'text': 'Please select a well from the dropdown above',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False,
                'font': dict(size=14, color='#6C757D')
            }]
        )
    
    return fig

