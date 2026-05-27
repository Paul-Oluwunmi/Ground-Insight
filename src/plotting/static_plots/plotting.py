"""
Static Plots Plotting Functions
================================

Functions for creating interactive static plots.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import List, Optional

from .constants import PLOT_HEIGHT, PLOT_WIDTH, INTERVAL_COLORS
from .cache import get_cached_well_data, cache_well_data
from .data_processing import preprocess_data_for_speed, calculate_all_intervals_for_well


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


def create_trace(x, y, name, color, group, visible=True, show_legend=True, mode='markers'):
    """Create a plotly trace with styling optimized for performance."""
    # Get seasons with colors for all dates
    seasons_with_colors = [get_nz_season_with_color(pd.Timestamp(d)) for d in x]
    season_texts = [f'<span style="color:{color}">{season}</span>' for season, color in seasons_with_colors]
    
    return go.Scatter(
        x=x,  
        y=y, 
        name=name,
        visible=visible,
        showlegend=show_legend,
        mode=mode,
        line=dict(
            color=color, 
            width=1.5,  # Slightly thicker
            shape='linear'
        ),
        marker=dict(
            size=5,  # Balanced size 
            color=color,
            symbol='circle',
            line=dict(width=1, color='white'),  # White border for contrast
            opacity=0.8  # Subtle transparency
        ),
        hovertemplate=(
            f"<b>{name}</b><br>" +
            "<b>Date: %{x}</b><br>" +
            "<b>Level: %{y:.2f} m</b><br>" +
            "<b>Season: %{customdata}</b><br>" +
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


def plot_interactive_hydrograph(data: pd.DataFrame, well_columns: List[str]) -> Optional[go.Figure]:
    """Create an interactive hydrograph with well selection dropdown - optimized for speed."""
    
    try:
        # Fast data preprocessing for maximum speed
        data = preprocess_data_for_speed(data)
        
        # Fast data hash for caching
        data_hash = str(hash(str(data.shape) + str(data.columns.tolist())))
        
        # Create figure with minimal overhead
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        traces = []
        well_trace_counts = {}  # Track how many traces each well gets

        # Process each well with maximum speed
        for well in well_columns:
            # Check cache first
            cached_intervals = get_cached_well_data(data_hash, well)
            
            if cached_intervals is not None:
                intervals = cached_intervals
            else:
                # Calculate intervals - optimized function
                intervals = calculate_all_intervals_for_well(data, well)
                
                # Cache results
                if intervals:
                    cache_well_data(data_hash, well, intervals)
            
            if not intervals:
                continue
            
            trace_count = 0
            
            # Create traces efficiently - only create what we have data for
            # First subplot: Original data
            if 'original' in intervals:
                x, y = intervals['original']
                traces.append((
                    create_trace(x, y, "Original Data", INTERVAL_COLORS['original'], well, False, True, 'lines'),
                    1
                ))
                trace_count += 1
            
            # Second subplot: All resampled intervals in specific order
            interval_order = ['hourly', 'daily', 'monthly', '6-monthly', 'yearly']
            for interval in interval_order:
                if interval in intervals:
                    x, y = intervals[interval]
                    # Map interval names to display names
                    if interval == '6-monthly':
                        display_name = "6-Monthly"
                        actual_interval = '6-monthly'
                    else:
                        display_name = interval.capitalize()
                        actual_interval = interval
                    
                    mode = 'lines+markers' if actual_interval in ['monthly', '6-monthly', 'yearly'] else 'lines'
                    traces.append((
                        create_trace(x, y, display_name, INTERVAL_COLORS[actual_interval], well, False, True, mode),
                        2
                    ))
                    trace_count += 1
            
            well_trace_counts[well] = trace_count

        if not traces:
            return None
        
        # Add all traces at once - faster than individual adds
        for trace, row in traces:
            fig.add_trace(trace, row=row, col=1)

        # Create dropdown buttons efficiently
        buttons = []
        
        # Default button
        buttons.append({
            'label': "**Select Well**",
            'method': "restyle",
            'args': [{"visible": [False] * len(traces)}],
            'name': 'none'
        })
        
        # Well selection buttons
        trace_idx = 0
        for well in well_columns:
            if well in well_trace_counts:
                trace_count = well_trace_counts[well]
                visibility = [False] * len(traces)
                visibility[trace_idx:trace_idx + trace_count] = [True] * trace_count
                
                buttons.append({
                    'label': well,
                    'method': "restyle",
                    'args': [{"visible": visibility}]
                })
                trace_idx += trace_count

        # Simplified dropdown menu
        dropdown_menu = [{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'active': 0,
            'x': 1.00, 'xanchor': 'right',
            'y': 1.15, 'yanchor': 'top',
            'bgcolor': 'rgba(255, 255, 255, 0.95)',
            'bordercolor': '#E9ECEF',
            'borderwidth': 2,
            'font': dict(size=12, color='#495057', family='Arial, sans-serif')
        }]

        # Fast layout update - minimal styling for speed
        start_time = time.time()
        
        # Use minimal layout for large datasets to prevent hanging
        if len(traces) > 50:  # If we have many traces, use minimal layout
            try:
                fig.update_layout(
                    title=dict(
                        text="<b>Interactive Groundwater Hydrograph</b>",
                        font=dict(size=24, color='#2C3E50', family='Arial, sans-serif'),
                        x=0.5,
                        y=0.98,
                        xanchor='center',
                        yanchor='top'
                    ),
                    height=PLOT_HEIGHT,
                    width=PLOT_WIDTH,
                    updatemenus=dropdown_menu,
                    hovermode="closest",
                    showlegend=True,
                    paper_bgcolor='#F8F9FA',  # Light gray background
                    plot_bgcolor='white',
                    legend=dict(
                        title=dict(
                            text="Time Intervals",
                            font=dict(size=14, color='#2C3E50', family='Arial, sans-serif')
                        ),
                        bgcolor='rgba(255, 255, 255, 0.95)',
                        bordercolor='#E9ECEF',
                        borderwidth=1,
                        font=dict(size=11, color='#495057', family='Arial, sans-serif'),
                        itemsizing='constant',
                        orientation='h',  # Horizontal layout
                        yanchor='bottom',
                        y=1.02,
                        xanchor='center',
                        x=0.5,
                        traceorder='normal',
                        groupclick='toggleitem'  # Click on legend item toggles only that item
                    ),
                    margin=dict(t=120, b=80, l=80, r=80),  # Increased top margin for annotations
                    yaxis=dict(domain=[0.55, 1]),
                    yaxis2=dict(domain=[0.08, 0.47]),
                    uirevision=True,
                    dragmode='zoom',
                    selectdirection='any',
                    font=dict(family='Arial, sans-serif', size=12, color='#495057')
                )
            except Exception as e:
                # Fallback to absolute minimal layout
                fig.update_layout(
                    title="Interactive Groundwater Hydrograph",
                    height=PLOT_HEIGHT,
                    width=PLOT_WIDTH,
                    updatemenus=dropdown_menu
                )
        else:
            # Standard layout for smaller datasets
            try:
                fig.update_layout(
                    title=dict(
                        text="<b>Interactive Groundwater Hydrograph</b>",
                        font=dict(size=24, color='#2C3E50', family='Arial, sans-serif'),
                        x=0.5,
                        y=0.98,
                        xanchor='center',
                        yanchor='top'
                    ),
                    height=PLOT_HEIGHT,
                    width=PLOT_WIDTH,
                    updatemenus=dropdown_menu,
                    hovermode="closest",
                    showlegend=True,
                    paper_bgcolor='#F8F9FA',
                    plot_bgcolor='white',
                    legend=dict(
                        title=dict(
                            text="Time Intervals",
                            font=dict(size=14, color='#2C3E50', family='Arial, sans-serif')
                        ),
                        bgcolor='rgba(255, 255, 255, 0.95)',
                        bordercolor='#E9ECEF',
                        borderwidth=1,
                        font=dict(size=11, color='#495057', family='Arial, sans-serif'),
                        itemsizing='constant',
                        orientation='h',  # Horizontal layout
                        yanchor='bottom',
                        y=1.02,
                        xanchor='center',
                        x=0.5,
                        traceorder='normal',
                        groupclick='toggleitem'  # Click on legend item toggles only that item
                    ),
                    margin=dict(t=140, b=100, l=100, r=50),  # Increased top margin for annotations
                    yaxis=dict(domain=[0.55, 1]),
                    yaxis2=dict(domain=[0.08, 0.47]),
                    font=dict(family='Arial, sans-serif', size=12, color='#495057'),
                )
            except Exception as e:
                # Fallback to absolute minimal layout
                fig.update_layout(
                    title="Interactive Groundwater Hydrograph",
                    height=PLOT_WIDTH,
                    width=PLOT_WIDTH,
                    updatemenus=dropdown_menu
                )

        # Check timeout
        if time.time() - start_time > 30:  # 30 second timeout
            pass

        # Fast axes configuration - batch updates to prevent hanging
        try:
            # Configure axes 
            fig.update_xaxes(
                title_text="",  # No title for first subplot
                showgrid=True,
                gridcolor='rgba(52, 73, 94, 0.15)',
                gridwidth=0.8,
                zeroline=False,
                showline=True,
                linecolor='#BDC3C7',
                linewidth=1,
                tickfont=dict(size=12, color='#495057', family='Arial, sans-serif', weight='bold'),
                row=1, col=1
            )
            
            fig.update_xaxes(
                title_text="Date",  # Title only for second subplot
                title_font=dict(size=14, color='#2C3E50', family='Arial, sans-serif', weight='bold'),
                showgrid=True,
                gridcolor='rgba(52, 73, 94, 0.15)',
                gridwidth=0.8,
                zeroline=False,
                showline=True,
                linecolor='#BDC3C7',
                linewidth=1,
                tickfont=dict(size=12, color='#495057', family='Arial, sans-serif', weight='bold'),
                row=2, col=1
            )
            
            # Configure y-axes with shared label
            fig.update_yaxes(
                showgrid=True,
                gridcolor='rgba(52, 73, 94, 0.15)',
                gridwidth=0.8,
                zeroline=False,
                showline=True,
                linecolor='#BDC3C7',
                linewidth=1,
                title_font=dict(size=14, color='#2C3E50', family='Arial, sans-serif', weight='bold'),
                tickfont=dict(size=12, color='#495057', family='Arial, sans-serif', weight='bold')
            )
            
            # Add shared y-axis label
            fig.update_layout(
                yaxis_title=dict(
                    text="Groundwater Level",
                    font=dict(size=14, color='#2C3E50', family='Arial, sans-serif', weight='bold')
                ),
                yaxis2_title=dict(
                    text="Groundwater Level",
                    font=dict(size=14, color='#2C3E50', family='Arial, sans-serif', weight='bold')
                )
            )
        except Exception as e:
            pass
        
        # Fast date axis setup - single update to prevent hanging
        try:
            fig.update_layout(
                xaxis=dict(type='date'),
                xaxis2=dict(type='date')
            )
        except Exception as e:
            pass
        
        return fig

    except Exception as e:
        return None

