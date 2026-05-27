"""
Overlay Callbacks
=================

Dash callbacks for the overlay visualization.
"""

import polars as pl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gc
from dash import Input, Output

from .constants import STATIC_LAYOUT, AXIS_CONFIG, WELL_COLORS, BAR_WIDTH_MAP
from .data_processing import get_resampled_data


def register_callbacks(app, data_lazy, well_columns, data_folder, rainfall_filename):
    """
    Register all callbacks for the overlay Dash application.
    
    Parameters:
    -----------
    app : dash.Dash
        Dash application instance
    data_lazy : pl.LazyFrame
        Lazy polars DataFrame containing groundwater data
    well_columns : list
        List of well column names
    data_folder : str or None
        Path to data folder for loading rainfall data
    rainfall_filename : str or None
        Name of rainfall data file
    """
    @app.callback(
        Output('overlay-plot', 'figure'),
        [Input('well-selector', 'value'),
         Input('interval-selector', 'value')]
    )
    def update_plot(selected_wells, interval):
        """
        Update plot based on well selection and interval.
        
        Parameters:
        -----------
        selected_wells : list
            List of selected well names
        interval : str
            Selected time interval
            
        Returns:
        --------
        go.Figure
            Updated plotly figure
        """
        if not selected_wells:
            return {}
        
        try:
            # Get resampled data (includes both rainfall and groundwater)
            well_columns_tuple = tuple(well_columns)
            df = get_resampled_data(data_lazy, well_columns_tuple, interval)
            if df is None:
                raise ValueError("Failed to get resampled data")
            
            # Extract datetime array
            times = df['DateTime'].to_numpy()
            
            # Extract rainfall from resampled data
            rainfall_array = df['mm_Rain'].to_numpy().copy()
            # Fill NaN values with 0
            rainfall_array = np.nan_to_num(rainfall_array, 0)
            
            # Get groundwater columns
            groundwater_data = df.select(['DateTime'] + selected_wells)
            
            # Create figure with independent subplots
            fig = make_subplots(
                rows=2, 
                cols=1,
                row_heights=[0.5, 0.5],
                shared_xaxes=False,  # Independent x-axes
                vertical_spacing=0.15,  # More space between plots
                subplot_titles=('<b>Rainfall</b>', '<b>Groundwater Levels</b>')
            )

            # Add rainfall bars (using resampled data)
            max_rainfall = np.max(rainfall_array) if len(rainfall_array) > 0 else 1
            
            # Calculate bar width based on interval
            bar_width = BAR_WIDTH_MAP.get(interval, 24*60*60*1000)  # Default to 1 day
            
            # Add rainfall trace
            fig.add_trace(
                go.Bar(
                    x=times,
                    y=rainfall_array,
                    name='Rainfall',
                    marker=dict(
                        color='rgba(0, 114, 189, 0.6)',  # Blue with transparency
                        line=dict(
                            color='rgba(0, 114, 189, 1.0)',  # Solid blue outline
                            width=1
                        )
                    ),
                    width=bar_width,
                    offset=0,
                    hovertemplate="<b>Rainfall</b><br>Date: %{x}<br>Amount: %{y:.1f} mm<extra></extra>",
                ),
                row=1, col=1
            )
            
            # Add well traces with better colors
            for i, well in enumerate(selected_wells):
                well_data = groundwater_data[well].to_numpy().copy()
                mask = ~np.isnan(well_data)
                fig.add_trace(
                    go.Scatter(
                        x=times[mask],
                        y=well_data[mask],
                        name=well,
                        mode='lines',
                        line=dict(
                            width=2,
                            color=WELL_COLORS[i % len(WELL_COLORS)],
                            shape='spline'  # Smooth lines
                        ),
                        hovertemplate=(
                            f"<b>Well {well}</b><br>" +
                            "Date: %{x}<br>" +
                            "Level: %{y:.2f}m<extra></extra>"
                        )
                    ),
                    row=2, col=1
                )

            # Set up rainfall axes
            fig.update_yaxes(
                title=dict(
                    text="<b>Rainfall (mm)</b>",
                    font=dict(size=14)
                ),
                tickfont=dict(size=12, weight='bold'),
                row=1, col=1,
                range=[max_rainfall * 1.1, 0]  # Inverted
            )
            fig.update_xaxes(
                title="",  # No title for top plot
                tickfont=dict(size=12, weight='bold'),
                row=1, col=1,
                range=[times[0], times[-1]]  # Use shared date range
            )
            
            # Set up groundwater axes
            fig.update_yaxes(
                title=dict(
                    text="<b>Groundwater Level (m)</b>",
                    font=dict(size=14)
                ),
                tickfont=dict(size=12, weight='bold'),
                row=2, col=1
            )
            fig.update_xaxes(
                title=dict(
                    text="<b>Date</b>",
                    font=dict(size=14)
                ),
                tickfont=dict(size=12, weight='bold'),
                row=2, col=1,
                range=[times[0], times[-1]]  # Use shared date range
            )

            # Update layout with better styling
            layout_update = {
                **STATIC_LAYOUT,  # Base static configuration
                'width': 1200,    # Match control panel width
                'height': 1000,
                'legend': dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="center",
                    x=0.5,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(0, 0, 0, 0.2)',
                    borderwidth=1
                ),
                'xaxis': AXIS_CONFIG,
                'xaxis2': AXIS_CONFIG,
                'yaxis': AXIS_CONFIG,
                'yaxis2': AXIS_CONFIG
            }
            fig.update_layout(layout_update)
            
            # Free memory
            del groundwater_data, times, rainfall_array
            gc.collect()
            
            gc.collect()  # Force garbage collection
            
            return fig
            
        except Exception as e:
            print(f"Error in overlay plot: {e}")
            import traceback
            traceback.print_exc()
            return {}

