"""
Time series plotting functions for Extreme Value Analysis
"""

import logging
import traceback
import pandas as pd
import plotly.graph_objects as go
from ..constants import SEASON_COLORS, COLORS
from ..utils import format_resample_period


def create_time_series_plot(ts: pd.Series, block_maxima: pd.Series, well: str, 
                          data_type: str, resample: str) -> go.Figure:
    """Create time series plot with block boundaries and improved annotations"""
    try:
        # Create base figure
        fig = go.Figure()
        
        # Set y-axis label and legend label based on data type
        if data_type.lower() == 'gwl':
            y_label = "Groundwater Level (m)"
            legend_label = f"GWL ({format_resample_period(resample)})"
        elif data_type.lower() == 'rain':
            y_label = "Rainfall (mm)"
            legend_label = f"Rainfall ({format_resample_period(resample)})"
        else:
            y_label = "Value"
            legend_label = f"Value ({format_resample_period(resample)})"
            
        # Add main time series
        fig.add_trace(
            go.Scatter(
                x=ts.index,
                y=ts.values,
                mode='lines',
                name=f'{well} {legend_label}',
                line=dict(color=COLORS['primary'], width=1),
                hovertemplate=(
                    "<b>Date</b>: %{x}<br>"
                    f"<b>{y_label}</b>: %{{y:.2f}}<br>"
                    f"<b>Resample</b>: {format_resample_period(resample)}"
                    "<extra></extra>"
                )
            )
        )
        
        # Get seasonal categorization for block maxima
        seasons = pd.Series(index=block_maxima.index, dtype='object')
        for idx in block_maxima.index:
            month = idx.month
            if month in [12, 1, 2]:
                seasons[idx] = 'Summer'
            elif month in [3, 4, 5]:
                seasons[idx] = 'Autumn'
            elif month in [6, 7, 8]:
                seasons[idx] = 'Winter'
            else:  # [9, 10, 11]
                seasons[idx] = 'Spring'
        
        # Add block maxima points by season
        for season in seasons.unique():
            season_maxima = block_maxima[seasons == season]
            if not season_maxima.empty:
                fig.add_trace(
                    go.Scatter(
                        x=season_maxima.index,
                        y=season_maxima.values,
                        mode='markers',
                        name=f'{season} Maxima (n={len(season_maxima)})',
                        marker=dict(
                            color=SEASON_COLORS[season],
                            size=10,
                            symbol='circle',
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate=(
                            f"{season}<br>" +
                            "Date: %{x}<br>" +
                            f"Value: %{{y:.2f}}<br>" +
                            "<extra></extra>"
                        )
                    )
                )

        # Calculate date range 
        date_range = ts.index.max() - ts.index.min()
        date_range_years = date_range.total_seconds() / (365.25 * 24 * 3600)
        
        # Create dynamic title
        title_text = f"<b style='font-size:20px'>Time Series Analysis</b><br>{well} - Block Maxima (n={len(block_maxima)}) - ({date_range_years:.1f} years)"
        
        # Update layout 
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,  # Center the title
                xanchor='center',
                font=dict(size=14, weight='bold')
            ),
            height=550,
            width=1600,
            xaxis=dict(
                title=dict(
                    text="Date",
                    font=dict(size=12, weight='bold'),
                    standoff=15
                ),
                rangeslider=dict(
                    visible=True,
                    thickness=0.05
                ),
                type="date",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickfont=dict(size=10, color='black', weight='bold')
            ),
            yaxis=dict(
                title=dict(
                    text=y_label,
                    font=dict(size=12, weight='bold'),
                    standoff=15
                ),
                fixedrange=False,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                range=[ts.min() * 0.95, ts.max() * 1.05],
                tickfont=dict(size=10, weight='bold'),
                zeroline=True,
                zerolinecolor='rgba(128, 128, 128, 0.2)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=1.21,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor="Black",
                borderwidth=1,
                font=dict(size=12, weight='bold')
            ),
            margin=dict(
                l=80,  
                r=150,
                t=80,
                b=60  
            )
        )

        return fig
        
    except Exception as e:
        logging.error(f"Error in create_time_series_plot: {str(e)}")
        traceback.print_exc()  
        return go.Figure()

