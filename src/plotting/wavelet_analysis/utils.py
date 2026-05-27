"""
Wavelet Analysis Utility Functions
==================================

Helper functions for wavelet analysis dashboard.
"""

import pywt
import pandas as pd
import datetime
import dash_table
import numpy as np
from functools import lru_cache

from .constants import phase_analysis_settings


def get_phase_settings():
    """Safely get phase analysis settings."""
    global phase_analysis_settings
    if 'phase_analysis_settings' not in globals() or phase_analysis_settings is None:
        phase_analysis_settings = {'show_plots': [], 'target_period': 12}
    return phase_analysis_settings


def get_flat_wavelet_options(kind):
    """Get flat list of wavelet options for dropdown."""
    options = []
    families = pywt.families()
    for fam in families:
        wavelets = pywt.wavelist(fam)
        for w in wavelets:
            try:
                if kind == 'cwt':
                    pywt.ContinuousWavelet(w)
                else:
                    pywt.Wavelet(w)
                options.append({'label': f"{fam}: {w}", 'value': w})
            except Exception:
                continue
    return options


def summary_to_dash_table(summary_df, yaxis_label):
    """
    Convert a summary statistics DataFrame or Series to a styled Dash DataTable with integrated title.
    yaxis_label: str, the label for the value column (e.g., 'Groundwater Level (m)' or 'Rainfall (mm)')
    """
    if isinstance(summary_df, pd.Series):
        summary_df = summary_df.to_frame()
    summary_df = summary_df.reset_index()
    summary_df.columns = ['Statistic', yaxis_label]
    # Round the values to 4 decimal places
    summary_df[yaxis_label] = summary_df[yaxis_label].round(4)
    
    # Add title as first row
    title_row = {'Statistic': 'Summary Statistics', yaxis_label: ''}
    data_with_title = [title_row] + summary_df.to_dict('records')
    
    return dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in summary_df.columns],
        data=data_with_title,
        style_table={'width': '60%', 'margin': 'auto', 'backgroundColor': 'white'},
        style_cell={
            'backgroundColor': 'white',
            'color': 'black',
            'textAlign': 'center',
            'fontFamily': 'Arial',
            'fontSize': 16,
        },
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 0},  # Title row
                'backgroundColor': '#e8f5e8',
                'fontWeight': 'bold',
                'fontSize': 18
            },
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f9f9f9'
            }
        ],
    )


def get_nz_season_vrects(x_dates):
    """
    Returns a list of plotly vrect dicts for NZ seasons covering the range of x_dates.
    The rectangles are clipped to the min and max of x_dates.
    """
    if len(x_dates) == 0:
        return []
    # Convert to pandas datetime if not already
    x_dates = pd.to_datetime(x_dates)
    min_x = x_dates.min()
    max_x = x_dates.max()
    min_year = min_x.year
    max_year = max_x.year
    vrects = []
    season_colors = {
        'Summer': 'rgba(255, 230, 128, 0.15)',   # light yellow
        'Autumn': 'rgba(255, 153, 51, 0.10)',    # light orange
        'Winter': 'rgba(128, 179, 255, 0.12)',   # light blue
        'Spring': 'rgba(153, 255, 153, 0.12)'    # light green
    }
    for year in range(min_year, max_year + 2):
        # Summer (Dec 1 prev year - Feb 28/29)
        x0 = datetime.datetime(year-1, 12, 1)
        x1 = datetime.datetime(year, 3, 1)
        if x1 < min_x or x0 > max_x:
            pass
        else:
            vrects.append(dict(
                type="rect",
                x0=max(x0, min_x),
                x1=min(x1, max_x),
                yref="paper", y0=0, y1=1,
                fillcolor=season_colors['Summer'],
                line_width=0,
                layer="below"
            ))
        # Autumn (Mar 1 - May 31)
        x0 = datetime.datetime(year, 3, 1)
        x1 = datetime.datetime(year, 6, 1)
        if x1 < min_x or x0 > max_x:
            pass
        else:
            vrects.append(dict(
                type="rect",
                x0=max(x0, min_x),
                x1=min(x1, max_x),
                yref="paper", y0=0, y1=1,
                fillcolor=season_colors['Autumn'],
                line_width=0,
                layer="below"
            ))
        # Winter (Jun 1 - Aug 31)
        x0 = datetime.datetime(year, 6, 1)
        x1 = datetime.datetime(year, 9, 1)
        if x1 < min_x or x0 > max_x:
            pass
        else:
            vrects.append(dict(
                type="rect",
                x0=max(x0, min_x),
                x1=min(x1, max_x),
                yref="paper", y0=0, y1=1,
                fillcolor=season_colors['Winter'],
                line_width=0,
                layer="below"
            ))
        # Spring (Sep 1 - Nov 30)
        x0 = datetime.datetime(year, 9, 1)
        x1 = datetime.datetime(year, 12, 1)
        if x1 < min_x or x0 > max_x:
            pass
        else:
            vrects.append(dict(
                type="rect",
                x0=max(x0, min_x),
                x1=min(x1, max_x),
                yref="paper", y0=0, y1=1,
                fillcolor=season_colors['Spring'],
                line_width=0,
                layer="below"
            ))
    return vrects


def get_nz_season_for_dates(dates):
    """
    Given a pandas Series or array of datetime, return a list of NZ season names for each date.
    """
    seasons = []
    for dt in pd.to_datetime(dates):
        month = dt.month
        if (month == 12) or (month <= 2):
            seasons.append('Summer')
        elif 3 <= month <= 5:
            seasons.append('Autumn')
        elif 6 <= month <= 8:
            seasons.append('Winter')
        else:
            seasons.append('Spring')
    return seasons

