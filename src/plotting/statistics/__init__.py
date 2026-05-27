"""
Statistics Module for Groundwater Data

This module provides statistical analysis and visualization tools for
groundwater level data, including monthly statistics plots with customizable
percentile ranges.
"""

import traceback
import polars as pl
import plotly.graph_objects as go
from typing import List
from .data_processing import prepare_data, prepare_well_data, prepare_all_years_data
from .statistical_calculations import calculate_boxplot_statistics
from .plotting import (
    create_year_trace, create_all_years_trace, create_buttons,
    configure_layout, configure_axes
)
from .constants import COLORS
from .interactive import interactive_run_statistics


def run_statistics(
    data,
    well_columns: List[str],
    q1_percentile: float = 25,
    q3_percentile: float = 75
):
    """
    Create monthly statistics plot with customizable percentile range for boxplots.

    Parameters:
    -----------
    data : polars.DataFrame or pandas.DataFrame
        Groundwater data with DateTime column
    well_columns : List[str]
        List of well column names to plot
    q1_percentile : float
        Lower percentile for boxplot (default 25)
    q3_percentile : float
        Upper percentile for boxplot (default 75)
        
    Returns:
    --------
    go.Figure or None : Plotly figure object or None if error
    """
    try:
        # Validate percentiles
        if not (0 <= q1_percentile < q3_percentile <= 100):
            raise ValueError("Percentiles must satisfy 0 <= q1 < q3 <= 100.")
        
        # Prepare data
        data = prepare_data(data)
        
        # Create figure
        fig = go.Figure()
        
        # Initialize tracking dictionaries
        well_traces = {}
        
        for well in well_columns:
            well_traces[well] = []
            
            # Prepare well data
            well_data = prepare_well_data(data, well)
            years = sorted(well_data.get_column("Year").unique().to_list())
            
            # Pre-calculate statistics for all years at once
            stats_by_year = {}
            for year in years:
                year_data = well_data.filter(pl.col("Year") == year)
                well_values = year_data[well].to_numpy()
                stats_by_year[year] = calculate_boxplot_statistics(
                    well_values, q1_percentile, q3_percentile
                )
            
            # Create traces for each year
            for i, year in enumerate(years):
                year_data = well_data.filter(pl.col("Year") == year)
                well_values = year_data[well].to_numpy()
                month_values = year_data["Month"].to_numpy()
                season_values = year_data["Season"].to_numpy()
                stats = stats_by_year[year]
                
                trace = create_year_trace(
                    well, year, well_values, month_values, season_values,
                    stats, COLORS[i % len(COLORS)], q1_percentile, q3_percentile
                )
                
                fig.add_trace(trace)
                well_traces[well].append(len(fig.data) - 1)
            
            # Create all_years_data and trace
            all_years_data = prepare_all_years_data(data, well)
            well_values = all_years_data[well].to_numpy()
            months = all_years_data["Month"].to_numpy()
            seasons = all_years_data["Season"].to_numpy()
            
            all_years_stats = calculate_boxplot_statistics(
                well_values, q1_percentile, q3_percentile
            )
            
            all_years_trace = create_all_years_trace(
                well, well_values, months, seasons,
                all_years_stats, q1_percentile, q3_percentile
            )
            
            fig.add_trace(all_years_trace)
            well_traces[well].append(len(fig.data) - 1)
        
        # Create buttons
        buttons = create_buttons(well_columns, well_traces, len(fig.data))
        
        # Update initial state
        for trace in fig.data:
            trace.visible = False
            trace.showlegend = True
        
        # Configure layout and axes
        configure_layout(fig, buttons)
        configure_axes(fig)
        
        fig.show()
        return fig
    
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None


# Export public API
__all__ = ['run_statistics', 'interactive_run_statistics', 'calculate_significance']

# Import calculate_significance for backward compatibility
from .statistical_calculations import calculate_significance

