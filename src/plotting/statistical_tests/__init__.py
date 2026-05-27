"""
Statistical Analysis Module for Groundwater Data

This module provides interactive statistical analysis tools for comparing
groundwater levels across different wells and time periods.
"""

import dash
import polars as pl
from typing import List
from .layout import create_layout
from .callbacks import register_callbacks


def run_statistical_analysis(data, well_columns: List[str]) -> None:
    """
    Run the statistical analysis application with the provided data.
    
    Parameters:
    -----------
    data : pandas.DataFrame or polars.DataFrame
        Groundwater data with DateTime column and well columns
    well_columns : List[str]
        List of well column names to analyze
    """
    try:
        # Convert to polars if it's not already
        if not isinstance(data, pl.DataFrame):
            data = pl.from_pandas(data)
        
        # Extract year and month using polars expressions
        data = data.with_columns([
            pl.col('DateTime').dt.year().alias('Year'),
            pl.col('DateTime').dt.strftime('%b').alias('Month')
        ])
        
        # Create and configure the app
        app = dash.Dash(__name__, 
                       suppress_callback_exceptions=True,
                       update_title=None)  
        
        # Configure server
        server = app.server
        server.config['PROPAGATE_EXCEPTIONS'] = False  
        
        app.layout = create_layout()
        register_callbacks(app, data, well_columns)
        
        app.run(debug=False, port=8050)  
        
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        raise

