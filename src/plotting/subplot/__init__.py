"""
Subplot Module
==============

Interactive subplot visualization for groundwater data.

Main Functions:
--------------
- create_subplots: Create Dash application with multiple well subplots
- run_subplot: Main entry point for running the subplot application
"""

from dash import Dash
from .layout import create_layout
from .callbacks import register_callbacks


def create_subplots(data, well_columns):
    """
    Create an interactive Dash application with multiple well subplots.
    
    Parameters:
    -----------
    data : DataFrame or pl.DataFrame
        Input data containing DateTime and well measurements
    well_columns : list
        List of column names corresponding to well measurements
    
    Returns:
    --------
    dash.Dash
        Configured Dash application instance
    
    Features:
    ---------
    - Four independent plot panels
    - Time interval selection (Original, Hourly, Daily, Monthly, 6-Monthly, Yearly)
    - Individual well selection per panel
    - Interactive zoom and pan capabilities
    - Unified hover tooltips with seasonal information
    - Performance optimizations with caching
    """
    try:
        import polars as pl
        
        # Ensure data is a pandas DataFrame for consistency
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        
        app = Dash(__name__)
        
        # Create layout with enhanced styling
        app.layout = create_layout(well_columns)
        
        # Register callbacks
        register_callbacks(app, data, well_columns)
        
        return app
        
    except Exception as e:
        print(f"Error creating Dash app: {e}")
        return None


def run_subplot(data, well_columns, save_path=None):
    """
    Run the Dash application with automatic port selection.
    
    Parameters:
    -----------
    data : DataFrame
        Input data containing DateTime and well measurements
    well_columns : list
        List of well column names
    save_path : str, optional
        Path to save any generated outputs
    
    Notes:
    ------
    - Attempts to start server on ports 8050-8070
    - Includes error handling and port availability checking
    - Provides feedback on server status
    - Includes performance optimizations and caching
    """
    try:
        # Create the app
        app = create_subplots(data, well_columns)
        if app is None:
            raise ValueError("Failed to create Dash app")
            
        # Try ports from 8050 to 8070
        for port in range(8050, 8071):
            try:
                app.run(
                    host='localhost',
                    port=port,
                    debug=False,
                    use_reloader=False
                )
                break
            except OSError:
                print(f"Port {port} is in use, trying next port...")
                if port == 8070:
                    print("No available ports found between 8050 and 8070")
                    raise
                continue
            
    except Exception as e:
        print(f"Error: {e}")
        raise


__all__ = ['create_subplots', 'run_subplot']

