"""
Overlay Module
==============

Interactive overlay visualization for groundwater and rainfall data.

Main Functions:
--------------
- create_interactive_overlay: Create Dash application with overlay visualization
- run_overlay: Main entry point for running the overlay application
"""

from dash import Dash
from .layout import create_layout
from .callbacks import register_callbacks
from .data_processing import prepare_data_for_overlay


def create_interactive_overlay(data, well_columns, data_folder=None, rainfall_filename=None):
    """
    Create an interactive overlay plot with optimized performance.
    
    Args:
        data: DataFrame containing groundwater data
        well_columns: List of well column names
        data_folder: Path to data folder for loading rainfall data directly
        rainfall_filename: Name of the rainfall data file
        
    Returns:
        dash.Dash: Configured Dash application instance
    """
    try:
        app = Dash(__name__)
        
        # Prepare data for overlay
        data_lazy = prepare_data_for_overlay(data)
        
        # Create layout
        app.layout = create_layout(well_columns)
        
        # Register callbacks
        register_callbacks(app, data_lazy, well_columns, data_folder, rainfall_filename)
        
        return app
        
    except Exception as e:
        print(f"Error creating overlay app: {e}")
        return None


def run_overlay(data, well_columns, data_folder=None, rainfall_filename=None):
    """
    Run the overlay app with minimal configuration.
    
    Parameters:
    -----------
    data : DataFrame
        Groundwater data
    well_columns : list
        List of well column names
    data_folder : str, optional
        Path to data folder
    rainfall_filename : str, optional
        Name of rainfall file
    """
    try:
        # Use global variables if not provided (for notebook compatibility)
        if data_folder is None:
            import inspect
            frame = inspect.currentframe().f_back
            if frame and 'data_folder' in frame.f_globals:
                data_folder = frame.f_globals['data_folder']
        
        if rainfall_filename is None:
            import inspect
            frame = inspect.currentframe().f_back
            if frame and 'rainfall_filename' in frame.f_globals:
                rainfall_filename = frame.f_globals['rainfall_filename']
        
        app = create_interactive_overlay(data, well_columns, data_folder, rainfall_filename)
        if app is None:
            raise ValueError("Failed to create overlay app")
            
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
                if port == 8070:
                        raise ValueError("No available ports found between 8050 and 8070")
                continue
            
    except Exception as e:
        raise


__all__ = ['create_interactive_overlay', 'run_overlay']

