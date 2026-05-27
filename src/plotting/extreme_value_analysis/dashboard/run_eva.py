"""
Main entry point for running the EVA analysis dashboard
"""

import logging
import traceback
import pandas as pd
from typing import List, Optional
from .layout import init_eva_dashboard
from ..constants import global_store


def run_eva_analysis(data: pd.DataFrame, well_columns: List[str], 
                     data_folder: Optional[str] = None, 
                     rainfall_filename: Optional[str] = None):
    """
    Run the EVA analysis dashboard.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Groundwater data DataFrame
    well_columns : List[str]
        List of well column names
    data_folder : Optional[str]
        Path to data folder (for loading rainfall data)
    rainfall_filename : Optional[str]
        Name of rainfall CSV file
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
        
        # Convert to absolute path for Dash callbacks (if data_folder is provided)
        if data_folder:
            import os
            data_folder = os.path.abspath(data_folder)
        
        # Store data_folder and rainfall_filename in global store for use in callbacks
        global_store['data_folder'] = data_folder
        global_store['rainfall_filename'] = rainfall_filename
        
        # Initialize the dashboard (callbacks are registered inside init_eva_dashboard)
        app = init_eva_dashboard(data, well_columns)
        
        # Run the server with error handling for already running apps
        try:
            app.run(debug=False, port=8050, host='127.0.0.1')
        except OSError as e:
            if "Address already in use" in str(e):
                logging.warning("Port 8050 is already in use. Trying port 8051...")
                try:
                    app.run(debug=False, port=8051, host='127.0.0.1')
                except OSError:
                    logging.warning("Port 8051 is also in use. Trying port 8052...")
                    app.run(debug=False, port=8052, host='127.0.0.1')
            else:
                raise e
        
    except Exception as e:
        logging.error(f"Error in EVA analysis: {str(e)}")
        traceback.print_exc()
        return

