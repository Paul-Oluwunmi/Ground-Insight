"""
POT Dashboard Runner
====================

Function to launch the POT analysis dashboard.
"""

from dash import Dash
import logging

from .create_dashboard import create_pot_dashboard


def run_pot_dashboard(data, location_columns):
    """
    Launch the POT analysis dashboard.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data containing DateTime and location columns
    location_columns : list
        List of location column names
    """
    try:
        # Create dashboard using modular structure
        app = create_pot_dashboard(data, location_columns)
        
        # Run the server with error handling for already running apps
        print("Starting POT Analysis Dashboard...")
        
        try:
            app.run(debug=False, port=8050, host='127.0.0.1')
        except OSError as e:
            if "Address already in use" in str(e):
                print("Port 8050 is already in use. Trying port 8051...")
                try:
                    app.run(debug=False, port=8051, host='127.0.0.1')
                except OSError:
                    print("Port 8051 is also in use. Trying port 8052...")
                    app.run(debug=False, port=8052, host='127.0.0.1')
            else:
                raise e
        
    except Exception as e:
        print(f"Error launching POT dashboard: {str(e)}")
        logging.error(f"POT dashboard error: {str(e)}")
        raise
