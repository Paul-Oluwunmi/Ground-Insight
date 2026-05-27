"""
Wavelet Analysis Dashboard Runner
=================================

Launches the wavelet analysis dashboard.
"""

import dash
import logging

from .create_dashboard import create_wavelet_dashboard


def run_wavelet_analysis(data, well_columns, rainfall_col=None, pumping_col=None, 
                         tide_col=None, remove_rainfall=False, remove_pumping=False, 
                         remove_tide=False, confounder_lags=7, date_range=None):
    """
    Launches a Dash dashboard for basic and advanced (wavelet) analysis
    of groundwater level and rainfall data.
    Optionally removes the effects of rainfall, pumping, and/or tidal data from GWL before analysis.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The main dataset
    well_columns : list
        List of well column names
    rainfall_col : str or None
        Column name for rainfall data
    pumping_col : str or None
        Column name for pumping data
    tide_col : str or None
        Column name for tidal data
    remove_rainfall : bool
        Whether to remove rainfall effects
    remove_pumping : bool
        Whether to remove pumping effects
    remove_tide : bool
        Whether to remove tidal effects
    confounder_lags : int
        Number of lags to use in regression
    date_range : tuple or None
        (start_date, end_date) or None
    """
    try:
        # Create dashboard
        app = create_wavelet_dashboard(
            data, well_columns, rainfall_col, pumping_col, tide_col,
            remove_rainfall, remove_pumping, remove_tide, confounder_lags, date_range
        )
        
        # Run the server
        print("Starting Wavelet Analysis Dashboard...")
        
        try:
            app.run(debug=True, port=8050, host='127.0.0.1')
        except OSError as e:
            if "Address already in use" in str(e):
                print("Port 8050 is already in use. Trying port 8051...")
                try:
                    app.run(debug=True, port=8051, host='127.0.0.1')
                except OSError:
                    print("Port 8051 is also in use. Trying port 8052...")
                    app.run(debug=True, port=8052, host='127.0.0.1')
            else:
                raise e
        
    except Exception as e:
        print(f"Error launching Wavelet dashboard: {str(e)}")
        logging.error(f"Wavelet dashboard error: {str(e)}")
        raise

