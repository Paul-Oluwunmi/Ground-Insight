"""
POT Dashboard Creation
=======================

Creates the POT analysis dashboard with layout and callbacks.
This file imports helper functions from the modular structure.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import genpareto
from scipy import stats
from scipy.signal import find_peaks as scipy_find_peaks
from scipy.stats import linregress
import traceback
import warnings
import logging
import polars as pl
from scipy.optimize import minimize
from dash import Dash       
import emcee
import corner
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import dash_table
from typing import Optional, Dict, Any
import threading
import time

# Import from modular structure
from ..constants import COLORS, SEASON_COLORS, global_store_pot
from ..data_processing import find_peaks_data, calculate_exceedances
from ..statistical_calculations import (
    calculate_return_periods,
    calculate_confidence_intervals,
    fit_gpd_by_season,
    calculate_return_levels_by_season,
    fit_gpd_with_trend,
    calculate_return_levels_trend,
    fit_gpd_by_covariate,
    calculate_return_levels_covariate
)
from ..bayesian_analysis import (
    perform_bayesian_gpd_stationary,
    perform_bayesian_gpd_non_stationary
)
from ..plotting import (
    update_eva_plots,
    create_corner_plot_gpd,
    generate_posterior_predictive_checks_gpd
)

# Configure logging and warnings
logging.getLogger('werkzeug').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


def create_pot_dashboard(data, location_columns):
    """
    Create an interactive POT analysis dashboard.
    
    This function uses the modular layout and callbacks extracted from the backup file.
    """
    import dash
    from .layout import create_layout
    from .callbacks import register_callbacks
    
    app = dash.Dash(__name__, 
                    suppress_callback_exceptions=True,
                    update_title=None)

    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                html, body, #react-entry-point {
                    height: 100%;
                    width: 100%;
                    margin: 0;
                    padding: 0;
                    background-color: white;
                    overflow-y: auto;
                }
                body {
                    min-height: 100vh;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Disable debug prints from Dash
    app.logger.handlers = []
    
    # Create layout
    app.layout = create_layout(location_columns)
    
    # Register callbacks
    register_callbacks(app, data, location_columns)
    
    return app

