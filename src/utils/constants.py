"""
Constants Module
--------------
Contains all constant values used throughout the application.
"""

# Display settings
PLOT_HEIGHT = 800
PLOT_WIDTH = 1400

# Colour palette for wells - Use tuple for immutability and faster lookups
WELL_COLOURS = (
    'blue', 'darkorange', 'forestgreen', 'crimson', 'rebeccapurple',
    'saddlebrown', 'hotpink', 'grey', 'olive', 'darkcyan',
    'darkslateblue', 'darkolivegreen', 'peru', 'maroon', 'indigo'
)

# Colours for subplots
SUBPLOT_COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Colours for different intervals
INTERVAL_COLOURS = {
    '15min': '#1f77b4',    # Blue
    'daily': '#2ca02c',    # Green
    'monthly': '#ff7f0e',  # Orange
    '6monthly': '#d62728'  # Red
}

# Time intervals for resampling
TIME_INTERVALS = {
    '15min': '15T',
    'daily': 'D',
    'monthly': 'M',
    '6monthly': '6M'
}

# Default values
DEFAULT_PORT = 8050
PORT_RANGE = (8050, 8100)

# File paths and extensions
VALID_EXTENSIONS = ('.csv', '.xlsx', '.xls')
DEFAULT_ENCODING = 'utf-8'

# Plot settings
RANGESLIDER_HEIGHT = 0.05
OPACITY = 0.7
LINE_WIDTH = 1.5
GRID_OPACITY = 0.2

# Dashboard settings
DASHBOARD_HEIGHTS = {
    'single': 600,
    'double': 800,
    'full': 1000
}

# Date formats
DATE_FORMATS = {
    'display': '%d/%m/%Y %H:%M',
    'storage': '%Y-%m-%d %H:%M:%S',
    'filename': '%Y%m%d_%H%M%S'
}

# Error messages
ERROR_MESSAGES = {
    'data_load': 'Error whilst loading data: {}',
    'processing': 'Error whilst processing data: {}',
    'plotting': 'Error whilst creating plot: {}',
    'saving': 'Error whilst saving file: {}',
    'conversion': 'Error whilst converting data: {}',
    'validation': 'Error whilst validating input: {}'
}

# Success messages
SUCCESS_MESSAGES = {
    'data_load': 'Data loaded successfully',
    'processing': 'Data processed successfully',
    'plotting': 'Plot created successfully',
    'saving': 'File saved successfully: {}',
    'server_start': 'Server started on port {}'
}