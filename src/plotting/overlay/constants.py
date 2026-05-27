"""
Overlay Module Constants
========================

Static styling and layout configurations for the overlay visualization.
"""

# Static styling configurations
PLOT_COLORS = {
    'grid': 'rgba(0, 0, 0, 0.1)',
    'background': 'white',
    'marker_line': 'white'
}

# Static layout configuration
STATIC_LAYOUT = {
    'plot_bgcolor': PLOT_COLORS['background'],
    'paper_bgcolor': PLOT_COLORS['background'],
    'showlegend': True,
    'margin': dict(l=80, r=80, t=100, b=50)
}

# Static axis configuration
AXIS_CONFIG = {
    'showgrid': True,
    'gridwidth': 1,
    'gridcolor': PLOT_COLORS['grid']
}

# Well colors for plotting
WELL_COLORS = [
    '#E41A1C',  # Red
    '#377EB8',  # Blue
    '#4DAF4A',  # Green
    '#984EA3',  # Purple
    '#FF7F00',  # Orange
    '#FFFF33',  # Yellow
    '#A65628',  # Brown
    '#F781BF'   # Pink
]

# Bar width mapping for different intervals (in milliseconds)
BAR_WIDTH_MAP = {
    'original': 24*60*60*1000,     # 1 day for original data
    'D': 24*60*60*1000,           # 1 day
    'W': 7*24*60*60*1000,         # 7 days
    'ME': 30*24*60*60*1000,       # ~30 days
    '3ME': 90*24*60*60*1000,      # ~90 days
    '6ME': 180*24*60*60*1000,     # ~180 days
    'Y': 365*24*60*60*1000        # ~365 days
}

