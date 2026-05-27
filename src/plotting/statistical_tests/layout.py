"""
Layout creation for Statistical Analysis Dashboard
"""

from dash import html, dcc
from .styles import styles
from .components import create_statistical_guide, create_dropdowns


def create_layout():
    """Create the application layout with improved statistical guide."""
    return html.Div([
        # Title
        html.H1('Groundwater Statistical Analysis',
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'marginBottom': '30px',
                    'fontSize': '32px',
                    'fontWeight': 'bold'
                }),
        
        # Statistical Guide Section
        create_statistical_guide(),
        
        # Main container
        html.Div([
            # Slider section
            html.Div([
                html.Label('Number of Wells to Compare:', 
                          style={
                              'fontSize': '18px',
                              'color': '#2c3e50',
                              'marginBottom': '15px',
                              'display': 'block',
                              'textAlign': 'center',
                              'fontWeight': 'bold'
                          }),
                dcc.Slider(
                    id='num-selections-slider',
                    min=1,
                    max=12,
                    value=1,
                    marks={i: {'label': str(i), 
                              'style': {'fontSize': '16px', 
                                       'fontWeight': 'bold',
                                       'color': '#2c3e50'}} 
                           for i in range(1, 13)},
                    step=1,
                    included=True,
                    tooltip={'placement': 'bottom', 
                            'always_visible': True}
                )
            ], style={
                'width': '80%',
                'margin': '30px auto',
                'padding': '20px',
                'backgroundColor': '#ffffff',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'
            }),
            
            # Dropdown container
            html.Div(id='dropdown-container', style={'margin': '20px'}),
            
            # Update button
            html.Div([
                html.Button('Generate Statistical Comparison', 
                           id='update-button', 
                           style=styles['button'])
            ], style={
                'display': 'flex',
                'justifyContent': 'center',
                'marginTop': '20px',
                'marginBottom': '20px'
            }),
            
            # Plot container
            html.Div(id='plot-container')
        ], style={'backgroundColor': '#ffffff', 'padding': '20px'})
    ], style=styles['container'])

