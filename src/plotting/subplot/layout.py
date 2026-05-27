"""
Subplot Layout
==============

Dash layout components for the subplot visualization.
"""

from dash import dcc, html


def create_layout(well_columns):
    """
    Create the Dash layout for the subplot application.
    
    Parameters:
    -----------
    well_columns : list
        List of well column names
        
    Returns:
    --------
    html.Div
        Complete Dash layout structure
    """
    return html.Div([
        # Title and interval selector
        html.H1("Interactive Groundwater Well Plots", 
                style={
                    'textAlign': 'center',
                    'color': '#2C3E50',
                    'marginBottom': '20px',
                    'fontFamily': 'Arial, sans-serif',
                    'fontWeight': 'bold',
                    'fontSize': '28px'
                }),           
        # Time interval selector
        html.Div([
            dcc.RadioItems(
                id='interval-selector',
                options=[
                    {'label': 'Original Data', 'value': 'original'},
                    {'label': 'Hourly', 'value': 'hourly'},  
                    {'label': 'Daily', 'value': 'daily'},
                    {'label': 'Monthly', 'value': 'monthly'},
                    {'label': '6-Monthly', 'value': '6-monthly'},
                    {'label': 'Yearly', 'value': 'yearly'}
                ],
                value='original',
                inline=True,
                style={
                    'marginBottom': '20px',
                    'color': '#2C3E50',
                    'fontWeight': 'bold'
                }
            )
        ], style={'textAlign': 'center'}),

        # Data store for efficient updates
        dcc.Store(id='resampled-data'),
        
        # Four plots in vertical layout
        html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id=f'well-selector-{i}',
                            options=[{'label': well, 'value': well} for well in well_columns],
                            placeholder='**Select Well**',
                            style={
                                'marginBottom': '10px',
                                'width': '200px',
                                'color': '#2C3E50',
                                'fontWeight': 'bold',
                                'backgroundColor': 'white'
                            }
                        ),
                        dcc.Loading(
                            id=f'loading-{i}',
                            type="default",
                            children=[dcc.Graph(
                                id=f'plot-{i}',
                                style={'height': '400px', 'width': '100%'}, 
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'scrollZoom': True,
                                    'responsive': True,  
                                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 
                                                        'drawclosedpath', 'drawcircle', 
                                                        'drawrect', 'eraseshape']
                                }
                            )]
                        )
                    ], style={
                        'marginBottom': '20px', 
                        'width': '100%',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'stretch'  
                    })
                    for i in range(4)
                ], style={
                    'display': 'flex', 
                    'flexDirection': 'column',
                    'width': '100%', 
                    'alignItems': 'stretch'  
                })
            ], style={
                'padding': '20px',
                'backgroundColor': '#F8F9FA',
                'minHeight': '100vh',
                'width': '100%',  
                'boxSizing': 'border-box' 
            })

