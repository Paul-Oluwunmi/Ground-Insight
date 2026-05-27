"""
Overlay Layout
==============

Dash layout components for the overlay visualization.
"""

from dash import dcc, html


def create_layout(well_columns):
    """
    Create the Dash layout for the overlay application.
    
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
        html.H1("Interactive Well Overlay", 
               style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#2c3e50'}),
        
        html.Div([
            html.Div([
                html.Label("Select Wells:", 
                         style={'marginBottom': '5px', 'color': '#2c3e50'}),
                dcc.Dropdown(
                    id='well-selector',
                    options=[{'label': well, 'value': well} for well in well_columns],
                    value=[well_columns[0]],
                    multi=True,
                    style={'width': '500px'}
                )
            ], style={'marginRight': '40px'}),
            
            html.Div([
                html.Label("Select Interval:", 
                         style={'marginRight': '15px', 'color': '#2c3e50'}),
                dcc.RadioItems(
                    id='interval-selector',
                    options=[
                        {'label': 'Original ', 'value': 'original'},
                        {'label': 'Daily ', 'value': 'D'},
                        {'label': 'Weekly ', 'value': 'W'},
                        {'label': 'Monthly ', 'value': 'ME'},
                        {'label': '6-Monthly ', 'value': '6ME'},
                        {'label': 'Yearly ', 'value': 'Y'}
                    ],
                    value='D',  # Set daily as default
                    inline=True,
                    style={'display': 'inline-block', 'color': '#2c3e50'},
                    labelStyle={'marginRight': '20px'}
                )
            ])
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'marginBottom': '20px',
            'padding': '15px',
            'backgroundColor': 'white',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
        
        dcc.Loading(
            id="loading",
            type="default",
            children=[
                dcc.Graph(
                    id='overlay-plot',
                    style={'height': '1000px'},
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'scrollZoom': True,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                    }
                )
            ]
        )
    ], style={
        'padding': '20px',
        'backgroundColor': '#f5f6fa',
        'minHeight': '100vh',
        'maxWidth': '1200px',  # Match plot width
        'margin': '0 auto'  # Center in page
    })

