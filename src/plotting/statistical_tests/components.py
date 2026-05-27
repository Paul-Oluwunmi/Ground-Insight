"""
UI Components for Statistical Analysis Dashboard
"""

from dash import html
from .styles import styles


def create_statistical_guide():
    """Create the statistical comparison guide table."""
    return html.Div([
        html.H3("Statistical Comparison Guide", 
                style={'marginBottom': '20px', 'color': '#2c3e50'}),
        
        html.Table([
            # Header
            html.Thead(
                html.Tr([
                    html.Th("Measure", style={'width': '20%', 'textAlign': 'left', 'padding': '12px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6'}),
                    html.Th("Description", style={'width': '40%', 'textAlign': 'left', 'padding': '12px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6'}),
                    html.Th("Interpretation", style={'width': '40%', 'textAlign': 'left', 'padding': '12px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6'})
                ])
            ),
            # Body
            html.Tbody([
                # Connection Colours
                html.Tr([
                    html.Td("Connection Colours", style=styles['table_cell']),
                    html.Td([
                        html.Div([
                            html.Span("● ", style={'color': 'green', 'fontSize': '20px'}),
                            "Green: Similar water levels"
                        ]),
                        html.Div([
                            html.Span("● ", style={'color': 'red', 'fontSize': '20px'}),
                            "Red: Different water levels"
                        ])
                    ], style=styles['table_cell']),
                    html.Td([
                        "Green: Wells show similar patterns",
                        html.Br(),
                        "Red: Wells show different patterns"
                    ], style=styles['table_cell'])
                ]),
                # p-value
                html.Tr([
                    html.Td("p-value", style=styles['table_cell']),
                    html.Td("Measures how likely the difference is by chance", style=styles['table_cell']),
                    html.Td([
                        html.Span("p ≤ 0.05:", style={'fontWeight': 'bold'}),
                        " Strong evidence of real difference",
                        html.Br(),
                        html.Span("p > 0.05:", style={'fontWeight': 'bold'}),
                        " Difference could be by chance"
                    ], style=styles['table_cell'])
                ], style={'backgroundColor': '#f8f9fa'}),
                # t-value
                html.Tr([
                    html.Td("t-value", style=styles['table_cell']),
                    html.Td("Shows how large the difference is", style=styles['table_cell']),
                    html.Td([
                        html.Span("|t| > 2:", style={'fontWeight': 'bold'}),
                        " Large difference",
                        html.Br(),
                        html.Span("1 < |t| < 2:", style={'fontWeight': 'bold'}),
                        " Medium difference",
                        html.Br(),
                        html.Span("|t| < 1:", style={'fontWeight': 'bold'}),
                        " Small difference"
                    ], style=styles['table_cell'])
                ]),
                # Effect Size
                html.Tr([
                    html.Td("Effect Size (d)", style=styles['table_cell']),
                    html.Td("Practical importance of the difference", style=styles['table_cell']),
                    html.Td([
                        html.Span("|d| < 0.2:", style={'fontWeight': 'bold'}),
                        " Minor practical difference",
                        html.Br(),
                        html.Span("0.2 ≤ |d| < 0.8:", style={'fontWeight': 'bold'}),
                        " Moderate practical difference",
                        html.Br(),
                        html.Span("|d| ≥ 0.8:", style={'fontWeight': 'bold'}),
                        " Major practical difference"
                    ], style=styles['table_cell'])
                ], style={'backgroundColor': '#f8f9fa'})
            ])
        ], style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'border': '1px solid #dee2e6',
            'marginTop': '10px'
        })
    ], style=styles['guide_container'])


def create_dropdowns(num_selections, well_options):
    """Create dropdown components for well, year, and month selection."""
    from dash import dcc
    
    dropdown_groups = []
    for i in range(num_selections):
        group = html.Div([
            html.Div([
                html.Div([
                    html.Label(f'Well {i+1}:', style=styles['dropdown_label']),
                    dcc.Dropdown(
                        id={'type': 'well-dropdown', 'index': i},
                        options=well_options,
                        placeholder='Select a well...',
                        style={'width': '100%'}
                    )
                ], style=styles['dropdown_container']),
                
                html.Div([
                    html.Label(f'Year {i+1}:', style=styles['dropdown_label']),
                    dcc.Dropdown(
                        id={'type': 'year-dropdown', 'index': i},
                        options=[],  # Will be populated dynamically
                        placeholder='Select a year...',
                        style={'width': '100%'}
                    )
                ], style=styles['dropdown_container']),
                
                html.Div([
                    html.Label(f'Month {i+1}:', style=styles['dropdown_label']),
                    dcc.Dropdown(
                        id={'type': 'month-dropdown', 'index': i},
                        options=[
                            {'label': month, 'value': month} 
                            for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        ],
                        placeholder='Select a month...',
                        style={'width': '100%'}
                    )
                ], style=styles['dropdown_container'])
            ], style=styles['selection_group'])
        ], style={'marginBottom': '20px'})
        dropdown_groups.append(group)
    return dropdown_groups

