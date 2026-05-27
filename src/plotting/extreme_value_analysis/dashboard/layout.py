"""
Dashboard layout for Extreme Value Analysis
"""

from dash import Dash, html, dcc
import dash_table
from typing import List
import pandas as pd
import logging
import traceback
from ..constants import COLORS, global_store
from ..utils import (
    get_suitable_block_periods,
    get_resample_options,
    get_default_block_period
)


def init_eva_dashboard(data, well_columns):
    """Initialize the EVA dashboard"""
    try:
        # Store data globally
        global_store['df_pl'] = data
        
        # Calculate data length for block periods
        date_range = data['DateTime'].max() - data['DateTime'].min()
        data_length_years = date_range.days / 365.25
        
        # Determine if rainfall data is present (more flexible detection)
        has_rainfall = any(
            'rain' in col.lower() or 
            'rainfall' in col.lower() or 
            'precipitation' in col.lower() 
            for col in well_columns
        )
        
        # Get initial data type and well
        initial_data_type = 'rain' if has_rainfall else 'gwl'
        initial_well = next(
            (col for col in well_columns if 'rain' in col.lower()),
            well_columns[0]
        ) if has_rainfall else well_columns[0]

        # Get suitable periods with default for initial data type
        suitable_periods, default_period = get_suitable_block_periods(data_length_years, initial_data_type)
        
        # Get initial resample options
        initial_resample_options = get_resample_options(default_period, initial_data_type)
        initial_resample = '10T' if initial_data_type == 'rain' else '15T'

        # Create Dash app
        app = Dash(__name__, suppress_callback_exceptions=True, update_title=None)
        server = app.server
        server.config['PROPAGATE_EXCEPTIONS'] = False

        # Create app layout
        app.layout = html.Div([
            # Header
            html.H1(
                'Extreme Value Analysis Dashboard',
                style={
                    'textAlign': 'center',
                    'color': COLORS['text'],
                    'padding': '20px'
                }
            ),
            
            # Guide Section
            html.Div([
                html.H2('Guide to Extreme Value Analysis: Block Maximum Method', 
                       style={'color': COLORS['text'], 'marginBottom': '15px'}),
                
                html.Div([
                    html.H3('Overview', style={'color': COLORS['primary']}),
                    html.P([
                        "This dashboard utilises the Generalised Extreme Value (GEV) distribution for analysing extreme events in hydrological data. ",
                        "It provides comprehensive analytical tools for both rainfall and groundwater level data, enabling the assessment and prediction of rare events."
                    ]),
                    
                    html.H3('Detailed Analysis Guide', style={'color': COLORS['primary']}),
                    
                    # Analysis Components Table
                    html.Table([
                        # Header
                        html.Thead(
                            html.Tr([
                                html.Th('Component', style={'width': '15%'}),
                                html.Th('Description', style={'width': '45%'}),
                                html.Th('Interpretation Guidelines', style={'width': '40%'})
                            ], style={
                            'backgroundColor': COLORS['primary'], 
                            'color': 'white',
                            'padding': '12px',
                            'textAlign': 'left',
                            'verticalAlign': 'top',
                            'borderBottom': f'2px solid {COLORS["secondary"]}'
                        })
                        ),
                        # Body
                        html.Tbody([
                            # Data Selection
                            html.Tr([
                                html.Td('Data Selection Controls'),
                                html.Td([
                                    html.Strong('Data Type: '), "Choose between rainfall and groundwater analysis",
                                    html.Br(),
                                    html.Strong('Well Selection: '), "Select specific monitoring locations",
                                    html.Br(),
                                    html.Strong('Block Period: '), "Define temporal blocks for extreme value extraction",
                                    html.Br(),
                                    html.Strong('Resample Period: '), "Adjust data resolution"
                                ]),
                                html.Td([
                                    "• Select appropriate block periods based on data length",
                                    html.Br(),
                                    "• Consider measurement frequency when resampling",
                                    html.Br(),
                                    "• Ensure sufficient data points within each block"
                                ])
                            ], style={'backgroundColor': COLORS['light']}),
                            
                            # Seasonal Analysis
                            html.Tr([
                                html.Td('1. Seasonal Analysis'),
                                html.Td([
                                    "Temporal distribution of extreme events across seasons, highlighting seasonal patterns and long-term trends. ",
                                    "Colour-coded by season: Summer (orange), Autumn (brown), Winter (blue), Spring (green)"
                                ]),
                                html.Td([
                                    "• Identify dominant seasons for extreme events",
                                    html.Br(),
                                    "• Assess temporal clustering",
                                    html.Br(),
                                    "• Evaluate long-term trends",
                                    html.Br(),
                                    "• Consider climate change implications"
                                ])
                            ]),

                            # Probability Density
                            html.Tr([
                                html.Td('2. Probability Density'),
                                html.Td([
                                    "Distribution of extreme values with fitted GEV curve. ",
                                    "Displays frequency distribution of extremes and theoretical fit.",
                                    html.Br(), html.Br(),
                                    html.Strong("Distribution Characteristics:"),
                                    html.Ul([
                                        html.Li("Shape Parameter (ξ) determines tail behaviour:"),
                                        html.Ul([
                                            html.Li("ξ > 0: Heavy/long tail (Fréchet type)"),
                                            html.Li("ξ = 0: Light tail (Gumbel type)"),
                                            html.Li("ξ < 0: Bounded/short tail (Weibull type)")
                                        ]),
                                        html.Li("Tail Position:"),
                                        html.Ul([
                                            html.Li("Right-skewed: Long tail extends to the right"),
                                            html.Li("Left-skewed: Long tail extends to the left"),
                                            html.Li("Symmetric: Similar to normal distribution")
                                        ]),
                                        html.Li("Common Patterns:"),
                                        html.Ul([
                                            html.Li([
                                                "Bell shape: Data clusters around the middle with equal spread on both sides",
                                                html.Ul([
                                                    html.Li("Implies: Regular and predictable extreme events"),
                                                    html.Li("Good for: Reliable forecasting and design values"),
                                                    html.Li("Risk: Moderate, with predictable extremes")
                                                ])
                                            ]),
                                            html.Li([
                                                "Positive skew: Long tail extends to the right",
                                                html.Ul([
                                                    html.Li("Implies: More frequent high extreme events"),
                                                    html.Li("Good for: Identifying flood risks"),
                                                    html.Li("Risk: Higher chance of extreme high values")
                                                ])
                                            ]),
                                            html.Li([
                                                "Negative skew: Long tail extends to the left",
                                                html.Ul([
                                                    html.Li("Implies: More frequent low extreme events"),
                                                    html.Li("Good for: Identifying drought risks"),
                                                    html.Li("Risk: Higher chance of extreme low values")
                                                ])
                                            ])
                                        ])
                                    ]),
                                    html.Br(),
                                    "Note: The shape of the distribution provides insights into the nature of extreme events in your dataset."
                                ]),
                                html.Td([
                                    html.Strong("Interpretation Guidelines:"),
                                    html.Ul([
                                        html.Li([
                                            html.Strong("Heavy Tails (ξ > 0):"),
                                            html.Br(),
                                            "• Higher probability of extreme events",
                                            html.Br(),
                                            "• More uncertainty in predictions",
                                            html.Br(),
                                            "• Common in intense rainfall events"
                                        ]),
                                        html.Li([
                                            html.Strong("Light Tails (ξ ≈ 0):"),
                                            html.Br(),
                                            "• Moderate extreme behaviour",
                                            html.Br(),
                                            "• More reliable predictions",
                                            html.Br(),
                                            "• Common in many natural processes"
                                        ]),
                                        html.Li([
                                            html.Strong("Short Tails (ξ < 0):"),
                                            html.Br(),
                                            "• Upper limit on possible values",
                                            html.Br(),
                                            "• More stable predictions",
                                            html.Br(),
                                            "• Common in bounded physical systems"
                                        ])
                                    ])
                                ])
                            ], style={'backgroundColor': COLORS['light']}),
                                                        
                            # Return Level Plot
                            html.Tr([
                                html.Td('3. Return Level Plot'),
                                html.Td([
                                    "Estimated magnitude of events for different return periods. ",
                                    "Includes 95% confidence intervals and frequency zones.",
                                    html.Br(), html.Br(),
                                    html.Strong("Return Period Interpretation:"),
                                    html.Br(),
                                    "• T-year return period → 1/T annual probability",
                                    html.Br(),
                                    "Common examples:",
                                    html.Ul([
                                        html.Li("10-year event: 1/10 = 10% chance per year"),
                                        html.Li("50-year event: 1/50 = 2% chance per year"),
                                        html.Li("100-year event: 1/100 = 1% chance per year")
                                    ], style={'marginLeft': '20px', 'marginTop': '5px'}),
                                    html.Br(),
                                    "Note: A 100-year event doesn't mean it occurs every 100 years, ",
                                    "but rather has a 1% chance of occurring in any given year."
                                ]),
                                html.Td([
                                    "• Use for design event estimation",
                                    html.Br(),
                                    "• Consider uncertainty bounds",
                                    html.Br(),
                                    "• Convert between return periods and annual probabilities:",
                                    html.Br(),
                                    html.Em("Annual Probability (%) = (1/T) × 100"),
                                    html.Br(),
                                    "• Remember these are statistical estimates",
                                    html.Br(),
                                    "• Consider climate change implications for future risk"
                                ])
                            ], style={'backgroundColor': COLORS['light']}),
                            
                            # Q-Q Plot
                            html.Tr([
                                html.Td('5. Q-Q Plot'),
                                html.Td([
                                    "Quantile-Quantile plot comparing theoretical vs observed distributions. ",
                                    html.Br(), html.Br(),
                                    html.Strong("Key Features:"),
                                    html.Ul([
                                        html.Li("X-axis: Theoretical quantiles from GEV distribution"),
                                        html.Li("Y-axis: Observed data quantiles"),
                                        html.Li("1:1 line represents perfect theoretical fit")
                                    ]),
                                    html.Br(),
                                    "Points following the 1:1 line indicate the GEV distribution accurately models the data's behaviour across all quantiles."
                                ]),
                                html.Td([
                                    html.Strong("Interpretation:"),
                                    html.Ul([
                                        html.Li("Perfect fit: Points lie exactly on 1:1 line"),
                                        html.Li("Good fit: Points closely follow 1:1 line"),
                                        html.Li("Poor fit: Systematic deviation from 1:1 line"),
                                        html.Li("Check especially the tail behaviour (extreme values)")
                                    ])
                                ])
                            ]),
                            
                            # P-P Plot
                            html.Tr([
                                html.Td('6. P-P Plot'),
                                html.Td([
                                    "Probability-Probability plot comparing cumulative probabilities. ",
                                    html.Br(), html.Br(),
                                    html.Strong("Key Features:"),
                                    html.Ul([
                                        html.Li("X-axis: Theoretical probabilities from GEV"),
                                        html.Li("Y-axis: Empirical probabilities from data"),
                                        html.Li("1:1 line represents perfect probability matching")
                                    ]),
                                    html.Br(),
                                    "Particularly useful for assessing fit in the critical tail regions of the distribution."
                                ]),
                                html.Td([
                                    html.Strong("Interpretation:"),
                                    html.Ul([
                                        html.Li("Ideal: Points parallel to 1:1 line"),
                                        html.Li("S-shaped curve: Indicates poor tail fit"),
                                        html.Li("Systematic deviation: Model bias present"),
                                        html.Li("Use alongside Q-Q plot for complete assessment")
                                    ]),
                                    html.Br(),
                                    "Note: P-P plots are more sensitive to discrepancies in the middle of the distribution, while Q-Q plots are more sensitive to tail behaviour."
                                ])
                            ], style={'backgroundColor': COLORS['light']}),
                            
                            # Statistical Indicators
                            html.Tr([
                                html.Td('5. Statistical Indicators'),
                                html.Td([
                                    html.Strong("GEV Parameters:"),
                                    html.Br(),
                                    "ξ (shape): Tail behaviour control",
                                    html.Br(),
                                    "μ (location): Central tendency",
                                    html.Br(),
                                    "σ (scale): Distribution spread",
                                    html.Br(),
                                    html.Strong("Goodness of Fit:"),
                                    html.Br(),
                                    "R², r, p-value"
                                ]),
                                html.Td([
                                    "• R² > 0.95 indicates excellent fit",
                                    html.Br(),
                                    "• p < 0.05 indicates significant fit",
                                    html.Br(),
                                    "• Consider both Q-Q and P-P plots",
                                    html.Br(),
                                    "• Evaluate parameter uncertainty"
                                ])
                            ]),
                            
                            # Corner Plot Analysis (Bayesian Models Only)
                            html.Tr([
                                html.Td('6. Corner Plot Analysis'),
                                html.Td([
                                    html.Strong("Bayesian Parameter Diagnostics:"), " Visual analysis of posterior parameter distributions and correlations.",
                                    html.Br(),
                                    html.Strong("Diagonal Elements:"), " Marginal histograms showing individual parameter distributions (shape ξ, location μ, scale σ).",
                                    html.Br(),
                                    html.Strong("Off-Diagonal Elements:"), " Scatter plots and contours showing parameter correlations and joint distributions.",
                                    html.Br(),
                                    html.Strong("Red Lines:"), " Posterior median values for each parameter.",
                                    html.Br(), html.Br(),
                                    "This plot is only available for Bayesian models and provides essential diagnostics for MCMC convergence and parameter uncertainty assessment."
                                ]),
                                html.Td([
                                    "• Check for smooth, unimodal parameter distributions.",
                                    html.Br(),
                                    "• Look for excessive parameter correlations (elongated contours).",
                                    html.Br(),
                                    "• Assess MCMC convergence quality (no gaps or artifacts).",
                                    html.Br(),
                                    "• Shape parameter interpretation:",
                                    html.Ul([
                                        html.Li("ξ > 0: Fréchet distribution (heavy tail)"),
                                        html.Li("ξ < 0: Weibull distribution (bounded tail)"),
                                        html.Li("ξ ≈ 0: Gumbel distribution (exponential tail)")
                                    ]),
                                    html.B("• Available only when using Bayesian analysis methods.")
                                ])
                            ], style={'backgroundColor': COLORS['light']}),
                            
                            # Posterior Predictive Checks (Bayesian Models Only)
                            html.Tr([
                                html.Td('7. Posterior Predictive Checks'),
                                html.Td([
                                    html.Strong("Model Validation:"), " Comparison between observed data and model predictions using posterior samples.",
                                    html.Br(),
                                    html.Strong("ECDF Comparison:"), " Empirical cumulative distribution functions of observed vs. simulated data.",
                                    html.Br(),
                                    html.Strong("Confidence Intervals:"), " 90% posterior prediction intervals showing model uncertainty.",
                                    html.Br(),
                                    html.Strong("Summary Statistics:"), " Comparison of key statistics (mean, variance, percentiles) between observed and predicted data.",
                                    html.Br(), html.Br(),
                                    "Good model fit should show observed data falling within the prediction intervals and similar distributional characteristics."
                                ]),
                                html.Td([
                                    "• Observed ECDF should fall within posterior prediction bands.",
                                    html.Br(),
                                    "• Check summary statistics agreement in the comparison table.",
                                    html.Br(),
                                    "• Systematic deviations indicate model inadequacy or poor fit.",
                                    html.Br(),
                                    "• Use for model selection and validation:",
                                    html.Ul([
                                        html.Li("Large deviations → Consider different model"),
                                        html.Li("Good agreement → Model adequately captures data"),
                                        html.Li("Narrow intervals → Low model uncertainty")
                                    ]),
                                    html.B("• Essential for Bayesian model validation and credibility assessment.")
                                ])
                            ])
                        ])
                    ], style={
                        'width': '100%',
                        'borderCollapse': 'collapse',
                        'border': f'1px solid {COLORS["secondary"]}',
                        'marginTop': '20px',
                        'tableLayout': 'fixed'
                    })
                ], style={
                    'backgroundColor': COLORS['background'],
                    'padding': '20px',
                    'borderRadius': '5px',
                    'marginBottom': '20px'
                })
            ], style={
                'margin': '20px',
                'padding': '20px',
                'backgroundColor': COLORS['background'],
                'borderRadius': '5px',
                'boxShadow': f'0 2px 4px {COLORS["shadow"]}'
            }),
            
            # Control Panel
            html.Div([
                # Data Type Selection
                html.Div([
                    html.Label('Data Type:', style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='data-type-selector',
                        options=[
                            {'label': 'Rainfall', 'value': 'rain'},
                            {'label': 'Groundwater', 'value': 'gwl'}
                        ],
                        value=initial_data_type,
                        inline=True
                    )
                ], style={'width': '25%', 'display': 'inline-block', 'marginRight': '20px'}),
                
                # Well Selection
                html.Div([
                    html.Label('Select Well:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='well-selector',
                        options=[{'label': col, 'value': col} for col in well_columns],
                        value=initial_well,
                        clearable=False
                    )
                ], style={'width': '25%', 'display': 'inline-block', 'marginRight': '20px'}),
                
                # Block Period Selection
                html.Div([
                    html.Label('Block Period:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='block-selector',
                        options=suitable_periods,
                        value=default_period,
                        clearable=False
                    )
                ], style={'width': '25%', 'display': 'inline-block', 'marginRight': '20px'}),
                
                # Resample Period Selection
                html.Div([
                    html.Label('Resample Period:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='resample-selector',
                        options=initial_resample_options,
                        value=initial_resample,
                        clearable=False
                    )
                ], style={'width': '20%', 'display': 'inline-block'}),

                # Analysis Type Selection
                html.Div([
                    html.Label('Analysis Type:', style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='analysis-type-selector',
                        options=[
                            {'label': 'Stationary', 'value': 'stationary'},
                            {'label': 'Bayesian (Stationary)', 'value': 'bayesian_stationary'},
                            {'label': 'Bayesian (Non-Stationary)', 'value': 'bayesian_non_stationary'}
                        ],
                        value='stationary',
                        inline=True
                    )
                ], style={'width': '20%', 'display': 'inline-block', 'marginLeft': '20px'})
            ], style={
                'display': 'flex',
                'justifyContent': 'space-around',
                'alignItems': 'center',
                'padding': '20px',
                'backgroundColor': COLORS['hover'],
                'borderRadius': '5px',
                'margin': '20px',
                'boxShadow': f'0 2px 4px {COLORS["shadow"]}'
            }),
            
            # Store for keeping track of data type changes
            dcc.Store(id='data-type-store'),
            
            # Store for managing Bayesian analysis state
            dcc.Store(id='bayesian-analysis-state', data={
                'status': 'idle',  # idle, running, completed, failed
                'analysis_type': 'stationary',
                'parameters': {},
                'results': None,
                'error': None
            }),
            
            # Interval for checking Bayesian completion (minimal, no UI components)
            dcc.Interval(
                id='bayesian-check-interval',
                interval=8000,  # Check every 8 seconds (reduced interference)
                n_intervals=0,
                disabled=True
            ),
            
            # Store for current analysis parameters
            dcc.Store(id='current-analysis-params'),
            
            # Main EVA Plots Container
            html.Div([
                # Time Series Plot with separate loading
                dcc.Loading(
                    id="loading-time-series",
                    type="circle",
                    children=[
                        dcc.Graph(id='time-series-plot')
                    ]
                ),
                
                # EVA Plot with separate loading
                dcc.Loading(
                    id="loading-eva-plot",
                    type="circle",
                    children=[
                        dcc.Graph(id='eva-plot')
                    ]
                )
            ], style={
                'padding': '25px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': f'0 4px 8px {COLORS["shadow"]}',
                'marginBottom': '30px',
                'border': f'1px solid {COLORS["secondary"]}'
            }),

            # Bayesian Analysis Container (Separate Section)
            html.Div([
                html.H2([
                    "Bayesian MCMC Diagnostics",
                    html.Span(" | Parameter Estimation & Model Validation", style={
                        'fontSize': '16px',
                        'fontWeight': 'normal',
                        'color': COLORS['secondary']
                    })
                ], style={
                    'color': COLORS['primary'],
                    'marginBottom': '20px',
                    'borderBottom': f'2px solid {COLORS["primary"]}',
                    'paddingBottom': '10px'
                }),
                
                # Corner Plot Section
                html.Div([
                    dcc.Loading(
                        id="loading-corner-plot",
                        type="default",
                        children=[dcc.Graph(id='corner-plot', style={'height': '900px', 'marginBottom': '20px'})]
                    )
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '8px',
                    'marginBottom': '25px',
                    'border': '1px solid #dee2e6'
                }),
                
                # Posterior Predictive Checks Section
                html.Div([
                    dcc.Loading(
                        id="loading-ppc-plot",
                        type="default",
                        children=[dcc.Graph(id='ppc-ecdf-plot')]
                    ),
                    
                    html.H4("Summary Statistics Comparison", style={
                        'color': COLORS['text'],
                        'marginTop': '25px',
                        'marginBottom': '15px'
                    }),
                    html.P([
                        html.Strong("Note: "), 
                        "Statistics calculated from block maxima. Posterior intervals show model uncertainty."
                    ], style={
                        'fontSize': '12px',
                        'color': COLORS['secondary'], 
                        'marginBottom': '15px',
                        'fontStyle': 'italic'
                    }),
                    dash_table.DataTable(
                        id='ppc-summary-table',
                        columns=[
                            {"name": "Statistic", "id": "Statistic"},
                            {"name": "Observed Value (Block Maxima)", "id": "Observed"},
                            {"name": "Posterior 90% CI (Lower, Upper)", "id": "Posterior_CI"}
                        ],
                        data=[],
                        style_table={
                            'overflowX': 'auto',
                            'marginBottom': '20px'
                        },
                        style_cell={
                            'textAlign': 'center', 
                            'padding': '10px',
                            'fontSize': '12px',
                            'fontFamily': 'Arial, sans-serif'
                        },
                        style_header={
                            'backgroundColor': COLORS['primary'], 
                            'color': 'white', 
                            'fontWeight': 'bold',
                            'fontSize': '13px'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': COLORS['light']
                            }
                        ]
                    )
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '8px',
                    'border': '1px solid #dee2e6'
                })
            ], id='bayesian-plots-container', style={
                'display': 'none',
                'padding': '25px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': f'0 4px 8px {COLORS["shadow"]}',
                'border': f'1px solid {COLORS["secondary"]}'
            })
        ], style={
            'backgroundColor': COLORS['light'],
            'minHeight': '100vh',
            'padding': '20px'
        })
        
        # Register callbacks
        from .callbacks import register_callbacks
        register_callbacks(app)
        
        return app
        
    except Exception as e:
        logging.error(f"Error initializing dashboard: {str(e)}")
        traceback.print_exc()
        raise

__all__ = ['init_eva_dashboard']
