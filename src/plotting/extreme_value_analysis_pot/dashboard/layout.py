"""
POT Dashboard Layout
====================

Dash layout for the POT analysis dashboard.
"""

from dash import dcc, html
import dash_table

from ..constants import COLORS


def create_layout(location_columns):
    """
    Create the Dash application layout for the POT analysis dashboard.
    
    Parameters:
    -----------
    location_columns : list
        List of location column names
        
    Returns:
    --------
    html.Div
        Dash application layout
    """
    return html.Div([
        # Header 
        html.Div([
            html.H1("Extreme Value Analysis Dashboard", 
                style={
                    'textAlign': 'center', 
                    'color': '#2c3e50',
                    'fontSize': '40px',
                    'fontWeight': 'bold',
                    'margin': '0',
                    'padding': '30px 0',
                    'fontFamily': 'Arial, sans-serif',
                    'backgroundColor': 'white'  
                })
        ], style={
            'backgroundColor': 'white',
            'width': '100%',
            'marginBottom': '25px',
            'maxWidth': '1800px',
            'margin': '0 auto',   
            'boxSizing': 'border-box'
        }),

        # Guide Section
        html.Div([
            html.H2('Guide to Extreme Value Analysis: Peak Over Threshold (POT) method', 
                style={'color': '#2c3e50', 'marginBottom': '15px'}),
            
            html.Div([
                html.H3('Overview', style={'color': '#3498db'}),
                html.P([
                    "This dashboard utilises the Peak Over Threshold (POT) method with Generalised Pareto Distribution (GPD) for analysing extreme events. ",
                    html.Br(),
                    html.B("Important:"), " Extreme value analysis (EVA) is a statistical approach that requires a sufficiently large dataset to produce reliable and meaningful results. The more data you have, the more robust your estimates of rare event probabilities and return periods will be. ",
                    html.Br(),
                    "For meaningful EVA, it is recommended to have at least 10 years of high-quality, continuous data, and preferably much more. Short or incomplete records can lead to misleading results, especially for rare events.",
                    html.Br(),
                    "This dashboard provides comprehensive tools for both rainfall and groundwater level data, enabling the assessment of extreme events and their return periods."
                ]),
                
                html.H3('Detailed Analysis Guide', style={'color': '#3498db'}),
                
                # Analysis Components Table
                html.Table([
                    # Header
                    html.Thead(
                        html.Tr([
                            html.Th('Component', style={'width': '15%'}),
                            html.Th('Description', style={'width': '45%'}),
                            html.Th('Interpretation Guidelines', style={'width': '40%'})
                        ], style={
                            'background-color': '#3498db', 
                            'color': 'white',
                            'padding': '12px',
                            'textAlign': 'left',
                            'verticalAlign': 'top',
                            'borderBottom': '2px solid #2980b9'
                        })
                    ),
                    # Body
                    html.Tbody([
                        # Data Selection
                        html.Tr([
                            html.Td('Data Selection Controls', style={
                                'padding': '12px',
                                'verticalAlign': 'top',
                                'fontWeight': 'bold',
                                'borderRight': '1px solid #bdc3c7'
                            }),
                            html.Td([
                                html.Strong('Data Type: '), "Choose between rainfall and groundwater analysis.",
                                html.Br(),
                                html.Strong('Location: '), "Select monitoring location.",
                                html.Br(),
                                html.Strong('Threshold Method: '), "Select method for determining threshold.",
                                html.Br(),
                                html.Strong('Analysis Type: '), "Choose between high extremes (floods) or low extremes (droughts).",
                                html.Br(),
                                html.B("Note: A large, continuous dataset is essential for robust extreme value analysis. Data gaps, short records, or poor quality can significantly reduce the reliability of results.")
                            ], style={
                                'padding': '12px',
                                'verticalAlign': 'top',
                                'borderRight': '1px solid #bdc3c7',
                                'lineHeight': '1.5'
                            }),
                            html.Td([
                                "• Consider data quality, length, and continuity.",
                                html.Br(),
                                "• Check for data gaps, outliers, and measurement errors.",
                                html.Br(),
                                "• Ensure sufficient events above/below threshold (ideally 3-5 per year, over at least 10 years).",
                                html.Br(),
                                html.B("• The reliability of return period estimates increases with the number of years and events in your dataset.")
                            ], style={
                                'padding': '12px',
                                'verticalAlign': 'top',
                                'lineHeight': '1.5'
                            })
                        ], style={'background-color': '#f8f9fa'}),
                        
                        # Threshold Selection
                        html.Tr([
                            html.Td('1. Threshold Selection'),
                            html.Td([
                                "Methods for determining threshold value:",
                                html.Ul([
                                    html.Li("Percentile: Based on data distribution (e.g., 95th percentile for high extremes)."),
                                    html.Li("Fixed: User-defined constant value, based on physical or regulatory criteria."),
                                    html.Li("Mean ± SD: Statistical approach using mean and standard deviation."),
                                    html.Li("Moving Average: Adaptive threshold that accounts for local variability."),
                                    html.Li("Seasonal: Season-specific values to account for seasonal cycles."),
                                    html.Li("POT: Selects a threshold to achieve a target number of events per year."),
                                    html.Li("Declustering: Identifies independent events by separating clusters."),
                                    html.Li("Minimum Annual Maximum: Uses annual block maxima/minima as threshold.")
                                ]),
                                html.Br(),
                                html.B("Tip: The threshold should be high enough to focus on extremes, but not so high that too few events remain for analysis.")
                            ]),
                            html.Td([
                                "• Balance between number of events and their independence.",
                                html.Br(),
                                "• Consider the physical meaning and context of the threshold.",
                                html.Br(),
                                "• Check for seasonal variations and adjust if needed.",
                                html.Br(),
                                "• Aim for 3-5 events per year on average, over as many years as possible.",
                                html.Br(),
                                html.B("• Too few events or too short a record will make return period estimates unreliable.")
                            ])
                        ]),
                        
                        # Time Series Analysis
                        html.Tr([
                            html.Td('2. Time Series Plot'),
                            html.Td([
                                "Visualisation of data with threshold and identified peaks. ",
                                "Shows temporal distribution of extreme events and their magnitudes.",
                                html.Br(),
                                "Colour-coded by season for pattern identification.",
                                html.Br(),
                                html.B("A long, continuous time series is needed to identify trends, clusters, and seasonal patterns in extremes.")
                            ]),
                            html.Td([
                                "• Check event independence and spacing.",
                                html.Br(),
                                "• Look for temporal clustering or gaps.",
                                html.Br(),
                                "• Identify seasonal or long-term patterns.",
                                html.Br(),
                                "• Assess threshold appropriateness and event frequency.",
                                html.Br(),
                                html.B("• Short or gappy records may hide important patterns or bias results.")
                            ])
                        ], style={'background-color': '#f8f9fa'}),
                        
                        # Return Level Plot
                        html.Tr([
                            html.Td('3. Return Level Plot'),
                            html.Td([
                                "Shows relationship between event magnitude and return period. ",
                                "Includes confidence intervals for uncertainty assessment.",
                                html.Br(), html.Br(),
                                "Return Period = 1 / (annual exceedance probability)",
                                html.Br(), html.Br(),
                                html.Strong("Return Period Zones:"),
                                html.Ul([
                                    html.Li("Frequent: 1-2 years (common events)"),
                                    html.Li("Common: 2-5 years"),
                                    html.Li("Uncommon: 5-10 years"),
                                    html.Li("Rare: 10-25 years"),
                                    html.Li("Very Rare: 25-50 years"),
                                    html.Li("Extreme: 50+ years (very rare events)")
                                ]),
                                html.Br(),
                                html.B("Note: Reliable estimation of rare (e.g., 50- or 100-year) events requires much longer records than the return period itself. For example, a 100-year event estimate is highly uncertain if you only have 20 years of data.")
                            ]),
                            html.Td([
                                "• Use for design event estimation (e.g., infrastructure planning).",
                                html.Br(),
                                "• Consider uncertainty bounds and confidence intervals.",
                                html.Br(),
                                "• Evaluate fit at high return periods (beware of extrapolation).",
                                html.Br(),
                                "• Check physical plausibility and compare with historical events.",
                                html.Br(),
                                "• Example interpretations:",
                                html.Ul([
                                    html.Li("10-year event: 10% annual chance"),
                                    html.Li("50-year event: 2% annual chance"),
                                    html.Li("100-year event: 1% annual chance")
                                ]),
                                html.Br(),
                                html.B("• The longer your dataset, the more reliable your return level estimates, especially for rare events.")
                            ])
                        ]),
                        
                        # Diagnostic Plots
                        html.Tr([
                            html.Td('4. Diagnostic Plots'),
                            html.Td([
                                html.Strong("Q-Q Plot:"), " Theoretical vs observed quantiles. Points should follow the 1:1 line if the GPD fit is good.",
                                html.Br(),
                                html.Strong("P-P Plot:"), " Probability comparisons between empirical and theoretical distributions.",
                                html.Br(),
                                html.Strong("Exceedance Plot:"), " Empirical exceedance probabilities compared to fitted GPD.",
                                html.Br(), html.Br(),
                                "These plots help assess the quality of the GPD fit and identify any systematic deviations. Deviations may indicate poor fit, insufficient data, or the need for a different threshold."
                            ]),
                            html.Td([
                                "• Q-Q Plot: Points should follow 1:1 line (good fit).",
                                html.Br(),
                                "• P-P Plot: Uniform distribution along diagonal.",
                                html.Br(),
                                "• Assess model fit quality and adequacy of threshold.",
                                html.Br(),
                                "• Look for systematic deviations, especially in the tails (extremes).",
                                html.Br(),
                                html.B("• More data improves the reliability of diagnostic plots and fit assessment.")
                            ])
                        ], style={'background-color': '#f8f9fa'}),
                        
                        # Statistical Indicators
                        html.Tr([
                            html.Td('5. Statistics'),
                            html.Td([
                                html.Strong("GPD Parameters:"),
                                html.Br(),
                                "ξ (shape): Controls tail behaviour (how heavy or light the extremes are).",
                                html.Br(),
                                "σ (scale): Controls spread of exceedances.",
                                html.Br(),
                                html.Strong("Fit Metrics:"),
                                html.Br(),
                                "R², correlation coefficient, p-value (statistical measures of fit quality).",
                                html.Br(),
                                html.B("Note: Statistical significance and fit quality are more meaningful with larger datasets.")
                            ]),
                            html.Td([
                                "• ξ > 0: Heavy tail (unlimited extremes possible).",
                                html.Br(),
                                "• ξ < 0: Light tail (upper bound exists).",
                                html.Br(),
                                "• R² > 0.95: Excellent fit (but check for overfitting with small data).",
                                html.Br(),
                                "• p < 0.05: Statistically significant.",
                                html.Br(),
                                html.B("• All statistical indicators are more robust and reliable with a large, high-quality dataset.")
                            ])
                        ], style={'background-color': '#f8f9fa'}),
                        
                        # Corner Plot Analysis (Bayesian Models Only)
                        html.Tr([
                            html.Td('6. Corner Plot Analysis'),
                            html.Td([
                                html.Strong("Bayesian Parameter Diagnostics:"), " Visual analysis of posterior parameter distributions and correlations.",
                                html.Br(),
                                html.Strong("Diagonal Elements:"), " Marginal histograms showing individual parameter distributions (shape ξ, scale σ).",
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
                                    html.Li("ξ > 0: Heavy-tailed distribution"),
                                    html.Li("ξ < 0: Light-tailed (bounded) distribution"),
                                    html.Li("ξ ≈ 0: Exponential-like tail behavior")
                                ]),
                                html.B("• Available only when using Bayesian analysis methods.")
                            ])
                        ]),
                        
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
                        ], style={'background-color': '#f8f9fa'})
                    ])
                ], style={
                    'width': '100%',
                    'borderCollapse': 'collapse',
                    'border': '1px solid #95a5a6',
                    'marginTop': '20px',
                    'tableLayout': 'fixed'
                })
            ], style={
                'background-color': '#ffffff',
                'padding': '20px',
                'border-radius': '5px',
                'margin-bottom': '20px'
            }, className='guide-table-container')
        ], style={
            'margin': '20px',
            'padding': '20px',
            'background-color': '#ffffff',
            'border-radius': '5px',
            'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
                
        # Main Controls Container
        html.Div([
            # Data Configuration Section
            html.Div([
                html.H4("Data Configuration", style={
                    'color': '#2c3e50', 
                    'marginBottom': '25px',
                    'fontFamily': 'Arial, sans-serif',
                    'textAlign': 'center',
                    'fontSize': '24px',
                    'borderBottom': '2px solid #eee',
                    'paddingBottom': '10px'
                }),
                
                html.Div([
                    # Analysis Type Selection
                    html.Div([
                        html.Label("Analysis Type:", 
                            style={
                                'fontWeight': 'bold',
                                'fontSize': '16px',
                                'color': '#34495e',
                                'marginBottom': '10px',
                                'display': 'block'
                            }),
                        dcc.RadioItems(
                            id='data-type',
                            options=[
                                {'label': ' Groundwater Levels', 'value': 'gwl'},
                                {'label': ' Rainfall', 'value': 'rain'}
                            ],
                            value='gwl',
                            className='custom-radio',
                            style={'fontSize': '16px'}
                        ),
                    ], style={
                        'padding': '15px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '8px',
                        'flex': '1',
                        'marginRight': '15px'
                    }),

                    # Well Selection
                    html.Div([
                        html.Label("Select Well:", 
                            style={
                                'fontWeight': 'bold',
                                'fontSize': '16px',
                                'color': '#34495e',
                                'marginBottom': '10px',
                                'display': 'block'
                            }),
                        dcc.Dropdown(
                            id='location-selector',
                            options=[{'label': loc, 'value': loc} for loc in location_columns],
                            value=location_columns[0] if location_columns else None,
                            style={'width': '100%'}
                        ),
                    ], style={
                        'padding': '15px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '8px',
                        'flex': '1',
                        'marginRight': '15px'
                    }),

                    # Drought Toggle
                    html.Div([
                        html.Div([
                            html.Label("Analyse Low Flows:", 
                                style={
                                    'fontWeight': 'bold',
                                    'fontSize': '16px',
                                    'color': '#34495e',
                                    'marginRight': '10px',  
                                    'display': 'inline-block',  
                                    'verticalAlign': 'middle'  
                                }),
                            dcc.RadioItems(
                                id='drought-toggle',
                                options=[
                                    {'label': ' No', 'value': False},
                                    {'label': ' Yes', 'value': True}
                                ],
                                value=False,
                                className='custom-radio',
                                style={
                                    'fontSize': '16px',
                                    'display': 'inline-block',  
                                    'verticalAlign': 'middle',  
                                    'gap': '20px'
                                }
                            ),
                        ], style={
                            'display': 'flex',
                            'alignItems': 'center',  
                            'justifyContent': 'center',  
                            'gap': '10px'  
                        }),
                    ], id='drought-toggle-container', 
                    style={
                        'padding': '15px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '8px',
                        'flex': '1',
                        'marginRight': '15px'
                    }),

                    # Time Resolution Selection
                html.Div([
                        html.Label("Time Resolution:", 
                            style={
                                'fontWeight': 'bold',
                                'fontSize': '16px',
                                'color': '#34495e',
                                'marginBottom': '10px',
                                'display': 'block'
                            }),
                        dcc.Dropdown(
                            id='sampling-interval',
                            options=[
                                {'label': 'Original', 'value': 'original'},
                                {'label': 'Hourly', 'value': 'H'},
                                {'label': 'Daily', 'value': 'D'},
                                {'label': 'Weekly', 'value': 'W'},
                                {'label': 'Monthly', 'value': 'M'},
                                {'label': '3-Monthly', 'value': '3M'},
                                {'label': '6-Monthly', 'value': '6M'},
                                {'label': 'Yearly', 'value': 'Y'}
                            ],
                            value='W',  
                            style={'width': '100%'}
                        ),
                    ], style={
                        'padding': '15px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '8px',
                        'flex': '1'
                    }),

                    # GPD Model Type Selection
                    html.Div([
                        html.Label("GPD Model Type:",
                            style={
                                'fontWeight': 'bold',
                                'fontSize': '16px',
                                'color': '#34495e',
                                'marginBottom': '10px',
                                'display': 'block'
                            }),
                        dcc.Dropdown(
                            id='gpd-model-type',
                            options=[
                                {'label': 'Stationary (MLE)', 'value': 'stationary'},
                                {'label': 'Bayesian Stationary', 'value': 'bayesian_stationary'},
                                {'label': 'Bayesian Non-Stationary', 'value': 'bayesian_non_stationary'},
                                {'label': 'Seasonal Nonstationary', 'value': 'seasonal'},
                                {'label': 'Trend Nonstationary', 'value': 'trend'},
                                {'label': 'Covariate Nonstationary', 'value': 'covariate'}
                            ],
                            value='stationary',
                            style={'width': '100%'}
                        ),
                    ], style={
                        'padding': '15px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '8px',
                        'flex': '1',
                        'marginRight': '15px'
                    }),
                ], style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'alignItems': 'flex-start',
                    'marginBottom': '20px',
                    'gap': '10px'
                }),
                # Covariate selector (conditionally shown)
                html.Div([
                    html.Label("Select Covariate:",
                        style={
                            'fontWeight': 'bold',
                            'fontSize': '16px',
                            'color': '#34495e',
                            'marginBottom': '10px',
                            'display': 'block'
                        }),
                    dcc.Dropdown(
                        id='covariate-selector',
                        options=[],  # To be populated dynamically
                        value=None,
                        style={'width': '100%'}
                    ),
                ], id='covariate-selector-container', style={'display': 'none', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'flex': '1', 'marginRight': '15px'}),
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '25px',
                'width': '100%',
                'maxWidth': '1600px',  
                'boxSizing': 'border-box'
            }),

            # Threshold Settings Section
            html.Div([
                html.H4("Threshold Settings", style={
                    'color': '#2c3e50', 
                    'marginBottom': '25px',
                    'fontFamily': 'Arial, sans-serif',
                    'textAlign': 'center',
                    'fontSize': '24px'
                }),
                html.Div([
                    # Threshold Method Selection
                    html.Div([
                        html.Label("Threshold Method:", 
                            style={
                                'fontWeight': 'bold',
                                'marginBottom': '10px',
                                'display': 'block',
                                'fontSize': '16px'
                            }),
                        dcc.Dropdown(
                            id='threshold-method',
                            options=[
                                {'label': '1. Percentile (e.g., 90th percentile)', 'value': 'percentile'},
                                {'label': '2. Fixed Value (enter actual threshold)', 'value': 'fixed'},
                                {'label': '3. Mean + N×Std Dev', 'value': 'mean_std'},
                                {'label': '4. Moving Average + N×Std Dev', 'value': 'moving_avg'},
                                {'label': '5. Seasonal Percentile', 'value': 'seasonal_percentile'},
                                {'label': '6. Peak Over Threshold (POT)', 'value': 'pot'},
                                {'label': '7. Declustering Method', 'value': 'declustering'},
                                {'label': '8. Minimum Annual Maximum', 'value': 'min_annual_max'}
                            ],
                            value='percentile',
                            style={'width': '100%', 'marginBottom': '20px', 'fontSize': '16px'}
                        ),
                        html.Div(id='threshold-help-text', style={
                            'marginTop': '15px',
                            'padding': '15px',
                            'backgroundColor': '#f8f9fa',
                            'borderRadius': '5px',
                            'fontSize': '14px'
                        })
                    ], style={'flex': '2', 'marginRight': '30px'}),
                    
                    # Threshold Value Input
                    html.Div([
                        html.Label("Threshold Value:", 
                            style={
                                'fontWeight': 'bold',
                                'marginBottom': '10px',
                                'display': 'block',
                                'fontSize': '16px'
                            }),
                        dcc.Input(
                            id='threshold-value',
                            type='number',
                            value=90.0,
                            style={
                                'width': '100%', 
                                'padding': '10px',
                                'borderRadius': '4px',
                                'border': '1px solid #ddd',
                                'fontSize': '16px'
                            }
                        ),
                        html.Div(id='threshold-range-text', style={
                            'fontSize': '14px',
                            'color': '#666',
                            'marginTop': '8px'
                        })
                    ], style={'flex': '1'})
                ], style={'display': 'flex'})
            ], style={
                'backgroundColor': 'white',
                'padding': '30px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '25px',
                'width': '100%',
                'maxWidth': '1600px',
                'boxSizing': 'border-box'
            }),

            # Time Series Plot Section
            html.Div([
                html.H4("Time Series Analysis", style={
                    'color': '#2c3e50', 
                    'marginBottom': '25px',
                    'fontFamily': 'Arial, sans-serif',
                    'textAlign': 'center',
                    'fontSize': '24px'
                }),
                dcc.Graph(
                    id='time-series-plot',
                    style={
                        'height': '130vh',  
                        'width': '105%',
                        'minHeight': '400px'
                    },
                    config={
                        'responsive': True,
                        'displayModeBar': True,
                        'scrollZoom': True
                    }
                )
            ], style={
                'padding': '30px',
                'backgroundColor': 'white',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '25px',
                'width': '100%',
                'boxSizing': 'border-box'
            }),

            # Store for managing Bayesian analysis state
            dcc.Store(id='bayesian-analysis-state-pot', data={
                'status': 'idle',  # idle, running, completed, failed
                'analysis_type': 'stationary',
                'parameters': {},
                'results': None,
                'error': None
            }),
            
            # Store for current analysis parameters
            dcc.Store(id='current-analysis-params-pot'),
            


            # Main EVA Analysis Section
            html.Div([
                html.H2([
                    "POT Analysis"
                ], style={
                    'color': COLORS['primary'],
                    'marginBottom': '20px',
                    'borderBottom': f'2px solid {COLORS["primary"]}',
                    'paddingBottom': '10px',
                    'textAlign': 'center',
                    'fontWeight': 'bold'
                }),
                
                dcc.Loading(
                    id="loading-main-plots",
                    type="circle",
                    children=[
                        dcc.Graph(id='main-plot', style={'height': '1000px'})
                    ],
                    style={'height': '1000px', 'overflow': 'hidden'}
                )
            ], id='eva-content', style={
                'display': 'none',
                'padding': '25px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': f'0 4px 8px {COLORS["shadow"]}',
                'marginBottom': '300px',
                'border': f'1px solid {COLORS["secondary"]}',
                'height': '1100px',
                'overflow': 'hidden'
            }),

            
            # Physical Separator - Guaranteed Layout Break
            html.Div(style={
                'height': '400px',
                'width': '100%',
                'clear': 'both',
                'display': 'block',
                'backgroundColor': 'transparent'
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
                        children=[dcc.Graph(id='corner-plot-gpd', style={'height': '900px', 'marginBottom': '20px'})]
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
                        children=[dcc.Graph(id='ppc-ecdf-plot-gpd', style={'height': '600px', 'marginBottom': '20px'})]
                    ),
                    
                    html.H4("Summary Statistics Comparison", style={
                        'color': COLORS['text'],
                        'marginTop': '25px',
                        'marginBottom': '15px'
                    }),
                    html.P([
                        html.Strong("Note: "), 
                        "Statistics calculated from exceedances over threshold. Posterior intervals show model uncertainty."
                    ], style={
                        'fontSize': '12px', 
                        'color': COLORS['secondary'], 
                        'marginBottom': '15px',
                        'fontStyle': 'italic'
                    }),
                    dash_table.DataTable(
                        id='ppc-summary-table-gpd',
                        columns=[
                            {"name": "Statistic", "id": "Statistic"},
                            {"name": "Observed Value (Exceedances)", "id": "Observed"},
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
                    ),
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '8px',
                    'border': '1px solid #dee2e6'
                })
            ], id='bayesian-plots-container-gpd', style={
                'display': 'none',
                'padding': '25px',
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'boxShadow': f'0 4px 8px {COLORS["shadow"]}',
                'border': f'1px solid {COLORS["secondary"]}',
                'marginTop': '100px',
                'clear': 'both',
                'position': 'relative',
                'top': '0px'
            })
        ], style={
            'backgroundColor': '#f8f9fa',
            'minHeight': '100vh',
            'padding': '20px'
        })
    ])

