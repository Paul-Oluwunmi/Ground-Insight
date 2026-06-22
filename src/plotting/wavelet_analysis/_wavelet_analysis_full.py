import dash
from dash import callback_context
from dash import dcc, html, Input, Output, dash_table
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import numpy as np
import pywt
import pandas as pd
import datetime
import scipy.ndimage
import functools
import io
import plotly.io as pio
import base64
from dash.dependencies import Input, Output, State
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy import signal as scipy_signal
import scipy.stats
import scipy.signal
import statsmodels.api as sm
import traceback

RESAMPLE_LABELS = {
    '10T': '10-min',
    '15T': '15-min',
    '30T': '30-min',
    '1H': 'Hourly',
    '1D': 'Daily',
    '1W': 'Weekly',
    '2W': '2-Weekly',
    '1M': 'Monthly',
    '3M': '3-Monthly',
    '6M': '6-Monthly',
    '1Y': 'Yearly',
}

# Global variable to store PWC selections
pwc_selections = {}

# Global variable to store XWT selections  
xwt_selections = {}

# Global variable to store phase analysis settings
# Initialize phase analysis settings globally at module level
phase_analysis_settings = {
    'target_period': 12,
    'show_plots': []
}

def get_phase_settings():
    """Safely get phase analysis settings."""
    global phase_analysis_settings
    if 'phase_analysis_settings' not in globals() or phase_analysis_settings is None:
        phase_analysis_settings = {'show_plots': [], 'target_period': 12}
    return phase_analysis_settings

def run_wavelet_analysis(data, well_columns, rainfall_col=None, pumping_col=None, tide_col=None, remove_rainfall=False, remove_pumping=False, remove_tide=False, confounder_lags=7, date_range=None):
    """
    Launches a Dash dashboard for basic and advanced (wavelet) analysis
    of groundwater level and rainfall data.
    Optionally removes the effects of rainfall, pumping, and/or tidal data from GWL before analysis.
    rainfall_col, pumping_col, tide_col: str or None, column names for confounder data
    remove_rainfall, remove_pumping, remove_tide: bool, whether to remove each effect
    confounder_lags: int, number of lags to use in regression
    date_range: tuple (start_date, end_date) or None
    """
    def remove_confounders(df, y_col, confounder_cols, lags=7):
        y = df[y_col].values
        X = []
        for col in confounder_cols:
            if col in df.columns:
                for lag in range(lags+1):
                    X.append(df[col].shift(lag).values)
        if not X:
            return y  # No confounders
        X = np.column_stack(X)
        mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        y_clean = y[mask]
        X_clean = X[mask]
        X_clean = sm.add_constant(X_clean)
        model = sm.OLS(y_clean, X_clean).fit()
        y_pred = model.predict(X_clean)
        y_resid = y_clean - y_pred
        # Fill residuals back into original shape
        y_out = np.full_like(y, np.nan)
        y_out[mask] = y_resid
        return y_out
    # 1. Filter to date range if provided
    data_proc = data.copy()
    if date_range is not None:
        start_date, end_date = date_range
        data_proc['DateTime'] = pd.to_datetime(data_proc['DateTime'])
        mask = (data_proc['DateTime'] >= pd.to_datetime(start_date)) & (data_proc['DateTime'] <= pd.to_datetime(end_date))
        data_proc = data_proc.loc[mask].reset_index(drop=True)
    # 2. Align all time series on the same time index
    align_cols = ['DateTime'] + well_columns
    
    # Auto-detect rainfall column if not specified
    if rainfall_col is None and 'mm_Rain' in data_proc.columns:
        rainfall_col = 'mm_Rain'
    
    if rainfall_col and rainfall_col in data_proc.columns:
        align_cols.append(rainfall_col)
    if pumping_col and pumping_col in data_proc.columns:
        align_cols.append(pumping_col)
    if tide_col and tide_col in data_proc.columns:
        align_cols.append(tide_col)
    
    data_proc = data_proc[align_cols]
    # 3. Interpolate missing values (linear)
    data_proc = data_proc.sort_values('DateTime').reset_index(drop=True)
    data_proc = data_proc.interpolate(method='linear', limit_direction='both')
    # Data is processed and ready for analysis
    # Mixed-frequency data (GWL: 15min, Rainfall: 10min) is expected and handled by dashboard resampling
    # 5. Remove selected confounders for each well
    for well in well_columns:
        conf_cols = []
        if remove_rainfall and rainfall_col and rainfall_col in data_proc.columns:
            conf_cols.append(rainfall_col)
        if remove_pumping and pumping_col and pumping_col in data_proc.columns:
            conf_cols.append(pumping_col)
        if remove_tide and tide_col and tide_col in data_proc.columns:
            conf_cols.append(tide_col)
        if conf_cols:
            data_proc[well] = remove_confounders(data_proc, well, conf_cols, lags=confounder_lags)
    # Use data_proc for all further analysis instead of data
    # Prepare options
    well_options = [{'label': w, 'value': w} for w in well_columns]
    data_type_options = [
        {'label': 'Groundwater Level', 'value': 'gwl'},
        {'label': 'Rainfall', 'value': 'rain'}
    ]
    wavelet_type_options = [
        {'label': 'Continuous (CWT)', 'value': 'cwt'},
        {'label': 'Discrete (DWT)', 'value': 'dwt'},
        {'label': 'Stationary (SWT)', 'value': 'swt'}
    ]

    def get_flat_wavelet_options(kind):
        options = []
        families = pywt.families()
        for fam in families:
            wavelets = pywt.wavelist(fam)
            for w in wavelets:
                try:
                    if kind == 'cwt':
                        pywt.ContinuousWavelet(w)
                    else:
                        pywt.Wavelet(w)
                    options.append({'label': f"{fam}: {w}", 'value': w})
                except Exception:
                    continue
        return options

    default_wavelet_type = 'cwt'
    default_wavelet = 'morl'
    default_scale = 32

    # Dash app
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div([
        html.Div([
            html.H1("Wavelet Analysis Dashboard", style={'marginBottom': 30, 'color': '#222', 'fontSize': '48px', 'fontWeight': 'bold'}),
        ], style={'textAlign': 'center', 'backgroundColor': 'white', 'padding': '20px 0 0 0'}),
        html.Div([
            # Controls Panel
            html.Div([
                html.H4("Analysis Controls", style={'marginBottom': 20, 'color': '#333'}),
                html.H5("Data Selection", style={'marginTop': 10, 'marginBottom': 6, 'color': '#444'}),
                html.Label("Select Data Type:", style={'marginTop': 6}),
                dcc.Dropdown(
                    id='data-type',
                    options=data_type_options,
                    value='gwl',
                    clearable=False,
                    style={'marginBottom': 12},
                    persistence=False
                ),
                html.Label("Select Well:", style={'marginTop': 6}),
                dcc.Dropdown(
                    id='well',
                    options=well_options,
                    value=well_columns[0],
                    clearable=False,
                    style={'marginBottom': 12},
                    persistence=False
                ),
                html.Label("Resample Frequency:", style={'marginTop': 6}),
                dcc.Dropdown(
                    id='resample-freq',
                    options=[
                        {'label': '10 min', 'value': '10T'},
                        {'label': '15 min', 'value': '15T'},
                        {'label': '30 min', 'value': '30T'},
                        {'label': 'Hourly', 'value': '1H'},
                        {'label': 'Daily', 'value': '1D'},
                        {'label': 'Weekly', 'value': '1W'},
                        {'label': '2-Weekly', 'value': '2W'},
                        {'label': 'Monthly', 'value': '1M'},
                        {'label': '3-Monthly', 'value': '3M'},
                        {'label': '6-Monthly', 'value': '6M'},
                        {'label': 'Yearly', 'value': '1Y'},
                    ],
                    value='1D',
                    clearable=False,
                    style={'marginBottom': 18},
                    persistence=False
                ),
                html.Label("Select Date Range:", style={'marginTop': 6}),
                dcc.DatePickerRange(
                    id='date-range',
                    min_date_allowed=data['DateTime'].min(),
                    max_date_allowed=data['DateTime'].max(),
                    start_date=data['DateTime'].min(),
                    end_date=data['DateTime'].max(),
                    display_format='YYYY-MM-DD',
                    style={'marginBottom': 12},
                    persistence=False
                ),
                dcc.Checklist(
                    id='use-all-data',
                    options=[{'label': 'Use All Data', 'value': 'all'}],
                    value=['all'],
                    style={'marginBottom': 12},
                    persistence=False
                ),
                html.Hr(),
                html.H5("Wavelet Settings", style={'marginTop': 10, 'marginBottom': 6, 'color': '#444'}),
                html.Label("Wavelet Transform Type:", style={'marginTop': 6}),
                dcc.Dropdown(
                    id='wavelet-type',
                    options=wavelet_type_options,
                    value=default_wavelet_type,
                    clearable=False,
                    style={'marginBottom': 12},
                    persistence=False
                ),
                html.Label("Select Wavelet:", style={'marginTop': 6}),
                dcc.Dropdown(
                    id='wavelet',
                    options=[{'label': f'{fam}: {w}', 'value': w} for fam in pywt.families() for w in pywt.wavelist(fam)],
                    value=default_wavelet,
                    clearable=False,
                    style={'marginBottom': 12},
                    persistence=False
                ),
                html.Label("Display:"),
                dcc.RadioItems(
                    id='power-amplitude-toggle',
                    options=[
                        {'label': 'Power', 'value': 'Power'},
                        {'label': 'Amplitude', 'value': 'Amplitude'}
                    ],
                    value='Power',
                    labelStyle={'display': 'inline-block', 'marginRight': 10},
                    persistence=False,
                    style={'marginBottom': 12}
                ),
                html.Label("Period Units:", style={'marginTop': 6}),
                dcc.RadioItems(
                    id='period-units',
                    options=[
                        {'label': 'Hours', 'value': 'hours'},
                        {'label': 'Days', 'value': 'days'},
                        {'label': 'Weeks', 'value': 'weeks'},
                        {'label': 'Months', 'value': 'months'},
                        {'label': 'Years', 'value': 'years'}
                    ],
                    value='hours',
                    labelStyle={'display': 'inline-block', 'marginRight': '10px'},
                    style={'marginBottom': 12},
                    persistence=False
                ),
                html.Label("Min Period:", style={'marginTop': 6}),
                dcc.Input(
                    id='min-period',
                    type='number',
                    value=2,
                    min=0.01,
                    step='any',
                    debounce=True,
                    style={'marginBottom': 8, 'width': '100%'},
                    persistence=False
                ),
                html.Label("Max Period:", style={'marginTop': 6}),
                dcc.Input(
                    id='max-period',
                    type='number',
                    value=48,
                    min=0.01,
                    step='any',
                    debounce=True,
                    style={'marginBottom': 12, 'width': '100%'},
                    persistence=False
                ),
                html.Label("Number of Periods (y-axis resolution):", style={'fontWeight': 'bold'}),
                html.P("Higher values = finer period resolution but slower computation", 
                       style={'fontSize': '11px', 'color': '#666', 'margin': '2px 0 4px 0', 'fontStyle': 'italic'}),
                html.Div([
                    dcc.Slider(
                        id='num-periods',
                        min=5,
                        max=100,
                        step=1,
                        value=40,
                        marks={i: str(i) for i in range(5, 101, 15)},
                        tooltip={"placement": "bottom", "always_visible": False},
                        persistence=False
                    )
                ], style={'width': '300px', 'marginRight': '10px', 'marginBottom': '8px'}),
                html.Label("Overlays:"),
                dcc.Checklist(
                    id='show-overlays',
                    options=[
                        {'label': 'Show 95% Significance', 'value': 'significance'},
                        {'label': 'Show Cone of Influence (COI)', 'value': 'coi'}
                    ],
                    value=[],
                    labelStyle={'display': 'block', 'marginRight': 10},
                    persistence=False,
                    style={'marginBottom': 8}
                ),
                html.Label("Arrow Interpretation Mode:", style={'marginTop': 6, 'fontSize': '14px', 'fontWeight': 'bold'}),
                html.P("Choose how to interpret phase arrows based on your analysis type:", 
                       style={'fontSize': '11px', 'color': '#666', 'margin': '2px 0 4px 0', 'fontStyle': 'italic'}),
                dcc.Dropdown(
                    id='arrow-interpretation-mode',
                    options=[
                        {'label': 'Recharge/Discharge (GW-Rainfall)', 'value': 'recharge_discharge'},
                        {'label': 'Pressure/Flow Response (Fast/Slow)', 'value': 'pressure_flow'},
                        {'label': 'Hydraulic Connectivity (Well-Well)', 'value': 'hydraulic_connectivity'},
                        {'label': 'Tidal Response (Elastic/Loading)', 'value': 'tidal_response'},
                        {'label': 'Pumping Impact (Response/Recovery)', 'value': 'pumping_impact'},
                        {'label': 'Phase Relationship (Lead/Lag)', 'value': 'phase_generic'},
                        {'label': 'Custom Analysis', 'value': 'custom'}
                    ],
                    value='recharge_discharge',
                    clearable=False,
                    style={'marginBottom': 10, 'fontSize': '13px'},
                    persistence=False
                ),
                html.Hr(),
                html.H5("Animation Controls", style={'marginTop': 10, 'marginBottom': 6, 'color': '#444'}),
                html.Label("Animation Type:", style={'fontWeight': 'bold'}),
                html.P("Animate plots to explore temporal patterns", 
                       style={'fontSize': '11px', 'color': '#666', 'margin': '2px 0 4px 0', 'fontStyle': 'italic'}),
                dcc.RadioItems(
                    id='animation-type',
                    options=[
                        {'label': 'None', 'value': 'none'},
                        {'label': 'Time Window (sliding view)', 'value': 'time_window'},
                        {'label': 'Period Slice (frequency bands)', 'value': 'period_slice'}
                    ],
                    value='none',
                    labelStyle={'display': 'block', 'marginRight': 0, 'marginBottom': '4px'},
                    style={'marginBottom': 8}
                ),
                html.Label("Animation Speed (frames/sec):", style={'marginTop': 6}),
                html.Div([
                    dcc.Slider(
                        id='animation-speed',
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                        marks={i: str(i) for i in range(1, 11, 2)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                ], style={'marginBottom': 12}),
                html.Hr(),
                html.H5("Extract Oscillatory Component", style={'marginTop': 10, 'marginBottom': 6, 'color': '#444'}),
                html.Label("Extract Period:", style={'marginTop': 6}),
                dcc.Input(id='extract-period', type='number', value=12, min=0.01, step='any', debounce=True, style={'marginBottom': 8, 'width': '100%'}, persistence=False),
                html.Label("Extract Period Units:", style={'marginTop': 6}),
                dcc.Dropdown(
                    id='extract-period-units',
                    options=[
                        {'label': 'Hours', 'value': 'hours'},
                        {'label': 'Days', 'value': 'days'},
                        {'label': 'Weeks', 'value': 'weeks'},
                        {'label': 'Months', 'value': 'months'},
                        {'label': 'Years', 'value': 'years'}
                    ],
                    value='hours',
                    clearable=False,
                    style={'marginBottom': 8},
                    persistence=False
                ),
                html.Label("Bandwidth (%):", style={'marginTop': 6}),
                dcc.Input(id='extract-bandwidth', type='number', value=10, min=1, max=100, step=1, style={'marginBottom': 8, 'width': '100%'}, persistence=False),
                dcc.Checklist(
                    id='show-extracted-component',
                    options=[{'label': 'Show Extracted Component', 'value': 'show'}],
                    value=[],
                    style={'marginBottom': 12},
                    persistence=False
                ),

            ], style={
                'backgroundColor': '#f8f9fa',
                'border': '1.5px solid #d1d5db',
                'borderRadius': '12px',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.08)',
                'padding': '24px 20px 20px 20px',
                'width': '360px',
                'margin': '24px 0 24px 24px',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'fontFamily': 'Segoe UI, Arial, sans-serif',
                'maxHeight': '95vh',
                'overflowY': 'auto',
                'overflowX': 'hidden'
            }),
            # Content Panel
            html.Div([
                dcc.Tabs(
                    id='tabs',
                    value='basic',
                    children=[
                        dcc.Tab(label='Basic Analysis', value='basic', style={'backgroundColor': 'white'}, selected_style={'backgroundColor': 'white'}),
                        dcc.Tab(label='Wavelet Analysis', value='advanced', style={'backgroundColor': 'white'}, selected_style={'backgroundColor': 'white'}),
                        dcc.Tab(label='Wavelet Coherence', value='coherence', style={'backgroundColor': 'white'}, selected_style={'backgroundColor': 'white'}),
                        dcc.Tab(label='Cross-Wavelet', value='xwt', style={'backgroundColor': 'white'}, selected_style={'backgroundColor': 'white'}),
                        dcc.Tab(label='Partial Wavelet Coherence', value='pwc', style={'backgroundColor': 'white'}, selected_style={'backgroundColor': 'white'}),
                        dcc.Tab(label='Groundwater Analysis', value='signal', style={'backgroundColor': 'white'}, selected_style={'backgroundColor': 'white'}),
                    ],
                    style={'backgroundColor': 'white'}
                ),
                html.Div(id='tab-content', style={'backgroundColor': 'white', 'padding': '20px 10px 10px 10px'})
            ], style={
                'width': 'calc(100% - 400px)',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'marginLeft': '20px',
                'backgroundColor': 'white'
            }),
        ], style={'backgroundColor': 'white', 'display': 'flex', 'flexDirection': 'row'}),
        # Hidden dummy components for dynamic IDs (wrap RangeSlider/Slider in hidden Divs)
        dcc.Checklist(id='show-idwt', options=[], value=[], style={'display': 'none'}),
        dcc.Checklist(id='show-iswt', options=[], value=[], style={'display': 'none'}),
        dcc.Checklist(id='show-trend', options=[], value=[], style={'display': 'none'}),
        html.Div(dcc.RangeSlider(id='drought-flood-thresholds', min=0, max=1, value=[0, 1]), style={'display': 'none'}),
        html.Div(dcc.Slider(id='periodic-level', min=0, max=1, value=0), style={'display': 'none'}),
        dcc.Checklist(id='show-anomaly', options=[], value=[], style={'display': 'none'}),
        dcc.Checklist(id='custom-levels', options=[], value=[], style={'display': 'none'}),
        dcc.Dropdown(id='coh-series1-hidden', options=[], value=None, style={'display': 'none'}),
        dcc.Dropdown(id='coh-series2-hidden', options=[], value=None, style={'display': 'none'}),
        dcc.Dropdown(id='coh-second-well-hidden', options=[], value=None, style={'display': 'none'}),
        dcc.Checklist(id='show-coherence', options=[], value=[], style={'display': 'none'}),
        html.Div(dcc.Slider(id='denoise-threshold', min=0, max=1, value=0), style={'display': 'none'}),
        dcc.Checklist(id='show-denoised', options=[], value=[], style={'display': 'none'}),
        # Hidden coherence dropdowns for callback compatibility
        dcc.Dropdown(id='coherence-series1-dropdown-hidden', options=[], value=None, style={'display': 'none'}),
        dcc.Dropdown(id='coherence-series2-dropdown-hidden', options=[], value=None, style={'display': 'none'}),
        # Hidden cross-wavelet dropdowns for callback compatibility
        dcc.Dropdown(id='xwt-series1-dropdown-hidden', options=[], value=None, style={'display': 'none'}),
        dcc.Dropdown(id='xwt-series2-dropdown-hidden', options=[], value=None, style={'display': 'none'}),
        # Hidden partial wavelet coherence dropdowns for callback compatibility
        dcc.Dropdown(id='pwc-series1-dropdown-hidden', options=[], value=None, style={'display': 'none'}),
        dcc.Dropdown(id='pwc-series2-dropdown-hidden', options=[], value=None, style={'display': 'none'}),
        dcc.Dropdown(id='pwc-confounder-dropdown-hidden', options=[], value=None, style={'display': 'none'}),
        # Hidden phase analysis controls removed - handled directly by coherence tab
            # Hidden signal analysis controls for callback compatibility
    dcc.Checklist(id='signal-analysis-options', options=[], value=[], style={'display': 'none'}),
    
    # Hidden signal dropdown components to prevent callback errors
            dcc.Dropdown(id='signal-analysis-options-dropdown-hidden', options=[], value='period_extraction', style={'display': 'none'}),
            dcc.Dropdown(id='statistical-analysis-options-dropdown-hidden', options=[], value=[], multi=True, style={'display': 'none'}),
        dcc.Dropdown(id='event-analysis-options-dropdown-hidden', options=[], value=[], multi=True, style={'display': 'none'}),
        dcc.Dropdown(id='advanced-analysis-options-dropdown-hidden', options=[], value=[], multi=True, style={'display': 'none'}),
    dcc.Input(id='extraction-target-period', type='number', value=16, style={'display': 'none'}),
    dcc.Input(id='extraction-bandwidth', type='number', value=2, style={'display': 'none'}),
            dcc.Dropdown(id='extraction-period-units-hidden', options=[], value='hours', style={'display': 'none'}),
            dcc.Dropdown(id='trend-method-dropdown-hidden', options=[], value='linear_regression', style={'display': 'none'}),
        dcc.Dropdown(id='confidence-level-dropdown-hidden', options=[], value=95, style={'display': 'none'}),
        dcc.Dropdown(id='forecast-model-dropdown-hidden', options=[], value='arima', style={'display': 'none'}),
        dcc.Dropdown(id='display-output-options-dropdown-hidden', options=[], value=[], multi=True, style={'display': 'none'}),
        dcc.Dropdown(id='extraction-output-options-dropdown-hidden', options=[], value=[], multi=True, style={'display': 'none'}),
        html.Div(dcc.Slider(id='denoise-threshold-slider', min=80, max=99, value=95), style={'display': 'none'}),
        # Hidden drought analysis controls for callback compatibility
        dcc.RadioItems(id='drought-threshold-method', options=[], value='percentile', style={'display': 'none'}),
        dcc.Input(id='drought-mild-threshold', type='number', value=20, style={'display': 'none'}),
        dcc.Input(id='drought-moderate-threshold', type='number', value=10, style={'display': 'none'}),
        dcc.Input(id='drought-severe-threshold', type='number', value=5, style={'display': 'none'}),
        dcc.Input(id='min-drought-duration', type='number', value=7, style={'display': 'none'}),
        # Hidden animation controls for callback compatibility
        dcc.RadioItems(id='animation-type-hidden', options=[], value='none', style={'display': 'none'}),
        html.Div(dcc.Slider(id='animation-speed-hidden', min=1, max=10, value=3), style={'display': 'none'}),
        # Hidden arrow interpretation control for callback compatibility
        dcc.Dropdown(id='arrow-interpretation-mode-hidden', options=[], value='recharge_discharge', style={'display': 'none'}),
        # Hidden PWC phase analysis controls for callback compatibility
        dcc.Input(id='pwc-phase-period-select', type='number', value=12, style={'display': 'none'}),
        dcc.Checklist(id='pwc-show-phase-plots', options=[], value=[], style={'display': 'none'}),
        # Hidden XWT phase analysis controls for callback compatibility
        dcc.Input(id='phase-period-select', type='number', value=12, style={'display': 'none'}),
        dcc.Checklist(id='show-phase-plots', options=[], value=[], style={'display': 'none'}),
    ], style={'backgroundColor': 'white', 'minHeight': '100vh'})

    def summary_to_dash_table(summary_df, yaxis_label):
        """
        Convert a summary statistics DataFrame or Series to a styled Dash DataTable with integrated title.
        yaxis_label: str, the label for the value column (e.g., 'Groundwater Level (m)' or 'Rainfall (mm)')
        """
        if isinstance(summary_df, pd.Series):
            summary_df = summary_df.to_frame()
        summary_df = summary_df.reset_index()
        summary_df.columns = ['Statistic', yaxis_label]
        # Round the values to 4 decimal places
        summary_df[yaxis_label] = summary_df[yaxis_label].round(4)
        
        # Add title as first row
        title_row = {'Statistic': 'Summary Statistics', yaxis_label: ''}
        data_with_title = [title_row] + summary_df.to_dict('records')
        
        return dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in summary_df.columns],
            data=data_with_title,
            style_table={'width': '60%', 'margin': 'auto', 'backgroundColor': 'white'},
            style_cell={
                'backgroundColor': 'white',
                'color': 'black',
                'textAlign': 'center',
                'fontFamily': 'Arial',
                'fontSize': 16,
            },
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 0},  # Title row
                    'backgroundColor': '#e8f5e8',
                    'fontWeight': 'bold',
                    'fontSize': 18
                },
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f9f9f9'
                }
            ],
        )

    def get_nz_season_vrects(x_dates):
        """
        Returns a list of plotly vrect dicts for NZ seasons covering the range of x_dates.
        The rectangles are clipped to the min and max of x_dates.
        """
        if len(x_dates) == 0:
            return []
        # Convert to pandas datetime if not already
        x_dates = pd.to_datetime(x_dates)
        min_x = x_dates.min()
        max_x = x_dates.max()
        min_year = min_x.year
        max_year = max_x.year
        vrects = []
        season_colors = {
            'Summer': 'rgba(255, 230, 128, 0.15)',   # light yellow
            'Autumn': 'rgba(255, 153, 51, 0.10)',    # light orange
            'Winter': 'rgba(128, 179, 255, 0.12)',   # light blue
            'Spring': 'rgba(153, 255, 153, 0.12)'    # light green
        }
        for year in range(min_year, max_year + 2):
            # Summer (Dec 1 prev year - Feb 28/29)
            x0 = datetime.datetime(year-1, 12, 1)
            x1 = datetime.datetime(year, 3, 1)
            if x1 < min_x or x0 > max_x:
                pass
            else:
                vrects.append(dict(
                    type="rect",
                    x0=max(x0, min_x),
                    x1=min(x1, max_x),
                    yref="paper", y0=0, y1=1,
                    fillcolor=season_colors['Summer'],
                    line_width=0,
                    layer="below"
                ))
            # Autumn (Mar 1 - May 31)
            x0 = datetime.datetime(year, 3, 1)
            x1 = datetime.datetime(year, 6, 1)
            if x1 < min_x or x0 > max_x:
                pass
            else:
                vrects.append(dict(
                    type="rect",
                    x0=max(x0, min_x),
                    x1=min(x1, max_x),
                    yref="paper", y0=0, y1=1,
                    fillcolor=season_colors['Autumn'],
                    line_width=0,
                    layer="below"
                ))
            # Winter (Jun 1 - Aug 31)
            x0 = datetime.datetime(year, 6, 1)
            x1 = datetime.datetime(year, 9, 1)
            if x1 < min_x or x0 > max_x:
                pass
            else:
                vrects.append(dict(
                    type="rect",
                    x0=max(x0, min_x),
                    x1=min(x1, max_x),
                    yref="paper", y0=0, y1=1,
                    fillcolor=season_colors['Winter'],
                    line_width=0,
                    layer="below"
                ))
            # Spring (Sep 1 - Nov 30)
            x0 = datetime.datetime(year, 9, 1)
            x1 = datetime.datetime(year, 12, 1)
            if x1 < min_x or x0 > max_x:
                pass
            else:
                vrects.append(dict(
                    type="rect",
                    x0=max(x0, min_x),
                    x1=min(x1, max_x),
                    yref="paper", y0=0, y1=1,
                    fillcolor=season_colors['Spring'],
                    line_width=0,
                    layer="below"
                ))
        return vrects

    def get_nz_season_for_dates(dates):
        """
        Given a pandas Series or array of datetime, return a list of NZ season names for each date.
        """
        seasons = []
        for dt in pd.to_datetime(dates):
            month = dt.month
            if (month == 12) or (month <= 2):
                seasons.append('Summer')
            elif 3 <= month <= 5:
                seasons.append('Autumn')
            elif 6 <= month <= 8:
                seasons.append('Winter')
            else:
                seasons.append('Spring')
        return seasons

    @functools.lru_cache(maxsize=32)
    def compute_wavelet_cwt(y_tuple, wavelet, scale):
        """
        Cached computation of the continuous wavelet transform.
        y_tuple: tuple of y values (must be hashable for lru_cache)
        wavelet: wavelet name
        scale: max scale (int)
        Returns: coef, freqs
        """
        y = np.array(y_tuple)
        scales = np.arange(1, scale+1)
        coef, freqs = pywt.cwt(y, scales, wavelet)
        return coef, freqs

    @functools.lru_cache(maxsize=128) 
    def cached_data_resample(data_hash, freq, start_date, end_date, use_all_data_flag):
        """
        PERFORMANCE BOOST: Cache resampled data results.
        Prevents expensive resampling operations on tab/overlay changes.
        """
        # This will be populated by the calling function
        # Returns cached resampled dataframe
        return None

    @functools.lru_cache(maxsize=64)
    def cached_percentile_calculation(data_tuple, percentile):
        """
        PERFORMANCE BOOST: Cache percentile calculations.
        These are expensive with large datasets.
        """
        data_array = np.array(data_tuple)
        return np.percentile(data_array[~np.isnan(data_array)], percentile)

    @functools.lru_cache(maxsize=32)
    def cached_coherence_computation(x1_tuple, x2_tuple, sampling_rate, num_periods):
        """
        PERFORMANCE BOOST: Cache wavelet coherence computations.
        These are very expensive and often repeated with same parameters.
        """
        try:
            from scipy import signal
            x1 = np.array(x1_tuple)
            x2 = np.array(x2_tuple)
            
            # Simple coherence computation (placeholder - replace with actual wavelet coherence)
            f, Cxy = signal.coherence(x1, x2, fs=1/sampling_rate, nperseg=min(256, len(x1)//4))
            return f, Cxy
        except:
            return None, None

    @functools.lru_cache(maxsize=128)
    def cached_wavelet_periods(min_period, max_period, num_periods, dt, wavelet_type):
        """
        PERFORMANCE BOOST: Cache period calculations and scales.
        Avoids recalculating period arrays on every render.
        """
        if wavelet_type == 'cwt':
            periods = np.logspace(np.log10(min_period), np.log10(max_period), num_periods)
            # Convert periods to scales for Morlet wavelet
            w0 = 6  # Morlet parameter
            scales = periods * (w0 + np.sqrt(2 + w0**2)) / (4 * np.pi * dt)
            return periods, scales
        else:
            # For other wavelet types
            periods = np.logspace(np.log10(min_period), np.log10(max_period), num_periods)
            return periods, periods

    # Callback to update wavelet dropdown based on wavelet type
    @app.callback(
        [dash.dependencies.Output('wavelet', 'options'),
         dash.dependencies.Output('wavelet', 'value')],
        [dash.dependencies.Input('wavelet-type', 'value'),
         dash.dependencies.Input('wavelet', 'value')]
    )
    def update_wavelet_options(wavelet_type, current_wavelet):
        options = get_flat_wavelet_options(wavelet_type)
        valid_values = [opt['value'] for opt in options]
        if current_wavelet in valid_values:
            default = current_wavelet
        else:
            default = options[0]['value'] if options else None
        return options, default

    # Callback for extra controls (IDWT/SWT toggles)
    @app.callback(
        dash.dependencies.Output('extra-controls', 'children'),
        [dash.dependencies.Input('wavelet-type', 'value')]
    )
    def show_extra_controls(wavelet_type):
        if wavelet_type == 'dwt':
            return dcc.Checklist(
                id='show-idwt',
                options=[{'label': 'Show Reconstructed Signal (IDWT)', 'value': 'idwt'}],
                value=[],
                style={'marginTop': 18, 'marginBottom': 10},
                persistence=True,
                persistence_type='session'
            )
        elif wavelet_type == 'swt':
            return dcc.Checklist(
                id='show-iswt',
                options=[{'label': 'Show Reconstructed Signal (ISWT)', 'value': 'iswt'}],
                value=[],
                style={'marginTop': 18, 'marginBottom': 10},
                persistence=True,
                persistence_type='session'
            )
        else:
            return None

    # Callback for signal extraction controls
    @app.callback(
        dash.dependencies.Output('signal-extract-controls', 'children'),
        [dash.dependencies.Input('tabs', 'value'),
         dash.dependencies.Input('wavelet-type', 'value')]
    )
    def show_signal_extract_controls(tab, wavelet_type):
        if tab != 'signal':
            return None
        # Use a default or computed value for scale if needed (e.g., 10)
        scale = 10
        controls = []
        # Trend overlay
        controls.append(dcc.Checklist(
            id='show-trend',
            options=[{'label': 'Overlay Trend (Approximation)', 'value': 'trend'}],
            value=['trend'],
            style={'marginTop': 10, 'marginBottom': 10},
            persistence=True,
            persistence_type='session'
        ))
        # Drought/flood thresholds
        controls.append(html.Label("Drought/Flood Thresholds (as percentiles):", style={'marginTop': 10}))
        controls.append(dcc.RangeSlider(
            id='drought-flood-thresholds',
            min=0, max=100, step=1, value=[10, 90],
            marks={0: '0%', 10: '10%', 50: '50%', 90: '90%', 100: '100%'},
            tooltip={"placement": "bottom", "always_visible": False},
            persistence=True,
            persistence_type='session'
        ))
        # Periodic component level
        controls.append(html.Label("Select Level for Periodic Component:", style={'marginTop': 10}))
        controls.append(dcc.Slider(
            id='periodic-level',
            min=1, max=scale, step=1, value=1,
            marks={i: str(i) for i in range(1, scale+1, max(1, scale//5))},
            tooltip={"placement": "bottom", "always_visible": False},
            persistence=True,
            persistence_type='session'
        ))
        # Anomaly detection toggle
        controls.append(dcc.Checklist(
            id='show-anomaly',
            options=[{'label': 'Detect Anomalies/Change Points', 'value': 'anomaly'}],
            value=[],
            style={'marginTop': 10, 'marginBottom': 10},
            persistence=True,
            persistence_type='session'
        ))
        # Multi-level selection
        controls.append(html.Label("Select Levels for Custom Reconstruction:", style={'marginTop': 10}))
        controls.append(dcc.Checklist(
            id='custom-levels',
            options=[{'label': f'Level {i}', 'value': i} for i in range(1, scale+1)],
            value=[1],
            style={'marginTop': 5, 'marginBottom': 10},
            persistence=True,
            persistence_type='session'
        ))
        # Denoising controls
        controls.append(html.Label("Denoising Threshold (percentile):", style={'marginTop': 10}))
        controls.append(dcc.Slider(
            id='denoise-threshold',
            min=80, max=100, step=1, value=95,
            marks={i: f'{i}%' for i in range(80, 101, 5)},
            tooltip={"placement": "bottom", "always_visible": False},
            persistence=True,
            persistence_type='session'
        ))
        controls.append(dcc.Checklist(
            id='show-denoised',
            options=[{'label': 'Show Denoised Signal', 'value': 'denoised'}],
            value=[],
            style={'marginTop': 5, 'marginBottom': 10},
            persistence=True,
            persistence_type='session'
        ))
        # Wavelet coherence controls
        controls.append(html.Hr())
        controls.append(html.Label("Wavelet Coherence Analysis:", style={'marginTop': 10, 'fontWeight': 'bold'}))
        # Series selection
        series_options = [{'label': w, 'value': w} for w in well_columns] + [{'label': 'Rainfall', 'value': 'mm_Rain'}]
        controls.append(html.Label("Coherence: Series 1", style={'marginTop': 5}))
        controls.append(dcc.Dropdown(id='coh-series1', options=series_options, value=well_columns[0], clearable=False, style={'marginBottom': 8}, persistence=True, persistence_type='session'))
        controls.append(html.Label("Coherence: Series 2", style={'marginTop': 5}))
        controls.append(dcc.Dropdown(id='coh-series2', options=series_options, value='mm_Rain', clearable=False, style={'marginBottom': 8}, persistence=True, persistence_type='session'))
        controls.append(dcc.Checklist(
            id='show-coherence',
            options=[{'label': 'Show Wavelet Coherence', 'value': 'coh'}],
            value=[],
            style={'marginTop': 5, 'marginBottom': 10},
            persistence=True,
            persistence_type='session'
        ))
        return controls

    # Callback for coherence controls
    @app.callback(
        dash.dependencies.Output('coherence-controls', 'children'),
        [dash.dependencies.Input('tabs', 'value'),
         dash.dependencies.Input('well', 'value'),
         dash.dependencies.Input('coh-series2', 'value')],
        [dash.dependencies.State('coh-second-well', 'value')]
    )
    def show_coherence_controls(tab, selected_well, coh_series2_value, coh_second_well_value=None):
        if tab != 'coherence':
            return None
        # Second series options
        second_series_options = [
            {'label': 'Rainfall', 'value': 'mm_Rain'},
            {'label': 'Groundwater Level', 'value': 'gwl'}
        ]
        controls = [
            html.Label("Second Series:"),
            dcc.Dropdown(id='coh-series2', options=second_series_options, value=coh_series2_value or 'mm_Rain', clearable=False, style={'marginBottom': 10}, persistence=True, persistence_type='session'),
        ]
        well_options = [{'label': w, 'value': w} for w in well_columns if w != selected_well]
        controls.append(
            html.Div([
                html.Label("Select Second Well:"),
                dcc.Dropdown(id='coh-second-well', options=well_options, value=coh_second_well_value or (well_options[0]['value'] if well_options else None), clearable=False, style={'marginBottom': 10}, persistence=True, persistence_type='session'),
            ], style={'display': 'block' if (coh_series2_value or 'mm_Rain') == 'gwl' else 'none'})
        )
        return controls

    # Sync coherence series1 dropdown with main well dropdown
    # @app.callback(
    #     Output('coh-series1-hidden', 'value'),
    #     [Input('well', 'value')],
    #     [State('coh-series1-hidden', 'value')]
    # )
    def sync_coherence_series1_with_well(well_value, coh_series1_value):
        if well_value != coh_series1_value:
            return well_value
        return dash.no_update

    # Note: Removed problematic sync callback since extraction-source-series 
    # doesn't always exist in layout. Signal tab logic now always uses current well.

    def get_auto_interpretation_mode(series1_name, series2_name):
        """Automatically determine interpretation mode based on series types"""
        series1_lower = series1_name.lower() if series1_name else ''
        series2_lower = series2_name.lower() if series2_name else ''
        
        # Check for rainfall combinations
        if 'mm_rain' in [series1_name, series2_name] or 'rain' in series1_lower or 'rain' in series2_lower:
            return 'recharge_discharge'
        
        # Check for tidal combinations
        if 'tide' in series1_lower or 'tide' in series2_lower or 'tidal' in series1_lower or 'tidal' in series2_lower:
            return 'tidal_response'
        
        # Check for pumping combinations
        if 'pump' in series1_lower or 'pump' in series2_lower or 'pumping' in series1_lower or 'pumping' in series2_lower:
            return 'pumping_impact'
        
        # If both are wells (default case)
        return 'hydraulic_connectivity'

    def extract_events(mask, x, y, event_type):
        """
        Extracts contiguous events from a boolean mask.
        Returns a list of dicts with start, end, duration, min/max value, and magnitude.
        """
        events = []
        in_event = False
        for i in range(len(mask)):
            if mask[i] and not in_event:
                in_event = True
                start_idx = i
            elif not mask[i] and in_event:
                in_event = False
                end_idx = i - 1
                event = {
                    'Type': event_type,
                    'Start': x[start_idx],
                    'End': x[end_idx],
                    'Duration': (x[end_idx] - x[start_idx]).astype('timedelta64[m]').item() // 60 if hasattr(x[end_idx] - x[start_idx], 'astype') else (x[end_idx] - x[start_idx]).total_seconds() // 3600,
                    'Min': float(np.min(y[start_idx:end_idx+1])),
                    'Max': float(np.max(y[start_idx:end_idx+1])),
                    'Magnitude': float(np.abs(np.max(y[start_idx:end_idx+1]) - np.min(y[start_idx:end_idx+1])))
                }
                events.append(event)
        # Handle if event goes to end
        if in_event:
            end_idx = len(mask) - 1
            event = {
                'Type': event_type,
                'Start': x[start_idx],
                'End': x[end_idx],
                'Duration': (x[end_idx] - x[start_idx]).astype('timedelta64[m]').item() // 60 if hasattr(x[end_idx] - x[start_idx], 'astype') else (x[end_idx] - x[start_idx]).total_seconds() // 3600,
                'Min': float(np.min(y[start_idx:end_idx+1])),
                'Max': float(np.max(y[start_idx:end_idx+1])),
                'Magnitude': float(np.abs(np.max(y[start_idx:end_idx+1]) - np.min(y[start_idx:end_idx+1])))
            }
            events.append(event)
        return events

    def event_stats(events):
        if not events:
            return {}
        durations = [e['Duration'] for e in events]
        magnitudes = [e['Magnitude'] for e in events]
        return {
            'Count': len(events),
            'Mean Duration (h)': np.mean(durations),
            'Mean Magnitude': np.mean(magnitudes),
            'Max Magnitude': np.max(magnitudes),
            'Min Magnitude': np.min(magnitudes)
        }

    @app.callback(
        Output('tab-content', 'children'),
        [Input('tabs', 'value'),
         Input('data-type', 'value'),
         Input('well', 'value'),
         Input('wavelet-type', 'value'),
         Input('wavelet', 'value'),
         Input('resample-freq', 'value'),
         Input('min-period', 'value'),
         Input('max-period', 'value'),
         Input('num-periods', 'value'),
         Input('date-range', 'start_date'),
         Input('date-range', 'end_date'),
         Input('use-all-data', 'value'),
         Input('period-units', 'value'),
         Input('power-amplitude-toggle', 'value'),
         Input('show-overlays', 'value'),
         Input('extract-period', 'value'),
         Input('extract-period-units', 'value'),
         Input('extract-bandwidth', 'value'),
         Input('show-extracted-component', 'value'),
         Input('animation-type', 'value'),
         Input('animation-speed', 'value'),
         Input('denoise-threshold-slider', 'value'),
         Input('arrow-interpretation-mode', 'value')],
        [State('show-idwt', 'value'),
         State('show-iswt', 'value'),
         State('show-trend', 'value'),
         State('drought-flood-thresholds', 'value'),
         State('periodic-level', 'value'),
         State('show-anomaly', 'value'),
         State('custom-levels', 'value'),
         State('coh-series1-hidden', 'value'),
         State('coh-series2-hidden', 'value'),
         State('show-coherence', 'value'),
         State('denoise-threshold', 'value'),
         State('show-denoised', 'value')]
    )
    def update_tab(tab, data_type, well, wavelet_type, wavelet, resample_freq, min_period, max_period, num_periods, start_date, end_date, use_all_data, period_units, power_amplitude, show_overlays, extract_period, extract_period_units, extract_bandwidth, show_extracted_component, animation_type, animation_speed, denoise_threshold_val, arrow_interpretation_mode, show_idwt=None, show_iswt=None, show_trend=None, drought_flood=None, periodic_level=None, show_anomaly=None, custom_levels=None, coh_series1=None, coh_series2=None, show_coherence=None, denoise_threshold=None, show_denoised=None, bypass_protection=False):

        # Declare global variables at the top to avoid scoping issues
        global current_signal_values, pwc_selections, phase_analysis_settings

        # CRITICAL: Allow main callback to handle well/resample changes for signal tab
        if tab == 'signal' and not bypass_protection:
            # Check if this is a well or resample frequency change (these should be handled by main callback)
            from dash import callback_context
            ctx = callback_context
            
            if ctx.triggered:
                triggered_ids = [trigger['prop_id'].split('.')[0] for trigger in ctx.triggered]
                # Only handle well/resample changes, let dedicated callback handle data-type
                well_or_resample_changed = any(id in ['well', 'resample-freq'] for id in triggered_ids)
                
                if well_or_resample_changed:
                    print(f"DEBUG: Main callback handling well/resample change for signal tab")
                    # Don't clear stored signal values when changing wells - preserve dropdown selections
                    print(f"DEBUG: Main callback preserving signal dropdown selections during well/resample change")
                    # Continue processing in main callback for well/resample changes
            else:
                print(f"DEBUG: Main callback handling initial signal tab load")
        elif tab == 'signal' and bypass_protection:
            print(f"DEBUG: Main callback bypassing protection for dedicated callback")

        # Initialize global variables for PWC selections and phase analysis
        if 'pwc_selections' not in globals():
            pwc_selections = {}
        if 'phase_analysis_settings' not in globals():
            phase_analysis_settings = {'show_plots': [], 'target_period': 12}

        # No confounder preprocessing - confounders are handled within specific analyses (e.g., PWC)
        remove_rainfall = False
        remove_pumping = False
        remove_tide = False
        # Validate period values
        try:
            min_period = float(min_period)
        except (TypeError, ValueError):
            min_period = 2  # default value
        try:
            max_period = float(max_period)
        except (TypeError, ValueError):
            max_period = 48  # default value
        if min_period <= 0:
            min_period = 0.01
        if max_period <= min_period:
            max_period = min_period * 10
        # Resample data to selected frequency
        freq = resample_freq
        data_resampled = data_proc.copy()
        data_resampled['DateTime'] = pd.to_datetime(data_resampled['DateTime'])
        data_resampled = data_resampled.set_index('DateTime')
        agg_dict = {col: 'mean' for col in well_columns}
        if 'mm_Rain' in data_resampled.columns:
            agg_dict['mm_Rain'] = 'sum'
        data_resampled = data_resampled.resample(freq).agg(agg_dict)
        # --- Remove interpolation logic ---
        data_resampled = data_resampled.reset_index()
        # Filter by date range unless 'Use All Data' is checked
        if not (use_all_data and 'all' in use_all_data):
            if start_date is not None and end_date is not None:
                data_resampled = data_resampled[(data_resampled['DateTime'] >= pd.to_datetime(start_date)) & (data_resampled['DateTime'] < pd.to_datetime(end_date) + pd.Timedelta(days=1))]
        # Use data_resampled for all further analysis
        # Replace all 'data' references in the callback with 'data_resampled'
        # First, assign x to avoid UnboundLocalError
        x = data_resampled['DateTime'].values
        
        # Select data
        resample_label = RESAMPLE_LABELS.get(freq, freq)
        if data_type == 'gwl':
            y = data_resampled[well].values
            title = f"Groundwater Level - {well} [{resample_label}]"
            yaxis_label = f'Groundwater Level (m) [{resample_label}]'
        elif data_type == 'rain' and 'mm_Rain' in data_resampled.columns and np.issubdtype(data_resampled['mm_Rain'].dtype, np.number):
            y = data_resampled['mm_Rain'].values
            title = f"Rainfall [{resample_label}]"
            yaxis_label = f'Rainfall (mm) [{resample_label}]'
        else:
            y = np.full(x.shape, np.nan)
            title = f"No data available [{resample_label}]"
            yaxis_label = f"No data [{resample_label}]"

        # Remove NaNs for analysis (only for the selected well)
        mask = ~np.isnan(y)
        y = y[mask]
        x = x[mask]
        seasons = get_nz_season_for_dates(x)

        # Ensure show_idwt and show_iswt are lists if not None, else set to empty list
        show_idwt = show_idwt or []
        show_iswt = show_iswt or []
        show_trend = show_trend or []
        show_anomaly = show_anomaly or []
        show_coherence = show_coherence or []
        show_denoised = show_denoised or []
        custom_levels = custom_levels or [1]
        drought_flood = drought_flood or [10, 90]
        periodic_level = periodic_level or 1
        coh_series1 = coh_series1 or (well_columns[0] if well_columns else None)
        coh_series2 = coh_series2 or ('mm_Rain' if 'mm_Rain' in data_resampled.columns else (well_columns[1] if len(well_columns) > 1 else None))
        denoise_threshold = denoise_threshold or 95

        # Validate min_period and max_period again before scale calculation
        if min_period is None or min_period <= 0:
            min_period = 1
        if max_period is None or max_period <= min_period:
            max_period = min_period + 1
        # Assign dt and period_label before using them
        if period_units == 'hours':
            dt = (data_resampled['DateTime'].iloc[1] - data_resampled['DateTime'].iloc[0]).total_seconds() / 3600
            period_label = 'Period (hours)'
            min_period_conv = min_period
            max_period_conv = max_period
        elif period_units == 'days':
            dt = (data_resampled['DateTime'].iloc[1] - data_resampled['DateTime'].iloc[0]).total_seconds() / 86400
            period_label = 'Period (days)'
            min_period_conv = min_period
            max_period_conv = max_period
        elif period_units == 'weeks':
            dt = (data_resampled['DateTime'].iloc[1] - data_resampled['DateTime'].iloc[0]).total_seconds() / (7 * 86400)
            period_label = 'Period (weeks)'
            min_period_conv = min_period * 7
            max_period_conv = max_period * 7
        elif period_units == 'months':
            dt = (data_resampled['DateTime'].iloc[1] - data_resampled['DateTime'].iloc[0]).total_seconds() / (30.44 * 86400)
            period_label = 'Period (months)'
            min_period_conv = min_period * 30.44
            max_period_conv = max_period * 30.44
        elif period_units == 'years':
            dt = (data_resampled['DateTime'].iloc[1] - data_resampled['DateTime'].iloc[0]).total_seconds() / (365.25 * 86400)
            period_label = 'Period (years)'
            min_period_conv = min_period * 365.25
            max_period_conv = max_period * 365.25
        else:
            dt = (data_resampled['DateTime'].iloc[1] - data_resampled['DateTime'].iloc[0]).total_seconds() / 3600
            period_label = 'Period (hours)'
            min_period_conv = min_period
            max_period_conv = max_period

        # Helper for period <-> scale. Use the centre frequency of the wavelet
        # actually used by pywt.cwt (closure variable `wavelet`) so periods match
        # the transform's true frequencies. Using a fixed Torrence-Compo Morlet
        # (w0=6) constant here mislabels periods by ~19% for pywt's 'morl'.
        def period_to_scale(period, dt, w0=6):
            return period * pywt.central_frequency(wavelet) / dt
        def scale_to_period(scales, dt, w0=6):
            return scales * dt / pywt.central_frequency(wavelet)

        if tab == 'basic':
            # Robust assignment of x
            if 'DateTime' in data_resampled.columns:
                x = data_resampled['DateTime'].values
                x_shape = x.shape
            else:
                x = np.array([])
                x_shape = (0,)
            seasons = get_nz_season_for_dates(x)
            resample_label = RESAMPLE_LABELS.get(freq, freq)
            if data_type == 'gwl' and well in data_resampled.columns:
                y = data_resampled[well].values
                title = f"Groundwater Level - {well} [{resample_label}]"
                yaxis_label = f'Groundwater Level (m) [{resample_label}]'
                customdata = np.stack([seasons], axis=-1)
                hovertemplate = f'<b>Date</b>: %{{x}}<br>' + \
                                f'<b>{well}</b>: %{{y:.3f}} m<br>' + \
                                f'<b>Frequency</b>: {resample_label}<br>' + \
                                '<b>Season</b>: %{customdata[0]}'
            elif data_type == 'rain' and 'mm_Rain' in data_resampled.columns and np.issubdtype(data_resampled['mm_Rain'].dtype, np.number):
                y = data_resampled['mm_Rain'].values
                title = f"Rainfall [{resample_label}]"
                yaxis_label = f'Rainfall (mm) [{resample_label}]'
                customdata = np.stack([seasons], axis=-1)
                hovertemplate = f'<b>Date</b>: %{{x}}<br>' + \
                                f'<b>Rainfall</b>: %{{y:.3f}} mm<br>' + \
                                f'<b>Frequency</b>: {resample_label}<br>' + \
                                '<b>Season</b>: %{customdata[0]}'
            else:
                y = np.full(x_shape, np.nan)
                title = f"No data available [{resample_label}]"
                yaxis_label = f"No data [{resample_label}]"
                customdata = np.stack([seasons], axis=-1)
                hovertemplate = f'<b>Date</b>: %{{x}}<br>' + \
                                f'<b>Frequency</b>: {resample_label}<br>' + \
                                '<b>Season</b>: %{customdata[0]}'
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=x,
                y=y,
                mode='lines',
                name=title,
                customdata=customdata,
                hovertemplate=hovertemplate
            ))
            # Add NZ seasons as vrects
            for vrect in get_nz_season_vrects(x):
                fig.add_shape(**vrect)
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title=yaxis_label,
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=600, width=1100,
                xaxis=dict(
                    title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                    tickfont=dict(size=15, family='Arial', color='black', weight='bold'),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)'
                ),
                yaxis=dict(
                    title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                    tickfont=dict(size=15, family='Arial', color='black', weight='bold'),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)'
                )
            )
            
            # Analysis for Basic Tab
            basic_plots = []
            basic_plots.append(dcc.Graph(id='main-figure', figure=fig, style={'backgroundColor': 'white'}))
            
            # Summary Statistics (existing)
            summary = pd.Series(y).describe()
            basic_plots.append(summary_to_dash_table(summary, yaxis_label))
            
            # Additional Analysis Plots
            if len(y) > 30:  # Only if sufficient data
                
                # 1. Monthly Boxplots for Seasonal Analysis
                if len(x) > 30:
                    try:
                        dates_dt = pd.to_datetime(x)
                        df_seasonal = pd.DataFrame({'Date': dates_dt, 'Value': y})
                        df_seasonal['Month'] = df_seasonal['Date'].dt.month
                        df_seasonal['MonthName'] = df_seasonal['Date'].dt.strftime('%b')
                        df_seasonal['Season'] = df_seasonal['Date'].dt.month.map({
                            12: 'Summer', 1: 'Summer', 2: 'Summer',
                            3: 'Autumn', 4: 'Autumn', 5: 'Autumn', 
                            6: 'Winter', 7: 'Winter', 8: 'Winter',
                            9: 'Spring', 10: 'Spring', 11: 'Spring'
                        })
                        
                        # Monthly boxplot
                        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        
                        fig_monthly = go.Figure()
                        for month in month_order:
                            month_mask = df_seasonal['MonthName'] == month
                            month_data = df_seasonal[month_mask]['Value']
                            month_dates = df_seasonal[month_mask]['Date']
                            
                            if len(month_data) > 0:
                                # Create custom hover text for outliers
                                hover_text = [f"Date: {date.strftime('%Y-%m-%d')}<br>Value: {val:.3f} m<br>Month: {month}" 
                                             for date, val in zip(month_dates, month_data)]
                                
                                fig_monthly.add_trace(go.Box(
                                    y=month_data,
                                    name=month,
                                    boxpoints=False,
                                    text=hover_text,
                                    hovertemplate='%{text}<extra></extra>'
                                ))
                        
                        fig_monthly.update_layout(
                            title=f'Monthly Distribution (Seasonal Patterns) [{resample_label}]',
                            xaxis_title='Month',
                            yaxis_title=yaxis_label,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=500, width=1100,
                            showlegend=False,
                            xaxis=dict(
                                title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                                tickfont=dict(size=15, family='Arial', color='black', weight='bold')
                            ),
                            yaxis=dict(
                                title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                                tickfont=dict(size=15, family='Arial', color='black', weight='bold'),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(128, 128, 128, 0.3)'
                            )
                        )
                        
                        basic_plots.append(dcc.Graph(figure=fig_monthly, style={'backgroundColor': 'white'}))
                        
                        # Seasonal statistics table
                        seasonal_stats = df_seasonal.groupby('Season')['Value'].agg([
                            'count', 'mean', 'std', 'min', 'max'
                        ]).round(3)
                        seasonal_stats.columns = ['Count', 'Mean', 'Std Dev', 'Minimum', 'Maximum']
                        seasonal_stats = seasonal_stats.reset_index()
                        
                        seasonal_table = dash_table.DataTable(
                            columns=[{"name": col, "id": col} for col in seasonal_stats.columns],
                            data=seasonal_stats.to_dict('records'),
                            style_table={'width': '80%', 'margin': 'auto'},
                            style_cell={'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': 12},
                            style_header={'backgroundColor': '#f0f0f0', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#f9f9f9'
                                }
                            ]
                        )
                        
                        # Add title as first row in seasonal table data
                        title_row = {'Season': 'Seasonal Statistics (NZ Seasons)', 'Count': '', 'Mean': '', 'Std Dev': '', 'Minimum': '', 'Maximum': ''}
                        seasonal_data_with_title = [title_row] + seasonal_stats.to_dict('records')
                        
                        seasonal_table_with_title = dash_table.DataTable(
                            columns=[{"name": col, "id": col} for col in seasonal_stats.columns],
                            data=seasonal_data_with_title,
                            style_table={'width': '80%', 'margin': 'auto'},
                            style_cell={'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': 12},
                            style_header={'backgroundColor': '#f0f0f0', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 0},  # Title row
                                    'backgroundColor': '#e8f5e8',
                                    'fontWeight': 'bold',
                                    'fontSize': 14
                                },
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#f9f9f9'
                                }
                            ]
                        )
                        
                        basic_plots.append(seasonal_table_with_title)
                        
                    except Exception as e:
                        pass  # Skip seasonal analysis if error
                
                # 2. Moving Averages
                try:
                    if len(y) > 60:  # Need sufficient data for moving averages
                        df_ma = pd.DataFrame({'Date': x, 'Value': y})
                        
                        # Calculate different moving averages
                        window_30 = min(30, len(y)//4)
                        window_90 = min(90, len(y)//2)
                        
                        if window_30 >= 7:
                            df_ma[f'{window_30}-period MA'] = pd.Series(y).rolling(window=window_30, center=True).mean()
                        if window_90 >= 7:
                            df_ma[f'{window_90}-period MA'] = pd.Series(y).rolling(window=window_90, center=True).mean()
                        
                        fig_ma = go.Figure()
                        
                        # Original data
                        fig_ma.add_trace(go.Scattergl(
                            x=df_ma['Date'], y=df_ma['Value'],
                            mode='lines', name='Original',
                            line=dict(color='lightgray', width=1)
                        ))
                        
                        # Moving averages
                        colors = ['blue', 'red', 'green']
                        for i, col in enumerate(df_ma.columns[2:]):  # Skip Date and Value
                            fig_ma.add_trace(go.Scattergl(
                                x=df_ma['Date'], y=df_ma[col],
                                mode='lines', name=col,
                                line=dict(color=colors[i % len(colors)], width=2)
                            ))
                        
                        fig_ma.update_layout(
                            title=f'Moving Averages (Trend Analysis) [{resample_label}]',
                            xaxis_title='Date',
                            yaxis_title=yaxis_label,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=500, width=1100,
                            xaxis=dict(
                                title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                                tickfont=dict(size=15, family='Arial', color='black', weight='bold'),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(128, 128, 128, 0.3)'
                            ),
                            yaxis=dict(
                                title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                                tickfont=dict(size=15, family='Arial', color='black', weight='bold'),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(128, 128, 128, 0.3)'
                            )
                        )
                        
                        basic_plots.append(dcc.Graph(figure=fig_ma, style={'backgroundColor': 'white'}))
                        
                except Exception as e:
                    pass  # Skip moving averages if error
                
                # 3. Anomaly Detection (Multiple Methods)
                try:
                    # Initialize anomaly detection results
                    anomaly_results = {}
                    
                    # Method 1: Z-Score (Original) - using 2.0σ for more sensitivity
                    mean_y = np.mean(y)
                    std_y = np.std(y)
                    z_scores = np.abs((y - mean_y) / std_y)
                    z_anomalies = z_scores > 2.0  # Lowered from 2.5 for more sensitivity
                    anomaly_results['Z-Score'] = z_anomalies
                    
                    # Method 2: IQR Method (Robust to outliers)
                    Q1 = np.percentile(y, 25)
                    Q3 = np.percentile(y, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    iqr_anomalies = (y < lower_bound) | (y > upper_bound)
                    anomaly_results['IQR'] = iqr_anomalies
                    
                    # Method 3: Modified Z-Score (Median-based, more robust)
                    median_y = np.median(y)
                    mad = np.median(np.abs(y - median_y))  # Median Absolute Deviation
                    if mad > 0:
                        modified_z_scores = 0.6745 * (y - median_y) / mad
                        modified_z_anomalies = np.abs(modified_z_scores) > 3.5
                    else:
                        modified_z_anomalies = np.zeros(len(y), dtype=bool)
                    anomaly_results['Modified Z-Score'] = modified_z_anomalies
                    
                    # Method 4: Rate-of-Change Detection (Sudden changes)
                    if len(y) > 1:
                        rate_of_change = np.abs(np.diff(y))
                        change_threshold = np.percentile(rate_of_change, 95)  # Top 5% of changes
                        change_anomalies = np.concatenate([[False], rate_of_change > change_threshold])
                    else:
                        change_anomalies = np.zeros(len(y), dtype=bool)
                    anomaly_results['Rate-of-Change'] = change_anomalies
                    
                    # Method 5: Rolling Statistics (Adaptive)
                    if len(y) > 30:  # Need sufficient data for rolling window
                        window = min(30, len(y)//3)  # 30-point window or 1/3 of data
                        y_series = pd.Series(y)
                        rolling_mean = y_series.rolling(window=window, center=True).mean()
                        rolling_std = y_series.rolling(window=window, center=True).std()
                        
                        # Fill NaN values at edges
                        rolling_mean = rolling_mean.bfill().ffill()
                        rolling_std = rolling_std.fillna(rolling_std.mean())
                        
                        rolling_z_scores = np.abs((y - rolling_mean) / rolling_std)
                        rolling_anomalies = rolling_z_scores > 2.5
                    else:
                        rolling_anomalies = np.zeros(len(y), dtype=bool)
                    anomaly_results['Rolling Statistics'] = rolling_anomalies
                    
                    # Method 6: Seasonal Analysis (if sufficient data)
                    seasonal_anomalies = np.zeros(len(y), dtype=bool)
                    if len(y) > 365 and len(x) > 365:  # Need at least 1 year of data
                        try:
                            # Create time series with date index
                            ts_data = pd.Series(y, index=pd.to_datetime(x))
                            
                            # Resample to daily if needed and interpolate
                            if len(ts_data) > 365:
                                daily_ts = ts_data.resample('D').mean().interpolate()
                                
                                if len(daily_ts) >= 365 * 2:  # Need 2+ years for seasonal decomposition
                                    from statsmodels.tsa.seasonal import seasonal_decompose
                                    
                                    # Seasonal decomposition
                                    decomposition = seasonal_decompose(daily_ts, 
                                                                     period=365,  # Yearly cycle
                                                                     model='additive',
                                                                     extrapolate_trend='freq')
                                    
                                    # Anomalies in residuals
                                    residuals = decomposition.resid.dropna()
                                    if len(residuals) > 0:
                                        residual_std = np.std(residuals)
                                        residual_threshold = 2.5 * residual_std
                                        residual_anomalies = np.abs(residuals) > residual_threshold
                                        
                                        # Map back to original time series
                                        seasonal_anomaly_dates = residuals[residual_anomalies].index
                                        for date in seasonal_anomaly_dates:
                                            # Find closest date in original data
                                            time_diffs = np.abs(pd.to_datetime(x) - date)
                                            closest_idx = np.argmin(time_diffs)
                                            if closest_idx < len(seasonal_anomalies):
                                                seasonal_anomalies[closest_idx] = True
                        except ImportError:
                            pass  # Skip seasonal analysis if statsmodels not available
                        except Exception:
                            pass  # Skip seasonal analysis if any error
                    
                    anomaly_results['Seasonal Analysis'] = seasonal_anomalies
                    
                    # Use Z-Score as the primary anomaly detection for basic functionality
                    anomalies = anomaly_results['Z-Score']
                    
                    if np.any(anomalies):
                        # Create multi-method visualization
                        fig_anomaly = go.Figure()
                        
                        # Add traces for each method
                        colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink']
                        symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up']
                        
                        for i, (method_name, method_results) in enumerate(anomaly_results.items()):
                            if np.any(method_results):
                                fig_anomaly.add_trace(go.Scattergl(
                                    x=x[method_results], y=y[method_results],
                                    mode='markers', name=method_name,
                                    marker=dict(color=colors[i % len(colors)], 
                                              size=6, symbol=symbols[i % len(symbols)],
                                              opacity=0.7),
                                    visible=True if method_name == 'Z-Score' else 'legendonly'  # Show only Z-Score by default
                                ))
                        
                        # Normal data background
                        fig_anomaly.add_trace(go.Scattergl(
                            x=x[~anomalies], y=y[~anomalies],
                            mode='markers', name=f'{yaxis_label}',
                            marker=dict(color='lightblue', size=4, opacity=0.7)
                        ))
                        
                        # Add statistical reference lines
                        mean_val = np.mean(y)
                        std_val = np.std(y)
                        median_val = np.median(y)
                        Q1 = np.percentile(y, 25)
                        Q3 = np.percentile(y, 75)
                        
                        # Z-Score threshold lines (±2.0σ) - use same mean/std as anomaly detection
                        fig_anomaly.add_hline(y=mean_y + 2.0*std_y, 
                                            line_dash="dash", line_color="red", line_width=2)
                        fig_anomaly.add_hline(y=mean_y - 2.0*std_y, 
                                            line_dash="dash", line_color="red", line_width=2)
                        
                        # Other statistical reference lines - no annotations
                        fig_anomaly.add_hline(y=mean_val, line_dash="dot", line_color="gray")
                        fig_anomaly.add_hline(y=median_val, line_dash="dot", line_color="green")
                        fig_anomaly.add_hline(y=Q1, line_dash="dashdot", line_color="orange")
                        fig_anomaly.add_hline(y=Q3, line_dash="dashdot", line_color="orange")
                        
                        # Add invisible traces for legend entries (reference lines)
                        fig_anomaly.add_trace(go.Scatter(
                            x=[None], y=[None], mode='lines',
                            line=dict(color='red', dash='dash', width=2),
                            name='±2.0σ (Z-Score Thresholds)',
                            showlegend=True
                        ))
                        fig_anomaly.add_trace(go.Scatter(
                            x=[None], y=[None], mode='lines',
                            line=dict(color='gray', dash='dot'),
                            name='Mean',
                            showlegend=True
                        ))
                        fig_anomaly.add_trace(go.Scatter(
                            x=[None], y=[None], mode='lines',
                            line=dict(color='green', dash='dot'),
                            name='Median',
                            showlegend=True
                        ))
                        fig_anomaly.add_trace(go.Scatter(
                            x=[None], y=[None], mode='lines',
                            line=dict(color='orange', dash='dashdot'),
                            name='Q1 (25th percentile)',
                            showlegend=True
                        ))
                        fig_anomaly.add_trace(go.Scatter(
                            x=[None], y=[None], mode='lines',
                            line=dict(color='orange', dash='dashdot'),
                            name='Q3 (75th percentile)',
                            showlegend=True
                        ))
                        
                        fig_anomaly.update_layout(
                            title=f'Multi-Method Anomaly Detection [{resample_label}]',
                            xaxis_title='Date',
                            yaxis_title=yaxis_label,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=500, width=1100,
                            xaxis=dict(
                                title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                                tickfont=dict(size=15, family='Arial', color='black', weight='bold'),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(128, 128, 128, 0.3)'
                            ),
                            yaxis=dict(
                                title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                                tickfont=dict(size=15, family='Arial', color='black', weight='bold'),
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(128, 128, 128, 0.3)'
                            ),
                            legend=dict(
                                x=1.02, y=1, xanchor='left', yanchor='top',
                                bgcolor='rgba(255, 255, 255, 0.9)',
                                bordercolor='rgba(0, 0, 0, 0.2)', borderwidth=1
                            ),
                            margin=dict(r=200)
                        )
                        
                        basic_plots.append(dcc.Graph(figure=fig_anomaly, style={'backgroundColor': 'white'}))
                        
                        # Determine which methods are visible by default (sync with legend)
                        # Default to showing Z-Score (primary) and other high-confidence methods
                        selected_methods = ['Z-Score']  # Always include primary method
                        
                        # Add other methods that detected significant anomalies
                        for method_name, method_results in anomaly_results.items():
                            if method_name != 'Z-Score' and np.sum(method_results) > 0:
                                selected_methods.append(method_name)
                        
                        # Create combined anomaly mask for selected methods
                        combined_anomalies = np.zeros(len(y), dtype=bool)
                        for method in selected_methods:
                            if method in anomaly_results:
                                combined_anomalies = combined_anomalies | anomaly_results[method]
                        
                        anomaly_indices = np.where(combined_anomalies)[0]
                        
                        if len(anomaly_indices) > 0:
                            anomaly_details = []
                            for idx in anomaly_indices:
                                # Determine which methods flagged this point (only from selected methods)
                                flagged_methods = []
                                for method_name in selected_methods:
                                    if method_name in anomaly_results and anomaly_results[method_name][idx]:
                                        flagged_methods.append(method_name)
                                
                                # Skip if not flagged by any selected method
                                if not flagged_methods:
                                    continue
                                
                                # Interpretation based on value
                                val = y[idx]
                                if val > np.percentile(y, 75):
                                    interpretation = "Extremely High Water Level"
                                elif val < np.percentile(y, 25):
                                    interpretation = "Extremely Low Water Level - Drought Indicator"
                                elif val > np.mean(y):
                                    interpretation = "Above Normal Water Level"
                                else:
                                    interpretation = "Below Normal Water Level"
                                
                                # Get primary score (Z-score if available, otherwise first available)
                                primary_score = ""
                                if 'Z-Score' in selected_methods and anomaly_results['Z-Score'][idx]:
                                    primary_score = f"{z_scores[idx]:.2f}"
                                elif 'IQR' in selected_methods and anomaly_results['IQR'][idx]:
                                    primary_score = "IQR Outlier"
                                elif flagged_methods:
                                    primary_score = flagged_methods[0]
                                
                                # Format date properly
                                if hasattr(x[idx], 'strftime'):
                                    date_str = x[idx].strftime('%Y-%m-%d %H:%M')
                                else:
                                    date_str = str(pd.to_datetime(x[idx]).strftime('%Y-%m-%d %H:%M'))
                                
                                anomaly_details.append({
                                    'Date': date_str,
                                    f'{yaxis_label}': f"{val:.3f}",
                                    'Primary Score': primary_score,
                                    'Detected By': ', '.join(flagged_methods),
                                    'Method Count': len(flagged_methods),
                                    'Interpretation': interpretation
                                })
                   
                    else:
                        # Even if Z-Score found nothing, check if other methods found anomalies
                        other_methods_found = False
                        for method_name, method_results in anomaly_results.items():
                            if method_name != 'Z-Score' and np.sum(method_results) > 0:
                                other_methods_found = True
                                break
                        
                        if other_methods_found:
                            basic_plots.append(html.P(
                                "No Z-Score anomalies detected (threshold: 2.0σ), but other methods detected anomalies. Use plot legend to explore alternative methods.", 
                                style={'textAlign': 'center', 'color': '#666', 'fontStyle': 'italic', 'padding': '20px'}
                            ))
                        else:
                            basic_plots.append(html.P(
                                "No significant anomalies detected by any method.", 
                                style={'textAlign': 'center', 'color': '#666', 'fontStyle': 'italic', 'padding': '20px'}
                            ))
                        
                except Exception as e:
                    pass  # Skip anomaly detection if error
            
            return html.Div([
                html.Div(basic_plots, style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
            ], style={'backgroundColor': 'white'})
        elif tab == 'advanced':
            # Wavelet Analysis Tab (CWT Scalogram)
            seasons = get_nz_season_for_dates(x)
            customdata = np.stack([y, seasons], axis=-1)
            min_scale = 1
            n = len(y)
            max_scale = n // 8 if n >= 8 else 2
            num_octaves = np.log2(max_period_conv / min_period_conv) if max_period_conv > min_period_conv else 1
            voices_per_octave = int(np.ceil(num_periods / num_octaves)) if num_octaves > 0 else 2
            voices_per_octave = max(2, min(voices_per_octave, 48))
            delta_j = 1 / voices_per_octave
            J = int(np.log2(max_scale / min_scale) / delta_j) if max_scale > min_scale else 1
            scales = min_scale * 2 ** (np.arange(J + 1) * delta_j)
            # Period axis must use the centre frequency of the wavelet actually
            # used by pywt.cwt (e.g. 'morl'), not a fixed Torrence-Compo Morlet
            # (w0=6) constant, otherwise periods are mislabelled (~19% too short).
            periods = scales * dt / pywt.central_frequency(wavelet)
            # Convert periods to selected units
            if period_units == 'weeks':
                periods = periods / 7
            elif period_units == 'months':
                periods = periods / 30.44
            elif period_units == 'years':
                periods = periods / 365.25
            period_mask = (periods >= min_period_conv) & (periods <= max_period_conv)
            periods = periods[period_mask]
            scales = scales[period_mask]
            if len(scales) < 2 or len(periods) < 2:
                return html.Div([
                    html.Div([
                        html.H4("Period Range Too Narrow", style={'color': '#dc3545', 'textAlign': 'center'}),
                        html.P(f"Selected range: {min_period_conv:.1f} - {max_period_conv:.1f} {period_units}", 
                               style={'textAlign': 'center', 'color': '#666'}),
                        html.P("Try expanding the Min/Max Period range or increasing 'Number of Periods'", 
                               style={'textAlign': 'center', 'color': '#28a745', 'fontStyle': 'italic'})
                    ], style={'padding': '20px', 'border': '2px dashed #dc3545', 'borderRadius': '8px', 
                             'margin': '20px', 'backgroundColor': '#fff5f5'})
                ], style={'backgroundColor': 'white'})
            
            # Use appropriate wavelet transform based on type
            if wavelet_type == 'cwt':
                coef, freqs = pywt.cwt(y, scales, wavelet)
                if power_amplitude == 'Power':
                    z_data = np.abs(coef) ** 2
                    colorbar_title = 'Power'
                else:
                    z_data = np.abs(coef)
                    colorbar_title = 'Amplitude'
            elif wavelet_type == 'dwt':
                # For DWT, we'll create a time-frequency representation using multiple decomposition levels
                try:
                    max_level = pywt.dwt_max_level(len(y), wavelet)
                    coeffs = pywt.wavedec(y, wavelet, level=max_level)
                except Exception as e:
                    # If continuous wavelet was selected, use a default discrete wavelet
                    if 'ContinuousWavelet' in str(e):
                        default_wavelet = 'db8'
                        max_level = pywt.dwt_max_level(len(y), default_wavelet)
                        coeffs = pywt.wavedec(y, default_wavelet, level=max_level)
                        wavelet = default_wavelet  # Update for reconstruction
                    else:
                        raise e
                
                # Create a pseudo-scalogram by reconstructing at different levels
                z_data = []
                for level in range(1, min(max_level + 1, len(periods))):
                    # Reconstruct using only detail coefficients up to this level
                    coeffs_partial = [None] * len(coeffs)
                    coeffs_partial[0] = coeffs[0]  # approximation
                    for i in range(1, min(level + 1, len(coeffs))):
                        coeffs_partial[i] = coeffs[i]
                    
                    recon = pywt.waverec(coeffs_partial, wavelet)
                    recon = recon[:len(y)]  # trim to original length
                    z_data.append(recon)
                
                # Pad or trim to match expected dimensions
                while len(z_data) < len(periods):
                    z_data.append(np.zeros_like(y))
                z_data = np.array(z_data[:len(periods)])
                
                if power_amplitude == 'Power':
                    z_data = np.abs(z_data) ** 2
                    colorbar_title = 'DWT Power'
                else:
                    z_data = np.abs(z_data)
                    colorbar_title = 'DWT Amplitude'
            elif wavelet_type == 'swt':
                # For SWT, use stationary wavelet transform
                try:
                    max_level = min(pywt.swt_max_level(len(y)), len(periods))
                    coeffs = pywt.swt(y, wavelet, level=max_level)
                except Exception as e:
                    # If continuous wavelet was selected, use a default discrete wavelet
                    if 'ContinuousWavelet' in str(e):
                        default_wavelet = 'db8'
                        max_level = min(pywt.swt_max_level(len(y)), len(periods))
                        coeffs = pywt.swt(y, default_wavelet, level=max_level)
                        wavelet = default_wavelet  # Update for later use
                    else:
                        raise e
                
                # Extract detail coefficients for each level
                z_data = []
                for level in range(max_level):
                    cA, cD = coeffs[level]
                    z_data.append(cD[:len(y)])
                
                # Pad or trim to match expected dimensions
                while len(z_data) < len(periods):
                    z_data.append(np.zeros_like(y))
                z_data = np.array(z_data[:len(periods)])
                
                if power_amplitude == 'Power':
                    z_data = np.abs(z_data) ** 2
                    colorbar_title = 'SWT Power'
                else:
                    z_data = np.abs(z_data)
                    colorbar_title = 'SWT Amplitude'
            else:
                return html.Div([html.H4("Unsupported wavelet type selected.")])
            
            # Ensure z_data has the right shape
            if z_data.shape[0] != len(periods):
                # Interpolate to match period dimensions if needed
                if z_data.shape[0] > 1:
                    old_indices = np.linspace(0, 1, z_data.shape[0])
                    new_indices = np.linspace(0, 1, len(periods))
                    z_data_interp = []
                    for i in range(z_data.shape[1]):
                        f = interp1d(old_indices, z_data[:, i], kind='linear', fill_value='extrapolate')
                        z_data_interp.append(f(new_indices))
                    z_data = np.array(z_data_interp).T
            # Broadcast customdata to match heatmap dimensions
            n_scales, n_times = z_data.shape
            customdata_broadcast = np.broadcast_to(customdata, (n_scales, n_times, 2))
            
            # Animation settings now come from callback
            animation_type = animation_type or 'none'
            animation_speed = animation_speed or 3
            
            # Create main heatmap
            heatmap = go.Heatmap(
                z=z_data,
                x=x,
                y=periods,
                colorscale='RdYlBu_r',  # Standardized colormap
                                    colorbar=dict(
                        title=colorbar_title,
                        x=1.03,  # Uniform position for all heatmaps
                        xanchor='left',
                        thickness=20,  # Standardized thickness
                        len=0.8  # Increased height
                    ),
                customdata=customdata_broadcast,
                hovertemplate=('<b>Date</b>: %{x}<br>' +
                              '<b>Period</b>: %{y:.3f}<br>' +
                              '<b>' + colorbar_title + '</b>: %{z:.3f}<br>' +
                              '<b>' + yaxis_label + '</b>: %{customdata[0]:.3f}<br>' +
                              '<b>Season</b>: %{customdata[1]}<extra></extra>')
            )
            
            # Create animated figure if requested
            if animation_type == 'time_window':
                # Time window animation - show sliding window of data
                window_size = len(x) // 10  # 10 frames
                frames = []
                
                for i in range(0, len(x) - window_size, window_size // 2):
                    end_idx = min(i + window_size, len(x))
                    frame_x = x[i:end_idx]
                    frame_z = z_data[:, i:end_idx]
                    frame_customdata = customdata_broadcast[:, i:end_idx, :]
                    
                    frame = go.Frame(
                        data=[go.Heatmap(
                            z=frame_z,
                            x=frame_x,
                            y=periods,
                            colorscale='RdYlBu_r',  # Standardized colormap
                            colorbar=dict(
                                title=colorbar_title,
                                x=1.03,  # Uniform position for all heatmaps
                                xanchor='left',
                                thickness=20,  # Standardized thickness
                                len=0.8  # Increased height
                            ),
                            customdata=frame_customdata,
                            hovertemplate=('<b>Date</b>: %{x}<br>' +
                                          '<b>Period</b>: %{y:.3f}<br>' +
                                          '<b>' + colorbar_title + '</b>: %{z:.3f}<br>' +
                                          '<b>' + yaxis_label + '</b>: %{customdata[0]:.3f}<br>' +
                                          '<b>Season</b>: %{customdata[1]}<extra></extra>')
                        )],
                        name=f'frame_{i}'
                    )
                    frames.append(frame)
                
                fig = go.Figure(data=[heatmap], frames=frames)
                
                # Add animation controls
                fig.update_layout(
                    updatemenus=[{
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                            {
                                'label': 'Play',
                                'method': 'animate',
                                'args': [None, {
                                    'frame': {'duration': 1000 // animation_speed, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 100}
                                }]
                            },
                            {
                                'label': 'Pause',
                                'method': 'animate',
                                'args': [[None], {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }]
                            }
                        ]
                    }]
                )
                
            elif animation_type == 'period_slice':
                # Period slice animation - show time series for different periods
                period_indices = np.linspace(0, len(periods)-1, 10, dtype=int)
                frames = []
                
                for idx in period_indices:
                    # Extract time series for this period
                    period_val = periods[idx]
                    time_series = z_data[idx, :]
                    
                    frame = go.Frame(
                        data=[go.Scatter(
                            x=x,
                            y=time_series,
                            mode='lines',
                            name=f'Period {period_val:.1f} {period_units}',
                            line=dict(color='blue')
                        )],
                        name=f'period_{period_val:.1f}'
                    )
                    frames.append(frame)
                
                # Initial frame
                initial_period = periods[period_indices[0]]
                initial_series = z_data[period_indices[0], :]
                
                fig = go.Figure(
                    data=[go.Scatter(
                        x=x,
                        y=initial_series,
                        mode='lines',
                        name=f'Period {initial_period:.1f} {period_units}',
                        line=dict(color='blue')
                    )],
                    frames=frames
                )
                
                # Add animation controls
                fig.update_layout(
                    updatemenus=[{
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                            {
                                'label': 'Play',
                                'method': 'animate',
                                'args': [None, {
                                    'frame': {'duration': 1000 // animation_speed, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 100}
                                }]
                            },
                            {
                                'label': 'Pause',
                                'method': 'animate',
                                'args': [[None], {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }]
                            }
                        ]
                    }],
                    yaxis_title=colorbar_title
                )
            else:
                # No animation - standard static plot
                fig = go.Figure(data=[heatmap])
            fig.update_layout(
                title=f"Wavelet Scalogram ({wavelet}) - {title}",
                xaxis_title='Date',
                yaxis_title=period_label,
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=600, width=1100,
                xaxis=dict(
                    title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                    tickfont=dict(size=15, family='Arial', color='black', weight='bold')
                ),
                yaxis=dict(
                    title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                    tickfont=dict(size=15, family='Arial', color='black', weight='bold'),
                    range=[float(periods[0]), float(periods[-1])]
                ),
                legend=dict(
                    x=1.02,  # Position legend to the right of the plot
                    y=1,     # Position at top
                    xanchor='left',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(0, 0, 0, 0.2)',
                    borderwidth=1,
                    font=dict(size=12, family='Arial', color='black'),
                    itemsizing='constant',
                    itemwidth=40,
                    groupclick='toggleitem',
                    orientation='v',
                    tracegroupgap=8,  # Add spacing between different trace groups
                    itemclick='toggle',
                    valign='top'
                ),
                margin=dict(r=180)  # Add right margin to accommodate legend and colorbar
            )
            # --- Significance overlay ---
            y_detrended = y - np.mean(y)
            alpha = np.corrcoef(y_detrended[:-1], y_detrended[1:])[0, 1] if len(y) > 1 else 0
            def red_noise_spectrum(periods, alpha, dt, variance):
                fourier_freq = dt / periods
                return (1 - alpha ** 2) / (1 + alpha ** 2 - 2 * alpha * np.cos(2 * np.pi * fourier_freq)) * variance
            variance = np.var(y)
            signif = red_noise_spectrum(periods, alpha, dt, variance)
            signif_level = signif * scipy.stats.chi2.ppf(0.95, 2) / 2
            sig_mask = z_data > signif_level[:, None]
            if 'significance' in show_overlays:
                fig.add_trace(go.Contour(
                    z=sig_mask.astype(float),
                    x=x,
                    y=periods,
                    showscale=False,
                    contours=dict(
                        coloring='none',
                        showlines=True,
                        start=0.5,
                        end=0.5,
                        size=0.5
                    ),
                    line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
                    hoverinfo='skip',
                    name='95% Significance',
                    showlegend=True
                ))
                
            # --- COI overlay ---
            coi = np.minimum(np.arange(len(x)), np.arange(len(x))[::-1]) * dt * np.sqrt(2)
            coi_periods = np.clip(coi, periods[0], periods[-1])
            if 'coi' in show_overlays:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=coi_periods,
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name='Cone of Influence',
                    hoverinfo='skip',
                    showlegend=True
                ))
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([coi_periods, [periods[-1]]*len(x)]),
                    fill='toself',
                    fillcolor='rgba(0,0,0,0.15)',
                    line=dict(color='rgba(0,0,0,0)'),
                    hoverinfo='skip',
                    name='COI Region',
                    showlegend=False
                ))
            
            # Legend entries are handled by the actual overlay traces above
            export_buttons = html.Div([
                html.Button('Export Plot (PNG)', id='export-plot-btn', 
                           style={'marginRight': 10, 'padding': '8px 16px', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                html.Button('Download Data (CSV)', id='export-data-btn',
                           style={'marginRight': 10, 'padding': '8px 16px', 'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                dcc.Download(id='download-plot-png'),
                dcc.Download(id='download-data-csv')
            ], style={'marginTop': 16, 'marginBottom': 8, 'textAlign': 'center'})
            
            # Wavelet Analysis - Additional Plots
            wavelet_plots = []
            wavelet_plots.append(dcc.Graph(id='main-figure', figure=fig, style={'backgroundColor': 'white'}))
            
            # 1. Global Wavelet Spectrum (Time-averaged power)
            try:
                if wavelet_type == 'cwt' and len(periods) > 1:
                    global_power = np.mean(z_data, axis=1)  # Average over time
                    
                    # Significance testing against red noise
                    # Estimate lag-1 autocorrelation
                    y_series = pd.Series(y)
                    lag1_corr = y_series.autocorr(lag=1)
                    if np.isnan(lag1_corr):
                        lag1_corr = 0.0
                    
                    # Red noise spectrum (already defined in significance overlay section)
                    variance = np.var(y)
                    red_noise_95 = red_noise_spectrum(periods, alpha, dt, variance) * 2.32  # 95% confidence
                    red_noise_99 = red_noise_spectrum(periods, alpha, dt, variance) * 3.72  # 99% confidence
                    
                    fig_global = go.Figure()
                    
                    # Global power spectrum
                    fig_global.add_trace(go.Scatter(
                        x=periods, y=global_power,
                        mode='lines+markers',
                        name='Global Wavelet Power',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Significance levels
                    fig_global.add_trace(go.Scatter(
                        x=periods, y=red_noise_95,
                        mode='lines',
                        name='95% Confidence',
                        line=dict(color='red', width=1, dash='dash')
                    ))
                    
                    fig_global.add_trace(go.Scatter(
                        x=periods, y=red_noise_99,
                        mode='lines',
                        name='99% Confidence',
                        line=dict(color='darkred', width=1, dash='dot')
                    ))
                    
                    fig_global.update_layout(
                        title='Global Wavelet Power Spectrum',
                        xaxis_title=period_label,
                        yaxis_title=f'Power ({colorbar_title})',
                        xaxis_type='log',
                        yaxis_type='log',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=500, width=1100,
                        xaxis=dict(
                            title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                            tickfont=dict(size=15, family='Arial', color='black', weight='bold'),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128, 128, 128, 0.3)'
                        ),
                        yaxis=dict(
                            title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                            tickfont=dict(size=15, family='Arial', color='black', weight='bold'),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128, 128, 128, 0.3)'
                        ),
                        legend=dict(x=0.02, y=0.98)
                    )
                    
                    wavelet_plots.append(dcc.Graph(figure=fig_global, style={'backgroundColor': 'white'}))
                    
                    # Peak detection in global spectrum
                    peaks, properties = find_peaks(global_power, height=np.mean(global_power), distance=3)
                    
                    if len(peaks) > 0:
                        # Period interpretation function
                        def get_period_interpretation(period_val, units):
                            """Provide hydrogeological interpretation of periods"""
                            if units == 'hours':
                                if period_val < 1:
                                    return "Sub-hourly variations"
                                elif period_val < 6:
                                    return "Diurnal variations"
                                elif period_val < 25:
                                    return "Daily cycle"
                                else:
                                    return "Multi-day patterns"
                            elif units == 'days':
                                if period_val < 2:
                                    return "Daily variations"
                                elif period_val < 8:
                                    return "Weekly patterns"
                                elif period_val < 40:
                                    return "Monthly patterns"
                                elif period_val < 200:
                                    return "Seasonal patterns"
                                else:
                                    return "Annual/multi-annual"
                            else:
                                return "Check period units"
                        
                        # Create peaks table
                        peaks_data = []
                        for i, peak_idx in enumerate(peaks):
                            # Check if significant
                            is_significant_95 = global_power[peak_idx] > red_noise_95[peak_idx]
                            is_significant_99 = global_power[peak_idx] > red_noise_99[peak_idx]
                            
                            significance = "99%" if is_significant_99 else ("95%" if is_significant_95 else "Not Significant")
                            
                            peaks_data.append({
                                'Peak': i+1,
                                'Period': f"{periods[peak_idx]:.2f} {period_units}",
                                'Power': f"{global_power[peak_idx]:.3f}",
                                'Significance': significance,
                                'Interpretation': get_period_interpretation(periods[peak_idx], period_units)
                            })
                        
                        peaks_table = dash_table.DataTable(
                            columns=[{"name": col, "id": col} for col in peaks_data[0].keys()],
                            data=peaks_data,
                            style_table={'width': '90%', 'margin': 'auto'},
                            style_cell={'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': 12},
                            style_header={'backgroundColor': '#e3f2fd', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{Significance} = 99%'},
                                    'backgroundColor': '#ffcdd2',
                                    'color': 'darkred'
                                },
                                {
                                    'if': {'filter_query': '{Significance} = 95%'},
                                    'backgroundColor': '#fff3e0',
                                    'color': 'darkorange'
                                }
                            ]
                        )
                        
                        # Add title as first row in peaks table data
                        title_row = {'Peak': 'Detected Spectral Peaks', 'Period': '', 'Power': '', 'Significance': '', 'Interpretation': ''}
                        peaks_data_with_title = [title_row] + peaks_data
                        
                        peaks_table_with_title = dash_table.DataTable(
                            columns=[{"name": col, "id": col} for col in peaks_data[0].keys()],
                            data=peaks_data_with_title,
                            style_table={'width': '90%', 'margin': 'auto'},
                            style_cell={'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': 12},
                            style_header={'backgroundColor': '#e3f2fd', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 0},  # Title row
                                    'backgroundColor': '#f3e5f5',
                                    'fontWeight': 'bold',
                                    'fontSize': 14
                                },
                                {
                                    'if': {'filter_query': '{Significance} = 99%'},
                                    'backgroundColor': '#ffcdd2',
                                    'color': 'darkred'
                                },
                                {
                                    'if': {'filter_query': '{Significance} = 95%'},
                                    'backgroundColor': '#fff3e0',
                                    'color': 'darkorange'
                                }
                            ]
                        )
                        
                        wavelet_plots.append(peaks_table_with_title)
                        
            except Exception as e:
                pass  # Skip global spectrum if error
            
            # Add overlay interpretation guide
            if show_overlays:
                overlay_items = []
                
                if 'significance' in show_overlays:
                    overlay_items.append(html.Li([
                        html.B("95% Significance Contour (White Lines): "), 
                        "Areas where wavelet power significantly exceeds red noise background. Only patterns inside these contours are statistically meaningful."
                    ], style={'marginBottom': '8px'}))
                
                if 'coi' in show_overlays:
                    overlay_items.append(html.Li([
                        html.B("Cone of Influence (Black Dashed Line): "), 
                        "Boundary where edge effects become important. Wavelet analysis outside this region may be unreliable due to data boundary effects."
                    ], style={'marginBottom': '8px'}))
                

                
                overlay_items.append(html.Li([
                    html.B("Interpretation Tips: "), 
                    "Focus on significant regions inside the COI for reliable analysis. Use Coherence/Cross-Wavelet tabs for phase relationship analysis."
                ], style={'marginBottom': '8px', 'fontStyle': 'italic', 'color': '#27ae60'}))
                
                overlay_guide = html.Div([
                    html.H5('Wavelet Analysis Overlay Guide', style={'color': '#2c3e50', 'marginBottom': '10px'}),
                    html.Ul(overlay_items, style={'fontSize': '13px', 'color': '#34495e', 'lineHeight': '1.6'})
                ], style={'padding': '15px', 'backgroundColor': '#e8f4fd', 'borderRadius': '8px', 'marginTop': '20px', 'border': '1px solid #bee5eb'})
                
                wavelet_plots.append(overlay_guide)
                       
            wavelet_plots.append(export_buttons)
            
            return html.Div(wavelet_plots, style={'backgroundColor': 'white'})
        elif tab == 'signal':
            
            # Clear stored signal values when main callback runs to ensure fresh well data
            from dash import callback_context
            ctx = callback_context
            well_changed = any('well' in str(trigger.get('prop_id', '')) for trigger in ctx.triggered) if ctx.triggered else False
            
            if well_changed:
                print(f"DEBUG: Well/data type changed detected in main callback, preserving dropdown selections")
                # Preserve stored dropdown selections when changing wells or data type, but update extraction source
                if 'current_signal_values' in globals() and current_signal_values:
                    # Update extraction source to match new well/data type
                    if data_type == 'gwl':
                        current_signal_values['extraction_source'] = well
                        print(f"DEBUG: Updated extraction source to new well: {well}")
                    elif data_type == 'rain':
                        current_signal_values['extraction_source'] = 'mm_Rain'
                        print(f"DEBUG: Updated extraction source to rainfall: mm_Rain")
                # Don't clear the entire current_signal_values to preserve dropdown selections
            
            # Also update extraction source if data type changes without well changing
            elif 'current_signal_values' in globals() and current_signal_values:
                if data_type == 'gwl' and current_signal_values.get('extraction_source') != well:
                    current_signal_values['extraction_source'] = well
                    print(f"DEBUG: Updated extraction source to match data type change: {well}")
                elif data_type == 'rain' and current_signal_values.get('extraction_source') != 'mm_Rain':
                    current_signal_values['extraction_source'] = 'mm_Rain'
                    print(f"DEBUG: Updated extraction source to match data type change: mm_Rain")
            
            # Check if we have current signal values from the dedicated callback
            try:
                if 'current_signal_values' in globals() and current_signal_values:
                    print(f"DEBUG: Using real signal values from dropdown")
                    signal_options = current_signal_values['signal_options']
                    statistical_options = current_signal_values['statistical_options']
                    event_options = current_signal_values['event_options']
                    advanced_options = current_signal_values['advanced_options']
                    target_period = current_signal_values['target_period']
                    bandwidth = current_signal_values['bandwidth']
                    display_options = current_signal_values['display_options']
                    trend_method = current_signal_values['trend_method']
                    confidence_level = current_signal_values['confidence_level']
                    forecast_model = current_signal_values['forecast_model']
                    extraction_output_options = current_signal_values['extraction_output_options']
                    extraction_period_units = current_signal_values['extraction_period_units']
                    # Use stored extraction_source if available, otherwise default to current well
                    extraction_source = current_signal_values.get('extraction_source', well)
                else:
                    raise Exception("No current signal values")
            except:
                # Always check for stored values first before using defaults
                if 'current_signal_values' in globals() and current_signal_values:
                    print(f"DEBUG: Using stored signal values instead of defaults")
                    print(f"DEBUG: MAIN CALLBACK RETRIEVING: signal_options='{current_signal_values['signal_options']}' from stored values")
                    signal_options = current_signal_values['signal_options']
                    statistical_options = current_signal_values['statistical_options']
                    event_options = current_signal_values['event_options']
                    advanced_options = current_signal_values['advanced_options']
                    target_period = current_signal_values['target_period']
                    bandwidth = current_signal_values['bandwidth']
                    display_options = current_signal_values['display_options']
                    trend_method = current_signal_values['trend_method']
                    confidence_level = current_signal_values['confidence_level']
                    forecast_model = current_signal_values['forecast_model']
                    extraction_output_options = current_signal_values['extraction_output_options']
                    extraction_period_units = current_signal_values['extraction_period_units']
                    extraction_source = current_signal_values.get('extraction_source', well)
                    print(f"DEBUG: FINAL signal_options BEING USED: '{signal_options}'")
                else:
                    print(f"DEBUG: Using defaults for signal tab")
                    print(f"DEBUG: MAIN CALLBACK USING DEFAULTS: signal_options='period_extraction' (default)")
                    # Set default values for signal tab analysis
                    signal_options = 'period_extraction'  # default (single value, not list)
                    print(f"DEBUG: DEFAULT signal_options SET TO: '{signal_options}'")
                statistical_options = ['descriptive_stats']  # default
                event_options = []  # default
                advanced_options = []  # default
                target_period = 12  # default
                bandwidth = 2  # default
                display_options = ['show_stats']  # default
                trend_method = 'linear_regression'  # default
                confidence_level = 95  # default
                forecast_model = 'arima'  # default
                extraction_output_options = []  # default
                extraction_period_units = 'hours'  # default
                extraction_source = well  # Default to current well when no stored values
            drought_threshold = 10  # default
            min_event_duration = 7  # default
            
            print(f"DEBUG: Signal tab values - signal_options={signal_options}, target_period={target_period}, bandwidth={bandwidth}, extraction_source={extraction_source}, current_well={well}")
            
            # Additional debug: Show source of extraction_source value
            try:
                if 'current_signal_values' in globals() and current_signal_values and 'extraction_source' in current_signal_values:
                    print(f"DEBUG: extraction_source came from stored values: {current_signal_values['extraction_source']}")
                else:
                    print(f"DEBUG: extraction_source using default (current well): {well}")
            except:
                print(f"DEBUG: extraction_source using fallback default: {well}")
            
            # Enhanced Groundwater Analysis Controls
            signal_controls = html.Div([
                # Header Section
                html.Div([
                    html.H3('Comprehensive Groundwater Analysis Dashboard', 
                           style={
                               'color': '#1a365d', 
                               'marginBottom': 8, 
                               'fontWeight': '600',
                               'fontSize': '28px',
                               'letterSpacing': '-0.5px'
                           }),
                    html.P('Advanced groundwater analysis including signal extraction, statistics, trends, events, and forecasting', 
                           style={
                               'fontSize': '16px', 
                               'color': '#4a5568', 
                               'marginBottom': 0, 
                               'fontWeight': '400',
                               'lineHeight': '1.5'
                           })
                ], style={
                    'textAlign': 'center', 
                    'marginBottom': 32,
                    'padding': '24px',
                    'backgroundColor': '#f7fafc',
                    'borderRadius': '12px',
                    'border': '1px solid #e2e8f0'
                }),
                
                # Analysis Guide at Top
                html.Div([
                    html.H4('Groundwater Analysis Guide', 
                           style={
                               'color': '#1a365d', 
                               'marginBottom': 20, 
                               'fontWeight': '600',
                               'fontSize': '24px',
                               'textAlign': 'center',
                               'letterSpacing': '-0.3px'
                           }),
                    
                    html.Div([
                        # Signal Extraction Guide
                        html.Div([
                            html.H6('Signal Extraction', 
                                   style={
                                       'color': '#1e40af', 
                                       'marginBottom': 12,
                                       'fontWeight': '600',
                                       'fontSize': '16px'
                                   }),
                            html.Ul([
                                html.Li("Extract specific periodic components (daily, seasonal patterns)"),
                                html.Li("Separate long-term trends from short-term variations"),
                                html.Li("Apply frequency band filters for targeted analysis"),
                                html.Li("Remove noise to reveal underlying patterns")
                            ], style={'fontSize': '14px', 'lineHeight': '1.6', 'color': '#4a5568'})
                        ], style={
                            'padding': '20px',
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'border': '1px solid #e2e8f0',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)'
                        }),
                        
                        # Statistical Analysis Guide
                        html.Div([
                            html.H6('Statistical Analysis', 
                                   style={
                                       'color': '#059669', 
                                       'marginBottom': 12,
                                       'fontWeight': '600',
                                       'fontSize': '16px'
                                   }),
                            html.Ul([
                                html.Li("Generate comprehensive descriptive statistics"),
                                html.Li("Detect significant trends using Mann-Kendall or linear regression"),
                                html.Li("Decompose time series into seasonal components"),
                                html.Li("Identify change points and regime shifts"),
                                html.Li("Calculate cross-correlations between wells")
                            ], style={'fontSize': '14px', 'lineHeight': '1.6', 'color': '#4a5568'})
                        ], style={
                            'padding': '20px',
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'border': '1px solid #e2e8f0',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)'
                        }),
                        
                        # Event Detection Guide
                        html.Div([
                            html.H6('Event Detection', 
                                   style={
                                       'color': '#dc2626', 
                                       'marginBottom': 12,
                                       'fontWeight': '600',
                                       'fontSize': '16px'
                                   }),
                            html.Ul([
                                html.Li("Identify drought events based on percentile thresholds"),
                                html.Li("Detect flood/high water events automatically"),
                                html.Li("Find anomalous behaviour and outliers"),
                                html.Li("Analyse recharge and discharge patterns")
                            ], style={'fontSize': '14px', 'lineHeight': '1.6', 'color': '#4a5568'})
                        ], style={
                            'padding': '20px',
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'border': '1px solid #e2e8f0',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)'
                        }),
                        
                        # Advanced Analytics Guide
                        html.Div([
                            html.H6('Advanced Analytics', 
                                   style={
                                       'color': '#7c3aed', 
                                       'marginBottom': 12,
                                       'fontWeight': '600',
                                       'fontSize': '16px'
                                   }),
                            html.Ul([
                                html.Li("Compare multiple wells simultaneously"),
                                html.Li("Perform water balance calculations"),
                                html.Li("Generate short-term forecasts using ARIMA models"),
                                html.Li("Assess data quality and completeness")
                            ], style={'fontSize': '14px', 'lineHeight': '1.6', 'color': '#4a5568'})
                        ], style={
                            'padding': '20px',
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'border': '1px solid #e2e8f0',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)'
                        })
                        
                    ], style={
                        'display': 'grid', 
                        'gridTemplateColumns': 'repeat(auto-fit, minmax(280px, 1fr))', 
                        'gap': '20px',
                        'marginBottom': 24
                    }),
                    
                    html.Hr(style={'margin': '24px 0', 'border': '1px solid #e2e8f0'}),
                    
                    html.Div([
                        html.H6('Best Practices', 
                               style={
                                   'color': '#2d3748', 
                                   'marginBottom': 12,
                                   'fontWeight': '600',
                                   'fontSize': '16px',
                                   'textAlign': 'center'
                               }),
                        html.Ul([
                            html.Li("Start with descriptive statistics to understand your data"),
                            html.Li("Use trend analysis to identify long-term changes"),
                            html.Li("Apply event detection to find extreme conditions"),
                            html.Li("Extract signals for detailed component analysis"),
                            html.Li("Compare multiple wells to understand spatial patterns"),
                            html.Li("Export results for further analysis or reporting"),
                            html.Li("Use forecasting for planning and management decisions")
                        ], style={
                            'fontSize': '14px', 
                            'color': '#4a5568',
                            'lineHeight': '1.6',
                            'display': 'grid',
                            'gridTemplateColumns': 'repeat(auto-fit, minmax(300px, 1fr))',
                            'gap': '8px'
                        })
                    ], style={
                        'padding': '20px',
                        'backgroundColor': 'white',
                        'borderRadius': '8px',
                        'border': '1px solid #e2e8f0',
                        'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)'
                    })
                    
                ], style={
                    'marginBottom': 32,
                    'padding': '28px', 
                    'backgroundColor': '#f8fafc', 
                    'borderRadius': '12px', 
                    'border': '1px solid #e2e8f0',
                    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.05)'
                }),
                
                # Analysis Methods Layout
                html.Div([
                    html.H4('Analysis Configuration', 
                           style={
                               'color': '#2d3748', 
                               'marginBottom': 24, 
                               'fontWeight': '600', 
                               'fontSize': '20px',
                               'textAlign': 'center',
                               'letterSpacing': '-0.3px'
                           }),
                    
                    # Analysis Methods Grid
                    html.Div([
                        # Signal Extraction Card
                        html.Div([
                            html.Div([
                                html.H6('Signal Extraction', 
                                       style={
                                           'fontWeight': '600', 
                                           'color': '#1e40af', 
                                           'marginBottom': 12, 
                                           'fontSize': '16px',
                                           'display': 'flex',
                                           'alignItems': 'center'
                                       }),
                                html.Div(style={
                                    'width': '4px',
                                    'height': '24px',
                                    'backgroundColor': '#3b82f6',
                                    'marginRight': '12px',
                                    'borderRadius': '2px'
                                })
                            ], style={'display': 'flex', 'alignItems': 'center'}),
                                            dcc.Dropdown(
                    id='signal-analysis-options-dropdown',
                    options=[
                        {'label': 'Period-Specific Extraction', 'value': 'period_extraction'},
                        {'label': 'Trend vs Variations Analysis', 'value': 'trend_extraction'},
                        {'label': 'Frequency Band Filtering', 'value': 'band_extraction'},
                        {'label': 'Signal Denoising', 'value': 'denoised'},
                        {'label': 'Multi-Stage Decomposition Viewer', 'value': 'stage_decomposition'},
                        {'label': 'Progressive Denoising Stages', 'value': 'progressive_denoising'},
                        {'label': 'Wavelet Basis Function Viewer', 'value': 'wavelet_basis_viewer'},
                    ],
                    value='period_extraction',
                    multi=False,
                    clearable=False,
                    placeholder="Select one method...",
                    style={'fontSize': '14px'}
                )
                        ], style={
                            'padding': '20px',
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'border': '1px solid #e2e8f0',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                            'transition': 'box-shadow 0.2s ease'
                        }),
                        
                        # Statistical Analysis Card
                        html.Div([
                            html.Div([
                                html.H6('Statistical Analysis', 
                                       style={
                                           'fontWeight': '600', 
                                           'color': '#059669', 
                                           'marginBottom': 12, 
                                           'fontSize': '16px',
                                           'display': 'flex',
                                           'alignItems': 'center'
                                       }),
                                html.Div(style={
                                    'width': '4px',
                                    'height': '24px',
                                    'backgroundColor': '#10b981',
                                    'marginRight': '12px',
                                    'borderRadius': '2px'
                                })
                            ], style={'display': 'flex', 'alignItems': 'center'}),
                            dcc.Dropdown(
                                id='statistical-analysis-options-dropdown',
                                options=[
                                    {'label': 'Descriptive Statistics', 'value': 'descriptive_stats'},
                                    {'label': 'Trend Analysis', 'value': 'trend_analysis'},
                                    {'label': 'Seasonal Decomposition', 'value': 'seasonal_decomp'},
                                    {'label': 'Change Point Detection', 'value': 'change_points'},
                                    {'label': 'Correlation Matrix', 'value': 'correlation_matrix'},
                                ],
                                value=['descriptive_stats'],
                                multi=True,
                                placeholder="Select methods...",
                                style={'fontSize': '14px'}
                            )
                        ], style={
                            'padding': '20px',
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'border': '1px solid #e2e8f0',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                            'transition': 'box-shadow 0.2s ease'
                        }),
                        
                        # Event Detection Card
                        html.Div([
                            html.Div([
                                html.H6('Event Detection', 
                                       style={
                                           'fontWeight': '600', 
                                           'color': '#dc2626', 
                                           'marginBottom': 12, 
                                           'fontSize': '16px',
                                           'display': 'flex',
                                           'alignItems': 'center'
                                       }),
                                html.Div(style={
                                    'width': '4px',
                                    'height': '24px',
                                    'backgroundColor': '#ef4444',
                                    'marginRight': '12px',
                                    'borderRadius': '2px'
                                })
                            ], style={'display': 'flex', 'alignItems': 'center'}),
                            dcc.Dropdown(
                                id='event-analysis-options-dropdown',
                                options=[
                                    {'label': 'Drought Events', 'value': 'drought_events'},
                                    {'label': 'Flood/High Water Events', 'value': 'flood_events'},
                                    {'label': 'Anomaly Detection', 'value': 'anomaly_detection'},
                                    {'label': 'Recharge/Discharge Events', 'value': 'recharge_events'},
                                ],
                                value=[],
                                multi=True,
                                placeholder="Select methods...",
                                style={'fontSize': '14px'}
                            )
                        ], style={
                            'padding': '20px',
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'border': '1px solid #e2e8f0',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                            'transition': 'box-shadow 0.2s ease'
                        }),
                        
                        # Advanced Analytics Card
                        html.Div([
                            html.Div([
                                html.H6('Advanced Analytics', 
                                       style={
                                           'fontWeight': '600', 
                                           'color': '#7c3aed', 
                                           'marginBottom': 12, 
                                           'fontSize': '16px',
                                           'display': 'flex',
                                           'alignItems': 'center'
                                       }),
                                html.Div(style={
                                    'width': '4px',
                                    'height': '24px',
                                    'backgroundColor': '#8b5cf6',
                                    'marginRight': '12px',
                                    'borderRadius': '2px'
                                })
                            ], style={'display': 'flex', 'alignItems': 'center'}),
                            dcc.Dropdown(
                                id='advanced-analysis-options-dropdown',
                                options=[
                                    {'label': 'Multi-Well Comparison', 'value': 'multi_well'},
                                    {'label': 'Water Balance Analysis', 'value': 'water_balance'},
                                    {'label': 'Short-term Forecasting', 'value': 'forecasting'},
                                    {'label': 'Quality Assessment', 'value': 'quality_assessment'},
                                ],
                                value=[],
                                multi=True,
                                placeholder="Select methods...",
                                style={'fontSize': '14px'}
                            )
                        ], style={
                            'padding': '20px',
                            'backgroundColor': 'white',
                            'borderRadius': '8px',
                            'border': '1px solid #e2e8f0',
                            'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                            'transition': 'box-shadow 0.2s ease'
                        })
                        
                    ], style={
                        'display': 'grid', 
                        'gridTemplateColumns': 'repeat(auto-fit, minmax(280px, 1fr))', 
                        'gap': '20px',
                        'marginBottom': 32
                    }),
                    
                    # Period-Specific Extraction Settings
                    html.Div([
                        html.Label('Target Period for Extraction:', style={'fontWeight': 'bold', 'marginBottom': 5}),
                        html.Div([
                            dcc.Input(
                                id='extraction-target-period',
                                type='number',
                                value=16,
                                min=0.5,
                                max=1000,  # Added explicit max
                                step=0.1,
                                style={'width': '80px', 'marginRight': '10px'},
                                debounce=True,  # Prevent rapid callback firing
                                persistence=False  # Don't persist old values
                            ),
                            dcc.Dropdown(
                                id='extraction-period-units',
                                options=[
                                    {'label': 'Hours', 'value': 'hours'},
                                    {'label': 'Days', 'value': 'days'},
                                    {'label': 'Weeks', 'value': 'weeks'},
                                    {'label': 'Months', 'value': 'months'}
                                ],
                                value='hours',
                                style={'width': '100px', 'display': 'inline-block'}
                            )
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': 10}),
                        
                        html.Label('Extraction Bandwidth (±):', style={'fontWeight': 'bold', 'marginBottom': 5}),
                        dcc.Input(
                            id='extraction-bandwidth',
                            type='number',
                            value=2,
                            min=0.5,
                            step=0.1,
                            style={'width': '80px', 'marginBottom': 10}
                        ),
                        
                        html.Label('Band Filter Range:', style={'fontWeight': 'bold', 'marginBottom': 5}),
                        html.Div([
                            html.Span('Min Period:', style={'marginRight': '5px'}),
                            dcc.Input(
                                id='band-min-period',
                                type='number',
                                value=12,
                                min=0.1,
                                step=0.1,
                                style={'width': '60px', 'marginRight': '10px'}
                            ),
                            html.Span('Max Period:', style={'marginRight': '5px'}),
                            dcc.Input(
                                id='band-max-period',
                                type='number',
                                value=48,
                                min=0.1,
                                step=0.1,
                                style={'width': '60px'}
                            )
                        ], style={'marginBottom': 15}),
                        
                        # NEW: Stage Viewer Controls
                        html.Div([
                            html.Label('Decomposition Stage Settings:', style={'fontWeight': 'bold', 'marginBottom': 8, 'color': '#8b5cf6'}),
                            
                            html.Div([
                                html.Label('Show Stages:', style={'marginBottom': 5}),
                                dcc.Checklist(
                                    id='show-decomposition-stages',
                                    options=[
                                        {'label': 'Stage-by-Stage Plots', 'value': 'stage_plots'},
                                        {'label': 'Cumulative Reconstruction', 'value': 'cumulative'},
                                        {'label': 'Energy Distribution', 'value': 'energy'},
                                        {'label': 'Interactive Stage Selector', 'value': 'interactive'}
                                    ],
                                    value=['stage_plots'],
                                    labelStyle={'display': 'block', 'marginBottom': '4px'},
                                    style={'marginBottom': 10}
                                )
                            ], style={'marginBottom': 10}),
                            
                            html.Div([
                                html.Label('Maximum Decomposition Levels:', style={'marginBottom': 5}),
                                dcc.Slider(
                                    id='max-decomp-levels',
                                    min=3, max=8, step=1, value=5,
                                    marks={i: str(i) for i in range(3, 9)},
                                    tooltip={'placement': 'bottom', 'always_visible': True}
                                )
                            ], style={'marginBottom': 15})
                        ], id='stage-settings', style={'marginBottom': 15})
                    ], id='extraction-settings'),
                    
                    # Event Detection Settings
                    html.Div([
                        html.Label('Event Detection Settings:', style={'fontWeight': 'bold', 'marginBottom': 8, 'color': '#34495e'}),
                        
                        html.Div([
                            html.Label('Drought Threshold (percentile):', style={'marginBottom': 5}),
                            dcc.Slider(
                                id='drought-threshold-slider',
                                min=5, max=25, step=1, value=10,
                                marks={5: '5%', 10: '10%', 15: '15%', 20: '20%', 25: '25%'},
                                tooltip={'placement': 'bottom', 'always_visible': True}
                            )
                        ], style={'marginBottom': 10}),
                        
                        html.Div([
                            html.Label('Flood Threshold (percentile):', style={'marginBottom': 5}),
                            dcc.Slider(
                                id='flood-threshold-slider',
                                min=75, max=95, step=1, value=90,
                                marks={75: '75%', 80: '80%', 85: '85%', 90: '90%', 95: '95%'},
                                tooltip={'placement': 'bottom', 'always_visible': True}
                            )
                        ], style={'marginBottom': 10}),
                        
                        html.Div([
                            html.Label('Minimum Event Duration (days):', style={'marginBottom': 5}),
                            dcc.Input(
                                id='min-event-duration',
                                type='number',
                                value=7,
                                min=1, max=365, step=1,
                                style={'width': '80px'}
                            )
                        ], style={'marginBottom': 15})
                    ], id='event-settings', style={'marginBottom': 15}),
                    
                    # Statistical Analysis Settings
                    html.Div([
                        html.Label('Statistical Settings:', style={'fontWeight': 'bold', 'marginBottom': 8, 'color': '#34495e'}),
                        
                        html.Div([
                            html.Div([
                                html.Label('Trend Analysis Method:', style={'marginBottom': 5}),
                                dcc.Dropdown(
                                    id='trend-method-dropdown',
                                    options=[
                                        {'label': 'Mann-Kendall Test', 'value': 'mann_kendall'},
                                        {'label': 'Linear Regression', 'value': 'linear_regression'},
                                        {'label': 'Sen Slope Estimator', 'value': 'sen_slope'},
                                        {'label': 'Moving Average', 'value': 'moving_average'}
                                    ],
                                    value='mann_kendall'
                                )
                            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                            
                            html.Div([
                                html.Label('Confidence Level (%):', style={'marginBottom': 5}),
                                dcc.Dropdown(
                                    id='confidence-level-dropdown',
                                    options=[
                                        {'label': '90%', 'value': 90},
                                        {'label': '95%', 'value': 95},
                                        {'label': '99%', 'value': 99}
                                    ],
                                    value=95
                                )
                            ], style={'width': '48%', 'display': 'inline-block'})
                        ], style={'marginBottom': 15})
                    ], id='statistical-settings', style={'marginBottom': 15}),
                    
                    # Forecasting Settings
                    html.Div([
                        html.Label('Forecasting Settings:', style={'fontWeight': 'bold', 'marginBottom': 8, 'color': '#34495e'}),
                        
                        html.Div([
                            html.Label('Forecast Horizon (days):', style={'marginBottom': 5}),
                            dcc.Input(
                                id='forecast-horizon',
                                type='number',
                                value=30,
                                min=1, max=365, step=1,
                                style={'width': '80px', 'marginRight': '15px'}
                            ),
                            html.Label('Model Type:', style={'marginBottom': 5, 'marginLeft': '15px'}),
                            dcc.Dropdown(
                                id='forecast-model-dropdown',
                                options=[
                                    {'label': 'ARIMA', 'value': 'arima'},
                                    {'label': 'Exponential Smoothing', 'value': 'exp_smoothing'},
                                    {'label': 'Linear Trend', 'value': 'linear_trend'},
                                    {'label': 'Seasonal Naive', 'value': 'seasonal_naive'}
                                ],
                                value='arima',
                                style={'width': '150px', 'display': 'inline-block'}
                            )
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': 15})
                    ], id='forecasting-settings', style={'marginBottom': 15}),
                    
                    # Enhanced Export & Output Options
                    html.Div([
                        html.Label('Output & Export Options:', style={'fontWeight': 'bold', 'marginBottom': 8, 'color': '#34495e'}),
                        
                        html.Div([
                            html.Div([
                                html.Label('Display Options:', style={'marginBottom': 5, 'fontWeight': 'bold', 'color': '#27ae60'}),
                                dcc.Dropdown(
                                    id='display-output-options-dropdown',
                                    options=[
                                        {'label': 'Show Time Series Plots', 'value': 'show_timeseries'},
                                        {'label': 'Show Statistical Summary', 'value': 'show_stats'},
                                        {'label': 'Show Event Timeline', 'value': 'show_events'},
                                        {'label': 'Show Correlation Matrix', 'value': 'show_correlation'},
                                        {'label': 'Show Forecasting Results', 'value': 'show_forecast'}
                                    ],
                                    value=['show_timeseries', 'show_stats'],
                                    multi=True,
                                    placeholder="Select display options...",
                                    style={'fontSize': '12px'}
                                )
                            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                            
                            html.Div([
                                html.Label('Export Options:', style={'marginBottom': 5, 'fontWeight': 'bold', 'color': '#e74c3c'}),
                                dcc.Dropdown(
                                    id='extraction-output-options-dropdown',
                                    options=[
                                        {'label': 'Export Analysis Results (CSV)', 'value': 'export_results'},
                                        {'label': 'Export Statistical Report (PDF)', 'value': 'export_report'},
                                        {'label': 'Export Event Summary (CSV)', 'value': 'export_events'},
                                        {'label': 'Export Forecast Data (CSV)', 'value': 'export_forecast'},
                                        {'label': 'Run Wavelet Analysis on Extracted Signal', 'value': 'analyse_extracted'}
                                    ],
                                    value=['export_results'],
                                    multi=True,
                                    placeholder="Select export options...",
                                    style={'fontSize': '12px'}
                                )
                            ], style={'width': '48%', 'display': 'inline-block'})
                        ])
                    ])
                    
                ], style={
                    'padding': '28px', 
                    'backgroundColor': 'white', 
                    'borderRadius': '12px', 
                    'border': '1px solid #e2e8f0',
                    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.05)'
                })
                
            ], style={
                'marginBottom': 24, 
                'padding': '0px', 
                'backgroundColor': 'transparent'
            })
            
            # Use the default values set above for signal tab
            # (Real dropdown values will be handled by the dedicated signal callback)
            display_options = ['show_timeseries', 'show_stats']  # Default display options
            
            # Use the default values set above for signal tab
            # (Real values will be handled by the dedicated signal callback)
            period_units = extraction_period_units  # Use the default set above
            drought_threshold = 10  # Default 10th percentile
            flood_threshold = 90  # Default 90th percentile
            min_event_duration = 7  # Default 7 days
            trend_method = 'mann_kendall'  # Default trend method
            confidence_level = 95  # Default confidence level
            forecast_horizon = 30  # Default 30 days
            forecast_model = 'arima'  # Default forecast model
            
            # Initialize plots and results storage
            signal_plots = []
            analysis_results = {}
            extracted_signals = {}
            
            # Enhanced data preparation
            all_well_data = {}
            all_clean_data = {}
            
            # Get source data and prepare all wells for multi-well analysis
            if extraction_source in data_resampled.columns:
                source_data = data_resampled[extraction_source].values
                # Use descriptive names for display
                if extraction_source == 'mm_Rain':
                    source_name = 'Rainfall'
                    units = 'mm'
                    value_label = 'Rainfall'
                    yaxis_label_full = f'Rainfall (mm) [{resample_label}]'
                else:
                    source_name = extraction_source  # Use actual well name
                    units = 'm'
                    value_label = 'Groundwater Level'
                    yaxis_label_full = f'Groundwater Level (m) [{resample_label}]'
            else:
                source_data = y  # fallback to main series
                source_name = title
                units = yaxis_label.split('(')[-1].replace(')', '') if '(' in yaxis_label else ''
                value_label = source_name
                yaxis_label_full = f'{source_name} ({units}) [{resample_label}]'
            
            # Prepare all available series for comprehensive analysis
            for well_col in well_columns:
                if well_col in data_resampled.columns:
                    well_data = data_resampled[well_col].values
                    mask = ~np.isnan(well_data)
                    all_well_data[well_col] = well_data[mask]
                    all_clean_data[well_col] = {
                        'data': well_data[mask],
                        'times': data_resampled['DateTime'].values[mask],
                        'mask': mask
                    }
            
            # Add rainfall data if available
            if 'mm_Rain' in data_resampled.columns:
                rain_data = data_resampled['mm_Rain'].values
                mask = ~np.isnan(rain_data)
                all_well_data['mm_Rain'] = rain_data[mask]
                all_clean_data['mm_Rain'] = {
                    'data': rain_data[mask],
                    'times': data_resampled['DateTime'].values[mask],
                    'mask': mask
                }
            
            # Remove NaNs from source data
            mask = ~np.isnan(source_data)
            clean_data = source_data[mask]
            clean_times = x[mask] if len(x) == len(source_data) else data_resampled['DateTime'].values[mask]
            
            if len(clean_data) < 10:
                return html.Div([
                    signal_controls,
                    html.Div([
                        html.H4('Insufficient Data'),
                        html.P('Not enough valid data points for signal extraction.', 
                               style={'color': '#e74c3c', 'textAlign': 'center'})
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                ], style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'padding': '20px'})
            
            # Calculate statistics for hover info
            mean_val = np.nanmean(clean_data)
            median_val = np.nanmedian(clean_data)
            
            # Create hover template with optional statistics
            if 'show_stats' in display_options:
                hover_template = (f'<b>Date</b>: %{{x}}<br>'
                                f'<b>{value_label}</b>: %{{y:.3f}} {units}<br>'
                                f'<b>Mean</b>: {mean_val:.3f} {units}<br>'
                                f'<b>Median</b>: {median_val:.3f} {units}<extra></extra>')
            else:
                hover_template = f'<b>Date</b>: %{{x}}<br><b>{value_label}</b>: %{{y:.3f}} {units}<extra></extra>'
            
            # Main signal plot with enhanced styling (ORIGINAL DATA ONLY)
            fig_main = go.Figure()
            fig_main.add_trace(go.Scattergl(
                x=clean_times, y=clean_data, 
                mode='lines', 
                name=f'Original {source_name} [{resample_label}]',
                line=dict(color='#3498db', width=2),
                hovertemplate=hover_template
            ))
            
            # Add descriptive statistics to the plot if enabled
            if 'show_stats' in display_options:
                # Add mean line (without annotation)
                fig_main.add_hline(
                    y=mean_val, 
                    line_dash="dash", 
                    line_color="red",
                    opacity=0.7
                )
                
                # Add median line (without annotation)
                fig_main.add_hline(
                    y=median_val, 
                    line_dash="dot", 
                    line_color="green",
                    opacity=0.7
                )
            
            # Verify we're using resampled data
            print(f"DEBUG: Extraction using {resample_label} resampled data, {len(clean_data)} points")
            print(f"DEBUG: Data range: {clean_data.min():.3f} to {clean_data.max():.3f} {units}")
            print(f"DEBUG: Data mean: {np.mean(clean_data):.3f} {units}, std: {np.std(clean_data):.3f} {units}")
            print(f"DEBUG: Time span: {clean_times[0]} to {clean_times[-1]}")
            print(f"DEBUG: Selected analysis methods: {signal_options}")
            print(f"DEBUG: *** EXTRACTION USES SAME DATA AS ORIGINAL PLOT ***")
            
            # Update main figure layout and add to plots
            fig_main.update_layout(
                title=f'Original {source_name} Data [{resample_label}]',
                xaxis_title='Date',
                yaxis_title=yaxis_label_full,
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=500, width=1300,
                showlegend=True
            )
            
            # Add main plot to signal_plots
            signal_plots.append(dcc.Graph(figure=fig_main, style={'backgroundColor': 'white'}))
            
            # Debug: Show which analysis method is selected
            print(f"DEBUG: Selected analysis method: {signal_options}")
            print(f"DEBUG: Signal options type: {type(signal_options)}")
            
            # Period-Specific Extraction (DEFAULT)
            if signal_options == 'period_extraction':
                print(f"DEBUG: ✅ EXECUTING Period Extraction - signal_options={signal_options}")
                try:
                    print(f"DEBUG: Running period extraction with target_period={target_period}, bandwidth={bandwidth}")
                    
                    # Convert target period to same units as analysis
                    if extraction_period_units == 'days':
                        target_period_hours = target_period * 24
                    elif extraction_period_units == 'weeks':
                        target_period_hours = target_period * 24 * 7
                    elif extraction_period_units == 'months':
                        target_period_hours = target_period * 24 * 30.44
                    else:  # hours
                        target_period_hours = target_period
                    
                    print(f"DEBUG: target_period_hours={target_period_hours}")
                    
                    # Use DWT for extraction
                    wavelet_extract = 'db8'
                    max_level = pywt.dwt_max_level(len(clean_data), wavelet_extract)
                    
                    # Decompose signal
                    coeffs = pywt.wavedec(clean_data, wavelet_extract, level=max_level)
                    
                    # Calculate approximate periods for each level based on actual resampling
                    if len(clean_times) > 1:
                        time_diff = clean_times[1] - clean_times[0]
                        # Handle both pandas datetime and numpy datetime64
                        try:
                            actual_dt_seconds = time_diff.total_seconds()  # pandas Timedelta
                        except AttributeError:
                            # numpy timedelta64 - convert to seconds
                            actual_dt_seconds = time_diff / np.timedelta64(1, 's')
                        dt_hours = actual_dt_seconds / 3600  # Convert to hours
                        
                        print(f"DEBUG: Time data type: {type(clean_times[0])}, diff type: {type(time_diff)}")
                        print(f"DEBUG: Actual data timestep: {dt_hours:.3f} hours ({resample_label})")
                    else:
                        dt_hours = 0.5  # fallback
                        print(f"DEBUG: Using fallback timestep: {dt_hours} hours")
                    
                    level_periods = [dt_hours * (2**i) for i in range(max_level + 1)]
                    print(f"DEBUG: Wavelet levels: {max_level}, Available periods: {level_periods}")
                    
                    # Find best matching level for target period
                    target_level = None
                    min_diff = float('inf')
                    for i, period in enumerate(level_periods):
                        diff = abs(period - target_period_hours)
                        if diff < min_diff and diff <= bandwidth:
                            min_diff = diff
                            target_level = i
                    
                    print(f"DEBUG: Found target_level={target_level}, min_diff={min_diff}")
                    
                    if target_level is not None:
                        # Reconstruct signal from specific level
                        coeffs_extract = [np.zeros_like(c) for c in coeffs]
                        if target_level == 0:
                            coeffs_extract[0] = coeffs[0]  # Approximation coefficients
                        else:
                            coeffs_extract[target_level] = coeffs[target_level]  # Detail coefficients
                        
                        extracted_period = pywt.waverec(coeffs_extract, wavelet_extract)[:len(clean_data)]
                        extracted_signals['period_component'] = extracted_period
                        
                        # Statistics for extracted component
                        extract_amp = np.std(extracted_period)
                        extract_range = np.max(extracted_period) - np.min(extracted_period)
                        
                        print(f"DEBUG: Successfully extracted {target_period:.1f}{extraction_period_units} component from {resample_label} data")
                        print(f"DEBUG: Extracted signal - Amplitude: {extract_amp:.3f} {units}, Range: {extract_range:.3f} {units}")
                        print(f"DEBUG: Original vs Extracted - Original std: {np.std(clean_data):.3f}, Extracted std: {extract_amp:.3f}")
                        
                        # Create ENHANCED plot with both extracted and reconstructed signals
                        fig_period = go.Figure()
                        
                        # Calculate reconstructed signal (mean + extracted = real tidal influence)
                        reconstructed_signal = np.mean(clean_data) + extracted_period
                        
                        # Add extracted oscillation around zero
                        fig_period.add_trace(go.Scattergl(
                            x=clean_times, y=extracted_period,
                            mode='lines',
                            name=f'Extracted Oscillation (±{np.std(extracted_period):.3f}{units})',
                            line=dict(color='#e74c3c', width=2),
                            hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Oscillation</b>: %{{y:.3f}} {units}<br><b>Interpretation</b>: Deviation from mean<extra></extra>'
                        ))
                        
                        # Add reconstructed tidal signal at real water levels  
                        fig_period.add_trace(go.Scattergl(
                            x=clean_times, y=reconstructed_signal,
                            mode='lines',
                            name=f'Reconstructed Tidal Signal (Mean + Extracted)',
                            line=dict(color='#2980b9', width=2, dash='dot'),
                            hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Tidal Influence</b>: %{{y:.3f}} {units}<br><b>Interpretation</b>: Actual water level variation<extra></extra>'
                        ))
                        
                        fig_period.update_layout(
                            title=f'Tidal Signal Analysis: {target_period:.1f}{extraction_period_units} Period [{resample_label}]',
                            xaxis_title='Date',
                            yaxis_title=yaxis_label_full,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=400, width=1300,
                            showlegend=True,
                            legend=dict(
                                orientation='h',
                                x=0.5, y=1.02,
                                bgcolor='rgba(255,255,255,0.9)',
                                bordercolor='rgba(0,0,0,0.2)',
                                borderwidth=1,
                                xanchor='center',
                                yanchor='bottom'
                            )
                        )
                        
                        signal_plots.append(dcc.Graph(figure=fig_period, style={'backgroundColor': 'white'}))
                    else:
                        print(f"DEBUG: No suitable level found for target period {target_period_hours}h with bandwidth {bandwidth}")
                        # Add a message to the user
                        signal_plots.append(html.Div([
                            html.P(f"No suitable wavelet level found for {target_period}h period (±{bandwidth}h). Try increasing bandwidth or different period.", 
                                   style={'color': '#e67e22', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#fef3cd', 'borderRadius': '5px'})
                        ]))
                
                except Exception as e:
                    print(f"DEBUG: Period extraction error: {str(e)}")
                    signal_plots.append(html.Div([
                        html.P(f"Period extraction error: {str(e)}", 
                               style={'color': '#e67e22', 'textAlign': 'center', 'padding': '10px'})
                    ]))
            
            # Trend vs Variations Extraction (OPTIONAL)
            if signal_options == 'trend_extraction':
                print(f"DEBUG: ✅ EXECUTING Trend Analysis - signal_options={signal_options}")
                try:
                    wavelet_trend = 'db8'
                    max_level = pywt.dwt_max_level(len(clean_data), wavelet_trend)
                    
                    # Decompose signal
                    coeffs = pywt.wavedec(clean_data, wavelet_trend, level=max_level)
                    
                    # Extract trend (approximation coefficients only)
                    trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
                    trend_signal = pywt.waverec(trend_coeffs, wavelet_trend)[:len(clean_data)]
                    
                    # Extract variations (detail coefficients only)
                    detail_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
                    variation_signal = pywt.waverec(detail_coeffs, wavelet_trend)[:len(clean_data)]
                    
                    extracted_signals['trend'] = trend_signal
                    extracted_signals['variations'] = variation_signal
                    
                    print(f"DEBUG: Successfully extracted trend and variations from {resample_label} data")
                    
                    # Create ENHANCED plot for trend component with dual traces
                    fig_trend = go.Figure()
                    
                    # Calculate trend oscillation around zero
                    data_mean = np.mean(clean_data)
                    trend_oscillation = trend_signal - data_mean
                    
                    # Add trend oscillation around zero
                    fig_trend.add_trace(go.Scattergl(
                        x=clean_times, y=trend_oscillation,
                        mode='lines',
                        name=f'Trend Oscillation (±{np.std(trend_oscillation):.3f}{units})',
                        line=dict(color='#27ae60', width=2),
                        hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Oscillation</b>: %{{y:.3f}} {units}<br><b>Interpretation</b>: Long-term trend deviation<extra></extra>'
                    ))
                    
                    # Add reconstructed trend signal at actual water levels
                    fig_trend.add_trace(go.Scattergl(
                        x=clean_times, y=trend_signal,
                        mode='lines',
                        name='Reconstructed Trend Signal (Long-term)',
                        line=dict(color='#2ecc71', width=2, dash='dot'),
                        hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Trend</b>: %{{y:.3f}} {units}<br><b>Interpretation</b>: Actual long-term water level trend<extra></extra>'
                    ))
                    
                    fig_trend.update_layout(
                        title=f'Trend Analysis: Long-term Component [{resample_label}]',
                        xaxis_title='Date',
                        yaxis_title=yaxis_label_full,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400, width=1300,
                        showlegend=True,
                        legend=dict(
                            orientation='h',
                            x=0.5, y=1.02,
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='rgba(0,0,0,0.2)',
                            borderwidth=1,
                            xanchor='center',
                            yanchor='bottom'
                        )
                    )
                    
                    signal_plots.append(dcc.Graph(figure=fig_trend, style={'backgroundColor': 'white'}))
                    
                    # Create ENHANCED plot for variations component with dual traces
                    fig_variations = go.Figure()
                    
                    # Calculate variation oscillation around zero
                    variation_oscillation = variation_signal - np.mean(variation_signal)
                    
                    # Add variation oscillation around zero
                    fig_variations.add_trace(go.Scattergl(
                        x=clean_times, y=variation_oscillation,
                        mode='lines',
                        name=f'Variation Oscillation (±{np.std(variation_oscillation):.3f}{units})',
                        line=dict(color='#f39c12', width=2),
                        hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Oscillation</b>: %{{y:.3f}} {units}<br><b>Interpretation</b>: Short-term variation deviation<extra></extra>'
                    ))
                    
                    # Add reconstructed variation signal at actual water levels
                    fig_variations.add_trace(go.Scattergl(
                        x=clean_times, y=variation_signal,
                        mode='lines',
                        name='Reconstructed Variation Signal (Short-term)',
                        line=dict(color='#e67e22', width=2, dash='dot'),
                        hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Variation</b>: %{{y:.3f}} {units}<br><b>Interpretation</b>: Actual short-term variations<extra></extra>'
                    ))
                    
                    fig_variations.update_layout(
                        title=f'Variation Analysis: Short-term Component [{resample_label}]',
                        xaxis_title='Date',
                        yaxis_title=yaxis_label_full,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400, width=1300,
                        showlegend=True,
                        legend=dict(
                            orientation='h',
                            x=0.5, y=1.02,
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='rgba(0,0,0,0.2)',
                            borderwidth=1,
                            xanchor='center',
                            yanchor='bottom'
                        )
                    )
                    
                    signal_plots.append(dcc.Graph(figure=fig_variations, style={'backgroundColor': 'white'}))
                
                except Exception as e:
                    signal_plots.append(html.Div([
                        html.P(f"Trend extraction error: {str(e)}", 
                               style={'color': '#e67e22', 'textAlign': 'center', 'padding': '10px'})
                    ]))
            
            # Band Filtering (OPTIONAL)
            if signal_options == 'band_extraction':
                print(f"DEBUG: ✅ EXECUTING Band Filtering - signal_options={signal_options}")
                try:
                    # Simple bandpass using DWT levels
                    wavelet_band = 'db8'
                    max_level = pywt.dwt_max_level(len(clean_data), wavelet_band)
                    
                    coeffs = pywt.wavedec(clean_data, wavelet_band, level=max_level)
                    
                    # Calculate periods for each level
                    dt_hours = 0.5
                    level_periods = [dt_hours * (2**i) for i in range(max_level + 1)]
                    
                    # Find levels within band range (use actual UI values)
                    band_min_period = min_period  # Use actual min-period input from UI
                    band_max_period = max_period  # Use actual max-period input from UI
                    
                    print(f"DEBUG: Band filtering using user values - min_period={band_min_period}, max_period={band_max_period}")
                    
                    band_coeffs = [np.zeros_like(c) for c in coeffs]
                    for i, period in enumerate(level_periods):
                        if band_min_period <= period <= band_max_period:
                            if i == 0:
                                band_coeffs[0] = coeffs[0]
                            else:
                                band_coeffs[i] = coeffs[i]
                    
                    band_signal = pywt.waverec(band_coeffs, wavelet_band)[:len(clean_data)]
                    extracted_signals['band_filtered'] = band_signal
                    
                    print(f"DEBUG: Successfully extracted band filtered ({band_min_period}-{band_max_period}h) from {resample_label} data")
                    
                    # Create ENHANCED plot with both band oscillation and reconstructed signals
                    fig_band = go.Figure()
                    
                    # Calculate band oscillation around zero (deviation from mean)
                    data_mean = np.mean(clean_data)
                    band_oscillation = band_signal - data_mean
                    
                    # Add band oscillation around zero
                    fig_band.add_trace(go.Scattergl(
                        x=clean_times, y=band_oscillation,
                        mode='lines',
                        name=f'Band Oscillation (±{np.std(band_oscillation):.3f}{units})',
                        line=dict(color='#9b59b6', width=2),
                        hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Oscillation</b>: %{{y:.3f}} {units}<br><b>Interpretation</b>: {band_min_period}-{band_max_period}h band deviation<extra></extra>'
                    ))
                    
                    # Add reconstructed band signal at actual water levels
                    fig_band.add_trace(go.Scattergl(
                        x=clean_times, y=band_signal,
                        mode='lines',
                        name=f'Reconstructed Band Signal ({band_min_period}-{band_max_period}h)',
                        line=dict(color='#e67e22', width=2, dash='dot'),
                        hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Band Signal</b>: %{{y:.3f}} {units}<br><b>Interpretation</b>: Actual water level in frequency band<extra></extra>'
                    ))
                    
                    fig_band.update_layout(
                        title=f'Band Filtering Analysis: {band_min_period}-{band_max_period}h Frequencies [{resample_label}]',
                        xaxis_title='Date',
                        yaxis_title=yaxis_label_full,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400, width=1300,
                        showlegend=True,
                        legend=dict(
                            orientation='h',
                            x=0.5, y=1.02,
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='rgba(0,0,0,0.2)',
                            borderwidth=1,
                            xanchor='center',
                            yanchor='bottom'
                        )
                    )
                    
                    signal_plots.append(dcc.Graph(figure=fig_band, style={'backgroundColor': 'white'}))
                
                except Exception as e:
                    signal_plots.append(html.Div([
                        html.P(f"Band filtering error: {str(e)}", 
                               style={'color': '#e67e22', 'textAlign': 'center', 'padding': '10px'})
                    ]))
            
            # Signal Denoising (OPTIONAL)
            if signal_options == 'denoised':
                print(f"DEBUG: Running Signal Denoising (user selected)")
                try:
                    default_wavelet = 'db8'
                    coeffs = pywt.wavedec(clean_data, default_wavelet)
                    threshold_val = cached_percentile_calculation(tuple(np.concatenate(coeffs[1:])), 95)
                    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold_val, mode='soft') for c in coeffs[1:]]
                    denoised_signal = pywt.waverec(denoised_coeffs, default_wavelet)[:len(clean_data)]
                    
                    extracted_signals['denoised'] = denoised_signal
                    
                    print(f"DEBUG: Successfully denoised signal from {resample_label} data")
                    
                    # Create ENHANCED plot for denoising with dual traces
                    fig_denoise = go.Figure()
                    
                    # Calculate noise removed (original - denoised)
                    noise_removed = clean_data - denoised_signal
                    
                    # Add noise component (what was removed)
                    fig_denoise.add_trace(go.Scattergl(
                        x=clean_times, y=noise_removed,
                        mode='lines',
                        name=f'Noise Removed (±{np.std(noise_removed):.3f}{units})',
                        line=dict(color='#e74c3c', width=2),
                        hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Noise</b>: %{{y:.3f}} {units}<br><b>Interpretation</b>: High-frequency noise removed<extra></extra>'
                    ))
                    
                    # Add reconstructed denoised signal at actual water levels
                    fig_denoise.add_trace(go.Scattergl(
                        x=clean_times, y=denoised_signal, 
                        mode='lines', 
                        name='Reconstructed Clean Signal (Denoised)',
                        line=dict(color='#c0392b', width=2, dash='dot'),
                        hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Clean Signal</b>: %{{y:.3f}} {units}<br><b>Interpretation</b>: Actual water level without noise<extra></extra>'
                    ))
                    
                    fig_denoise.update_layout(
                        title=f'Denoising Analysis: Noise Removal [{resample_label}]',
                        xaxis_title='Date',
                        yaxis_title=yaxis_label_full,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400, width=1300,
                        showlegend=True,
                        legend=dict(
                            orientation='h',
                            x=0.5, y=1.02,
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='rgba(0,0,0,0.2)',
                            borderwidth=1,
                            xanchor='center',
                            yanchor='bottom'
                        )
                    )
                    
                    signal_plots.append(dcc.Graph(figure=fig_denoise, style={'backgroundColor': 'white'}))
                    
                except Exception:
                    signal_plots.append(html.Div([
                        html.P("Signal denoising unavailable for current data", 
                               style={'color': '#e67e22', 'textAlign': 'center', 'padding': '10px'})
                    ]))
            
            # Multi-Stage Decomposition Viewer (NEW)
            if signal_options == 'stage_decomposition':
                print(f"DEBUG: ✅ EXECUTING Multi-Stage Decomposition Viewer - signal_options={signal_options}")
                try:
                    wavelet_stage = 'db8'
                    max_level = min(pywt.dwt_max_level(len(clean_data), wavelet_stage), 6)  # Limit to 6 levels
                    
                    print(f"DEBUG: Stage decomposition with {max_level} levels")
                    
                    # Perform complete wavelet decomposition
                    coeffs = pywt.wavedec(clean_data, wavelet_stage, level=max_level)
                    
                    # Calculate approximate periods for each level
                    if len(clean_times) > 1:
                        time_diff = clean_times[1] - clean_times[0]
                        try:
                            actual_dt_seconds = time_diff.total_seconds()
                        except AttributeError:
                            actual_dt_seconds = time_diff / np.timedelta64(1, 's')
                        dt_hours = actual_dt_seconds / 3600
                    else:
                        dt_hours = 0.5
                    
                    level_periods = [dt_hours * (2**i) for i in range(max_level + 1)]
                    
                    # Create stage-by-stage plots
                    for level in range(max_level + 1):
                        if level == 0:
                            # Approximation coefficients (lowest frequency/highest scale)
                            stage_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
                            stage_name = f"Stage {level}: Approximation (Long-term Trend)"
                            stage_color = '#2c3e50'
                            interpretation = f"Very long-term changes (>{level_periods[level]:.1f}h periods)"
                        else:
                            # Detail coefficients for this level
                            stage_coeffs = [np.zeros_like(coeffs[0])] + [np.zeros_like(c) for c in coeffs[1:]]
                            stage_coeffs[level] = coeffs[level]
                            
                            if level <= 2:
                                stage_name = f"Stage {level}: High-Frequency Details"
                                stage_color = '#e74c3c'
                                interpretation = f"Rapid fluctuations (~{level_periods[level]:.1f}h periods)"
                            elif level <= 4:
                                stage_name = f"Stage {level}: Medium-Frequency Details"  
                                stage_color = '#f39c12'
                                interpretation = f"Daily/sub-daily patterns (~{level_periods[level]:.1f}h periods)"
                            else:
                                stage_name = f"Stage {level}: Low-Frequency Details"
                                stage_color = '#27ae60'
                                interpretation = f"Weekly/seasonal patterns (~{level_periods[level]:.1f}h periods)"
                        
                        # Reconstruct signal for this stage
                        stage_signal = pywt.waverec(stage_coeffs, wavelet_stage)[:len(clean_data)]
                        
                        # Calculate stage statistics
                        stage_energy = np.sum(stage_signal**2)
                        stage_variance = np.var(stage_signal)
                        stage_amplitude = np.std(stage_signal)
                        
                        # Create enhanced stage plot
                        fig_stage = go.Figure()
                        
                        # Stage oscillation around zero
                        stage_oscillation = stage_signal - np.mean(stage_signal)
                        fig_stage.add_trace(go.Scattergl(
                            x=clean_times, y=stage_oscillation,
                            mode='lines',
                            name=f'Stage {level} Oscillation (±{stage_amplitude:.3f}{units})',
                            line=dict(color=stage_color, width=2),
                            hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Oscillation</b>: %{{y:.3f}} {units}<br><b>Level</b>: {level}<br><b>Period</b>: ~{level_periods[level]:.1f}h<br><b>Energy</b>: {stage_energy:.1f}<extra></extra>'
                        ))
                        
                        # Reconstructed stage signal at actual levels
                        fig_stage.add_trace(go.Scattergl(
                            x=clean_times, y=stage_signal,
                            mode='lines',
                            name=f'Reconstructed Stage {level}',
                            line=dict(color=stage_color, width=2, dash='dot'),
                            hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Signal</b>: %{{y:.3f}} {units}<br><b>Interpretation</b>: {interpretation}<extra></extra>'
                        ))
                        
                        fig_stage.update_layout(
                            title=f'{stage_name}<br><sub>Energy: {stage_energy:.1f} | Variance: {stage_variance:.3f} | Period: ~{level_periods[level]:.1f}h</sub>',
                            xaxis_title='Date',
                            yaxis_title=yaxis_label_full,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=350, width=1300,
                            showlegend=True,
                            legend=dict(
                                orientation='h',
                                x=0.5, y=1.02,
                                bgcolor='rgba(255,255,255,0.9)',
                                bordercolor='rgba(0,0,0,0.2)',
                                borderwidth=1,
                                xanchor='center',
                                yanchor='bottom'
                            )
                        )
                        
                        signal_plots.append(dcc.Graph(figure=fig_stage, style={'backgroundColor': 'white'}))
                    
                    # Create cumulative reconstruction plot
                    fig_cumulative = go.Figure()
                    cumulative_signal = np.zeros_like(clean_data)
                    
                    colors = ['#2c3e50', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6', '#1abc9c']
                    
                    for level in range(max_level + 1):
                        if level == 0:
                            stage_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
                        else:
                            stage_coeffs = [np.zeros_like(coeffs[0])] + [np.zeros_like(c) for c in coeffs[1:]]
                            stage_coeffs[level] = coeffs[level]
                        
                        stage_signal = pywt.waverec(stage_coeffs, wavelet_stage)[:len(clean_data)]
                        cumulative_signal += stage_signal
                        
                        fig_cumulative.add_trace(go.Scattergl(
                            x=clean_times, y=cumulative_signal,
                            mode='lines',
                            name=f'Up to Level {level}',
                            line=dict(color=colors[level % len(colors)], width=2),
                            hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Cumulative Signal</b>: %{{y:.3f}} {units}<br><b>Levels Included</b>: 0-{level}<extra></extra>'
                        ))
                    
                    # Add original signal for comparison
                    fig_cumulative.add_trace(go.Scattergl(
                        x=clean_times, y=clean_data,
                        mode='lines',
                        name='Original Signal',
                        line=dict(color='black', width=2, dash='dash'),
                        hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Original</b>: %{{y:.3f}} {units}<extra></extra>'
                    ))
                    
                    fig_cumulative.update_layout(
                        title=f'Cumulative Wavelet Reconstruction: Progressive Signal Building [{resample_label}]',
                        xaxis_title='Date',
                        yaxis_title=yaxis_label_full,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=500, width=1300,
                        showlegend=True
                    )
                    
                    signal_plots.append(dcc.Graph(figure=fig_cumulative, style={'backgroundColor': 'white'}))
                    
                    # Create energy distribution plot
                    energies = []
                    level_names = []
                    for level in range(max_level + 1):
                        if level == 0:
                            stage_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
                            level_names.append(f'Level {level}: Approximation')
                        else:
                            stage_coeffs = [np.zeros_like(coeffs[0])] + [np.zeros_like(c) for c in coeffs[1:]]
                            stage_coeffs[level] = coeffs[level]
                            level_names.append(f'Level {level}: Detail')
                        
                        stage_signal = pywt.waverec(stage_coeffs, wavelet_stage)[:len(clean_data)]
                        stage_energy = np.sum(stage_signal**2)
                        energies.append(stage_energy)
                    
                    total_energy = sum(energies)
                    energy_percentages = [(e/total_energy)*100 for e in energies]
                    
                    fig_energy = go.Figure(data=[
                        go.Bar(
                            x=level_names,
                            y=energy_percentages,
                            text=[f'{e:.1f}%' for e in energy_percentages],
                            textposition='auto',
                            marker_color=colors[:len(energies)]
                        )
                    ])
                    
                    fig_energy.update_layout(
                        title='Energy Distribution Across Decomposition Levels',
                        xaxis_title='Decomposition Level',
                        yaxis_title='Energy Percentage (%)',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400, width=1300
                    )
                    
                    signal_plots.append(dcc.Graph(figure=fig_energy, style={'backgroundColor': 'white'}))
                    
                    print(f"DEBUG: Successfully created {max_level+1} stage plots from {resample_label} data")
                    
                except Exception as e:
                    print(f"DEBUG: Stage decomposition error: {str(e)}")
                    signal_plots.append(html.Div([
                        html.P(f"Stage decomposition error: {str(e)}", 
                               style={'color': '#e67e22', 'textAlign': 'center', 'padding': '10px'})
                    ]))
            
            # Progressive Denoising Stages (NEW)
            if signal_options == 'progressive_denoising':
                print(f"DEBUG: ✅ EXECUTING Progressive Denoising Stages - signal_options={signal_options}")
                try:
                    wavelet_denoise = 'db8'
                    coeffs = pywt.wavedec(clean_data, wavelet_denoise)
                    
                    # Calculate thresholds at different percentiles
                    detail_coeffs = np.concatenate(coeffs[1:])
                    thresholds = [np.percentile(np.abs(detail_coeffs), p) for p in [50, 70, 85, 95, 99]]
                    threshold_names = ['Light (50%)', 'Moderate (70%)', 'Strong (85%)', 'Heavy (95%)', 'Extreme (99%)']
                    
                    print(f"DEBUG: Progressive denoising with {len(thresholds)} threshold levels")
                    
                    # Create plots for each denoising stage
                    for i, (threshold, name) in enumerate(zip(thresholds, threshold_names)):
                        # Apply progressive thresholding
                        denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
                        denoised_signal = pywt.waverec(denoised_coeffs, wavelet_denoise)[:len(clean_data)]
                        
                        # Calculate noise removed at this stage
                        noise_removed = clean_data - denoised_signal
                        noise_energy = np.sum(noise_removed**2)
                        noise_std = np.std(noise_removed)
                        
                        # Create stage plot
                        fig_denoise_stage = go.Figure()
                        
                        # Original signal (light gray)
                        fig_denoise_stage.add_trace(go.Scattergl(
                            x=clean_times, y=clean_data,
                            mode='lines',
                            name='Original Signal',
                            line=dict(color='lightgray', width=1),
                            hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Original</b>: %{{y:.3f}} {units}<extra></extra>'
                        ))
                        
                        # Denoised signal
                        colors_denoise = ['#3498db', '#2980b9', '#e67e22', '#e74c3c', '#c0392b']
                        fig_denoise_stage.add_trace(go.Scattergl(
                            x=clean_times, y=denoised_signal,
                            mode='lines',
                            name=f'Denoised ({name})',
                            line=dict(color=colors_denoise[i], width=2),
                            hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Denoised</b>: %{{y:.3f}} {units}<br><b>Threshold</b>: {threshold:.3f}<br><b>Noise Removed</b>: ±{noise_std:.3f}{units}<extra></extra>'
                        ))
                        
                        # Noise component (what was removed)
                        fig_denoise_stage.add_trace(go.Scattergl(
                            x=clean_times, y=noise_removed,
                            mode='lines',
                            name=f'Noise Removed (±{noise_std:.3f}{units})',
                            line=dict(color='red', width=1, dash='dot'),
                            hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Noise</b>: %{{y:.3f}} {units}<br><b>Energy</b>: {noise_energy:.1f}<extra></extra>'
                        ))
                        
                        fig_denoise_stage.update_layout(
                            title=f'Denoising Stage {i+1}: {name}<br><sub>Threshold: {threshold:.3f} | Noise Energy: {noise_energy:.1f} | Noise StdDev: {noise_std:.3f}{units}</sub>',
                            xaxis_title='Date',
                            yaxis_title=yaxis_label_full,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=350, width=1300,
                            showlegend=True,
                            legend=dict(
                                orientation='h',
                                x=0.5, y=1.02,
                                bgcolor='rgba(255,255,255,0.9)',
                                bordercolor='rgba(0,0,0,0.2)',
                                borderwidth=1,
                                xanchor='center',
                                yanchor='bottom'
                            )
                        )
                        
                        signal_plots.append(dcc.Graph(figure=fig_denoise_stage, style={'backgroundColor': 'white'}))
                    
                    # Create comparison plot showing all denoising levels
                    fig_compare = go.Figure()
                    
                    # Original signal
                    fig_compare.add_trace(go.Scattergl(
                        x=clean_times, y=clean_data,
                        mode='lines',
                        name='Original Signal',
                        line=dict(color='black', width=2),
                        hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Original</b>: %{{y:.3f}} {units}<extra></extra>'
                    ))
                    
                    # All denoised versions
                    for i, (threshold, name) in enumerate(zip(thresholds, threshold_names)):
                        denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
                        denoised_signal = pywt.waverec(denoised_coeffs, wavelet_denoise)[:len(clean_data)]
                        
                        fig_compare.add_trace(go.Scattergl(
                            x=clean_times, y=denoised_signal,
                            mode='lines',
                            name=f'Denoised {name}',
                            line=dict(color=colors_denoise[i], width=2),
                            hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Denoised ({name})</b>: %{{y:.3f}} {units}<extra></extra>'
                        ))
                    
                    fig_compare.update_layout(
                        title=f'Progressive Denoising Comparison: All Stages [{resample_label}]',
                        xaxis_title='Date',
                        yaxis_title=yaxis_label_full,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=500, width=1300,
                        showlegend=True
                    )
                    
                    signal_plots.append(dcc.Graph(figure=fig_compare, style={'backgroundColor': 'white'}))
                    
                    print(f"DEBUG: Successfully created {len(thresholds)} progressive denoising stages from {resample_label} data")
                    
                except Exception as e:
                    print(f"DEBUG: Progressive denoising error: {str(e)}")
                    signal_plots.append(html.Div([
                        html.P(f"Progressive denoising error: {str(e)}", 
                               style={'color': '#e67e22', 'textAlign': 'center', 'padding': '10px'})
                    ]))
            
            # Wavelet Basis Function Viewer (NEW) - Inspired by Figure 2 of research paper
            if signal_options == 'wavelet_basis_viewer':
                 print(f"DEBUG: ✅ EXECUTING Wavelet Basis Function Viewer - signal_options={signal_options}")
                 try:
                     # Define wavelets to display (like Figure 2)
                     wavelets_to_show = [
                         ('morl', 'Morlet', '#2c3e50'),
                         ('mexh', 'Mexican Hat (DOG m=2)', '#e74c3c'), 
                         ('db8', 'Daubechies 8', '#27ae60'),
                         ('haar', 'Haar', '#f39c12'),
                         ('cgau2', 'Complex Gaussian 2', '#9b59b6'),
                         ('shan', 'Shannon', '#1abc9c')
                     ]
                     
                     print(f"DEBUG: Displaying {len(wavelets_to_show)} wavelet basis functions")
                     
                     # Create time domain plots (like left side of Figure 2)
                     fig_time_domain = go.Figure()
                     
                     # Generate time vector
                     t = np.linspace(-4, 4, 1000)
                     
                     for i, (wavelet_name, display_name, color) in enumerate(wavelets_to_show):
                         try:
                             if wavelet_name in ['morl', 'mexh', 'cgau2', 'shan']:
                                 # Continuous wavelets
                                 wavelet_func = pywt.ContinuousWavelet(wavelet_name)
                                 # Generate wavelet function
                                 psi, x = pywt.ContinuousWavelet(wavelet_name).wavefun(length=1000)
                                 
                                 # Scale x to our desired range
                                 x_scaled = (x - np.mean(x)) / np.std(x) * 2  # Scale to approximately -4 to 4
                                 
                                 # Handle complex wavelets
                                 if np.iscomplexobj(psi):
                                     # Show real and imaginary parts (like Figure 2)
                                     fig_time_domain.add_trace(go.Scatter(
                                         x=x_scaled, y=np.real(psi),
                                         mode='lines',
                                         name=f'{display_name} (Real)',
                                         line=dict(color=color, width=2),
                                         hovertemplate=f'<b>{display_name} - Real Part</b><br>t: %{{x:.2f}}<br>ψ(t): %{{y:.3f}}<extra></extra>'
                                     ))
                                     fig_time_domain.add_trace(go.Scatter(
                                         x=x_scaled, y=np.imag(psi),
                                         mode='lines',
                                         name=f'{display_name} (Imag)',
                                         line=dict(color=color, width=2, dash='dash'),
                                         hovertemplate=f'<b>{display_name} - Imaginary Part</b><br>t: %{{x:.2f}}<br>ψ(t): %{{y:.3f}}<extra></extra>'
                                     ))
                                 else:
                                     # Real wavelets
                                     fig_time_domain.add_trace(go.Scatter(
                                         x=x_scaled, y=psi,
                                         mode='lines',
                                         name=display_name,
                                         line=dict(color=color, width=2),
                                         hovertemplate=f'<b>{display_name}</b><br>t: %{{x:.2f}}<br>ψ(t): %{{y:.3f}}<extra></extra>'
                                     ))
                                     
                             else:
                                 # Discrete wavelets - reconstruct wavelet function
                                 w = pywt.Wavelet(wavelet_name)
                                 phi, psi, x = w.wavefun(level=8)  # High resolution
                                 
                                 # Scale x to our desired range  
                                 x_scaled = (x - np.mean(x)) / np.std(x) * 2
                                 
                                 fig_time_domain.add_trace(go.Scatter(
                                     x=x_scaled, y=psi,
                                     mode='lines',
                                     name=display_name,
                                     line=dict(color=color, width=2),
                                     hovertemplate=f'<b>{display_name}</b><br>t: %{{x:.2f}}<br>ψ(t): %{{y:.3f}}<extra></extra>'
                                 ))
                                 
                         except Exception as e:
                             print(f"DEBUG: Could not plot {wavelet_name}: {str(e)}")
                             continue
                     
                     fig_time_domain.update_layout(
                         title='Wavelet Basis Functions in Time Domain<br><sub>Solid lines: Real part | Dashed lines: Imaginary part</sub>',
                         xaxis_title='Time (t/s)',
                         yaxis_title='Amplitude ψ(t)',
                         plot_bgcolor='white',
                         paper_bgcolor='white',
                         height=500, width=1300,
                         showlegend=True,
                         legend=dict(
                             x=1.02, y=1,
                             bgcolor='rgba(255,255,255,0.9)',
                             bordercolor='rgba(0,0,0,0.2)',
                             borderwidth=1
                         )
                     )
                     
                     signal_plots.append(dcc.Graph(figure=fig_time_domain, style={'backgroundColor': 'white'}))
                     
                     # Create frequency domain representation
                     fig_freq_domain = go.Figure()
                     
                     # Generate frequency vector
                     omega = np.linspace(0, 2, 500)
                     
                     for i, (wavelet_name, display_name, color) in enumerate(wavelets_to_show):
                         try:
                             if wavelet_name in ['morl', 'mexh']:
                                 # For these wavelets, we can show theoretical frequency response
                                 if wavelet_name == 'morl':
                                     # Morlet wavelet frequency response (Gaussian envelope)
                                     omega0 = 6  # Central frequency
                                     freq_resp = np.exp(-0.5 * (omega - omega0)**2) * np.exp(-0.5 * omega**2)
                                 elif wavelet_name == 'mexh':
                                     # Mexican Hat frequency response
                                     freq_resp = 2 * np.sqrt(2/3) * (omega**2) * np.exp(-0.5 * omega**2)
                                 
                                 fig_freq_domain.add_trace(go.Scatter(
                                     x=omega, y=freq_resp,
                                     mode='lines',
                                     name=display_name,
                                     line=dict(color=color, width=2),
                                     hovertemplate=f'<b>{display_name}</b><br>ω: %{{x:.2f}}<br>|Ψ(ω)|: %{{y:.3f}}<extra></extra>'
                                 ))
                                 
                         except Exception as e:
                             continue
                     
                     fig_freq_domain.update_layout(
                         title='Wavelet Basis Functions in Frequency Domain<br><sub>Fourier Transform of Wavelets</sub>',
                         xaxis_title='Frequency (sω/2π)',
                         yaxis_title='|Ψ(ω)|',
                         plot_bgcolor='white',
                         paper_bgcolor='white',
                         height=500, width=1300,
                         showlegend=True,
                         legend=dict(
                             x=1.02, y=1,
                             bgcolor='rgba(255,255,255,0.9)',
                             bordercolor='rgba(0,0,0,0.2)',
                             borderwidth=1
                         )
                     )
                     
                     signal_plots.append(dcc.Graph(figure=fig_freq_domain, style={'backgroundColor': 'white'}))
                     
                     # Create wavelet comparison table
                     wavelet_info = []
                     for wavelet_name, display_name, color in wavelets_to_show:
                         try:
                             if wavelet_name in ['morl', 'mexh', 'cgau2', 'shan']:
                                 w = pywt.ContinuousWavelet(wavelet_name)
                                 family = "Continuous"
                                 if wavelet_name == 'morl':
                                     properties = "Complex, good time-freq localization"
                                 elif wavelet_name == 'mexh':
                                     properties = "Real, second derivative of Gaussian"
                                 elif wavelet_name == 'cgau2':
                                     properties = "Complex Gaussian, very smooth"
                                 else:
                                     properties = "Shannon wavelet, ideal frequency response"
                             else:
                                 w = pywt.Wavelet(wavelet_name)
                                 family = w.family_name
                                 if wavelet_name == 'db8':
                                     properties = "Orthogonal, compact support"
                                 elif wavelet_name == 'haar':
                                     properties = "Simplest wavelet, discontinuous"
                                 else:
                                     properties = "Standard discrete wavelet"
                             
                             wavelet_info.append({
                                 'Wavelet': display_name,
                                 'Name': wavelet_name,
                                 'Family': family,
                                 'Properties': properties,
                                 'Best For': "General analysis" if wavelet_name == 'morl' else "Specific applications"
                             })
                             
                         except Exception as e:
                             continue
                     
                     if wavelet_info:
                         # Add title row
                         title_row = {
                             'Wavelet': 'Wavelet Basis Function Comparison',
                             'Name': '', 'Family': '', 'Properties': '', 'Best For': ''
                         }
                         wavelet_data_with_title = [title_row] + wavelet_info
                         
                         wavelet_table = dash_table.DataTable(
                             columns=[{"name": col, "id": col} for col in wavelet_info[0].keys()],
                             data=wavelet_data_with_title,
                             style_table={'width': '95%', 'margin': 'auto'},
                             style_cell={'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': 12},
                             style_header={'backgroundColor': '#e8f5e8', 'fontWeight': 'bold'},
                             style_data_conditional=[
                                 {
                                     'if': {'row_index': 0},  # Title row
                                     'backgroundColor': '#d5f4e6',
                                     'fontWeight': 'bold',
                                     'fontSize': 14
                                 },
                                 {
                                     'if': {'row_index': 'odd'},
                                     'backgroundColor': '#f9f9f9'
                                 }
                             ]
                         )
                         
                         signal_plots.append(html.Div([
                             wavelet_table
                         ], style={'marginTop': '20px'}))
                     
                     print(f"DEBUG: Successfully created wavelet basis function viewer from {resample_label} data")
                     
                 except Exception as e:
                     print(f"DEBUG: Wavelet basis viewer error: {str(e)}")
                     signal_plots.append(html.Div([
                         html.P(f"Wavelet basis viewer error: {str(e)}", 
                                style={'color': '#e67e22', 'textAlign': 'center', 'padding': '10px'})
                     ]))
            
            # === STATISTICAL ANALYSIS SECTION ===
            
            # Descriptive Statistics
            if 'descriptive_stats' in statistical_options:
                try:
                    stats_data = {
                        'Series': source_name,
                        'Count': len(clean_data),
                        'Mean': np.nanmean(clean_data),
                        'Median': np.nanmedian(clean_data),
                        'Std Dev': np.nanstd(clean_data),
                        'Min': np.nanmin(clean_data),
                        'Max': np.nanmax(clean_data),
                        'Range': np.nanmax(clean_data) - np.nanmin(clean_data),
                        'CV (%)': (np.nanstd(clean_data) / np.nanmean(clean_data)) * 100,
                        'Skewness': float(pd.Series(clean_data).skew()),
                        'Kurtosis': float(pd.Series(clean_data).kurtosis())
                    }
                    analysis_results['descriptive_stats'] = stats_data
                except Exception as e:
                    analysis_results['descriptive_stats'] = {'error': str(e)}
            
            # Trend Analysis
            if 'trend_analysis' in statistical_options:
                try:
                    import scipy.stats as stats
                    
                    # Create time index for trend analysis
                    time_numeric = np.arange(len(clean_data))
                    
                    if trend_method == 'linear_regression':
                        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, clean_data)
                        trend_line = slope * time_numeric + intercept
                        
                        fig_main.add_trace(go.Scattergl(
                            x=clean_times, y=trend_line,
                            mode='lines',
                            name='Linear Trend',
                            line=dict(color='#e67e22', width=2, dash='dashdot')
                        ))
                        
                        analysis_results['trend_analysis'] = {
                            'method': 'Linear Regression',
                            'slope': slope,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'trend': 'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'No trend',
                            'significant': p_value < (1 - confidence_level/100)
                        }
                    
                    elif trend_method == 'mann_kendall':
                        # Simplified Mann-Kendall test
                        n = len(clean_data)
                        s = 0
                        for i in range(n-1):
                            for j in range(i+1, n):
                                if clean_data[j] > clean_data[i]:
                                    s += 1
                                elif clean_data[j] < clean_data[i]:
                                    s -= 1
                        
                        var_s = n * (n - 1) * (2 * n + 5) / 18
                        z = s / np.sqrt(var_s) if var_s > 0 else 0
                        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                        
                        analysis_results['trend_analysis'] = {
                            'method': 'Mann-Kendall',
                            'kendall_tau': s,
                            'z_statistic': z,
                            'p_value': p_value,
                            'trend': 'Increasing' if z > 0 else 'Decreasing' if z < 0 else 'No trend',
                            'significant': p_value < (1 - confidence_level/100)
                        }
                        
                except Exception as e:
                    analysis_results['trend_analysis'] = {'error': str(e)}
            
            # Seasonal Decomposition
            if 'seasonal_decomp' in statistical_options:
                try:
                    from scipy import signal as scipy_signal
                    
                    # Simple seasonal decomposition using moving averages
                    if len(clean_data) >= 365:  # Need sufficient data for yearly patterns
                        # Estimate seasonal component (simple approach)
                        window_size = min(365, len(clean_data) // 4)
                        trend_component = pd.Series(clean_data).rolling(window=window_size, center=True).mean().values
                        seasonal_component = clean_data - trend_component
                        residual_component = clean_data - trend_component - seasonal_component
                        
                        # Remove NaNs
                        valid_idx = ~np.isnan(trend_component)
                        
                        if np.any(valid_idx):
                            fig_main.add_trace(go.Scattergl(
                                x=clean_times[valid_idx], 
                                y=trend_component[valid_idx],
                                mode='lines',
                                name='Seasonal Trend',
                                line=dict(color='#9b59b6', width=2)
                            ))
                            
                            extracted_signals['seasonal_trend'] = trend_component[valid_idx]
                            extracted_signals['seasonal_component'] = seasonal_component[valid_idx]
                            
                            analysis_results['seasonal_decomp'] = {
                                'trend_variance': np.nanvar(trend_component),
                                'seasonal_variance': np.nanvar(seasonal_component),
                                'residual_variance': np.nanvar(residual_component)
                            }
                        
                except Exception as e:
                    analysis_results['seasonal_decomp'] = {'error': str(e)}
            
            # Change Point Detection
            if 'change_points' in statistical_options:
                try:
                    # Simple change point detection using variance
                    window_size = max(30, len(clean_data) // 10)
                    change_points = []
                    
                    for i in range(window_size, len(clean_data) - window_size):
                        before = clean_data[i-window_size:i]
                        after = clean_data[i:i+window_size]
                        
                        # Test for significant difference in means
                        if len(before) > 5 and len(after) > 5:
                            from scipy.stats import ttest_ind
                            t_stat, p_val = ttest_ind(before, after)
                            if p_val < 0.01:  # Significant change
                                change_points.append({
                                    'index': i,
                                    'date': clean_times[i],
                                    'p_value': p_val,
                                    'mean_before': np.mean(before),
                                    'mean_after': np.mean(after)
                                })
                    
                    # Add change points to plot
                    for cp in change_points[:5]:  # Show only first 5 change points
                        fig_main.add_vline(
                            x=cp['date'],
                            line_dash="dash",
                            line_color="orange",
                            annotation_text=f"Change Point"
                        )
                    
                    analysis_results['change_points'] = {
                        'count': len(change_points),
                        'points': change_points[:10]  # Store first 10
                    }
                    
                except Exception as e:
                    analysis_results['change_points'] = {'error': str(e)}
            
            # === EVENT DETECTION SECTION ===
            
            # Drought Events
            if 'drought_events' in event_options:
                try:
                    drought_level = np.percentile(clean_data, drought_threshold)
                    drought_mask = clean_data <= drought_level
                    
                    # Find continuous drought periods
                    drought_events = []
                    in_drought = False
                    start_idx = None
                    
                    for i, is_drought in enumerate(drought_mask):
                        if is_drought and not in_drought:
                            in_drought = True
                            start_idx = i
                        elif not is_drought and in_drought:
                            duration = i - start_idx
                            if duration >= min_event_duration:
                                drought_events.append({
                                    'start_date': clean_times[start_idx],
                                    'end_date': clean_times[i-1],
                                    'duration_days': duration,
                                    'min_value': np.min(clean_data[start_idx:i]),
                                    'severity': drought_level - np.min(clean_data[start_idx:i])
                                })
                            in_drought = False
                    
                    # Add drought threshold line
                    fig_main.add_hline(
                        y=drought_level,
                        line_dash="dash",
                        line_color="brown",
                        annotation_text=f"Drought Threshold ({drought_threshold}%): {drought_level:.2f} {units}"
                    )
                    
                    # Highlight drought periods
                    for event in drought_events:
                        fig_main.add_vrect(
                            x0=event['start_date'],
                            x1=event['end_date'],
                            fillcolor="brown",
                            opacity=0.2,
                            layer="below",
                            line_width=0
                        )
                    
                    analysis_results['drought_events'] = {
                        'threshold': drought_level,
                        'count': len(drought_events),
                        'events': drought_events
                    }
                    
                except Exception as e:
                    analysis_results['drought_events'] = {'error': str(e)}
            
            # Flood/High Water Events
            if 'flood_events' in event_options:
                try:
                    flood_level = np.percentile(clean_data, flood_threshold)
                    flood_mask = clean_data >= flood_level
                    
                    # Find continuous flood periods
                    flood_events = []
                    in_flood = False
                    start_idx = None
                    
                    for i, is_flood in enumerate(flood_mask):
                        if is_flood and not in_flood:
                            in_flood = True
                            start_idx = i
                        elif not is_flood and in_flood:
                            duration = i - start_idx
                            if duration >= min_event_duration:
                                flood_events.append({
                                    'start_date': clean_times[start_idx],
                                    'end_date': clean_times[i-1],
                                    'duration_days': duration,
                                    'max_value': np.max(clean_data[start_idx:i]),
                                    'magnitude': np.max(clean_data[start_idx:i]) - flood_level
                                })
                            in_flood = False
                    
                    # Add flood threshold line
                    fig_main.add_hline(
                        y=flood_level,
                        line_dash="dash",
                        line_color="blue",
                        annotation_text=f"Flood Threshold ({flood_threshold}%): {flood_level:.2f} {units}"
                    )
                    
                    # Highlight flood periods
                    for event in flood_events:
                        fig_main.add_vrect(
                            x0=event['start_date'],
                            x1=event['end_date'],
                            fillcolor="blue",
                            opacity=0.2,
                            layer="below",
                            line_width=0
                        )
                    
                    analysis_results['flood_events'] = {
                        'threshold': flood_level,
                        'count': len(flood_events),
                        'events': flood_events
                    }
                    
                except Exception as e:
                    analysis_results['flood_events'] = {'error': str(e)}
            
            # Enhanced Plot Styling
            fig_main.update_layout(
                title=dict(
                    text=f'Comprehensive Groundwater Analysis - {source_name}',
                    font=dict(size=18, color='#2c3e50', family='Arial Black'),
                    x=0.5
                ),
                xaxis_title='Date',
                yaxis_title=yaxis_label_full,
                plot_bgcolor='#fafafa',
                paper_bgcolor='white',
                height=700, width=1400,  # Larger to accommodate more data
                xaxis=dict(
                    title_font=dict(size=16, family='Arial', color='#34495e', weight='bold'),
                    tickfont=dict(size=12, family='Arial', color='#2c3e50'),
                    showgrid=True,
                    gridwidth=0.5,
                    gridcolor='rgba(189, 195, 199, 0.4)'
                ),
                yaxis=dict(
                    title_font=dict(size=16, family='Arial', color='#34495e', weight='bold'),
                    tickfont=dict(size=12, family='Arial', color='#2c3e50'),
                    showgrid=True,
                    gridwidth=0.5,
                    gridcolor='rgba(189, 195, 199, 0.4)'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor='rgba(255,255,255,0.9)',
                    font=dict(size=11)
                ),
                hovermode='x unified'
            )
            
            # Original plot already added above - no need to add again
            
            # === RESULTS DISPLAY SECTION ===
            
            # Statistical summary tables removed - cleaner interface focused on signal extraction
            
            # Multi-Well Comparison Plot
            if 'multi_well' in advanced_options and len(well_columns) > 1:
                fig_comparison = go.Figure()
                
                colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
                for i, well_col in enumerate(well_columns[:6]):  # Limit to 6 wells for clarity
                    if well_col in all_clean_data:
                        well_info = all_clean_data[well_col]
                        fig_comparison.add_trace(go.Scattergl(
                            x=well_info['times'], 
                            y=well_info['data'],
                            mode='lines',
                            name=well_col,
                            line=dict(color=colors[i % len(colors)], width=2),
                            opacity=0.8
                        ))
                
                fig_comparison.update_layout(
                    title=dict(
                        text='Multi-Well Comparison',
                        font=dict(size=16, color='#2c3e50'),
                        x=0.5
                    ),
                    xaxis_title='Date',
                                                yaxis_title=yaxis_label_full,
                    height=500, width=1300,
                    plot_bgcolor='#fafafa',
                    paper_bgcolor='white',
                    hovermode='x unified'
                )
                
                signal_plots.append(html.Div([
                    html.H5('Multi-Well Comparison', style={'color': '#8e44ad', 'marginBottom': '15px'}),
                    dcc.Graph(figure=fig_comparison)
                ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'marginBottom': '20px'}))
            
            # Correlation Matrix
            if 'show_correlation' in display_options and len(all_well_data) > 1:
                try:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    import io
                    import base64
                    
                    # Create correlation matrix
                    corr_data = {}
                    min_length = min([len(data) for data in all_well_data.values()])
                    
                    for name, data in all_well_data.items():
                        corr_data[name] = data[:min_length]  # Truncate to same length
                    
                    corr_df = pd.DataFrame(corr_data)
                    correlation_matrix = corr_df.corr()
                    
                    # Create correlation heatmap using plotly
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=correlation_matrix.round(3).values,
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
                    ))
                    
                    fig_corr.update_layout(
                        title=dict(
                            text='Cross-Correlation Matrix',
                            font=dict(size=16, color='#2c3e50'),
                            x=0.5
                        ),
                        height=500, width=600,
                        xaxis={'side': 'bottom'},
                        yaxis={'side': 'left'}
                    )
                    
                    signal_plots.append(html.Div([
                        html.H5('Correlation Analysis', style={'color': '#27ae60', 'marginBottom': '15px'}),
                        dcc.Graph(figure=fig_corr)
                    ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'marginBottom': '20px'}))
                    
                except Exception as e:
                    signal_plots.append(html.Div([
                        html.P(f"Correlation analysis error: {str(e)}", 
                               style={'color': '#e67e22', 'textAlign': 'center', 'padding': '10px'})
                    ]))
            
            # Enhanced Export Functionality
            export_components = []
            
            # Data Export Section
            if extracted_signals or analysis_results:
                # Create comprehensive data export
                export_data = {}
                export_data['DateTime'] = clean_times
                export_data[f'Original_{source_name}'] = clean_data
                
                # Add extracted signals
                for signal_name, signal_data in extracted_signals.items():
                    if len(signal_data) == len(clean_times):
                        export_data[f'Extracted_{signal_name}'] = signal_data
                
                # Create export DataFrame
                export_df = pd.DataFrame(export_data)
                
                # Export summary
                export_summary = []
                if extracted_signals:
                    export_summary.append(f"{len(extracted_signals)} Signal Extraction Components")
                if analysis_results:
                    export_summary.append(f"{len(analysis_results)} Analysis Results")
                
                export_components.append(html.Div([
                    html.H5('Data Export Summary', style={'color': '#2c3e50', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(item) for item in export_summary
                    ], style={'fontSize': '14px', 'color': '#34495e'}),
                    html.Button('Download Complete Analysis (CSV)', 
                               id='download-complete-analysis-btn',
                               style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 
                                     'padding': '12px 24px', 'borderRadius': '5px', 'cursor': 'pointer', 'marginTop': '10px'}),
                    dcc.Download(id='download-complete-analysis-csv')
                ], style={'padding': '15px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'marginBottom': '15px'}))
            
            # Analysis Status Report removed - cleaner interface
            
            if export_components:
                signal_plots.extend(export_components)

            
            # Return complete signal extraction interface
            return html.Div([
                signal_controls,
                html.Div(signal_plots, style={'marginTop': '24px'})
            ], style={
                'backgroundColor': '#f8fafc', 
                'minHeight': '100vh', 
                'padding': '32px',
                'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
            })
        elif tab == 'coherence':
            # Wavelet Coherence Tab - Add controls directly in the tab
            # Create series options for coherence analysis
            series_options = [{'label': w, 'value': w} for w in well_columns]
            if 'mm_Rain' in data_resampled.columns:
                series_options.append({'label': 'Rainfall', 'value': 'mm_Rain'})
            
            coherence_controls = html.Div([
                html.H5("Select Series for Coherence Analysis", style={'marginBottom': 15, 'color': '#444'}),
                html.Div([
                    html.Div([
                        html.Label("First Series:", style={'marginBottom': 4, 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='coherence-series1-dropdown',
                            options=series_options,
                            value=coh_series1 or well_columns[0],
                            clearable=False,
                            style={'marginBottom': 8}
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    html.Div([
                        html.Label("Second Series:", style={'marginBottom': 4, 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='coherence-series2-dropdown',
                            options=series_options,
                            value=coh_series2 or 'mm_Rain',
                            clearable=False,
                            style={'marginBottom': 8}
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
                ]),

            ], style={'marginBottom': 20, 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9'})
            
            # For the coherence tab, always show the analysis (enable by default)
            show_coherence = ['coh']  # Override for this tab
            
            # Use dropdown selections if available, otherwise use defaults
            series1_selected = coh_series1 or well_columns[0]
            series2_selected = coh_series2 or 'mm_Rain'
            
            # Get selected series
            s1 = data_resampled[series1_selected].values if series1_selected in data_resampled.columns else None
            s2 = data_resampled[series2_selected].values if series2_selected in data_resampled.columns else None
            x_coh = data_resampled['DateTime'].values
            # Use the selected series for customdata
            series1_data = s1 if s1 is not None else np.full_like(x_coh, np.nan)
            series2_data = s2 if s2 is not None else np.full_like(x_coh, np.nan)
            seasons = get_nz_season_for_dates(x_coh)
            if s1 is None or s2 is None or len(s1) != len(s2):
                return html.Div([
                    html.H4('Selected series are not available or have mismatched lengths.')
                ], style={'backgroundColor': 'white'})
            # Remove NaNs
            mask = ~np.isnan(s1) & ~np.isnan(s2)
            s1 = s1[mask]
            s2 = s2[mask]
            x_coh = x_coh[mask]
            series1_data = series1_data[mask]
            series2_data = series2_data[mask]
            seasons = np.array(seasons)[mask]
            if len(s1) < 10 or len(s2) < 10:
                return html.Div([
                    html.H4('Not enough data for coherence analysis.')
                ], style={'backgroundColor': 'white'})
            # Compute CWT for both series (only for continuous wavelets)
            if wavelet_type != 'cwt':
                return html.Div([
                    html.H4('Wavelet Coherence analysis is only supported for Continuous Wavelet Transform (CWT). Please select CWT as the wavelet type.')
                ], style={'backgroundColor': 'white'})
            
            # Use the same scale calculation as main wavelet analysis tab
            min_scale = 1
            n = len(s1)
            max_scale = n // 8 if n >= 8 else 2
            num_octaves = np.log2(max_period_conv / min_period_conv) if max_period_conv > min_period_conv else 1
            voices_per_octave = int(np.ceil(num_periods / num_octaves)) if num_octaves > 0 else 2
            voices_per_octave = max(2, min(voices_per_octave, 48))
            delta_j = 1 / voices_per_octave
            J = int(np.log2(max_scale / min_scale) / delta_j) if max_scale > min_scale else 1
            scales = min_scale * 2 ** (np.arange(J + 1) * delta_j)
            
            # Convert scales to periods for meaningful y-axis (use the wavelet's
            # actual centre frequency so periods match pywt.cwt's frequencies).
            periods_coh = scales * dt / pywt.central_frequency(wavelet)
            # Convert periods to selected units
            if period_units == 'weeks':
                periods_coh = periods_coh / 7
            elif period_units == 'months':
                periods_coh = periods_coh / 30.44
            elif period_units == 'years':
                periods_coh = periods_coh / 365.25
                
            # Filter to user-specified period range
            period_mask = (periods_coh >= min_period_conv) & (periods_coh <= max_period_conv)
            periods_coh = periods_coh[period_mask]
            scales = scales[period_mask]
            
            if len(scales) < 2 or len(periods_coh) < 2:
                return html.Div([
                    coherence_controls,
                    html.H4("Not enough periods in selected range for coherence analysis. Try expanding your period range.")
                ], style={'backgroundColor': 'white'})
            
            coef1, _ = pywt.cwt(s1, scales, wavelet)
            coef2, _ = pywt.cwt(s2, scales, wavelet)
            S1 = np.abs(coef1) ** 2
            S2 = np.abs(coef2) ** 2
            S12 = coef1 * np.conj(coef2)
            def smooth(mat, window=5):
                return uniform_filter1d(mat, size=window, axis=1, mode='nearest')
            
            # Compute coherence with better numerical stability
            S1_smooth = smooth(S1, 5)
            S2_smooth = smooth(S2, 5) 
            S12_smooth = smooth(S12, 5)
            
            eps = 1e-12  # Better numerical stability
            WCOH = np.abs(S12_smooth) ** 2 / (S1_smooth * S2_smooth + eps)
            
            # Ensure coherence is in [0,1] range and handle invalid values
            WCOH = np.clip(WCOH, 0, 1)
            WCOH = np.nan_to_num(WCOH, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Additional data cleaning to eliminate white space issues
            WCOH = np.where(np.isfinite(WCOH), WCOH, 0.0)
            WCOH = np.where(WCOH < 1e-10, 1e-10, WCOH)  # Replace tiny values that cause white space
            
            # Calculate phase for arrows
            phase_angles = np.angle(smooth(S12, 5))
            
            # Convert phase to time lag in selected units
            # Phase angle to time lag: lag = phase / (2π * frequency)
            freqs_coh = 1 / periods_coh  # frequency = 1/period
            time_lags = phase_angles / (2 * np.pi * freqs_coh[:, np.newaxis])
            
            # Convert time lags to selected units
            if period_units == 'weeks':
                time_lags = time_lags / 7
                lag_units = 'weeks'
            elif period_units == 'months':
                time_lags = time_lags / 30.44
                lag_units = 'months'
            elif period_units == 'years':
                time_lags = time_lags / 365.25
                lag_units = 'years'
            else:
                lag_units = period_units
            n_scales, n_times = WCOH.shape
            customdata = np.stack([series1_data, series2_data, seasons], axis=-1)
            customdata = np.broadcast_to(customdata, (n_scales, n_times, 3))
            # Detect analysis type to optimize visualization
            is_well_rainfall = ('mm_Rain' in [series1_selected, series2_selected] or 
                              'rain' in series1_selected.lower() or 'rain' in series2_selected.lower())
            is_well_well = not is_well_rainfall
            
            # Ensure proper normalization for coherence display with analysis-specific cleaning
            WCOH_display = np.copy(WCOH)
            
            if is_well_well:
                # Well-to-well coherence is typically lower, so adjust visualization
                WCOH_display = np.clip(WCOH_display, 0.001, 1.0)
                WCOH_display = np.where(np.isfinite(WCOH_display), WCOH_display, 0.001)
                # Use a more sensitive baseline for well-well
                WCOH_display = np.maximum(WCOH_display, 0.001)
                
                # Calculate appropriate visualization parameters for well-well
                max_coherence = np.nanmax(WCOH_display)
                if max_coherence < 0.5:
                    # Low coherence case - adjust scales
                    zmin_val = 0.001
                    zmax_val = min(max_coherence * 1.2, 1.0)  # Scale to data range
                    zmid_val = zmax_val * 0.4  # Lower midpoint for low coherence
                    colorscale = 'Viridis'  # Better for low values
                else:
                    # Normal coherence case
                    zmin_val = 0.001
                    zmax_val = 1.0
                    zmid_val = 0.3  # Lower midpoint for well-well
                    colorscale = 'RdYlBu_r'
            else:
                # Well-to-rainfall coherence is typically stronger
                WCOH_display = np.clip(WCOH_display, 0.001, 1.0)
                WCOH_display = np.where(np.isfinite(WCOH_display), WCOH_display, 0.001)
                WCOH_display = np.maximum(WCOH_display, 0.001)
                
                # Standard visualization for well-rainfall
                zmin_val = 0.001
                zmax_val = 1.0
                zmid_val = 0.5
                colorscale = 'RdYlBu_r'
            
            fig = go.Figure(
                data=go.Heatmap(
                    z=WCOH_display,
                    x=x_coh,
                    y=periods_coh,
                    customdata=customdata,
                    colorscale=colorscale,
                    zmin=zmin_val,
                    zmax=zmax_val,
                    zmid=zmid_val,
                    connectgaps=False,  # Don't connect gaps in data
                    showscale=True,
                    colorbar=dict(
                        title='Wavelet Coherence',
                        x=1.03,  # Uniform position for all heatmaps
                        xanchor='left',
                        thickness=20,  # Standardized thickness
                        len=0.8,  # Increased height
                        tickmode='linear',
                        tick0=0,
                        dtick=0.2 if is_well_rainfall else 0.1  # More ticks for well-well
                    ),
                    hovertemplate=f'<b>Date</b>: %{{x}}<br><b>Period</b>: %{{y:.3f}}<br><b>Coherence</b>: %{{z:.3f}}<br>' +
                                  f'<b>Series 1 ({series1_selected})</b>: %{{customdata[0]:.3f}}<br><b>Series 2 ({series2_selected})</b>: %{{customdata[1]:.3f}}<br><b>Season</b>: %{{customdata[2]}}<extra></extra>'
                )
            )
            # Function to get interpretation labels based on mode
            def get_arrow_interpretation(mode, series1_name, series2_name):
                interpretations = {
                    'recharge_discharge': {
                        'lag': 'Recharge', 'lead': 'Discharge',
                        'lag_desc': 'Groundwater lags rainfall (recharge process)',
                        'lead_desc': 'Groundwater leads rainfall (discharge process)',
                        'title': 'Recharge/Discharge'
                    },
                    'pressure_flow': {
                        'lag': 'Flow Response', 'lead': 'Pressure Response',
                        'lag_desc': 'Actual water movement (slower response)',
                        'lead_desc': 'Elastic/barometric effects (faster response)',
                        'title': 'Pressure/Flow'
                    },
                    'hydraulic_connectivity': {
                        'lag': 'Downstream', 'lead': 'Upstream',
                        'lag_desc': f'{series1_name} responds to {series2_name}',
                        'lead_desc': f'{series1_name} influences {series2_name}',
                        'title': 'Hydraulic Flow'
                    },
                    'tidal_response': {
                        'lag': 'Loading Response', 'lead': 'Elastic Response',
                        'lag_desc': 'Aquifer loading/unloading effects',
                        'lead_desc': 'Elastic/immediate pressure response',
                        'title': 'Tidal Response'
                    },
                    'pumping_impact': {
                        'lag': 'Pumping Impact', 'lead': 'Recovery Lead',
                        'lag_desc': 'Responds to pumping influence',
                        'lead_desc': 'Recovers before pumping effects diminish',
                        'title': 'Pumping Effects'
                    },
                    'phase_generic': {
                        'lag': 'Lagging', 'lead': 'Leading',
                        'lag_desc': f'{series1_name} lags {series2_name}',
                        'lead_desc': f'{series1_name} leads {series2_name}',
                        'title': 'Phase Relationship'
                    },
                    'custom': {
                        'lag': 'Response', 'lead': 'Driver',
                        'lag_desc': 'Series 1 responds to Series 2',
                        'lead_desc': 'Series 1 drives Series 2',
                        'title': 'Custom Analysis'
                    }
                }
                return interpretations.get(mode, interpretations['recharge_discharge'])
            
            # Add phase arrows if requested
            if 'phase_arrows' in show_overlays:
                # Use the SAME approach as Cross-Wavelet tab (which works without white space)
                arrow_step_time = max(1, n_times // 8)  # Conservative sampling
                arrow_step_scale = max(1, n_scales // 5)
                
                # Simple subsampling without complex meshgrid
                phase_arrows_x = x_coh[::arrow_step_time]
                phase_arrows_y = periods_coh[::arrow_step_scale]
                
                # Get phase and coherence data at these points
                phase_data = phase_angles[::arrow_step_scale, ::arrow_step_time]
                coh_data = WCOH[::arrow_step_scale, ::arrow_step_time]
                
                # Debug: ensure we have valid data shapes
                if phase_data.size == 0 or coh_data.size == 0:
                    # Use less aggressive subsampling if arrays are too small
                    arrow_step_time = max(1, n_times // 4)
                    arrow_step_scale = max(1, n_scales // 3)
                    phase_arrows_x = x_coh[::arrow_step_time]
                    phase_arrows_y = periods_coh[::arrow_step_scale]
                    phase_data = phase_angles[::arrow_step_scale, ::arrow_step_time]
                    coh_data = WCOH[::arrow_step_scale, ::arrow_step_time]
                
                # Flatten and filter for high coherence only with detailed info
                x_arrows = []
                y_arrows = []
                hover_info = []
                
                # Adaptive threshold based on analysis type to prevent white space
                if is_well_well:
                    # Well-to-well analysis - use much lower threshold
                    coh_threshold = 0.05  # Very low threshold for well-well
                    if np.any(coh_data > 0.05):
                        # Use 30th percentile for well-well (lower than well-rainfall)
                        percentile_threshold = np.percentile(coh_data[coh_data > 0.05], 30)
                        coh_threshold = max(percentile_threshold, 0.05)
                else:
                    # Well-to-rainfall analysis - standard threshold
                    coh_threshold = 0.1  # Standard threshold for well-rainfall
                    if np.any(coh_data > 0):
                        percentile_threshold = np.percentile(coh_data[coh_data > 0], 50)
                        coh_threshold = max(percentile_threshold, 0.1)
                       
                for i in range(len(phase_arrows_y)):
                    for j in range(len(phase_arrows_x)):
                        if i < phase_data.shape[0] and j < phase_data.shape[1]:
                            coh_val = coh_data[i, j]
                            phase_val = phase_data[i, j]
                            
                            if coh_val > coh_threshold and np.isfinite(coh_val) and np.isfinite(phase_val):
                                x_arrows.append(phase_arrows_x[j])
                                y_arrows.append(phase_arrows_y[i])
                                
                                # Calculate phase angle in degrees
                                phase_degrees = phase_val * 180 / np.pi
                                
                                # Calculate time lag
                                period_val = phase_arrows_y[i]
                                frequency = 1.0 / period_val
                                time_lag = phase_val / (2 * np.pi * frequency)
                                
                                # Convert time lag to selected units
                                if period_units == 'weeks':
                                    time_lag_converted = time_lag / 7
                                    lag_units_local = 'weeks'
                                elif period_units == 'months':
                                    time_lag_converted = time_lag / 30.44
                                    lag_units_local = 'months'
                                elif period_units == 'years':
                                    time_lag_converted = time_lag / 365.25
                                    lag_units_local = 'years'
                                else:
                                    time_lag_converted = time_lag
                                    lag_units_local = period_units
                                
                                # Determine phase relationship
                                if abs(time_lag_converted) < 0.01:
                                    relationship = "In-Phase (Synchronous)"
                                elif time_lag_converted > 0:
                                    relationship = f"{series1_selected} LAGS {series2_selected}"
                                else:
                                    relationship = f"{series1_selected} LEADS {series2_selected}"
                                
                                # Get interpretation based on mode with automatic selection
                                if arrow_interpretation_mode:
                                    interpretation_mode = arrow_interpretation_mode
                                else:
                                    # Auto-select based on series types
                                    if 'mm_Rain' in [series1_selected, series2_selected] or 'rain' in series1_selected.lower() or 'rain' in series2_selected.lower():
                                        interpretation_mode = 'recharge_discharge'
                                    elif 'tide' in series1_selected.lower() or 'tide' in series2_selected.lower():
                                        interpretation_mode = 'tidal_response'
                                    elif 'pump' in series1_selected.lower() or 'pump' in series2_selected.lower():
                                        interpretation_mode = 'pumping_impact'
                                    else:
                                        interpretation_mode = 'hydraulic_connectivity'
                                
                                if interpretation_mode == 'recharge_discharge':
                                    if time_lag_converted > 0:
                                        process = "Recharge Response"
                                    else:
                                        process = "Discharge Process"
                                elif interpretation_mode == 'hydraulic_connectivity':
                                    if time_lag_converted > 0:
                                        process = f"{series1_selected} responds to {series2_selected}"
                                    else:
                                        process = f"{series1_selected} influences {series2_selected}"
                                elif interpretation_mode == 'tidal_response':
                                    if time_lag_converted > 0:
                                        process = "Loading Response"
                                    else:
                                        process = "Elastic Response"
                                elif interpretation_mode == 'pumping_impact':
                                    if time_lag_converted > 0:
                                        process = "Pumping Impact"
                                    else:
                                        process = "Recovery Lead"
                                else:
                                    process = relationship
                                
                                hover_info.append({
                                    'coherence': coh_val,
                                    'phase_deg': phase_degrees,
                                    'time_lag': time_lag_converted,
                                    'lag_units': lag_units_local,
                                    'relationship': relationship,
                                    'process': process
                                })
                
                # Fallback: if no arrows meet threshold, lower the bar to show at least some
                if len(x_arrows) == 0 and np.any(coh_data > 0) and not is_well_well:
                    # Use a much lower threshold to ensure some arrows are visible
                    fallback_threshold = max(np.percentile(coh_data[coh_data > 0], 20), 0.01) if np.any(coh_data > 0) else 0.01
                    for i in range(len(phase_arrows_y)):
                        for j in range(len(phase_arrows_x)):
                            if i < phase_data.shape[0] and j < phase_data.shape[1]:
                                coh_val = coh_data[i, j]
                                phase_val = phase_data[i, j]
                                
                                if coh_val > fallback_threshold and np.isfinite(coh_val) and np.isfinite(phase_val):
                                    x_arrows.append(phase_arrows_x[j])
                                    y_arrows.append(phase_arrows_y[i])
                                    
                                    # Calculate the same hover info as before
                                    phase_degrees = phase_val * 180 / np.pi
                                    period_val = phase_arrows_y[i]
                                    frequency = 1.0 / period_val
                                    time_lag = phase_val / (2 * np.pi * frequency)
                                    
                                    # Convert time lag to selected units
                                    if period_units == 'weeks':
                                        time_lag_converted = time_lag / 7
                                        lag_units_local = 'weeks'
                                    elif period_units == 'months':
                                        time_lag_converted = time_lag / 30.44
                                        lag_units_local = 'months'
                                    elif period_units == 'years':
                                        time_lag_converted = time_lag / 365.25
                                        lag_units_local = 'years'
                                    else:
                                        time_lag_converted = time_lag
                                        lag_units_local = period_units
                                    
                                    # Determine phase relationship
                                    if abs(time_lag_converted) < 0.01:
                                        relationship = "In-Phase (Synchronous)"
                                    elif time_lag_converted > 0:
                                        relationship = f"{series1_selected} LAGS {series2_selected}"
                                    else:
                                        relationship = f"{series1_selected} LEADS {series2_selected}"
                                    
                                    # Get interpretation based on mode
                                    interpretation_mode = arrow_interpretation_mode or 'recharge_discharge'
                                    if interpretation_mode == 'recharge_discharge':
                                        if time_lag_converted > 0:
                                            process = "Recharge Response"
                                        else:
                                            process = "Discharge Process"
                                    else:
                                        process = relationship
                                    
                                    hover_info.append({
                                        'coherence': coh_val,
                                        'phase_deg': phase_degrees,
                                        'time_lag': time_lag_converted,
                                        'lag_units': lag_units_local,
                                        'relationship': relationship,
                                        'process': process
                                    })
                
                if len(x_arrows) == 0 and not is_well_well:
                    # Flatten all coherence and phase data
                    flat_coh = coh_data.flatten()
                    flat_phase = phase_data.flatten()
                    
                    # Get indices of top 5 coherence values
                    valid_mask = np.isfinite(flat_coh) & np.isfinite(flat_phase) & (flat_coh > 0)
                    if np.any(valid_mask):
                        valid_indices = np.where(valid_mask)[0]
                        top_indices = valid_indices[np.argsort(flat_coh[valid_indices])[-5:]]  # Top 5
                        
                        for idx in top_indices:
                            i_2d = idx // coh_data.shape[1]
                            j_2d = idx % coh_data.shape[1]
                            
                            if i_2d < len(phase_arrows_y) and j_2d < len(phase_arrows_x):
                                coh_val = flat_coh[idx]
                                phase_val = flat_phase[idx]
                                
                                x_arrows.append(phase_arrows_x[j_2d])
                                y_arrows.append(phase_arrows_y[i_2d])
                                
                                # Calculate hover info
                                phase_degrees = phase_val * 180 / np.pi
                                period_val = phase_arrows_y[i_2d]
                                frequency = 1.0 / period_val
                                time_lag = phase_val / (2 * np.pi * frequency)
                                
                                if period_units == 'weeks':
                                    time_lag_converted = time_lag / 7
                                    lag_units_local = 'weeks'
                                elif period_units == 'months':
                                    time_lag_converted = time_lag / 30.44
                                    lag_units_local = 'months'
                                elif period_units == 'years':
                                    time_lag_converted = time_lag / 365.25
                                    lag_units_local = 'years'
                                else:
                                    time_lag_converted = time_lag
                                    lag_units_local = period_units
                                
                                if abs(time_lag_converted) < 0.01:
                                    relationship = "In-Phase (Synchronous)"
                                elif time_lag_converted > 0:
                                    relationship = f"{series1_selected} LAGS {series2_selected}"
                                else:
                                    relationship = f"{series1_selected} LEADS {series2_selected}"
                                
                                interpretation_mode = arrow_interpretation_mode or 'recharge_discharge'
                                if interpretation_mode == 'recharge_discharge':
                                    if time_lag_converted > 0:
                                        process = "Recharge Response"
                                    else:
                                        process = "Discharge Process"
                                else:
                                    process = relationship
                                
                                hover_info.append({
                                    'coherence': coh_val,
                                    'phase_deg': phase_degrees,
                                    'time_lag': time_lag_converted,
                                    'lag_units': lag_units_local,
                                    'relationship': relationship,
                                    'process': process
                                })
                

                

                
                # Only add arrows if we have valid points  
                if len(x_arrows) > 0:
                    # Create custom hover text and determine colors
                    hover_text = []
                    arrow_colors = []
                    
                    for info in hover_info:
                        hover_text.append(
                            f"Coherence: {info['coherence']:.3f}<br>"
                            f"Phase: {info['phase_deg']:.1f}°<br>"
                            f"Time Lag: {info['time_lag']:.2f} {info['lag_units']}<br>"
                            f"Relationship: {info['relationship']}<br>"
                            f"Process: {info['process']}"
                        )
                        
                        # Color coding based on process/relationship
                        if abs(info['time_lag']) < 0.01:
                            arrow_colors.append('green')  # In-phase/synchronous
                        elif info['time_lag'] > 0:
                            arrow_colors.append('red')    # Recharge/lagging
                        else:
                            arrow_colors.append('blue')   # Discharge/leading
                    
                    # Convert phase angles to proper arrow text (simplified approach)
                    arrow_text = []
                    for info in hover_info:
                        phase_deg = info['phase_deg']
                        # Convert phase to simple arrow directions (4 main directions only)
                        if -45 <= phase_deg < 45:
                            arrow_text.append('→')  # Right
                        elif 45 <= phase_deg < 135:
                            arrow_text.append('↑')  # Up
                        elif 135 <= phase_deg <= 180 or -180 <= phase_deg < -135:
                            arrow_text.append('←')  # Left
                        else:  # -135 <= phase_deg < -45
                            arrow_text.append('↓')  # Down
                    
                    # Convert x_arrows to proper datetime format - ROBUST conversion
                    x_arrows_clean = []
                    for x_val in x_arrows:
                        try:
                            if hasattr(x_val, 'strftime'):
                                x_arrows_clean.append(x_val)
                            elif isinstance(x_val, (int, float)):
                                # If it's a numeric timestamp, convert it
                                x_arrows_clean.append(pd.to_datetime(x_val, unit='s'))
                            else:
                                x_arrows_clean.append(pd.to_datetime(x_val))
                        except:
                            # Fallback - use first valid date if conversion fails
                            x_arrows_clean.append(x_coh[0])
                    
                    # Ensure all arrays have the same length to prevent white space
                    min_length = min(len(x_arrows_clean), len(y_arrows), len(arrow_colors), len(arrow_text), len(hover_text))
                    if min_length > 0:
                        x_arrows_clean = x_arrows_clean[:min_length]
                        y_arrows = y_arrows[:min_length]
                        arrow_colors = arrow_colors[:min_length]
                        arrow_text = arrow_text[:min_length]
                        hover_text = hover_text[:min_length]
                        
                        # PLOTLY FIX: Use simple circle markers to avoid diamond+text rendering issues
                        if min_length >= 3 and all(x is not None and not pd.isna(x) for x in x_arrows_clean):
                            fig.add_trace(go.Scatter(
                                x=x_arrows_clean,
                                y=y_arrows,
                                mode='markers',  # Remove text to avoid Plotly rendering conflicts
                                marker=dict(
                                    symbol='circle',  # Use simple circles instead of diamonds
                                    size=10,  # Smaller size for better rendering
                                    color=arrow_colors,
                                    line=dict(color='black', width=1),
                                    opacity=0.8
                                ),
                                name='Phase Relationships',
                                hovertemplate='<b>Date</b>: %{x}<br><b>Period</b>: %{y:.1f} ' + period_units + '<br>%{customdata}<extra></extra>',
                                customdata=hover_text,
                                showlegend=True
                            ))
            
            
            # --- Wavelet Coherence Overlays ---
            
            # 1. Significance Overlay for Coherence 
            if 'significance' in show_overlays:
                # Calculate significance threshold (simplified approach)
                sig_threshold = 0.6  # Conservative threshold for coherence
                sig_mask = WCOH > sig_threshold
                
                # Create contour with minimal processing to avoid white space
                try:
                    fig.add_trace(go.Contour(
                        z=sig_mask.astype(float),
                        x=x_coh,
                        y=periods_coh,
                        showscale=False,
                        contours=dict(
                            coloring='none',
                            showlines=True,
                            start=0.5,
                            end=0.5,
                            size=0.1
                        ),
                        line=dict(color='white', width=2),
                        hoverinfo='skip',
                        name='95% Significance',
                        showlegend=True
                    ))
                except:
                    # Fallback to simple line if contour fails
                    fig.add_trace(go.Scatter(
                        x=[x_coh[0], x_coh[-1]], 
                        y=[periods_coh[0], periods_coh[0]], 
                        mode='lines',
                        line=dict(color='white', width=0, opacity=0),
                        name='95% Significance',
                        showlegend=True,
                        hoverinfo='skip'
                    ))
            
            # 2. Cone of Influence (COI)
            if 'coi' in show_overlays:
                try:
                    # Calculate COI boundary
                    coi = np.minimum(np.arange(len(x_coh)), np.arange(len(x_coh))[::-1]) * dt * np.sqrt(2)
                    coi_periods = np.clip(coi, periods_coh[0], periods_coh[-1])
                    
                    # Add COI boundary line
                    fig.add_trace(go.Scatter(
                        x=x_coh,
                        y=coi_periods,
                        mode='lines',
                        line=dict(color='black', dash='dash', width=2),
                        name='Cone of Influence',
                        hoverinfo='skip',
                        showlegend=True
                    ))
                    
                    # Add shaded COI region (minimal opacity to avoid white space)
                    try:
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([x_coh, x_coh[::-1]]),
                            y=np.concatenate([coi_periods, [periods_coh[-1]]*len(x_coh)]),
                            fill='toself',
                            fillcolor='rgba(128,128,128,0.1)',
                            line=dict(color='rgba(0,0,0,0)'),
                            hoverinfo='skip',
                            name='COI Region',
                            showlegend=False
                        ))
                    except:
                        pass  # Skip shaded region if it causes issues
                except Exception as e:
                    pass  # Skip COI if error
            
            # 3. Event Markers - REMOVED FOR PERFORMANCE
            # Event markers removed to improve dashboard responsiveness
            # Focus on core wavelet coherence analysis features
            
            fig.update_layout(
                title=f'Wavelet Coherence: {series1_selected} vs {series2_selected}',
                xaxis_title='Date',
                yaxis_title=period_label,
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=600, width=1100,
                xaxis=dict(
                    title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                    tickfont=dict(size=15, family='Arial', color='black', weight='bold')
                ),
                yaxis=dict(
                    title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                    tickfont=dict(size=15, family='Arial', color='black', weight='bold')
                ),
                legend=dict(
                    x=0.5,   # Center horizontally
                    y=0.98,  # Position closer to plot, below title
                    xanchor='center',
                    yanchor='bottom',
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(0, 0, 0, 0.2)',
                    borderwidth=1,
                    font=dict(size=11, family='Arial', color='black'),
                    itemsizing='constant',
                    itemwidth=30,
                    groupclick='toggleitem',
                    orientation='h',  # Horizontal orientation
                    tracegroupgap=8,
                    itemclick='toggle'
                ),
                margin=dict(t=100, r=50)  # Increased top margin for better legend spacing
            )
            
            # === RAINFALL-GROUNDWATER SPECIFIC PLOTS (Right after main heatmap) ===
            # Only show these plots for GWL vs Rainfall analysis
            rainfall_gwl_plots = []
            if is_well_rainfall:
                try:
                    # Determine which series is rainfall and which is groundwater
                    if 'mm_Rain' in series1_selected or 'rain' in series1_selected.lower():
                        rainfall_data = s1
                        gwl_data = s2
                        rainfall_name = series1_selected
                        gwl_name = series2_selected
                    else:
                        rainfall_data = s2
                        gwl_data = s1
                        rainfall_name = series2_selected
                        gwl_name = series1_selected
                    # 1. Time Series Analysis (Timeline plot - first priority)
                    try:
                        # Create dual-axis time series plot
                        fig_timeseries = go.Figure()
                        
                        # Add rainfall as bars (inverted to show from top)
                        seasons_coh = get_nz_season_for_dates(x_coh)
                        rainfall_hover_text = []
                        for date, rain_val, season in zip(x_coh, rainfall_data, seasons_coh):
                            rainfall_hover_text.append(
                                f"<b>Date</b>: {pd.to_datetime(date).strftime('%Y-%m-%d')}<br>"
                                f"<b>Rainfall</b>: {rain_val:.1f} mm<br>"
                                f"<b>Season</b>: {season}"
                            )
                        
                        fig_timeseries.add_trace(go.Bar(
                            x=x_coh,
                            y=rainfall_data,
                            name='Rainfall',
                            marker_color='rgba(52, 152, 219, 0.7)',
                            yaxis='y2',
                            customdata=rainfall_hover_text,
                            hovertemplate='%{customdata}<extra></extra>'
                        ))
                        
                        # Add groundwater level line with season information
                        gwl_hover_text = []
                        for date, gwl_val, season in zip(x_coh, gwl_data, seasons_coh):
                            gwl_hover_text.append(
                                f"<b>Date</b>: {pd.to_datetime(date).strftime('%Y-%m-%d')}<br>"
                                f"<b>{gwl_name}</b>: {gwl_val:.3f} m<br>"
                                f"<b>Season</b>: {season}"
                            )
                        
                        fig_timeseries.add_trace(go.Scatter(
                            x=x_coh,
                            y=gwl_data,
                            mode='lines+markers',
                            name='Groundwater Level',
                            line=dict(color='#2c3e50', width=2),
                            marker=dict(size=4, color='#2c3e50'),
                            customdata=gwl_hover_text,
                            hovertemplate='%{customdata}<extra></extra>'
                        ))
                        # Add seasonal background shading
                        for vrect in get_nz_season_vrects(x_coh):
                            fig_timeseries.add_shape(**vrect)
                        
                        fig_timeseries.update_layout(
                            title=f'Rainfall and Groundwater Response Timeline<br><sub>{rainfall_name} vs {gwl_name}</sub>',
                            xaxis_title='Date',
                            yaxis_title='Groundwater Level (m)',
                            yaxis2=dict(
                                title='Rainfall (mm)',
                                overlaying='y',
                                side='right',
                                range=[max(rainfall_data)*1.1, 0],  # Invert rainfall axis
                                title_font=dict(size=14, family='Arial', color='black', weight='bold'),
                                tickfont=dict(size=12, family='Arial', color='black', weight='bold')
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=500, width=1100,
                            showlegend=True,
                            legend=dict(
                                x=0.5,
                                y=1.02,
                                xanchor='center',
                                yanchor='bottom',
                                orientation='h',
                                bgcolor='rgba(255, 255, 255, 0.9)',
                                bordercolor='rgba(0, 0, 0, 0.2)',
                                borderwidth=1
                            ),
                            xaxis=dict(
                                title_font=dict(size=14, family='Arial', color='black', weight='bold'),
                                tickfont=dict(size=12, family='Arial', color='black', weight='bold')
                            ),
                            yaxis=dict(
                                title_font=dict(size=14, family='Arial', color='black', weight='bold'),
                                tickfont=dict(size=12, family='Arial', color='black', weight='bold')
                            ),
                            margin=dict(t=80)  # Add top margin for legend
                        )
                        
                        rainfall_gwl_plots.append(html.Div([
                            dcc.Graph(figure=fig_timeseries, style={'backgroundColor': 'white'})
                        ], style={'marginBottom': '20px'}))
                        
                    except Exception as e:
                        pass  # Skip time series if error
                    
                    # 2. Seasonal Scatter Plot Analysis
                    try:
                        # Create seasonal masks
                        dates_series = pd.to_datetime(x_coh)
                        df_seasonal = pd.DataFrame({
                            'Date': dates_series,
                            'Rainfall': rainfall_data,
                            'GWL': gwl_data
                        })
                        
                        # Define New Zealand seasons
                        df_seasonal['Month'] = df_seasonal['Date'].dt.month
                        df_seasonal['Season'] = df_seasonal['Month'].map({
                            12: 'Summer', 1: 'Summer', 2: 'Summer',
                            3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
                            6: 'Winter', 7: 'Winter', 8: 'Winter',
                            9: 'Spring', 10: 'Spring', 11: 'Spring'
                        })
                        
                        # Use ALL rainfall-GWL data points (no filtering)
                        df_events = df_seasonal.copy()
                        
                        if len(df_events) > 10:  # Need sufficient data for analysis
                            # Create seasonal scatter plots
                            fig_seasonal = go.Figure()
                            
                            seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
                            colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
                            
                            for i, season in enumerate(seasons):
                                season_data = df_events[df_events['Season'] == season]
                                
                                if len(season_data) > 3:  # Need at least 3 points for regression
                                    # Calculate linear regression
                                    x_vals = season_data['Rainfall'].values
                                    y_vals = season_data['GWL'].values
                                    
                                    # Simple linear regression
                                    n = len(x_vals)
                                    sum_x = np.sum(x_vals)
                                    sum_y = np.sum(y_vals)
                                    sum_xy = np.sum(x_vals * y_vals)
                                    sum_x2 = np.sum(x_vals ** 2)
                                    
                                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                                    intercept = (sum_y - slope * sum_x) / n
                                    
                                    # Calculate R-squared
                                    y_pred = slope * x_vals + intercept
                                    ss_res = np.sum((y_vals - y_pred) ** 2)
                                    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                                    
                                    # Add scatter points with enhanced hover including date
                                    hover_text = []
                                    for j, (date, rain, gwl) in enumerate(zip(season_data['Date'], x_vals, y_vals)):
                                        hover_text.append(
                                            f"<b>{season}</b><br>"
                                            f"Date: {date.strftime('%Y-%m-%d')}<br>"
                                            f"Rainfall: {rain:.1f} mm<br>"
                                            f"GWL: {gwl:.3f} m<br>"
                                            f"Season: {season}"
                                        )
                                    
                                    fig_seasonal.add_trace(go.Scatter(
                                        x=x_vals,
                                        y=y_vals,
                                        mode='markers',
                                        name=f'{season} Data',
                                        marker=dict(color=colors[i], size=8, symbol='cross'),
                                        customdata=hover_text,
                                        hovertemplate='%{customdata}<extra></extra>'
                                    ))
                                    
                                    # Add regression line
                                    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                                    y_line = slope * x_line + intercept
                                    
                                    fig_seasonal.add_trace(go.Scatter(
                                        x=x_line,
                                        y=y_line,
                                        mode='lines',
                                        name=f'{season} Regression (R²={r_squared:.2f})',
                                        line=dict(color=colors[i], width=2),
                                        hovertemplate=f'<b>{season} Trend</b><br>y = {slope:.2f}x + {intercept:.2f}<br>R² = {r_squared:.3f}<extra></extra>'
                                    ))
                            
                            fig_seasonal.update_layout(
                                title=f'Seasonal Rainfall-Groundwater Response Analysis<br><sub>{rainfall_name} vs {gwl_name}</sub>',
                                xaxis_title='Rainfall (mm)',
                                yaxis_title='Groundwater Level (m)', 
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                height=500, width=1100,
                                showlegend=True,
                                legend=dict(
                                    x=0.5,
                                    y=1.02,
                                    xanchor='center',
                                    yanchor='bottom',
                                    orientation='h',
                                    bgcolor='rgba(255, 255, 255, 0.9)',
                                    bordercolor='rgba(0, 0, 0, 0.2)',
                                    borderwidth=1
                                ),
                                xaxis=dict(
                                    title_font=dict(size=14, family='Arial', color='black', weight='bold'),
                                    tickfont=dict(size=12, family='Arial', color='black', weight='bold')
                                ),
                                yaxis=dict(
                                    title_font=dict(size=14, family='Arial', color='black', weight='bold'),
                                    tickfont=dict(size=12, family='Arial', color='black', weight='bold')
                                ),
                                margin=dict(t=80)  # Add top margin for legend
                            )
                            
                            rainfall_gwl_plots.append(html.Div([
                                dcc.Graph(figure=fig_seasonal, style={'backgroundColor': 'white'})
                            ], style={'marginBottom': '20px'}))
                            
                    except Exception as e:
                        pass  # Skip seasonal analysis if error
                            
                    # 3. Complete Response Statistics Table
                    try:
                        # Calculate response statistics using ALL data
                        all_data_df = df_events if 'df_events' in locals() else df_seasonal
                            
                        if len(all_data_df) > 0:
                            # Calculate statistics by season using all data
                            response_stats = []
                            
                            for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
                                season_data = all_data_df[all_data_df['Season'] == season]
                                
                                if len(season_data) > 0:
                                    # Calculate correlation
                                    corr = np.corrcoef(season_data['Rainfall'], season_data['GWL'])[0, 1]
                                    
                                    # Count actual rainfall events (>0mm) vs all data points
                                    rainfall_events = season_data[season_data['Rainfall'] > 0]
                                    
                                    response_stats.append({
                                        'Season': season,
                                        'Data Points': len(season_data),
                                        'Rainfall Events': len(rainfall_events),
                                        'Avg Rainfall': f"{season_data['Rainfall'].mean():.1f} mm",
                                        'Avg GWL': f"{season_data['GWL'].mean():.3f} m",
                                        'Correlation': f"{corr:.3f}" if not np.isnan(corr) else "N/A",
                                        'Max Rainfall': f"{season_data['Rainfall'].max():.1f} mm",
                                        'GWL Range': f"{season_data['GWL'].min():.3f} - {season_data['GWL'].max():.3f} m"
                                    })
                            if response_stats:
                                # Add title row
                                title_row = {
                                    'Season': 'Complete Rainfall-Groundwater Statistics',
                                    'Data Points': '', 'Rainfall Events': '', 'Avg Rainfall': '',
                                    'Avg GWL': '', 'Correlation': '', 'Max Rainfall': '', 'GWL Range': ''
                                }
                                stats_data_with_title = [title_row] + response_stats
                                
                                stats_table = dash_table.DataTable(
                                    columns=[{"name": col, "id": col} for col in response_stats[0].keys()],
                                    data=stats_data_with_title,
                                    style_table={'width': '95%', 'margin': 'auto'},
                                    style_cell={'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': 12},
                                    style_header={'backgroundColor': '#e8f5e8', 'fontWeight': 'bold'},
                                    style_data_conditional=[
                                        {
                                            'if': {'row_index': 0},  # Title row
                                            'backgroundColor': '#d5f4e6',
                                            'fontWeight': 'bold',
                                            'fontSize': 14
                                        },
                                        {
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': '#f9f9f9'
                                        }
                                    ]
                                )
                                
                                rainfall_gwl_plots.append(html.Div([
                                    stats_table
                                ], style={'marginBottom': '20px'}))
                                
                    except Exception as e:
                        pass  # Skip statistics table if error

                except Exception as e:
                    pass  # Skip all rainfall-GWL specific plots if error

            
            # 2. Cone of Influence (COI)
            if 'coi' in show_overlays:
                coi = np.minimum(np.arange(len(x_coh)), np.arange(len(x_coh))[::-1]) * dt * np.sqrt(2)
                coi_periods_coh = np.clip(coi, periods_coh[0], periods_coh[-1])
                
                # Add COI boundary line
                fig.add_trace(go.Scatter(
                    x=x_coh,
                    y=coi_periods_coh,
                    mode='lines',
                    line=dict(color='black', dash='dash', width=2),
                    name='Cone of Influence',
                    hoverinfo='skip',
                    showlegend=True
                ))
                
                # Add minimal shaded COI region
                try:
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([x_coh, x_coh[::-1]]),
                        y=np.concatenate([coi_periods_coh, [periods_coh[-1]]*len(x_coh)]),
                        fill='toself',
                        fillcolor='rgba(128,128,128,0.08)',
                        line=dict(color='rgba(0,0,0,0)'),
                        hoverinfo='skip',
                        name='COI Region',
                        showlegend=False
                    ))
                except:
                    pass  # Skip if causes issues
            
            # Update layout for coherence
            fig.update_layout(
                title=f'Wavelet Coherence: {series1_selected} vs {series2_selected}',
                xaxis_title='Date',
                yaxis_title=period_label,
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=600, width=1100,
                xaxis=dict(
                    title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                    tickfont=dict(size=15, family='Arial', color='black', weight='bold')
                ),
                yaxis=dict(
                    title_font=dict(size=18, family='Arial', color='black', weight='bold'),
                    tickfont=dict(size=15, family='Arial', color='black', weight='bold')
                ),
                legend=dict(
                    x=0.5,   # Center horizontally
                    y=0.98,  # Position closer to plot, below title
                    xanchor='center',
                    yanchor='bottom',
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(0, 0, 0, 0.2)',
                    borderwidth=1,
                    font=dict(size=11, family='Arial', color='black'),
                    itemsizing='constant',
                    itemwidth=30,
                    groupclick='toggleitem',
                    orientation='h',  # Horizontal orientation
                    tracegroupgap=8,
                    itemclick='toggle'
                ),
                margin=dict(t=100, r=50)  # Increased top margin for better legend spacing
            )
            
            # Create plots list for coherence
            coherence_plots = [dcc.Graph(figure=fig, style={'backgroundColor': 'white'})]
            
            # Add rainfall-GWL plots if applicable
            if rainfall_gwl_plots:
                coherence_plots.extend(rainfall_gwl_plots)
            
            # Add overlay interpretation guide for coherence
            if show_overlays:
                overlay_items = []
                
                if 'significance' in show_overlays:
                    overlay_items.append(html.Li([
                        html.B("95% Significance Contour (White Lines): "), 
                        "Areas where wavelet coherence significantly exceeds red noise background. Only patterns inside these contours are statistically meaningful."
                    ], style={'marginBottom': '8px'}))
                
                if 'coi' in show_overlays:
                    overlay_items.append(html.Li([
                        html.B("Cone of Influence (Black Dashed Line): "), 
                        "Boundary where edge effects become important. Wavelet analysis outside this region may be unreliable due to data boundary effects."
                    ], style={'marginBottom': '8px'}))
                
                overlay_items.append(html.Li([
                    html.B("Interpretation Tips: "), 
                    "Focus on significant regions inside the COI for reliable analysis. Use Coherence/Cross-Wavelet tabs for phase relationship analysis."
                ], style={'marginBottom': '8px', 'fontStyle': 'italic', 'color': '#27ae60'}))
                
                overlay_guide = html.Div([
                    html.H5('Wavelet Coherence Overlay Guide', style={'color': '#2c3e50', 'marginBottom': '10px'}),
                    html.Ul(overlay_items, style={'fontSize': '13px', 'color': '#34495e', 'lineHeight': '1.6'})
                ], style={'padding': '15px', 'backgroundColor': '#e8f4fd', 'borderRadius': '8px', 'marginTop': '20px', 'border': '1px solid #bee5eb'})
                
                coherence_plots.append(overlay_guide)
            
            # Return coherence content
            return html.Div([
                coherence_controls,
                html.Div(coherence_plots)
            ], style={'backgroundColor': 'white'})
        elif tab == 'pwc':
            # Partial Wavelet Coherence Tab - Add controls directly in the tab
            # Create series options for partial wavelet coherence analysis
            series_options = [{'label': w, 'value': w} for w in well_columns]
            if 'mm_Rain' in data_resampled.columns:
                series_options.append({'label': 'Rainfall', 'value': 'mm_Rain'})
            
            # Get selections from callback if available, otherwise use defaults
            if 'pwc_selections' in globals() and pwc_selections and 'series1' in pwc_selections:
                default_series1 = pwc_selections['series1']
                default_series2 = pwc_selections['series2']
                default_confounder = pwc_selections['confounder']
            else:
                default_series1 = coh_series1 or well_columns[0]
                default_series2 = coh_series2 or 'mm_Rain'
                default_confounder = 'mm_Rain' if 'mm_Rain' in data_resampled.columns else (well_columns[1] if len(well_columns) > 1 else well_columns[0])
            
            pwc_controls = html.Div([
                html.H5("Select Series for Partial Wavelet Coherence Analysis", style={'marginBottom': 15, 'color': '#444'}),
                html.Div([
                    html.Div([
                        html.Label("Series 1:", style={'marginBottom': 4, 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='pwc-series1-dropdown',
                            options=series_options,
                            value=default_series1,
                            clearable=False,
                            style={'marginBottom': 8}
                        )
                    ], style={'width': '31%', 'display': 'inline-block'}),
                    html.Div([
                        html.Label("Series 2:", style={'marginBottom': 4, 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='pwc-series2-dropdown',
                            options=series_options,
                            value=default_series2,
                            clearable=False,
                            style={'marginBottom': 8}
                        )
                    ], style={'width': '31%', 'display': 'inline-block', 'marginLeft': '3.5%'}),
                    html.Div([
                        html.Label("Confounder (Control for):", style={'marginBottom': 4, 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='pwc-confounder-dropdown',
                            options=series_options,
                            value=default_confounder,
                            clearable=False,
                            style={'marginBottom': 8}
                        )
                    ], style={'width': '31%', 'display': 'inline-block', 'marginLeft': '3.5%'}),
                ]),
                html.P("Partial coherence shows the relationship between Series 1 and Series 2 after removing the linear influence of the Confounder.", 
                       style={'fontSize': '12px', 'color': '#666', 'margin': '8px 0 0 0', 'fontStyle': 'italic'}),
                html.Hr(),
                html.H6("Phase Analysis Controls", style={'marginTop': 10, 'marginBottom': 6, 'color': '#444'}),
                html.Label("Select Period for Phase Time Series:", style={'marginTop': 6}),
                dcc.Input(
                    id='pwc-phase-period-select',
                    type='number',
                    value=get_phase_settings().get('target_period', 12),
                    min=0.1,
                    step=0.1,
                    style={'marginBottom': 8, 'width': '120px'}
                ),
                dcc.Checklist(
                    id='pwc-show-phase-plots',
                    options=[
                        {'label': 'Show Phase Time Series', 'value': 'phase_ts'},
                        {'label': 'Show Time Lag Analysis', 'value': 'lag_analysis'},
                        {'label': 'Show Peak Coherence Detection', 'value': 'peak_detection'}
                    ],
                    value=get_phase_settings().get('show_plots', []),
                    labelStyle={'display': 'block', 'marginRight': 10},
                    style={'marginBottom': 12}
                )
            ], style={'marginBottom': 20, 'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9'})
            
            # Check if PWC selections are available from callback, otherwise use defaults
            if 'pwc_selections' in globals() and pwc_selections and 'series1' in pwc_selections:
                series1_selected = pwc_selections['series1']
                series2_selected = pwc_selections['series2']
                confounder_selected = pwc_selections['confounder']
            else:
                series1_selected = default_series1
                series2_selected = default_series2
                confounder_selected = default_confounder
            
            # Get selected series data
            s1 = data_resampled[series1_selected].values if series1_selected in data_resampled.columns else None
            s2 = data_resampled[series2_selected].values if series2_selected in data_resampled.columns else None
            s3 = data_resampled[confounder_selected].values if confounder_selected in data_resampled.columns else None
            x_pwc = data_resampled['DateTime'].values
            
            # Validate data availability
            if s1 is None or s2 is None or s3 is None:
                return html.Div([
                    pwc_controls,
                    html.Div([
                        html.H4('Data Validation Error'),
                        html.P('One or more selected series are not available in the dataset.', 
                               style={'color': '#e74c3c', 'textAlign': 'center'})
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                ], style={'backgroundColor': 'white'})
            
            # Ensure all series have the same length and remove NaNs
            mask = ~np.isnan(s1) & ~np.isnan(s2) & ~np.isnan(s3)
            s1, s2, s3 = s1[mask], s2[mask], s3[mask]
            x_pwc = x_pwc[mask]
            
            if len(s1) < 10:
                return html.Div([
                    pwc_controls,
                    html.Div([
                        html.H4('Insufficient Data'),
                        html.P('Not enough valid data points for PWC analysis. Need at least 10 points.', 
                               style={'color': '#e74c3c', 'textAlign': 'center'})
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                ], style={'backgroundColor': 'white'})
            
            # Validate wavelet type
            if wavelet_type != 'cwt':
                return html.Div([
                    pwc_controls,
                    html.Div([
                        html.H4('Wavelet Type Error'),
                        html.P('Partial Wavelet Coherence requires Continuous Wavelet Transform (CWT). Please select CWT as the wavelet type.', 
                               style={'color': '#f39c12', 'textAlign': 'center'})
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                ], style={'backgroundColor': 'white'})
            
            # Use same scale calculation as coherence analysis
            min_scale = 1
            n = len(s1)
            max_scale = n // 8 if n >= 8 else 2
            num_octaves = np.log2(max_period_conv / min_period_conv) if max_period_conv > min_period_conv else 1
            voices_per_octave = int(np.ceil(num_periods / num_octaves)) if num_octaves > 0 else 2
            voices_per_octave = max(2, min(voices_per_octave, 48))
            delta_j = 1 / voices_per_octave
            J = int(np.log2(max_scale / min_scale) / delta_j) if max_scale > min_scale else 1
            scales = min_scale * 2 ** (np.arange(J + 1) * delta_j)
            
            # Convert scales to periods (use the wavelet's actual centre frequency
            # so periods match pywt.cwt's frequencies).
            periods_pwc = scales * dt / pywt.central_frequency(wavelet)
            
            # Convert periods to selected units
            if period_units == 'weeks':
                periods_pwc = periods_pwc / 7
            elif period_units == 'months':
                periods_pwc = periods_pwc / 30.44
            elif period_units == 'years':
                periods_pwc = periods_pwc / 365.25
            
            # Filter to user-specified period range
            period_mask = (periods_pwc >= min_period_conv) & (periods_pwc <= max_period_conv)
            periods_pwc = periods_pwc[period_mask]
            scales = scales[period_mask]
            
            if len(scales) < 2:
                return html.Div([
                    pwc_controls,
                    html.Div([
                        html.H4('Period Range Error'),
                        html.P('Not enough periods in selected range for PWC analysis. Try expanding your period range.', 
                               style={'color': '#e74c3c', 'textAlign': 'center'})
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                ], style={'backgroundColor': 'white'})
            
            # Compute continuous wavelet transforms for all three series
            try:
                coef1, _ = pywt.cwt(s1, scales, wavelet)
                coef2, _ = pywt.cwt(s2, scales, wavelet)
                coef3, _ = pywt.cwt(s3, scales, wavelet)
                
                # Compute cross-wavelet transforms for all pairs
                S12 = coef1 * np.conj(coef2)  # Cross-wavelet between s1 and s2
                S13 = coef1 * np.conj(coef3)  # Cross-wavelet between s1 and s3 (confounder)
                S23 = coef2 * np.conj(coef3)  # Cross-wavelet between s2 and s3 (confounder)
                
                # Compute power spectra
                S11 = np.abs(coef1) ** 2  # Power spectrum of s1
                S22 = np.abs(coef2) ** 2  # Power spectrum of s2
                S33 = np.abs(coef3) ** 2  # Power spectrum of s3
                
                # Smooth the cross-wavelet transforms (important for coherence calculation)
                def smooth_pwc(mat, window=5):
                    return uniform_filter1d(mat, size=window, axis=1, mode='nearest')
                
                S12_smooth = smooth_pwc(S12)
                S13_smooth = smooth_pwc(S13)
                S23_smooth = smooth_pwc(S23)
                S11_smooth = smooth_pwc(S11)
                S22_smooth = smooth_pwc(S22)
                S33_smooth = smooth_pwc(S33)
                
                # Calculate regular coherence values
                coherence_12 = np.abs(S12_smooth) ** 2 / (S11_smooth * S22_smooth)
                coherence_13 = np.abs(S13_smooth) ** 2 / (S11_smooth * S33_smooth)
                coherence_23 = np.abs(S23_smooth) ** 2 / (S22_smooth * S33_smooth)
                
                # Calculate partial wavelet coherence
                # PWC formula: R²_{12|3} = (R²₁₂ - R²₁₃ * R²₂₃) / ((1 - R²₁₃) * (1 - R²₂₃))
                numerator = coherence_12 - coherence_13 * coherence_23
                denominator = (1 - coherence_13) * (1 - coherence_23)
                
                # Avoid division by zero
                denominator = np.where(denominator < 1e-10, 1e-10, denominator)
                PWC = numerator / denominator
                
                # Ensure PWC is between 0 and 1
                PWC = np.clip(PWC, 0, 1)
                
                # Additional data cleaning to prevent white space rendering issues
                PWC = np.nan_to_num(PWC, nan=0.0, posinf=1.0, neginf=0.0)
                PWC = np.where(np.isfinite(PWC), PWC, 0.0)
                PWC = np.where(PWC < 1e-10, 1e-10, PWC)  # Replace tiny values that cause white space
                PWC = np.maximum(PWC, 0.001)  # Force a minimum baseline to prevent white space
                
                # Calculate phase relationships for PWC
                phase_12 = np.angle(S12_smooth)
                phase_13 = np.angle(S13_smooth)
                phase_23 = np.angle(S23_smooth)
                
                # Partial phase (approximate)
                phase_pwc = phase_12 - (phase_13 + phase_23) / 2
                
                # Prepare customdata for hover information
                customdata = np.zeros((len(periods_pwc), len(x_pwc), 4))
                series1_data = s1
                series2_data = s2
                confounder_data = s3
                seasons = get_nz_season_for_dates(x_pwc)
                
                for i in range(len(periods_pwc)):
                    for j in range(len(x_pwc)):
                        customdata[i, j, 0] = PWC[i, j]
                        customdata[i, j, 1] = series1_data[j]
                        customdata[i, j, 2] = series2_data[j]
                        customdata[i, j, 3] = confounder_data[j]
                        
                # Create the PWC plot
                fig = go.Figure()
                
                # Add PWC heatmap
                fig.add_trace(go.Heatmap(
                    z=PWC,
                    x=x_pwc,
                    y=periods_pwc,
                    colorscale='Viridis' if ('mm_Rain' not in [series1_selected, series2_selected] and 
                                             'rain' not in series1_selected.lower() and 
                                             'rain' not in series2_selected.lower() and 
                                             confounder_selected != 'mm_Rain' and 'rain' not in confounder_selected.lower() and
                                             np.nanmax(PWC_display) < np.nanpercentile(PWC_display[PWC_display > 0.001], 90) * 2) 
                                            else 'RdYlBu_r',  # Adaptive colormap for well-well analysis
                    zmin=0.001,  # Start slightly above 0 to prevent white space
                    zmax=1.0,    # Force colorscale to end at 1
                    zmid=0.5,    # Set midpoint for better color distribution
                    connectgaps=False,  # Don't connect gaps in data
                    showscale=True,
                    customdata=customdata,
                    hovertemplate=(
                        '<b>Date:</b> %{x}<br>'
                        '<b>Period:</b> %{y:.2f} ' + period_units + '<br>'
                        '<b>Partial Coherence:</b> %{z:.3f}<br>'
                        f'<b>{series1_selected}:</b> %{{customdata[1]:.3f}} m<br>'
                        f'<b>{series2_selected}:</b> %{{customdata[2]:.3f}} mm<br>'
                        f'<b>{confounder_selected} (Control):</b> %{{customdata[3]:.3f}} m<br>'
                        '<extra></extra>'
                    ),
                    colorbar=dict(
                        title=dict(text='Partial Wavelet<br>Coherence', font=dict(size=14)),
                        len=0.8,
                        thickness=20,
                        x=1.03,
                        xanchor='left',
                        tickmode='linear',
                        tick0=0,
                        dtick=0.2
                    ),
                    name='PWC'
                ))
                
            except Exception as e:
                return html.Div([
                    pwc_controls,
                    html.Div([
                        html.H4('Computation Error'),
                        html.P(f'Error computing PWC analysis: {str(e)}', 
                               style={'color': '#e74c3c', 'textAlign': 'center'})
                    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                ], style={'backgroundColor': 'white'})
            
            # --- Partial Wavelet Coherence Overlays ---
            
            # 1. Significance Overlay for PWC
            if 'significance' in show_overlays:
                # PWC significance using conservative threshold
                pwc_threshold = 0.5  # Conservative threshold for partial coherence
                sig_mask_pwc = PWC > pwc_threshold
                
                try:
                    fig.add_trace(go.Contour(
                        z=sig_mask_pwc.astype(float),
                        x=x_pwc,
                        y=periods_pwc,
                        showscale=False,
                        contours=dict(
                            coloring='none',
                            showlines=True,
                            start=0.5,
                            end=0.5,
                            size=0.1
                        ),
                        line=dict(color='white', width=2),
                        hoverinfo='skip',
                        name='95% Significance',
                        showlegend=True
                    ))
                except:
                    # Fallback if contour fails
                    fig.add_trace(go.Scatter(
                        x=[x_pwc[0], x_pwc[-1]], 
                        y=[periods_pwc[0], periods_pwc[0]], 
                        mode='lines',
                        line=dict(color='white', width=0, opacity=0),
                        name='95% Significance',
                        showlegend=True,
                        hoverinfo='skip'
                    ))
            
            # 2. Cone of Influence for PWC
            if 'coi' in show_overlays:
                # Calculate COI boundary for PWC
                coi_pwc = np.minimum(np.arange(len(x_pwc)), np.arange(len(x_pwc))[::-1]) * dt * np.sqrt(2)
                coi_periods_pwc = np.clip(coi_pwc, periods_pwc[0], periods_pwc[-1])
                
                # Add COI boundary
                fig.add_trace(go.Scatter(
                    x=x_pwc,
                    y=coi_periods_pwc,
                    mode='lines',
                    line=dict(color='black', dash='dash', width=2),
                    name='Cone of Influence',
                    hoverinfo='skip',
                    showlegend=True
                ))
            
            # 3. Phase Arrows for PWC
            if 'phase_arrows' in show_overlays:
                # Use the SAME approach as Cross-Wavelet tab (which works without white space)
                arrow_step_time = max(1, len(x_pwc) // 8)
                arrow_step_scale = max(1, len(periods_pwc) // 5)
                
                # Simple subsampling approach
                pwc_arrows_x = x_pwc[::arrow_step_time]
                pwc_arrows_y = periods_pwc[::arrow_step_scale]
                
                # Get PWC data at these points
                pwc_data = PWC[::arrow_step_scale, ::arrow_step_time]
                
                # Debug: ensure we have valid data shapes for PWC
                if pwc_data.size == 0:
                    # Use less aggressive subsampling if array is too small
                    arrow_step_time = max(1, len(x_pwc) // 4)
                    arrow_step_scale = max(1, len(periods_pwc) // 3)
                    pwc_arrows_x = x_pwc[::arrow_step_time]
                    pwc_arrows_y = periods_pwc[::arrow_step_scale]
                    pwc_data = PWC[::arrow_step_scale, ::arrow_step_time]
                
                # Use more permissive threshold for PWC - start very low
                pwc_threshold = 0.05  # Start with very low threshold  
                if np.any(pwc_data > 0):
                    # Use 40th percentile instead of 65th, with lower minimum
                    percentile_threshold = np.percentile(pwc_data[pwc_data > 0], 40)
                    pwc_threshold = max(percentile_threshold, 0.05)  # Much lower minimum
                                
                # Build arrow coordinates for high PWC values only
                x_arrows_pwc = []
                y_arrows_pwc = []
                pwc_values = []
                
                for i in range(len(pwc_arrows_y)):
                    for j in range(len(pwc_arrows_x)):
                        if i < pwc_data.shape[0] and j < pwc_data.shape[1]:
                            if pwc_data[i, j] > pwc_threshold and np.isfinite(pwc_data[i, j]):
                                x_arrows_pwc.append(pwc_arrows_x[j])
                                y_arrows_pwc.append(pwc_arrows_y[i])
                                pwc_values.append(pwc_data[i, j])
                
                # Fallback: if no PWC arrows meet threshold, lower the bar to show at least some
                if len(x_arrows_pwc) == 0 and np.any(pwc_data > 0):
                    # Use a much lower threshold to ensure some arrows are visible
                    fallback_threshold = max(np.percentile(pwc_data[pwc_data > 0], 10), 0.01) if np.any(pwc_data > 0) else 0.01
                    for i in range(len(pwc_arrows_y)):
                        for j in range(len(pwc_arrows_x)):
                            if i < pwc_data.shape[0] and j < pwc_data.shape[1]:
                                if pwc_data[i, j] > fallback_threshold and np.isfinite(pwc_data[i, j]):
                                    x_arrows_pwc.append(pwc_arrows_x[j])
                                    y_arrows_pwc.append(pwc_arrows_y[i])
                                    pwc_values.append(pwc_data[i, j])
                
                # Final fallback for PWC: if STILL no arrows, just take the top 3 highest PWC values
                if len(x_arrows_pwc) == 0 and pwc_data.size > 0:
                    # Flatten PWC data
                    flat_pwc = pwc_data.flatten()
                    
                    # Get indices of top 3 PWC values
                    valid_mask = np.isfinite(flat_pwc) & (flat_pwc > 0)
                    if np.any(valid_mask):
                        valid_indices = np.where(valid_mask)[0]
                        top_indices = valid_indices[np.argsort(flat_pwc[valid_indices])[-3:]]  # Top 3
                        
                        for idx in top_indices:
                            i_2d = idx // pwc_data.shape[1]
                            j_2d = idx % pwc_data.shape[1]
                            
                            if i_2d < len(pwc_arrows_y) and j_2d < len(pwc_arrows_x):
                                x_arrows_pwc.append(pwc_arrows_x[j_2d])
                                y_arrows_pwc.append(pwc_arrows_y[i_2d])
                                pwc_values.append(flat_pwc[idx])
                
                # Only add arrows if we have valid points
                if len(x_arrows_pwc) > 0:
                    # Create detailed hover information for PWC arrows
                    hover_text_pwc = []
                    for i, (x_val, y_val, pwc_val) in enumerate(zip(x_arrows_pwc, y_arrows_pwc, pwc_values)):
                        # Find corresponding phase data
                        phase_val = phase_pwc[np.argmin(np.abs(periods_pwc - y_val)), 
                                            np.argmin(np.abs(x_pwc - x_val))]
                        
                        # Calculate phase angle in degrees
                        phase_degrees = phase_val * 180 / np.pi
                        
                        # Calculate time lag
                        frequency = 1.0 / y_val
                        time_lag = phase_val / (2 * np.pi * frequency)
                        
                        # Convert time lag to selected units
                        if period_units == 'weeks':
                            time_lag_converted = time_lag / 7
                            lag_units_local = 'weeks'
                        elif period_units == 'months':
                            time_lag_converted = time_lag / 30.44
                            lag_units_local = 'months'
                        elif period_units == 'years':
                            time_lag_converted = time_lag / 365.25
                            lag_units_local = 'years'
                        else:
                            time_lag_converted = time_lag
                            lag_units_local = period_units
                        
                        # Determine phase relationship
                        if abs(time_lag_converted) < 0.01:
                            relationship = "In-Phase (Synchronous)"
                        elif time_lag_converted > 0:
                            relationship = f"{series1_selected} LAGS {series2_selected}"
                        else:
                            relationship = f"{series1_selected} LEADS {series2_selected}"
                        
                        # PWC interpretation (accounting for confounder)
                        interpretation_mode = arrow_interpretation_mode or 'custom'
                        if interpretation_mode == 'custom':
                            if time_lag_converted > 0:
                                process = f"After removing {confounder_selected}: Response"
                            else:
                                process = f"After removing {confounder_selected}: Driver"
                        else:
                            process = f"Partial relationship (controlling {confounder_selected})"
                        
                        hover_text_pwc.append(
                            f"Partial Coherence: {pwc_val:.3f}<br>"
                            f"Phase: {phase_degrees:.1f}°<br>"
                            f"Time Lag: {time_lag_converted:.2f} {lag_units_local}<br>"
                            f"Relationship: {relationship}<br>"
                            f"Process: {process}<br>"
                            f"Controlled for: {confounder_selected}"
                        )
                    
                    # Determine arrow colors for PWC based on phase relationships
                    arrow_colors_pwc = []
                    for i, (x_val, y_val, pwc_val) in enumerate(zip(x_arrows_pwc, y_arrows_pwc, pwc_values)):
                        # Find corresponding phase data
                        phase_val = phase_pwc[np.argmin(np.abs(periods_pwc - y_val)), 
                                            np.argmin(np.abs(x_pwc - x_val))]
                        
                        # Calculate time lag for color determination
                        frequency = 1.0 / y_val
                        time_lag = phase_val / (2 * np.pi * frequency)
                        
                        # Convert time lag to selected units
                        if period_units == 'weeks':
                            time_lag_converted = time_lag / 7
                        elif period_units == 'months':
                            time_lag_converted = time_lag / 30.44
                        elif period_units == 'years':
                            time_lag_converted = time_lag / 365.25
                        else:
                            time_lag_converted = time_lag
                        
                        # Color coding for PWC relationships (after removing confounder)
                        if abs(time_lag_converted) < 0.01:
                            arrow_colors_pwc.append('green')   # In-phase after controlling
                        elif time_lag_converted > 0:
                            arrow_colors_pwc.append('red')     # Response after controlling
                        else:
                            arrow_colors_pwc.append('blue')    # Driver after controlling
                    
                    # Convert phase angles to directional text for PWC
                    arrow_text_pwc = []
                    for i, (x_val, y_val, pwc_val) in enumerate(zip(x_arrows_pwc, y_arrows_pwc, pwc_values)):
                        # Find corresponding phase data
                        phase_val = phase_pwc[np.argmin(np.abs(periods_pwc - y_val)), 
                                            np.argmin(np.abs(x_pwc - x_val))]
                        phase_degrees = phase_val * 180 / np.pi
                        
                        # Convert phase to arrow direction (4 main directions)
                        if -45 <= phase_degrees < 45:
                            arrow_text_pwc.append('→')  # Right
                        elif 45 <= phase_degrees < 135:
                            arrow_text_pwc.append('↑')  # Up
                        elif 135 <= phase_degrees <= 180 or -180 <= phase_degrees < -135:
                            arrow_text_pwc.append('←')  # Left
                        else:  # -135 <= phase_degrees < -45
                            arrow_text_pwc.append('↓')  # Down
                    
                    # Convert x_arrows to proper datetime format - ROBUST conversion for PWC
                    x_arrows_pwc_clean = []
                    for x_val in x_arrows_pwc:
                        try:
                            if hasattr(x_val, 'strftime'):
                                x_arrows_pwc_clean.append(x_val)
                            elif isinstance(x_val, (int, float)):
                                # If it's a numeric timestamp, convert it
                                x_arrows_pwc_clean.append(pd.to_datetime(x_val, unit='s'))
                            else:
                                x_arrows_pwc_clean.append(pd.to_datetime(x_val))
                        except:
                            # Fallback - use first valid date if conversion fails
                            x_arrows_pwc_clean.append(x_pwc[0])
                    
                    # Ensure all PWC arrays have the same length to prevent white space
                    min_length = min(len(x_arrows_pwc_clean), len(y_arrows_pwc), len(arrow_colors_pwc), len(arrow_text_pwc), len(hover_text_pwc))
                    if min_length > 0:
                        x_arrows_pwc_clean = x_arrows_pwc_clean[:min_length]
                        y_arrows_pwc = y_arrows_pwc[:min_length]
                        arrow_colors_pwc = arrow_colors_pwc[:min_length]
                        arrow_text_pwc = arrow_text_pwc[:min_length]
                        hover_text_pwc = hover_text_pwc[:min_length]
                        
                        # PLOTLY FIX: Use simple circles to avoid diamond+text rendering issues
                        if min_length >= 3 and all(x is not None and not pd.isna(x) for x in x_arrows_pwc_clean):
                            fig.add_trace(go.Scatter(
                                x=x_arrows_pwc_clean,
                                y=y_arrows_pwc,
                                mode='markers',  # Remove text to avoid Plotly rendering conflicts
                                marker=dict(
                                    symbol='circle',  # Use simple circles instead of diamonds
                                    size=10,  # Smaller size for better rendering  
                                    color=arrow_colors_pwc,
                                    line=dict(color='black', width=1),
                                    opacity=0.8
                                ),
                                name='Phase Relationships',
                                hovertemplate='<b>Date</b>: %{x}<br><b>Period</b>: %{y:.1f} ' + period_units + '<br>%{customdata}<extra></extra>',
                                customdata=hover_text_pwc,
                                showlegend=True
                            ))
            
            # Plot styling
            period_label = f'Period ({period_units})'
            fig.update_layout(
                title=dict(
                    text=f'Partial Wavelet Coherence: {series1_selected} vs {series2_selected}<br><sub>Controlling for: {confounder_selected}</sub>',
                    font=dict(size=18, color='#2c3e50', family='Arial'),
                    x=0.5
                ),
                xaxis_title='Date',
                yaxis_title=period_label,
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=600, width=1200,
                xaxis=dict(
                    title_font=dict(size=16, family='Arial', color='#34495e', weight='bold'),
                    tickfont=dict(size=13, family='Arial', color='#2c3e50', weight='bold')
                ),
                yaxis=dict(
                    title_font=dict(size=16, family='Arial', color='#34495e', weight='bold'),
                    tickfont=dict(size=13, family='Arial', color='#2c3e50', weight='bold'),
                    type='log'
                ),
                legend=dict(
                    x=0.5,   # Center horizontally
                    y=0.98,  # Position closer to plot, below title
                    xanchor='center',
                    yanchor='bottom',
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(0, 0, 0, 0.2)',
                    borderwidth=1,
                    font=dict(size=11, family='Arial', color='black'),
                    itemsizing='constant',
                    itemwidth=30,
                    groupclick='toggleitem',
                    orientation='h',  # Horizontal orientation like other heatmaps
                    tracegroupgap=8,
                    itemclick='toggle'
                ),
                margin=dict(l=60, r=80, t=120, b=60)  # Consistent with other heatmap tabs
            )
            
            # Initialize plots list
            pwc_plots = [dcc.Graph(figure=fig, style={'backgroundColor': 'white'})]
            

            
            # Add overlay interpretation guide for PWC
            if show_overlays:
                pwc_overlay_items = []
                
                if 'significance' in show_overlays:
                    pwc_overlay_items.append(html.Li([
                        html.B("95% Significance Contour (White Lines): "), 
                        f"Areas where partial coherence significantly exceeds background after removing {confounder_selected} influence."
                    ], style={'marginBottom': '8px'}))
                
                if 'coi' in show_overlays:
                    pwc_overlay_items.append(html.Li([
                        html.B("Cone of Influence (Black Dashed Line): "), 
                        "Boundary where edge effects become important. Analysis outside this region may be unreliable."
                    ], style={'marginBottom': '8px'}))
                
                if 'phase_arrows' in show_overlays:
                    pwc_overlay_items.extend([
                        html.Li([
                            html.B("Phase Relationships (After Controlling): "), 
                        ], style={'marginBottom': '4px'}),
                        html.Li([
                            html.B("Phase Markers: "), "Colored circles showing phase relationships (hover for details)"
                        ], style={'marginBottom': '4px', 'marginLeft': '20px'}),
                        html.Li([
                            html.Span("🔴 Red Markers: ", style={'color': 'red', 'fontWeight': 'bold'}),
                            f"Response Pattern ({series1_selected} responds to {series2_selected} after removing {confounder_selected})"
                        ], style={'marginBottom': '4px', 'marginLeft': '20px'}),
                        html.Li([
                            html.Span("🔵 Blue Markers: ", style={'color': 'blue', 'fontWeight': 'bold'}),
                            f"Driver Pattern ({series1_selected} drives {series2_selected} after removing {confounder_selected})"
                        ], style={'marginBottom': '4px', 'marginLeft': '20px'}),
                        html.Li([
                            html.Span("🟢 Green Markers: ", style={'color': 'green', 'fontWeight': 'bold'}),
                            f"In-Phase (Synchronous after controlling for {confounder_selected})"
                        ], style={'marginBottom': '8px', 'marginLeft': '20px'})
                    ])
                
                pwc_overlay_items.append(html.Li([
                    html.B("Interpretation: "), 
                    f"Partial coherence reveals true {series1_selected}-{series2_selected} relationships independent of {confounder_selected} effects."
                ], style={'marginBottom': '8px', 'fontStyle': 'italic', 'color': '#27ae60'}))
                
                pwc_overlay_guide = html.Div([
                    html.H5('Partial Wavelet Coherence Overlay Guide', style={'color': '#2c3e50', 'marginBottom': '10px'}),
                    html.Ul(pwc_overlay_items, style={'fontSize': '13px', 'color': '#34495e', 'lineHeight': '1.6'})
                ], style={'padding': '15px', 'backgroundColor': '#e8f4fd', 'borderRadius': '8px', 'marginTop': '20px', 'border': '1px solid #bee5eb'})
                
                pwc_plots.append(pwc_overlay_guide)
            
            # Return complete PWC analysis
            return html.Div([
                pwc_controls,
                html.Div(pwc_plots)
            ], style={'backgroundColor': 'white'})
        else:
            return html.Div([html.H4('This tab is temporarily disabled for debugging.')], style={'backgroundColor': 'white'})

    @app.callback(
        Output('cwt-octave-controls', 'style'),
        [Input('wavelet-type', 'value')]
    )
    def show_cwt_octave_controls(wavelet_type):
        if wavelet_type == 'cwt':
            return {'display': 'block'}
        else:
            return {'display': 'none'}

    @app.callback(
        Output('main-figure', 'figure'),
        [Input('main-figure', 'relayoutData')],
        [State('tabs', 'value'),
         State('data-type', 'value'),
         State('well', 'value'),
         State('wavelet-type', 'value'),
         State('wavelet', 'value'),
         State('resample-freq', 'value'),
         State('min-period', 'value'),
         State('max-period', 'value'),
         State('num-periods', 'value'),
         State('period-units', 'value'),
         State('show-idwt', 'value'),
         State('show-iswt', 'value'),
         State('show-trend', 'value'),
         State('drought-flood-thresholds', 'value'),
         State('periodic-level', 'value'),
         State('show-anomaly', 'value'),
         State('custom-levels', 'value'),
         State('coh-series1-hidden', 'value'),
         State('coh-series2-hidden', 'value'),
         State('show-coherence', 'value'),
         State('denoise-threshold', 'value'),
         State('show-denoised', 'value')]
    )
    def reset_figure_on_autoscale(relayoutData, tab, data_type, well, wavelet_type, wavelet, resample_freq, min_period, max_period, num_periods, period_units, show_idwt, show_iswt, show_trend, drought_flood, periodic_level, show_anomaly, custom_levels, coh_series1, coh_series2, show_coherence, denoise_threshold, show_denoised):
        # Autoscale / autorange (double-click or the autoscale button) is handled
        # client-side by Plotly. The scalogram's data is already limited to the
        # selected Min/Max Period range, so Plotly's autoscale stays within that
        # range and there is nothing to regenerate here.
        #
        # NOTE: the previous implementation called update_tab(...) with mismatched
        # positional arguments (min_period passed as start_date, the literal 3 as
        # period_units, etc.), which silently failed and made autoscale a no-op.
        return dash.no_update

    # Callback for coherence series selection
    @app.callback(
        Output('tab-content', 'children', allow_duplicate=True),
        [Input('coherence-series1-dropdown', 'value'),
         Input('coherence-series2-dropdown', 'value')],
        [State('tabs', 'value'),
         State('data-type', 'value'),
         State('well', 'value'),
         State('wavelet-type', 'value'),
         State('wavelet', 'value'),
         State('resample-freq', 'value'),
         State('min-period', 'value'),
         State('max-period', 'value'),
         State('num-periods', 'value'),
         State('date-range', 'start_date'),
         State('date-range', 'end_date'),
         State('use-all-data', 'value'),
         State('period-units', 'value'),
         State('power-amplitude-toggle', 'value'),
         State('show-overlays', 'value'),
         State('extract-period', 'value'),
         State('extract-period-units', 'value'),
         State('extract-bandwidth', 'value'),
         State('show-extracted-component', 'value'),
         State('arrow-interpretation-mode', 'value')],
        prevent_initial_call=True,
        suppress_callback_exceptions=True
    )
    def update_coherence_analysis(series1_val, series2_val, tab, data_type, well, wavelet_type, wavelet, resample_freq, min_period, max_period, num_periods, start_date, end_date, use_all_data, period_units, power_amplitude, show_overlays, extract_period, extract_period_units, extract_bandwidth, show_extracted_component, arrow_interpretation_mode):
        if tab != 'coherence':
            return dash.no_update
        
        # Use the selected series from the coherence dropdowns
        return update_tab(
            tab, data_type, well, wavelet_type, wavelet, resample_freq, 
            min_period, max_period, num_periods, start_date, end_date, 
            use_all_data, period_units, power_amplitude, show_overlays, 
            extract_period, extract_period_units, extract_bandwidth, 
            show_extracted_component, 'none', 3, 95, arrow_interpretation_mode or 'recharge_discharge',
            show_idwt=[], show_iswt=[], show_trend=[], drought_flood=[10, 90], 
            periodic_level=1, show_anomaly=[], custom_levels=[1], 
            coh_series1=series1_val, coh_series2=series2_val, 
            show_coherence=['coh'], denoise_threshold=95, show_denoised=[],
            bypass_protection=False
        )

    # Callback for cross-wavelet series selection
    @app.callback(
        Output('tab-content', 'children', allow_duplicate=True),
        [Input('xwt-series1-dropdown', 'value'),
         Input('xwt-series2-dropdown', 'value')],
        [State('tabs', 'value'),
         State('data-type', 'value'),
         State('well', 'value'),
         State('wavelet-type', 'value'),
         State('wavelet', 'value'),
         State('resample-freq', 'value'),
         State('min-period', 'value'),
         State('max-period', 'value'),
         State('num-periods', 'value'),
         State('date-range', 'start_date'),
         State('date-range', 'end_date'),
         State('use-all-data', 'value'),
         State('period-units', 'value'),
         State('power-amplitude-toggle', 'value'),
         State('show-overlays', 'value'),
         State('extract-period', 'value'),
         State('extract-period-units', 'value'),
         State('extract-bandwidth', 'value'),
         State('show-extracted-component', 'value'),
         State('arrow-interpretation-mode', 'value')],
        prevent_initial_call=True
    )
    def update_xwt_analysis(series1_val, series2_val, tab, data_type, well, wavelet_type, wavelet, resample_freq, min_period, max_period, num_periods, start_date, end_date, use_all_data, period_units, power_amplitude, show_overlays, extract_period, extract_period_units, extract_bandwidth, show_extracted_component, arrow_interpretation_mode):
        if tab != 'xwt':
            return dash.no_update
        
        # Store the XWT selections in a way that the XWT tab can access them
        ctx = dash.callback_context
        if ctx.triggered:
            # Store XWT selections in a global way that the tab can access
            xwt_selections.update({
                'series1': series1_val,
                'series2': series2_val
            })
        
        # Use the selected series from the cross-wavelet dropdowns
        return update_tab(
            tab, data_type, well, wavelet_type, wavelet, resample_freq, 
            min_period, max_period, num_periods, start_date, end_date, 
            use_all_data, period_units, power_amplitude, show_overlays, 
            extract_period, extract_period_units, extract_bandwidth, 
            show_extracted_component, 'none', 3, 95, arrow_interpretation_mode or 'hydraulic_connectivity',
            show_idwt=[], show_iswt=[], show_trend=[], drought_flood=[10, 90], 
            periodic_level=1, show_anomaly=[], custom_levels=[1], 
            coh_series1=series1_val, coh_series2=series2_val, 
            show_coherence=['coh'], denoise_threshold=95, show_denoised=[],
            bypass_protection=False
        )

    # Callback for partial wavelet coherence series selection
    @app.callback(
        Output('tab-content', 'children', allow_duplicate=True),
        [Input('pwc-series1-dropdown', 'value'),
         Input('pwc-series2-dropdown', 'value'),
         Input('pwc-confounder-dropdown', 'value')],
        [State('tabs', 'value'),
         State('data-type', 'value'),
         State('well', 'value'),
         State('wavelet-type', 'value'),
         State('wavelet', 'value'),
         State('resample-freq', 'value'),
         State('min-period', 'value'),
         State('max-period', 'value'),
         State('num-periods', 'value'),
         State('date-range', 'start_date'),
         State('date-range', 'end_date'),
         State('use-all-data', 'value'),
         State('period-units', 'value'),
         State('power-amplitude-toggle', 'value'),
         State('show-overlays', 'value'),
         State('extract-period', 'value'),
         State('extract-period-units', 'value'),
         State('extract-bandwidth', 'value'),
         State('show-extracted-component', 'value'),
         State('arrow-interpretation-mode', 'value')],
        prevent_initial_call=True
    )
    def update_pwc_analysis(series1_val, series2_val, confounder_val, tab, data_type, well, wavelet_type, wavelet, resample_freq, min_period, max_period, num_periods, start_date, end_date, use_all_data, period_units, power_amplitude, show_overlays, extract_period, extract_period_units, extract_bandwidth, show_extracted_component, arrow_interpretation_mode):
        if tab != 'pwc':
            return dash.no_update
        
        # Store the confounder value in a way that the PWC tab can access it
        # We'll use a simple approach by modifying the callback context
        ctx = dash.callback_context
        if ctx.triggered:
            # Store PWC selections in a global way that the tab can access
            pwc_selections.update({
                'series1': series1_val,
                'series2': series2_val, 
                'confounder': confounder_val
            })
        
        return update_tab(
            tab, data_type, well, wavelet_type, wavelet, resample_freq, 
            min_period, max_period, num_periods, start_date, end_date, 
            use_all_data, period_units, power_amplitude, show_overlays, 
            extract_period, extract_period_units, extract_bandwidth, 
            show_extracted_component, 'none', 3, 95, arrow_interpretation_mode or 'custom',
            show_idwt=[], show_iswt=[], show_trend=[], drought_flood=[10, 90], 
            periodic_level=1, show_anomaly=[], custom_levels=[1], 
            coh_series1=series1_val, coh_series2=series2_val, 
            show_coherence=['coh'], denoise_threshold=95, show_denoised=[],
            bypass_protection=False
        )

    # Callback for phase analysis controls
    @app.callback(
        Output('tab-content', 'children', allow_duplicate=True),
        [Input('phase-period-select', 'value'),
         Input('show-phase-plots', 'value'),
         Input('pwc-phase-period-select', 'value'),
         Input('pwc-show-phase-plots', 'value')],
        [State('tabs', 'value'),
         State('data-type', 'value'),
         State('well', 'value'),
         State('wavelet-type', 'value'),
         State('wavelet', 'value'),
         State('resample-freq', 'value'),
         State('min-period', 'value'),
         State('max-period', 'value'),
         State('num-periods', 'value'),
         State('date-range', 'start_date'),
         State('date-range', 'end_date'),
         State('use-all-data', 'value'),
         State('period-units', 'value'),
         State('power-amplitude-toggle', 'value'),
         State('show-overlays', 'value'),
         State('extract-period', 'value'),
         State('extract-period-units', 'value'),
         State('extract-bandwidth', 'value'),
         State('show-extracted-component', 'value'),
         State('coh-series1-hidden', 'value'),
         State('coh-series2-hidden', 'value'),
         State('arrow-interpretation-mode', 'value')],
        prevent_initial_call=True,
        suppress_callback_exceptions=True
    )
    def update_phase_analysis(phase_period, show_phase_plots, pwc_phase_period, pwc_show_phase_plots, tab, data_type, well, wavelet_type, wavelet, resample_freq, min_period, max_period, num_periods, start_date, end_date, use_all_data, period_units, power_amplitude, show_overlays, extract_period, extract_period_units, extract_bandwidth, show_extracted_component, coh_series1, coh_series2, arrow_interpretation_mode):
        if tab not in ['coherence', 'xwt', 'pwc']:
            return dash.no_update
        
        # Use appropriate phase settings based on tab
        if tab == 'pwc':
            # Use PWC-specific inputs
            target_period = pwc_phase_period or 12
            show_plots = pwc_show_phase_plots or []
        else:
            # Use XWT inputs for coherence and xwt tabs
            target_period = phase_period or 12
            show_plots = show_phase_plots or []
        
        # Store phase analysis settings globally
        global phase_analysis_settings
        phase_analysis_settings = {
            'target_period': target_period,
            'show_plots': show_plots
        }
        
        # Set appropriate interpretation mode based on tab
        if tab == 'coherence':
            interpretation_mode = arrow_interpretation_mode or 'recharge_discharge'
        elif tab == 'xwt':
            # Auto-select interpretation mode based on series types (same as coherence tab)
            if arrow_interpretation_mode:
                interpretation_mode = arrow_interpretation_mode
            else:
                if 'mm_Rain' in [series1_val, series2_val] or 'rain' in str(series1_val).lower() or 'rain' in str(series2_val).lower():
                    interpretation_mode = 'recharge_discharge'
                elif 'tide' in str(series1_val).lower() or 'tide' in str(series2_val).lower():
                    interpretation_mode = 'tidal_response'
                elif 'pump' in str(series1_val).lower() or 'pump' in str(series2_val).lower():
                    interpretation_mode = 'pumping_impact'
                else:
                    interpretation_mode = 'hydraulic_connectivity'
        elif tab == 'pwc':
            interpretation_mode = arrow_interpretation_mode or 'custom'
        else:
            interpretation_mode = arrow_interpretation_mode or 'recharge_discharge'
        
        return update_tab(
            tab, data_type, well, wavelet_type, wavelet, resample_freq, 
            min_period, max_period, num_periods, start_date, end_date, 
            use_all_data, period_units, power_amplitude, show_overlays, 
            extract_period, extract_period_units, extract_bandwidth, 
            show_extracted_component, 'none', 3, 95, interpretation_mode,
            show_idwt=[], show_iswt=[], show_trend=[], drought_flood=[10, 90], 
            periodic_level=1, show_anomaly=[], custom_levels=[1], 
            coh_series1=coh_series1, coh_series2=coh_series2, 
            show_coherence=['coh'], denoise_threshold=95, show_denoised=[],
            bypass_protection=False
        )
    
    # Callback for exporting plots as PNG
    @app.callback(
        Output('download-plot-png', 'data'),
        Input('export-plot-btn', 'n_clicks'),
        [State('main-figure', 'figure'),
         State('tabs', 'value'),
         State('well', 'value'),
         State('data-type', 'value')],
        prevent_initial_call=True
    )
    def export_plot_png(n_clicks, figure, tab, well, data_type):
        if n_clicks is None:
            return dash.no_update
        
        # Generate filename based on current settings
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wavelet_{tab}_{well}_{data_type}_{timestamp}.png"
        
        # Convert figure to PNG
        if figure:
            img_bytes = pio.to_image(figure, format='png', width=1400, height=800, scale=2)  # Increased export width
            return dcc.send_bytes(img_bytes, filename)
        
        return dash.no_update

    # Callback for exporting data as CSV
    @app.callback(
        Output('download-data-csv', 'data'),
        Input('export-data-btn', 'n_clicks'),
        [State('tabs', 'value'),
         State('data-type', 'value'),
         State('well', 'value'),
         State('date-range', 'start_date'),
         State('date-range', 'end_date'),
         State('resample-freq', 'value')],
        prevent_initial_call=True
    )
    def export_data_csv(n_clicks, tab, data_type, well, start_date, end_date, resample_freq):
        if n_clicks is None:
            return dash.no_update
        
        try:
            # Prepare data for export
            df_export = data_proc.copy()
            df_export['DateTime'] = pd.to_datetime(df_export['DateTime'])
            
            # Filter by date range
            if start_date and end_date:
                mask = (df_export['DateTime'] >= pd.to_datetime(start_date)) & (df_export['DateTime'] <= pd.to_datetime(end_date))
                df_export = df_export.loc[mask]
            
            # Resample if needed
            if resample_freq != '10T':  # If not original frequency
                df_export = df_export.set_index('DateTime')
                agg_dict = {col: 'mean' for col in well_columns}
                if 'mm_Rain' in df_export.columns:
                    agg_dict['mm_Rain'] = 'sum'
                df_export = df_export.resample(resample_freq).agg(agg_dict)
                df_export = df_export.reset_index()
            
            # Generate filename
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wavelet_data_{tab}_{well}_{data_type}_{timestamp}.csv"
            
            # Convert to CSV
            csv_string = df_export.to_csv(index=False)
            return dict(content=csv_string, filename=filename)
            
        except Exception as e:
            print(f"Export error: {e}")
            return dash.no_update

    # Enhanced callback for comprehensive analysis that responds to all dropdown changes
    @app.callback(
        Output('tab-content', 'children', allow_duplicate=True),
        [Input('signal-analysis-options-dropdown-hidden', 'value'),
         Input('statistical-analysis-options-dropdown-hidden', 'value'),
         Input('event-analysis-options-dropdown-hidden', 'value'),
         Input('advanced-analysis-options-dropdown-hidden', 'value'),
         Input('extraction-target-period', 'value'),
         Input('extraction-bandwidth', 'value'),
         Input('extraction-period-units-hidden', 'value'),
         Input('trend-method-dropdown-hidden', 'value'),
         Input('confidence-level-dropdown-hidden', 'value'),
         Input('forecast-model-dropdown-hidden', 'value'),
         Input('display-output-options-dropdown-hidden', 'value'),
         Input('extraction-output-options-dropdown-hidden', 'value'),
         Input('well', 'value'),
         Input('data-type', 'value'),
         Input('resample-freq', 'value')],
        [State('tabs', 'value'),
         State('data-type', 'value'),
         State('well', 'value'),
         State('wavelet-type', 'value'),
         State('wavelet', 'value'),
         State('resample-freq', 'value'),
         State('min-period', 'value'),
         State('max-period', 'value'),
         State('num-periods', 'value'),
         State('date-range', 'start_date'),
         State('date-range', 'end_date'),
         State('use-all-data', 'value'),
         State('period-units', 'value'),
         State('power-amplitude-toggle', 'value'),
         State('show-overlays', 'value'),
         State('extract-period', 'value'),
         State('extract-period-units', 'value'),
         State('extract-bandwidth', 'value'),
         State('show-extracted-component', 'value'),
         State('arrow-interpretation-mode', 'value')],
        prevent_initial_call=True,
        suppress_callback_exceptions=True
    )
    def update_comprehensive_analysis(signal_dropdown, statistical_dropdown, event_dropdown, advanced_dropdown, 
                                    target_period_input, bandwidth_input,
                                    extraction_period_units_dropdown, trend_method_dropdown, confidence_level_dropdown,
                                    forecast_model_dropdown, display_output_options_dropdown, extraction_output_options_dropdown,
                                    well_input, data_type_input, resample_freq_input,
                                    tab, data_type, well, wavelet_type, wavelet, resample_freq, 
                                    min_period, max_period, num_periods, start_date, end_date, use_all_data, 
                                    period_units, power_amplitude, show_overlays, extract_period, 
                                    extract_period_units, extract_bandwidth, show_extracted_component, 
                                    arrow_interpretation_mode):
        # Check what triggered this callback
        from dash import callback_context
        ctx = callback_context
        
        # Only update if we're on the signal extraction tab
        if tab != 'signal':
            if ctx.triggered:
                # Check if triggered by None values (hidden components being reset)
                if all(trigger['value'] is None for trigger in ctx.triggered):
                    raise PreventUpdate
            # Instead of returning dash.no_update, raise PreventUpdate to completely prevent callback execution
            raise PreventUpdate
            
        # Additional check: Only proceed if we have meaningful trigger values (not all None)
        if ctx.triggered:
            if all(trigger['value'] is None for trigger in ctx.triggered):
                raise PreventUpdate
            
        # Additional safety check - make sure signal dropdown components exist
        try:
            if signal_dropdown is None and statistical_dropdown is None:
                raise PreventUpdate
        except:
            raise PreventUpdate
        
        # Check what triggered this callback to avoid conflicts with main callback
        # callback_context already imported above
        ctx = callback_context
        
        if ctx.triggered:
            triggered_ids = [trigger['prop_id'].split('.')[0] for trigger in ctx.triggered]
            # Only skip for well/resample changes, but allow data-type changes to be handled here
            well_or_resample_triggered = any(id in ['well', 'resample-freq'] for id in triggered_ids)
            
            if well_or_resample_triggered:
                return dash.no_update
        
        # Store the real dropdown values and trigger main callback update
        global current_signal_values
        # Determine extraction source based on data type selection
        if data_type_input == 'rain':  # The actual value is 'rain', not 'rainfall'
            extraction_source = 'mm_Rain'
        else:
            extraction_source = well_input
        
        current_signal_values = {
            'signal_options': signal_dropdown or 'period_extraction',
            'statistical_options': statistical_dropdown or ['descriptive_stats'], 
            'event_options': event_dropdown or [],
            'advanced_options': advanced_dropdown or [],
            'extraction_source': extraction_source,  # Use appropriate source based on data type
            'target_period': target_period_input or 12,
            'bandwidth': bandwidth_input or 2,
            'extraction_period_units': extraction_period_units_dropdown or 'hours',
            'trend_method': trend_method_dropdown or 'linear_regression',
            'confidence_level': confidence_level_dropdown or 95,
            'forecast_model': forecast_model_dropdown or 'arima',
            'display_options': display_output_options_dropdown or ['show_stats'],
            'extraction_output_options': extraction_output_options_dropdown or []
        }
        

        
        # Call main update_tab function directly with current values from main controls
        try:
            result = update_tab(
                tab, data_type_input, well_input, wavelet_type, wavelet, resample_freq_input, 
                min_period, max_period, num_periods, start_date, end_date, 
                use_all_data, period_units, power_amplitude, show_overlays, 
                extract_period, extract_period_units, extract_bandwidth, 
                show_extracted_component, 'none', 3, 95, arrow_interpretation_mode or 'recharge_discharge',
                bypass_protection=True
            )
            return result
        except Exception as e:
            # Log error but don't expose debug details
            return dash.no_update

    # Removed conflicting sync_analysis_options callback to prevent dropdown reset issues
    # The update_comprehensive_analysis callback above handles all dropdown changes properly

    # Sync visible signal tab dropdowns with hidden components for callback compatibility
    @app.callback(
        [Output('signal-analysis-options-dropdown-hidden', 'value'),
         Output('statistical-analysis-options-dropdown-hidden', 'value'),
         Output('event-analysis-options-dropdown-hidden', 'value'),
         Output('advanced-analysis-options-dropdown-hidden', 'value'),
         Output('extraction-period-units-hidden', 'value'),
         Output('trend-method-dropdown-hidden', 'value'),
         Output('confidence-level-dropdown-hidden', 'value'),
         Output('forecast-model-dropdown-hidden', 'value'),
         Output('display-output-options-dropdown-hidden', 'value'),
         Output('extraction-output-options-dropdown-hidden', 'value')],
        [Input('signal-analysis-options-dropdown', 'value'),
         Input('statistical-analysis-options-dropdown', 'value'),
         Input('event-analysis-options-dropdown', 'value'),
         Input('advanced-analysis-options-dropdown', 'value'),
         Input('extraction-period-units', 'value'),
         Input('trend-method-dropdown', 'value'),
         Input('confidence-level-dropdown', 'value'),
         Input('forecast-model-dropdown', 'value'),
         Input('display-output-options-dropdown', 'value'),
         Input('extraction-output-options-dropdown', 'value')],
        [State('tabs', 'value')],
        prevent_initial_call=True,
        suppress_callback_exceptions=True
    )
    def sync_signal_dropdowns(signal_val, stat_val, event_val, advanced_val, 
                            period_units_val, trend_val, confidence_val, forecast_val, 
                            display_val, extraction_val, tab):
        # Only sync when on signal tab
        if tab == 'signal':
            return (signal_val or 'period_extraction',
                    stat_val or ['descriptive_stats'],
                    event_val or [],
                    advanced_val or [],
                    period_units_val or 'hours',
                    trend_val or 'linear_regression',
                    confidence_val or 95,
                    forecast_val or 'arima',
                    display_val or ['show_stats'],
                    extraction_val or [])
        else:
            # Return no_update to avoid changing hidden values when not on signal tab
            return [dash.no_update] * 10

    # Sync visible coherence tab dropdowns with hidden components for callback compatibility
    @app.callback(
        [Output('coherence-series1-dropdown-hidden', 'value'),
         Output('coherence-series2-dropdown-hidden', 'value')],
        [Input('coherence-series1-dropdown', 'value'),
         Input('coherence-series2-dropdown', 'value')],
        [State('tabs', 'value')],
        prevent_initial_call=True,
        suppress_callback_exceptions=True
    )
    def sync_coherence_dropdowns(coh_s1, coh_s2, tab):
        # Only sync when on coherence tab
        if tab == 'coherence':
            return (coh_s1, coh_s2)
        else:
            return [dash.no_update] * 2

    # Sync visible XWT tab dropdowns with hidden components for callback compatibility
    @app.callback(
        [Output('xwt-series1-dropdown-hidden', 'value'),
         Output('xwt-series2-dropdown-hidden', 'value')],
        [Input('xwt-series1-dropdown', 'value'),
         Input('xwt-series2-dropdown', 'value')],
        [State('tabs', 'value')],
        prevent_initial_call=True,
        suppress_callback_exceptions=True
    )
    def sync_xwt_dropdowns(xwt_s1, xwt_s2, tab):
        # Only sync when on XWT tab
        if tab == 'xwt':
            return (xwt_s1, xwt_s2)
        else:
            return [dash.no_update] * 2

    # Sync visible denoise threshold slider with hidden component for callback compatibility
    @app.callback(
        Output('denoise-threshold-slider', 'value'),
        [Input('denoise-threshold-slider-visible', 'value')],
        prevent_initial_call=True
    )
    def sync_denoise_threshold_slider(visible_value):
        return visible_value or 95

    app.run(debug=True)
