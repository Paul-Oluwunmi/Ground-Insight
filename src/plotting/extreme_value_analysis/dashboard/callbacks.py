"""
Dashboard callbacks for Extreme Value Analysis
"""

from dash import Input, Output, State
from dash import callback_context
from functools import lru_cache
import plotly.graph_objects as go
import logging
import traceback
from ..constants import COLORS, global_store
from ..data_processing import get_processed_data, get_global_data
from ..block_maxima import get_block_maxima
from ..eva_analysis import (
    perform_extreme_value_analysis,
    perform_bayesian_eva_stationary,
    perform_bayesian_eva_non_stationary
)
from ..plotting import (
    create_time_series_plot,
    create_eva_plots,
    create_corner_plot,
    generate_posterior_predictive_checks
)
from ..utils import (
    get_suitable_block_periods,
    get_resample_options
)


def register_callbacks(app):
    """Register all callbacks for the dashboard"""
    
    @lru_cache(maxsize=32)
    def run_full_analysis(well, data_type, resample, block, analysis_type):
        
        df_pl = global_store.get('df_pl')
        if df_pl is None:
            raise ValueError("No data available")

        # Process the data
        ts = get_processed_data(df_pl, well, resample, data_type)
        if ts is None or len(ts) < 10:
            raise ValueError(f"Insufficient data for {well}")

        # Calculate block maxima
        block_maxima = get_block_maxima(ts, block, data_type)
        if block_maxima is None or len(block_maxima) < 10:
            raise ValueError("Insufficient block maxima for analysis")

        logging.info(f"Calculated {len(block_maxima)} block maxima using {block} period")

        effective_analysis_type = analysis_type
        warning_message = None

        # Perform EVA analysis based on selected type
        if analysis_type == 'stationary':
            stats = perform_extreme_value_analysis(block_maxima)
        elif analysis_type == 'bayesian_stationary':
            stats = perform_bayesian_eva_stationary(block_maxima)
            if stats is None:
                logging.warning("Bayesian stationary analysis failed. Falling back to MLE.")
                stats = perform_extreme_value_analysis(block_maxima)
                effective_analysis_type = 'stationary'
                warning_message = "Warning: Bayesian analysis failed. Showing standard MLE results."
        elif analysis_type == 'bayesian_non_stationary':
            stats = perform_bayesian_eva_non_stationary(block_maxima)
            if stats is None:
                logging.warning("Bayesian non-stationary analysis failed. Falling back to MLE.")
                stats = perform_extreme_value_analysis(block_maxima)
                effective_analysis_type = 'stationary'
                warning_message = "Warning: Bayesian analysis failed. Showing standard MLE results."
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        if stats is None:
            raise ValueError("Error in extreme value analysis")
        
        # The time series data is needed for the time series plot, so return it as well.
        # It's a pandas Series, which is not ideal for caching, but the primary
        # cost is the analysis, so this is an acceptable trade-off.
        return ts, block_maxima, stats, effective_analysis_type, warning_message

    @app.callback(
        [Output('well-selector', 'options'),
         Output('well-selector', 'value'),
         Output('block-selector', 'options'),
         Output('block-selector', 'value'),
         Output('resample-selector', 'options'),
         Output('resample-selector', 'value')],
        [Input('data-type-selector', 'value'),
         Input('block-selector', 'value')]
    )
    def update_all_options(data_type, block_period):
        """Update all dropdown options based on data type and block period"""
        try:
            ctx = callback_context
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
            
            df_pl = global_store.get('df_pl')
            if df_pl is None:
                raise ValueError("No data available in global store")
            
            # Get all column names
            if hasattr(df_pl, 'columns'):
                columns = df_pl.columns
            else:
                columns = df_pl.get_column_names()
            
            # Filter well columns based on data type
            if data_type == 'rain':
                well_columns = [
                    col for col in columns 
                    if any(term in col.lower() 
                          for term in ['rain', 'rainfall', 'precipitation'])
                ]
                # Set default to mm_Rain if available
                default_well = 'mm_Rain' if 'mm_Rain' in well_columns else well_columns[0] if well_columns else None
            else:  # gwl
                well_columns = [
                    col for col in columns 
                    if col not in ['DateTime'] and 
                    not any(term in col.lower() 
                           for term in ['rain', 'rainfall', 'precipitation'])
                ]
                default_well = well_columns[0] if well_columns else None
            
            well_options = [{'label': col, 'value': col} for col in well_columns]
            
            # Calculate data length for block periods
            if hasattr(df_pl, 'to_pandas'):
                df = df_pl.to_pandas()
            else:
                df = df_pl
                
            date_range = df['DateTime'].max() - df['DateTime'].min()
            data_length_years = date_range.days / 365.25
            
            # Get suitable block periods
            block_options, default_block = get_suitable_block_periods(data_length_years, data_type)
            
            # Use existing block period if valid, otherwise use default
            current_block = block_period if block_period and any(opt['value'] == block_period for opt in block_options) else default_block
            
            # Get resample options based on current block period
            resample_options = get_resample_options(current_block, data_type)
            default_resample = '10T' if data_type == 'rain' else '15T'
            
            return (
                well_options,
                default_well,
                block_options,
                current_block,
                resample_options,
                default_resample
            )
            
        except Exception as e:
            logging.error(f"Error updating options: {str(e)}")
            # Return safe defaults
            default_options = [{'label': 'No data available', 'value': 'none'}]
            return (
                default_options,
                'none',
                [{'label': '1 Month', 'value': '1M'}],
                '1M',
                [{'label': '15 Minutes', 'value': '15T'}],
                '15T'
            )

    # Separate callback for time series plot (stays visible during Bayesian analysis)
    @app.callback(
        Output('time-series-plot', 'figure'),
        [Input('well-selector', 'value'),
         Input('data-type-selector', 'value'),
         Input('resample-selector', 'value'),
         Input('block-selector', 'value')],
        prevent_initial_call=False
    )
    def update_time_series_plot(well, data_type, resample, block):
        """Update time series plot only - completely independent of analysis type and cache"""
        try:
            if not well or not data_type:
                return go.Figure()
            
            # Get data independently without using shared cache
            df = get_global_data()
            if df is None:
                return go.Figure()
            
            # Process data directly for time series (no cache sharing)
            ts = get_processed_data(df, well, resample, data_type)
            if ts is None:
                return go.Figure()
            
            # Get block maxima independently
            block_maxima = get_block_maxima(ts, block, data_type)
            if block_maxima is None or len(block_maxima) == 0:
                return go.Figure()
            
            # Create time series plot directly
            return create_time_series_plot(
                ts=ts,
                block_maxima=block_maxima,
                well=well,
                data_type=data_type,
                resample=resample
            )
            
        except Exception as e:
            logging.error(f"Error updating time series plot: {str(e)}")
            return go.Figure()

    # Separate callback for EVA analysis (updates during Bayesian analysis)
    @app.callback(
        [Output('eva-plot', 'figure'),
         Output('corner-plot', 'figure'),
         Output('ppc-ecdf-plot', 'figure'),
         Output('ppc-summary-table', 'data'),
         Output('bayesian-plots-container', 'style'),
         Output('current-analysis-params', 'data')],
        [Input('well-selector', 'value'),
         Input('data-type-selector', 'value'),
         Input('resample-selector', 'value'),
         Input('block-selector', 'value'),
         Input('analysis-type-selector', 'value')],
        prevent_initial_call=False
    )
    def update_eva_analysis(well, data_type, resample, block, analysis_type):
        """Update EVA analysis when parameters change (simplified POT-style approach)"""
        try:
            # Run analysis directly (like POT)
            ts, block_maxima, stats, effective_analysis_type, warning_message = run_full_analysis(
                well, data_type, resample, block, analysis_type
            )
            
            # Get current analysis parameters
            current_params = {
                'well': well, 'data_type': data_type, 'resample': resample, 
                'block': block, 'analysis_type': analysis_type
            }
            
            # Create EVA plots
            eva_fig = create_eva_plots(
                stats=stats,
                well=well,
                data_type=data_type,
                analysis_type=effective_analysis_type,
                warning=warning_message
            )
            
            # Create Bayesian-specific plots
            if effective_analysis_type in ['bayesian_stationary', 'bayesian_non_stationary']:
                corner_fig = create_corner_plot(stats, effective_analysis_type)
                ppc_fig, ppc_summary = generate_posterior_predictive_checks(
                    stats, block_maxima, effective_analysis_type
                )
                # Format summary for DataTable
                ppc_table_data = []
                for k, v in ppc_summary.items():
                    if "(90% CI)" not in k:
                        ci_key = k + " (90% CI)"
                        ci_value = ppc_summary.get(ci_key, "N/A")
                        ppc_table_data.append({
                            "Statistic": k,
                            "Observed": v,
                            "Posterior_CI": ci_value
                        })
                bayesian_style = {'display': 'block'}
            else:
                corner_fig = go.Figure()
                ppc_fig = go.Figure()
                ppc_table_data = []
                bayesian_style = {'display': 'none'}
            
            return (eva_fig, corner_fig, ppc_fig, ppc_table_data, 
                   bayesian_style, current_params)
            
        except Exception as e:
            logging.error(f"Error updating plots: {str(e)}")
            traceback.print_exc()
            
            # Create informative error figure
            error_fig = go.Figure()
            error_fig.update_layout(
                title=dict(
                    text=f"Error: {str(e)}",
                    font=dict(color=COLORS['danger'])
                ),
                plot_bgcolor=COLORS['background'],
                paper_bgcolor=COLORS['background']
            )
            
            return (error_fig, go.Figure(), go.Figure(), [], 
                   {'display': 'none'}, {})

    @app.callback(
        Output('data-type-store', 'data'),
        [Input('data-type-selector', 'value')]
    )
    def store_data_type_and_cleanup(data_type):
        """Store the current data type and handle cleanup"""
        try:
            # Clear cache when data type changes
            if data_type:
                run_full_analysis.cache_clear()
            
            # Note: We keep the data in global store since it doesn't change
            # Only clear cache, not the data itself
            
            return data_type
            
        except Exception as e:
            logging.error(f"Error in cleanup callback: {str(e)}")
            return data_type

    logging.info("Callbacks registered successfully")

__all__ = ['register_callbacks']
