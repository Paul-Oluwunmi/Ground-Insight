"""
Subplot Callbacks
=================

Dash callbacks for the subplot visualization.
"""

import pandas as pd
from dash.dependencies import Input, Output

from .data_processing import resample_data_optimized
from .plotting import create_plot_figure


def register_callbacks(app, data, well_columns):
    """
    Register all callbacks for the subplot Dash application.
    
    Parameters:
    -----------
    app : dash.Dash
        Dash application instance
    data : pd.DataFrame
        Input data containing DateTime and well measurements
    well_columns : list
        List of well column names
    """
    @app.callback(
        Output('resampled-data', 'data'),
        Input('interval-selector', 'value')
    )
    def resample_data(interval):
        """
        Resample data based on selected time interval with caching.
        
        Parameters:
        -----------
        interval : str
            Selected time interval ('original', 'hourly', 'daily', etc.)
        
        Returns:
        --------
        dict
            Resampled data in dictionary format
        """
        try:
            resampled_df = resample_data_optimized(data, well_columns, interval)
            return resampled_df.to_dict('records')
        except Exception as e:
            print(f"Error in resampling: {str(e)}")
            return data.to_dict('records')
    
    # Create callbacks for each plot
    # Use a closure to properly capture the plot index
    def make_callback(plot_index):
        @app.callback(
            Output(f'plot-{plot_index}', 'figure'),
            [Input(f'well-selector-{plot_index}', 'value'),
            Input('resampled-data', 'data'),
            Input('interval-selector', 'value')]
        )
        def update_plot(selected_well, resampled_data, interval):
            """
            Update individual plot based on well selection and time interval.
            
            Parameters:
            -----------
            selected_well : str
                Selected well name
            resampled_data : dict
                Resampled data dictionary
            interval : str
                Selected time interval
            
            Returns:
            --------
            go.Figure
                Updated plot figure
            """
            return create_plot_figure(selected_well, resampled_data, interval, plot_index)
        return update_plot
    
    # Register callbacks for all 4 plots
    for i in range(4):
        make_callback(i)

