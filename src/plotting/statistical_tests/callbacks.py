"""
Callbacks for Statistical Analysis Dashboard
"""

import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State, ALL
import polars as pl
import plotly.graph_objects as go
from .components import create_dropdowns
from .plotting import create_box_plot

# Global variables (set by main function)
wells = []
data = None


def register_callbacks(app, data_df, well_columns):
    """Register all callbacks for the application."""
    global wells, data
    wells = well_columns
    data = data_df
    
    @app.callback(
        Output('dropdown-container', 'children'),
        [Input('num-selections-slider', 'value')]
    )
    def update_dropdowns(num_selections):
        well_options = [{'label': well, 'value': well} for well in wells]
        return create_dropdowns(num_selections, well_options)
    
    @app.callback(
        Output({'type': 'year-dropdown', 'index': ALL}, 'options'),
        [Input({'type': 'well-dropdown', 'index': ALL}, 'value')]
    )
    def update_year_options(selected_wells):
        year_options_list = []
        
        for i, well in enumerate(selected_wells):
            if well:
                # Filter data to only get years with actual data for this well
                available_years = (
                    data
                    .filter(
                        pl.col(well).is_not_null()
                    )
                    .get_column('Year')
                    .unique()
                    .to_list()
                )
                
                # Only create options for years that have data
                if available_years:
                    year_options = [
                        {'label': year, 'value': year} 
                        for year in sorted(available_years)
                    ]
                else:
                    year_options = []
            else:
                year_options = []
            
            year_options_list.append(year_options)
        
        return year_options_list

    @app.callback(
        Output({'type': 'month-dropdown', 'index': ALL}, 'options'),
        [Input({'type': 'well-dropdown', 'index': ALL}, 'value'),
         Input({'type': 'year-dropdown', 'index': ALL}, 'value')]
    )
    def update_month_options(selected_wells, selected_years):
        month_options_list = []
        
        for i, (well, year) in enumerate(zip(selected_wells, selected_years)):
            if well and year:
                # Filter data using polars expressions to only get months with actual data
                available_months = (
                    data
                    .filter(
                        (pl.col('Year') == int(year)) &
                        pl.col(well).is_not_null()
                    )
                    .get_column('Month')
                    .unique()
                    .to_list()
                )
                
                # Only create options for months that have data
                if available_months:
                    month_options = [
                        {'label': month, 'value': month} 
                        for month in sorted(available_months, 
                                        key=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].index(x))
                    ]
                else:
                    month_options = []
            else:
                month_options = []
            
            month_options_list.append(month_options)
        
        return month_options_list

    @app.callback(
        [Output({'type': 'year-dropdown', 'index': ALL}, 'value'),
         Output({'type': 'month-dropdown', 'index': ALL}, 'value')],
        [Input({'type': 'well-dropdown', 'index': ALL}, 'value')],
        [State({'type': 'year-dropdown', 'index': ALL}, 'value'),
         State({'type': 'month-dropdown', 'index': ALL}, 'value')]
    )
    def reset_selections_when_well_changes(selected_wells, current_years, current_months):
        """Reset year and month selections only for wells that changed."""
        ctx = dash.callback_context
        
        # Handle initial load or no inputs
        if not ctx.inputs or len(ctx.inputs) == 0:
            return current_years, current_months
        
        # Find which well changed
        well_changed = False
        changed_index = None
        
        for input_item in ctx.inputs:
            if isinstance(input_item, dict) and 'well-dropdown' in input_item:
                well_changed = True
                changed_index = input_item.get('index')
                break
        
        if well_changed and changed_index is not None:
            # Only reset the year and month for the changed well
            new_years = current_years.copy() if current_years else [None] * len(selected_wells)
            new_months = current_months.copy() if current_months else [None] * len(selected_wells)
            
            if changed_index < len(new_years):
                new_years[changed_index] = None
                new_months[changed_index] = None
            
            return new_years, new_months
        
        return current_years, current_months

    @app.callback(
        Output('plot-container', 'children'),
        [Input('update-button', 'n_clicks')],
        [State('num-selections-slider', 'value'),
        State({'type': 'well-dropdown', 'index': ALL}, 'value'),
        State({'type': 'year-dropdown', 'index': ALL}, 'value'),
        State({'type': 'month-dropdown', 'index': ALL}, 'value')]
    )
    def update_plot(n_clicks, num_selections, wells_selected, years_selected, months_selected):
        if not n_clicks:
            return no_update
        
        try:
            if not all(wells_selected) or not all(years_selected) or not all(months_selected):
                return dcc.Graph(figure=go.Figure())
            
            datasets = []
            valid_selections = []
            
            # Add box plots for each selection
            for i in range(len(wells_selected)):
                if wells_selected[i] and years_selected[i] and months_selected[i]:
                    well_data = (
                        data
                        .filter(
                            (pl.col('Year') == int(years_selected[i])) &
                            (pl.col('Month') == months_selected[i])
                        )
                        .get_column(wells_selected[i])
                        .drop_nulls()
                        .to_numpy()
                    )
                    
                    if len(well_data) > 0:
                        valid_selections.append(i)
                        month_abbr = months_selected[i][:3]
                        year_abbr = str(years_selected[i])[2:]
                        x_label = f"{wells_selected[i]}\n({month_abbr}-{year_abbr})"
                        if wells_selected.count(wells_selected[i]) > 1:
                            x_label = f"{x_label} ({i+1})"
                        
                        datasets.append({
                            'label': x_label,
                            'data': well_data,
                            'index': len(valid_selections) - 1
                        })
            
            if len(datasets) >= 2:
                return create_box_plot(datasets, wells_selected, years_selected, months_selected)
            else:
                return dcc.Graph(figure=go.Figure())
            
        except Exception as e:
            print(f"Error in update_plot: {str(e)}")
            import traceback
            traceback.print_exc()
            return dcc.Graph(figure=go.Figure())

