"""
Interactive widgets for Jupyter notebook interface
"""


def interactive_run_statistics(data, well_columns):
    """
    Display interactive input boxes for Q1 and Q3 percentiles and a button to plot statistics.
    For use in Jupyter notebooks. Calls run_statistics with selected percentiles.
    
    Parameters:
    -----------
    data : polars.DataFrame or pandas.DataFrame
        Groundwater data
    well_columns : List[str]
        List of well column names to plot
    """
    from . import run_statistics
    _interactive_run_statistics_impl(data, well_columns, run_statistics)


def _interactive_run_statistics_impl(data, well_columns, run_statistics_func):
    """
    Internal implementation of interactive statistics.
    
    Parameters:
    -----------
    data : polars.DataFrame or pandas.DataFrame
        Groundwater data
    well_columns : List[str]
        List of well column names to plot
    run_statistics_func : callable
        Function to call with (data, well_columns, q1_percentile, q3_percentile)
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    q1_widget = widgets.BoundedFloatText(
        value=25,
        min=0,
        max=100,
        step=1,
        description='Q1 Percentile:',
        style={'description_width': 'initial'}
    )
    q3_widget = widgets.BoundedFloatText(
        value=75,
        min=0,
        max=100,
        step=1,
        description='Q3 Percentile:',
        style={'description_width': 'initial'}
    )
    plot_button = widgets.Button(description="Plot Statistics", button_style='success')
    output = widgets.Output()

    def on_plot_button_clicked(b):
        with output:
            clear_output()
            q1 = q1_widget.value
            q3 = q3_widget.value
            if not (0 <= q1 < q3 <= 100):
                print("Percentiles must satisfy 0 ≤ Q1 < Q3 ≤ 100.")
                return
            print(f"Plotting with Q1={q1} and Q3={q3} percentiles...")
            run_statistics_func(data, well_columns, q1_percentile=q1, q3_percentile=q3)

    plot_button.on_click(on_plot_button_clicked)
    display(widgets.HBox([q1_widget, q3_widget, plot_button]), output)

