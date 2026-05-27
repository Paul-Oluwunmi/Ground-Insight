"""
Plotting functions for statistics visualization
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Dict
from .constants import MONTH_ORDER, COLORS
from .statistical_calculations import calculate_boxplot_statistics
from .data_processing import prepare_data, prepare_well_data, prepare_all_years_data


def create_hover_template(well, year, q1_percentile, q3_percentile, stats_dict, is_all_years=False):
    """
    Create hover template string for boxplot traces.
    
    Parameters:
    -----------
    well : str
        Well name
    year : int or str
        Year (or "All Years")
    q1_percentile : float
        Q1 percentile value
    q3_percentile : float
        Q3 percentile value
    stats_dict : dict
        Statistics dictionary
    is_all_years : bool
        Whether this is for all years trace
        
    Returns:
    --------
    str : Hover template string
    """
    year_label = "All Years" if is_all_years else str(year)
    
    template = (
        f"<b>Well: {well}</b><br>" +
        f"<b>Year: {year_label}</b><br>" +
        "<b>Month:</b> %{x}<br>" +
        "<b>Season:</b> %{customdata[0]}<br>" +
        "<b>Value:</b> %{y:.2f} m<br>" +
        "<b>Statistics:</b><br>" +
        f"Median: {stats_dict['median']:.2f} m<br>" +
        f"Mean: {stats_dict['mean']:.2f} m<br>" +
        f"Q{q1_percentile}: {stats_dict['q1']:.2f} m<br>" +
        f"Q{q3_percentile}: {stats_dict['q3']:.2f} m<br>" +
        f"SD: {stats_dict['std']:.2f} m<br>" +
        "<b>Outlier:</b> %{text}<br>" +
        "<extra></extra>"
    )
    
    return template


def create_year_trace(well, year, well_values, month_values, season_values, 
                      stats_dict, color, q1_percentile, q3_percentile):
    """
    Create a boxplot trace for a specific year.
    
    Parameters:
    -----------
    well : str
        Well name
    year : int
        Year
    well_values : array-like
        Well values for the year
    month_values : array-like
        Month values
    season_values : array-like
        Season values
    stats_dict : dict
        Statistics dictionary
    color : str
        Color for the trace
    q1_percentile : float
        Q1 percentile
    q3_percentile : float
        Q3 percentile
        
    Returns:
    --------
    go.Box : Plotly box trace
    """
    # Determine outliers
    is_outlier = (well_values < stats_dict['lower_bound']) | (well_values > stats_dict['upper_bound'])
    outlier_text = np.where(is_outlier, "Yes", "No")
    
    hover_template = create_hover_template(well, year, q1_percentile, q3_percentile, stats_dict)
    
    trace = go.Box(
        x=month_values,
        y=well_values,
        name=f"{year}",
        marker_color=color,
        visible="legendonly",
        showlegend=True,
        boxpoints='outliers',
        boxmean='sd',
        notched=True,
        line=dict(width=2),
        fillcolor='rgba(255,255,255,0.8)',
        legendgroup=well,
        legendgrouptitle_text=well,
        hovertemplate=hover_template,
        text=outlier_text,
        customdata=np.array([season_values]).T,
        quartilemethod="linear"
    )
    
    return trace


def create_all_years_trace(well, well_values, months, seasons, stats_dict, q1_percentile, q3_percentile):
    """
    Create boxplot trace for all years combined.
    
    Parameters:
    -----------
    well : str
        Well name
    well_values : array-like
        All well values
    months : array-like
        Month values
    seasons : array-like
        Season values
    stats_dict : dict
        Statistics dictionary
    q1_percentile : float
        Q1 percentile
    q3_percentile : float
        Q3 percentile
        
    Returns:
    --------
    go.Box : Plotly box trace
    """
    # Determine outliers
    is_outlier = (well_values < stats_dict['lower_bound']) | (well_values > stats_dict['upper_bound'])
    outlier_text = np.where(is_outlier, "Yes", "No")
    
    hover_template = create_hover_template(well, None, q1_percentile, q3_percentile, stats_dict, is_all_years=True)
    
    trace = go.Box(
        x=months,
        y=well_values,
        name="All Years",
        marker_color='rgba(0,0,0,0.5)',
        boxmean=True,
        boxpoints='outliers',
        notched=True,
        line=dict(width=2),
        fillcolor='rgba(255,255,255,0.8)',
        legendgroup=well,
        legendgrouptitle_text=well,
        hovertemplate=hover_template,
        text=outlier_text,
        customdata=np.array([seasons]).T,
        quartilemethod="linear"
    )
    
    return trace


def create_buttons(well_columns, well_traces, num_traces):
    """
    Create update menu buttons for well selection.
    
    Parameters:
    -----------
    well_columns : List[str]
        List of well column names
    well_traces : Dict[str, List[int]]
        Dictionary mapping well names to trace indices
    num_traces : int
        Total number of traces
        
    Returns:
    --------
    List[dict] : List of button configurations
    """
    buttons = []
    
    # "Select Well" button - everything hidden, no legend
    buttons.append({
        "args": [
            {"visible": [False] * num_traces},
            {"showlegend": False}
        ],
        "label": "Select Well",
        "method": "update"
    })
    
    # Well-specific buttons
    for well in well_columns:
        visibility = []
        
        for i in range(num_traces):
            if i in well_traces[well]:
                visibility.append("legendonly")
            else:
                visibility.append(False)
        
        well_button = {
            "args": [
                {"visible": visibility},
                {"showlegend": True}
            ],
            "label": well,
            "method": "update"
        }
        buttons.append(well_button)
    
    return buttons


def configure_layout(fig, buttons):
    """
    Configure the plot layout.
    
    Parameters:
    -----------
    fig : go.Figure
        Plotly figure object
    buttons : List[dict]
        Button configurations for update menu
    """
    fig.update_layout(
        height=800,
        width=1600,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            groupclick="toggleitem",
            itemclick="toggle",
            tracegroupgap=5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            y=0.5,
            yanchor='middle',
            x=1.02,
            xanchor='left',
            traceorder='grouped'
        ),
        margin=dict(
            t=120,
            r=100
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=dict(
            text="Monthly Statistics<br><sup>Distribution By Year</sup>",
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top',
            font=dict(size=20, weight='bold')
        ),
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1.02,
            'xanchor': 'right',
            'y': 1.15,
            'yanchor': 'top'
        }],
        boxmode='group',
        boxgap=0.1,
        boxgroupgap=0.2
    )


def configure_axes(fig):
    """
    Configure x and y axes.
    
    Parameters:
    -----------
    fig : go.Figure
        Plotly figure object
    """
    # Update x-axis
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        categoryorder='array',
        categoryarray=MONTH_ORDER,
        tickangle=0,
        title_font=dict(
            size=16,
            family="Arial",
            color="black",
            weight="bold"
        ),
        title_standoff=15,
        title_text="Month",
        tickfont=dict(
            family="Arial",
            size=14,
            color="black",
            weight="bold"
        )
    )
    
    # Update y-axis
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128, 128, 128, 0.2)',
        title_font=dict(
            size=16,
            family="Arial",
            color="black",
            weight="bold"
        ),
        title_standoff=15,
        title_text="Groundwater Level (m)",
        tickfont=dict(
            family="Arial",
            size=14,
            color="black",
            weight="bold"
        )
    )

