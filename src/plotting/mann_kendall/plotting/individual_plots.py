"""
Individual Well Plotting Functions

This module contains functions for creating plots for individual wells.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import traceback
from ..constants import (
    ALPHA_LEVEL,
    ROLLING_WINDOW_SIZE,
    SAVE_PLOTS
)
from ..utils import (
    ensure_datetime_index,
    generate_well_display_name,
    generate_well_filename
)
from ..core_tests import mann_kendall_test
from ..analysis import (
    calculate_yearly_max_duration_metrics,
    calculate_percentile_low_flow_metrics
)


def create_gwl_rolling_mean_plot(data, well_name, rolling_window=None):
    """
    Create GWL time series plot with rolling mean.
    
    Args:
        data: pandas Series with datetime index and groundwater levels
        well_name: Display name of the well
        rolling_window: Rolling window size for smoothing
    
    Returns:
        plotly.graph_objects.Figure: Plot figure
    """
    # Use global rolling window if none specified
    if rolling_window is None:
        rolling_window = ROLLING_WINDOW_SIZE
        print(f"Using global rolling window size: {rolling_window} days")
    
    try:
        # Calculate rolling mean
        rolling_mean = data.rolling(window=rolling_window, center=False).mean()
        
        # Create figure
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name='Original GWL',
            line=dict(color='lightblue', width=1),
            opacity=0.7
        ))
        
        # Add rolling mean
        fig.add_trace(go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean.values,
            mode='lines',
            name=f'{rolling_window}-Day Rolling Mean',
            line=dict(color='red', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Groundwater Level Time Series - {well_name}',
            xaxis_title='Date',
            yaxis_title='Groundwater Level (m)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        # Add statistics annotations
        mean_level = data.mean()
        min_level = data.min()
        max_level = data.max()
        
        fig.add_annotation(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=f'Mean: {mean_level:.3f} m<br>Min: {min_level:.3f} m<br>Max: {max_level:.3f} m',
            showarrow=False,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating GWL rolling mean plot: {e}")
        return None


def create_seasonal_analysis_plot(data, well_name, seasonal_period=12):
    """
    Create seasonal analysis plot showing monthly/seasonal patterns.
    
    Args:
        data: pandas Series with datetime index and groundwater levels
        well_name: Display name of the well
        seasonal_period: Seasonal period (12 for monthly, 4 for quarterly)
    
    Returns:
        plotly.graph_objects.Figure: Plot figure
    """
    try:
        # Create seasonal data
        if seasonal_period == 12:
            # Monthly analysis
            seasonal_data = data.groupby(data.index.month).agg(['mean', 'std', 'min', 'max'])
            seasonal_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            title_suffix = 'Monthly'
        else:
            # Quarterly analysis
            seasonal_data = data.groupby(data.index.quarter).agg(['mean', 'std', 'min', 'max'])
            seasonal_labels = ['Q1', 'Q2', 'Q3', 'Q4']
            title_suffix = 'Quarterly'
        
        # Create figure with subplots
        fig = go.Figure()
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=seasonal_labels,
            y=seasonal_data['mean'].values,
            mode='lines+markers',
            name='Mean',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Add min-max range
        fig.add_trace(go.Scatter(
            x=seasonal_labels + seasonal_labels[::-1],
            y=list(seasonal_data['max'].values) + list(seasonal_data['min'].values)[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Min-Max Range',
            showlegend=True
        ))
        
        # Add standard deviation error bars
        fig.add_trace(go.Scatter(
            x=seasonal_labels,
            y=seasonal_data['mean'].values,
            error_y=dict(
                type='data',
                array=seasonal_data['std'].values,
                visible=True
            ),
            mode='markers',
            name='±1 Std Dev',
            marker=dict(size=0),
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{title_suffix} Seasonal Analysis - {well_name}',
            xaxis_title=f'{title_suffix} Period',
            yaxis_title='Groundwater Level (m)',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating seasonal analysis plot: {e}")
        return None


def create_mk_trend_plot(data, well_name, alpha=None):
    """
    Create Mann-Kendall trend analysis plot.
    
    Args:
        data: pandas Series with datetime index and groundwater levels
        well_name: Display name of the well
        alpha: Significance level
    
    Returns:
        plotly.graph_objects.Figure: Plot figure
    """
    # Use global alpha if none specified
    if alpha is None:
        alpha = ALPHA_LEVEL
        print(f"Using global alpha level: {alpha}")
    
    try:
        # Run Mann-Kendall test
        mk_result = mann_kendall_test(data, alpha=alpha)
        
        if 'error' in mk_result:
            print(f"Error in MK test for {well_name}: {mk_result['error']}")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name='Original GWL',
            line=dict(color='lightblue', width=1),
            opacity=0.7
        ))
        
        # Add trend line if significant
        if mk_result.get('trend') in ['increasing', 'decreasing']:
            # Calculate trend line
            x_numeric = np.arange(len(data))
            slope = mk_result.get('slope', 0)
            intercept = data.iloc[0] - slope * 0
            
            trend_line = slope * x_numeric + intercept
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=trend_line,
                mode='lines',
                name=f"Trend Line (slope: {slope*1000:.2f} mm/year)",
                line=dict(color='red', width=2, dash='dash')
            ))
        
        # Update layout
        trend_text = mk_result.get('trend', 'no_trend')
        p_value = mk_result.get('p_value', 1.0)
        significance = "Significant" if p_value < alpha else "Not Significant"
        
        fig.update_layout(
            title=f'Mann-Kendall Trend Analysis - {well_name}<br>'
                  f'Trend: {trend_text.title()}, P-value: {p_value:.4f} ({significance})',
            xaxis_title='Date',
            yaxis_title='Groundwater Level (m)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        # Add statistics annotations
        mean_level = data.mean()
        tau = mk_result.get('tau', 0)
        z_score = mk_result.get('z_score', 0)
        
        fig.add_annotation(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=f'Tau: {tau:.3f}<br>Z-score: {z_score:.3f}<br>Mean: {mean_level:.3f} m',
            showarrow=False,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating MK trend plot: {e}")
        return None


def create_duration_analysis_plot(duration_results, well_name):
    """
    Create duration analysis plot.
    
    Args:
        duration_results: Results from calculate_yearly_max_duration_metrics
        well_name: Display name of the well
    
    Returns:
        plotly.graph_objects.Figure: Plot figure
    """
    try:
        if 'error' in duration_results:
            return None
        
        # Extract data
        years = list(duration_results['yearly_breakdown'].keys())
        max_durations = [duration_results['yearly_breakdown'][year]['max_duration'] 
                        for year in years]
        mean_durations = [duration_results['yearly_breakdown'][year]['mean_duration'] 
                         for year in years]
        
        # Create figure
        fig = go.Figure()
        
        # Add max duration
        fig.add_trace(go.Bar(
            x=years,
            y=max_durations,
            name='Max Duration',
            marker_color='red',
            opacity=0.7
        ))
        
        # Add mean duration
        fig.add_trace(go.Bar(
            x=years,
            y=mean_durations,
            name='Mean Duration',
            marker_color='blue',
            opacity=0.7
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Duration Analysis - {well_name}',
            xaxis_title='Year',
            yaxis_title='Duration (days)',
            template='plotly_white',
            height=500,
            barmode='group',
            showlegend=True
        )
        
        # Add summary statistics
        total_events = duration_results['summary']['total_events']
        avg_duration = duration_results['summary']['average_duration']
        
        fig.add_annotation(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=f'Total Events: {total_events}<br>Avg Duration: {avg_duration:.1f} days',
            showarrow=False,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating duration analysis plot: {e}")
        return None


def create_low_flow_analysis_plot(low_flow_results, well_name):
    """
    Create low flow analysis plot.
    
    Args:
        low_flow_results: Results from calculate_percentile_low_flow_metrics
        well_name: Display name of the well
    
    Returns:
        plotly.graph_objects.Figure: Plot figure
    """
    try:
        if 'error' in low_flow_results:
            return None
        
        # Extract data
        years = list(low_flow_results['yearly_statistics'].keys())
        q10_values = [low_flow_results['yearly_statistics'][year]['q10'] 
                     for year in years]
        q25_values = [low_flow_results['yearly_statistics'][year]['q25'] 
                     for year in years]
        q50_values = [low_flow_results['yearly_statistics'][year]['q50'] 
                     for year in years]
        
        # Create figure
        fig = go.Figure()
        
        # Add percentile lines
        fig.add_trace(go.Scatter(
            x=years,
            y=q10_values,
            mode='lines+markers',
            name='Q10 (10th percentile)',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=q25_values,
            mode='lines+markers',
            name='Q25 (25th percentile)',
            line=dict(color='orange', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=q50_values,
            mode='lines+markers',
            name='Q50 (median)',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Low Flow Analysis - {well_name}',
            xaxis_title='Year',
            yaxis_title='Groundwater Level (m)',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        # Add summary statistics
        avg_q10 = low_flow_results['summary']['average_q10']
        avg_q25 = low_flow_results['summary']['average_q25']
        
        fig.add_annotation(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=f'Avg Q10: {avg_q10:.3f} m<br>Avg Q25: {avg_q25:.3f} m',
            showarrow=False,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating low flow analysis plot: {e}")
        return None


def create_gwl_rolling_mean_plot_simple(data, well_name, save_dir, rolling_window=None):
    """
    Create simple GWL time series plot with rolling mean for a single well.
    
    Args:
        data: pandas Series with datetime index and groundwater levels
        well_name: Name of the well
        save_dir: Directory to save the plot
        rolling_window: Rolling window size for smoothing (default 30 days)
    
    Returns:
        str: Path to saved plot file, or None if error
    """
    # Use global rolling window if none specified
    if rolling_window is None:
        rolling_window = ROLLING_WINDOW_SIZE
        print(f"Using global rolling window size: {rolling_window} days")
    
    try:
        # Ensure data has datetime index using our robust function
        data = ensure_datetime_index(data)
        if not isinstance(data.index, pd.DatetimeIndex):
            print(f"Error: Data for {well_name} must have datetime index or DateTime column")
            return None
        
        # Remove NaN values
        clean_data = data.dropna()
        # No minimum data requirement - analyze whatever data exists
        if len(clean_data) == 0:
            print(f"Error: No data available for {well_name}")
            return None
        
        # Resample to daily frequency for consistent plotting
        daily_data = clean_data.resample('D').mean()
        
        # Create figure 
        fig = go.Figure()
        
        # Calculate rolling mean using daily resampled data for consistency
        rolling_mean = daily_data.rolling(window=rolling_window, center=False).mean()
        
        # Add daily resampled data with professional styling
        fig.add_trace(go.Scatter(
            x=daily_data.index,
            y=daily_data.values,
            mode='lines',
            name='Daily GWL (Resampled)',
            line=dict(
                color='#2E86AB',  # Professional blue
                width=3.0,        # Increased from 1.5 to 3.0
                shape='linear'
            ),
            opacity=0.5,          # Reduced from 0.7 to 0.5 for better contrast
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>GWL:</b> %{y:.3f} m<extra></extra>',
            hoverlabel=dict(
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#2E86AB',
                font=dict(size=12, color='#1A1A1A')
            )
        ))
        
        # Add rolling mean with much more prominent styling
        rolling_label = f'Rolling Mean ({rolling_window} days)'
            
        fig.add_trace(go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean.values,
            mode='lines+markers',  # Added markers for better visibility
            name=rolling_label,
            line=dict(
                color='#FF4500',  # Changed to bright orange-red for visibility
                width=6.0,        # Increased from 3.5 to 6.0 - much thicker
                shape='linear'
            ),
            marker=dict(
                size=4,           # Small markers for key points
                color='#FF4500',
                opacity=0.8
            ),
            opacity=1.0,          # Full opacity for maximum visibility
            fill='tonexty',       # Fill area below the line for visibility
            fillcolor='rgba(255, 69, 0, 0.1)',  # Very light orange fill
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Rolling Mean:</b> %{y:.3f} m<extra></extra>',
            hoverlabel=dict(
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#FF4500',
                font=dict(size=12, color='#1A1A1A')
            )
        ))
        # layout 
        fig.update_layout(
            title=dict(
                text=f'Groundwater Level Time Series - {well_name}',
                font=dict(size=20, color='#1A1A1A', family='Arial, sans-serif', weight='bold'),
                x=0.5,
                xanchor='center',
                y=0.95
            ),
            xaxis_title=dict(
                text='Year',
                font=dict(size=14, color='#1A1A1A', family='Arial, sans-serif', weight='bold')
            ),
            yaxis_title=dict(
                text='Groundwater Level (m)',
                font=dict(size=14, color='#1A1A1A', family='Arial, sans-serif', weight='bold')
            ),
            hovermode='x unified',
            template='plotly_white',
            height=700,
            width=1000,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#2E86AB',
                borderwidth=2,
                font=dict(size=13, color='#1A1A1A', weight='bold')
            ),
            margin=dict(l=100, r=100, t=120, b=100),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='#FFFFFF'
        )
        # Professional axis styling with better grid for visibility
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1.0,        # Reduced grid width for subtlety
            gridcolor='#F0F0F0',  # Lighter grid color
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='#2E86AB',
            tickfont=dict(size=13, color='#1A1A1A', weight='bold'),
            tickformat='%Y-%m',
            dtick='M12',  # Show every 12 months
            tickangle=0,
            tickmode='auto',
            nticks=8
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=1.0,        # Reduced grid width for subtlety
            gridcolor='#F0F0F0',  # Lighter grid color
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='#2E86AB',
            tickfont=dict(size=13, color='#1A1A1A', weight='bold'),
            tickformat='.3f',     # Changed to 3 decimal places for precision
            tickmode='auto',
            nticks=8
        )
        
        # Save plot
        plot_path = os.path.join(save_dir, f"{well_name}_gwl_rolling_mean.html")
        
        try:
            fig.write_html(plot_path)
            print(f"✓ Successfully saved GWL rolling mean plot: {plot_path}")
            
            return plot_path
        except Exception as e:
            print(f"✗ Error saving plot: {e}")
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f"Error creating GWL rolling mean plot for {well_name}: {e}")
        return None


def create_individual_well_plots(well_data, well_name, save_dir, mapping_dict=None, 
                                rolling_window=None, seasonal_period=12, alpha=None, save_plots=None):
    """
    Create comprehensive individual well plots including GWL, rolling mean, and seasonal analysis.
    
    Args:
        well_data: pandas Series with datetime index and groundwater levels
        well_name: Name of the well
        save_dir: Directory to save plots
        mapping_dict: Dictionary mapping well names to display names
        rolling_window: Rolling window size for smoothing (default 30 days)
        seasonal_period: Seasonal period for analysis (default 12 months)
        alpha: Significance level for trend tests
        save_plots: Whether to save plots (default uses global SAVE_PLOTS setting)
    
    Returns:
        list: Paths to saved plot files
    """
    # Use global settings if none specified
    if alpha is None:
        alpha = ALPHA_LEVEL
        print(f"Using global alpha level: {alpha}")
    if save_plots is None:
        save_plots = SAVE_PLOTS
        print(f"Using global save plots setting: {save_plots}")
    
    # Use global rolling window if none specified
    if rolling_window is None:
        rolling_window = ROLLING_WINDOW_SIZE
        print(f"Using global rolling window size: {rolling_window} days")
    
    try:
        # Get well display name
        well_display = generate_well_display_name(well_name, mapping_dict)
        
        # Ensure data has datetime index using our robust function
        well_data = ensure_datetime_index(well_data)
        if not isinstance(well_data.index, pd.DatetimeIndex):
            print(f"Error: Data for {well_name} must have datetime index or DateTime column")
            return []
        
        # Remove NaN values and resample to daily frequency for consistency with batch analysis
        clean_data = well_data.dropna()
        if len(clean_data) == 0:
            print(f"Error: No data available for {well_name}")
            return []
        
        # Resample to daily frequency to match batch analysis preprocessing
        clean_data = clean_data.resample('D').mean()
        print(f"Resampled {well_name} data to daily frequency for consistency with batch analysis")
        
        saved_plots = []
        
        # 1. GWL Time Series with Rolling Mean
        fig1 = create_gwl_rolling_mean_plot(clean_data, well_display, rolling_window)
        if fig1 and save_plots:
            plot_path1 = os.path.join(save_dir, generate_well_filename(well_name, "_gwl_rolling_mean.html", mapping_dict))
            fig1.write_html(plot_path1)
            saved_plots.append(plot_path1)
            print(f"Saved GWL rolling mean plot: {plot_path1}")
        elif fig1:
            print(f"GWL rolling mean plot created but not saved (save_plots=False)")
        
        # 2. Seasonal Analysis Plot
        fig2 = create_seasonal_analysis_plot(clean_data, well_display, seasonal_period)
        if fig2 and save_plots:
            plot_path2 = os.path.join(save_dir, generate_well_filename(well_name, "_seasonal_analysis.html", mapping_dict))
            fig2.write_html(plot_path2)
            saved_plots.append(plot_path2)
            print(f"Saved seasonal analysis plot: {plot_path2}")
        elif fig2:
            print(f"Seasonal analysis plot created but not saved (save_plots=False)")
        
        # 3. Mann-Kendall Trend Analysis Plot
        fig3 = create_mk_trend_plot(clean_data, well_display, alpha)
        if fig3 and save_plots:
            plot_path3 = os.path.join(save_dir, generate_well_filename(well_name, "_mk_trend_analysis.html", mapping_dict))
            fig3.write_html(plot_path3)
            saved_plots.append(plot_path3)
            print(f"Saved MK trend analysis plot: {plot_path3}")
        elif fig3:
            print(f"MK trend analysis plot created but not saved (save_plots=False)")
        
        # 4. Duration Analysis Plot
        duration_results = calculate_yearly_max_duration_metrics(clean_data)
        if 'error' not in duration_results:
            fig4 = create_duration_analysis_plot(duration_results, well_display)
            if fig4 and save_plots:
                plot_path4 = os.path.join(save_dir, generate_well_filename(well_name, "_duration_analysis.html", mapping_dict))
                fig4.write_html(plot_path4)
                saved_plots.append(plot_path4)
                print(f"Saved duration analysis plot: {plot_path4}")
            elif fig4:
                print(f"Duration analysis plot created but not saved (save_plots=False)")
        
        # 5. Low Flow Analysis Plot
        low_flow_results = calculate_percentile_low_flow_metrics(clean_data)
        if 'error' not in low_flow_results:
            fig5 = create_low_flow_analysis_plot(low_flow_results, well_display)
            if fig5 and save_plots:
                plot_path5 = os.path.join(save_dir, generate_well_filename(well_name, "_low_flow_analysis.html", mapping_dict))
                fig5.write_html(plot_path5)
                saved_plots.append(plot_path5)
                print(f"Saved low flow analysis plot: {plot_path5}")
            elif fig5:
                print(f"Low flow analysis plot created but not saved (save_plots=False)")
        
        return saved_plots
        
    except Exception as e:
        print(f"Error creating plots for {well_name}: {e}")
        return []


__all__ = [
    'create_gwl_rolling_mean_plot',
    'create_seasonal_analysis_plot',
    'create_mk_trend_plot',
    'create_duration_analysis_plot',
    'create_low_flow_analysis_plot',
    'create_gwl_rolling_mean_plot_simple',
    'create_individual_well_plots'
]
