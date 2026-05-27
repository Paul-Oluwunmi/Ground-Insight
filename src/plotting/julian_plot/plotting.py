"""
Julian Plot Plotting Functions
===============================

This module contains functions for creating Julian-Day plots for groundwater data.
"""

import polars as pl
import plotly.graph_objects as go


def create_julian_plot(data):
    """
    Create an interactive Julian-Day plot using processed data.
    
    Parameters:
    -----------
    data : pl.DataFrame or pd.DataFrame
        Data containing DateTime and well columns
        
    Returns:
    --------
    go.Figure or None
        Plotly figure object, or None if error occurs
    """
    try:
        print("\nProcessing and generating Julian-Day plot...")
        
        # Convert pandas DataFrame to polars if needed
        if not isinstance(data, pl.DataFrame):
            data = pl.from_pandas(data)
        
        # Ensure DateTime column exists and is datetime type
        if 'DateTime' not in data.columns:
            raise ValueError("Data must contain a 'DateTime' column")
        
        # Create plot data with Julian Day and Year using with_columns
        plot_data = data.with_columns([
            pl.col('DateTime').dt.ordinal_day().alias('Julian_Day'),
            pl.col('DateTime').dt.year().alias('Year')
        ])
        
        # Get the most recent year
        max_year = plot_data['Year'].max()
        
        # Extract well columns (exclude DateTime, Julian_Day, mm_Rain, Year)
        wells = [col for col in plot_data.columns if col not in ['DateTime', 'Julian_Day', 'mm_Rain', 'Year']]
        
        if not wells:
            raise ValueError("No well columns found in data")

        # Create a figure
        fig = go.Figure()

        # Add traces for each well (historical and latest year)
        for i, well_name in enumerate(wells):
            # Historical data (excluding latest year)
            historical_data = plot_data.filter(pl.col('Year') < max_year)
            
            # Filter out null values for this well in historical data
            if len(historical_data) > 0 and well_name in historical_data.columns:
                hist_filtered = historical_data.filter(pl.col(well_name).is_not_null())
                hist_julian = hist_filtered.get_column('Julian_Day').to_list() if len(hist_filtered) > 0 else []
                hist_values = hist_filtered.get_column(well_name).to_list() if len(hist_filtered) > 0 else []
            else:
                hist_julian = []
                hist_values = []
            
            # Add historical trace (always add, even if empty)
            fig.add_trace(go.Scattergl(
                x=hist_julian,
                y=hist_values,
                mode='lines',
                name=f"{well_name} (Historical)",
                visible=False,
                line=dict(color='rgba(255, 127, 14, 0.3)'), 
                hovertemplate=(
                    f"<b>Well: {well_name}</b><br>"
                    "Julian Day: %{x}<br>"
                    "Groundwater Level: %{y:.2f} m<br>"
                    "<extra></extra>"
                )
            ))
            
            # Latest year data
            latest_data = plot_data.filter(pl.col('Year') == max_year)
            
            # Filter out null values for this well in latest year data
            if len(latest_data) > 0 and well_name in latest_data.columns:
                latest_filtered = latest_data.filter(pl.col(well_name).is_not_null())
                latest_julian = latest_filtered.get_column('Julian_Day').to_list() if len(latest_filtered) > 0 else []
                latest_values = latest_filtered.get_column(well_name).to_list() if len(latest_filtered) > 0 else []
            else:
                latest_julian = []
                latest_values = []
            
            # Add latest year trace (always add, even if empty)
            fig.add_trace(go.Scattergl(
                x=latest_julian,
                y=latest_values,
                mode='lines',
                name=f"{well_name} ({max_year})",
                visible=False,
                line=dict(color='#1f77b4', width=2),  
                hovertemplate=(
                    f"<b>Well: {well_name}</b><br>"
                    "Julian Day: %{x}<br>"
                    "Groundwater Level: %{y:.2f} m<br>"
                    f"Year: {max_year}<br>"
                    "<extra></extra>"
                )
            ))

        # Create dropdown menu buttons
        # Total traces = len(wells) * 2 (historical + latest for each well)
        total_traces = len(wells) * 2
        
        buttons = [
            dict(
                label="Select Well",
                method="update",
                args=[{"visible": [False] * total_traces},
                      {"title": "Julian-Day Plot - Select a Well"}]
            )
        ]
        
        # Add individual well buttons
        for i, well_name in enumerate(wells):
            visibility = [False] * total_traces
            # Each well has 2 traces: historical (i*2) and latest (i*2+1)
            visibility[i*2] = True      # Historical trace
            visibility[i*2 + 1] = True  # Latest year trace
            buttons.append(dict(
                label=well_name,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Julian-Day Plot for {well_name}"}]
            ))

        # Update layout with dropdown and styling
        fig.update_layout(
            title=dict(
                text="Julian-Day Plot - Select a Well", 
                x=0.5,
                xanchor='center',
                font=dict(size=20, weight='bold')
            ),
            xaxis=dict(
                title=dict(
                    text="Julian Day (0-365)",
                    font=dict(size=14, weight='bold')  
                ),
                tickfont=dict(size=12, weight='bold')
            ),
            yaxis=dict(
                title=dict(
                    text="Groundwater Level (m)",
                    font=dict(size=14, weight='bold')  
                ),
                tickfont=dict(size=12, weight='bold')
            ),
            template="plotly_white",
            legend_title="Wells",
            hovermode="x unified",
            width=1600,
            height=800,
            updatemenus=[dict(
                active=0,  
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.15,
                xanchor="left",
                y=1.10,
                yanchor="top",
                font=dict(size=12, weight='bold'),
                bgcolor='white',
                bordercolor='#999999'
            )],
            margin=dict(t=100, r=200)
        )

        return fig

    except Exception as e:
        print(f"Error generating Julian-Day plot: {str(e)}")
        return None

