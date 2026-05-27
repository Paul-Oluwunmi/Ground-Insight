"""
Plotting functions for statistical analysis visualization
"""

import plotly.graph_objects as go
from dash import dcc
from .statistics import calculate_statistics, format_significance_text


def create_box_plot(datasets, wells_selected, years_selected, months_selected):
    """
    Create box plot with statistical comparisons.
    
    Parameters:
    -----------
    datasets : list
        List of dataset dictionaries with 'label', 'data', and 'index'
    wells_selected : list
        List of selected well names
    years_selected : list
        List of selected years
    months_selected : list
        List of selected months
        
    Returns:
    --------
    dcc.Graph : Plotly graph component
    """
    fig = go.Figure()
    annotations = []
    comparison_count = 0
    
    # Add box plots for each dataset
    for dataset in datasets:
        fig.add_trace(go.Box(
            y=dataset['data'],
            name=dataset['label'],
            boxmean=True,
            marker_color=f'hsl({dataset["index"] * 360/len(datasets)}, 70%, 50%)',
            width=0.8
        ))
    
    # Calculate number of wells
    num_wells = len([w for w in wells_selected if w])
    
    # Statistical comparisons
    if len(datasets) >= 2:
        max_y = max([max(d['data']) for d in datasets])
        min_y = min([min(d['data']) for d in datasets])
        y_range = max_y - min_y
        
        num_comparisons = sum(range(len(datasets)))
        base_spacing = y_range * (0.3 if num_comparisons <= 3 else 0.35)
        
        for i, d1 in enumerate(datasets):
            for j, d2 in enumerate(datasets[i+1:], i+1):
                current_level = comparison_count
                spacing = base_spacing * (1 + current_level * 1.5)
                
                # Calculate statistics
                stats_result = calculate_statistics(d1['data'], d2['data'])
                color = 'red' if stats_result['is_significant'] else 'green'
                
                # Get well names and dates
                well1_parts = d1['label'].split('\n')
                well2_parts = d2['label'].split('\n')
                well1_name = well1_parts[0]
                well2_name = well2_parts[0]
                well1_date = well1_parts[1].strip('()')
                well2_date = well2_parts[1].strip('()')
                
                # Format significance text
                significance_text = format_significance_text(
                    well1_name, well1_date, well2_name, well2_date,
                    stats_result['mean_diff'], stats_result
                )
                
                # Add connecting lines
                fig.add_shape(
                    type='line',
                    x0=d1['index'],
                    x1=d2['index'],
                    y0=max_y + spacing,
                    y1=max_y + spacing,
                    line=dict(color=color, width=2)
                )
                
                # Add vertical lines
                for x in [d1['index'], d2['index']]:
                    fig.add_shape(
                        type='line',
                        x0=x,
                        x1=x,
                        y0=max_y + spacing - base_spacing/6,
                        y1=max_y + spacing,
                        line=dict(color=color, width=2)
                    )
                
                # Smart annotation positioning
                annotation_x = (d1['index'] + d2['index']) / 2
                annotation_y = max_y + spacing + base_spacing/3
                
                # Adjust font size based on number of wells
                if num_wells <= 4:
                    font_size = 12
                elif num_wells <= 8:
                    font_size = 10
                else:
                    font_size = 9
                
                # Add annotation
                annotations.append(dict(
                    x=annotation_x,
                    y=annotation_y,
                    text=significance_text,
                    showarrow=False,
                    font=dict(size=font_size),
                    bgcolor='white',
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=4,
                    align='center',
                    xanchor='center',
                    yanchor='bottom'
                ))
                
                comparison_count += 1
    
    # Calculate plot dimensions
    plot_width, height, tick_angle = _calculate_plot_dimensions(num_wells, len(annotations))
    
    # Create title
    if num_wells > 8:
        title_text = "Statistical Comparison of Groundwater Levels (Many wells - scroll horizontally)"
    else:
        title_text = "Statistical Comparison of Groundwater Levels"
    
    # Update layout
    fig.update_layout(
        title={
            'text': title_text,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, weight='bold')
        },
        yaxis_title={
            'text': "Groundwater Level (m)",
            'font': dict(size=14, weight='bold')
        },
        xaxis_title={
            'text': "Wells and Time Periods",
            'font': dict(size=14, weight='bold')
        },
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=plot_width,
        height=height,
        margin=dict(t=120, b=120, l=100, r=100),
        annotations=annotations,
        xaxis=dict(
            tickangle=tick_angle, 
            tickmode='array',
            ticktext=[f"{wells_selected[i]}\n({months_selected[i][:3]}-{str(years_selected[i])[2:]})" 
                     for i in range(len(wells_selected)) if wells_selected[i] and years_selected[i] and months_selected[i]],
            tickvals=[i for i in range(num_wells)],
            tickfont=dict(size=12, color='#2c3e50', weight='bold'),
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            domain=[0.02, 0.98],
            title_font=dict(size=14, weight='bold', color='#2c3e50'),
            ticklabelstep=1
        ),
        bargap=0.3,
        bargroupgap=0.1
    )
    
    # Update y-axis settings
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='LightGray',
        tickfont=dict(size=12, color='#2c3e50', weight='bold')
    )
    
    return dcc.Graph(
        figure=fig,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'scrollZoom': True
        }
    )


def _calculate_plot_dimensions(num_wells, num_annotations):
    """Calculate plot width, height, and tick angle based on number of wells."""
    # Smart spacing based on number of wells
    if num_wells <= 4:
        min_box_width = 350
        box_spacing = 150
        tick_angle = 0
    elif num_wells <= 8:
        min_box_width = 300
        box_spacing = 120
        tick_angle = 15
    else:
        min_box_width = 250
        box_spacing = 80
        tick_angle = 30
    
    # Calculate total width needed
    total_box_width = num_wells * min_box_width
    total_spacing = (num_wells - 1) * box_spacing
    plot_width = max(1200, total_box_width + total_spacing + 200)
    
    height = min(1200, max(800, 600 + (num_annotations * 50)))
    
    return plot_width, height, tick_angle

