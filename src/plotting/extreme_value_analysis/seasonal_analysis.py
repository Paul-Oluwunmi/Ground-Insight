"""
Seasonal pattern analysis functions
"""

import logging
import traceback
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict
from .constants import SEASON_COLORS


def get_seasons_numpy(timestamps: np.ndarray) -> np.ndarray:
    """
    Convert timestamps to seasons using numpy operations.
    
    Parameters:
    -----------
    timestamps : np.ndarray
        Unix timestamps (seconds since epoch)
        
    Returns:
    --------
    np.ndarray : Array of season names
    """
    # Convert Unix timestamps to datetime64
    dates = np.datetime64('1970-01-01') + np.timedelta64(1, 's') * timestamps
    
    # Extract months using numpy datetime operations
    months = dates.astype('datetime64[M]').astype(int) % 12 + 1
    
    # Define seasons using numpy operations
    seasons = np.full_like(months, 'Unknown', dtype='object')
    seasons[(months == 12) | (months <= 2)] = 'Summer'
    seasons[(months >= 3) & (months <= 5)] = 'Autumn'
    seasons[(months >= 6) & (months <= 8)] = 'Winter'
    seasons[(months >= 9) & (months <= 11)] = 'Spring'
    
    return seasons


def analyse_seasonal_patterns(block_maxima: pd.Series) -> dict:
    """Analyse seasonal patterns in block maxima"""
    try:
        # Ensure we're working with a Series that has a datetime index
        if not isinstance(block_maxima.index, pd.DatetimeIndex):
            block_maxima.index = pd.to_datetime(block_maxima.index)

        # Create DataFrame with the values and their dates
        block_maxima_df = pd.DataFrame({
            'value': block_maxima,
            'month': block_maxima.index.month,
            'date': block_maxima.index
        })

        # Define NZ seasons (months are 1-12)
        nz_seasons = {
            'Summer': [12, 1, 2],     # December to February
            'Autumn': [3, 4, 5],      # March to May
            'Winter': [6, 7, 8],      # June to August
            'Spring': [9, 10, 11]     # September to November
        }
        
        # Categorize each maximum by season
        def get_nz_season(month):
            for season, months in nz_seasons.items():
                if month in months:
                    return season
            return 'Unknown'  # This shouldn't happen with valid months
            
        block_maxima_df['season'] = block_maxima_df['month'].map(get_nz_season)
        
        # Verify all points are categorized
        uncategorized = block_maxima_df[block_maxima_df['season'] == 'Unknown']
        if not uncategorized.empty:
            logging.warning(f"Found {len(uncategorized)} uncategorized points with months: {uncategorized['month'].unique()}")

        # Calculate statistics for each season
        seasonal_stats = {'seasons': {}}
        
        for season in nz_seasons.keys():
            season_data = block_maxima_df[block_maxima_df['season'] == season]
            if not season_data.empty:
                seasonal_stats['seasons'][season] = {
                    'count': len(season_data),
                    'mean': season_data['value'].mean(),
                    'max': season_data['value'].max(),
                    'min': season_data['value'].min(),
                    'std': season_data['value'].std(),
                    'dates': season_data['date'].tolist(),
                    'values': season_data['value'].tolist()  
                }

        # Add total count for verification
        seasonal_stats['total_points'] = len(block_maxima_df)
        
        # Log counts for verification
        total_categorized = sum(stats['count'] for stats in seasonal_stats['seasons'].values())
        logging.info(f"Total points: {len(block_maxima_df)}")
        logging.info(f"Total categorized: {total_categorized}")
        if total_categorized != len(block_maxima_df):
            logging.error(f"Mismatch in point counts: {total_categorized} categorized vs {len(block_maxima_df)} total")
            
        return seasonal_stats
        
    except Exception as e:
        logging.error(f"Error in analyse_seasonal_patterns: {str(e)}\n{traceback.format_exc()}")
        return {'seasons': {}, 'total_points': 0}


def add_seasonal_annotation(fig, seasonal_stats, data_type):
    """Add seasonal analysis to the plot"""
    try:
        # Collect all dates and values from each season
        all_dates = []
        all_values = []
        for season, stats in seasonal_stats['seasons'].items():
            if stats['count'] > 0:
                all_dates.extend(stats['dates'])
                all_values.extend(stats['values'])

        # Sort by date
        sorted_indices = np.argsort(all_dates)
        all_dates = np.array(all_dates)[sorted_indices]
        all_values = np.array(all_values)[sorted_indices]

        # Add time series line
        fig.add_trace(
            go.Scatter(
                x=all_dates,
                y=all_values,
                mode='lines',
                name='Time Series',
                line=dict(color='gray', width=1),
                showlegend=True
            ),
            row=1, col=1
        )

        # Add seasonal maxima points
        for season, stats in seasonal_stats['seasons'].items():
            if stats['count'] > 0:
                fig.add_trace(
                    go.Scatter(
                        x=stats['dates'],
                        y=stats['values'],
                        mode='markers',
                        name=f'{season} Maxima (n={stats["count"]})',
                        marker=dict(
                            color=SEASON_COLORS[season],
                            size=8,
                            symbol='circle'
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )

        # Update axes for seasonal plot
        fig.update_xaxes(
            title_text="Date",
            row=1, col=1,  
            tickformat="%Y"
        )
        
        value_label = 'Rainfall (mm)' if data_type == 'rain' else 'Groundwater Level (m)'
        fig.update_yaxes(
            title_text=f"Maximum {value_label}",
            row=1, col=1  
        )

        return fig

    except Exception as e:
        logging.error(f"Error in add_seasonal_annotation: {str(e)}")
        traceback.print_exc()
        return fig

