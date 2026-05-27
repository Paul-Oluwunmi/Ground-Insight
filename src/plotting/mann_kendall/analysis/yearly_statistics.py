"""
Yearly Statistics Analysis Module

This module contains functions for calculating yearly water level statistics.
"""

import pandas as pd
import numpy as np
from ..utils import ensure_datetime_index, get_nz_season


def calculate_yearly_water_level_statistics(series):
    """
    Calculate basic yearly water level statistics (mean, min, max, std) for each year.
    
    IMPORTANT: This function should use ORIGINAL groundwater level data, not resampled data.
    Resampling would change the actual measurement values and timing, which is not appropriate
    for yearly statistics that need to preserve the real data characteristics.
    
    Args:
        series: pandas Series with datetime index and ORIGINAL groundwater levels (not resampled)
    
    Returns:
        dict: Complete yearly water level statistics results
    """
    try:
        if series is None or len(series) == 0:
            return {'error': 'No data available'}
        
        # Ensure data has datetime index using our robust function
        series = ensure_datetime_index(series)
        if not isinstance(series.index, pd.DatetimeIndex):
            return {'error': 'Data must have datetime index or DateTime column'}
        # Check if we have sufficient data for meaningful analysis
        if len(series) < 2:
            return {'error': f'Insufficient data: only {len(series)} data point(s) available - need at least 2 for meaningful analysis'}
        
        # Group by year for annual statistics
        years = sorted(series.index.year.unique())
        yearly_stats = []
        
        # Check if we have data spanning multiple years
        if len(years) == 1 and len(series) < 10:
            print(f"Warning: Data spans only one year ({years[0]}) with {len(series)} points - may not provide meaningful yearly analysis")
        
        for year in years:
            year_data = series[series.index.year == year]
            
            # No minimum data requirement - analyze whatever data exists for each year
            if len(year_data) == 0:
                continue
            
            # Calculate basic statistics for this year
            mean_level = year_data.mean()
            max_level = year_data.max()
            min_level = year_data.min()
            
            # Get DATES for min and max
            min_date = year_data.idxmin() if len(year_data) > 0 else np.nan
            max_date = year_data.idxmax() if len(year_data) > 0 else np.nan
            
            # Get the season for the max and min dates
            max_season = get_nz_season(max_date) if pd.notna(max_date) else 'N/A'
            min_season = get_nz_season(min_date) if pd.notna(min_date) else 'N/A'
            
            # Create entries for both max and min occurrences
            if pd.notna(max_date):
                yearly_stats.append({
                    'year': max_date.strftime('%b-%Y'),
                    'season': max_season,
                    'max_level_m': max_level,
                    'min_level_m': '-',
                    'mean_level_m': mean_level
                })
            
            if pd.notna(min_date) and min_date != max_date:
                yearly_stats.append({
                    'year': min_date.strftime('%b-%Y'),
                    'season': min_season,
                    'max_level_m': '-',
                    'min_level_m': min_level,
                    'mean_level_m': mean_level
                })
        
        # Calculate overall seasonal summary statistics
        if yearly_stats:
            # Get overall maximum and minimum with seasons
            overall_max = series.max()
            overall_max_date = series.idxmax()
            overall_max_season = get_nz_season(overall_max_date)
            
            overall_min = series.min()
            overall_min_date = series.idxmin()
            overall_min_season = get_nz_season(overall_min_date)
            
            # Count occurrences of max and min by season from yearly data
            max_season_counts = {}
            min_season_counts = {}
            
            for stat in yearly_stats:
                # For yearly water level analysis, we'll use the year to determine season
                year = stat['year']
                # Count by year for now - you can modify this logic as needed
                max_season_counts[year] = max_season_counts.get(year, 0) + 1
                min_season_counts[year] = min_season_counts.get(year, 0) + 1
            
            # Format season counts (simplified for now)
            max_season_summary = ', '.join([f"Year {year} {count}" for year, count in max_season_counts.items()])
            min_season_summary = ', '.join([f"Year {year} {count}" for year, count in min_season_counts.items()])
            
            # Add seasonal summary to the first yearly entry
            if yearly_stats:
                yearly_stats[0].update({
                    'Overall_Max_Level_m': overall_max,
                    'Overall_Max_Season': overall_max_season,
                    'Overall_Max_Date': overall_max_date.strftime('%b-%Y'),
                    'Overall_Max_Description': 'Highest water level recorded',
                    'Overall_Min_Level_m': overall_min,
                    'Overall_Min_Season': overall_min_season,
                    'Overall_Min_Date': overall_min_date.strftime('%b-%Y'),
                    'Overall_Min_Description': 'Lowest water level recorded',
                    'Max_Season_Count': max_season_summary,
                    'Min_Season_Count': min_season_summary
                })
        
        return {
            'yearly_stats': yearly_stats,
            'total_years': len(yearly_stats),
            'original_data': series  # Add original data for overall statistics
        }
        
    except Exception as e:
        print(f"Error in calculate_yearly_water_level_statistics: {e}")
        return {'error': str(e)}


__all__ = [
    'calculate_yearly_water_level_statistics',
]
