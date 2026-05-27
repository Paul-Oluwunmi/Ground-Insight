"""
Hydrological Year Analysis Module

This module contains functions for hydrological year statistics analysis.
"""

import pandas as pd
import numpy as np
from ..constants import ROLLING_WINDOW_SIZE
from ..utils import (
    ensure_datetime_index,
    get_nz_season,
    generate_well_display_name
)


def calculate_hydrological_year_statistics(data, well_name, mapping_dict=None, include_well_column=False):
    """
    Calculate comprehensive hydrological year statistics (July 1st to June 30th).
    Works for both individual wells and all-wells analysis.
    
    IMPORTANT: This function should use ORIGINAL groundwater level data, not resampled data.
    The hydrological year analysis needs to preserve actual measurement values and timing
    to accurately show seasonal patterns and extreme values.
    
    Args:
        data: pandas Series with datetime index and ORIGINAL groundwater levels (not resampled)
        well_name: Name of the well being analysed
        mapping_dict: Dictionary mapping well names to display names
        include_well_column: True to add 'Well' column (for all-wells CSV), False for individual well CSV
    
    Returns:
        list: List of dictionaries containing hydrological year statistics
    """
    try:
        if data is None or len(data) == 0:
            print(f"Warning: No data available for {well_name}")
            return []
        
        # Ensure data has datetime index using our robust function
        data = ensure_datetime_index(data)
        if not isinstance(data.index, pd.DatetimeIndex):
            print(f"Warning: Data for {well_name} does not have datetime index or DateTime column")
            return []
        
        # Check if we have sufficient data for meaningful analysis
        if len(data) < 2:
            print(f"Warning: {well_name} has only {len(data)} data point(s) - insufficient for hydrological year analysis")
            return []
        
        # Group by hydrological year (July 1st to June 30th)
        hydrological_stats = []
        
        # Get the range of years in the data
        min_year = data.index.min().year
        max_year = data.index.max().year
        
        # Check if we have data spanning multiple years
        if min_year == max_year and len(data) < 10:
            print(f"Warning: {well_name} has data from only one year ({min_year}) with {len(data)} points - may not provide meaningful hydrological year analysis")
        
        # Process each hydrological year
        for start_year in range(min_year, max_year + 1):
            try:
                # Define hydrological year boundaries (July 1st to June 30th)
                start_date = pd.Timestamp(f"{start_year}-07-01")
                end_date = pd.Timestamp(f"{start_year + 1}-06-30")
                
                # Get data for this hydrological year
                hydro_year_data = data[(data.index >= start_date) & (data.index <= end_date)]
                
                # No minimum data requirement - analyze whatever data exists
                if len(hydro_year_data) == 0:
                    continue
                
                # Calculate basic statistics
                mean_level = hydro_year_data.mean()
                max_level = hydro_year_data.max()
                min_level = hydro_year_data.min()
                median_level = hydro_year_data.median()
                std_level = hydro_year_data.std()
                
                # Get DATES for min and max
                min_date = hydro_year_data.idxmin() if len(hydro_year_data) > 0 else np.nan
                max_date = hydro_year_data.idxmax() if len(hydro_year_data) > 0 else np.nan
                
                # Calculate MALF (Mean Annual Low Flow) - average of daily minimums
                daily_minima = hydro_year_data.resample('D').min().dropna()
                malf = daily_minima.mean() if len(daily_minima) > 0 else min_level
                malf_std = daily_minima.std() if len(daily_minima) > 1 else 0
                
                # Create hydrological year statistics dictionary
                hydro_year_stat = {
                    'Hydrological_Year': f"{start_year}-{start_year + 1}",
                    'Min_Season': get_nz_season(min_date) if pd.notna(min_date) else 'N/A',
                    'Minimum_Level_m': min_level,
                    'Min_Date': min_date.strftime('%d-%b-%Y') if pd.notna(min_date) else 'N/A',
                    'Max_Season': get_nz_season(max_date) if pd.notna(max_date) else 'N/A',
                    'Maximum_Level_m': max_level,
                    'Max_Date': max_date.strftime('%d-%b-%Y') if pd.notna(max_date) else 'N/A',
                    'Mean_Level_m': mean_level,
                    'Median_Level_m': median_level,
                    'Std_Dev_m': std_level,
                    'MALF_m': malf,
                    'MALF_Std_Dev_m': malf_std,
                    'Data_Points': len(hydro_year_data)
                }
                
                # Add Well column if requested (for all-wells CSV)
                if include_well_column:
                    well_display = generate_well_display_name(well_name, mapping_dict)
                    hydro_year_stat['Well'] = well_display
                
                hydrological_stats.append(hydro_year_stat)
                
            except Exception as e:
                print(f"Error calculating hydrological year statistics for {start_year}-{start_year + 1} in {well_name}: {e}")
                continue
        
        # Sort by hydrological year
        hydrological_stats.sort(key=lambda x: x['Hydrological_Year'])
        
        # Add seasonal summary statistics
        if hydrological_stats:
            # Calculate overall seasonal statistics
            all_data = data.copy()
            
            # Get overall maximum and minimum with seasons
            overall_max = all_data.max()
            overall_max_date = all_data.idxmax()
            overall_max_season = get_nz_season(overall_max_date)
            
            overall_min = all_data.min()
            overall_min_date = all_data.idxmin()
            overall_min_season = get_nz_season(overall_min_date)
            
            # Count occurrences of max and min by season
            max_season_counts = {}
            min_season_counts = {}
            
            for stat in hydrological_stats:
                max_season = stat['Max_Season']
                min_season = stat['Min_Season']
                
                if max_season != 'N/A':
                    max_season_counts[max_season] = max_season_counts.get(max_season, 0) + 1
                if min_season != 'N/A':
                    min_season_counts[min_season] = min_season_counts.get(min_season, 0) + 1
            
            # Format season counts
            max_season_summary = ', '.join([f"{season} {count}" for season, count in max_season_counts.items()])
            min_season_summary = ', '.join([f"{season} {count}" for season, count in min_season_counts.items()])
            
            # Add seasonal summary to the first hydrological year entry
            if hydrological_stats:
                hydrological_stats[0]['Overall_Max_Level_m'] = overall_max
                hydrological_stats[0]['Overall_Max_Season'] = overall_max_season
                hydrological_stats[0]['Overall_Max_Date'] = overall_max_date.strftime('%d-%b-%Y')
                hydrological_stats[0]['Overall_Max_Description'] = f"Highest water level recorded"
                
                hydrological_stats[0]['Overall_Min_Level_m'] = overall_min
                hydrological_stats[0]['Overall_Min_Season'] = overall_min_season
                hydrological_stats[0]['Overall_Min_Date'] = overall_min_date.strftime('%d-%b-%Y')
                hydrological_stats[0]['Overall_Min_Description'] = f"Lowest water level recorded"
                
                hydrological_stats[0]['Max_Season_Count'] = max_season_summary
                hydrological_stats[0]['Min_Season_Count'] = min_season_summary
        
        print(f"Calculated hydrological year statistics for {well_name}: {len(hydrological_stats)} years")
        return hydrological_stats
        
    except Exception as e:
        print(f"Error in calculate_hydrological_year_statistics for {well_name}: {e}")
        return []


def calculate_hydrological_year_statistics_rolling(data, well_name, mapping_dict=None, include_well_column=False, rolling_window=None):
    """
    Calculate comprehensive hydrological year statistics using rolling mean smoothing.
    Works for both individual wells and all-wells analysis.
    
    IMPORTANT: This function should use 7-day rolling mean data derived from ORIGINAL groundwater levels.
    The rolling mean smooths out daily fluctuations while preserving seasonal trends and patterns.
    
    Args:
        data: pandas Series with datetime index and 7-day rolling mean groundwater levels
        well_name: Name of the well being analysed
        mapping_dict: Dictionary mapping well names to display names
        include_well_column: True to add 'Well' column (for all-wells CSV), False for individual well CSV
        rolling_window: Rolling window size for smoothing (default uses global ROLLING_WINDOW_SIZE)
    
    Returns:
        list: List of dictionaries containing hydrological year statistics (7-day smoothed)
    """
    try:
        # Use global rolling window size if none specified
        if rolling_window is None:
            rolling_window = ROLLING_WINDOW_SIZE
            print(f"Using global rolling window size: {rolling_window} days")
        
        if data is None or len(data) == 0:
            print(f"Warning: No data available for {well_name}")
            return []
        
        # Ensure data has datetime index using our robust function
        data = ensure_datetime_index(data)
        if not isinstance(data.index, pd.DatetimeIndex):
            print(f"Warning: Data for {well_name} does not have datetime index or DateTime column")
            return []
        
        # Apply rolling mean smoothing using global window size
        smoothed_data = data.rolling(window=rolling_window, center=False, min_periods=1).mean()
        
        # No minimum data requirement after smoothing - analyze whatever smoothed data exists
        if len(smoothed_data) == 0:
            print(f"Warning: No data available after {rolling_window}-day smoothing for {well_name}")
            return []
        
        # Group by hydrological year (July 1st to June 30th)
        hydrological_stats = []
        
        # Get the range of years in the smoothed data
        min_year = smoothed_data.index.min().year
        max_year = smoothed_data.index.max().year
        
        # Process each hydrological year
        for start_year in range(min_year, max_year + 1):
            try:
                # Define hydrological year boundaries (July 1st to June 30th)
                start_date = pd.Timestamp(f"{start_year}-07-01")
                end_date = pd.Timestamp(f"{start_year + 1}-06-30")
                
                # Get smoothed data for this hydrological year
                hydro_year_data = smoothed_data[(smoothed_data.index >= start_date) & (smoothed_data.index <= end_date)]
                
                # No minimum data requirement - analyze whatever data exists
                if len(hydro_year_data) == 0:
                    continue
                
                # Calculate basic statistics on smoothed data
                mean_level = hydro_year_data.mean()
                max_level = hydro_year_data.max()
                min_level = hydro_year_data.min()
                median_level = hydro_year_data.median()
                std_level = hydro_year_data.std()
                
                # Get DATES for min and max
                min_date = hydro_year_data.idxmin() if len(hydro_year_data) > 0 else np.nan
                max_date = hydro_year_data.idxmax() if len(hydro_year_data) > 0 else np.nan
                
                # Calculate MALF (Mean Annual Low Flow) - average of daily minimums from smoothed data
                daily_minima = hydro_year_data.resample('D').min().dropna()
                malf = daily_minima.mean() if len(daily_minima) > 0 else min_level
                malf_std = daily_minima.std() if len(daily_minima) > 1 else 0
                
                # Create hydrological year statistics dictionary (7-day smoothed)
                hydro_year_stat = {
                    'Hydrological_Year': f"{start_year}-{start_year + 1}",
                    'Min_Season': get_nz_season(min_date) if pd.notna(min_date) else 'N/A',
                    'Minimum_Level_m': min_level,
                    'Min_Date': min_date.strftime('%d-%b-%Y') if pd.notna(min_date) else 'N/A',
                    'Max_Season': get_nz_season(max_date) if pd.notna(max_date) else 'N/A',
                    'Maximum_Level_m': max_level,
                    'Max_Date': max_date.strftime('%d-%b-%Y') if pd.notna(max_date) else 'N/A',
                    'Mean_Level_m': mean_level,
                    'Median_Level_m': median_level,
                    'Std_Dev_m': std_level,
                    'MALF_m': malf,
                    'MALF_Std_Dev_m': malf_std,
                    'Data_Points': len(hydro_year_data),
                    'Smoothing_Applied': f"{rolling_window}-day rolling mean (global setting)"
                }
                
                # Add Well column if requested (for all-wells CSV)
                if include_well_column:
                    well_display = generate_well_display_name(well_name, mapping_dict)
                    hydro_year_stat['Well'] = well_display
                
                hydrological_stats.append(hydro_year_stat)
                
            except Exception as e:
                print(f"Error calculating 7-day smoothed hydrological year statistics for {start_year}-{start_year + 1} in {well_name}: {e}")
                continue
        
        # Sort by hydrological year
        hydrological_stats.sort(key=lambda x: x['Hydrological_Year'])
        
        # Add seasonal summary statistics
        if hydrological_stats:
            # Calculate overall seasonal statistics
            all_data = data.copy()
            
            # Get overall maximum and minimum with seasons
            overall_max = all_data.max()
            overall_max_date = all_data.idxmax()
            overall_max_season = get_nz_season(overall_max_date)
            
            overall_min = all_data.min()
            overall_min_date = all_data.idxmin()
            overall_min_season = get_nz_season(overall_min_date)
            
            # Count occurrences of max and min by season
            max_season_counts = {}
            min_season_counts = {}
            
            for stat in hydrological_stats:
                max_season = stat['Max_Season']
                min_season = stat['Min_Season']
                
                if max_season != 'N/A':
                    max_season_counts[max_season] = max_season_counts.get(max_season, 0) + 1
                if min_season != 'N/A':
                    min_season_counts[min_season] = min_season_counts.get(min_season, 0) + 1
            
            # Format season counts
            max_season_summary = ', '.join([f"{season} {count}" for season, count in max_season_counts.items()])
            min_season_summary = ', '.join([f"{season} {count}" for season, count in min_season_counts.items()])
            
            # Add seasonal summary to the first hydrological year entry
            if hydrological_stats:
                hydrological_stats[0]['Overall_Max_Level_m'] = overall_max
                hydrological_stats[0]['Overall_Max_Season'] = overall_max_season
                hydrological_stats[0]['Overall_Max_Date'] = overall_max_date.strftime('%d-%b-%Y')
                hydrological_stats[0]['Overall_Max_Description'] = f"Highest water level recorded"
                
                hydrological_stats[0]['Overall_Min_Level_m'] = overall_min
                hydrological_stats[0]['Overall_Min_Season'] = overall_min_season
                hydrological_stats[0]['Overall_Min_Date'] = overall_min_date.strftime('%d-%b-%Y')
                hydrological_stats[0]['Overall_Min_Description'] = f"Lowest water level recorded"
                
                hydrological_stats[0]['Max_Season_Count'] = max_season_summary
                hydrological_stats[0]['Min_Season_Count'] = min_season_summary
        
        print(f"Calculated 7-day smoothed hydrological year statistics for {well_name}: {len(hydrological_stats)} years")
        return hydrological_stats
        
    except Exception as e:
        print(f"Error in calculate_hydrological_year_statistics_rolling for {well_name}: {e}")
        return []


__all__ = [
    'calculate_hydrological_year_statistics',
    'calculate_hydrological_year_statistics_rolling'
]
