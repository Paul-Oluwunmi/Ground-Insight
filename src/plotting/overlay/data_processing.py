"""
Overlay Data Processing
=======================

Functions for data resampling, caching, and optimization for the overlay visualization.
"""

import polars as pl
import pandas as pd
import numpy as np
import gc
from typing import Optional


# Optimized cache with memory management
cache = {}


def clear_old_cache(interval_to_keep):
    """Clear cache except for the most recently used interval"""
    for key in list(cache.keys()):
        if key != f"data_{interval_to_keep}":
            del cache[key]
    gc.collect()  # Force garbage collection


def get_resampled_data(data_lazy, well_columns_tuple, interval):
    """
    Resample data based on specified time interval.
    
    Parameters:
    -----------
    data_lazy : pl.LazyFrame
        Lazy polars DataFrame containing groundwater data
    well_columns_tuple : tuple
        Tuple of well column names (for hashing)
    interval : str
        Time interval for resampling. Options:
        - 'original': No resampling
        - 'D': Daily
        - 'W': Weekly
        - 'ME': Monthly
        - '6ME': 6-Monthly
        - 'Y': Yearly
            
    Returns:
    --------
    pl.DataFrame or None
        Resampled data containing DateTime, mm_Rain, and well columns
        
    Raises:
    -------
    ValueError
        If resampling fails or invalid interval is provided
    """
    cache_key = f"data_{interval}"
    
    try:
        if cache_key in cache:
            return cache[cache_key]
        
        # Clear old cache entries before creating new one
        clear_old_cache(interval)
        
        # Convert to pandas for resampling
        df_pd = data_lazy.collect().to_pandas()
        
        if interval == 'original':
            result = df_pd
        else:
            # Define resampling frequencies
            resample_dict = {
                'D': 'D',      # Daily
                'W': 'W',      # Weekly
                'ME': 'M',     # Monthly
                '6ME': '6M',   # 6-Monthly
                'Y': 'Y'       # Yearly
            }
            rule = resample_dict[interval]
            
            # Set DateTime as index for resampling
            df_pd = df_pd.set_index('DateTime')
            
            # Resample rainfall (sum) and groundwater (mean) separately
            well_columns = list(well_columns_tuple)
            resampled_rain = df_pd['mm_Rain'].resample(rule).sum()
            resampled_gw = df_pd[well_columns].resample(rule).mean()
            
            # Combine results
            result = pd.concat([resampled_rain, resampled_gw], axis=1).reset_index()
        
        # Convert back to polars
        result = pl.from_pandas(result)
        
        cache[cache_key] = result
        return result
        
    except Exception as e:
        print(f"Error in get_resampled_data: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_data_for_overlay(data):
    """
    Prepare and optimize data for overlay visualization.
    
    Parameters:
    -----------
    data : pl.DataFrame or pd.DataFrame
        Input data
        
    Returns:
    --------
    pl.LazyFrame
        Optimized lazy frame ready for resampling
    """
    # Pre-process data once at initialization using lazy evaluation
    if isinstance(data, pl.DataFrame):
        data = data.lazy()
    elif hasattr(data, 'to_pandas'):
        data = pl.from_pandas(data.to_pandas()).lazy()
    else:
        data = pl.from_pandas(data).lazy()
    
    # Initial data optimization - sort and format DateTime
    data = (data
        .with_columns(pl.col('DateTime').cast(pl.Datetime))
        .sort('DateTime')
    )
    
    # Pre-process data validation
    _ = data.collect().columns  # Validate data can be collected
    
    return data


def load_rainfall_data(data_folder, rainfall_filename, data_lazy):
    """
    Load rainfall data from file or use existing data.
    
    Parameters:
    -----------
    data_folder : str or None
        Path to data folder
    rainfall_filename : str or None
        Name of rainfall file
    data_lazy : pl.LazyFrame
        Main data lazy frame (fallback if file not found)
        
    Returns:
    --------
    pl.DataFrame
        Rainfall data with DateTime and mm_Rain columns
    """
    import os
    
    if data_folder and rainfall_filename:
        rain_file = os.path.join(data_folder, rainfall_filename)
        if os.path.exists(rain_file):
            # Read first line to check format
            with open(rain_file, 'r') as f:
                first_line = f.readline()
            
            # Skip header rows if needed
            skip_rows = 2 if "Rainfall" in first_line else 0
            rain_df = pd.read_csv(rain_file, skiprows=skip_rows)
            
            # Get the actual column names
            time_col = rain_df.columns[0]  # First column should be time
            rain_col = rain_df.columns[1]  # Second column should be rainfall
            
            # Rename columns to our standard names
            rain_df = rain_df.rename(columns={
                time_col: 'DateTime',
                rain_col: 'mm_Rain'
            })
            
            # Convert to datetime (British format DD/MM/YYYY)
            rain_df['DateTime'] = pd.to_datetime(rain_df['DateTime'], format='%d/%m/%Y %H:%M', dayfirst=True)
            rainfall_data = pl.from_pandas(rain_df)
            return rainfall_data
    
    # Use mm_Rain from the main dataset
    return data_lazy.collect().select(['DateTime', 'mm_Rain'])

