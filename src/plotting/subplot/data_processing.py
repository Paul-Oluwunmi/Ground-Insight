"""
Subplot Data Processing
========================

Functions for data preprocessing and resampling.
"""

import pandas as pd
from typing import List
import polars as pl

from .cache import get_cached_resampled_data, cache_resampled_data


def preprocess_data_for_speed(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data to optimize for fast plotting."""
    try:
        # Convert DateTime to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['DateTime']):
            data = data.copy()
            data['DateTime'] = pd.to_datetime(data['DateTime'])
        
        # Sort by DateTime once - this speeds up all subsequent operations
        data = data.sort_values('DateTime')
        
        # Reset index for faster column access
        data = data.reset_index(drop=True)
        
        return data
    except Exception:
        return data


def resample_data_optimized(data: pd.DataFrame, well_columns: List[str], interval: str) -> pd.DataFrame:
    """Optimized data resampling with caching."""
    try:
        # Fast data hash for caching
        data_hash = str(hash(str(data.shape) + str(data.columns.tolist())))
        
        # Check cache first
        cached_data = get_cached_resampled_data(interval, data_hash)
        if cached_data is not None:
            return cached_data
        
        # Preprocess data for speed
        data = preprocess_data_for_speed(data)
        
        if interval == 'original':
            # Return original data without resampling
            return data
        
        # Fast resampling operations
        resampling_ops = {
            'hourly': 'H',
            'daily': 'D',
            'monthly': 'M',
            '6-monthly': '6M',
            'yearly': 'Y'
        }
        
        if interval not in resampling_ops:
            return data
        
        # Select only numeric columns for resampling
        numeric_cols = ['DateTime'] + [col for col in well_columns 
                                     if pd.api.types.is_numeric_dtype(data[col])]
        df_resample = data[numeric_cols].copy()
        df_resample.set_index('DateTime', inplace=True)
        
        # Fast resampling
        resampled = df_resample.resample(resampling_ops[interval]).mean()
        resampled = resampled.reset_index()
        
        # Cache the result
        cache_resampled_data(interval, data_hash, resampled)
        
        return resampled
        
    except Exception as e:
        print(f"Error in optimized resampling: {str(e)}")
        return data

