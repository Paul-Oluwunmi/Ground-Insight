"""
Static Plots Data Processing
=============================

Functions for data preprocessing and interval calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union
import polars as pl


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


def calculate_all_intervals_for_well(data: Union[pd.DataFrame, pl.DataFrame], well_column: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Calculate all intervals for a single well in one pass for maximum performance."""
    try:
        # Skip Polars conversion if data is already pandas - faster for pandas DataFrames
        if isinstance(data, pd.DataFrame):
            pandas_df = data[['DateTime', well_column]].copy()
        else:
            # Only convert if it's Polars
            pandas_df = data.select(['DateTime', well_column]).drop_nulls().to_pandas()
        
        # Fast null removal and sorting
        pandas_df = pandas_df.dropna().sort_values('DateTime')
        
        if len(pandas_df) == 0:
            return {}
        
        # Set index once - faster than inplace
        pandas_df.set_index('DateTime', inplace=True)
        
        intervals = {}
        
        # Add original data (raw data without resampling)
        try:
            x_original = pandas_df.index.values
            y_original = pandas_df[well_column].values
            
            # Data decimation for very large datasets to make Plotly faster
            if len(x_original) > 2000:  # Higher threshold for original data
                step = max(1, len(x_original) // 2000)  # Keep max 2000 points for original data
                x_original = x_original[::step]
                y_original = y_original[::step]
            
            if len(x_original) > 0:
                intervals['original'] = (x_original, y_original)
                
        except Exception:
            pass
        
        # Pre-define resampling operations with optimized parameters
        resampling_ops = {
            'hourly': 'H', 
            'daily': 'D',
            'monthly': 'M',
            '6-monthly': '6M',
            'yearly': 'Y'
        }
        
        # Process all intervals in one loop - maximum efficiency
        for interval_name, resample_code in resampling_ops.items():
            try:
                # Fast resampling with minimal overhead
                resampled = pandas_df.resample(resample_code).mean()
                
                # Quick validation and conversion
                if not resampled.empty and resampled[well_column].notna().any():
                    # Direct NumPy conversion - fastest possible
                    x = resampled.index.values  # Faster than to_numpy()
                    y = resampled[well_column].values
                    
                    # Data decimation for very large datasets to make Plotly faster
                    if len(x) > 1000:  # If more than 1000 points, decimate
                        step = max(1, len(x) // 1000)  # Keep max 1000 points
                        x = x[::step]
                        y = y[::step]
                    
                    if len(x) > 0:
                        intervals[interval_name] = (x, y)
                        
            except Exception as e:
                # Silent fail - don't slow down with warnings
                continue
        
        # Minimal cleanup
        del pandas_df
        
        return intervals
            
    except Exception:
        return {}

