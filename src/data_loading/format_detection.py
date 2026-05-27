"""
Data format detection functions.
"""

import logging
import polars as pl
import pandas as pd


def detect_data_format(data):
    """
    Automatically detect the data format based on column structure.
    
    Parameters:
    -----------
    data : polars.DataFrame
        Input data to analyze
        
    Returns:
    --------
    str
        Format type: 'indexed', 'long', 'wide', or 'unknown'
    """
    columns = [col.lower() for col in data.columns]
    
    # Check for indexed format (SiteName, Time, Value columns)
    if ('index' in columns and 'sitename' in columns and 
        'measurementname' in columns and 'time' in columns and 'value' in columns):
        return 'indexed'
    
    # Check for long format (name/wellno/well + level/value columns)
    elif (('name' in columns or 'wellno' in columns or 'well' in columns) and 
          ('level' in columns or 'value' in columns)):
        return 'long'
    
    # Check for single well long format (datetime + value/level)
    elif (len(data.columns) == 2 and 
          any(col.lower() in ['datetime', 'date', 'time', 'timestamp'] for col in data.columns) and 
          any(col.lower() in ['value', 'level'] for col in data.columns)):
        return 'long'
    
    # Check for wide format (datetime column + multiple data columns)
    elif len(data.columns) >= 2 and any(col.lower() in ['datetime', 'date', 'time', 'timestamp'] for col in data.columns):
        return 'wide'
    
    else:
        # Check if first column contains datetime values (even if header doesn't suggest it)
        # This handles cases where CSV files have datetime data in first column but header is a well name
        if len(data.columns) >= 2:
            try:
                first_col = data.columns[0]
                # Sample first few non-null values to check if they're datetime-like
                sample_data = data.select([pl.col(first_col)]).head(10)
                sample_values = sample_data[first_col].drop_nulls().head(5)
                
                if len(sample_values) > 0:
                    # Try to parse as datetime
                    try:
                        # Convert to pandas for easier datetime parsing
                        sample_pd = sample_data.to_pandas()
                        parsed = pd.to_datetime(sample_pd[first_col].dropna().head(5), errors='coerce')
                        # If at least 80% of samples parse as datetime, assume it's a datetime column
                        if parsed.notna().sum() / max(len(parsed), 1) >= 0.8:
                            return 'wide'
                    except:
                        pass
            except Exception as e:
                logging.debug(f"Error checking first column for datetime: {e}")
        
        return 'unknown'

