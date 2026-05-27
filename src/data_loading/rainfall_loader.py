"""
Helper functions for loading and merging rainfall data.
"""

import os
import logging
import polars as pl
import pandas as pd
import numpy as np


def load_rainfall_data(rainfall_file_path):
    """
    Load rainfall data from CSV file.
    
    Parameters:
    -----------
    rainfall_file_path : str
        Path to rainfall CSV file
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with DateTime and mm_Rain columns, or None if loading fails
    """
    if not rainfall_file_path or not os.path.exists(rainfall_file_path):
        return None
    
    try:
        rain_data = pl.read_csv(rainfall_file_path, null_values=["NA", "", "None", "#N/A", "NaN"], infer_schema_length=None)
        if len(rain_data.columns) >= 2:
            rain_data = rain_data.select([
                pl.col(rain_data.columns[0]).alias("DateTime"), 
                pl.col(rain_data.columns[1]).alias("mm_Rain")
            ])
        
        # Try parsing datetime with common formats
        datetime_parsed = False
        date_formats = [
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ]
        
        for fmt in date_formats:
            try:
                rain_data = rain_data.with_columns([pl.col("DateTime").str.strptime(pl.Datetime, format=fmt)])
                datetime_parsed = True
                break
            except:
                continue
        
        # Handle "From X to Y" format
        if not datetime_parsed:
            try:
                rain_data_pd = rain_data.to_pandas()
                logging.info(f"Initial rainfall data from polars: {len(rain_data_pd)} rows")
                
                def extract_start_date(date_str):
                    if isinstance(date_str, str) and 'From' in date_str:
                        start_date = date_str.split('From')[1].split('to')[0].strip()
                        return start_date
                    return date_str
                
                rain_data_pd['DateTime'] = rain_data_pd['DateTime'].apply(extract_start_date)
                logging.info(f"After extracting start dates: {len(rain_data_pd)} rows")
                
                # Try parsing with comprehensive date formats
                datetime_parsed_format = False
                date_formats = [
                    "%d/%m/%Y %H:%M",
                    "%d/%m/%Y %H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M",
                    "%m/%d/%Y %H:%M",
                    "%m/%d/%Y %H:%M:%S",
                    "%d-%m-%Y %H:%M",
                    "%d-%m-%Y %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                    "%Y/%m/%d %H:%M",
                    "%d/%m/%Y",
                    "%Y-%m-%d",
                    "%m/%d/%Y",
                ]
                
                for fmt in date_formats:
                    try:
                        parsed = pd.to_datetime(rain_data_pd['DateTime'], format=fmt, errors='coerce')
                        if parsed.notna().sum() / len(rain_data_pd) >= 0.9:
                            rain_data_pd['DateTime'] = parsed
                            datetime_parsed_format = True
                            logging.info(f"Successfully parsed rainfall DateTime using format: {fmt}")
                            break
                    except:
                        continue
                
                if not datetime_parsed_format:
                    rain_data_pd['DateTime'] = pd.to_datetime(rain_data_pd['DateTime'], errors='coerce')
                
                valid_dates = rain_data_pd['DateTime'].notna().sum()
                logging.info(f"After datetime conversion: {len(rain_data_pd)} rows, {valid_dates} valid dates")
                rain_data = pl.from_pandas(rain_data_pd)
                datetime_parsed = True
                logging.info("Successfully parsed 'From X to Y' datetime format")
            except Exception as e:
                logging.warning(f"Could not parse 'From X to Y' format: {e}")
        
        if not datetime_parsed:
            raise ValueError("Unable to parse datetime column in rainfall data")
        
        rain_data = rain_data.with_columns([pl.col("mm_Rain").cast(pl.Float64, strict=False)])
        rain_data_pd = rain_data.to_pandas()
        rain_data_pd['DateTime'] = pd.to_datetime(rain_data_pd['DateTime'])
        rain_data_pd = rain_data_pd.dropna(subset=['DateTime'])
        rain_data_pd = rain_data_pd.sort_values('DateTime')
        
        return rain_data_pd
        
    except Exception as e:
        logging.warning(f"Error loading rainfall data: {str(e)}")
        import traceback
        logging.warning(f"Traceback: {traceback.format_exc()}")
        return None


def merge_rainfall_data(data, rainfall_data_pd):
    """
    Merge rainfall data into main dataframe.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Main groundwater data dataframe
    rainfall_data_pd : pandas.DataFrame
        Rainfall data with DateTime and mm_Rain columns
        
    Returns:
    --------
    pandas.DataFrame
        Merged dataframe with rainfall data
    """
    if rainfall_data_pd is None or len(rainfall_data_pd) == 0:
        logging.warning("No rainfall data to merge")
        if 'mm_Rain' not in data.columns:
            data['mm_Rain'] = np.nan
        return data
    
    # Ensure both dataframes have DateTime as datetime type
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data = data.sort_values('DateTime')
    rainfall_data_pd = rainfall_data_pd.sort_values('DateTime')
    
    # Handle potential duplicate DateTime values
    rain_for_merge = rainfall_data_pd[['DateTime', 'mm_Rain']].copy()
    if rain_for_merge['DateTime'].duplicated().any():
        rain_for_merge = rain_for_merge.drop_duplicates(subset=['DateTime'], keep='first')
    
    # Merge
    if 'mm_Rain' in data.columns:
        data_temp = data.drop(columns=['mm_Rain'])
        merged = pd.merge(data_temp, rain_for_merge, on='DateTime', how='outer')
    else:
        merged = pd.merge(data, rain_for_merge, on='DateTime', how='outer')
    
    merged = merged.sort_values('DateTime')
    
    # Log statistics
    if 'mm_Rain' in merged.columns:
        rainfall_count = merged['mm_Rain'].notna().sum()
        logging.info(f"Rainfall data merged successfully. {rainfall_count}/{len(merged)} rows have rainfall data.")
    else:
        logging.warning("mm_Rain column not found after merge - adding empty column")
        merged['mm_Rain'] = np.nan
    
    return merged

