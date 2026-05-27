"""
Data processing functions for Extreme Value Analysis
"""

import logging
import traceback
import os
import pandas as pd
import polars as pl
import numpy as np
from typing import Optional
from scipy.signal import find_peaks
from .utils import convert_time_unit
from .constants import global_store


def find_peaks_polars(df: pl.DataFrame, well: str) -> pl.DataFrame:
    """Find peaks with improved accuracy and time preservation"""
    # Ensure data is sorted by DateTime
    df = df.sort('DateTime')
    values = df[well].to_numpy()
    
    # Initial peak detection with more sensitive parameters
    peaks, properties = find_peaks(values,
                                 distance=3,        
                                 prominence=0.005,  
                                 width=1,          
                                 rel_height=0.1,   
                                 height=None)      
    
    # Refine peaks by checking local windows
    peak_indices = []
    for peak, prom in zip(peaks, properties['prominences']):
        # Use larger window for more prominent peaks
        window_size = max(2, int(prom * 2))
        start = max(0, peak - window_size)
        end = min(len(values), peak + window_size + 1)
        window = values[start:end]
        
        # Find true maximum in window
        local_max_idx = start + np.argmax(window)
        peak_indices.append(local_max_idx)
    
    # Return peaks as DataFrame with original DateTime preserved
    return df.filter(pl.arange(0, len(df)).is_in(peak_indices))


def get_processed_data(df_pl, well: str, resample: str, data_type: str, 
                      data_folder=None, rainfall_filename=None) -> Optional[pd.Series]:
    """Process the data for analysis with resampling using Polars for better performance."""
    try:
        # For rainfall data, try to load directly from file first
        if data_type == 'rain' and data_folder and rainfall_filename:
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
                
                # Convert to polars
                df_pl = pl.from_pandas(rain_df)
                df_pl = df_pl.rename({'mm_Rain': well})
            else:
                # Fallback to using the merged data
                if isinstance(df_pl, pd.DataFrame):
                    df_pl = pl.from_pandas(df_pl[['DateTime', well]])
                else:
                    df_pl = df_pl.select(['DateTime', well])
        else:
            # For groundwater data, just select the columns we need
            if isinstance(df_pl, pd.DataFrame):
                df_pl = pl.from_pandas(df_pl[['DateTime', well]])
            else:
                df_pl = df_pl.select(['DateTime', well])
        
        # Ensure DateTime is properly formatted and well column is float
        df_pl = df_pl.with_columns([
            pl.col('DateTime').cast(pl.Datetime),
            pl.col(well).cast(pl.Float64)
        ])
        
        logging.info(f"Initial data shape: {df_pl.shape}")
        
        # For rainfall data
        if data_type == 'rain':
            df_pl = df_pl.filter(pl.col(well) > 0)
            logging.info(f"Non-zero rainfall events: {len(df_pl)}")
            
            # Sort data BEFORE resampling (required by Polars group_by_dynamic)
            df_pl = df_pl.sort('DateTime')
            
            if resample:
                try:
                    polars_resample = convert_time_unit(resample)
                    df_pl = (df_pl
                            .group_by_dynamic('DateTime', 
                                           every=polars_resample,
                                           closed='left',
                                           label='left')
                            .agg(pl.col(well).sum()))
                except Exception as e:
                    logging.error(f"Error during rainfall resampling: {str(e)}")
                    df_pl = df_pl.sort('DateTime')
            
        else:  # groundwater levels
            df_pl = df_pl.filter(pl.col(well).is_not_null())
            logging.info(f"Valid groundwater measurements: {len(df_pl)}")
            
            # Sort data BEFORE resampling (required by Polars group_by_dynamic)
            df_pl = df_pl.sort('DateTime')
            
            if resample:
                try:
                    polars_resample = convert_time_unit(resample)
                    
                    # Find peaks first
                    peaks_df = find_peaks_polars(df_pl, well)
                    
                    # Perform basic resampling
                    resampled = (df_pl
                               .group_by_dynamic('DateTime', 
                                              every=polars_resample,
                                              closed='left',
                                              label='left')
                               .agg(pl.col(well).max())
                               .sort('DateTime'))
                    
                    # Ensure peaks are included
                    df_pl = (pl.concat([resampled, peaks_df])
                            .unique(subset=['DateTime'])
                            .sort('DateTime'))
                    
                    logging.info(f"Points after resampling with peak preservation: {len(df_pl)}")
                    
                    if len(df_pl) < 10:
                        logging.warning(f"Resampling resulted in too few points ({len(df_pl)}). Using original data.")
                        df_pl = df_pl.sort('DateTime')
                        
                except Exception as e:
                    logging.error(f"Error during groundwater resampling: {str(e)}")
                    logging.info("Falling back to original data")
                    df_pl = df_pl.sort('DateTime')
            else:
                df_pl = df_pl.sort('DateTime')

        # Drop null values and convert to pandas Series
        df_pl = df_pl.drop_nulls()
        
        # Log final data summary
        logging.info(f"Final points after processing: {len(df_pl)}")
        logging.info(f"Final data summary for {well}:")
        logging.info(f"Number of records: {len(df_pl):,}")
        logging.info(f"Date range: {df_pl['DateTime'].min()} to {df_pl['DateTime'].max()} ({(df_pl['DateTime'].max() - df_pl['DateTime'].min()).days / 365.25:.1f} years)")
        logging.info(f"Value range: {df_pl[well].min():.2f} to {df_pl[well].max():.2f}")
        logging.info(f"Resample period: {resample}")
        
        # Convert back to pandas Series
        return pd.Series(
            df_pl[well].to_numpy(),
            index=pd.DatetimeIndex(df_pl['DateTime'].to_numpy()),
            name=well
        )
        
    except Exception as e:
        logging.error(f"Error in get_processed_data: {str(e)}")
        traceback.print_exc()
        return None


def get_global_data() -> Optional[pd.DataFrame]:
    """Safely retrieve data from global store"""
    try:
        df_pl = global_store.get('df_pl')
        if df_pl is None:
            raise ValueError("No data available in global store")
            
        if hasattr(df_pl, 'to_pandas'):
            return df_pl.to_pandas()
        return df_pl
        
    except Exception as e:
        logging.error(f"Error accessing global store: {str(e)}")
        return None

