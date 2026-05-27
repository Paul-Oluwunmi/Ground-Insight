"""
Loader for indexed format data (SiteName, Time, Value columns).
"""

import logging
import polars as pl
import pandas as pd
import numpy as np
from .utils import clean_filename, convert_units, remove_duplicates


def load_indexed_format(file_path, convert_mm_to_m=False, remove_duplicates_flag=False):
    """
    Load and process indexed format data.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    convert_mm_to_m : bool, default False
        Whether to convert units from mm to meters
    remove_duplicates_flag : bool, default False
        Whether to remove duplicate records
        
    Returns:
    --------
    tuple : (pandas.DataFrame, list)
        Dataframe with data in wide format and list of well column names
        Returns (None, None) if loading fails
    """
    try:
        data = pl.read_csv(file_path, null_values=["NA", "", "None", "#N/A", "NaN"], infer_schema_length=None)
        if data is None or len(data) == 0:
            raise ValueError("No data loaded from file")
        
        data_pd = data.to_pandas()
        data_pd['SiteName'] = data_pd['SiteName'].astype(str).apply(clean_filename)
        
        try:
            data_pd['Time'] = pd.to_datetime(data_pd['Time'], errors='coerce')
        except Exception as e:
            logging.warning(f"Could not parse Time column: {e}")
            return None, None
        
        data_pd['Value'] = pd.to_numeric(data_pd['Value'], errors='coerce')
        data_pd = data_pd.dropna(subset=['Time', 'SiteName', 'Value'])
        
        if len(data_pd) == 0:
            raise ValueError("No valid data after cleaning")
        
        data_pd = remove_duplicates(data_pd, subset_columns=['Time', 'SiteName', 'Value'], 
                                    keep='first', remove_duplicates_flag=remove_duplicates_flag)
        
        # Pivot to wide format
        data_wide = data_pd.pivot_table(index='Time', columns='SiteName', values='Value', aggfunc='first').reset_index()
        data_wide.columns.name = None
        data_wide = data_wide.rename(columns={'Time': 'DateTime'})
        well_columns = [col for col in data_wide.columns if col != 'DateTime']
        data_wide['mm_Rain'] = np.nan
        
        # Apply unit conversion if enabled
        data_wide, well_columns = convert_units(data_wide, well_columns, convert_mm_to_m)
        
        logging.info(f"Indexed format loaded: {len(well_columns)} wells, {len(data_wide)} records")
        return data_wide, well_columns
        
    except Exception as e:
        logging.error(f"Error loading indexed format: {str(e)}")
        return None, None

