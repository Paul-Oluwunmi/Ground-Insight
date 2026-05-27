"""
Single file loader for groundwater data with automatic format detection.
"""

import os
import logging
import polars as pl
import pandas as pd
import numpy as np
from .format_detection import detect_data_format
from .indexed_loader import load_indexed_format
from .utils import clean_filename, convert_units, remove_duplicates
from .rainfall_loader import load_rainfall_data, merge_rainfall_data
from .datetime_parser import parse_datetime_column, COMMON_DATE_FORMATS


def load_mapping_file(mapping_file_path):
    """
    Load mapping file for well name to IRIS ID mapping.
    
    Parameters:
    -----------
    mapping_file_path : str
        Path to mapping CSV file
        
    Returns:
    --------
    dict or None
        Mapping dictionary or None if loading fails
    """
    if not mapping_file_path or mapping_file_path == 'None' or not os.path.exists(mapping_file_path):
        return None
    
    try:
        mapping_df = pd.read_csv(mapping_file_path)
        well_col = [col for col in mapping_df.columns if col.lower() in ['well', 'name', 'site']]
        irisid_col = [col for col in mapping_df.columns if col.lower() == 'irisid']
        if well_col and irisid_col:
            mapping_dict = {}
            for well_name, irisid in zip(mapping_df[well_col[0]].astype(str), mapping_df[irisid_col[0]].astype(str)):
                mapping_dict[well_name] = irisid
                transformed_name = well_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                mapping_dict[transformed_name] = irisid
            return mapping_dict
    except Exception as e:
        logging.warning(f"Could not load mapping file: {e}")
    return None


def load_and_process_data(groundwater_filename, data_folder=None, rainfall_filename=None, 
                          mapping_filename=None, convert_mm_to_m=False, 
                          remove_duplicates_flag=False):
    """
    Load and process single file data with automatic format detection.
    
    Parameters:
    -----------
    groundwater_filename : str
        Name of groundwater CSV file
    data_folder : str, optional
        Path to folder containing data files. If None, will try to use global 'data_folder' variable
    rainfall_filename : str, optional
        Name of rainfall CSV file
    mapping_filename : str, optional
        Name of mapping CSV file
    convert_mm_to_m : bool, default False
        Whether to convert units from mm to meters
    remove_duplicates_flag : bool, default False
        Whether to remove duplicate records
        
    Returns:
    --------
    tuple : (pandas.DataFrame, list, dict)
        Dataframe with data, list of well column names, and mapping dictionary
        Returns (None, None, None) if loading fails
    """
    try:
        # Use global data_folder if not provided (for notebook compatibility)
        if data_folder is None:
            import inspect
            # Try to get from calling module's globals (e.g., notebook cell)
            frame = inspect.currentframe().f_back
            if frame and 'data_folder' in frame.f_globals:
                data_folder = frame.f_globals['data_folder']
            else:
                raise ValueError("data_folder must be provided either as parameter or as global variable")
        
        print(f"Loading groundwater data from: {groundwater_filename}")
        
        # Load the data file
        file_path = os.path.join(data_folder, groundwater_filename)
        data = pl.read_csv(file_path, null_values=["NA", "", "None", "#N/A", "NaN"], infer_schema_length=None)
        
        if data is None or len(data) == 0:
            raise ValueError("No data loaded from file")
        
        # Detect data format
        format_type = detect_data_format(data)
        print(f"Detected data format: {format_type}")
        
        if format_type == 'indexed':
            data, well_columns = load_indexed_format(file_path, convert_mm_to_m, remove_duplicates_flag)
            if data is None:
                return None, None, None
                
        elif format_type == 'long':
            # Handle long format data (WellNo/Value or name/level)
            data_pd = data.to_pandas()
            
            # Standardize column names
            col_mapping = {}
            for col in data_pd.columns:
                col_lower = col.lower()
                if col_lower in ['datetime', 'date', 'time']:
                    col_mapping[col] = 'DateTime'
                elif col_lower in ['name', 'wellno', 'well']:
                    col_mapping[col] = 'name'
                elif col_lower in ['level', 'value']:
                    col_mapping[col] = 'level'
            
            if col_mapping:
                data_pd = data_pd.rename(columns=col_mapping)
            
            # Validate required columns
            if 'DateTime' not in data_pd.columns:
                datetime_col = data_pd.columns[0]
                data_pd = data_pd.rename(columns={datetime_col: 'DateTime'})
            
            if 'name' not in data_pd.columns or 'level' not in data_pd.columns:
                raise ValueError("Long format data must have 'name' and 'level' columns")
            
            # Convert DateTime
            data_pd['DateTime'] = pd.to_datetime(data_pd['DateTime'], errors='coerce')
            data_pd['level'] = pd.to_numeric(data_pd['level'], errors='coerce')
            data_pd = data_pd.dropna(subset=['DateTime', 'name', 'level'])
            
            # Clean well names
            data_pd['name'] = data_pd['name'].astype(str).apply(clean_filename)
            
            # Remove duplicates
            data_pd = remove_duplicates(data_pd, subset_columns=['DateTime', 'name', 'level'], 
                                       keep='first', remove_duplicates_flag=remove_duplicates_flag)
            
            # Pivot to wide format
            data_wide = data_pd.pivot_table(index='DateTime', columns='name', values='level', aggfunc='first').reset_index()
            data_wide.columns.name = None
            data_wide['mm_Rain'] = np.nan
            
            well_columns = [col for col in data_wide.columns if col not in ['DateTime', 'mm_Rain']]
            data = data_wide
            
        elif format_type == 'wide':
            # Handle wide format data
            data_pd = data.to_pandas()
            
            # Handle datetime column
            datetime_col = None
            for col in data_pd.columns:
                if col.lower() in ['datetime', 'date', 'time']:
                    datetime_col = col
                    break
            
            if datetime_col and datetime_col != "DateTime":
                data_pd = data_pd.rename(columns={datetime_col: "DateTime"})
            elif not datetime_col:
                first_col = data_pd.columns[0]
                data_pd = data_pd.rename(columns={first_col: "DateTime"})
            
            # Parse datetime using helper function
            data_pd = parse_datetime_column(data_pd, datetime_col='DateTime')
            
            # Convert numeric columns
            numeric_cols = [col for col in data_pd.columns if col != "DateTime"]
            for col in numeric_cols:
                data_pd[col] = pd.to_numeric(data_pd[col], errors='coerce')
            
            data_pd = remove_duplicates(data_pd, subset_columns=['DateTime'], 
                                       keep='first', remove_duplicates_flag=remove_duplicates_flag)
            data_pd['mm_Rain'] = np.nan
            
            well_columns = [col for col in data_pd.columns if col not in ['DateTime', 'mm_Rain']]
            data = data_pd
            
        else:
            raise ValueError(f"Unknown data format: {format_type}")
        
        if data is None:
            return None, None, None
        
        # Load rainfall data if specified
        if rainfall_filename and rainfall_filename != 'None':
            print(f"Loading rainfall data from: {rainfall_filename}")
            rainfall_file = os.path.join(data_folder, rainfall_filename)
            rainfall_data = load_rainfall_data(rainfall_file)
            if rainfall_data is not None:
                data = merge_rainfall_data(data, rainfall_data)
            else:
                if 'mm_Rain' not in data.columns:
                    data['mm_Rain'] = np.nan
        
        # Load mapping if specified
        mapping_dict = None
        if mapping_filename and mapping_filename != 'None':
            mapping_file = os.path.join(data_folder, mapping_filename)
            mapping_dict = load_mapping_file(mapping_file)
        
        # Apply unit conversion if enabled
        data, well_columns = convert_units(data, well_columns, convert_mm_to_m)
        
        # Keep DateTime as a column (not index) for better compatibility with visualization functions
        if 'DateTime' in data.columns:
            data = data.sort_values('DateTime')
        
        logging.info(f"Data loaded successfully: {len(well_columns)} wells, {len(data)} records")
        return data, well_columns, mapping_dict
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None, None, None

