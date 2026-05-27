"""
Utility functions for data loading and processing.
"""

import logging
import re
import pandas as pd


def clean_filename(filename):
    """Clean filename by replacing spaces and special characters with underscores"""
    return re.sub(r'[^\w\-_.]', '_', filename).replace('__', '_').strip('_')


def convert_units(data, well_columns, convert_mm_to_m=True):
    """
    Convert groundwater level measurements from mm to meters if conversion is enabled.
    Rainfall remains in mm as it's the standard unit.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataframe containing groundwater level data
    well_columns : list
        List of column names containing well data
    convert_mm_to_m : bool
        Whether to convert from mm to meters
        
    Returns:
    --------
    data : pandas.DataFrame
        Dataframe with converted units
    well_columns : list
        List of well column names (unchanged)
    """
    if not convert_mm_to_m:
        return data, well_columns
    
    try:
        # Convert groundwater level measurements from mm to meters
        for col in well_columns:
            if col in data.columns:
                data[col] = data[col] / 1000.0  # Convert mm to meters
                logging.info(f"Converted groundwater level {col} from mm to meters")
        
        # Keep rainfall in mm (standard unit) - no conversion needed
        if 'mm_Rain' in data.columns:
            logging.info("Rainfall data kept in mm (standard unit)")
        
        logging.info("Unit conversion completed: Groundwater levels mm → meters, Rainfall kept in mm")
        return data, well_columns
    except Exception as e:
        logging.warning(f"Error during unit conversion: {e}")
        return data, well_columns


def remove_duplicates(data, subset_columns=None, keep='first', remove_duplicates_flag=True):
    """
    Remove duplicate records from the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataframe to clean
    subset_columns : list, optional
        Columns to check for duplicates. If None, checks all columns.
    keep : str, default 'first'
        Which duplicate to keep ('first', 'last', or False)
    remove_duplicates_flag : bool, default True
        Whether to actually remove duplicates. If False, returns data unchanged.
        
    Returns:
    --------
    data : pandas.DataFrame
        Dataframe with duplicates removed
    """
    # Check if duplicate removal is enabled
    if not remove_duplicates_flag:
        logging.info("Duplicate removal is DISABLED - keeping all records")
        return data
    
    if subset_columns is None:
        subset_columns = data.columns.tolist()
    original_count = len(data)
    
    # Check if duplicates have different values (potential data loss)
    if len(subset_columns) < len(data.columns):
        # Only check if we're not checking all columns
        duplicate_mask = data.duplicated(subset=subset_columns, keep=False)
        if duplicate_mask.any():
            duplicate_rows = data[duplicate_mask]
            # Group by the subset columns and check if other columns differ
            for cols_to_check in [subset_columns]:
                grouped = duplicate_rows.groupby(subset_columns)
                different_value_count = 0
                for name, group in grouped:
                    if len(group) > 1:
                        # Check if any other columns have different values
                        other_cols = [c for c in data.columns if c not in subset_columns]
                        if other_cols:
                            # Check if values differ in other columns
                            if len(group[other_cols].drop_duplicates()) > 1:
                                different_value_count += len(group) - 1
                
                if different_value_count > 0:
                    logging.warning(f"WARNING: {different_value_count} duplicate records have DIFFERENT values in other columns! "
                                  f"Only the first occurrence will be kept (using keep='{keep}'). "
                                  f"Consider reviewing your data.")
                else:
                    logging.info(f"All {duplicate_mask.sum()} duplicate records have IDENTICAL values (true duplicates)")
    
    data_clean = data.drop_duplicates(subset=subset_columns, keep=keep)
    duplicates_removed = original_count - len(data_clean)
    if duplicates_removed > 0:
        logging.info(f"Removed {duplicates_removed} duplicate records")
    else:
        logging.info("No duplicates found")
    return data_clean

