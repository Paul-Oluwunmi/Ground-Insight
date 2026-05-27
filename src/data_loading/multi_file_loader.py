"""
Multi-file loader for loading multiple CSV files from a folder.
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
from .datetime_parser import parse_datetime_column
from .single_file_loader import load_mapping_file


def process_single_csv_file(file_path, csv_file, convert_mm_to_m=False, remove_duplicates_flag=False):
    """
    Process a single CSV file and return wide format dataframe.
    
    Parameters:
    -----------
    file_path : str
        Full path to CSV file
    csv_file : str
        Filename (for logging and well naming)
    convert_mm_to_m : bool, default False
        Whether to convert units
    remove_duplicates_flag : bool, default False
        Whether to remove duplicates
        
    Returns:
    --------
    tuple : (pandas.DataFrame, list)
        Wide format dataframe and list of well column names
        Returns (None, None) if processing fails
    """
    try:
        data = pl.read_csv(file_path, null_values=["NA", "", "None", "#N/A", "NaN"], infer_schema_length=None)
        if data is None or len(data) == 0:
            logging.warning(f"No data in file: {csv_file}")
            return None, None
        
        format_type = detect_data_format(data)
        logging.info(f"Detected format for {csv_file}: {format_type}")
        
        if format_type == 'indexed':
            data_wide, wells = load_indexed_format(file_path, convert_mm_to_m, remove_duplicates_flag)
            return data_wide, wells
            
        elif format_type == 'long':
            data_pd = data.to_pandas()
            
            # Standardize column names
            col_mapping = {}
            for col in data_pd.columns:
                col_lower = col.lower()
                if col_lower in ['datetime', 'date', 'time', 'timestamp']:
                    col_mapping[col] = 'DateTime'
                elif col_lower in ['name', 'wellno', 'well']:
                    col_mapping[col] = 'name'
                elif col_lower in ['level', 'value']:
                    col_mapping[col] = 'level'
            
            if col_mapping:
                data_pd = data_pd.rename(columns=col_mapping)
            
            # If no 'name' column, add it from filename
            if 'name' not in data_pd.columns:
                well_name = os.path.splitext(csv_file)[0]
                well_name = well_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                data_pd['name'] = well_name
                logging.info(f"Added site name '{well_name}' from filename for file: {csv_file}")
            
            data_pd['DateTime'] = pd.to_datetime(data_pd['DateTime'], errors='coerce')
            data_pd['level'] = pd.to_numeric(data_pd['level'], errors='coerce')
            data_pd = data_pd.dropna(subset=['DateTime', 'name', 'level'])
            data_pd['name'] = data_pd['name'].astype(str).apply(clean_filename)
            data_pd = remove_duplicates(data_pd, subset_columns=['DateTime', 'name', 'level'], 
                                       keep='first', remove_duplicates_flag=remove_duplicates_flag)
            
            data_wide = data_pd.pivot_table(index='DateTime', columns='name', values='level', aggfunc='first').reset_index()
            data_wide.columns.name = None
            data_wide['mm_Rain'] = np.nan
            wells = [col for col in data_wide.columns if col not in ['DateTime', 'mm_Rain']]
            
            data_wide, wells = convert_units(data_wide, wells, convert_mm_to_m)
            return data_wide, wells
            
        elif format_type == 'wide':
            data_pd = data.to_pandas()
            
            # Handle datetime column
            datetime_col = None
            for col in data_pd.columns:
                if col.lower() in ['datetime', 'date', 'time', 'timestamp']:
                    datetime_col = col
                    break
            
            if datetime_col and datetime_col != "DateTime":
                data_pd = data_pd.rename(columns={datetime_col: "DateTime"})
            elif not datetime_col:
                first_col = data_pd.columns[0]
                data_pd = data_pd.rename(columns={first_col: "DateTime"})
            
            # Parse datetime
            data_pd = parse_datetime_column(data_pd, datetime_col='DateTime')
            
            # Convert numeric columns
            numeric_cols = [col for col in data_pd.columns if col != "DateTime"]
            for col in numeric_cols:
                data_pd[col] = pd.to_numeric(data_pd[col], errors='coerce')
            
            data_pd = remove_duplicates(data_pd, subset_columns=['DateTime'], 
                                       keep='first', remove_duplicates_flag=remove_duplicates_flag)
            data_pd['mm_Rain'] = np.nan
            wells = [col for col in data_pd.columns if col not in ['DateTime', 'mm_Rain']]
            
            data_pd, wells = convert_units(data_pd, wells, convert_mm_to_m)
            return data_pd, wells
            
        else:
            logging.warning(f"Unknown format for file: {csv_file}")
            return None, None
            
    except Exception as e:
        logging.warning(f"Error processing file {csv_file}: {str(e)}")
        return None, None


def load_multiple_csv_enhanced(folder_path, data_folder=None, rainfall_csv_name=None, 
                               mapping_csv_name=None, convert_mm_to_m=False, 
                               remove_duplicates_flag=False):
    """
    Enhanced function to load multiple CSV files from a folder with automatic format detection.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing CSV files
    data_folder : str, optional
        Path to data folder (for rainfall and mapping files). If None, will try to use global 'data_folder' variable
    rainfall_csv_name : str, optional
        Name of rainfall CSV file
    mapping_csv_name : str, optional
        Name of mapping CSV file
    convert_mm_to_m : bool, default False
        Whether to convert units from mm to meters
    remove_duplicates_flag : bool, default False
        Whether to remove duplicate records
        
    Returns:
    --------
    tuple : (pandas.DataFrame, list, dict)
        Merged dataframe, list of well column names, and mapping dictionary
        Returns (None, None, None) if loading fails
    """
    all_dataframes = []
    well_columns = []
    mapping_dict = None
    
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
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in folder: {folder_path}")
        
        logging.info(f"Found {len(csv_files)} CSV files in folder: {folder_path}")
        
        # Load mapping file if specified
        if mapping_csv_name:
            mapping_file = os.path.join(data_folder, mapping_csv_name)
            mapping_dict = load_mapping_file(mapping_file)
        
        # Process each CSV file
        for i, csv_file in enumerate(csv_files):
            file_path = os.path.join(folder_path, csv_file)
            logging.info(f"Processing file {i+1}/{len(csv_files)}: {csv_file}")
            
            data_wide, wells = process_single_csv_file(file_path, csv_file, convert_mm_to_m, remove_duplicates_flag)
            
            if data_wide is not None and wells:
                all_dataframes.append(data_wide)
                well_columns.extend(wells)
        
        if not all_dataframes:
            raise ValueError("No valid data found in any CSV files")
        
        # Merge all dataframes
        logging.info("Combining all dataframes...")
        merged = all_dataframes[0]
        
        for i, df in enumerate(all_dataframes[1:], 1):
            # Handle mm_Rain columns - keep only one
            if 'mm_Rain' in df.columns and 'mm_Rain' in merged.columns:
                df = df.drop(columns=['mm_Rain'])
            
            # Handle 'value' columns without site names
            if 'value' in df.columns and 'name' not in df.columns and len(df.columns) == 2:
                if i < len(csv_files):
                    csv_file = csv_files[i]
                    well_name = os.path.splitext(csv_file)[0]
                    well_name = well_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    df = df.rename(columns={'value': well_name})
                    logging.info(f"Renamed 'value' column to '{well_name}' for file without site name: {csv_file}")
            
            # Merge dataframes
            merged = pd.merge(merged, df, on='DateTime', how='outer')
        
        # Load and merge rainfall data if specified
        if rainfall_csv_name:
            rainfall_file = os.path.join(data_folder, rainfall_csv_name)
            rainfall_data = load_rainfall_data(rainfall_file)
            if rainfall_data is not None:
                merged = merge_rainfall_data(merged, rainfall_data)
            else:
                if 'mm_Rain' not in merged.columns:
                    merged['mm_Rain'] = np.nan
        
        # Final processing
        merged = merged.sort_values('DateTime')
        original_count = len(merged)
        
        if remove_duplicates_flag:
            merged = merged.drop_duplicates(subset=['DateTime'], keep='first')
            final_duplicates_removed = original_count - len(merged)
            if final_duplicates_removed > 0:
                logging.info(f"Removed {final_duplicates_removed} duplicate timestamps from final dataset")
        else:
            logging.info("Duplicate removal is DISABLED - keeping all records with duplicate timestamps")
        
        # Re-derive well columns
        well_columns = [col for col in merged.columns if col not in ['DateTime', 'mm_Rain']]
        merged, well_columns = convert_units(merged, well_columns, convert_mm_to_m)
        
        well_columns = list(dict.fromkeys(well_columns))
        if not well_columns:
            raise ValueError("No well columns found in data")
        
        logging.info("Enhanced data loading completed successfully!")
        logging.info(f"Number of wells: {len(well_columns)}")
        logging.info(f"Well names: {', '.join(well_columns[:10])}{'...' if len(well_columns) > 10 else ''}")
        if 'DateTime' in merged.columns:
            logging.info(f"Date range: {merged['DateTime'].min()} to {merged['DateTime'].max()}")
        logging.info(f"Final dataset shape: {merged.shape}")
        
        return merged, well_columns, mapping_dict
        
    except Exception as e:
        logging.error(f"Error loading data from folder: {str(e)}")
        return None, None, None

