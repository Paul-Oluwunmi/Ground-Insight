"""
Batch analysis functions for Mann-Kendall

This module contains functions for running batch Mann-Kendall analyses
on multiple wells, including comprehensive batch processing with CSV export.
"""

import pandas as pd
import numpy as np
import os
import gc
import traceback
from .constants import (
    USE_ROLLING_MEAN,
    ROLLING_WINDOW_SIZE,
    ALPHA_LEVEL,
    RESAMPLE_FREQUENCY,
    RUN_DURATION_LOW_FLOW
)
from .core_tests import mann_kendall_test, seasonal_mann_kendall_test
from .utils import (
    ensure_datetime_index,
    check_rolling_mean_compatibility,
    generate_well_display_name,
    generate_well_folder_name,
    generate_well_filename,
    get_well_folder_name_consistent
)
from .export.csv_export import save_basic_mk_analysis_csv
# These will be imported from analysis and export modules once created
# from .analysis.duration_low_flow import calculate_yearly_max_duration_metrics, calculate_percentile_low_flow_metrics
# from .export.csv_export import save_duration_analysis_csv, save_low_flow_analysis_csv
# from .export.csv_export import create_all_wells_duration_analysis_csv, create_all_wells_low_flow_analysis_csv


def run_mann_kendall_analysis(data, well_columns, analysis_type='original', batch_mode=False, 
                             alpha=None, resample_freq=None, use_rolling_mean=None, rolling_window=None):
    """
    Run Mann-Kendall analysis on groundwater data.
    
    Args:
        data: DataFrame with datetime index and well columns
        well_columns: List of well column names
        analysis_type: 'original' or 'seasonal'
        batch_mode: Whether to run in batch mode
        alpha: Significance level
        resample_freq: Data resampling frequency
    
    Returns:
        dict: Analysis results for all wells
    """
    try:
        # Use global settings if none specified
        if alpha is None:
            alpha = ALPHA_LEVEL
        if resample_freq is None:
            resample_freq = RESAMPLE_FREQUENCY
        
        if data is None or well_columns is None or len(well_columns) == 0:
            return {'error': 'Invalid input data or well columns'}
        
        # Check and fix datetime index if needed
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'DateTime' in data.columns:
                print("Converting 'DateTime' column to index...")
                # Create a new DataFrame with DateTime as index
                data = data.set_index('DateTime').copy()
                # Convert index to datetime if it's not already
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)
                print("DateTime index set successfully")
            else:
                print("ERROR: Data must have datetime index or 'DateTime' column")
                print("Available columns:", list(data.columns))
                return {'error': 'Data must have datetime index or DateTime column'}
        
        # Use original data directly (no resampling) to preserve irregular intervals
        results = {}
        
        for well in well_columns:
            if well not in data.columns:
                continue
                
            well_data = data[well].dropna()
            
            # No minimum data requirement - analyze whatever data exists
            if len(well_data) == 0:
                results[well] = {'error': 'No data available for analysis'}
                continue
            
            # Run appropriate MK test
            if analysis_type == 'original':
                mk_result = mann_kendall_test(well_data, alpha=alpha, use_rolling_mean=use_rolling_mean, rolling_window=rolling_window)
            elif analysis_type == 'seasonal':
                mk_result = seasonal_mann_kendall_test(well_data, alpha=alpha)
            else:
                mk_result = {'error': f'Unknown analysis type: {analysis_type}'}
            
            # Add basic statistics
            if 'error' not in mk_result:
                mk_result.update({
                    'well_name': well,
                    'analysis_type': analysis_type,
                    'n_points': len(well_data),
                    'mean_level': well_data.mean(),
                    'min_level': well_data.min(),
                    'max_level': well_data.max(),
                    'std_level': well_data.std(),
                    'start_date': well_data.index[0],
                    'end_date': well_data.index[-1]
                })
            
            results[well] = mk_result
        
        return results
        
    except Exception as e:
        print(f"Error in run_mann_kendall_analysis: {e}")
        return {'error': str(e)}


# Note: run_comprehensive_original_mk_batch_analysis_separate_files will be added here
# once we extract it from the original file. For now, it's imported in the backward compatibility wrapper.


__all__ = [
    'run_mann_kendall_analysis'
]

