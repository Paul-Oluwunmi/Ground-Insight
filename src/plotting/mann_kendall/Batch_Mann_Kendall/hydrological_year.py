"""
Hydrological Year Batch Analysis

This module contains the batch wrapper function for hydrological year analysis
(both raw and 7-day rolling mean).
"""

import pandas as pd
import os
import traceback
import sys

# Use ONLY absolute imports to avoid relative import issues when loaded dynamically
# Relative imports cause "attempted relative import with no known parent package" errors
# when modules are loaded dynamically, even inside try/except blocks
from src.plotting.mann_kendall.analysis import (
    calculate_hydrological_year_statistics,
    calculate_hydrological_year_statistics_rolling
)
from src.plotting.mann_kendall.utils import (
    ensure_datetime_index,
    check_well_data_availability,
    display_well_data_availability,
    get_well_folder_name_consistent,
    generate_well_filename
)


def run_hydrological_year_analysis_both_types(data, well_columns, mapping_dict=None, save_directory='hydrological_year_results'):
    """
    Run both raw and 7-day rolling mean hydrological year analysis.
    Creates separate CSV files for each type.
    
    Args:
        data: DataFrame with datetime index and well columns
        well_columns: List of well column names
        mapping_dict: Dictionary mapping well names to display names
        save_directory: Directory to save results
    
    Returns:
        dict: Dictionary with file paths for both analysis types
    """
    try:
        # Check and fix datetime index if needed using our robust function
        data = ensure_datetime_index(data)
        if not isinstance(data.index, pd.DatetimeIndex):
            print("ERROR: Data must have datetime index or 'DateTime' column")
            print("Available columns:", list(data.columns))
            return {}
        
        # Check data availability before starting analysis
        print("Checking well data availability...")
        try:
            availability_info = check_well_data_availability(data, well_columns)
            if availability_info:
                display_well_data_availability(availability_info, mapping_dict)
            else:
                print("  ⚠ Data availability check returned no results, proceeding with analysis...")
        except Exception as e:
            print(f"  ⚠ Error during data availability check: {e}")
            print("  ⚠ Proceeding with analysis anyway...")
            availability_info = {}
        
        # Create save directory
        os.makedirs(save_directory, exist_ok=True)
        
        # Create well-specific folders using consistent naming
        well_folders = {}
        for well in well_columns:
            if well in data.columns:
                well_folder = os.path.join(save_directory, get_well_folder_name_consistent(well, mapping_dict))
                os.makedirs(well_folder, exist_ok=True)
                well_folders[well] = well_folder
        
        # Create all-wells folder
        all_wells_folder = os.path.join(save_directory, "all_wells")
        os.makedirs(all_wells_folder, exist_ok=True)
        
        # Store results for both types
        raw_results = {}
        smoothed_results = {}
        
        # Process each well
        for well in well_columns:
            if well in data.columns:
                well_data = data[well].dropna()
                # Check if we have sufficient data for meaningful analysis
                if len(well_data) < 2:
                    print(f"  ⚠ {well}: insufficient data ({len(well_data)} points) - skipping hydrological year analysis")
                    continue
                elif len(well_data) > 0:
                    # Save original data for this well (before any processing)
                    original_data_file = os.path.join(well_folders[well], generate_well_filename(well, "_original_data.csv", mapping_dict))
                    well_data.to_csv(original_data_file, index=True, index_label='Date')
                    print(f" Original data saved for {well}: {original_data_file}")
                    
                    # 1. RAW hydrological year analysis - use ORIGINAL data
                    print(f"Processing RAW hydrological year analysis for {well} (using original data)...")
                    raw_stats = calculate_hydrological_year_statistics(well_data, well, mapping_dict, False)
                    if raw_stats:
                        raw_results[well] = raw_stats
                        
                        # Save individual well raw results
                        raw_df = pd.DataFrame(raw_stats)
                        raw_file = os.path.join(well_folders[well], generate_well_filename(well, "_hydrological_year_raw.csv", mapping_dict))
                        raw_df.to_csv(raw_file, index=False)
                        print(f"  ✓ {well}: raw hydrological year analysis completed")
                    else:
                        print(f"  ⚠ {well}: raw hydrological year analysis returned no results")
                    
                    # 2. ROLLING MEAN hydrological year analysis - use 7-day rolling mean of original data
                    print(f"Processing ROLLING MEAN hydrological year analysis for {well} (using 7-day rolling mean)...")
                    # Create 7-day rolling mean for smoothed analysis
                    rolling_mean_data = well_data.rolling(window=7, center=False, min_periods=1).mean()
                    smoothed_stats = calculate_hydrological_year_statistics_rolling(rolling_mean_data, well, mapping_dict, False)
                    if smoothed_stats:
                        smoothed_results[well] = smoothed_stats
                        
                        # Save individual well smoothed results
                        smoothed_df = pd.DataFrame(smoothed_stats)
                        smoothed_file = os.path.join(well_folders[well], generate_well_filename(well, "_hydrological_year_rolling_mean.csv", mapping_dict))
                        smoothed_df.to_csv(smoothed_file, index=False)
                        print(f"  ✓ {well}: rolling mean hydrological year analysis completed")
                    else:
                        print(f"  ⚠ {well}: rolling mean hydrological year analysis returned no results")
        
        # Create combined all-wells files
        all_wells_raw = []
        all_wells_smoothed = []
        
        try:
            for well, stats in raw_results.items():
                if stats is None:
                    print(f"  ⚠ Skipping {well} in combined raw results: stats is None")
                    continue
                if not isinstance(stats, list):
                    print(f"  ⚠ Skipping {well} in combined raw results: stats is not a list, got {type(stats)}")
                    continue
                    
                well_display = mapping_dict.get(well, well) if mapping_dict else well
                for stat in stats:
                    if isinstance(stat, dict):
                        stat_copy = stat.copy()
                        stat_copy['Well'] = well_display
                        all_wells_raw.append(stat_copy)
                    else:
                        print(f"  ⚠ Skipping non-dict stat in {well}: {type(stat)}")
        except Exception as e:
            print(f"  ⚠ Error processing raw results for combined CSV: {e}")
        
        try:
            for well, stats in smoothed_results.items():
                if stats is None:
                    print(f"  ⚠ Skipping {well} in combined smoothed results: stats is None")
                    continue
                if not isinstance(stats, list):
                    print(f"  ⚠ Skipping {well} in combined smoothed results: stats is not a list, got {type(stats)}")
                    continue
                    
                well_display = mapping_dict.get(well, well) if mapping_dict else well
                for stat in stats:
                    if isinstance(stat, dict):
                        stat_copy = stat.copy()
                        stat_copy['Well'] = well_display
                        all_wells_smoothed.append(stat_copy)
                    else:
                        print(f"  ⚠ Skipping non-dict stat in {well}: {type(stat)}")
        except Exception as e:
            print(f"  ⚠ Error processing smoothed results for combined CSV: {e}")
        
        # Save combined files
        try:
            if all_wells_raw:
                all_wells_raw_df = pd.DataFrame(all_wells_raw)
                # Reorder columns to put 'Well' first
                if 'Well' in all_wells_raw_df.columns:
                    other_columns = [col for col in all_wells_raw_df.columns if col != 'Well']
                    all_wells_raw_df = all_wells_raw_df[['Well'] + other_columns]
                all_wells_raw_file = os.path.join(all_wells_folder, "all_wells_hydrological_year_raw.csv")
                all_wells_raw_df.to_csv(all_wells_raw_file, index=False)
                print(f"  ✓ Saved combined raw hydrological year results: {len(all_wells_raw)} records")
            else:
                print("  ⚠ No raw results available for combined CSV")
        except Exception as e:
            print(f"  ⚠ Error saving combined raw CSV: {e}")
        
        try:
            if all_wells_smoothed:
                all_wells_smoothed_df = pd.DataFrame(all_wells_smoothed)
                # Reorder columns to put 'Well' first
                if 'Well' in all_wells_smoothed_df.columns:
                    other_columns = [col for col in all_wells_smoothed_df.columns if col != 'Well']
                    all_wells_smoothed_df = all_wells_smoothed_df[['Well'] + other_columns]
                all_wells_smoothed_file = os.path.join(all_wells_folder, "all_wells_hydrological_year_rolling_mean.csv")
                all_wells_smoothed_df.to_csv(all_wells_smoothed_file, index=False)
                print(f"  ✓ Saved combined rolling mean hydrological year results: {len(all_wells_smoothed)} records")
            else:
                print("  ⚠ No smoothed results available for combined CSV")
        except Exception as e:
            print(f"  ⚠ Error saving combined smoothed CSV: {e}")
        
        # Calculate summary statistics
        total_wells = len([w for w in well_columns if w in data.columns])
        processed_wells = len(raw_results) + len(smoothed_results)
        skipped_wells = total_wells - processed_wells
        
        print(f"\nHydrological year analysis completed!")
        print(f"Total wells: {total_wells}")
        print(f"Successfully processed: {processed_wells} wells")
        print(f"Skipped (insufficient data): {skipped_wells} wells")
        print(f"Raw analysis: {len(raw_results)} wells")
        print(f"Rolling mean analysis: {len(smoothed_results)} wells")
        print(f"Results saved to: {save_directory}")
        
        return {
            'raw_results': raw_results,
            'smoothed_results': smoothed_results,
            'save_directory': save_directory
        }
        
    except Exception as e:
        print(f"Error in run_hydrological_year_analysis_both_types: {e}")
        traceback.print_exc()
        return {}


__all__ = ['run_hydrological_year_analysis_both_types']

