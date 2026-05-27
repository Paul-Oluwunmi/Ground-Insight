"""
Original MK Comprehensive Batch Analysis

This module contains the comprehensive batch wrapper function for Original MK analysis.
"""

import pandas as pd
import os
import gc
import traceback
import sys

# Use ONLY absolute imports to avoid relative import issues when loaded dynamically
# Relative imports cause "attempted relative import with no known parent package" errors
# when modules are loaded dynamically, even inside try/except blocks
from src.plotting.mann_kendall.constants import (
    ALPHA_LEVEL,
    RESAMPLE_FREQUENCY,
    USE_ROLLING_MEAN,
    ROLLING_WINDOW_SIZE,
    RUN_DURATION_LOW_FLOW
)
from src.plotting.mann_kendall.batch_analysis import run_mann_kendall_analysis
from src.plotting.mann_kendall.export.csv_export import (
    save_basic_mk_analysis_csv,
    save_duration_analysis_csv,
    save_low_flow_analysis_csv,
    create_all_wells_duration_analysis_csv,
    create_all_wells_low_flow_analysis_csv
)
from src.plotting.mann_kendall.analysis import (
    calculate_yearly_max_duration_metrics,
    calculate_percentile_low_flow_metrics
)
from src.plotting.mann_kendall.utils import (
    ensure_datetime_index,
    check_rolling_mean_compatibility,
    generate_well_folder_name,
    generate_well_filename,
    get_well_folder_name_consistent
)


def run_comprehensive_original_mk_batch_analysis_separate_files(data, well_columns, mapping_dict=None,
                                                             save_directory='original_mk_batch_results',
                                                             alpha=None, resample_freq=None,
                                                             start_date=None, end_date=None,
                                                             create_folders=False, save_plots=False,
                                                             use_rolling_mean=None, rolling_window=None,
                                                             custom_save_location=None):
    """
    Run comprehensive Original MK batch analysis and save results to separate CSV files.
    
    This function produces:
    1. Basic MK Analysis CSV (trend, p-value, slope, etc.)
    2. Duration Analysis CSV (declining/rising periods)
    3. Low Flow Analysis CSV (percentile thresholds)
    """
    # Use global settings if none specified
    if alpha is None:
        alpha = ALPHA_LEVEL
    if resample_freq is None:
        resample_freq = RESAMPLE_FREQUENCY
    
    # Use global rolling mean setting if none specified
    if use_rolling_mean is None:
        use_rolling_mean = USE_ROLLING_MEAN
    
    # Use global rolling window if none specified
    if rolling_window is None:
        rolling_window = ROLLING_WINDOW_SIZE
    
    try:
        print("Running COMPREHENSIVE Original MK Batch Analysis...")
        print("=" * 60)
        
        # Check and fix datetime index FIRST - before any other operations
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
                return [], []
        else:
            print("Index is already datetime")
        
        # Determine save location
        if custom_save_location:
            final_save_dir = custom_save_location
        else:
            final_save_dir = os.path.join(save_directory, "original_mk_analysis")
        
        print(f"Save location: {final_save_dir}")
        
        # Create output directory (always create, regardless of create_folders flag)
        os.makedirs(final_save_dir, exist_ok=True)
        
        # Check rolling mean compatibility AFTER datetime index is properly set and save directory is determined
        target_years = ROLLING_WINDOW_SIZE / 365.25
        print(f"Checking rolling mean compatibility for {len(well_columns)} wells...")
        
        compatibility_info = check_rolling_mean_compatibility(
            data=data, 
            well_columns=well_columns, 
            target_years=target_years
        )
        
        # Save compatibility check results
        compatibility_df = pd.DataFrame.from_dict(compatibility_info, orient='index')
        compatibility_csv_path = os.path.join(final_save_dir, '00_Rolling_Mean_Compatibility.csv')
        # Ensure directory exists before saving (redundant but safe)
        os.makedirs(os.path.dirname(compatibility_csv_path), exist_ok=True)
        compatibility_df.to_csv(compatibility_csv_path, index=True, index_label='Well')
        print(f"Rolling mean compatibility saved to: {compatibility_csv_path}")
        
        # Filter data by date range if specified
        filtered_data = data.copy()
        if start_date:
            filtered_data = filtered_data[filtered_data.index >= start_date]
        if end_date:
            filtered_data = filtered_data[filtered_data.index <= end_date]
        
        # Run MK analysis
        mk_results = run_mann_kendall_analysis(
            data=filtered_data,
            well_columns=well_columns,
            analysis_type='original',
            batch_mode=True,
            alpha=alpha,
            resample_freq=resample_freq,
            use_rolling_mean=use_rolling_mean,
            rolling_window=rolling_window
        )
        
        # Save basic MK analysis results
        basic_mk_csv = save_basic_mk_analysis_csv(mk_results, final_save_dir, mapping_dict)
        
        # Create well-specific folders and save individual well analysis (if enabled)
        duration_csv_files = []
        low_flow_csv_files = []
        
        # Start processing duration and low flow analysis
        
        for well in well_columns:
            if well in data.columns:
                # Get mapped well name for folder structure (not used for folder creation anymore)
                well_display = generate_well_folder_name(well, mapping_dict)
                # Use daily resampling and keep NaN values to match dashboard analysis and get December 18th
                # This matches the data processing used in dashboard plotting functions
                well_data = data[well].resample('D').mean()
                # Ensure the resampled data has proper datetime index
                well_data = ensure_datetime_index(well_data)
                # Process all wells with any data after cleaning
                if len(well_data) > 0:
                    # Create well-specific folder using consistent naming
                    well_folder_name = get_well_folder_name_consistent(well, mapping_dict)
                    well_folder = os.path.join(final_save_dir, well_folder_name)
                    os.makedirs(well_folder, exist_ok=True)
                    
                    # Save original data for this well
                    original_data_file = os.path.join(well_folder, generate_well_filename(well, "_original_data.csv", mapping_dict))
                    well_data.to_csv(original_data_file, index=True, index_label='Date')
                    print(f" Original data saved for {well}: {original_data_file}")
                    
                    # Save resampled data for this well
                    resampled_well_data = well_data.resample(resample_freq).mean().dropna()
                    resampled_data_file = os.path.join(well_folder, generate_well_filename(well, f"_resampled_{resample_freq}_data.csv", mapping_dict))
                    resampled_well_data.to_csv(resampled_data_file, index=True, index_label='Date')
                    print(f" Resampled data saved for {well}: {resampled_data_file}")
                    
                    # Clean up resampled data after saving
                    del resampled_well_data
                    
                    # Save duration analysis for this well (if enabled)
                    if RUN_DURATION_LOW_FLOW:
                        duration_results = calculate_yearly_max_duration_metrics(well_data)
                        if 'error' not in duration_results:
                            csv_file = save_duration_analysis_csv(duration_results, well_folder, well, mapping_dict)
                            if csv_file:
                                duration_csv_files.append(csv_file)
                        # Clean up intermediate results
                        del duration_results
                    
                    # Save low flow analysis for this well (if enabled)
                    if RUN_DURATION_LOW_FLOW:
                        print(f"\n{'='*80}")
                        print(f"LOW FLOW ANALYSIS FOR: {well}")
                        print(f"{'='*80}")
                        
                        # Run hybrid method only
                        print(f"\n Hybrid Method (Original data + Rolling mean extension)")
                        low_flow_results = calculate_percentile_low_flow_metrics(well_data)
                        if 'error' not in low_flow_results:
                            csv_file = save_low_flow_analysis_csv(low_flow_results, well_folder, well, mapping_dict)
                            if csv_file:
                                low_flow_csv_files.append(csv_file)
                        
                        # Show results summary
                        if 'error' not in low_flow_results:
                            print(f"\n Results Summary:")
                            print(f"   10th Percentile: {low_flow_results.get('percentile_10th', 'N/A')}")
                            print(f"   5th Percentile: {low_flow_results.get('percentile_5th', 'N/A')}")
                            print(f"   Total Days Below 10th: {low_flow_results.get('total_days_below_10th_per_year', 'N/A')}")
                            print(f"   Total Days Below 5th: {low_flow_results.get('total_days_below_5th_per_year', 'N/A')}")
                        
                        # Clean up intermediate results
                        del low_flow_results
                
                    # Clean up well data after processing
                    del well_data
                
                    # Progress indicator every 10 wells
                    if well_columns.index(well) % 10 == 9:  # Every 10th well (0-indexed)
                        print(f"  Progress: Processed {well_columns.index(well) + 1}/{len(well_columns)} wells")
        
        # Create all-wells folder and save combined CSVs (if enabled)
        all_wells_duration_csv = []
        all_wells_low_flow_csv = []
        
        if RUN_DURATION_LOW_FLOW:
            all_wells_folder = os.path.join(final_save_dir, "all_wells")
            os.makedirs(all_wells_folder, exist_ok=True)
            
            # Store results for all-wells combined files
            all_wells_duration_results = []
            all_wells_low_flow_results = []
            
            for well in well_columns:
                if well in filtered_data.columns:
                    # Get mapped well name for folder structure (not used for folder creation anymore)
                    well_display = generate_well_folder_name(well, mapping_dict)
                    well_data = filtered_data[well].dropna()
                    if len(well_data) > 0:
                        # Duration analysis
                        duration_results = calculate_yearly_max_duration_metrics(well_data)
                        if 'error' not in duration_results:
                            # Save individual well files first to get processed data
                            well_folder = os.path.join(final_save_dir, get_well_folder_name_consistent(well, mapping_dict))
                            os.makedirs(well_folder, exist_ok=True)
                            duration_save_result = save_duration_analysis_csv(duration_results, well_folder, well, mapping_dict)
                            if duration_save_result:
                                all_wells_duration_results.append(duration_save_result)
                        
                        # Low flow analysis
                        low_flow_results = calculate_percentile_low_flow_metrics(well_data)
                        if 'error' not in low_flow_results:
                            # Save individual well files first to get processed data
                            well_folder = os.path.join(final_save_dir, get_well_folder_name_consistent(well, mapping_dict))
                            os.makedirs(well_folder, exist_ok=True)
                            low_flow_save_result = save_low_flow_analysis_csv(low_flow_results, well_folder, well, mapping_dict)
                            if low_flow_save_result:
                                all_wells_low_flow_results.append(low_flow_save_result)
            
            # Create combined CSVs using the stored results
            if all_wells_duration_results:
                all_wells_duration_csv = create_all_wells_duration_analysis_csv(
                    all_wells_duration_results, all_wells_folder, mapping_dict
                )
            
            if all_wells_low_flow_results:
                all_wells_low_flow_csv = create_all_wells_low_flow_analysis_csv(
                    all_wells_low_flow_results, all_wells_folder, mapping_dict
                )
        
        # Collect all CSV files
        all_csv_files = []
        if basic_mk_csv:
            all_csv_files.append(basic_mk_csv)
        all_csv_files.extend(duration_csv_files)
        all_csv_files.extend(low_flow_csv_files)
        if all_wells_duration_csv:
            all_csv_files.append(all_wells_duration_csv)
        if all_wells_low_flow_csv:
            all_csv_files.append(all_wells_low_flow_csv)
        
        print(f"\nOriginal MK Analysis completed!")
        if RUN_DURATION_LOW_FLOW:
            print(f"Duration & Low Flow Analysis: Enabled")
        else:
            print(f"Duration & Low Flow Analysis: Skipped (RUN_DURATION_LOW_FLOW = False)")
        print(f"Total CSV files created: {len(all_csv_files)}")
        print(f"Output location: {final_save_dir}")
        
        # Clean up large data structures and force garbage collection
        if 'filtered_data' in locals():
            del filtered_data
        if 'data' in locals():
            del data
        
        # Force garbage collection to free memory
        gc.collect()
        
        return all_csv_files, [final_save_dir]
        
    except Exception as e:
        print(f"Error in run_comprehensive_original_mk_batch_analysis_separate_files: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return [], []


__all__ = ['run_comprehensive_original_mk_batch_analysis_separate_files']

