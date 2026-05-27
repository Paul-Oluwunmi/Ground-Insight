"""
Time Evolution Analysis Module

This module contains functions for time evolution analysis using rolling windows.
"""

import pandas as pd
import os
from ..constants import (
    TIME_EVOLUTION_WINDOW_SIZE,
    TIME_EVOLUTION_STEP_SIZE,
    TIME_EVOLUTION_OVERLAP,
    TIME_EVOLUTION_MIN_POINTS,
    RUN_TIME_EVOLUTION
)
from ..utils import (
    ensure_datetime_index,
    generate_well_display_name,
    generate_well_folder_name,
    generate_well_filename,
    get_well_folder_name_consistent,
    detect_data_format
)
from ..core_tests import mann_kendall_test


def calculate_simple_time_evolution_analysis(data, rolling_window=None, step_size=None, overlap=None):
    """
    Calculate time evolution analysis for a single well using rolling windows.
    FIRST applies rolling mean smoothing to the data, THEN creates time windows from the smoothed data.
    
    Args:
        data: pandas Series with datetime index and groundwater levels
        rolling_window: Size of rolling window for smoothing (days) AND analysis window size
        step_size: Step size between windows in days
        overlap: Whether windows should overlap
    
    Returns:
        dict: Dictionary containing window analysis results
    """
    try:
        # Use configurable parameters if not specified
        if rolling_window is None:
            rolling_window = TIME_EVOLUTION_WINDOW_SIZE
        if step_size is None:
            step_size = TIME_EVOLUTION_STEP_SIZE
        if overlap is None:
            overlap = TIME_EVOLUTION_OVERLAP
        
        # No minimum data requirement - analyze whatever data exists
        if len(data) == 0:
            return {'error': 'No data available for analysis'}
        
        # Step 1: Apply rolling mean smoothing FIRST (like your backup does)
        print(f"  Step 1: Applying {rolling_window}-day rolling mean smoothing...")
        smoothed_data = data.copy()
        
        # Only apply rolling mean if we have enough data and rolling window > 1
        if rolling_window > 1 and len(smoothed_data) >= rolling_window:
            # Apply rolling mean smoothing
            rolling_mean = smoothed_data.rolling(window=rolling_window, center=False).mean()
            # Remove NaN values from rolling mean
            smoothed_data = rolling_mean.dropna()
            print(f"  Applied rolling mean smoothing. Smoothed data length: {len(smoothed_data)} points")
        else:
            print(f"  Skipped rolling mean smoothing (window={rolling_window}, series_length={len(smoothed_data)})")
        
        # Resample to daily frequency if needed (after smoothing)
        if smoothed_data.index.freq is None or smoothed_data.index.freq != pd.Timedelta('1D'):
            smoothed_data = smoothed_data.resample('D').mean()
        
        windows = []
        window_number = 1
        
        # Calculate number of windows based on actual time span of SMOOTHED data
        start_date = smoothed_data.index[0]
        end_date = smoothed_data.index[-1]
        
        # Handle different datetime subtraction results
        time_diff = end_date - start_date
        if hasattr(time_diff, 'days'):
            total_time_span_days = time_diff.days
        elif hasattr(time_diff, 'total_seconds'):
            total_time_span_days = int(time_diff.total_seconds() / (24 * 3600))
        else:
            # Fallback: convert to days if it's a numeric value
            total_time_span_days = int(time_diff / pd.Timedelta(days=1))
        
        print(f"  Smoothed data time span: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({total_time_span_days} days)")
        
        if overlap:
            num_windows = max(1, (total_time_span_days - rolling_window) // step_size + 1)
        else:
            num_windows = max(1, total_time_span_days // step_size)
        
        print(f"  Creating {num_windows} windows with {rolling_window}-day analysis window and {step_size}-day step size")
        
        # Create windows based on time from SMOOTHED data
        for i in range(0, total_time_span_days - rolling_window + 1, step_size):
            try:
                # Extract window data based on time from SMOOTHED data
                window_start_date = start_date + pd.Timedelta(days=i)
                window_end_date = window_start_date + pd.Timedelta(days=rolling_window)
                
                # Get data within this time window from SMOOTHED data
                window_data = smoothed_data[(smoothed_data.index >= window_start_date) & (smoothed_data.index <= window_end_date)]
                
                print(f"  Window {window_number}: {window_start_date.strftime('%Y-%m-%d')} to {window_end_date.strftime('%Y-%m-%d')} - {len(window_data)} data points")
                
                if len(window_data) < TIME_EVOLUTION_MIN_POINTS:  # Use configurable minimum points
                    print(f"  Skipping window {window_number}: insufficient data points ({len(window_data)} < {TIME_EVOLUTION_MIN_POINTS})")
                    continue
                
                # Get window dates with safety checks
                if len(window_data) == 0:
                    print(f"  Skipping window {window_number}: no data in window")
                    continue
                
                window_start = window_data.index[0]
                window_end = window_data.index[-1]
                
                # Ensure we have valid datetime objects
                if not isinstance(window_start, pd.Timestamp) or not isinstance(window_end, pd.Timestamp):
                    print(f"  Skipping window {window_number}: invalid datetime objects")
                    continue
                
                # Format period as "dd-MMM-yyyy to dd-MMM-yyyy"
                try:
                    period = f"{window_start.strftime('%d-%b-%Y')} to {window_end.strftime('%d-%b-%Y')}"
                except Exception as e:
                    print(f"  Warning: Could not format period for window {window_number}: {e}")
                    period = f"Window {window_number}"
                
                # Run Mann-Kendall test on the already-smoothed window data (no additional rolling mean needed)
                mk_result = mann_kendall_test(window_data, use_rolling_mean=False)
                
                # Get trend direction with safe fallback
                trend = mk_result.get('trend', 'no_trend') if mk_result else 'no_trend'
                if trend is None or trend == 'None':
                    trend = 'no_trend'
                
                # Get p-value with safe fallback
                p_value = mk_result.get('p_value', 1.0) if mk_result else 1.0
                if p_value is None or pd.isna(p_value):
                    p_value = 1.0
                
                # Calculate Sen's slope in mm/year
                slope_mm_year = 0.0
                if mk_result and 'slope' in mk_result:
                    slope = mk_result['slope']
                    if pd.notna(slope) and slope is not None and not (isinstance(slope, (int, float)) and slope == 0):
                        try:
                            slope_mm_year = slope * 1000  # Convert m/day to mm/year
                        except (TypeError, ValueError):
                            slope_mm_year = 0.0
                    else:
                        slope_mm_year = 0.0
                else:
                    slope_mm_year = 0.0
                
                # Ensure slope_mm_year is a valid number
                if not isinstance(slope_mm_year, (int, float)) or pd.isna(slope_mm_year):
                    slope_mm_year = 0.0
                
                # Calculate mean level
                mean_level = window_data.mean()
                
                # Ensure mean_level is a valid number
                if not isinstance(mean_level, (int, float)) or pd.isna(mean_level):
                    mean_level = 0.0
                
                # Safe formatting - handle None/NaN values
                safe_slope = slope_mm_year if pd.notna(slope_mm_year) and slope_mm_year is not None else 0.0
                safe_mean_level = mean_level if pd.notna(mean_level) and mean_level is not None else 0.0
                safe_p_value = p_value if pd.notna(p_value) and p_value is not None else 1.0
                
                # Create interpretation
                if trend == 'increasing':
                    interpretation = f"Increasing trend of {safe_slope:.4f} mm/year"
                elif trend == 'decreasing':
                    interpretation = f"Decreasing trend of {safe_slope:.4f} mm/year"
                else:
                    interpretation = f"No significant trend"
                
                # Create window result - EXACT format as user's table
                window_result = {
                    'Block': f"Window {window_number}",
                    'Period': period,
                    'Trend': trend,
                    'P-value': f"{safe_p_value:.4f}",
                    'Slope (mm/year)': f"{safe_slope:.4f}",
                    'Mean Level (m)': f"{safe_mean_level:.4f}",
                    'Interpretation': interpretation
                }
                
                windows.append(window_result)
                window_number += 1
                
            except Exception as e:
                print(f"Error processing window {window_number}: {e}")
                continue
        
        print(f"  Created {len(windows)} windows for analysis")
        
        return {
            'windows': windows,
            'rolling_window': rolling_window,
            'step_size': step_size,
            'overlap': overlap,
            'total_windows': len(windows)
        }
        
    except Exception as e:
        print(f"Error in calculate_simple_time_evolution_analysis: {e}")
        return {'error': str(e)}


def run_time_evolution_analysis(data, well_columns, mapping_dict=None, save_directory='time_evolution'):
    """Run time evolution analysis for all wells."""
    try:
        print(f"Running Time Evolution Analysis...")
        print(f"Save location: {save_directory}")
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Use the existing simple time evolution analysis function
        time_evolution_results = run_simple_time_evolution_analysis(
            data=data,
            well_columns=well_columns,
            mapping_dict=mapping_dict,
            custom_save_location=save_directory,
            rolling_window=TIME_EVOLUTION_WINDOW_SIZE,
            step_size=TIME_EVOLUTION_STEP_SIZE,
            overlap=TIME_EVOLUTION_OVERLAP
        )
        
        if time_evolution_results and 'save_directory' in time_evolution_results:
            print(f"  Time Evolution Analysis completed successfully")
            return time_evolution_results
        else:
            print(f"  Time Evolution Analysis failed")
            return {}
            
    except Exception as e:
        print(f"Error in run_time_evolution_analysis: {e}")
        return {}


def run_simple_time_evolution_analysis(data, well_columns, mapping_dict=None, 
                                      rolling_window=None, step_size=None, 
                                      overlap=None, custom_save_location=None):
    """
    Run simple time evolution analysis using rolling windows for trend analysis.
    Produces the exact same table format as the user's original analysis.
    
    Args:
        data: pandas DataFrame with datetime index and groundwater levels
        well_columns: List of well column names
        mapping_dict: Dictionary mapping well names to display names
        rolling_window: Size of rolling window in days
        step_size: Step size between windows in days
        overlap: Whether windows should overlap
        custom_save_location: Custom directory to save results
    
    Returns:
        dict: Dictionary containing analysis results and file paths
    """
    try:
        # Check if time evolution analysis is enabled
        if not RUN_TIME_EVOLUTION:
            print("Time Evolution Analysis is disabled (RUN_TIME_EVOLUTION = False)")
            return {'error': 'Time evolution analysis is disabled'}
        
        # Use configurable parameters if not specified
        if rolling_window is None:
            rolling_window = TIME_EVOLUTION_WINDOW_SIZE
        if step_size is None:
            step_size = TIME_EVOLUTION_STEP_SIZE
        if overlap is None:
            overlap = TIME_EVOLUTION_OVERLAP
        
        # Detect data format and provide compatibility information
        print("Detecting data format for compatibility...")
        format_info = detect_data_format(data)
        if format_info.get('compatibility_notes'):
            print("Data format compatibility:")
            for note in format_info['compatibility_notes']:
                print(f"  • {note}")
        
        # Ensure data has proper datetime index
        print("Ensuring datetime index compatibility...")
        data = ensure_datetime_index(data)
        
        print(f"Running Simple Time Evolution Analysis...")
        print(f"Rolling Window: {rolling_window} days")
        print(f"Step Size: {step_size} days")
        print(f"Overlap: {overlap}")
        
        # Debug: Show data time span
        if len(data.columns) > 0:
            sample_well = data.columns[0]
            sample_data = data[sample_well].dropna()
            if len(sample_data) > 0:
                sample_start_date = sample_data.index[0]
                sample_end_date = sample_data.index[-1]
                sample_total_days = (sample_end_date - sample_start_date).days
                print(f"Sample data time span: {sample_start_date.strftime('%Y-%m-%d')} to {sample_end_date.strftime('%Y-%m-%d')} ({sample_total_days} days)")
                expected_windows = max(1, (sample_total_days - rolling_window) // step_size + 1)
                print(f"Expected windows: {expected_windows}")
        
        # Create save directory
        if custom_save_location:
            save_dir = custom_save_location
        else:
            save_dir = "time_evolution_analysis"
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Store results for each well
        all_results = {}
        combined_results = []
        
        # Import CSV export function
        from ..export.csv_export import save_simple_time_evolution_analysis_csv
        
        for well in well_columns:
            try:
                print(f"Processing well: {well}")
                
                # Get well data with enhanced compatibility
                if well in data.columns:
                    # Use the compatibility function to ensure proper datetime index
                    well_data = data[well].dropna()
                    if len(well_data) > 0:
                        # Ensure the index is datetime using our robust function
                        well_data = ensure_datetime_index(well_data)
                        if not isinstance(well_data.index, pd.DatetimeIndex):
                            print(f"  Warning: Could not convert index to datetime for {well}")
                            continue
                else:
                    print(f"  Warning: Well {well} not found in dataset")
                    continue
                
                # No minimum data requirement - analyze whatever data exists
                if len(well_data) == 0:
                    print(f"Warning: No data available for {well}")
                    continue
                
                # Check if well has enough data for the rolling window before processing
                data_years = len(well_data) / 365.25  # Approximate years of data
                rolling_years = rolling_window / 365.25  # Rolling window in years
                
                if data_years < rolling_years:
                    print(f"  Skipping {well}: insufficient data ({data_years:.1f} years) for {rolling_years:.1f}-year rolling window")
                    continue
                
                # Run time evolution analysis for this well
                well_results = calculate_simple_time_evolution_analysis(
                    well_data, rolling_window, step_size, overlap
                )
                
                if well_results and 'error' not in well_results and well_results.get('windows'):
                    all_results[well] = well_results
                    
                    # Add well identifier for combined results
                    for result in well_results.get('windows', []):
                        result_copy = result.copy()
                        result_copy['Well'] = mapping_dict.get(well, well) if mapping_dict else well
                        combined_results.append(result_copy)
                        
                    # Create well-specific folder and save individual well results ONLY for successful runs
                    well_display = generate_well_folder_name(well, mapping_dict)
                    well_folder = os.path.join(save_dir, get_well_folder_name_consistent(well, mapping_dict))
                    os.makedirs(well_folder, exist_ok=True)
                    
                    well_csv_path = save_simple_time_evolution_analysis_csv(
                        well_results, well_folder, well, mapping_dict
                    )
                    if well_csv_path:
                        print(f"✓ Created folder and saved results for {well}: {well_csv_path}")
                        
                else:
                    print(f"  Skipping {well}: analysis failed or produced no valid windows")
                    print(f"Warning: No valid results for {well}")
                    
            except Exception as e:
                print(f"Error processing {well}: {e}")
                continue
        
        # Create all-wells folder and save combined results
        all_wells_folder = os.path.join(save_dir, "all_wells")
        os.makedirs(all_wells_folder, exist_ok=True)
        
        combined_csv_path = None
        if combined_results:
            combined_results_df = pd.DataFrame(combined_results)
            # Reorder columns to put 'Well' first
            if 'Well' in combined_results_df.columns:
                other_columns = [col for col in combined_results_df.columns if col != 'Well']
                combined_results_df = combined_results_df[['Well'] + other_columns]
            combined_csv_path = os.path.join(all_wells_folder, f"all_wells_time_evolution_analysis.csv")
            combined_results_df.to_csv(combined_csv_path, index=False)
            print(f"Saved combined results: {combined_csv_path}")
        
        # Create summary CSV showing which wells succeeded and which were skipped
        summary_data = []
        for well in well_columns:
            if well in all_results:
                # Well succeeded
                summary_data.append({
                    'Well': mapping_dict.get(well, well) if mapping_dict else well,
                    'Status': 'SUCCESS',
                    'Rolling Window (days)': rolling_window,
                    'Windows Created': len(all_results[well].get('windows', [])),
                    'Data Points': len(data[well].dropna()) if well in data.columns else 0,
                    'Data Years': len(data[well].dropna()) / 365.25 if well in data.columns else 0,
                    'Failure Reason': 'None'
                })
            else:
                # Well was skipped
                well_data = data[well].dropna() if well in data.columns else pd.Series()
                data_years = len(well_data) / 365.25
                rolling_years = rolling_window / 365.25
                
                if data_years < rolling_years:
                    failure_reason = f"Insufficient data ({data_years:.1f} years) for {rolling_years:.1f}-year rolling window"
                else:
                    failure_reason = "Analysis failed or produced no valid windows"
                
                summary_data.append({
                    'Well': mapping_dict.get(well, well) if mapping_dict else well,
                    'Status': 'SKIPPED',
                    'Rolling Window (days)': rolling_window,
                    'Windows Created': 0,
                    'Data Points': len(well_data),
                    'Data Years': data_years,
                    'Failure Reason': failure_reason
                })
        
        # Save summary CSV
        if summary_data:
            summary_csv_path = os.path.join(all_wells_folder, "Time_Evolution_Summary.csv")
            summary_df = pd.DataFrame(summary_data)
            # Reorder columns to put 'Well' first
            if 'Well' in summary_df.columns:
                other_columns = [col for col in summary_df.columns if col != 'Well']
                summary_df = summary_df[['Well'] + other_columns]
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Saved summary CSV: {summary_csv_path}")
            print(f"Summary: {len([w for w in summary_data if w['Status'] == 'SUCCESS'])} wells succeeded, {len([w for w in summary_data if w['Status'] == 'SKIPPED'])} wells skipped")
        
        return {
            'individual_results': all_results,
            'combined_results': combined_results,
            'combined_csv_path': combined_csv_path,
            'save_directory': save_dir
        }
        
    except Exception as e:
        print(f"Error in run_simple_time_evolution_analysis: {e}")
        return None


__all__ = [
    'calculate_simple_time_evolution_analysis',
    'run_time_evolution_analysis',
    'run_simple_time_evolution_analysis'
]
