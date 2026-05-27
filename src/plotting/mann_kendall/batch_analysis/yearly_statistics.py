"""
Yearly Statistics Batch Analysis

This module contains the batch wrapper function for yearly water level statistics.
"""

import os
import traceback
from ..analysis import calculate_yearly_water_level_statistics
from ..export.csv_export import (
    save_yearly_water_level_statistics_csv,
    create_all_wells_yearly_water_level_statistics_csv
)
from ..utils import (
    check_well_data_availability,
    display_well_data_availability
)


def run_yearly_water_level_statistics(data, well_columns, mapping_dict=None, save_directory='yearly_statistics'):
    """Run yearly water level statistics analysis for all wells."""
    try:
        print(f"Running Yearly Water Level Statistics Analysis...")
        print(f"Save location: {save_directory}")
        
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
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
        
        all_wells_results = {}
        saved_files = []
        
        for well in well_columns:
            if well in data.columns:
                try:
                    well_data = data[well].dropna()
                    # Check if we have sufficient data for meaningful analysis
                    if len(well_data) < 2:
                        print(f"  ⚠ {well}: insufficient data ({len(well_data)} points) - skipping yearly analysis")
                        continue
                    elif len(well_data) > 0:
                        # Calculate yearly statistics for this well
                        yearly_results = calculate_yearly_water_level_statistics(well_data)
                        
                        # Safety check: handle None results or error results
                        if yearly_results is None:
                            print(f"  ⚠ {well}: function returned None, skipping")
                            continue
                        elif isinstance(yearly_results, dict) and 'error' in yearly_results:
                            print(f"  ⚠ {well}: {yearly_results['error']}")
                            continue
                        
                        # Additional validation: ensure yearly_results has the expected structure
                        if not isinstance(yearly_results, dict):
                            print(f"  ⚠ {well}: unexpected result type {type(yearly_results)}, skipping")
                            continue
                        
                        if 'yearly_stats' not in yearly_results:
                            print(f"  ⚠ {well}: missing 'yearly_stats' key, available keys: {list(yearly_results.keys())}")
                            continue
                        
                        if not isinstance(yearly_results['yearly_stats'], list):
                            print(f"  ⚠ {well}: 'yearly_stats' is not a list, got {type(yearly_results['yearly_stats'])}")
                            continue
                        
                        # Save individual well results
                        try:
                            well_files = save_yearly_water_level_statistics_csv(
                                yearly_results, save_directory, well, mapping_dict
                            )
                            saved_files.extend(well_files)
                        except Exception as save_error:
                            print(f"  ⚠ {well}: error saving individual results: {save_error}")
                            continue
                        
                        # Store for all-wells combined CSV
                        all_wells_results[well] = yearly_results
                        
                        print(f"  ✓ {well}: {len(yearly_results.get('yearly_stats', []))} years analysed")
                    else:
                        print(f"  ⚠ {well}: no data available")
                except Exception as e:
                    print(f"  ✗ {well}: error - {e}")
                    traceback.print_exc()
        
        # Create all-wells combined CSV
        if all_wells_results:
            try:
                all_wells_files = create_all_wells_yearly_water_level_statistics_csv(
                    all_wells_results, save_directory, mapping_dict
                )
                if all_wells_files:
                    saved_files.extend(all_wells_files)
                else:
                    print("  ⚠ Combined CSV creation returned no files")
            except Exception as e:
                print(f"  ⚠ Error creating combined CSV: {e}")
                traceback.print_exc()
        
        # Calculate summary statistics
        total_wells = len([w for w in well_columns if w in data.columns])
        processed_wells = len(all_wells_results)
        skipped_wells = total_wells - processed_wells
        
        print(f"Yearly Water Level Statistics completed: {len(saved_files)} files created")
        print(f"Total wells: {total_wells}")
        print(f"Successfully processed: {processed_wells} wells")
        print(f"Skipped (insufficient data): {skipped_wells} wells")
        
        # Final safety check: ensure we return a valid dictionary
        if not isinstance(all_wells_results, dict):
            print(f"  ⚠ Warning: all_wells_results is not a dictionary, got {type(all_wells_results)}")
            all_wells_results = {}
        
        return all_wells_results
        
    except Exception as e:
        print(f"Error in run_yearly_water_level_statistics: {e}")
        traceback.print_exc()
        return {}


__all__ = ['run_yearly_water_level_statistics']

