"""
Dashboard Function for Mann-Kendall Analysis

This module contains the main dashboard function that orchestrates
all analysis types.
"""

import pandas as pd
import os
import traceback
from .constants import (
    CUSTOM_SAVE_LOCATION,
    CREATE_FOLDERS,
    SAVE_PLOTS,
    USE_ROLLING_MEAN,
    ROLLING_WINDOW_SIZE,
    RUN_ORIGINAL_MK,
    RUN_SEASONAL_MK,
    RUN_HYDROLOGICAL_YEAR,
    RUN_YEARLY_STATISTICS,
    RUN_TIME_EVOLUTION
)
from .utils import safe_check_data_structure
from .plotting import run_gwl_rolling_mean_plots
from .analysis import run_time_evolution_analysis

# Import batch wrapper functions from original file for now
# TODO: Migrate these batch wrapper functions to appropriate modules
# These functions are still in the original mann_kendall.py file:
# - run_comprehensive_original_mk_batch_analysis_separate_files
# - run_comprehensive_seasonal_mk_batch_analysis_separate_files
# - run_hydrological_year_analysis_both_types
# - run_yearly_water_level_statistics
#
# For now, we import them from the original file which acts as a backward compatibility wrapper


def run_mann_kendall_dashboard(
    data=None,
    well_columns=None,
    mapping_dict=None,
    custom_save_location=None,
    create_folders=None,
    save_plots=None
):
    """
    Run BOTH Original and Seasonal Mann-Kendall analysis for dashboard.
    """
    # Import batch wrapper functions from modular locations
    from ..Batch_Mann_Kendall import (
        run_comprehensive_original_mk_batch_analysis_separate_files,
        run_comprehensive_seasonal_mk_batch_analysis_separate_files,
        run_hydrological_year_analysis_both_types,
        run_yearly_water_level_statistics
    )
    
    # Use global settings if none specified
    if custom_save_location is None:
        custom_save_location = CUSTOM_SAVE_LOCATION
        print(f"Using global custom save location: {custom_save_location}")
    if create_folders is None:
        create_folders = CREATE_FOLDERS
        print(f"Using global create folders setting: {create_folders}")
    if save_plots is None:
        save_plots = SAVE_PLOTS
        print(f"Using global save plots setting: {save_plots}")
    
    try:
        # Check if required parameters are provided
        if data is None:
            print("ERROR: 'data' parameter is required")
            return [], []
        if well_columns is None:
            print("ERROR: 'well_columns' parameter is required")
            return [], []
        if mapping_dict is None:
            mapping_dict = {}  # Use empty dict as default
        
        # Safe data structure check to prevent timestamp interval errors
        if not safe_check_data_structure(data, "run_mann_kendall_dashboard"):
            print("ERROR: Data structure check failed. Cannot proceed with analysis.")
            return [], []
        
        print("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
        print("║                                  MANN-KENDALL TREND ANALYSIS DASHBOARD                               ║")
        print("║                                       Groundwater Trend Assessment                                   ║")
        print("╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
        print()
        print("  Analysis Components:")
        print("   • Original Mann-Kendall Trend Tests")
        print("   • Seasonal Mann-Kendall Analysis")
        print("   • Hydrological Year Statistics")
        print("   • Groundwater Level Time Series Plots")
        print("   • Duration & Low Flow Analysis")
        print("   • Original & Resampled Data Files")
        print()
        print("=" * 80)
        print(f"  Analysis Parameters:")
        print(f"   • Total Wells: {len(well_columns)}")
        print(f"   • Custom Save Location: {custom_save_location}")
        print(f"   • Create Folders: {create_folders}")
        print(f"   • Save Plots: {save_plots}")
        print(f"   • Rolling Mean: {USE_ROLLING_MEAN}")
        if USE_ROLLING_MEAN:
            print(f"   • Rolling Window Size: {ROLLING_WINDOW_SIZE} days ({ROLLING_WINDOW_SIZE/365.25:.1f} years)")
        print("=" * 80)
        
        # Check rolling mean compatibility using STRICT requirements
        # NOTE: This runs AFTER datetime index is set in the individual analysis functions
        target_years = ROLLING_WINDOW_SIZE / 365.25
        print(f"  Rolling mean compatibility will be checked during individual analysis phases")
        
        # 1. RUN ORIGINAL MK BATCH ANALYSIS (if enabled)
        if RUN_ORIGINAL_MK:
            print("\n" + "=" * 80)
            print("  PHASE 1: ORIGINAL MANN-KENDALL TREND ANALYSIS")
            print("=" * 80)
            original_csv_files, original_dirs = run_comprehensive_original_mk_batch_analysis_separate_files(
                data=data,
                well_columns=well_columns,
                mapping_dict=mapping_dict,
                save_directory=os.path.join(custom_save_location, '01_Original_MK_Analysis'),
                alpha=None,
                resample_freq=None,
                create_folders=create_folders,
                save_plots=save_plots,
                custom_save_location=None
            )
        else:
            print("\n" + "=" * 80)
            print("  PHASE 1: ORIGINAL MANN-KENDALL TREND ANALYSIS (SKIPPED)")
            print("=" * 80)
            print("  • Original MK analysis is disabled (RUN_ORIGINAL_MK = False)")
            original_csv_files, original_dirs = [], []
        
        # 2. RUN SEASONAL MK BATCH ANALYSIS (if enabled)
        if RUN_SEASONAL_MK:
            print("\n" + "=" * 80)
            print("  PHASE 2: SEASONAL MANN-KENDALL ANALYSIS")
            print("=" * 80)
            seasonal_csv_files, seasonal_dirs = run_comprehensive_seasonal_mk_batch_analysis_separate_files(
                data=data,
                well_columns=well_columns,
                mapping_dict=mapping_dict,
                save_directory=os.path.join(custom_save_location, '02_Seasonal_MK_Analysis'),
                alpha=None,
                create_folders=create_folders,
                save_plots=save_plots,
                custom_save_location=None
            )
            
            # Debug: Check what we got back from seasonal analysis
            print(f"  DEBUG: Seasonal MK results:")
            print(f"    CSV files: {len(seasonal_csv_files) if seasonal_csv_files else 0}")
            print(f"    Directories: {len(seasonal_dirs) if seasonal_dirs else 0}")
            if seasonal_csv_files:
                print(f"    First few files: {seasonal_csv_files[:3]}")
        else:
            print("\n" + "=" * 80)
            print("  PHASE 2: SEASONAL MANN-KENDALL ANALYSIS (SKIPPED)")
            print("=" * 80)
            print("  • Seasonal analysis is disabled (RUN_SEASONAL_MK = False)")
            seasonal_csv_files, seasonal_dirs = [], []
        
        # 3. RUN HYDROLOGICAL YEAR ANALYSIS (BOTH RAW AND 7-DAY ROLLING) (if enabled)
        if RUN_HYDROLOGICAL_YEAR:
            print("\n" + "=" * 80)
            print("  PHASE 3: HYDROLOGICAL YEAR ANALYSIS")
            print("=" * 80)
            hydrological_results = run_hydrological_year_analysis_both_types(
                data=data,
                well_columns=well_columns,
                mapping_dict=mapping_dict,
                save_directory=os.path.join(custom_save_location, '03_Hydrological_Year_Analysis')
            )
        else:
            print("\n" + "=" * 80)
            print("  PHASE 3: HYDROLOGICAL YEAR ANALYSIS (SKIPPED)")
            print("=" * 80)
            print("  • Hydrological year analysis is disabled (RUN_HYDROLOGICAL_YEAR = False)")
            hydrological_results = None
        
        # 4. CREATE GWL AND ROLLING MEAN PLOTS
        if save_plots:
            print("\n" + "=" * 80)
            print("  PHASE 4: GROUNDWATER LEVEL PLOTS")
            print("=" * 80)
            
            # Ensure data has datetime index for GWL plotting
            if not isinstance(data.index, pd.DatetimeIndex):
                print("  Converting data index to datetime for GWL plotting...")
                try:
                    # Try to convert the index to datetime
                    if 'DateTime' in data.columns:
                        print("  Found 'DateTime' column, converting to index...")
                        data = data.set_index('DateTime')
                        print("  Successfully converted DateTime column to index")
                    elif hasattr(data.index, 'name') and data.index.name == 'DateTime':
                        print("  Found DateTime index name, resetting and converting...")
                        data = data.reset_index().set_index('DateTime')
                        print("  Successfully reset and converted DateTime index")
                    else:
                        print("  Warning: Could not find DateTime column for GWL plotting")
                        print(f"  Available columns: {data.columns.tolist()}")
                        print(f"  Index name: {data.index.name if hasattr(data.index, 'name') else 'No name'}")
                except Exception as e:
                    print(f"  Error converting data index: {e}")
                    traceback.print_exc()
            else:
                print("  Data already has datetime index")
            
            save_dir = os.path.join(custom_save_location, '04_GWL_Plots')
            
            gwl_plot_results = run_gwl_rolling_mean_plots(
                data=data,
                well_columns=well_columns,
                save_directory=save_dir,
                mapping_dict=mapping_dict,
                rolling_window=ROLLING_WINDOW_SIZE
            )
            print(f"  GWL plots created: {len(gwl_plot_results)} wells")
    
        else:
            gwl_plot_results = {}
            print("\n" + "=" * 80)
            print("   PHASE 4: GROUNDWATER LEVEL PLOTS (SKIPPED)")
            print("=" * 80)
            print("   • Plot saving is disabled (save_plots=False)")
        
        # 5. CREATE YEARLY WATER LEVEL STATISTICS (if enabled)
        if RUN_YEARLY_STATISTICS:
            print("\n" + "=" * 80)
            print("  PHASE 5: YEARLY WATER LEVEL STATISTICS")
            print("=" * 80)
            
            yearly_stats_results = run_yearly_water_level_statistics(
                data=data,
                well_columns=well_columns,
                mapping_dict=mapping_dict,
                save_directory=os.path.join(custom_save_location, '05_Yearly_Water_Level_Statistics')
            )
            print(f"  Yearly statistics created: {len(yearly_stats_results) if yearly_stats_results else 0} wells")
        else:
            print("\n" + "=" * 80)
            print("  PHASE 5: YEARLY WATER LEVEL STATISTICS (SKIPPED)")
            print("=" * 80)
            print("  • Yearly water level statistics is disabled (RUN_YEARLY_STATISTICS = False)")
            yearly_stats_results = None
        
        # 6. CREATE TIME EVOLUTION ANALYSIS
        if RUN_TIME_EVOLUTION:
            print("\n" + "=" * 80)
            print("  PHASE 6: TIME EVOLUTION ANALYSIS")
            print("=" * 80)
            
            time_evolution_results = run_time_evolution_analysis(
                data=data,
                well_columns=well_columns,
                mapping_dict=mapping_dict,
                save_directory=os.path.join(custom_save_location, '06_Time_Evolution_Analysis')
            )
            print(f"  Time evolution analysis created: {len(time_evolution_results) if time_evolution_results else 0} wells")
        else:
            print("\n" + "=" * 80)
            print("  PHASE 6: TIME EVOLUTION ANALYSIS (SKIPPED)")
            print("=" * 80)
            print("  • Time evolution analysis is disabled (RUN_TIME_EVOLUTION = False)")
            time_evolution_results = None
        
        # 7. CREATE COMPREHENSIVE SUMMARY
        print("\n" + "=" * 80)
        print("  PHASE 7: COMPREHENSIVE SUMMARY GENERATION")
        print("=" * 80)
        
        summary_data = {
            'Analysis_Type': ['Original MK', 'Seasonal MK', 'Hydrological Year', 'GWL Plots', 'Yearly Statistics', 'Time Evolution', 'TOTAL'],
            'CSV_Files_Created': [len(original_csv_files) if original_csv_files else 0, 
                                 len(seasonal_csv_files) if RUN_SEASONAL_MK and seasonal_csv_files else 'Skipped' if not RUN_SEASONAL_MK else 0, 
                                 'N/A' if RUN_HYDROLOGICAL_YEAR else 'Skipped', 
                                 len(gwl_plot_results) if gwl_plot_results else 0,
                                 len(yearly_stats_results) if RUN_YEARLY_STATISTICS and yearly_stats_results else 'Skipped' if not RUN_YEARLY_STATISTICS else 0,
                                 len(time_evolution_results) if RUN_TIME_EVOLUTION and time_evolution_results else 'Skipped' if not RUN_TIME_EVOLUTION else 0,
                                 (len(original_csv_files) if original_csv_files else 0) + 
                                 (len(seasonal_csv_files) if RUN_SEASONAL_MK and seasonal_csv_files else 0) + 
                                 (len(gwl_plot_results) if gwl_plot_results else 0) + 
                                 (len(yearly_stats_results) if RUN_YEARLY_STATISTICS and yearly_stats_results else 0) + 
                                 (len(time_evolution_results) if RUN_TIME_EVOLUTION and time_evolution_results else 0)],
            'Output_Directories': [len(original_dirs) if original_dirs else 0, 
                                  len(seasonal_dirs) if RUN_SEASONAL_MK and seasonal_dirs else 0, 
                                  1 if RUN_HYDROLOGICAL_YEAR else 0, 
                                  1,
                                  1 if RUN_YEARLY_STATISTICS else 0,
                                  1 if RUN_TIME_EVOLUTION else 0,
                                  (len(original_dirs) if original_dirs else 0) + 
                                  (len(seasonal_dirs) if RUN_SEASONAL_MK and seasonal_dirs else 0) + 
                                  (1 if RUN_HYDROLOGICAL_YEAR else 0) + 
                                  (1 if RUN_YEARLY_STATISTICS else 0) + 
                                  (1 if RUN_TIME_EVOLUTION else 0) + 1],
            'Status': ['Completed' if original_csv_files else 'Failed', 
                      'Completed' if RUN_SEASONAL_MK and seasonal_csv_files else 'Skipped' if not RUN_SEASONAL_MK else 'Failed',
                      'Completed' if RUN_HYDROLOGICAL_YEAR and hydrological_results else 'Skipped' if not RUN_HYDROLOGICAL_YEAR else 'Failed',
                      'Completed' if gwl_plot_results else 'Failed',
                      'Completed' if RUN_YEARLY_STATISTICS and yearly_stats_results else 'Skipped' if not RUN_YEARLY_STATISTICS else 'Failed',
                      'Completed' if RUN_TIME_EVOLUTION and time_evolution_results else 'Skipped' if not RUN_TIME_EVOLUTION else 'Failed',
                      'Completed' if (original_csv_files or (RUN_SEASONAL_MK and seasonal_csv_files)) else 'Failed']
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(custom_save_location, '00_Comprehensive_Analysis_Summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f" Comprehensive summary saved to: {summary_csv_path}")
        
        # 6. FINAL RESULTS
        print("\n" + "╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
        print("       ║                               ANALYSIS COMPLETED SUCCESSFULLY!                                       ║")
        print("       ╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝")
        print()
        print(" FINAL RESULTS SUMMARY:")
        print("=" * 60)
        total_csv_files = (len(original_csv_files) if original_csv_files else 0) + (len(seasonal_csv_files) if seasonal_csv_files else 0)
        print(f"   • Total CSV Files Created: {total_csv_files}")
        print(f"   • Main Output Location: {custom_save_location}")
        print("=" * 60)
        print()
        print("   ANALYSIS COMPONENTS COMPLETED:")
        if RUN_ORIGINAL_MK:
            print("   ✓ Original Mann-Kendall Analysis")
        else:
            print("   ⚠ Original Mann-Kendall Analysis (disabled)")
        if RUN_SEASONAL_MK:
            print("   ✓ Seasonal Mann-Kendall Analysis")
        else:
            print("   ⚠ Seasonal Mann-Kendall Analysis (disabled)")
        if RUN_HYDROLOGICAL_YEAR:
            print("   ✓ Hydrological Year Analysis (Raw + 7-Day Rolling)")
        else:
            print("   ⚠ Hydrological Year Analysis (disabled)")
        if save_plots:
            print("   ✓ Groundwater Level & Rolling Mean Plots")
        else:
            print("   ⚠ Groundwater Level Plots (disabled - save_plots=False)")
        if RUN_YEARLY_STATISTICS:
            print("   ✓ Yearly Water Level Statistics")
        else:
            print("   ⚠ Yearly Water Level Statistics (disabled)")
        if RUN_TIME_EVOLUTION:
            print("   ✓ Time Evolution Analysis")
        else:
            print("   ⚠ Time Evolution Analysis (disabled)")
        print()
        print("   All analysis results have been saved to your specified location!")
        print("   You can now review the comprehensive groundwater trend assessment.")
        
        return original_csv_files, seasonal_csv_files
        
    except Exception as e:
        print(f"Error in run_mann_kendall_dashboard: {e}")
        import traceback
        traceback.print_exc()
        return [], []


__all__ = ['run_mann_kendall_dashboard']
