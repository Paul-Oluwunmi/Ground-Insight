"""
Batch Plotting Functions

This module contains functions for batch plotting across multiple wells.
"""

import pandas as pd
import os
import traceback
from ..constants import (
    ALPHA_LEVEL,
    ROLLING_WINDOW_SIZE,
    SAVE_PLOTS
)
from ..utils import (
    get_well_folder_name_consistent
)
from .individual_plots import (
    create_individual_well_plots,
    create_gwl_rolling_mean_plot_simple
)


def run_individual_well_plotting(data, well_columns, save_directory, mapping_dict=None,
                                rolling_window=None, seasonal_period=12, alpha=None, save_plots=None):
    """
    Run individual well plotting for all wells.
    
    Args:
        data: DataFrame with datetime index and well columns
        well_columns: List of well column names
        save_directory: Directory to save plots
        mapping_dict: Dictionary mapping well names to display names
        rolling_window: Rolling window size for smoothing
        seasonal_period: Seasonal period for analysis
        alpha: Significance level for trend tests
        save_plots: Whether to save plots (default uses global SAVE_PLOTS setting)
    
    Returns:
        dict: Dictionary with well names as keys and lists of plot file paths as values
    """
    # Use global settings if none specified
    if alpha is None:
        alpha = ALPHA_LEVEL
        print(f"Using global alpha level: {alpha}")
    if save_plots is None:
        save_plots = SAVE_PLOTS
        print(f"Using global save plots setting: {save_plots}")
    
    # Use global rolling window if none specified
    if rolling_window is None:
        rolling_window = ROLLING_WINDOW_SIZE
        print(f"Using global rolling window size: {rolling_window} days")
    
    try:
        # Create plots directory
        plots_dir = os.path.join(save_directory, "individual_well_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        all_plots = {}
        
        for well in well_columns:
            if well in data.columns:
                well_data = data[well].dropna()
                # No minimum data requirement - process all wells with any data
                if len(well_data) > 0:
                    print(f"\nCreating plots for {well}...")
                    plot_files = create_individual_well_plots(
                        well_data=well_data,
                        well_name=well,
                        save_dir=plots_dir,
                        mapping_dict=mapping_dict,
                        rolling_window=rolling_window,
                        seasonal_period=seasonal_period,
                        alpha=alpha,
                        save_plots=save_plots
                    )
                    all_plots[well] = plot_files
                    print(f"Created {len(plot_files)} plots for {well}")
                else:
                    print(f"Skipping {well}: no data available")
        
        print(f"\nIndividual well plotting completed!")
        print(f"Total wells processed: {len(all_plots)}")
        print(f"Plots saved to: {plots_dir}")
        
        return all_plots
        
    except Exception as e:
        print(f"Error in run_individual_well_plotting: {e}")
        return {}


def run_gwl_rolling_mean_plots(data, well_columns, save_directory, mapping_dict=None, rolling_window=None):
    """
    Create GWL and rolling mean plots for all wells.
    
    Args:
        data: DataFrame with datetime index and well columns
        well_columns: List of well column names
        save_directory: Directory to save plots
        mapping_dict: Dictionary mapping well names to display names
        rolling_window: Rolling window size for smoothing
    
    Returns:
        dict: Dictionary with well names as keys and plot file paths as values
    """
    # Use global rolling window if none specified
    if rolling_window is None:
        rolling_window = ROLLING_WINDOW_SIZE
        print(f"Using global rolling window size: {rolling_window} days")
    
    try:
        print(f"Starting GWL plotting for {len(well_columns)} wells...")
        print(f"Save directory: {save_directory}")
        print(f"Rolling window: {rolling_window} days")
        
        all_plots = {}
        
        for well in well_columns:
            try:
                if well in data.columns:
                    # Ensure we have the correct data with datetime index
                    well_data = data[well].dropna()
                    
                    # Check if the data has datetime index
                    if not isinstance(data.index, pd.DatetimeIndex):
                        print(f"Error: Data for {well} must have datetime index")
                        continue
                    
                    print(f"Processing {well}: {len(well_data)} data points")
                    
                    # No minimum data requirement - process all wells with any data
                    if len(well_data) > 0:
                        # Create well-specific folder using consistent naming
                        well_folder = os.path.join(save_directory, get_well_folder_name_consistent(well, mapping_dict))
                        os.makedirs(well_folder, exist_ok=True)
                        print(f"Created folder: {well_folder}")
                        
                        print(f"Creating GWL rolling mean plot for {well}...")
                        plot_file = create_gwl_rolling_mean_plot_simple(
                            data=well_data,
                            well_name=well,
                            save_dir=well_folder,
                            rolling_window=rolling_window
                        )
                        if plot_file:
                            all_plots[well] = plot_file
                            print(f"✓ Created plot for {well}: {plot_file}")
                        else:
                            print(f"✗ Failed to create plot for {well}")
                    else:
                        print(f"Skipping {well}: no data available")
                else:
                    print(f"Warning: {well} not found in data columns")
            except Exception as e:
                print(f"Error processing {well}: {e}")
                traceback.print_exc()
                continue
        
        print(f"\nGWL rolling mean plotting completed!")
        print(f"Total wells processed: {len(all_plots)}")
        print(f"Plots saved to individual well folders")
        
        return all_plots
        
    except Exception as e:
        print(f"Error in run_gwl_rolling_mean_plots: {e}")
        return {}


__all__ = [
    'run_individual_well_plotting',
    'run_gwl_rolling_mean_plots'
]
